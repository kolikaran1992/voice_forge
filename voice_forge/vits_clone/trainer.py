# pylint: disable=W1203
# pylint: disable=W0641
# pylint: disable=W0612
# pylint: disable=W0613
# pylint: disable=W0718
import os
import random
from tqdm import tqdm

import numpy as np

import torch
import torch.nn.functional as F
import torch.amp
from voice_forge.vits_clone.losses import (
    discriminator_loss,
    generator_loss,
    kl_loss,
    feature_loss,
)
from voice_forge.vits_clone import utils
from voice_forge.vits_clone import commons
from voice_forge.vits_clone.mel_processing import (
    mel_spectrogram_torch,
    spec_to_mel_torch,
)


# -------------------------
# Trainer (corrected)
# -------------------------
class Trainer:
    def __init__(self, hps, net_g, net_d, optim_g, optim_d, train_loader, eval_loader):
        self.hps = hps
        self.net_g = net_g
        self.net_d = net_d
        self.optim_g = optim_g
        self.optim_d = optim_d
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        # -----------------------------
        # Device setup
        # -----------------------------
        self.device = torch.device(
            "cuda"
            if (torch.cuda.is_available() and hps.train.device == "cuda")
            else "cpu"
        )

        # AMP only if GPU
        self.use_amp = (self.device.type == "cuda") and getattr(
            hps.train, "fp16_run", False
        )
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        # Move models to device
        self.net_g.to(self.device)
        self.net_d.to(self.device)

        # -----------------------------
        # Convenience alias for mel config
        # (works if mel_config is dict or object)
        # -----------------------------
        self.mel = self.hps.data.mel_config

        # -----------------------------
        # Training state
        # -----------------------------
        # both updated when resumed from checkpoint
        self.global_step = 0
        self.start_epoch = 1

        # -----------------------------
        # Learning rate schedulers (ExponentialLR like VITS)
        # -----------------------------
        lr_decay = getattr(self.hps.train, "lr_decay", None)
        if lr_decay is not None:
            self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
                self.optim_g, gamma=lr_decay
            )
            self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
                self.optim_d, gamma=lr_decay
            )
        else:
            self.scheduler_g = None
            self.scheduler_d = None

        # -----------------------------
        # Logging
        # -----------------------------
        self.writer = utils.get_writer(hps.model_dir)
        self.writer_eval = utils.get_writer(os.path.join(hps.model_dir, "eval"))
        self.logger = utils.get_logger(hps.model_dir)

        # -----------------------------
        # Seeding (for reproducibility)
        # -----------------------------
        seed = getattr(self.hps.train, "seed", None)
        if seed is not None:
            self._seed_everything(seed)
            self.logger.info(f"Seed set to: {seed}")

        self.freeze_g = False

    def _seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Ensures deterministic ops where possible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # -----------------------------
    # Robust accessor for mel config
    # -----------------------------
    def _mel(self, key):
        """Return mel_config[key] or mel_config.key depending on type."""
        if hasattr(self.mel, key):
            return getattr(self.mel, key)
        try:
            return self.mel[key]
        except Exception:
            raise KeyError(f"Cannot find mel config key '{key}' in {type(self.mel)}")

    # ------------------------------------------------------
    # Utility: move tensors to device (handles None gracefully)
    # ------------------------------------------------------
    def _to_device(self, *tensors):
        result = []
        for t in tensors:
            if t is None:
                result.append(None)
            elif isinstance(t, torch.Tensor):
                result.append(t.to(self.device))
            else:
                # If not a tensor (e.g. metadata), pass as-is
                result.append(t)
        return result

    def _train_epoch(self, epoch):
        if not self.freeze_g:
            self.net_g.train()
        else:
            self.net_g.eval()

        self.net_d.train()

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
            colour="green",
            leave=False,
        )

        for batch_idx, batch in pbar:
            x, x_lengths, spec, spec_lengths, y, y_lengths, *_ = batch
            x, x_lengths, spec, spec_lengths, y, y_lengths = self._to_device(
                x, x_lengths, spec, spec_lengths, y, y_lengths
            )

            # 1. Backward pass: discriminator
            disc_out = self._backward_discriminator(
                x, x_lengths, spec, spec_lengths, y, y_lengths
            )

            # 2. Backward pass: generator (Only if not frozen)
            if not self.freeze_g:
                gen_out = self._backward_generator(disc_out)
            else:
                # Create a dummy gen_out for logging compatibility
                gen_out = {
                    k: 0.0
                    for k in [
                        "loss_total",
                        "loss_mel",
                        "loss_fm",
                        "loss_dur",
                        "loss_kl",
                        "grad_norm_g",
                    ]
                }

            # 3. Logging & Eval
            if self.global_step % int(self.hps.train.log_interval) == 0:
                self._log_step(epoch, batch_idx, disc_out, gen_out)

            if (
                self.eval_loader is not None
                and self.global_step % int(self.hps.train.eval_interval) == 0
            ):
                self._evaluate()
                self._save_checkpoint(epoch)

            self.global_step += 1

    # ------------------------------------------------------
    # Discriminator Backward
    # ------------------------------------------------------
    def _backward_discriminator(self, x, x_lengths, spec, spec_lengths, y, y_lengths):
        hps = self.hps

        # 1. Generator Forward Pass
        # If frozen, we wrap ONLY this part in no_grad to save memory
        if self.freeze_g:
            with torch.no_grad():
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, latent = self.net_g(
                    x, x_lengths, spec, spec_lengths
                )
        else:
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask, latent = self.net_g(
                    x, x_lengths, spec, spec_lengths
                )

        # 2. Pre-process segments (Still inside autocast if enabled)
        with torch.amp.autocast(self.device.type, enabled=self.use_amp):
            mel = spec_to_mel_torch(
                spec,
                self._mel("filter_length"),
                self._mel("n_mels"),
                hps.data.sampling_rate,
                self._mel("fmin"),
                self._mel("fmax"),
            )
            segment_frames = int(hps.train.segment_size // self._mel("hop_length"))
            y_mel = commons.slice_segments(mel, ids_slice, segment_frames)
            y_slice = commons.slice_segments(
                y, ids_slice * int(self._mel("hop_length")), int(hps.train.segment_size)
            )

            # 3. Discriminator Forward Pass
            # THIS MUST ALWAYS HAVE GRADIENTS ENABLED
            # We detach y_hat so gradients don't try to go back to the Generator
            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y_slice, y_hat.detach())

        # 4. Compute loss and backward
        with torch.amp.autocast(self.device.type, enabled=False):
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )

        self.optim_d.zero_grad()
        self.scaler.scale(
            loss_disc
        ).backward()  # Now loss_disc will have a grad_fn from net_d
        self.scaler.unscale_(self.optim_d)
        grad_norm_d = commons.clip_grad_value_(self.net_d.parameters(), None)
        self.scaler.step(self.optim_d)

        return {
            "y_hat": y_hat,
            "ids_slice": ids_slice,
            "l_length": l_length,
            "z_mask": z_mask,
            "latent": latent,
            "y_slice": y_slice,
            "y_mel": y_mel,
            "mel": mel,
            "loss_disc": loss_disc,
            "losses_disc_r": losses_disc_r,
            "losses_disc_g": losses_disc_g,
            "grad_norm_d": grad_norm_d,
            "attn": attn,
        }

    # ------------------------------------------------------
    # Generator Backward
    # ------------------------------------------------------
    def _backward_generator(self, do):
        hps = self.hps

        # Unpack latent robustly (tuple/list)
        try:
            z, z_p, m_p, logs_p, m_q, logs_q = do["latent"]
        except Exception:
            # if net_g returned different structure, try alternate unpack
            z = z_p = m_p = logs_p = m_q = logs_q = None

        # Compute mel from generated audio (do["y_hat"] shape: [B, 1, T] or [B, T])
        y_hat_tensor = do["y_hat"]
        if y_hat_tensor.dim() == 3 and y_hat_tensor.size(1) == 1:
            y_hat_for_mel = y_hat_tensor.squeeze(1)
        else:
            y_hat_for_mel = y_hat_tensor

        y_hat_mel = mel_spectrogram_torch(
            y_hat_for_mel,
            self._mel("filter_length"),
            self._mel("n_mels"),
            hps.data.sampling_rate,
            self._mel("hop_length"),
            self._mel("win_length"),
            self._mel("fmin"),
            self._mel("fmax"),
        )

        with torch.amp.autocast(self.device.type, enabled=self.use_amp):

            # D forward for generator update (no detach)
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(
                do["y_slice"], do["y_hat"]
            )

            # Losses in fp32 (disable autocast)
            with torch.amp.autocast(self.device.type, enabled=False):
                loss_mel = F.l1_loss(do["y_mel"], y_hat_mel) * hps.train.c_mel
                loss_dur = torch.sum(do["l_length"].float())

                loss_kl_val = (
                    kl_loss(z_p, logs_q, m_p, logs_p, do["z_mask"]) * hps.train.c_kl
                )

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gan, losses_gen = generator_loss(y_d_hat_g)

                loss_total = loss_gan + loss_fm + loss_mel + loss_dur + loss_kl_val

        self.optim_g.zero_grad()
        self.scaler.scale(loss_total).backward()
        self.scaler.unscale_(self.optim_g)
        grad_norm_g = commons.clip_grad_value_(self.net_g.parameters(), None)
        self.scaler.step(self.optim_g)
        self.scaler.update()

        return {
            "loss_total": loss_total,
            "loss_gen": loss_gan,
            "loss_fm": loss_fm,
            "loss_mel": loss_mel,
            "loss_dur": loss_dur,
            "loss_kl": loss_kl_val,
            "losses_gen": losses_gen,
            "grad_norm_g": grad_norm_g,
        }

    # ------------------------------------------------------
    # Logging
    # ------------------------------------------------------
    def _log_step(self, epoch, batch_idx, disc_out, gen_out):

        lr = self.optim_g.param_groups[0]["lr"]

        self.logger.info(f"Epoch {epoch} | Step {self.global_step} | Batch {batch_idx}")

        scalar_dict = {
            "loss/d": disc_out["loss_disc"],
            "loss/g_total": gen_out["loss_total"],
            "loss/g_mel": gen_out["loss_mel"],
            "loss/g_fm": gen_out["loss_fm"],
            "loss/g_dur": gen_out["loss_dur"],
            "loss/g_kl": gen_out["loss_kl"],
            "grad/g": gen_out["grad_norm_g"],
            "grad/d": disc_out["grad_norm_d"],
            "learning_rate": lr,
        }

        utils.write_scalars(self.writer, self.global_step, scalar_dict)

        # Write mel images
        # compute y_hat_mel for logging safely (use CPU numpy)
        # ensure y_hat is on device and in shape [B, T] for mel computation
        y_hat = disc_out["y_hat"]
        if y_hat.dim() == 3 and y_hat.size(1) == 1:
            y_hat_log = y_hat.squeeze(1).detach().cpu()
        else:
            y_hat_log = y_hat.detach().cpu()

        with torch.amp.autocast("cpu", enabled=False):
            # mel_spectrogram_torch expects a tensor shaped [B, T] or similar
            y_hat_mel_for_log = mel_spectrogram_torch(
                y_hat_log,
                self._mel("filter_length"),
                self._mel("n_mels"),
                self.hps.data.sampling_rate,
                self._mel("hop_length"),
                self._mel("win_length"),
                self._mel("fmin"),
                self._mel("fmax"),
            )

        images = {
            "mel/gt_slice": utils.plot_spectrogram_to_numpy(
                disc_out["y_mel"][0].detach().cpu().numpy()
            ),
            "mel/gen_slice": utils.plot_spectrogram_to_numpy(
                y_hat_mel_for_log[0].detach().cpu().numpy()
            ),
        }
        # Add attention logging
        if "attn" in disc_out:
            attn = disc_out["attn"]

            # VITS attention shape is usually [B, n_heads, T_dec, T_enc]
            # Plot only head 0, batch 0 like the reference trainer.
            attn_img = utils.plot_alignment_to_numpy(attn[0, 0].detach().cpu().numpy())

            images["attn/head0"] = attn_img

        utils.write_images(self.writer, self.global_step, images)

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------
    def _evaluate(self):
        self.net_g.eval()

        with torch.no_grad():
            # Create tqdm bar
            pbar = tqdm(
                enumerate(self.eval_loader),
                total=len(self.eval_loader),
                desc=f"Eval",
                colour="green",
                leave=False,  # keeps logs clean
            )

            for batch_idx, batch in pbar:
                x, x_lengths, spec, spec_lengths, y, y_lengths, *rest = self._to_device(
                    *batch
                )
                # Use single sample for eval
                x = x[:1]
                x_lengths = x_lengths[:1]
                spec = spec[:1]
                spec_lengths = spec_lengths[:1]
                y = y[:1]
                y_lengths = y_lengths[:1]
                break

            # Inference (adapt to your net_g.infer signature)
            # many implementations return (y_hat, attn, *_)
            y_hat, attn, mask, *rest = self.net_g.infer(x, x_lengths, max_len=1000)

            # GT mel
            mel = spec_to_mel_torch(
                spec,
                self._mel("filter_length"),
                self._mel("n_mels"),
                self.hps.data.sampling_rate,
                self._mel("fmin"),
                self._mel("fmax"),
            )

            # Generated mel
            if y_hat.dim() == 3 and y_hat.size(1) == 1:
                y_hat_for_mel = y_hat.squeeze(1).float()
            else:
                y_hat_for_mel = y_hat.float()

            y_hat_mel = mel_spectrogram_torch(
                y_hat_for_mel,
                self._mel("filter_length"),
                self._mel("n_mels"),
                self.hps.data.sampling_rate,
                self._mel("hop_length"),
                self._mel("win_length"),
                self._mel("fmin"),
                self._mel("fmax"),
            )

        utils.write_images(
            self.writer_eval,
            self.global_step,
            {"eval/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())},
        )

        # Add attention image in eval writer
        attn_img = utils.plot_alignment_to_numpy(attn[0, 0].cpu().numpy())
        utils.write_images(
            self.writer_eval,
            self.global_step,
            {"eval/attn": attn_img},
        )

        # If your writer supports audio and y_hat is on CPU:
        audio_for_write = y_hat_for_mel if "y_hat_for_mel" in locals() else y_hat
        utils.write_audio(
            self.writer_eval,
            self.global_step,
            {"eval/audio": audio_for_write[0].cpu()},
            self.hps.data.sampling_rate,
        )

        self.net_g.train()

    # ------------------------------------------------------
    # Saving checkpoints
    # ------------------------------------------------------
    def _save_checkpoint(self, epoch):
        utils.save_checkpoint(
            self.net_g,
            self.optim_g,
            self.hps.train.learning_rate,
            epoch,
            os.path.join(self.hps.model_dir, f"G_{self.global_step}.pth"),
        )

        utils.save_checkpoint(
            self.net_d,
            self.optim_d,
            self.hps.train.learning_rate,
            epoch,
            os.path.join(self.hps.model_dir, f"D_{self.global_step}.pth"),
        )

    def _resume_schedulers(self, start_epoch):
        """
        Advance schedulers so their internal step count matches the epoch
        we are resuming from.
        start_epoch is 1-based.
        """
        if start_epoch <= 1:
            return

        steps = start_epoch - 1

        if getattr(self, "scheduler_g", None) is not None:
            for _ in range(steps):
                self.scheduler_g.step()

        if getattr(self, "scheduler_d", None) is not None:
            for _ in range(steps):
                self.scheduler_d.step()

        self.logger.info(
            f"Schedulers resumed to epoch {start_epoch}. Applied {steps} step() calls."
        )

    # ------------------------------------------------------
    # Loading checkpoints (Generator + Discriminator)
    # ------------------------------------------------------
    def load_checkpoint(self, ckpt_dir):
        """
        Loads latest G and D checkpoints. Sets:
            - self.start_epoch
            - self.global_step
            - resumes schedulers
        """

        def _load(pattern, model, optim):
            ckpt = utils.latest_checkpoint_path(ckpt_dir, pattern)
            if ckpt is None:
                return None
            self.logger.info(f"Loading checkpoint: {ckpt}")
            _, _, lr, iteration = utils.load_checkpoint(ckpt, model, optim)
            return iteration  # iteration is epoch number in this repo

        epoch_g = _load("G_*.pth", self.net_g, self.optim_g)
        epoch_d = _load("D_*.pth", self.net_d, self.optim_d)

        if epoch_g is None and epoch_d is None:
            self.logger.info("No checkpoints found. Starting from epoch 1.")
            self.start_epoch = 1
            self.global_step = 0
            return

        # Use the max epoch among G/D checkpoints
        self.start_epoch = max(epoch_g or 1, epoch_d or 1)

        # Compute global step
        steps_per_epoch = len(self.train_loader)
        self.global_step = (self.start_epoch - 1) * steps_per_epoch

        # Resume schedulers here
        self._resume_schedulers(self.start_epoch)

        self.logger.info(
            f"Resumed training | start_epoch={self.start_epoch} | global_step={self.global_step}"
        )

    # ------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------
    def train(self):
        """
        runs the main training loop
        """
        for epoch in range(self.start_epoch, int(self.hps.train.epochs) + 1):
            self.logger.info(f"=== EPOCH {epoch} ===")
            self._train_epoch(epoch)

            # Step scheduler like original VITS
            if self.scheduler_g is not None:
                self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

    def train_from_pre_trained_generator(self, gen_path: str, strict: bool = True):
        """
        Load a pretrained SynthesizerTrn generator and begin training
        from epoch 1 with a freshly initialized discriminator.

        Args:
            gen_path: Path to pretrained generator .pth file.
            strict: Whether to strictly enforce weight key matching.
        """

        if not os.path.isfile(gen_path):
            raise FileNotFoundError(f"Pretrained generator file not found: {gen_path}")

        self.logger.info(f"=== Loading pretrained generator from: {gen_path} ===")

        ckpt = torch.load(gen_path, map_location=self.device)

        # Accept formats:
        #   - raw state_dict
        #   - {"model": state_dict}
        #   - {"generator": state_dict}
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "generator" in ckpt:
                state_dict = ckpt["generator"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        missing, unexpected = self.net_g.load_state_dict(state_dict, strict=strict)

        if missing:
            self.logger.warning(f"Missing keys when loading generator: {missing}")
        if unexpected:
            self.logger.warning(f"Unexpected keys when loading generator: {unexpected}")

        self.logger.info("Pretrained generator loaded successfully.")

        # --------------------------------------
        # Reset training state
        # --------------------------------------
        self.start_epoch = 1
        self.global_step = 0

        # Move generator to device (D already initialized on device in __init__)
        self.net_g.to(self.device)

        self.logger.info(
            "=== Starting training using pretrained generator (D is fresh) ==="
        )

        # --------------------------------------
        # Begin training normally
        # --------------------------------------
        self.train()

    def fine_tune_discriminator_only(self, gen_path: str, strict: bool = True):
        """
        Loads a pretrained generator, freezes its parameters, and
        starts a training loop where only the discriminator is updated.
        """
        if not os.path.isfile(gen_path):
            raise FileNotFoundError(f"Pretrained generator file not found: {gen_path}")

        self.logger.info(
            f"=== Loading generator for D-only fine-tuning: {gen_path} ==="
        )

        # 1. Load Generator Weights (matching your existing logic)
        ckpt = torch.load(gen_path, map_location=self.device)
        if isinstance(ckpt, dict):
            state_dict = ckpt.get("model", ckpt.get("generator", ckpt))
        else:
            state_dict = ckpt

        self.net_g.load_state_dict(state_dict, strict=strict)

        # 2. Freeze Generator
        for param in self.net_g.parameters():
            param.requires_grad = False
        self.net_g.eval()
        self.freeze_g = True

        # 3. Reset training state
        self.start_epoch = 1
        self.global_step = 0
        self.net_g.to(self.device)

        self.logger.info("Generator frozen. Starting Discriminator fine-tuning.")

        # 4. Start training
        self.train()
