import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging


# ---------------------------------------------------------
# CHECKPOINTING
# ---------------------------------------------------------


def load_checkpoint(checkpoint_path, model, optimizer=None):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])

    saved_state_dict = checkpoint_dict["model"]

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            logger.info(f"{k} is not in the checkpoint")
            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)

    logger.info(f"Loaded checkpoint '{checkpoint_path}' (iteration {iteration})")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        f"Saving model and optimizer state at iteration {iteration} to {checkpoint_path}"
    )

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


# ---------------------------------------------------------
# SUMMARIZATION / LOGGING
# ---------------------------------------------------------


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


# ---------------------------------------------------------
# MATPLOTLIB FIXED PLOTTING HELPERS
# ---------------------------------------------------------


def _ensure_matplotlib():
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True

        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)


def plot_spectrogram_to_numpy(spectrogram):
    """
    Works reliably across new Matplotlib versions (3.3–3.9+).
    Uses buffer_rgba() instead of deprecated tostring_rgb().
    """
    _ensure_matplotlib()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    data = buf[:, :, :3].copy()  # Drop alpha channel → RGB image
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    """
    Same fix as spectrogram plotting — avoids deprecated APIs.
    """
    _ensure_matplotlib()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(),
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    fig.colorbar(im, ax=ax)

    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    data = buf[:, :, :3].copy()
    plt.close()
    return data


# ---------------------------------------------------------
# WAV / TEXT HELPERS
# ---------------------------------------------------------


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


# ---------------------------------------------------------
# HPARAMS LOADING
# ---------------------------------------------------------


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.json",
        help="JSON file for configuration",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")

    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()

    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_dir(model_dir):
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    with open(config_path, "r") as f:
        data = f.read()
    config = json.loads(data)
    hparams = HParams(**config)
    return hparams


# ---------------------------------------------------------
# GIT HASH CHECK (Optional)
# ---------------------------------------------------------


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(f"{source_dir} is not a git repository, skipping git hash check.")
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    path = os.path.join(model_dir, "githash")

    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                f"git hash mismatch: {saved_hash[:8]} (saved) != {cur_hash[:8]} (current)"
            )
    else:
        open(path, "w").write(cur_hash)


# ---------------------------------------------------------
# LOGGER + WRITER HELPERS
# ---------------------------------------------------------


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    handler = logging.FileHandler(os.path.join(model_dir, filename))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_writer(log_dir):
    from torch.utils.tensorboard import SummaryWriter

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir=log_dir)


def write_scalars(writer, step, scalar_dict):
    for key, value in scalar_dict.items():
        if torch.is_tensor(value):
            value = value.item()
        writer.add_scalar(key, value, step)


def write_images(writer, step, image_dict):
    for key, img in image_dict.items():
        writer.add_image(key, img, step, dataformats="HWC")


def write_audio(writer, step, audio_dict, sampling_rate):
    for key, audio_tensor in audio_dict.items():
        if torch.is_tensor(audio_tensor) and audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        writer.add_audio(
            key,
            audio_tensor,
            step,
        )


# ---------------------------------------------------------
# HParams container
# ---------------------------------------------------------


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
