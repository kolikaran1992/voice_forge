# pylint: disable=E1137
# Load and resample audio function
import torch
import numpy as np
import librosa
from torch import nn
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any

from voice_forge.vits_clone.text import text_to_sequence


def load_and_resample_audio(audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    """
    Loads an audio file, converts it to mono, and resamples it to the target sampling rate.

    Args:
        audio_path (str): The file path to the input audio (e.g., 'audio.wav').
        target_sr (int): The required sampling rate for the model (e.g., 22050 Hz).

    Returns:
        tuple: A tuple containing:
            - wav (np.ndarray): The resampled mono waveform as a float32 NumPy array.
            - sr (int): The final sampling rate, which is equal to target_sr.
    """
    # librosa.load() reads the file, converts it to mono (default), and resamples
    # it to the specified 'sr' automatically using high-quality resampling.
    wav, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Ensure the waveform is a float32 type, which is standard for model inputs
    # wav = wav.astype(np.float32)

    return wav, sr


class MelSpectrogram(nn.Module):
    """
    Revised PyTorch module to compute BOTH the Log-Mel Spectrogram and the
    Linear Magnitude Spectrogram from a raw waveform tensor.
    """

    def __init__(self, hps_config):
        super().__init__()

        # 1. Retrieve Hyperparameters (Same as before)
        mel_config = hps_config.data.mel_config
        self.sampling_rate = hps_config.data.target_sampling_rate
        self.n_fft = mel_config.filter_length
        self.hop_size = mel_config.hop_length
        self.win_size = mel_config.win_length
        self.n_mels = mel_config.n_mels
        self.fmin = mel_config.fmin
        self.fmax = mel_config.fmax
        self.clip_val = 1e-5

        # 2. Create Hann Window (Same as before)
        self.register_buffer("window", torch.hann_window(self.win_size))

        # 3. Create Mel Filterbank (Same as before)
        mel_basis_np = librosa.filters.mel(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis_np).float())

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the Log-Mel Spectrogram AND the Linear Magnitude Spectrogram.

        Args:
            y (torch.Tensor): Raw waveform tensor [B, T_wav]

        Returns:
            tuple: (Log-Mel Spectrogram [B, n_mels, T_mel], Linear Spectrogram [B, n_fft/2 + 1, T_mel])
        """
        y = y.float()

        # --- Step 2.1: STFT ---
        D = torch.stft(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            center=False,
            return_complex=True,
        )

        # --- Step 2.2: Linear Magnitude Spectrogram ---
        spec_magnitude = (
            D.abs()
        )  # shape: [B, n_fft/2 + 1, T_frames] (This has 513 channels)
        # Note: We will use this spec_magnitude as the input 'y' for net_g

        # --- Step 2.3: Mel Filterbank Application (for Log-Mel Spectrogram) ---
        mel_spec = torch.matmul(self.mel_basis, spec_magnitude)

        # --- Step 2.4: Logarithmic Compression and Clipping ---
        mel_spec = torch.clamp(mel_spec, min=self.clip_val)
        log_mel_spec = torch.log(mel_spec)

        # Return BOTH the log_mel_spec (80 channels) and the linear spec (513 channels)
        return log_mel_spec, spec_magnitude


class TextAudioSpeakerDataset(Dataset):
    """
    A PyTorch Dataset for VITS, handling text-to-sequence conversion,
    audio loading, and mel-spectrogram computation.
    """

    def __init__(
        self,
        audiopaths_and_text: List[Dict[str, Any]],
        hps_config: Any,  # hparams instance
        cleaner_names: List[str] = ["english_cleaners"],
    ):

        super().__init__()

        # List of dictionaries, e.g., [{'wav_file_path': 'a.wav', 'utterance': 'Hello.', 'speaker_id': 0}, ...]
        self.audiopaths_and_text = audiopaths_and_text
        self.cleaner_names = cleaner_names

        # Store configuration
        self.hps = hps_config
        self.target_sr = self.hps.data.target_sampling_rate

        # Initialize the MelSpectrogram converter
        self.mel_converter = MelSpectrogram(hps_config)

        # Optional: Cache the length of the dataset
        self._length = len(self.audiopaths_and_text)

    def __len__(self) -> int:
        return self._length

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves one processed sample from the dataset.

        Returns:
            (text_tensor, mel_tensor, wav_tensor, sid_tensor)
        """
        sample_info = self.audiopaths_and_text[index]

        # 1. Text Processing
        text = sample_info["utterance"]
        # Use your provided function
        text_ids = text_to_sequence(text, self.cleaner_names)
        text_tensor = torch.LongTensor(text_ids)

        # 2. Audio Processing (Waveform and Mel-Spectrogram)
        audio_path = sample_info["wav_file_path"]
        # Use your provided function
        wav_np, sr = load_and_resample_audio(audio_path, self.target_sr)

        # Convert to tensor and compute mel-spectrogram
        wav_tensor = torch.from_numpy(wav_np).unsqueeze(0).float()

        # Your MelSpectrogram class expects a batch dimension [1, T_wav]
        _, mel_output = self.mel_converter(wav_tensor)
        mel_tensor = mel_output.squeeze(0)  # [n_mels, T_mel]

        # 3. Speaker ID Processing
        # Assumes 'speaker_id' is stored as an integer (0, 1, 2, etc.)
        sid = sample_info.get(
            "speaker_id", 0
        )  # Default to 0 if not present (single-speaker)
        sid_tensor = torch.tensor(sid, dtype=torch.long)

        # Return all necessary tensors
        return text_tensor, mel_tensor, wav_tensor, sid_tensor


class TextAudioSpeakerCollate:
    """Zero-pads model inputs and targets."""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # Sort by spectrogram length
        ids_sorted = self._sort_batch_by_spec_length(batch)

        # Process each modality independently
        text_padded, text_lengths = self._process_text_batch(batch, ids_sorted)
        spec_padded, spec_lengths = self._process_spec_batch(batch, ids_sorted)
        wav_padded, wav_lengths = self._process_wav_batch(batch, ids_sorted)
        sid = self._process_speaker_batch(batch, ids_sorted)

        if self.return_ids:
            return (
                text_padded,
                text_lengths,
                spec_padded,
                spec_lengths,
                wav_padded,
                wav_lengths,
                sid,
                ids_sorted,
            )

        return (
            text_padded,
            text_lengths,
            spec_padded,
            spec_lengths,
            wav_padded,
            wav_lengths,
            sid,
        )

    # --------------------------------------------------
    # Sorting Helper
    # --------------------------------------------------

    def _sort_batch_by_spec_length(self, batch):
        lengths = torch.LongTensor([x[1].size(1) for x in batch])
        _, ids = torch.sort(lengths, dim=0, descending=True)
        return ids

    # --------------------------------------------------
    # 4 Dedicated Private Methods
    # --------------------------------------------------

    def _process_text_batch(self, batch, ids_sorted):
        """Pad text sequences for the whole batch."""
        max_text_len = max(len(x[0]) for x in batch)
        B = len(batch)

        text_padded = torch.LongTensor(B, max_text_len).zero_()
        text_lengths = torch.LongTensor(B)

        for i, bi in enumerate(ids_sorted):
            text = batch[bi][0]
            L = text.size(0)
            text_padded[i, :L] = text
            text_lengths[i] = L

        return text_padded, text_lengths

    def _process_spec_batch(self, batch, ids_sorted):
        """Pad spectrograms for the whole batch."""
        max_spec_len = max(x[1].size(1) for x in batch)
        n_mels = batch[0][1].size(0)
        B = len(batch)

        spec_padded = torch.FloatTensor(B, n_mels, max_spec_len).zero_()
        spec_lengths = torch.LongTensor(B)

        for i, bi in enumerate(ids_sorted):
            spec = batch[bi][1]
            L = spec.size(1)
            spec_padded[i, :, :L] = spec
            spec_lengths[i] = L

        return spec_padded, spec_lengths

    def _process_wav_batch(self, batch, ids_sorted):
        """Pad waveforms for the whole batch."""
        max_wav_len = max(x[2].size(1) for x in batch)
        B = len(batch)

        wav_padded = torch.FloatTensor(B, 1, max_wav_len).zero_()
        wav_lengths = torch.LongTensor(B)

        for i, bi in enumerate(ids_sorted):
            wav = batch[bi][2]
            L = wav.size(1)
            wav_padded[i, :, :L] = wav
            wav_lengths[i] = L

        return wav_padded, wav_lengths

    def _process_speaker_batch(self, batch, ids_sorted):
        """Extract speaker IDs for the whole batch."""
        B = len(batch)
        sid = torch.LongTensor(B)

        for i, bi in enumerate(ids_sorted):
            sid[i] = batch[bi][3]

        return sid
