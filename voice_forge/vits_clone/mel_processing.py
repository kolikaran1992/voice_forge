# voice_forge/vits_clone/mel_processing.py  (patched)
import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read

# keep top-level import for other code, but do NOT rely on it for creating mel basis
import librosa.filters as librosa_filters


MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,  # REQUIRED FOR PYTORCH ≥ 2.4
    )

    spec = torch.abs(spec) + 1e-6  # NEW
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """
    Convert linear-frequency spectrogram to mel spectrogram using a mel-basis cached in `mel_basis`.
    Uses a local import of librosa.filters.mel to avoid any accidental symbol shadowing.
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    _key = str(fmax) + "_" + dtype_device

    if _key not in mel_basis:
        # compute mel basis (numpy), then convert to torch and store keyed by dtype+device
        _mel = librosa_filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )

        mel_basis[_key] = torch.from_numpy(_mel).to(
            dtype=spec.dtype, device=spec.device
        )

    spec = torch.matmul(mel_basis[_key], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """
    Compute mel-spectrogram directly from waveform `y`. This function also creates the mel-basis
    using a local import to avoid runtime symbol shadowing.
    """
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    _fmax_key = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_size) + "_" + dtype_device

    # Ensure mel-basis exists for this dtype/device/fmax combination
    if _fmax_key not in mel_basis:

        _mel = librosa_filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )

        mel_basis[_fmax_key] = torch.from_numpy(_mel).to(dtype=y.dtype, device=y.device)

    # Ensure hann window exists for this dtype/device/win_size
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=y.dtype, device=y.device
        )

    # pad, stft, magnitude
    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,  # REQUIRED FOR PYTORCH ≥ 2.4
    )

    spec = torch.abs(spec) + 1e-6  # NEW

    # apply mel basis & normalize
    spec = torch.matmul(mel_basis[_fmax_key], spec)
    spec = spectral_normalize_torch(spec)

    return spec
