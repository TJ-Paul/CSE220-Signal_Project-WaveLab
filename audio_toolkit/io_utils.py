"""Audio file input/output and metadata inspection.

Handles loading WAV/MP3 files (via libsndfile, with an audioread
fallback for formats libsndfile can't parse), reporting basic
properties, and writing processed signals back to disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import librosa


@dataclass
class AudioInfo:
    filename: str
    duration_s: float
    sample_rate: int
    channels: int
    subtype: str
    bit_depth: int | None
    bitrate_kbps: float
    is_lossy: bool


# Approximate bits-per-sample for common PCM subtypes reported by libsndfile.
_SUBTYPE_BITS = {
    "PCM_S8": 8, "PCM_U8": 8,
    "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT": 32, "DOUBLE": 64,
}


def load_audio(path: str, sr: int | None = None, mono: bool = True):
    """Load an audio file as a float32 waveform in [-1, 1].

    Uses librosa.load, which reads through soundfile (libsndfile — covers
    WAV and, on recent libsndfile builds, MP3) and falls back to
    audioread/ffmpeg for any container libsndfile can't parse.

    sr=None preserves the file's native sample rate instead of
    resampling to librosa's 22050 Hz default.
    """
    y, sr_out = librosa.load(path, sr=sr, mono=mono)
    return y.astype(np.float32), sr_out


def get_audio_info(path_or_buffer, filename: str | None = None, file_size_bytes: int | None = None) -> AudioInfo:
    """Return duration, sample rate, channel count, and an estimated bitrate.

    Accepts either a filesystem path or a file-like buffer (e.g. a
    Streamlit UploadedFile) — soundfile can read metadata from both.

    For uncompressed PCM, bitrate is exact: sample_rate * channels * bit_depth.
    For lossy formats (e.g. MP3) libsndfile does not expose the original
    encoder bitrate, so it is *estimated* from file size / duration —
    the same average-bitrate figure most media players show for
    variable-bitrate files. `file_size_bytes` lets the caller supply the
    size explicitly (needed for in-memory buffers); for a plain path it
    is read from disk automatically.
    """
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    info = sf.info(path_or_buffer)
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)

    bit_depth = _SUBTYPE_BITS.get(info.subtype)
    is_lossy = bit_depth is None

    if file_size_bytes is None and isinstance(path_or_buffer, (str, os.PathLike)):
        file_size_bytes = os.path.getsize(path_or_buffer)

    if is_lossy and file_size_bytes and info.duration > 0:
        bitrate_kbps = (file_size_bytes * 8 / info.duration) / 1000
    elif is_lossy:
        bitrate_kbps = 0.0
    else:
        bitrate_kbps = (info.samplerate * info.channels * bit_depth) / 1000

    if filename is None:
        filename = os.path.basename(path_or_buffer) if isinstance(path_or_buffer, (str, os.PathLike)) else "uploaded audio"

    return AudioInfo(
        filename=filename,
        duration_s=info.duration,
        sample_rate=info.samplerate,
        channels=info.channels,
        subtype=info.subtype,
        bit_depth=bit_depth,
        bitrate_kbps=bitrate_kbps,
        is_lossy=is_lossy,
    )


def save_audio(path: str, y: np.ndarray, sr: int) -> None:
    """Write a float waveform to disk as PCM_16 WAV (clipped to [-1, 1])."""
    y = np.clip(y, -1.0, 1.0)
    sf.write(path, y, sr, subtype="PCM_16")
