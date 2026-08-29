"""FFT / STFT spectral analysis.

The Fourier Transform re-expresses a time-domain signal as a sum of
sinusoids, revealing which frequencies are present and how strong they
are. The (discrete, fast) FFT gives one static spectrum for a whole
signal or frame; the Short-Time Fourier Transform (STFT) applies the
FFT to a sliding sequence of overlapping frames, producing a
spectrogram — how the frequency content evolves over time.
"""
from __future__ import annotations

import numpy as np
import librosa


def compute_fft(x: np.ndarray, sr: int):
    """Single-sided amplitude spectrum of a real signal x.

    Returns (freqs, magnitude) for frequencies in [0, sr/2] (Nyquist).
    Magnitude is normalized by frame length and doubled (except DC/Nyquist)
    so it approximates true sinusoid amplitude, independent of length.
    """
    n = len(x)
    window = np.hanning(n) if n > 1 else np.ones(n)
    spectrum = np.fft.rfft(x * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    mag = np.abs(spectrum) / (np.sum(window) + 1e-12)
    mag[1:-1] *= 2  # fold negative-frequency energy back onto positive side
    return freqs, mag


def compute_stft(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int | None = None):
    """Complex STFT: D[f, t] = FFT of the windowed frame centered at time t.

    n_fft sets frequency resolution (Δf = sr/n_fft); hop_length sets time
    resolution. This is the fundamental time/frequency resolution
    trade-off of the STFT (Heisenberg-Gabor limit) — you cannot make both
    arbitrarily fine at once.
    """
    hop_length = hop_length or n_fft // 4
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)
    return D, freqs, times


def spectrogram_db(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int | None = None):
    """Magnitude spectrogram in dB: 20*log10(|D|), for spectrogram display."""
    D, freqs, times = compute_stft(y, sr, n_fft, hop_length)
    mag_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return freqs, times, mag_db
