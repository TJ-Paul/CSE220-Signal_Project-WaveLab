"""Reusable Matplotlib figure builders for the Streamlit UI.

Kept separate from app.py so every plot is a pure function of data in,
Figure out — easy to reason about and reuse across tabs.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FCFDFD",
    "axes.edgecolor": "#DEE5E9",
    "axes.labelcolor": "#3B4A56",
    "text.color": "#1C2B36",
    "xtick.color": "#5B7184",
    "ytick.color": "#5B7184",
    "axes.grid": True,
    "grid.color": "#E7ECEE",
    "grid.alpha": 0.8,
    "grid.linewidth": 0.6,
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 12.5,
    "axes.titlecolor": "#0F1B24",
})

# Signal-processing accent system: teal = primary signal (time domain),
# amber = secondary signal / spectral peaks, green = detected/positive regions.
_ACCENT = "#0891B2"
_ACCENT2 = "#C2760F"
_ACCENT3 = "#15803D"


def plot_waveform(y, sr, title="Waveform", highlight_segments=None, xlim=None, figsize=(9, 3)):
    t = np.arange(len(y)) / sr
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="#C6D0D8", linewidth=0.8, zorder=1)
    ax.plot(t, y, color=_ACCENT, linewidth=0.7, zorder=2)
    if highlight_segments:
        for start, end in highlight_segments:
            ax.axvspan(start, end, color=_ACCENT3, alpha=0.18, zorder=0)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_fft_spectrum(freqs, mag, title="Frequency Spectrum", log_x=False, xlim=None, figsize=(9, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(freqs, mag, color=_ACCENT2, alpha=0.12)
    ax.plot(freqs, mag, color=_ACCENT2, linewidth=0.9)
    if log_x:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(xlim)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_time_freq_pair(y, sr, title_prefix=""):
    """Side-by-side time-domain snippet and its FFT spectrum."""
    from .spectral import compute_fft
    freqs, mag = compute_fft(y, sr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color=_ACCENT, linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"{title_prefix}Time Domain")

    ax2.plot(freqs, mag, color=_ACCENT2, linewidth=0.8)
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title(f"{title_prefix}Frequency Domain")
    fig.tight_layout()
    return fig


def plot_spectrogram(freqs, times, mag_db, title="Spectrogram", figsize=(9, 3.5)):
    fig, ax = plt.subplots(figsize=figsize)
    mesh = ax.pcolormesh(times, freqs, mag_db, shading="auto", cmap="magma")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_vad(y, sr, vad_result):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)

    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color=_ACCENT, linewidth=0.5)
    for start, end in vad_result.speech_segments:
        ax1.axvspan(start, end, color=_ACCENT3, alpha=0.3)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform — Speech regions highlighted")

    ax2.plot(vad_result.frame_times, vad_result.energy_db, color=_ACCENT2, label="Short-time energy (dB)")
    ax2b = ax2.twinx()
    ax2b.plot(vad_result.frame_times, vad_result.speech_band_ratio, color=_ACCENT3, alpha=0.7, label="Speech-band ratio")
    ax2.fill_between(
        vad_result.frame_times, ax2.get_ylim()[0], ax2.get_ylim()[1],
        where=vad_result.is_speech, color=_ACCENT3, alpha=0.12, step="mid"
    )
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Energy (dB)", color=_ACCENT2)
    ax2b.set_ylabel("Speech-band ratio", color=_ACCENT3)
    ax2.set_title("Per-frame features used for classification")
    fig.tight_layout()
    return fig


def plot_filter_response(freqs, mag_db, cutoffs=None, title="Filter Frequency Response"):
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(freqs, mag_db, color=_ACCENT, linewidth=1.2)
    if cutoffs:
        for c in cutoffs:
            ax.axvline(c, color=_ACCENT2, linestyle="--", linewidth=1, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title(title)
    ax.set_ylim(bottom=max(-100, mag_db.min()))
    fig.tight_layout()
    return fig


def plot_before_after(y1, sr1, label1, y2, sr2, label2, title="Before / After"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=False)
    t1 = np.arange(len(y1)) / sr1
    t2 = np.arange(len(y2)) / sr2
    ax1.plot(t1, y1, color=_ACCENT, linewidth=0.6)
    ax1.set_title(label1)
    ax1.set_ylabel("Amplitude")
    ax2.plot(t2, y2, color=_ACCENT2, linewidth=0.6)
    ax2.set_title(label2)
    ax2.set_ylabel("Amplitude")
    ax2.set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_sampling_demo(t_hi, x_hi, t_samples, x_samples, t_recon, x_recon, title="Sampling & Reconstruction"):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t_hi, x_hi, color="#9AA7B2", linewidth=1.2, label="Original (continuous-time reference)")
    ax.plot(t_recon, x_recon, color=_ACCENT2, linewidth=1, linestyle="--", label="Reconstructed")
    ax.stem(t_samples, x_samples, linefmt=_ACCENT, markerfmt="o", basefmt=" ")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig
