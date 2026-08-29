"""Butterworth frequency-selective filters (low/high/band-pass/stop).

A Butterworth filter is chosen for its maximally-flat passband (no
ripple), a good default for general-purpose audio filtering. Filters
are designed as second-order sections (SOS) rather than raw (b, a)
transfer-function coefficients — SOS cascades are numerically stable
at higher orders, where direct-form coefficients can blow up.

Filtering is applied with `sosfiltfilt`, which runs the filter forward
then backward. This cancels the phase distortion (delay) a causal
filter would otherwise introduce, at the cost of needing the whole
signal in memory (not suitable for real-time streaming, fine for
offline processing of a recorded file).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

FilterKind = str  # "lowpass" | "highpass" | "bandpass" | "bandstop"


def design_filter(kind: FilterKind, cutoff, sr: int, order: int = 6):
    """Design a Butterworth filter as SOS coefficients.

    cutoff: a single Hz value for lowpass/highpass, or (low_hz, high_hz)
    for bandpass/bandstop. Cutoff frequencies are normalized by the
    Nyquist frequency (sr/2) since scipy's digital filter design expects
    frequencies in [0, 1] where 1 == Nyquist.
    """
    nyq = sr / 2.0
    if kind in ("lowpass", "highpass"):
        wn = cutoff / nyq
    else:
        wn = (cutoff[0] / nyq, cutoff[1] / nyq)
    wn = np.clip(wn, 1e-6, 0.999999)
    return signal.butter(order, wn, btype=kind, output="sos")


def apply_filter(y: np.ndarray, sos) -> np.ndarray:
    """Zero-phase filtering: filtfilt applies the filter twice (forward +
    time-reversed) so net phase shift is zero, avoiding waveform smearing."""
    return signal.sosfiltfilt(sos, y).astype(np.float32)


def frequency_response(sos, sr: int, n_points: int = 2048):
    """Magnitude response H(f) in dB, for plotting what the filter does
    to each frequency before it's applied to actual audio."""
    w, h = signal.sosfreqz(sos, worN=n_points, fs=sr)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    return w, mag_db
