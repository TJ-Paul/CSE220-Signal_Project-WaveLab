"""Quantitative comparison metrics between two signals (e.g. original vs.
processed/filtered/denoised), used to objectively judge what a processing
step did instead of relying on listening alone."""
from __future__ import annotations

import numpy as np


def _align(x: np.ndarray, y: np.ndarray):
    """Trim both signals to their shared length so per-sample metrics
    are well-defined even if processing changed the sample count."""
    n = min(len(x), len(y))
    return x[:n], y[:n]


def mse(x: np.ndarray, y: np.ndarray) -> float:
    """Mean Squared Error: average squared sample-by-sample difference.
    Lower is more similar; 0 means identical signals."""
    x, y = _align(x, y)
    return float(np.mean((x.astype(np.float64) - y.astype(np.float64)) ** 2))


def snr_db(reference: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB, treating `reference` as the clean
    signal and (test - reference) as the "noise" introduced by processing:

        SNR = 10 * log10( sum(reference^2) / sum((test - reference)^2) )

    Higher is better (test more closely matches reference).
    """
    reference, test = _align(reference, test)
    signal_power = np.sum(reference.astype(np.float64) ** 2)
    noise_power = np.sum((test.astype(np.float64) - reference.astype(np.float64)) ** 2)
    if noise_power <= 1e-20:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation coefficient in [-1, 1]: how linearly similar
    the two waveforms' shapes are, independent of absolute amplitude.
    1 = identical shape, 0 = unrelated, -1 = perfectly inverted."""
    x, y = _align(x, y)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
