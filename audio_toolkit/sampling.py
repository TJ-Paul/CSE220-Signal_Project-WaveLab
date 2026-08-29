"""Sampling, aliasing, and reconstruction demonstrations.

The Nyquist-Shannon sampling theorem: a continuous signal band-limited
to f_max can be perfectly reconstructed from samples taken at a rate
sr >= 2*f_max (the Nyquist rate). Sampling below that rate causes
aliasing — frequencies above sr/2 fold back and masquerade as lower
("alias") frequencies, an effect that cannot be undone after the fact.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def generate_tone(freq_hz: float, duration_s: float, fs_reference: int = 44100):
    """A "continuous-time" reference signal: a sine sampled at a very high
    rate, standing in for the true analog signal for plotting/comparison."""
    t = np.arange(0, duration_s, 1.0 / fs_reference)
    x = np.sin(2 * np.pi * freq_hz * t)
    return t, x


def sample_signal(freq_hz: float, duration_s: float, sample_rate: int):
    """Sample a sine tone at `sample_rate`. If sample_rate < 2*freq_hz
    (below Nyquist), the discrete samples alone are indistinguishable
    from a lower-frequency sinusoid — that lower frequency is what a
    DAC/reconstruction filter will actually reproduce."""
    t = np.arange(0, duration_s, 1.0 / sample_rate)
    x = np.sin(2 * np.pi * freq_hz * t)
    return t, x


def aliased_frequency(freq_hz: float, sample_rate: int) -> float:
    """The frequency an under-sampled tone appears as after reconstruction.

    Frequencies fold around multiples of the Nyquist frequency (sr/2):
    any true frequency f maps to the alias in [0, sr/2] obtained by
    reflecting f into that range off multiples of sr.
    """
    nyquist = sample_rate / 2.0
    folded = freq_hz % sample_rate
    if folded > nyquist:
        folded = sample_rate - folded
    return folded


def reconstruct_signal(t_samples: np.ndarray, x_samples: np.ndarray, t_target: np.ndarray):
    """Reconstruct a continuous-looking signal from discrete samples via
    band-limited (sinc) interpolation — the theoretical ideal
    reconstruction filter for a perfectly sampled signal.

    Implemented as a direct Whittaker-Shannon sinc sum, which is O(N*M)
    but exact and simple to reason about for demo-sized signals.
    """
    if len(t_samples) < 2:
        return np.zeros_like(t_target)
    ts = t_samples[1] - t_samples[0]
    # sinc((t - t_n)/Ts) summed over all sample points n, weighted by x[n]
    sinc_matrix = np.sinc((t_target[:, None] - t_samples[None, :]) / ts)
    return sinc_matrix @ x_samples


def resample_audio(y: np.ndarray, orig_sr: int, new_sr: int):
    """Practical resampling (used for real audio, unlike the sinc demo
    above) via scipy's polyphase filter — resamples using rational
    factors with an anti-aliasing low-pass filter built in."""
    if orig_sr == new_sr:
        return y.copy()
    gcd = np.gcd(orig_sr, new_sr)
    up, down = new_sr // gcd, orig_sr // gcd
    return signal.resample_poly(y, up, down).astype(np.float32)
