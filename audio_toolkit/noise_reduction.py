"""Noise reduction via spectral subtraction.

METHOD
------
Classic spectral subtraction (Boll, 1979). Assumes background noise is
roughly stationary (its spectral shape doesn't change much over time)
and additive: noisy(t) = clean(t) + noise(t).

1. Estimate the noise's average magnitude spectrum |N(f)| from a
   segment known to contain only noise (e.g. the first ~0.5s before
   speech starts).
2. For every STFT frame of the full (noisy) signal, subtract that noise
   estimate from the magnitude spectrum, leaving phase untouched (the
   human ear is far less sensitive to phase than magnitude, so reusing
   the noisy phase is a standard, cheap approximation):

       |S_clean(f)| = max( |S_noisy(f)| - alpha*|N(f)| , beta*|S_noisy(f)| )

   `alpha` (oversubtraction factor) trades more noise removal for more
   speech distortion. `beta` is a spectral floor that prevents bins from
   being subtracted all the way to zero — without it, small subtraction
   errors leave isolated near-zero bins scattered across time/frequency,
   which the ear perceives as "musical noise" (random tonal chirps).
3. Reconstruct the audio via inverse STFT using the *original* noisy
   phase.

LIMITATIONS
-----------
- Only removes stationary noise (hums, hiss, fan/AC noise). Transient
  or time-varying noise (traffic, other voices, clicks) is not
  well-suppressed since the estimate doesn't track it.
- Requires a noise-only region to sample; a poor estimate (e.g. taken
  from a segment that already has speech in it) degrades results.
- Over-aggressive settings (`alpha` too high, `beta` too low) trade
  noise removal for audible musical-noise artifacts.
"""
from __future__ import annotations

import numpy as np
import librosa


def spectral_subtraction(
    y: np.ndarray,
    sr: int,
    noise_duration_s: float = 0.5,
    n_fft: int = 2048,
    hop_length: int = 512,
    alpha: float = 2.0,
    beta: float = 0.05,
):
    """Denoise y assuming the first `noise_duration_s` seconds are noise-only."""
    noise_samples = max(n_fft, int(noise_duration_s * sr))
    noise_clip = y[:noise_samples]

    D_noisy = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag_noisy, phase_noisy = np.abs(D_noisy), np.angle(D_noisy)

    D_noise = librosa.stft(noise_clip, n_fft=n_fft, hop_length=hop_length)
    noise_profile = np.mean(np.abs(D_noise), axis=1, keepdims=True)  # avg over noise frames

    subtracted = mag_noisy - alpha * noise_profile
    floor = beta * mag_noisy
    mag_clean = np.maximum(subtracted, floor)

    D_clean = mag_clean * np.exp(1j * phase_noisy)
    y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(y))
    return y_clean.astype(np.float32)
