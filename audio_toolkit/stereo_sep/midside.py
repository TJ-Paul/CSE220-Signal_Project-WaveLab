"""Stereo loading, validation and the Mid/Side transform.

THE TRANSFORM
-------------
    M(t) = (L(t) + R(t)) / 2      "Mid"  -- what the channels share
    S(t) = (L(t) - R(t)) / 2      "Side" -- how they differ

and its exact inverse

    L(t) = M(t) + S(t)
    R(t) = M(t) - S(t)

This is an orthogonal rotation of the (L, R) plane by 45 degrees (scaled by
1/2). It is lossless and perfectly invertible -- no information is created or
destroyed, only re-expressed. That is the entire reason to use it.

WHY IT HELPS
------------
Model the mixture as

    L = V_L + I_L
    R = V_R + I_R

For a source panned dead centre, its contribution is identical in both
channels (V_L = V_R = V), so:

    M = V + (I_L + I_R)/2        the vocal survives at FULL amplitude
    S = (I_L - I_R)/2            the vocal CANCELS COMPLETELY

S is therefore a vocal-free reference signal -- obtained exactly, with no
estimation, purely from the stereo geometry. That is an unusually strong
starting point, and it is available only if the stereo field is preserved.

WHAT IT IS NOT
--------------
    M = vocal        is FALSE
    S = instrumental is FALSE

M contains every centred source: kick, bass, snare, and often lead guitar or
piano. S contains only the *difference* between channels, so it omits all
centred accompaniment and is not a usable instrumental on its own -- playing
S alone sounds thin and hollow, missing its entire low end, which is exactly
the classic "vocal remover" failure.

Both are spatial *estimates*, used downstream as evidence among others.

WHERE THE ASSUMPTION BREAKS
---------------------------
1. Stereo/doubled/harmonised vocals: V_L != V_R, so cancellation in S is
   partial. The residual (V_L - V_R)/2 remains and no L-R method can remove it.
2. Stereo reverb on a centred dry vocal: the dry signal cancels, the reverb
   does not. The characteristic result is a karaoke track with no lead voice
   but an audible ghost of its reverb tail singing the melody.
3. Mono or near-mono recordings: S ~= 0, all spatial evidence vanishes, and
   the pipeline must fall back on non-spatial features. Detected and reported
   by `analyse_stereo` rather than silently producing garbage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf


@dataclass
class StereoReport:
    """Measured properties of the input stereo field."""

    n_channels_source: int
    sample_rate: int
    duration_s: float
    #: Pearson correlation between L and R over the whole file.
    #: ~1.0 => effectively mono (dual mono). ~0 => wide/decorrelated.
    #: Negative => channels partly out of phase (rare; often a mastering fault).
    lr_correlation: float
    #: Energy ratio 10*log10(E_S / E_M). Very negative means a narrow mix with
    #: little side information, so spatial separation will be weak.
    side_to_mid_db: float
    #: True when the two channels are identical (or the file was mono).
    #: In this case NO stereo-based separation is possible, by definition.
    is_effectively_mono: bool
    peak: float
    #: Fraction of samples at or beyond full scale in the source.
    clipping_fraction: float

    def summary(self) -> dict:
        return {
            "sourceChannels": self.n_channels_source,
            "sampleRate": self.sample_rate,
            "durationS": self.duration_s,
            "lrCorrelation": self.lr_correlation,
            "sideToMidDb": self.side_to_mid_db,
            "isEffectivelyMono": self.is_effectively_mono,
            "peak": self.peak,
            "clippingFraction": self.clipping_fraction,
        }


def load_stereo(path_or_buffer, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load audio as a (2, n) float32 array, PRESERVING the stereo field.

    This is the counterpart to io_utils.load_audio, which downmixes to mono.
    Downmixing computes (L+R)/2 = M and discards S irrecoverably; since S is
    the one signal guaranteed free of a centred vocal, that discard removes
    the most useful evidence the recording contains. Hence a separate loader.

    A genuinely mono source is duplicated into two identical channels so the
    rest of the pipeline has a uniform (2, n) contract. `analyse_stereo` then
    reports is_effectively_mono so callers can degrade gracefully instead of
    trusting spatial features that carry no information.
    """
    y, sr_out = librosa.load(path_or_buffer, sr=sr, mono=False)
    y = np.atleast_2d(np.asarray(y, dtype=np.float32))

    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    elif y.shape[0] > 2:
        # More than two channels (e.g. 5.1): fold to stereo by summing the
        # odd/even channel groups. Crude, but it keeps L/R distinction, which
        # a full downmix would not.
        left = y[0::2].mean(axis=0)
        right = y[1::2].mean(axis=0)
        y = np.stack([left, right])

    return np.ascontiguousarray(y, dtype=np.float32), int(sr_out)


def validate_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Check shape, rate and numerical sanity; return a clean (2, n) array.

    Non-finite samples (NaN/Inf) are replaced with zeros rather than allowed
    to propagate: a single NaN entering an FFT contaminates an entire frame,
    and after overlap-add that becomes a burst of silence or noise whose cause
    is very hard to trace back from the output.
    """
    y = np.atleast_2d(np.asarray(y, dtype=np.float32))
    if y.shape[0] != 2:
        raise ValueError(f"expected (2, n) stereo, got {y.shape}")
    if y.shape[1] == 0:
        raise ValueError("empty audio")
    if sr <= 0:
        raise ValueError(f"invalid sample rate {sr}")
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(y, dtype=np.float32)


def analyse_stereo(y: np.ndarray, sr: int, n_channels_source: int = 2) -> StereoReport:
    """Measure the stereo field before any processing decisions are made."""
    y = validate_audio(y, sr)
    L = y[0].astype(np.float64)
    R = y[1].astype(np.float64)

    if np.std(L) < 1e-12 or np.std(R) < 1e-12:
        corr = 1.0 if np.allclose(L, R) else 0.0
    else:
        corr = float(np.corrcoef(L, R)[0, 1])

    M, S = to_mid_side(y)
    e_m = float(np.sum(M.astype(np.float64) ** 2))
    e_s = float(np.sum(S.astype(np.float64) ** 2))
    side_db = float(10 * np.log10((e_s + 1e-20) / (e_m + 1e-20)))

    peak = float(np.max(np.abs(y)))
    clipped = float(np.mean(np.abs(y) >= 0.999))

    # Two tests, because either alone gives false positives: a wide mix can
    # still correlate highly at low frequencies, and a quiet-but-real side
    # channel can look negligible in energy while carrying useful geometry.
    effectively_mono = bool(corr > 0.999 or side_db < -60.0)

    return StereoReport(
        n_channels_source=n_channels_source,
        sample_rate=sr,
        duration_s=y.shape[1] / sr,
        lr_correlation=corr,
        side_to_mid_db=side_db,
        is_effectively_mono=effectively_mono,
        peak=peak,
        clipping_fraction=clipped,
    )


def to_mid_side(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(2, n) L/R  ->  (M, S). Lossless."""
    L, R = y[0], y[1]
    return (0.5 * (L + R)).astype(np.float32), (0.5 * (L - R)).astype(np.float32)


def to_left_right(M: np.ndarray, S: np.ndarray) -> np.ndarray:
    """(M, S) -> (2, n) L/R. Exact inverse of to_mid_side."""
    return np.stack([M + S, M - S]).astype(np.float32)


def mid_side_spectra(X_L: np.ndarray, X_R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mid/Side in the STFT domain.

        M(k,m) = (L(k,m) + R(k,m)) / 2
        S(k,m) = (L(k,m) - R(k,m)) / 2

    Valid because the DFT is linear, so the transform commutes with the
    Mid/Side rotation: doing M/S before or after the STFT gives the same
    result. Working in the STFT domain lets the M/S decision be made
    per frequency and per frame rather than once for the whole signal --
    which matters, since a mix's stereo width is strongly frequency
    dependent (bass centred, cymbals wide).
    """
    return 0.5 * (X_L + X_R), 0.5 * (X_L - X_R)
