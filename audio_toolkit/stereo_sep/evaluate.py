"""Objective evaluation of separation quality.

WITH GROUND TRUTH: the BSS_EVAL decomposition (Vincent, Gribonval & Fevotte
2006). The estimate is projected onto subspaces spanned by the true sources:

    s_hat = s_target + e_interf + e_artif

    SDR = 10 log10( ||s_target||^2 / ||e_interf + e_artif||^2 )   overall
    SIR = 10 log10( ||s_target||^2 / ||e_interf||^2 )             bleed
    SAR = 10 log10( ||s_target + e_interf||^2 / ||e_artif||^2 )   artefacts

The three separate concerns that a single number confuses. A mask can raise
SIR (less of the other source) while lowering SAR (more musical noise), and
that trade is exactly what the tuning parameters control -- so measuring only
SDR hides the decision being made.

Projections are computed with a least-squares filter of order `flen` rather
than a plain inner product, which allows the estimate to differ from the
truth by a short filter (a small delay or EQ) without being penalised. That
is the standard formulation, and it matters here because masking inherently
applies a time-varying filter.

WITHOUT GROUND TRUTH: proxy measures only. They are explicitly NOT quality
scores -- see `proxy_metrics`.
"""
from __future__ import annotations

import numpy as np


def _to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.mean(axis=0) if x.ndim > 1 else x


def _align(*sigs):
    n = min(len(s) for s in sigs)
    return [s[:n] for s in sigs]


def _proj(y: np.ndarray, refs: list[np.ndarray], flen: int) -> np.ndarray:
    """Least-squares projection of y onto the subspace of filtered references.

    Builds the Toeplitz design matrix of every reference at every lag up to
    flen, then solves the normal equations. lstsq is used rather than an
    explicit inverse because the design matrix is often ill-conditioned when
    references are correlated (as bass and kick are), and lstsq handles the
    rank deficiency gracefully instead of producing a huge spurious solution.
    """
    n = len(y)
    cols = []
    for r in refs:
        for lag in range(flen):
            shifted = np.zeros(n)
            if lag == 0:
                shifted[:] = r[:n]
            else:
                shifted[lag:] = r[: n - lag]
            cols.append(shifted)
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


def bss_eval(
    estimate: np.ndarray,
    target: np.ndarray,
    interference: np.ndarray,
    flen: int = 32,
) -> dict[str, float]:
    """SDR / SIR / SAR in dB for one estimated stem.

    `target` is the ground-truth source this stem is meant to be;
    `interference` is the other source. Higher is better for all three.

    flen is the allowed filter length. 32 taps is short enough to stay
    honest (it cannot fabricate a match) and long enough to absorb the small
    delay and colouration any masking scheme introduces.
    """
    est = _to_mono(estimate)
    tgt = _to_mono(target)
    itf = _to_mono(interference)
    est, tgt, itf = _align(est, tgt, itf)

    if np.allclose(est, 0) or np.allclose(tgt, 0):
        return {"sdrDb": float("-inf"), "sirDb": float("-inf"), "sarDb": float("-inf")}

    s_target = _proj(est, [tgt], flen)
    s_both = _proj(est, [tgt, itf], flen)

    e_interf = s_both - s_target
    e_artif = est - s_both

    def db(num, den):
        num = float(np.sum(num**2)); den = float(np.sum(den**2))
        if den <= 1e-20:
            return float("inf")
        if num <= 1e-20:
            return float("-inf")
        return float(10 * np.log10(num / den))

    return {
        "sdrDb": db(s_target, e_interf + e_artif),
        "sirDb": db(s_target, e_interf),
        "sarDb": db(s_target + e_interf, e_artif),
    }


def evaluate_result(result, true_instrumental, true_vocals, flen: int = 32) -> dict:
    """BSS_EVAL for both stems of a SeparationResult."""
    return {
        "method": result.method,
        "methodName": result.method_name,
        "instrumental": bss_eval(result.instrumental, true_instrumental, true_vocals, flen),
        "vocals": bss_eval(result.vocals, true_vocals, true_instrumental, flen),
    }


# ---------------------------------------------------------------------------
# No ground truth available
# ---------------------------------------------------------------------------

def proxy_metrics(mixture: np.ndarray, instrumental: np.ndarray,
                  vocals: np.ndarray, sr: int) -> dict:
    """Measurements computable without ground truth.

    READ THIS BEFORE USING THEM. None of these is a quality score, and each
    can be made to look excellent by a separator that is in fact useless:

      - `vocalBandSuppressionDb` measures energy lost in the 300-3400 Hz
        speech band. Deleting that band entirely maximises it and destroys
        the track. It only tells you SOMETHING was removed, never whether it
        was the voice.

      - `stemCorrelation` is near 0 when stems carry different content, but
        two stems sharing the mixture phase are correlated by construction,
        so a low value is necessary-but-not-sufficient.

      - `energyConservation` compares the summed stems against the mixture.
        Read it ONLY with level matching disabled, where it should sit near
        1: the stems ought to roughly sum back to the mixture, and a value
        far from 1 means energy was lost or duplicated -- a real bug signal,
        and how a non-partitioning pair of masks gets caught. With
        `level_match` on, both stems are independently rescaled to the
        mixture's RMS, so values of 2-4 are expected and mean nothing.

      - `spectralHoleFraction` counts bins driven near silence. High values
        predict audible musical noise even when every other number looks good.

    These are diagnostics for catching failure, not evidence of success.
    Judge quality by listening; use these to know where to listen.
    """
    mix = _to_mono(mixture); inst = _to_mono(instrumental); voc = _to_mono(vocals)
    mix, inst, voc = _align(mix, inst, voc)

    def band_rms(x, lo, hi):
        X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / sr)
        Y = X.copy(); Y[(f < lo) | (f > hi)] = 0
        return float(np.sqrt(np.mean(np.fft.irfft(Y, n=len(x)) ** 2)))

    mix_b = band_rms(mix, 300, 3400)
    inst_b = band_rms(inst, 300, 3400)

    summed = inst + voc
    e_mix = float(np.sum(mix**2)); e_sum = float(np.sum(summed**2))

    corr = 0.0
    if np.std(inst) > 1e-12 and np.std(voc) > 1e-12:
        corr = abs(float(np.corrcoef(inst, voc)[0, 1]))

    from scipy import signal as sps
    _, _, Z = sps.stft(voc, fs=sr, nperseg=1024)
    mag = np.abs(Z)
    thresh = np.max(mag) * 1e-4
    holes = float(np.mean(mag < thresh))

    return {
        "vocalBandSuppressionDb": float(20 * np.log10((mix_b + 1e-12) / (inst_b + 1e-12))),
        "stemCorrelation": corr,
        "energyConservation": float(e_sum / (e_mix + 1e-20)),
        "spectralHoleFraction": holes,
        "instrumentalPeak": float(np.max(np.abs(inst))),
        "vocalsPeak": float(np.max(np.abs(voc))),
        "note": "Proxy measures only. None indicates quality; use for diagnosis.",
    }


LISTENING_GUIDE = """
HOW TO JUDGE THESE OUTPUTS BY EAR
=================================
Numbers cannot settle this. Separation artefacts are perceptual, and two
files with identical SDR can sound completely different. Listen for:

KARAOKE / INSTRUMENTAL STEM
  1. Is the BASS still there? Play it after the original and listen to the
     low end specifically. Loss of bass is the most common failure and the
     one that most damages a karaoke track.
  2. Does the SNARE still crack, or has it gone soft and hollow? A centred
     snare sits exactly where the vocal-suppression logic is most confident.
  3. Residual vocal: a faint voice is acceptable. A vocal REVERB tail with no
     dry voice -- a ghost singing the melody -- means the vocal was stereo or
     heavily reverberated, and no L/R method can remove it.
  4. Warbling or "underwater" sound between notes indicates the masks are too
     aggressive. Lower vocal_suppression or raise smooth_time_frames.

VOCAL STEM
  1. Are CONSONANTS intact? Listen to "t", "k", "s". If the words are mumbled
     the mask is deleting broadband transients -- raise vocal_preservation.
  2. Telephone-like or boxy tone means the low fundamental or the high
     partials are being cut. Check the harmonic guard.
  3. Metallic ringing or chirping between phrases is musical noise -- raise
     mask_floor and the smoothing kernels.
  4. Drum bleed is expected and is usually the correct trade against losing
     consonants.

METHOD COMPARISON
  Compare m1 against m5 on the SAME file at MATCHED LEVEL. If m5 does not
  sound clearly better, the material likely breaks its assumptions -- a mono
  or very narrow mix defeats the spatial features, and through-composed music
  defeats the repetition feature. In that case the honest answer is that
  classical DSP has little to work with on that track.
"""
