"""Time-scale and pitch modification, and the measurement that proves it.

WHY SPEED AND PITCH ARE COUPLED AT ALL
---------------------------------------
Play a recording back faster by simply reading its samples at a higher
rate and you have compressed the time axis: every event in x(t) now
happens at x(rt). The Fourier scaling theorem says exactly what that
does to the spectrum —

        x(rt)  <-->  (1/|r|) X(f/r)

— every frequency component is multiplied by r. Speed the tape up by 2×
and a 220 Hz note becomes 440 Hz: one octave up, the "chipmunk" effect.
Time and pitch are two views of the same stretch, so separating them
takes real work.

THE PHASE VOCODER
------------------
To change duration while holding pitch, the signal is moved into the
short-time Fourier domain, where a frame's *content* (which frequencies
are present) is stored separately from *when* that frame occurs. The
STFT is then resynthesised with a different hop size: analyse every 512
samples, synthesise every 256, and the sound plays at half speed with
every frame's spectrum unchanged, so nothing is transposed.

The difficulty is phase. Each bin's phase must advance by the amount its
true frequency would accumulate over the *synthesis* hop, not the
analysis hop, or successive frames fight each other and the result turns
metallic and smeared. The fix is phase-gradient integration: estimate
each bin's instantaneous frequency from the phase difference between
consecutive analysis frames, then integrate that frequency across the
synthesis hop to produce a phase that stays coherent. This is
`librosa.effects.time_stretch`, and it is classical DSP — a transform,
an estimate, and an inverse transform. Nothing is learned from data.

The characteristic artefacts are worth knowing before a demo: transients
smear (a drum hit is spread over the frames it was stretched across) and
harmonics can lose their phase alignment relative to one another, which
is heard as a faint chorusing. Both grow with the stretch factor, so
0.8×–1.25× sounds near-transparent while 0.5× or 2× announces itself.

PITCH SHIFTING IS TIME STRETCHING, TWICE
-----------------------------------------
To move pitch without moving duration: stretch by the pitch ratio with
the phase vocoder (pitch unchanged, duration wrong), then resample by
the same ratio (pitch *and* duration both scaled). The duration errors
cancel and the pitch shifts cleanly. One semitone is a ratio of 2^(1/12),
because twelve equal steps must multiply to an octave.

VERIFYING THE RESULT
---------------------
Claims about pitch should be measured, not asserted, and this module
measures them two ways.

The primary method needs no pitch at all. A transposition multiplies
every frequency by the same ratio, and multiplication becomes *addition*
on a logarithmic axis — so transposing a signal slides its entire
spectrum sideways along a log-frequency axis without changing its shape.
Resampling the average spectrum onto a grid spaced in cents and
cross-correlating before against after therefore recovers the shift
directly, as the lag of the correlation peak. Because it matches the
whole spectral envelope rather than any single partial, it works on
polyphonic music, speech and noise alike, and it was measured against
known ground truth on this project's own demo signals to about one cent.

The secondary method is fundamental frequency, via YIN (Cheveigné &
Kawahara 2002): an autocorrelation method that searches for the lag at
which a signal best matches a delayed copy of itself, refined by a
cumulative mean normalised difference function that suppresses the
octave errors plain autocorrelation is prone to. F0 is only meaningful
for material that *has* one note at a time, so the estimate below is
gated on its own spread and declines to answer for chords or dense
mixes rather than reporting the median of several different notes.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import librosa
import numpy as np
from scipy import signal as sp_signal

#: Covers a low male speaking voice (~65 Hz) up to soprano / lead
#: instrument territory (~1050 Hz); outside this range YIN starts
#: reporting octave errors rather than useful answers.
F0_MIN_HZ = 65.0
F0_MAX_HZ = 1050.0

#: Resolution of the log-frequency grid used for transposition matching.
#: 10 cents is a twentieth of a semitone — finer than anyone can hear,
#: and parabolic interpolation of the correlation peak refines it further.
CENTS_PER_BIN = 10.0
LOG_GRID_FMIN_HZ = 55.0
LOG_GRID_FMAX_HZ = 8000.0

#: An F0 track whose interquartile spread exceeds this is not one note —
#: it is a chord, a melody, or an octave-confused estimator. Measured
#: spreads on this project's demos: steady tone ~0, solo melody ~315,
#: speech ~770, full mix ~2360 cents.
F0_STABILITY_LIMIT_CENTS = 150.0


def semitone_ratio(semitones: float) -> float:
    """Frequency ratio of a shift in equal temperament: 2^(n/12)."""
    return float(2 ** (semitones / 12))


def cents_between(reference_hz: float | None, test_hz: float | None) -> float | None:
    """Pitch distance in cents — 1200 per octave, 100 per semitone.

    Cents rather than hertz because pitch is perceived multiplicatively:
    10 Hz is an octave at the bottom of a bass and a rounding error at
    the top of a violin. Roughly 5–10 cents is the threshold of audibility
    for a sustained tone.
    """
    if not reference_hz or not test_hz or reference_hz <= 0 or test_hz <= 0:
        return None
    return float(1200 * np.log2(test_hz / reference_hz))


@dataclass
class PitchEstimate:
    """A fundamental frequency reading, with the evidence for trusting it."""

    f0_hz: float | None
    #: Interquartile spread of the F0 track, in cents. Small means one
    #: steady note; large means the estimator was handed several at once.
    spread_cents: float | None
    stable: bool


def estimate_f0(
    y: np.ndarray,
    sr: int,
    fmin: float = F0_MIN_HZ,
    fmax: float = F0_MAX_HZ,
    max_seconds: float = 12.0,
) -> PitchEstimate:
    """Fundamental frequency over the voiced part of the signal.

    Three refinements make this fit to quote as evidence:

    1. Only frames within 25 dB of the loudest frame are counted. Silence
       and room tone have no pitch, but YIN will still return *some* lag
       for them, and those values would drag the median around.
    2. The median, not the mean — a handful of octave errors then cannot
       move the answer, whereas a mean would split the difference and
       report a pitch that never occurred.
    3. The interquartile spread is reported alongside, and `stable` is
       False when it is wide. A chord has no single F0; measuring one
       anyway produces a number that looks authoritative and means
       nothing. On a dense mix this correctly declines to answer, and
       `estimate_transposition_cents` should be used instead.
    """
    y = np.asarray(y, dtype=np.float32)
    frame_length, hop_length = 2048, 512
    if sr <= 0 or len(y) < frame_length:
        return PitchEstimate(None, None, False)
    if len(y) > max_seconds * sr:
        y = y[: int(max_seconds * sr)]

    try:
        f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                         frame_length=frame_length, hop_length=hop_length)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    except Exception:
        return PitchEstimate(None, None, False)

    n = min(len(f0), len(rms))
    f0, rms = f0[:n], rms[:n]
    if n == 0:
        return PitchEstimate(None, None, False)

    rms_db = 20 * np.log10(rms + 1e-12)
    voiced = (rms_db > rms_db.max() - 25) & np.isfinite(f0) & (f0 > fmin * 1.01) & (f0 < fmax * 0.99)
    if voiced.sum() < 3:
        return PitchEstimate(None, None, False)

    values = f0[voiced]
    q1, q3 = np.percentile(values, [25, 75])
    spread = float(1200 * np.log2(q3 / q1)) if q1 > 0 else None
    stable = spread is not None and spread <= F0_STABILITY_LIMIT_CENTS
    return PitchEstimate(float(np.median(values)), spread, stable)


def _resample_by_ratio(y: np.ndarray, ratio: float) -> np.ndarray:
    """Read the signal back `ratio`× faster, anti-aliased.

    Implemented with a polyphase rational resampler rather than plain
    interpolation because speeding audio up compresses its spectrum
    *upward*: content that was just under Nyquist is pushed past it and
    folds back as aliasing. `resample_poly` runs the required low-pass
    filter as part of the rate change, so the content that cannot survive
    is removed rather than mirrored back into the audible band.
    """
    frac = Fraction(1.0 / ratio).limit_denominator(2000)
    up, down = frac.numerator, frac.denominator
    if up == down or up == 0:
        return np.asarray(y, dtype=np.float32)
    return sp_signal.resample_poly(np.asarray(y, dtype=np.float64), up, down).astype(np.float32)


def change_speed(y: np.ndarray, sr: int, rate: float, preserve_pitch: bool = True) -> np.ndarray:
    """Retime the signal by `rate` (>1 faster, <1 slower).

    preserve_pitch=True  — phase vocoder. Duration changes, pitch does not.
                           The musically useful mode, and the one with
                           artefacts to listen for.
    preserve_pitch=False — resampling, i.e. varispeed / tape. Duration and
                           pitch both scale by `rate`. Mathematically exact
                           (no estimation anywhere), which is why it stays
                           clean at extreme ratios where the vocoder does not.
    """
    y = np.asarray(y, dtype=np.float32)
    rate = float(np.clip(rate, 0.25, 4.0))
    if abs(rate - 1.0) < 1e-6 or len(y) == 0:
        return y.copy()

    if not preserve_pitch:
        return _resample_by_ratio(y, rate)

    stretched = librosa.effects.time_stretch(y, rate=rate)
    return np.asarray(stretched, dtype=np.float32)


def shift_pitch(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Transpose by `semitones` while holding duration constant."""
    y = np.asarray(y, dtype=np.float32)
    semitones = float(np.clip(semitones, -24.0, 24.0))
    if abs(semitones) < 1e-6 or len(y) == 0:
        return y.copy()
    shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)
    return np.asarray(shifted, dtype=np.float32)


@dataclass
class Transposition:
    """How far a signal's spectrum moved, and how well the match held up."""

    cents: float | None
    #: Peak of the normalised cross-correlation, in [-1, 1]. Near 1 means
    #: the two spectra really are the same shape at an offset; a low value
    #: means the processing changed the spectrum's shape as well as its
    #: position, so the reported offset is a weaker claim.
    confidence: float | None


def _log_frequency_spectrum(y: np.ndarray, sr: int) -> np.ndarray | None:
    """Average magnitude spectrum resampled onto a grid spaced in cents.

    Averaging across STFT frames is what makes this comparable before and
    after a time stretch: the two signals have different durations, so any
    frame-by-frame comparison would be aligning different moments. The
    average discards timing and keeps the spectral envelope, which is
    exactly the part a transposition translates.

    The log is taken before comparison so that the quiet upper harmonics
    count as much as the loud fundamental — on a linear magnitude scale
    the correlation would be decided almost entirely by the few loudest
    bins.
    """
    y = np.asarray(y, dtype=np.float32)
    n_fft = 4096
    if len(y) < n_fft:
        return None

    spectrum = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4)).mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    fmax = min(LOG_GRID_FMAX_HZ, sr / 2 * 0.98)
    if fmax <= LOG_GRID_FMIN_HZ:
        return None
    n_bins = int(1200 * np.log2(fmax / LOG_GRID_FMIN_HZ) / CENTS_PER_BIN)
    grid = LOG_GRID_FMIN_HZ * 2 ** (np.arange(n_bins) * CENTS_PER_BIN / 1200)

    magnitude = np.log(np.interp(grid, freqs, spectrum) + 1e-10)
    return magnitude - magnitude.mean()


def estimate_transposition_cents(
    reference: np.ndarray,
    test: np.ndarray,
    sr: int,
    max_cents: float = 2600.0,
) -> Transposition:
    """Measure how far `test` is transposed relative to `reference`.

    Works by sliding one log-frequency spectrum across the other and
    taking the lag of the best normalised correlation, then refining that
    lag with parabolic interpolation through its two neighbours — the
    standard trick for locating a correlation peak to a fraction of a bin.

    Unlike F0 this needs no pitch to be present and no assumption of
    monophony, so it is the right tool for verifying what a stretch or a
    shift did to real music.
    """
    a = _log_frequency_spectrum(reference, sr)
    b = _log_frequency_spectrum(test, sr)
    if a is None or b is None:
        return Transposition(None, None)

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    max_lag = int(max_cents / CENTS_PER_BIN)
    if n < 4 * max_lag // 3:
        max_lag = max(1, n // 3)

    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.full(len(lags), -1.0)
    for i, lag in enumerate(lags):
        x, y2 = (a[: n - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[: n + lag])
        if len(x) < 32:
            continue
        denom = np.linalg.norm(x) * np.linalg.norm(y2)
        if denom > 1e-12:
            scores[i] = float(np.dot(x, y2) / denom)

    k = int(np.argmax(scores))
    if scores[k] <= -1.0:
        return Transposition(None, None)

    delta = 0.0
    if 0 < k < len(scores) - 1:
        y0, y1, y2_ = scores[k - 1], scores[k], scores[k + 1]
        curvature = y0 - 2 * y1 + y2_
        if abs(curvature) > 1e-12:
            delta = float(np.clip(0.5 * (y0 - y2_) / curvature, -1.0, 1.0))

    return Transposition(float((lags[k] + delta) * CENTS_PER_BIN), float(scores[k]))
