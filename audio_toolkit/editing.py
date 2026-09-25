"""Sample-exact edit primitives: trim, cut, splice, merge and fade.

Every operation in this module is plain sample arithmetic — no transform,
no estimation, nothing to tune. The audio that survives an edit is
bit-identical to the audio that went in. That makes the *seam* the only
place where quality can be lost, and seams are what this module is
really about.

THE SEAM PROBLEM
----------------
A waveform is a continuous pressure curve. Cut it at an arbitrary sample
and butt two pieces together and the result almost always contains a step
discontinuity — the signal jumps from, say, +0.42 to -0.31 within a
single sample period. In the frequency domain a step is catastrophically
broadband: its spectrum rolls off as only 1/f, so energy is smeared
across the entire audible range. That is heard as a click, and no amount
of care about *where* the edit lands removes it.

The fix is to never let the amplitude jump. Every function below that
creates a seam ramps one side down while ramping the other up, over a
few milliseconds — short enough to be inaudible as a level change, long
enough that the fastest slope in the signal is now finite.

TWO CROSSFADE LAWS, AND WHY THE CHOICE MATTERS
----------------------------------------------
When two signals overlap during a crossfade, what the listener perceives
depends on whether those signals are correlated.

- **Linear** gains (g_out = 1-t, g_in = t) sum to 1 at every instant, so
  two *coherent* signals — the same music, the same phase, as in a cut
  inside one continuous take — hold a constant amplitude across the
  join. Use this when the two sides are the same source.

- **Equal-power** gains (g_out = cos(t·π/2), g_in = sin(t·π/2)) satisfy
  g_out² + g_in² = 1, so the *power* sum is constant. Two unrelated
  signals — different songs, different takes — add incoherently, meaning
  their powers add rather than their amplitudes; a linear fade between
  them would dip about 3 dB in the middle. Use this when merging
  different sources.

Picking the wrong law is not fatal, it just costs a shallow audible dip
(or bump) at the join, which is precisely why both are exposed.
"""
from __future__ import annotations

import numpy as np

FADE_SHAPES = ("linear", "exponential", "logarithmic", "scurve", "equal_power")
CROSSFADE_LAWS = ("equal_power", "linear")

#: Short enough to be inaudible as a level change, long enough to bound
#: the waveform slope at the join. 5 ms ≈ 220 samples at 44.1 kHz.
DEFAULT_DECLICK_MS = 5.0


def fade_ramp(n: int, shape: str = "linear") -> np.ndarray:
    """A rising gain ramp from 0 to 1 over `n` samples.

    The shape is a perceptual choice, not a technical one — all five
    reach the same endpoints, they differ in how the loudness travels
    between them. Human loudness perception is roughly logarithmic, so a
    gain that rises linearly in *amplitude* is heard as rising fast at
    the start and crawling at the end.

    linear       g = t              amplitude-linear; the familiar default
    exponential  g = t²             slow, late-blooming start
    logarithmic  g = √t             quick start, long tail — good for fade-outs
                                    that must not sound like a cliff
    scurve       g = (1-cos πt)/2   raised cosine; zero slope at *both* ends,
                                    the smoothest option and the one least
                                    likely to be noticed at all
    equal_power  g = sin(tπ/2)      pairs with cos to hold constant power
    """
    if n <= 0:
        return np.ones(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    t = np.linspace(0.0, 1.0, n)
    if shape == "linear":
        return t
    if shape == "exponential":
        return t**2
    if shape == "logarithmic":
        return np.sqrt(t)
    if shape == "scurve":
        return 0.5 - 0.5 * np.cos(np.pi * t)
    if shape == "equal_power":
        return np.sin(t * np.pi / 2)
    raise ValueError(f"Unknown fade shape '{shape}' (expected one of {FADE_SHAPES})")


def apply_fades(
    y: np.ndarray,
    sr: int,
    fade_in_s: float = 0.0,
    fade_out_s: float = 0.0,
    shape: str = "scurve",
) -> np.ndarray:
    """Return a copy of `y` with fades applied at its head and tail.

    The two fades are clamped so they can never overlap: on a 2 s clip a
    requested 1.5 s in and 1.5 s out become 1.0 s each, preserving the
    ratio the user asked for rather than silently dropping one.
    """
    out = np.array(y, dtype=np.float64, copy=True)
    n = len(out)
    if n == 0:
        return out.astype(np.float32)

    n_in = max(0, min(n, int(round(fade_in_s * sr))))
    n_out = max(0, min(n, int(round(fade_out_s * sr))))
    if n_in + n_out > n:
        scale = n / (n_in + n_out)
        n_in, n_out = int(n_in * scale), int(n_out * scale)

    if n_in > 0:
        out[:n_in] *= fade_ramp(n_in, shape)
    if n_out > 0:
        out[n - n_out :] *= fade_ramp(n_out, shape)[::-1]
    return out.astype(np.float32)


def _crossfade_gains(n: int, law: str) -> tuple[np.ndarray, np.ndarray]:
    """(outgoing, incoming) gain pair for an n-sample overlap."""
    t = np.linspace(0.0, 1.0, n)
    if law == "linear":
        return 1.0 - t, t
    if law == "equal_power":
        return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    raise ValueError(f"Unknown crossfade law '{law}' (expected one of {CROSSFADE_LAWS})")


def crossfade(a: np.ndarray, b: np.ndarray, n: int, law: str = "equal_power") -> np.ndarray:
    """Join `a` to `b` with an `n`-sample overlap.

    The result is `len(a) + len(b) - n` samples long: a crossfade buys its
    click-free seam by consuming material from both sides. Callers that
    must preserve total length should use a butt join with declick fades
    (see `splice` with crossfade_s = 0) instead.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = int(min(n, len(a), len(b)))
    if n <= 0:
        return np.concatenate([a, b]).astype(np.float32)

    g_out, g_in = _crossfade_gains(n, law)
    middle = a[len(a) - n :] * g_out + b[:n] * g_in
    return np.concatenate([a[: len(a) - n], middle, b[n:]]).astype(np.float32)


def splice(
    segments: list[np.ndarray],
    sr: int,
    crossfade_s: float = 0.0,
    law: str = "equal_power",
    gap_s: float = 0.0,
) -> np.ndarray:
    """Join a list of segments end to end, declicking every seam.

    With `crossfade_s > 0` neighbours overlap and fade through each other.
    With `crossfade_s == 0` they are butted together, but each side still
    gets a ~5 ms micro-fade first: a hard butt of two unrelated buffers is
    exactly the step discontinuity described at the top of this module.
    `gap_s` inserts digital silence between segments, which needs no
    crossfade because silence has no discontinuity to hide.
    """
    clips = [np.asarray(s, dtype=np.float64) for s in segments if len(s) > 0]
    if not clips:
        return np.zeros(0, dtype=np.float32)
    if len(clips) == 1:
        return clips[0].astype(np.float32)

    gap = np.zeros(max(0, int(round(gap_s * sr))), dtype=np.float64)
    n_cross = int(round(crossfade_s * sr))

    if n_cross <= 0 or len(gap) > 0:
        declick = int(round(DEFAULT_DECLICK_MS * sr / 1000))
        joined: list[np.ndarray] = []
        for i, clip in enumerate(clips):
            # Only the edges that actually meet another clip need tapering;
            # the outermost head and tail are left exactly as the user cut them.
            head = declick if i > 0 else 0
            tail = declick if i < len(clips) - 1 else 0
            joined.append(apply_fades(clip, sr, head / sr, tail / sr, "scurve").astype(np.float64))
            if len(gap) and i < len(clips) - 1:
                joined.append(gap)
        return np.concatenate(joined).astype(np.float32)

    out = clips[0]
    for clip in clips[1:]:
        out = np.asarray(crossfade(out, clip, n_cross, law), dtype=np.float64)
    return out.astype(np.float32)


def resolve_span(y: np.ndarray, sr: int, start_s: float, end_s: float) -> tuple[int, int]:
    """Clamp a time span to valid, non-empty sample indices."""
    start = int(np.clip(round(start_s * sr), 0, len(y)))
    end = int(np.clip(round(end_s * sr), 0, len(y)))
    if end <= start:
        end = min(len(y), start + 1)
    return start, end


def trim(y: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    """Keep only [start_s, end_s). No seam is created inside the audio, so
    no crossfade is needed — the cut edges become the new file boundaries."""
    start, end = resolve_span(y, sr, start_s, end_s)
    return np.array(y[start:end], dtype=np.float32, copy=True)


def cut(
    y: np.ndarray,
    sr: int,
    start_s: float,
    end_s: float,
    crossfade_ms: float = DEFAULT_DECLICK_MS,
) -> np.ndarray:
    """Delete [start_s, end_s) and close the gap.

    Unlike `trim` this *does* create an interior seam, between material
    that was previously seconds apart, so the two sides are crossfaded.
    Both sides come from one continuous recording and are therefore
    treated as coherent — hence the linear law.
    """
    start, end = resolve_span(y, sr, start_s, end_s)
    head, tail = y[:start], y[end:]
    if len(head) == 0:
        return np.array(tail, dtype=np.float32, copy=True)
    if len(tail) == 0:
        return np.array(head, dtype=np.float32, copy=True)
    return crossfade(head, tail, int(round(crossfade_ms * sr / 1000)), law="linear")


def peak_normalize(y: np.ndarray, target_dbfs: float = -1.0) -> np.ndarray:
    """Scale so the loudest sample sits at `target_dbfs`.

    Peak normalisation changes only the gain — every sample is multiplied
    by one constant, so the waveform shape, and with it the timbre, is
    untouched. A silent buffer is returned unchanged rather than amplified
    into its own noise floor.
    """
    y64 = np.asarray(y, dtype=np.float64)
    peak = float(np.max(np.abs(y64))) if len(y64) else 0.0
    if peak < 1e-9:
        return np.asarray(y, dtype=np.float32)
    return (y64 * (10 ** (target_dbfs / 20) / peak)).astype(np.float32)


def match_loudness(y: np.ndarray, reference: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    """Scale `y` to the RMS level of `reference`, without letting it clip.

    This exists so A/B comparisons are honest. Processing that removes
    energy — vocal removal, filtering, gating — leaves the output quieter,
    and a quieter version of the same audio is reliably judged as worse
    regardless of what the processing actually did. Matching levels first
    means the listener compares the processing, not the gain.
    """
    y64 = np.asarray(y, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if len(y64) == 0:
        return np.asarray(y, dtype=np.float32)

    rms_y = float(np.sqrt(np.mean(y64**2)))
    rms_ref = float(np.sqrt(np.mean(ref**2))) if len(ref) else 0.0
    if rms_y < 1e-9 or rms_ref < 1e-9:
        return np.asarray(y, dtype=np.float32)

    scaled = y64 * (rms_ref / rms_y)
    peak = float(np.max(np.abs(scaled)))
    if peak > ceiling:
        scaled *= ceiling / peak
    return scaled.astype(np.float32)
