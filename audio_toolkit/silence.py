"""Automatic silence detection and removal.

The goal is the one every podcast editor wants: find the long dead
stretches in a recording — the pause while someone reaches for a glass of
water, the run-up before the first word — and close them, without
touching the natural pauses that carry meaning.

WHY NOT JUST THRESHOLD THE WAVEFORM
------------------------------------
Comparing individual samples against a level fails immediately, because
every waveform crosses zero twice per cycle: a 100 Hz tone at full
volume is "below any threshold" 200 times a second. Silence is a
property of a *span* of audio, not of a sample, so the signal is first
reduced to a frame-rate envelope — the RMS level of each ~20 ms frame —
and the decision is made on that.

THE THRESHOLD IS RELATIVE, NOT ABSOLUTE
----------------------------------------
A fixed threshold like -45 dBFS is meaningless across recordings: a
quiet interview and a mastered track differ by 30 dB of gain before
anyone has said anything. The threshold here is therefore stated
relative to the loudest frame in the signal itself (the same convention
as librosa's `top_db`), so "-40 dB" means "40 dB below this recording's
own peak" and behaves the same on both files.

HYSTERESIS: ONE THRESHOLD IS NEVER ENOUGH
------------------------------------------
A single threshold makes the detector chatter. Real speech hovers right
around the boundary during the decay of a word, so the state flips
silent/loud/silent/loud many times a second, shattering one pause into
dozens of unusable fragments. The classic fix, borrowed from analogue
comparator design, is a Schmitt trigger: use a *higher* threshold to
declare audio present and a *lower* one to declare it absent. Between
the two the detector simply keeps its current opinion, so noise around
the boundary cannot flip it.

TWO GUARDS AGAINST OVER-EDITING
--------------------------------
`min_silence_s` — a pause must be genuinely long before it is a candidate
for removal. Short gaps are the rhythm of speech; deleting them produces
the breathless, unnaturally tight result that gives automatic silence
removal a bad name.

`pad_s` — even a long pause keeps a margin of silence at each end, so
words are not clipped and the edit keeps a natural breath around it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import editing
from .framing import frame_signal


@dataclass
class SilenceAnalysis:
    """What the detector saw, in enough detail to plot and to justify."""

    frame_times: np.ndarray
    level_db: np.ndarray
    is_silent: np.ndarray
    #: Spans that would actually be deleted (long enough, already padded).
    silent_regions: list[tuple[float, float]]
    #: Complement of the above — the spans that survive, in order.
    kept_regions: list[tuple[float, float]]
    threshold_db: float  # absolute dBFS, derived from peak_db + relative offset
    peak_db: float
    total_duration: float

    @property
    def removed_duration(self) -> float:
        return sum(end - start for start, end in self.silent_regions)

    @property
    def kept_duration(self) -> float:
        return sum(end - start for start, end in self.kept_regions)


def frame_levels(y: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0):
    """Per-frame RMS level in dBFS, plus each frame's centre time.

    RMS rather than peak: peak reacts to a single stray sample, while RMS
    measures the energy actually present over the frame, which is what
    "how loud is it here" means perceptually.
    """
    frames, times = frame_signal(y, sr, frame_ms, hop_ms)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return 20 * np.log10(rms + 1e-12), times


def suggest_threshold_db(y: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0) -> float:
    """Pick a threshold from the recording's own level distribution.

    A fixed "40 dB below peak" is a guess about the noise floor, and it is
    wrong in both directions: a clean studio take sits 60 dB below its
    peak and keeps its pauses, while a phone recording sits 20 dB down and
    has every pause missed. Neither is unusual.

    What is actually stable across recordings is the *shape* of the
    distribution. Audio with real pauses is bimodal — a cluster of frames
    at the noise floor and a cluster at speaking level — so the 10th
    percentile estimates the floor, the 90th estimates the content, and a
    threshold placed in the lower quarter of that gap separates them. The
    quarter, rather than the midpoint, biases toward keeping audio: the
    cost of missing a pause is a slightly long edit, while the cost of
    cutting too aggressively is clipped words.

    The result is returned relative to the peak, matching the units of
    `analyze_silence`'s `threshold_db`, so it can be dropped straight into
    a slider the user can then override.
    """
    level_db, _ = frame_levels(y, sr, frame_ms, hop_ms)
    if len(level_db) == 0:
        return -40.0

    floor_db, content_db = np.percentile(level_db, [10, 90])
    peak_db = float(np.max(level_db))
    gap = content_db - floor_db

    # Below this the distribution is one cluster, not two: continuous music,
    # or a tone whose level merely wanders. Splitting a unimodal
    # distribution would place the threshold inside the content and cut
    # away quiet passages, so fall back to a conservative fixed offset and
    # let the detector find nothing. Measured gaps on this project's demos:
    # continuous music 1.5 dB, a noisy tone 8 dB, speech with pauses 20 dB.
    if gap < 15.0:
        return -40.0

    absolute = floor_db + 0.25 * gap
    # Never call anything within 12 dB of the peak silence, however the
    # distribution looks — that is content by any reasonable definition.
    return float(np.clip(absolute - peak_db, -60.0, -12.0))


def _schmitt(level_db: np.ndarray, low: float, high: float) -> np.ndarray:
    """Dual-threshold state machine. True == silent.

    Starts in whichever state the first frame is unambiguously in, so a
    recording that opens mid-sentence is not mislabelled.
    """
    silent = np.empty(len(level_db), dtype=bool)
    state = bool(level_db[0] < high) if len(level_db) else True
    for i, level in enumerate(level_db):
        if state and level > high:
            state = False  # loud enough to be sure audio started
        elif not state and level < low:
            state = True  # quiet enough to be sure it stopped
        silent[i] = state
    return silent


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index spans [start, end) where `mask` is True."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _invert_regions(regions: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in regions:
        if start > cursor:
            kept.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        kept.append((cursor, total))
    return kept


def analyze_silence(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_s: float = 0.35,
    pad_s: float = 0.08,
    hysteresis_db: float = 6.0,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> SilenceAnalysis:
    """Find the silent stretches worth removing.

    threshold_db is relative to the loudest frame (negative, e.g. -40).
    hysteresis_db is the total gap between the two Schmitt thresholds:
    audio must exceed threshold + h/2 to count as present and fall below
    threshold - h/2 to count as absent.
    """
    total = len(y) / sr if sr else 0.0
    level_db, times = frame_levels(y, sr, frame_ms, hop_ms)

    peak_db = float(np.max(level_db)) if len(level_db) else -120.0
    absolute = peak_db + threshold_db
    high = absolute + hysteresis_db / 2
    low = absolute - hysteresis_db / 2

    is_silent = _schmitt(level_db, low, high)

    half_frame = frame_ms / 2000.0
    regions: list[tuple[float, float]] = []
    for i0, i1 in _runs(is_silent):
        start = max(0.0, float(times[i0]) - half_frame)
        end = min(total, float(times[i1 - 1]) + half_frame)
        if end - start < min_silence_s:
            continue  # a natural pause, not dead air
        start, end = start + pad_s, end - pad_s
        if end - start > 0:
            regions.append((start, end))

    kept = _invert_regions(regions, total)
    # Refuse to delete everything: a signal that is silent end to end (or
    # thresholded into oblivion) should come back untouched, not empty.
    if not kept:
        regions, kept = [], [(0.0, total)]

    return SilenceAnalysis(
        frame_times=times,
        level_db=level_db,
        is_silent=is_silent,
        silent_regions=regions,
        kept_regions=kept,
        threshold_db=absolute,
        peak_db=peak_db,
        total_duration=total,
    )


def remove_silence(
    y: np.ndarray,
    sr: int,
    analysis: SilenceAnalysis,
    crossfade_ms: float = editing.DEFAULT_DECLICK_MS,
) -> np.ndarray:
    """Splice the kept regions back together.

    Each join is a seam between material that was originally separated by
    the deleted pause, so every one is crossfaded. The sides are the same
    recording and therefore coherent, which is why the linear law applies.
    """
    if len(analysis.silent_regions) == 0:
        return np.asarray(y, dtype=np.float32)

    pieces = [
        np.asarray(y[int(start * sr) : int(end * sr)], dtype=np.float64)
        for start, end in analysis.kept_regions
    ]
    pieces = [p for p in pieces if len(p) > 0]
    if not pieces:
        return np.asarray(y, dtype=np.float32)

    n_cross = int(round(crossfade_ms * sr / 1000))
    out = pieces[0]
    for piece in pieces[1:]:
        out = np.asarray(editing.crossfade(out, piece, n_cross, law="linear"), dtype=np.float64)
    return out.astype(np.float32)
