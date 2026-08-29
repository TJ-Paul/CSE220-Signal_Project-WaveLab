"""Voice Activity Detection (VAD): classify frames as Speech or Silence.

Two complementary features are combined:

1. Short-time energy  E_n = (1/N) * sum(x_n[i]^2)
   Speech has substantially higher energy than background silence/noise,
   but energy alone is fooled by loud non-speech noise bursts.

2. A frequency-domain feature — the fraction of a frame's FFT energy
   that falls in the speech band (~300-3400 Hz, the classic telephony
   band where voiced/unvoiced speech energy concentrates). Noise bursts
   (e.g. broadband clicks, low-frequency hums) tend to score low on
   this ratio even if their raw energy is high, so it helps reject
   false positives that energy thresholding alone would accept.

A frame is labeled Speech only if it clears an energy threshold AND a
speech-band-ratio threshold. Thresholds are set adaptively from the
frame energy distribution (percentile-based) so the detector is not
hand-tuned to one specific recording's loudness/noise floor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .framing import frame_signal


@dataclass
class VadResult:
    frame_times: np.ndarray
    energy_db: np.ndarray
    speech_band_ratio: np.ndarray
    is_speech: np.ndarray
    speech_segments: list[tuple[float, float]]  # (start_s, end_s)


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """E_n = mean(x^2) per frame — average power of the frame."""
    return np.mean(frames.astype(np.float64) ** 2, axis=1)


def speech_band_energy_ratio(frames: np.ndarray, sr: int, band=(300.0, 3400.0)) -> np.ndarray:
    """Fraction of each frame's FFT magnitude energy inside `band` Hz."""
    n = frames.shape[1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag2 = np.abs(np.fft.rfft(frames, axis=1)) ** 2  # per-frame power spectrum

    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    total = mag2.sum(axis=1) + 1e-12
    in_band = mag2[:, band_mask].sum(axis=1)
    return in_band / total


def _merge_segments(is_speech: np.ndarray, frame_times: np.ndarray, frame_ms: float, hop_ms: float):
    """Merge consecutive Speech frames into contiguous (start, end) segments."""
    segments = []
    half_frame_s = frame_ms / 2000.0
    in_seg = False
    seg_start = 0.0
    for i, speech in enumerate(is_speech):
        if speech and not in_seg:
            in_seg = True
            seg_start = frame_times[i] - half_frame_s
        elif not speech and in_seg:
            in_seg = False
            segments.append((max(0.0, seg_start), frame_times[i - 1] + half_frame_s))
    if in_seg:
        segments.append((max(0.0, seg_start), frame_times[-1] + half_frame_s))
    return segments


def detect_speech(
    y: np.ndarray,
    sr: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_percentile: float = 40.0,
    band_ratio_thresh: float = 0.35,
) -> VadResult:
    """Run frame-based VAD over the whole signal.

    energy_percentile sets the adaptive energy threshold: any frame at or
    below this percentile of the signal's own energy distribution is
    treated as background level. Raising it flags more frames as speech
    (more sensitive, more false positives); lowering it is stricter.
    """
    frames, frame_times = frame_signal(y, sr, frame_ms, hop_ms)

    energy = short_time_energy(frames)
    energy_db = 10 * np.log10(energy + 1e-12)

    band_ratio = speech_band_energy_ratio(frames, sr)

    energy_thresh_db = np.percentile(energy_db, energy_percentile)
    is_speech = (energy_db > energy_thresh_db) & (band_ratio > band_ratio_thresh)

    segments = _merge_segments(is_speech, frame_times, frame_ms, hop_ms)

    return VadResult(
        frame_times=frame_times,
        energy_db=energy_db,
        speech_band_ratio=band_ratio,
        is_speech=is_speech,
        speech_segments=segments,
    )
