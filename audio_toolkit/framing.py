"""Short-time framing: split a continuous signal into overlapping frames.

Audio is non-stationary — its statistics change over time — but on a
short enough window (20-30 ms) speech and music can be treated as
quasi-stationary. This is the basis of virtually all frame-based DSP
(VAD, STFT, MFCCs, ...): analyze many short, (mostly) stationary
snippets instead of the whole non-stationary signal at once.
"""
from __future__ import annotations

import numpy as np


def frame_signal(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0):
    """Slice y into overlapping frames.

    frame_len = frame_ms of samples, hop_len = hop_ms of samples
    (hop < frame_len gives overlapping frames, which avoids losing
    information at frame boundaries).

    Returns
    -------
    frames : (n_frames, frame_len) array
    frame_times : (n_frames,) array — the time in seconds of each frame's center
    """
    frame_len = max(1, int(round(sr * frame_ms / 1000)))
    hop_len = max(1, int(round(sr * hop_ms / 1000)))

    if len(y) < frame_len:
        y = np.pad(y, (0, frame_len - len(y)))

    n_frames = 1 + (len(y) - frame_len) // hop_len
    # Build an (n_frames, frame_len) view via stride tricks — equivalent to a
    # sliding window with step hop_len, without copying each frame separately.
    frames = np.stack([y[i * hop_len: i * hop_len + frame_len] for i in range(n_frames)])

    frame_times = (np.arange(n_frames) * hop_len + frame_len / 2) / sr
    return frames, frame_times


def apply_window(frames: np.ndarray, window: str = "hamming") -> np.ndarray:
    """Taper each frame with a window function to reduce spectral leakage.

    A rectangular (unwindowed) frame has abrupt edges, which the DFT
    interprets as high-frequency content, smearing the spectrum. A
    Hamming/Hann window tapers the edges to near zero, producing a
    cleaner frequency-domain estimate.
    """
    frame_len = frames.shape[1]
    if window == "hamming":
        w = np.hamming(frame_len)
    elif window == "hann":
        w = np.hanning(frame_len)
    else:
        w = np.ones(frame_len)
    return frames * w
