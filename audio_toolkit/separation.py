"""Vocal / instrumental separation via nearest-neighbor spectral filtering.

METHOD
------
This implements a classical (non-machine-learning) source-separation
technique related to REPET-SIM (Repeating Pattern Extraction Technique,
Rafii & Pardo 2012), rather than pretending a simple EQ/frequency cut
could isolate a voice — vocals and instruments overlap too heavily in
frequency for that to work.

The idea: instrumental backing tracks (bass, drums, chords/loops) tend
to *repeat* similar spectral patterns over time, while a lead vocal
melody is comparatively non-repetitive. For every time frame of the
STFT magnitude spectrogram, we find its nearest-neighbor frames
elsewhere in the song (by cosine similarity) and take their median.
Because the vocal in any single frame is (mostly) uncorrelated with
the vocal in its similarity-matched neighbors, but the instrumental
pattern *is* shared, the median across neighbors approximates a
"vocal-removed" version of that frame. Comparing this estimated
background against the original spectrum, frequency-and-time bins are
soft-masked into a "background" (repetitive / instrumental-like)
stream and a "foreground" (non-repetitive / vocal-like) stream.

LIMITATIONS (important — read before trusting the output)
-----------------------------------------------------------
- This is signal self-similarity, not semantic understanding of "voice"
  vs "instrument". It will only work well when the backing track is
  genuinely repetitive (loops, riffs, held chords) and the vocal is the
  main non-repeating element — a common case in pop/rock but not
  universal.
- Non-repetitive instrumental parts (rubato piano, live improvisation,
  ambient pads, sparse arrangements) leak into the "foreground"/vocal
  output.
- Backing vocals, doubled/harmonized vocals, or a lead vocal that
  repeats melodically (choruses) can leak into the "background"/
  instrumental output.
- Output quality is far below modern deep-learning separators
  (e.g. Spleeter, Demucs), which are trained on large paired datasets
  to directly predict a vocal/instrumental mask. Those require heavy
  ML dependencies (PyTorch/TensorFlow + multi-hundred-MB pretrained
  weights) and are intentionally out of scope here so the toolkit runs
  anywhere with just NumPy/SciPy/librosa — but that tradeoff is why
  results will sound comparatively artifact-y (phasey, with bleed).
"""
from __future__ import annotations

import numpy as np
import librosa


def separate_vocals_instrumental(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    similarity_window_s: float = 2.0,
    margin_background: float = 2.0,
    margin_foreground: float = 10.0,
    power: float = 2.0,
):
    """Split y into (foreground, background) waveforms of the same length.

    foreground  ~= vocal-like / non-repetitive component
    background  ~= instrumental-like / repetitive component

    similarity_window_s bounds how far in time a frame may search for
    similar neighbors (avoids matching frames that are musically
    unrelated just because they look alike numerically).
    margin_background/foreground control how aggressively each mask
    commits a bin to one stream vs. leaving it blended (higher margin =
    more aggressive, more artifacts; lower = safer but more bleed).
    """
    S_full, phase = librosa.magphase(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))

    n_frames = S_full.shape[1]
    max_width = max(1, (n_frames - 1) // 2 - 1)
    width_frames = int(librosa.time_to_frames(similarity_window_s, sr=sr, hop_length=hop_length))
    width_frames = int(np.clip(width_frames, 1, max_width))

    # Median spectrum of each frame's nearest neighbors == estimated repeating
    # (instrumental-like) content, since the non-repeating vocal averages out.
    S_filter = librosa.decompose.nn_filter(
        S_full, aggregate=np.median, metric="cosine", width=width_frames
    )
    # A neighbor-median can occasionally exceed the true bin's energy
    # (e.g. near a vocal onset); cap it so it never "creates" energy.
    S_filter = np.minimum(S_full, S_filter)

    mask_background = librosa.util.softmask(
        S_filter, margin_background * (S_full - S_filter), power=power
    )
    mask_foreground = librosa.util.softmask(
        S_full - S_filter, margin_foreground * S_filter, power=power
    )

    S_background = mask_background * S_full
    S_foreground = mask_foreground * S_full

    background = librosa.istft(S_background * phase, hop_length=hop_length, length=len(y))
    foreground = librosa.istft(S_foreground * phase, hop_length=hop_length, length=len(y))

    return foreground.astype(np.float32), background.astype(np.float32)
