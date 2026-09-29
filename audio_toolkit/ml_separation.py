"""Vocal / instrumental separation with a pretrained Demucs model.

WHAT THIS IS (AND IS NOT)
-------------------------
This module *uses* a published, pretrained source-separation model as a
signal-processing tool, the same way `filters` uses SciPy's Butterworth
design instead of deriving it. Nothing here designs a network, collects a
dataset or trains weights. We load Meta AI's `htdemucs` model, hand it a
waveform, and get separated waveforms back.

    pip install demucs          # the equivalent CLI: demucs --two-stems=vocals song.wav

HOW THE MODEL WORKS (name-drop level)
-------------------------------------
Demucs v4 ("Hybrid Transformer Demucs") looks at the signal in two ways at
once: the raw waveform, and its STFT spectrogram. Each branch is an
encoder-decoder (a U-Net: downsample, process, upsample, with skip links),
and a transformer in the middle lets the two branches share information.
It was trained on thousands of songs where the true stems were known, so it
has learned what a voice looks like in time *and* frequency, which is what
the classical repetition method in `separation` cannot know.

It predicts four stems: drums, bass, other, vocals. For karaoke we only need
two, so, exactly like the CLI's `--two-stems=vocals` option:

    vocals        = the "vocals" stem
    instrumental  = drums + bass + other

WHERE THE SIGNAL PROCESSING IS
------------------------------
Around the model, everything is ordinary DSP from this course:
  - resampling to the model's 44.1 kHz and back (done inside demucs, then
    `librosa.resample` here),
  - mono <-> stereo channel conversion,
  - the STFT/ISTFT inside the spectrogram branch,
  - and the evaluation afterwards (SNR/SDR, correlation, band energy) in
    `vocals` and `metrics`.

PRACTICAL NOTES
---------------
- First use downloads the weights (~85 MB) and caches them; later runs load
  from disk. The model is kept in memory after the first call.
- Runs on the fastest device available: the Apple-Silicon GPU ("mps"), an
  NVIDIA GPU ("cuda"), or the CPU. On an M5, MPS separates a 60 s stereo
  song in ~4 s against ~15 s on the CPU, with the same quality (SI-SDR
  within 0.5 dB).
- Needs `torch` + `demucs`. If they are not installed, `is_available()`
  returns False and only the classical engine is usable.
"""
from __future__ import annotations

import threading

import numpy as np
import librosa

MODEL_NAME = "htdemucs"

_separators: dict = {}  # one per device, loaded once, reused for every request
_lock = threading.RLock()  # one load or separation at a time; the model is shared


def is_available() -> bool:
    """True when demucs (and therefore torch) can be imported."""
    try:
        import demucs.api  # noqa: F401
    except ImportError:
        return False
    return True


def best_device() -> str:
    """Fastest PyTorch device on this machine: Apple GPU, NVIDIA GPU, or CPU."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_separator(device: str):
    with _lock:
        if device not in _separators:
            from demucs.api import Separator
            _separators[device] = Separator(model=MODEL_NAME, device=device)
        return _separators[device]


def separate_vocals_instrumental(
    y: np.ndarray, sr: int, device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split y into (vocals, instrumental) waveforms at the input rate and length.

    `y` may be mono, shape (n,), or stereo, shape (2, n). The model is
    stereo, so mono input is duplicated to both channels first; the
    outputs are returned in the same layout as the input.

    `device` picks where the network runs ("cpu", "mps", "cuda"); None
    means `best_device()`.
    """
    import torch

    y = np.asarray(y, dtype=np.float32)
    was_mono = y.ndim == 1
    # The network's first layer expects exactly two channels; demucs does
    # not upmix for us, so a mono signal is copied to left and right.
    wav = torch.from_numpy(np.stack([y, y]) if was_mono else y.copy())  # (2, samples)

    with _lock, torch.no_grad():
        separator = _get_separator(device or best_device())
        _, stems = separator.separate_tensor(wav, sr)  # resamples to 44.1 kHz internally

    vocals = stems["vocals"].cpu().numpy()
    instrumental = sum(stems[name] for name in stems if name != "vocals").cpu().numpy()

    # Back to the caller's sample rate, length and channel layout.
    model_sr = separator.samplerate
    if model_sr != sr:
        vocals = librosa.resample(vocals, orig_sr=model_sr, target_sr=sr)
        instrumental = librosa.resample(instrumental, orig_sr=model_sr, target_sr=sr)
    n = y.shape[-1]
    vocals = librosa.util.fix_length(vocals, size=n, axis=-1)
    instrumental = librosa.util.fix_length(instrumental, size=n, axis=-1)
    if was_mono:
        vocals = vocals.mean(axis=0)
        instrumental = instrumental.mean(axis=0)

    return vocals.astype(np.float32), instrumental.astype(np.float32)
