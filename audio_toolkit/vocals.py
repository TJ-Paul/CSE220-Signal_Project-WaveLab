"""Karaoke and a cappella production — the application layer over separation.

Two separation engines can sit underneath:

    "ml"         a pretrained Demucs model (`ml_separation`), the default
                 when it is installed: it has learned what a voice sounds
                 like, so it separates *voice* from *instruments*.
    "classical"  repetition-based masking (`separation`): no model, but it
                 separates *non-repeating* from *repeating* content, which
                 is only the same thing on loop-based music.

Either engine answers "which part of this mix is the vocal". This module
turns that answer into
the three things anyone actually asks for —

    karaoke     the backing track, vocals suppressed
    a cappella  the isolated lead, backing suppressed
    both        the pair, for A/B listening

— and, just as importantly, measures how well it worked, with the same
metrics for both engines so they can be compared fairly. Everything here
is composition: nothing in this file can make a bad split good. What it can do is present the split
fairly and quantify it honestly.

LEVEL MATCHING IS NOT COSMETIC
-------------------------------
Removing the vocal removes energy, so a raw karaoke stem is several dB
quieter than the mix it came from. In a blind comparison the quieter of
two otherwise identical clips is reliably judged worse — a well-known
confound in audio evaluation, and the reason mastering engineers insist
on level-matched shootouts. Outputs are therefore scaled back to the
source's RMS (peak-limited so nothing clips), which means a listener
switching between mix and karaoke is judging the separation rather than
the gain. Every metric below is computed *before* that rescaling, so the
numbers describe the DSP and not the makeup gain.

THE SPEECH BAND
----------------
Suppression is measured over 300–3400 Hz. That is the classic telephony
band, kept for a century because it carries the intelligibility of a
voice: below it sits mostly the fundamental and chest resonance shared
with bass and low instruments, above it sit sibilance and air shared with
cymbals. Measuring there isolates the part of the spectrum where vocal
removal either worked or did not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import editing, filters, metrics, ml_separation, separation

ENGINES = ("ml", "classical")

#: Mask aggressiveness for the classical engine, as (background margin,
#: foreground margin). The ML engine has no equivalent knob.
#: Higher margins commit each bin more decisively to one stream: cleaner
#: on paper, more artefacts in the ear. "Balanced" matches the defaults in
#: `separation`, so a preset is a considered starting point, not a secret.
PRESETS: dict[str, tuple[float, float]] = {
    "gentle": (1.5, 5.0),
    "balanced": (2.0, 10.0),
    "aggressive": (4.0, 16.0),
}

VOCAL_BAND_HZ = (300.0, 3400.0)


@dataclass
class VocalStems:
    instrumental: np.ndarray
    vocals: np.ndarray
    sr: int
    #: dB of vocal-band energy the instrumental gives up relative to the mix.
    #: Larger means more of the voice is gone — but see `stem_correlation`,
    #: since removing everything would also score well here.
    vocal_suppression_db: float
    #: |corr| between the two stems. Near 0 means they carry genuinely
    #: different content; a high value means one stem is mostly a quieter
    #: copy of the other, i.e. the split did not really happen.
    stem_correlation: float
    #: Share of total energy that ended up in the vocal stem, 0–1.
    vocal_energy_share: float
    preset: str
    engine: str


def _band_rms(y: np.ndarray, sr: int, band: tuple[float, float] = VOCAL_BAND_HZ) -> float:
    """RMS level inside `band`, via a zero-phase Butterworth bandpass."""
    nyquist = sr / 2
    low, high = band[0], min(band[1], nyquist * 0.98)
    if high <= low or len(y) < 32:
        return float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2))) if len(y) else 0.0
    sos = filters.design_filter("bandpass", (low, high), sr, order=4)
    banded = filters.apply_filter(np.asarray(y, dtype=np.float64), sos)
    return float(np.sqrt(np.mean(banded.astype(np.float64) ** 2)))


def make_stems(
    y: np.ndarray,
    sr: int,
    preset: str = "balanced",
    level_match: bool = True,
    engine: str = "ml",
    y_stereo: np.ndarray | None = None,
) -> VocalStems:
    """Split a mix into a karaoke stem and an a cappella stem, with metrics.

    `y` is the mono mix. `y_stereo`, when given, is passed to the ML engine
    instead, because Demucs is a stereo model and uses the channel
    difference as extra evidence. Output stems are always mono.
    """
    if engine == "ml":
        vocals, instrumental = ml_separation.separate_vocals_instrumental(
            y_stereo if y_stereo is not None else y, sr)
        if vocals.ndim == 2:
            vocals, instrumental = vocals.mean(axis=0), instrumental.mean(axis=0)
    else:
        margin_background, margin_foreground = PRESETS.get(preset, PRESETS["balanced"])
        vocals, instrumental = separation.separate_vocals_instrumental(
            y, sr,
            margin_background=margin_background,
            margin_foreground=margin_foreground,
        )

    # Measured on the raw stems: level matching afterwards would inflate the
    # instrumental back toward the mix and erase the very difference being
    # reported.
    mix_band = _band_rms(y, sr)
    inst_band = _band_rms(instrumental, sr)
    suppression = float(20 * np.log10((mix_band + 1e-12) / (inst_band + 1e-12)))

    energy_vocals = float(np.sum(np.asarray(vocals, dtype=np.float64) ** 2))
    energy_inst = float(np.sum(np.asarray(instrumental, dtype=np.float64) ** 2))
    share = energy_vocals / (energy_vocals + energy_inst + 1e-20)

    correlation = abs(metrics.correlation(vocals, instrumental))

    if level_match:
        instrumental = editing.match_loudness(instrumental, y)
        vocals = editing.match_loudness(vocals, y)

    return VocalStems(
        instrumental=np.asarray(instrumental, dtype=np.float32),
        vocals=np.asarray(vocals, dtype=np.float32),
        sr=sr,
        vocal_suppression_db=suppression,
        stem_correlation=correlation,
        vocal_energy_share=share,
        preset=preset if preset in PRESETS else "balanced",
        engine=engine,
    )
