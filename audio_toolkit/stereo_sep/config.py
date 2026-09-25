"""Tunable parameters for the classical stereo separation pipeline.

Every number the pipeline uses lives here. Nothing downstream hard-codes a
constant, so an experiment is a config change rather than an edit to the DSP.

The defaults are not arbitrary; each field documents the reasoning and the
audible consequence of moving it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class STFTConfig:
    """Time-frequency resolution.

    The STFT of x[n] is

        X(k,m) = SUM_n x[n + mH] w[n] exp(-j 2 pi k n / N)

    with N = n_fft, H = hop_length, w = window. Resolution obeys the
    Gabor limit: dt * df >= 1/(4 pi). You buy frequency detail with time
    detail and vice versa; there is no setting that gives both.

    n_fft = 4096 at 44.1 kHz gives df = sr/N = 10.8 Hz and a 93 ms window.
    That choice is deliberate: separation decisions depend on resolving
    individual harmonics. A singer at f0 = 110 Hz has partials 110 Hz
    apart, so the 21.5 Hz bins of an n_fft=2048 analysis smear a low male
    voice's first partials together with the bass. 10.8 Hz resolves them.
    The cost is transient smearing over 93 ms, which is why the percussive
    evidence stream (HPSS) is computed at a *separate*, shorter window --
    see hpss_n_fft.

    hop_length = n_fft // 4 gives 75% overlap. With a Hann window that
    satisfies COLA (constant overlap-add), so ISTFT(STFT(x)) == x to
    floating-point precision. Raising hop toward n_fft/2 halves the cost
    but makes time-varying masks audibly steppy.
    """

    n_fft: int = 4096
    hop_length: int = 1024
    window: str = "hann"
    center: bool = True

    #: Percussive/transient evidence is computed at this shorter window so
    #: drum hits stay sharp. 512 samples = 11.6 ms at 44.1 kHz, short enough
    #: to localise a snare, too short to resolve pitch -- which is fine,
    #: because this branch is only asked "is this a transient?".
    hpss_n_fft: int = 1024
    hpss_hop_length: int = 256


@dataclass
class MaskConfig:
    """Soft-mask shaping.

    A Wiener-style mask splits each bin by relative estimated power:

        M_v = P_v^p / (P_v^p + P_i^p + eps)

    p = mask_power controls sharpness. p -> 0 gives M -> 1/2 (no
    separation, both stems are half the mix). p -> inf gives a binary mask
    (maximum separation, maximum musical noise). p = 2 corresponds to the
    classical Wiener filter, which is the MMSE estimator under the
    assumption that V and I are uncorrelated within a bin.
    """

    mask_power: float = 2.0

    #: Smoothing kernel in (frequency bins, time frames). MEASURED COST:
    #: a (3, 5) kernel cost 2.1 dB of instrumental SDR on the ground-truth
    #: benchmark -- the single largest loss in the pipeline. It is kept
    #: anyway, at a reduced (2, 3), because this is a case where the metric
    #: and the ear disagree: SDR penalises the blur, while the flicker it
    #: removes is heard as musical noise. Do not tune this parameter on SDR
    #: alone. Raw masks flicker
    #: bin-to-bin; flicker is heard as musical noise -- isolated spectral
    #: peaks winking in and out, perceived as warbling or metallic ringing.
    #: Smoothing in time enforces the physical prior that sources persist
    #: across a few milliseconds. Too much time smoothing smears consonants
    #: (a "t" or "k" is ~20 ms) and pre-echoes drum hits.
    smooth_freq_bins: int = 2
    smooth_time_frames: int = 3

    #: Mask floor/ceiling. A mask that reaches exactly 0 creates a spectral
    #: hole: total absence of energy in a bin surrounded by energy, which the
    #: ear detects as a hole rather than as silence. Leaving a floor of
    #: -30 dB (0.03 in amplitude) keeps a whisper of the original there and
    #: masks the artefact. This is standard practice in noise reduction.
    mask_floor: float = 0.03
    mask_ceiling: float = 1.0

    #: Bias correction applied to the foreground residual before the Wiener
    #: partition. mag_M - A_i systematically UNDERSTATES vocal magnitude
    #: because magnitudes add vectorially, not scalar-wise. Without this term
    #: the mask degenerates to "keep everything" -- measured, not theorised.
    #: Useful range ~2-6. Higher suppresses more vocal and damages more
    #: accompaniment; lower leaves more vocal bleed in the karaoke stem.
    vocal_margin: float = 4.0

    eps: float = 1e-10


@dataclass
class EvidenceWeights:
    """Relative trust in each independent estimator of "is this vocal?".

    Each feature below returns a score in [0, 1] per time-frequency bin.
    They are combined as a weighted mean, so weights are meaningful only
    in ratio to each other; they are normalised to sum to 1 internally.

    The defaults encode an explicit ranking of reliability:

    spatial (0.35) -- highest, because it rests on the least shaky
        assumption. In the overwhelming majority of commercial mixes the
        lead vocal is panned centre. This is a mixing convention, not a
        law, but it is a strong and testable one.

    repetition (0.25) -- REPET-SIM. Strong when the backing is loop-based,
        weak for through-composed or live material.

    harmonic (0.20) -- vocals are harmonic, but so are guitar, piano and
        strings. This feature discriminates *pitched from unpitched*, which
        is useful but far from "is this a voice".

    percussive (0.15) -- near-certain evidence of NOT-vocal when high.
        Drums are not sung. Used mostly as a veto.

    spectral (0.05) -- weakest. Formant-band shaping is a broad, blunt
        prior that overlaps heavily with guitars and synths, so it only
        breaks ties.
    """

    spatial: float = 0.35
    repetition: float = 0.25
    harmonic: float = 0.20
    percussive: float = 0.15
    spectral: float = 0.05

    def normalised(self) -> dict[str, float]:
        raw = {
            "spatial": self.spatial,
            "repetition": self.repetition,
            "harmonic": self.harmonic,
            "percussive": self.percussive,
            "spectral": self.spectral,
        }
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("evidence weights must sum to a positive value")
        return {k: v / total for k, v in raw.items()}


@dataclass
class SpatialConfig:
    """Stereo-geometry analysis.

    Spatial confidence per bin:

        C(k,m) = |M(k,m)| / (|M(k,m)| + |S(k,m)| + eps)

    C near 1 means the bin is centre-dominant; C near 0 means it lives in
    the sides. C is NOT "vocal probability" -- kick, snare and bass are
    usually centred too. It is one piece of evidence.
    """

    #: Bins below this frequency are exempt from centre-based vocal
    #: attribution. Below ~120 Hz a mix is nearly always mono by
    #: engineering necessity (vinyl cutting, club systems, and because
    #: low-frequency localisation cues are weak), so "centred" carries no
    #: information there -- it just means "bass". Treating centred low
    #: frequencies as vocal is the single most common way a karaoke track
    #: loses its bass line.
    bass_protect_hz: float = 120.0

    #: Above this, centre-panning is likewise less informative: cymbals and
    #: air are broadly spread, and sibilance rides on top. Softer guard.
    air_guard_hz: float = 12000.0

    #: How much Mid-channel energy is attributed to panned instruments per
    #: unit of Side energy. Stereo geometry sets the floor: a source with
    #: channel gains (gL, gR) has M = (gL+gR)/2 s and S = (gL-gR)/2 s, so a
    #: hard-panned source contributes |S| to Mid (ratio 1) while a
    #: half-panned one contributes 3|S|. Values above 1 therefore assume the
    #: average instrument is not hard-panned, which is usually true.
    #: Raising it removes more accompaniment from the vocal stem but starts
    #: eating centred vocal energy; lowering it leaves more bleed.
    side_leak: float = 1.6

    #: Inter-channel coherence is measured over this many frames. Coherence
    #: distinguishes a genuinely centred dry source (stable phase relation)
    #: from stereo reverb (decorrelated). 9 frames at hop 1024 is ~0.2 s --
    #: long enough to average out noise, short enough to track a phrase.
    coherence_frames: int = 9


@dataclass
class HPSSConfig:
    """Harmonic/percussive decomposition by median filtering.

    Harmonic content is horizontally coherent in a spectrogram (a steady
    pitch is a horizontal ridge). Percussive content is vertically coherent
    (a drum hit is a vertical stripe, broadband and brief). Median filtering
    along each axis therefore estimates one while rejecting the other --
    the median is used rather than the mean because it is robust to the
    outliers that the *other* component contributes.
    """

    #: Horizontal (time) kernel: how long a partial must persist to read as
    #: harmonic. 17 frames at hop 256 / 44.1 kHz is ~99 ms.
    kernel_harmonic: int = 17
    #: Vertical (frequency) kernel: how broadband an event must be to read
    #: as percussive. 17 bins at n_fft 1024 is ~730 Hz.
    kernel_percussive: int = 17
    power: float = 2.0


@dataclass
class HarmonicConfig:
    """Harmonic-comb scoring.

    A voiced sound has energy at f0, 2*f0, 3*f0, ... Scoring a bin by how
    well it sits on a detected comb rewards pitched content. Crucially this
    does NOT identify voice -- a guitar has the same comb structure. It is
    used to protect vocal partials from over-suppression, not to find them.
    """

    f0_min_hz: float = 70.0    # below a low male speaking/singing f0
    f0_max_hz: float = 1100.0  # above a soprano's top notes
    n_harmonics: int = 8
    #: Half-width, in bins, of the tolerance band around each predicted
    #: partial. Real singers vibrato and drift, so a rigid comb misses.
    comb_tolerance_bins: int = 2


@dataclass
class RepetitionConfig:
    """REPET-SIM repeating-background estimation.

    For each frame, find spectrally similar frames elsewhere and take their
    median. A repeating accompaniment survives the median; a non-repeating
    vocal does not. The residual is the foreground estimate.
    """

    #: Number of similar frames to aggregate.
    n_neighbours: int = 12
    #: Minimum temporal separation, in seconds, between a frame and a frame
    #: eligible to be its neighbour. NOTE: this is an EXCLUSION radius --
    #: librosa's nn_filter admits neighbours where |i - j| >= width. The
    #: previous implementation documented this backwards. Excluding the
    #: immediate neighbourhood is correct and necessary: adjacent frames
    #: contain the *same* vocal note, so including them would let the vocal
    #: survive the median and end up in the "background" estimate.
    min_separation_s: float = 1.0
    #: Cap on frames analysed for similarity. The similarity matrix is
    #: O(n_frames^2); a 5-minute song at hop 1024 is ~13k frames, so a dense
    #: matrix would be ~1.4 GB in float64. Beyond this cap, the signal is
    #: processed in overlapping blocks.
    max_frames_dense: int = 4000


@dataclass
class OutputConfig:
    """The karaoke/vocal trade-off, and output conditioning."""

    #: How hard to push the lead vocal down in the instrumental stem, 0-1.
    #: Raising this removes more voice AND more of everything that looks
    #: like voice -- centred snare body, pitched lead guitar. A karaoke
    #: track with a hole where the snare was is worse than one with faint
    #: residual voice, which is why the default is moderate rather than max.
    vocal_suppression: float = 0.85

    #: Counterweight: a floor on how much of the accompaniment must survive.
    #: Implemented as a hard lower bound on the instrumental mask, so no
    #: amount of vocal confidence can fully mute a bin. Directly prevents
    #: the "missing bass/snare" failure.
    music_preservation: float = 0.10

    #: Same trade-off, mirrored, for the vocal stem. Lower values keep more
    #: accompaniment bleed but sound more natural; higher values isolate
    #: more but risk telephone-band, phasey artefacts.
    accompaniment_suppression: float = 0.92

    #: Protects vocal partials in the vocal stem from being masked below
    #: this value where harmonic confidence is high -- guards intelligibility.
    vocal_preservation: float = 0.15

    #: True peak ceiling before write. Leaves headroom so inter-sample peaks
    #: do not clip on conversion to PCM.
    peak_ceiling: float = 0.97

    #: Match output RMS to the input. Honest A/B requires it, but note the
    #: gain also amplifies residual bleed -- metrics are reported for BOTH
    #: the raw and the level-matched signal so this is visible.
    level_match: bool = True


@dataclass
class SeparationConfig:
    """Top-level configuration object passed through the whole pipeline."""

    stft: STFTConfig = field(default_factory=STFTConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    weights: EvidenceWeights = field(default_factory=EvidenceWeights)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    hpss: HPSSConfig = field(default_factory=HPSSConfig)
    harmonic: HarmonicConfig = field(default_factory=HarmonicConfig)
    repetition: RepetitionConfig = field(default_factory=RepetitionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    #: Emit intermediate spectrograms, masks and statistics.
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SeparationConfig":
        """Build a config from a (possibly partial) nested dict."""
        sub = {
            "stft": STFTConfig, "mask": MaskConfig, "weights": EvidenceWeights,
            "spatial": SpatialConfig, "hpss": HPSSConfig,
            "harmonic": HarmonicConfig, "repetition": RepetitionConfig,
            "output": OutputConfig,
        }
        kwargs: dict[str, Any] = {}
        for name, klass in sub.items():
            payload = data.get(name) or {}
            valid = {f for f in klass.__dataclass_fields__}
            kwargs[name] = klass(**{k: v for k, v in payload.items() if k in valid})
        kwargs["debug"] = bool(data.get("debug", False))
        return cls(**kwargs)
