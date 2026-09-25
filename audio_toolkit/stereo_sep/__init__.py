"""Classical (non-ML) stereo source separation.

Splits a stereo mixture into a karaoke/instrumental estimate and a
vocal-focused estimate using only mathematics and signal processing:
Fourier analysis, stereo geometry, phase relationships, spectral analysis,
soft masking and consistent complex-domain reconstruction. No neural
networks, no learned models, no training data.

    config    -> every tunable parameter, with its justification
    stft      -> STFT/ISTFT plus the REQUIRED round-trip verification
    midside   -> stereo loading, validation, Mid/Side transform
    features  -> independent evidence: spatial, HPSS, harmonic, repetition
    masks     -> power estimation, Wiener masks, smoothing, application
    pipeline  -> the five methods, from naive baseline to full pipeline
    analysis  -> debug mode: spectrograms, masks and measurements
    evaluate  -> SDR/SIR/SAR against ground truth, proxies without it
    compare   -> A/B harness across all five methods

THE CENTRAL CAVEAT, STATED ONCE
-------------------------------
Per time-frequency bin the mixture gives one complex equation,
X = V + I, in two complex unknowns. The problem is underdetermined: V and I
are NOT uniquely determined by X, and no method recovers them exactly from
an already-mixed recording. Every result here is an ESTIMATE conditioned on
stated assumptions (centre-panned lead, repeating accompaniment, harmonic
voice, percussive drums, time-frequency sparsity), and it degrades as those
assumptions fail. Nothing in this package identifies a voice; it identifies
energy that behaves the way the assumptions say a voice behaves.
"""
from .config import (
    SeparationConfig, STFTConfig, MaskConfig, EvidenceWeights,
    SpatialConfig, HPSSConfig, HarmonicConfig, RepetitionConfig, OutputConfig,
)
from .pipeline import separate, SeparationResult, METHODS, METHOD_NAMES
from .midside import load_stereo, analyse_stereo, to_mid_side, to_left_right
from .stft import verify_reconstruction, check_cola
from .analysis import debug_report, signal_stats
from .evaluate import bss_eval, evaluate_result, proxy_metrics, LISTENING_GUIDE

__all__ = [
    "SeparationConfig", "STFTConfig", "MaskConfig", "EvidenceWeights",
    "SpatialConfig", "HPSSConfig", "HarmonicConfig", "RepetitionConfig",
    "OutputConfig", "separate", "SeparationResult", "METHODS", "METHOD_NAMES",
    "load_stereo", "analyse_stereo", "to_mid_side", "to_left_right",
    "verify_reconstruction", "check_cola", "debug_report", "signal_stats",
    "bss_eval", "evaluate_result", "proxy_metrics", "LISTENING_GUIDE",
]
