"""Evidence fusion, soft-mask construction, smoothing and application.

WHY SOFT MASKS
--------------
A binary mask assigns each bin wholly to one source. But a bin is a ~11 Hz by
~93 ms cell that genuinely contains BOTH sources; forcing it to one is a lie
the ear detects. The audible result is "musical noise": isolated surviving
bins appear and vanish frame to frame, heard as warbling, chirping or
metallic ringing, because a lone short sinusoid at a random frequency is
exactly what a binary mask leaves behind.

A continuous mask 0 <= M <= 1 instead splits each bin in proportion to
estimated energy share, so a bin containing 70% accompaniment contributes 70%
of its energy to the instrumental and 30% to the vocal. Nothing switches, so
nothing chirps.

THE WIENER FORM
---------------
    M_v = P_v^p / (P_v^p + P_i^p + eps)
    M_i = P_i^p / (P_v^p + P_i^p + eps)

With p = 2 and P as power spectra this is the classical Wiener filter, the
minimum mean-squared-error estimator of one component given the mixture,
under the assumption that V and I are uncorrelated within a bin. That
assumption is imperfect (they are correlated at shared harmonics) but it is
explicit, and it makes p = 2 a principled default rather than a taste.

Note M_v + M_i = 1 exactly: the masks PARTITION the mixture, so no energy is
lost or duplicated. The previous implementation used two independently
computed masks with different margins, so their sum was not 1 and the stems
were not a decomposition of the input.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import MaskConfig, EvidenceWeights, OutputConfig


def combine_evidence(
    scores: dict[str, np.ndarray],
    weights: EvidenceWeights,
) -> np.ndarray:
    """Fuse independent evidence streams into one vocal confidence C_v.

        C_v(k,m) = SUM_i w_i * s_i(k,m) / SUM_i w_i,  over PRESENT features i

    A weighted arithmetic mean is used rather than a product. A product acts
    as a logical AND -- any single feature near 0 vetoes the bin -- which is
    too brittle when each feature is individually unreliable: an unvoiced
    consonant scores ~0 on harmonicity yet is unmistakably vocal, and a
    product would delete it. The mean lets strong evidence outvote one
    dissenting feature, which is the intended behaviour when no single
    feature is trusted.

    The percussive stream is inverted before fusion: high percussive energy
    is evidence AGAINST voice (drums are not sung), so it enters as
    (1 - percussive).

    RENORMALISATION OVER PRESENT FEATURES ONLY. A method that does not
    compute a feature must omit it, not pass a neutral 0.5 placeholder. This
    is not a detail -- it was a measured defect. Feeding 0.5 for the absent
    repetition and spectral streams while still spending their 0.30 of the
    weight budget dragged the fused confidence toward 0.5 everywhere,
    collapsing its spread from sd 0.23 to sd 0.11 and making the
    HPSS-enabled method score WORSE than the one without it. Omitting an
    absent feature and renormalising the remaining weights keeps the
    confidence's dynamic range intact, so adding a feature can never dilute
    the features already present.
    """
    contributions = {
        "spatial": lambda: scores["spatial"],
        "repetition": lambda: scores["repetition"],
        "harmonic": lambda: scores["harmonic"],
        "percussive": lambda: 1.0 - scores["percussive"],
        "spectral": lambda: scores["spectral"],
    }

    w = weights.normalised()
    present = {k: v for k, v in w.items() if k in scores}
    if not present:
        raise ValueError("no evidence features supplied")
    total_w = sum(present.values())

    shape = next(iter(scores.values())).shape
    out = np.zeros(shape, dtype=np.float32)
    for name, weight in present.items():
        out += np.float32(weight / total_w) * contributions[name]().astype(np.float32)

    return np.clip(out, 0.0, 1.0)


def confidence_to_mask(
    C: np.ndarray,
    sharpness: float,
    midpoint: float = 0.5,
) -> np.ndarray:
    """Map confidence to a mask through a smooth bounded function.

        M = 1 / (1 + exp(-sharpness * (C - midpoint)))

    A logistic curve is used rather than the identity because confidence and
    gain are not the same quantity. Passing C straight through would leave
    every bin at partial gain, so nothing is ever fully removed or fully
    kept, and the separation sounds washed out. The logistic pushes confident
    bins toward the extremes while keeping the transition differentiable --
    no discontinuity, so no spectral edge to hear.

    sharpness is the trade-off dial:
      low  (~4): gentle, lots of bleed, very few artefacts
      high (~16): decisive, cleaner separation, more musical noise
    midpoint sets where the curve crosses 0.5, i.e. how much confidence is
    required before a bin is treated as more vocal than not.
    """
    return 1.0 / (1.0 + np.exp(-sharpness * (C - midpoint)))


def wiener_masks(
    P_v: np.ndarray,
    P_i: np.ndarray,
    cfg: MaskConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Complementary Wiener masks from estimated vocal/instrumental power.

    Returns (M_v, M_i) with M_v + M_i == 1 exactly.
    """
    p = cfg.mask_power
    pv = np.power(np.maximum(P_v, 0.0), p)
    pi = np.power(np.maximum(P_i, 0.0), p)
    denom = pv + pi + cfg.eps
    M_v = pv / denom
    return M_v.astype(np.float32), (1.0 - M_v).astype(np.float32)


def smooth_mask(M: np.ndarray, cfg: MaskConfig) -> np.ndarray:
    """Smooth a mask across frequency and time.

    Raw masks vary abruptly between neighbouring bins because the evidence
    does. Those discontinuities are the direct cause of musical noise,
    buzzing, warbling and spectral holes.

    Smoothing encodes two physical priors:
      - across TIME: a source does not appear and vanish within 20 ms, so a
        bin's classification should persist across a few frames.
      - across FREQUENCY: a harmonic partial occupies several adjacent bins
        (it is convolved with the window's main lobe), so its classification
        should not flip between them.

    A separable uniform filter is used, applied in frequency then time.
    Over-smoothing is a real cost, not a hypothetical one: too wide a TIME
    kernel smears consonants (a plosive lasts ~20 ms) and pre-echoes drum
    hits, because the mask starts opening before the transient arrives. That
    is why the time kernel defaults to 5 frames and is exposed as a parameter.
    """
    kf = max(1, int(cfg.smooth_freq_bins))
    kt = max(1, int(cfg.smooth_time_frames))
    out = M.astype(np.float32)
    if kf > 1:
        out = ndimage.uniform_filter1d(out, size=kf, axis=0, mode="nearest")
    if kt > 1:
        out = ndimage.uniform_filter1d(out, size=kt, axis=1, mode="nearest")
    return out


def clamp_mask(M: np.ndarray, cfg: MaskConfig) -> np.ndarray:
    """Apply the mask floor and ceiling.

    The floor matters more than it looks. A mask reaching exactly 0 creates a
    spectral hole: a bin of total silence surrounded by energy. The ear is
    very good at detecting holes -- they sound like a notch or a "hollow"
    quality -- whereas heavily attenuated content just sounds quiet. Leaving
    a floor around -30 dB keeps a trace of the original signal in the bin,
    which perceptually fills the hole at a level too low to compromise the
    separation.
    """
    return np.clip(M, cfg.mask_floor, cfg.mask_ceiling).astype(np.float32)


def estimate_source_powers(
    mag_M: np.ndarray,
    mag_S: np.ndarray,
    scores: dict[str, np.ndarray],
    side_leak: float,
    vocal_margin: float,
    mask_cfg: MaskConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate accompaniment and vocal POWER per bin, for a Wiener partition.

    WHY THIS SHAPE, AND WHAT WAS MEASURED
    -------------------------------------
    Two earlier designs were built and rejected on measurement, and the
    reasons are worth keeping because both are natural mistakes.

    (1) A fused confidence C_v in [0,1] mapped through a logistic. This fails
        because in a real mix MOST energy is centre-panned -- bass, kick,
        snare and voice all sit in the middle -- so centre-dominance averaged
        0.74, the logistic saturated, and the mask degenerated to "keep
        everything". It scored BELOW plain Mid/Side. A mask is a ratio of
        energies and must be built from estimated energies, not from a score
        with arbitrary units.

    (2) A_i = max over several lower bounds. Each estimator IS a valid lower
        bound on accompaniment magnitude, so their maximum is the tightest --
        in exact arithmetic. With noisy estimates it is badly biased upward:
        the max of several noisy variables exceeds each of their means. The
        measured A_i came out at 1.085 against a true 0.729, which drove the
        residual mag_M - A_i toward zero and again produced "keep everything".

    THE FORMULATION THAT MEASURED WELL
    ----------------------------------
        A_i = accompaniment magnitude estimate    (REPET background, primary)
        A_v = vocal_margin * max(mag_M - A_i, 0)  (foreground residual)

        P_i = A_i^p,  P_v = A_v^p   ->  Wiener partition

    `vocal_margin` is the critical term and its omission was the defect in
    the first two designs. The residual mag_M - A_i UNDERSTATES vocal energy
    systematically, for a specific reason: magnitudes add vectorially, so for
    components with phase difference theta,

        |X| = sqrt(|I|^2 + |V|^2 + 2|I||V| cos theta)  <=  |I| + |V|

    with equality only when they are exactly in phase. Subtracting an
    accompaniment magnitude therefore leaves less than the true vocal
    magnitude almost everywhere. The margin compensates for that bias.

    It is a bias correction, not a free gain: it trades vocal suppression
    against accompaniment damage, and a sweep on ground truth put the useful
    range around 2-6, with 4 the default.

    A_i is built primarily from the REPET background, which measured a 0.979
    correlation with true accompaniment magnitude -- by far the strongest
    estimator available. Where it is absent (methods that do not compute
    repetition), stereo geometry stands in: a source contributing |S| to Side
    must contribute at least |S| to Mid, since M = (gL+gR)/2 s and
    S = (gL-gR)/2 s, so side_leak * |S| is a geometric lower bound on
    accompaniment in Mid, assuming nothing about what the source is.
    """
    eps = mask_cfg.eps

    have_repet = "repetition_background" in scores
    if have_repet:
        # Primary estimator. Blended, not maxed, with the geometric bound so
        # hard-panned content that REPET may treat as non-repeating is still
        # credited as accompaniment -- while avoiding the upward bias of max.
        A_i = scores["repetition_background"]
        geo = np.float32(side_leak) * mag_S
        A_i = np.maximum(A_i, 0.5 * (A_i + geo))
    else:
        A_i = np.float32(side_leak) * mag_S
        if "percussive" in scores:
            A_i = np.maximum(A_i, scores["percussive"] * mag_M)

    A_i = np.minimum(A_i, mag_M)

    # The margin corrects the bias of a MAGNITUDE SUBTRACTION, so it applies
    # only where a real subtraction happened. Without a repetition estimate
    # the accompaniment model is just stereo geometry, and for centre-panned
    # bass and drums |S| ~ 0, so A_i ~ 0 and mag_M - A_i is essentially all
    # of Mid -- nothing has been subtracted and there is no bias to correct.
    # Applying the full margin there amplifies an unsubtracted residual and
    # declares the entire centre vocal, which measured WORSE than doing
    # nothing at all. Scaling the margin by how much was actually explained
    # keeps every method at or above the do-nothing baseline.
    if have_repet:
        margin = np.float32(vocal_margin)
    else:
        explained = float(np.mean(A_i) / (np.mean(mag_M) + eps))
        margin = np.float32(1.0 + (vocal_margin - 1.0) * min(1.0, 2.0 * explained))

    A_v = margin * np.maximum(mag_M - A_i, 0.0)

    # Plausibility modulates the residual multiplicatively and gently. It can
    # reject a candidate but never manufacture vocal energy where the
    # residual is zero -- the correct asymmetry, since evidence should be
    # able to veto an estimate, not invent one.
    #
    # The coefficients are deliberately mild. An earlier, stronger version
    # (0.5-0.7 floors) drove the combined factor to a mean of 0.36, damping
    # the vocal estimate roughly threefold and costing measured SDR: the
    # features are individually too unreliable to justify that much authority
    # over an estimate the residual already supports.
    plaus = np.ones_like(A_v, dtype=np.float32)
    if "harmonic" in scores:
        plaus *= (0.85 + 0.15 * scores["harmonic"])
    if "spectral" in scores:
        plaus *= (0.85 + 0.15 * scores["spectral"])
    if "spatial" in scores:
        plaus *= (0.7 + 0.3 * scores["spatial"])
    if "percussive" in scores:
        # Drums are never sung: percussive energy argues against vocal.
        plaus *= (1.0 - 0.3 * scores["percussive"])
    A_v = A_v * plaus

    p = mask_cfg.mask_power
    return (np.power(A_v, p) + eps).astype(np.float32), (np.power(A_i, p) + eps).astype(np.float32)


def spatial_gate_masks(
    C_v: np.ndarray,
    scores: dict[str, np.ndarray],
    mask_cfg: MaskConfig,
    out_cfg: OutputConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Confidence-driven masks for methods with NO accompaniment magnitude model.

    WHY A SEPARATE PATH EXISTS
    --------------------------
    The Wiener path in `estimate_source_powers` needs an estimate of
    accompaniment MAGNITUDE. Methods that compute no repetition analysis do
    not have one: their only accompaniment evidence is stereo geometry, and
    for centre-panned bass and drums |S| ~ 0, so the estimate is silent
    exactly where it is needed. Measured against ground truth, that estimate
    correlated 0.165 with true accompaniment magnitude -- near-random. Driving
    a Wiener partition from it produced masks WORSE than doing nothing.

    So these methods take a different and more modest route: a per-bin
    generalisation of the Mid/Side baseline. Method 2 attenuates Mid by a
    single constant above the bass guard; this attenuates Mid by an amount
    that VARIES with per-bin spatial confidence. That needs only a relative
    ordering of bins, not a calibrated magnitude, which is all the spatial
    feature can honestly supply.

        M_i = 1 - vocal_suppression * g(C_v)
        M_v = accompaniment_suppression * g(C_v) + (1 - accompaniment_suppression)

    g is a logistic centred on the MEDIAN of C_v rather than on a fixed 0.5.
    That detail matters: absolute centre-dominance is not comparable across
    tracks, because a narrow mix has high C everywhere and a wide mix low C
    everywhere. Centring on the median makes the gate adaptive -- it asks
    "is this bin more centred than the rest of THIS track?", which is the
    question the evidence can actually answer, and it keeps the gate from
    saturating at one extreme.
    """
    mid = float(np.median(C_v))
    spread = float(np.std(C_v)) + 1e-6
    # Normalise to the track's own confidence distribution, then sharpen.
    z = (C_v - mid) / spread
    gate = 1.0 / (1.0 + np.exp(-1.5 * z))

    M_i = 1.0 - out_cfg.vocal_suppression * gate
    M_i = np.maximum(M_i, out_cfg.music_preservation)

    perc = scores.get("percussive")
    if perc is not None:
        M_i = np.maximum(M_i, out_cfg.music_preservation
                         + (1.0 - out_cfg.music_preservation) * perc * (1.0 - gate))

    a = out_cfg.accompaniment_suppression
    M_v = a * gate + (1.0 - a)

    return (clamp_mask(smooth_mask(M_v, mask_cfg), mask_cfg),
            clamp_mask(smooth_mask(M_i, mask_cfg), mask_cfg))


def karaoke_mask(
    M_i_base: np.ndarray,
    scores: dict[str, np.ndarray],
    C_v: np.ndarray,
    mask_cfg: MaskConfig,
    out_cfg: OutputConfig,
) -> np.ndarray:
    """Instrumental mask. Priority: PRESERVE THE MUSIC.

    Starts from the Wiener instrumental mask and applies the karaoke-specific
    safety bias. The asymmetry against `vocal_mask` is the core design
    decision: for karaoke the failure modes are not equally bad.

      - residual lead vocal: mildly annoying, the live singer masks it
      - missing bass or snare: ruins the track, there is nothing to sing to

    So this mask fails safe toward keeping music.

        M_i = 1 - vocal_suppression * (1 - M_i_base)
        M_i = max(M_i, music_preservation)                 (hard floor)
        M_i = max(M_i, drum guard)

    vocal_suppression < 1 means even a maximally confident vocal bin is
    attenuated rather than deleted, so the residual sounds like quiet voice
    rather than like a hole.

    music_preservation is a hard lower bound no evidence can override. It
    directly answers the "weak bass / missing snare" failure: a centred kick
    looks vocal to the spatial feature, and without this floor it would go.
    """
    M_i = 1.0 - out_cfg.vocal_suppression * (1.0 - M_i_base)
    M_i = np.maximum(M_i, out_cfg.music_preservation)

    # Drum guard: percussive AND not-vocal. The (1 - C_v) gate is essential
    # and was missing in the first version. Percussive energy is not only
    # drums -- vocal consonants are broadband transients that score just as
    # highly -- so an ungated guard protected the singer's consonants inside
    # the karaoke stem, measurably degrading it.
    perc = scores.get("percussive")
    if perc is not None:
        drum_evidence = perc * (1.0 - C_v)
        M_i = np.maximum(
            M_i,
            out_cfg.music_preservation
            + (1.0 - out_cfg.music_preservation) * drum_evidence,
        )

    return clamp_mask(smooth_mask(M_i, mask_cfg), mask_cfg)


def vocal_mask(
    M_v_base: np.ndarray,
    scores: dict[str, np.ndarray],
    C_v: np.ndarray,
    mask_cfg: MaskConfig,
    out_cfg: OutputConfig,
) -> np.ndarray:
    """Vocal mask. Priority: PRESERVE VOICE QUALITY AND INTELLIGIBILITY.

    A different objective, so a different mask -- deliberately NOT the
    complement of the karaoke mask. Here the bad outcomes are:

      - accompaniment bleed: tolerable, the voice still reads clearly
      - a thin, telephone-like, phasey or consonant-less voice: useless

    so this mask fails safe toward keeping voice.

        M_v = accompaniment_suppression * M_v_base + (1 - accompaniment_suppression)

    then lifted by two guards, both GATED BY VOCAL CONFIDENCE:

      - harmonic partial guard, which prevents the hollow "robotic" quality
        that appears when a voice loses its upper harmonics
      - consonant guard, because consonants are broadband and noise-like,
        score ~0 on both harmonicity and repetition, and are therefore
        exactly what a confidence-driven mask deletes -- which is why
        over-processed isolated vocals sound mumbled

    The C_v gate on both is not optional. Harmonicity alone does not indicate
    voice (guitar and piano are harmonic), so an ungated harmonic guard pins
    a floor under every pitched instrument and the "vocal" stem becomes the
    whole song at reduced level. Likewise an ungated transient guard protects
    every drum hit.
    """
    a = out_cfg.accompaniment_suppression
    M_v = a * M_v_base + (1.0 - a)

    if "harmonic" in scores:
        M_v = np.maximum(M_v, out_cfg.vocal_preservation * scores["harmonic"] * C_v)
    if "percussive" in scores:
        M_v = np.maximum(
            M_v, out_cfg.vocal_preservation * _transient_emphasis(scores) * C_v)

    return clamp_mask(smooth_mask(M_v, mask_cfg), mask_cfg)


def _transient_emphasis(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Detect broadband onsets, to protect consonants in the vocal stem.

    Consonants are short, broadband and noise-like. They carry most of the
    intelligibility of sung text and are invisible to harmonic and repetition
    features alike, so they need an explicit detector.

    A positive half-wave rectified time difference of the percussive-stream
    energy (spectral flux) marks bins where energy rises sharply. Only RISES
    count -- a decay is not an onset -- which is why the difference is
    rectified rather than taken as an absolute value.

    This also emphasises drum onsets, which is an accepted cost: a little
    extra drum transient in the vocal stem is far less damaging than a voice
    with no consonants.
    """
    perc = scores["percussive"]
    flux = np.diff(perc, axis=1, prepend=perc[:, :1])
    flux = np.maximum(flux, 0.0)
    m = float(np.max(flux))
    if m < 1e-9:
        return np.zeros_like(perc)
    return (flux / m).astype(np.float32)


def apply_mask_complex(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a real mask to a complex spectrum, preserving mixture phase.

        Y(k,m) = M(k,m) * X(k,m) = M * |X| * exp(j phi)

    Because M is real and non-negative it scales magnitude only; the phase
    phi(k,m) is carried through untouched. This is the right default. The
    true phase of the isolated source is unknown and unknowable from the
    mixture, and the mixture phase is its minimum-mean-squared-error estimate
    whenever one source dominates the bin -- which is exactly where the mask
    is near 1 and phase matters most.

    Inventing or randomising phase instead would destroy the waveform
    coherence that overlap-add depends on, producing the smeared, reverberant,
    "phasey" sound characteristic of magnitude-only reconstruction.
    """
    return (X * M.astype(X.real.dtype)).astype(X.dtype)


def complex_residual(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Exact complementary component: X - Y, in the COMPLEX domain.

    If Y = M * X then X - Y = (1 - M) * X, so the two sum back to X exactly
    and the decomposition is lossless.

    This must never be done on magnitudes. Magnitudes add vectorially, not
    scalar-wise: for components with phase difference theta,

        |X| = sqrt(|I|^2 + |V|^2 + 2|I||V| cos theta)

    so |X| - |I| is a biased estimate of |V| whose error swings with
    cos theta. Subtracting magnitude spectrograms -- as the earlier
    implementation did -- introduces a phase-dependent error that is heard as
    a restless, fluttering residual.
    """
    return X - Y
