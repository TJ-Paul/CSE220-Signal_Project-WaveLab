"""Independent per-bin evidence streams for "is this energy vocal-like?".

Each function returns a score array in [0, 1] with the same shape as the
spectrogram. None of them is a vocal detector. Each tests one narrow,
falsifiable hypothesis, and each is wrong in a *different* way -- which is the
point. Combining estimators whose errors are uncorrelated is what lets a
deterministic rule outperform any single rule, with no learning involved.

Every score is documented with: the equation, what it assumes, and how it
fails. Read the failure modes before trusting any output.
"""
from __future__ import annotations

import numpy as np
import librosa
from scipy import ndimage

from .config import (
    SpatialConfig, HPSSConfig, HarmonicConfig, RepetitionConfig, STFTConfig,
)


# ---------------------------------------------------------------------------
# 1. SPATIAL -- stereo geometry
# ---------------------------------------------------------------------------

def spatial_score(
    X_M: np.ndarray,
    X_S: np.ndarray,
    X_L: np.ndarray,
    X_R: np.ndarray,
    freqs: np.ndarray,
    cfg: SpatialConfig,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Centre-dominance and inter-channel coherence.

    EQUATION
        C(k,m) = |M(k,m)| / (|M(k,m)| + |S(k,m)| + eps)

    C -> 1: energy is common to both channels (centre-panned).
    C -> 0: energy differs between channels (panned wide, or decorrelated).

    ASSUMES lead vocals are mixed centre. True for the large majority of
    commercial popular music -- a production convention, not physics.

    FAILS when: the vocal is deliberately widened or doubled (C drops, the
    voice looks like accompaniment); bass/kick/snare are centred (C is high,
    they look like voice -- mitigated by bass_protect_hz); the recording is
    mono (C == 1 everywhere, zero information).

    COHERENCE
        gamma(k,m) = |<L conj(R)>| / sqrt(<|L|^2> <|R|^2>)

    averaged over a short window of frames. This is the magnitude-squared
    coherence's square root, i.e. normalised cross-spectrum. It separates two
    cases that C alone confuses:

      - a dry centred source: L and R are near-identical, so gamma -> 1
      - stereo reverb / ambience: L and R are decorrelated, so gamma -> 0

    A bin can be centre-heavy (high C) yet incoherent (low gamma) when it
    holds diffuse reverb that happens to balance across channels. Treating
    such bins as dry vocal is what leaves a "singing reverb ghost" in a
    karaoke track, so coherence is carried as a separate signal.

    PHASE DIFFERENCE
        dphi(k,m) = phi_L(k,m) - phi_R(k,m)

    Near 0 means in-phase (centred). Near +-pi means out-of-phase, which in
    real music usually indicates a wide stereo effect or a phase problem
    rather than a source. Reported as cos(dphi) so it is bounded and has no
    wrap-around discontinuity at +-pi.
    """
    mag_M = np.abs(X_M)
    mag_S = np.abs(X_S)

    # --- Centre dominance -------------------------------------------------
    C = mag_M / (mag_M + mag_S + eps)

    # --- Inter-channel coherence -----------------------------------------
    # Smoothed cross- and auto-spectra. Smoothing is essential: an
    # instantaneous single-frame coherence is identically 1 by construction
    # (any two complex numbers are "perfectly coherent" on their own), so
    # coherence only becomes meaningful once averaged over several frames.
    w = max(1, int(cfg.coherence_frames))
    kernel = np.ones(w) / w

    def smooth_t(A):
        return ndimage.convolve1d(A, kernel, axis=1, mode="nearest")

    Sxy = smooth_t(X_L * np.conj(X_R))
    Sxx = smooth_t(np.abs(X_L) ** 2)
    Syy = smooth_t(np.abs(X_R) ** 2)
    coherence = np.abs(Sxy) / (np.sqrt(Sxx * Syy) + eps)
    coherence = np.clip(coherence, 0.0, 1.0)

    # --- Inter-channel phase difference ----------------------------------
    dphi = np.angle(X_L) - np.angle(X_R)
    phase_alignment = 0.5 * (np.cos(dphi) + 1.0)  # map [-1,1] -> [0,1]

    # --- Frequency-dependent trust ---------------------------------------
    # Below bass_protect_hz, "centred" means "bass", not "voice": mixes are
    # mono in the low end by engineering convention. Setting the spatial
    # score to a neutral 0.5 there removes the evidence rather than letting
    # it argue confidently for the wrong answer -- this is the single guard
    # that keeps a karaoke stem's bass line intact.
    trust = np.ones_like(C)
    low = freqs < cfg.bass_protect_hz
    trust[low, :] = 0.0
    # Smooth taper above the guard so there is no audible step at the
    # boundary; an abrupt change in mask behaviour across one bin is heard
    # as a resonance at that frequency.
    taper = (freqs >= cfg.bass_protect_hz) & (freqs < cfg.bass_protect_hz * 2)
    if np.any(taper):
        ramp = (freqs[taper] - cfg.bass_protect_hz) / cfg.bass_protect_hz
        trust[taper, :] = ramp[:, None]
    high = freqs > cfg.air_guard_hz
    trust[high, :] = 0.5

    # Combine: a bin argues for "vocal" only when it is both centre-dominant
    # AND coherent. The product is deliberate -- either condition failing
    # should veto, which an average would not do.
    raw = C * (0.5 + 0.5 * coherence)
    score = 0.5 + trust * (raw - 0.5)

    return {
        "score": np.clip(score, 0.0, 1.0).astype(np.float32),
        "centre_dominance": C.astype(np.float32),
        "coherence": coherence.astype(np.float32),
        "phase_alignment": phase_alignment.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# 2. HPSS -- harmonic / percussive structure
# ---------------------------------------------------------------------------

def hpss_scores(
    y_mono: np.ndarray,
    cfg: HPSSConfig,
    stft_cfg: STFTConfig,
    target_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Harmonic and percussive energy, by median filtering the spectrogram.

    EQUATION
        H(k,m) = median over time   of |X| in a 1 x kernel_harmonic window
        P(k,m) = median over freq   of |X| in a kernel_percussive x 1 window

        M_H = H^p / (H^p + P^p),   M_P = P^p / (H^p + P^p)

    A steady pitch draws a horizontal ridge in the spectrogram; a drum hit
    draws a vertical stripe. Median-filtering along time therefore preserves
    sustained tones and removes transients, and vice versa. The median is
    chosen over the mean because it is robust: a percussive spike inside a
    time window is an outlier, and the median ignores outliers while the mean
    would be dragged by them.

    ASSUMES vocals are predominantly harmonic and drums predominantly
    percussive. Both are broadly true.

    FAILS as a vocal detector, and must not be used as one: guitar, piano,
    strings, bass and synth pads are all harmonic too. Consonants ("t", "k",
    "s") are percussive and broadband, so a naive "harmonic = vocal" rule
    actively deletes the consonants that carry intelligibility. Used here
    only as (a) evidence AGAINST vocal when percussive energy dominates, and
    (b) structural context.

    Computed at a SHORTER window than the main analysis (hpss_n_fft) because
    the percussive judgement needs time resolution, then resampled onto the
    main grid. Using the main 93 ms window would smear a snare across the
    whole window and make it look harmonic.
    """
    D = librosa.stft(
        np.ascontiguousarray(y_mono, dtype=np.float32),
        n_fft=stft_cfg.hpss_n_fft,
        hop_length=stft_cfg.hpss_hop_length,
        window=stft_cfg.window,
        center=stft_cfg.center,
    )
    mag = np.abs(D)

    H = ndimage.median_filter(mag, size=(1, cfg.kernel_harmonic), mode="nearest")
    P = ndimage.median_filter(mag, size=(cfg.kernel_percussive, 1), mode="nearest")

    p = cfg.power
    denom = H**p + P**p + 1e-12
    mask_h = (H**p) / denom
    mask_p = (P**p) / denom

    return {
        "harmonic": _resample_grid(mask_h, target_shape).astype(np.float32),
        "percussive": _resample_grid(mask_p, target_shape).astype(np.float32),
    }


def _resample_grid(A: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resample a (bins, frames) array onto a different grid.

    Needed because the percussive analysis runs at its own time-frequency
    resolution. Bilinear (order=1) rather than nearest-neighbour so the
    resampled mask has no staircase edges, which would reintroduce exactly
    the bin-to-bin discontinuities that mask smoothing exists to remove.
    """
    if A.shape == shape:
        return A
    zoom = (shape[0] / A.shape[0], shape[1] / A.shape[1])
    out = ndimage.zoom(A, zoom, order=1, mode="nearest")
    # zoom can be off by a row/column from rounding; pad or crop to be exact.
    out = out[: shape[0], : shape[1]]
    if out.shape != shape:
        pad = ((0, shape[0] - out.shape[0]), (0, shape[1] - out.shape[1]))
        out = np.pad(out, pad, mode="edge")
    return out


# ---------------------------------------------------------------------------
# 3. HARMONIC COMB -- pitched structure at a detected f0
# ---------------------------------------------------------------------------

def harmonic_score(
    mag: np.ndarray,
    freqs: np.ndarray,
    sr: int,
    cfg: HarmonicConfig,
    stft_cfg: STFTConfig,
) -> dict[str, np.ndarray]:
    """Score bins by whether they sit on a harmonic series f0, 2f0, 3f0, ...

    METHOD
    Per frame, estimate a dominant f0 by harmonic sum spectrum:

        HSS(f0) = SUM_{h=1..H} |X(h * f0)|

    The f0 maximising HSS is the pitch best explaining the frame's partials.
    Bins within comb_tolerance_bins of any h*f0 then score high.

    WHY THIS HELPS
    It does not find voice. It finds *pitched* content, and its real job here
    is protective: once a bin is known to lie on a coherent harmonic comb,
    suppressing it will audibly damage a musical tone. In the vocal stem this
    guards the partials that carry timbre and intelligibility; in the
    instrumental stem it flags tonal content worth preserving.

    ASSUMES voiced singing is harmonic with a resolvable f0.

    FAILS on: unvoiced consonants (no f0 at all -- noise-like, so they score
    0 and must be protected by other means); polyphonic frames where the
    strongest comb belongs to a guitar chord, not the voice; octave errors,
    since HSS(f0/2) also collects all of f0's partials. The octave ambiguity
    is why the score is deliberately soft evidence, never a gate.
    """
    n_bins, n_frames = mag.shape
    n_fft = stft_cfg.n_fft

    bin_min = max(1, int(np.floor(cfg.f0_min_hz * n_fft / sr)))
    bin_max = min(n_bins - 1, int(np.ceil(cfg.f0_max_hz * n_fft / sr)))
    if bin_max <= bin_min:
        z = np.zeros_like(mag, dtype=np.float32)
        return {"score": z, "f0_hz": np.zeros(n_frames, dtype=np.float32),
                "salience": np.zeros(n_frames, dtype=np.float32)}

    candidates = np.arange(bin_min, bin_max + 1)

    # Harmonic sum over candidate fundamentals, vectorised over frames.
    hss = np.zeros((len(candidates), n_frames), dtype=np.float32)
    for h in range(1, cfg.n_harmonics + 1):
        idx = candidates * h
        valid = idx < n_bins
        if not np.any(valid):
            break
        # 1/h weighting reflects the typical spectral rolloff of a voiced
        # source: upper partials carry less energy, so weighting them equally
        # would let a bright instrument outvote the true fundamental.
        hss[valid] += mag[idx[valid], :] / h

    best = np.argmax(hss, axis=0)
    f0_bins = candidates[best]
    f0_hz = f0_bins * sr / n_fft

    peak = hss[best, np.arange(n_frames)]
    mean_h = np.mean(hss, axis=0) + 1e-12
    # Salience: how much the winning f0 stands out from the field. A frame
    # with no clear pitch produces a flat HSS and low salience, which
    # correctly reports "no reliable comb here" instead of a spurious f0.
    salience = np.clip((peak / mean_h - 1.0) / 4.0, 0.0, 1.0)

    # Paint the comb.
    score = np.zeros_like(mag, dtype=np.float32)
    tol = int(cfg.comb_tolerance_bins)
    frame_idx = np.arange(n_frames)
    for h in range(1, cfg.n_harmonics + 1):
        centre = f0_bins * h
        for off in range(-tol, tol + 1):
            b = centre + off
            ok = (b >= 0) & (b < n_bins)
            if not np.any(ok):
                continue
            # Triangular weighting across the tolerance band: exact-centre
            # bins score fully, edges partially. A rectangular band would
            # create hard edges in the mask around every partial.
            weight = (1.0 - abs(off) / (tol + 1.0)) * salience[ok]
            np.maximum.at(score, (b[ok], frame_idx[ok]), weight)

    return {
        "score": np.clip(score, 0.0, 1.0).astype(np.float32),
        "f0_hz": f0_hz.astype(np.float32),
        "salience": salience.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# 4. REPETITION -- REPET-SIM foreground estimation
# ---------------------------------------------------------------------------

def repetition_score(
    mag: np.ndarray,
    sr: int,
    cfg: RepetitionConfig,
    stft_cfg: STFTConfig,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Non-repeating (foreground) energy, by nearest-neighbour median filtering.

    METHOD (REPET-SIM, Rafii & Pardo 2012)
    For frame m, find the n_neighbours most spectrally similar frames
    elsewhere in the piece (cosine similarity) and take their element-wise
    median:

        B(k,m) = median_{j in N(m)} |X(k,j)|        repeating estimate
        F(k,m) = max(|X(k,m)| - B(k,m), 0)          non-repeating residual

        score = F / (F + B + eps)

    A looped accompaniment appears near-identically in all neighbours, so it
    survives the median. A vocal line differs between them, so it does not.
    The residual therefore concentrates non-repeating content.

    ASSUMES the accompaniment repeats and the lead does not.

    FAILS on: through-composed, rubato, live or ambient material, where the
    backing does not repeat and leaks into the "vocal" residual; repeated
    choruses, where the vocal itself repeats and leaks into the background;
    and any piece shorter than a couple of repetitions.

    min_separation_s is an EXCLUSION radius, not a search window. librosa's
    nn_filter admits neighbours satisfying |i - j| >= width. Excluding nearby
    frames is essential and deliberate: adjacent frames hold the SAME sung
    note, so allowing them as neighbours would let the vocal survive its own
    median and be classified as repeating background. The previous
    implementation in audio_toolkit/separation.py documented this parameter
    with the opposite meaning.
    """
    n_frames = mag.shape[1]

    width = int(librosa.time_to_frames(
        cfg.min_separation_s, sr=sr, hop_length=stft_cfg.hop_length))
    max_width = max(1, (n_frames - 1) // 2 - 1)
    width = int(np.clip(width, 1, max_width))

    if n_frames <= cfg.max_frames_dense:
        B = _nn_median(mag, width, cfg.n_neighbours)
    else:
        B = _nn_median_blocked(mag, width, cfg.n_neighbours, cfg.max_frames_dense)

    # A neighbour median can exceed the frame's own energy (e.g. just before a
    # vocal onset, where neighbours are louder). Capping prevents the
    # background estimate from "creating" energy the mixture never had, which
    # would drive the residual negative and the score to nonsense.
    B = np.minimum(mag, B)
    F = np.maximum(mag - B, 0.0)

    return {
        "score": (F / (F + B + eps)).astype(np.float32),
        "background_mag": B.astype(np.float32),
        "foreground_mag": F.astype(np.float32),
    }


def _nn_median(mag, width, k):
    return librosa.decompose.nn_filter(
        mag, aggregate=np.median, metric="cosine", width=width, k=k)


def _nn_median_blocked(mag, width, k, block):
    """Process long signals in overlapping blocks.

    The similarity matrix is O(n_frames^2); a five-minute track at hop 1024
    is ~13k frames, so a dense float64 matrix would need ~1.4 GB. Blocking
    with 50% overlap bounds memory while letting each frame still see a wide
    span of the piece. Blocks are cross-faded so no discontinuity appears at
    a boundary -- an abrupt change in the background estimate would produce
    an audible click after masking.
    """
    n = mag.shape[1]
    step = max(1, block // 2)
    out = np.zeros_like(mag, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)

    for start in range(0, n, step):
        end = min(n, start + block)
        if end - start < 8:
            break
        seg = mag[:, start:end]
        w_local = int(np.clip(width, 1, max(1, (seg.shape[1] - 1) // 2 - 1)))
        est = _nn_median(seg, w_local, k)
        # Hann taper across the block for the cross-fade.
        taper = np.hanning(end - start + 2)[1:-1]
        if np.all(taper <= 0):
            taper = np.ones(end - start)
        out[:, start:end] += est * taper
        wsum[start:end] += taper
        if end >= n:
            break

    wsum[wsum < 1e-9] = 1.0
    return out / wsum


# ---------------------------------------------------------------------------
# 5. SPECTRAL -- broad formant-region shaping
# ---------------------------------------------------------------------------

def spectral_score(
    mag: np.ndarray,
    freqs: np.ndarray,
    eps: float = 1e-10,
) -> dict[str, np.ndarray]:
    """Weak prior from the frequency regions where voices concentrate energy.

    This is deliberately the LOWEST-weighted feature, and it is not a
    band-pass. Deleting everything outside a "vocal range" is the classic
    naive approach and it fails badly: it removes a singer's fundamental
    (below the band) and sibilance (above it) while keeping every guitar and
    synth inside it.

    Instead a smooth Gaussian emphasis is placed over the first two formant
    regions, which for sung vowels cluster around ~500 Hz (F1) and ~1500 Hz
    (F2), plus a gentle rise in the 4-8 kHz sibilance region. The score never
    reaches 0 or 1, so it can only tilt a decision that other features have
    left balanced -- it can never decide one alone.

    A local spectral-flatness term is included as evidence AGAINST voice:
    voiced sound is tonal (peaky, low flatness), whereas broadband noise-like
    energy (cymbals, distortion, hiss) has high flatness. Flatness is

        flatness = geometric_mean(|X|) / arithmetic_mean(|X|)

    computed over a local frequency neighbourhood, which is 1 for white noise
    and near 0 for a pure tone.
    """
    f = np.maximum(freqs, 1.0)
    logf = np.log(f)

    def bump(centre_hz, octaves, height):
        return height * np.exp(-0.5 * ((logf - np.log(centre_hz)) / (octaves * np.log(2))) ** 2)

    profile = 0.35 + bump(500.0, 1.0, 0.30) + bump(1500.0, 1.0, 0.25) + bump(6000.0, 0.8, 0.15)
    profile = np.clip(profile, 0.0, 1.0)

    # Local spectral flatness over a 9-bin neighbourhood.
    logmag = np.log(mag + eps)
    k = 9
    kern = np.ones(k) / k
    geo = np.exp(ndimage.convolve1d(logmag, kern, axis=0, mode="nearest"))
    ari = ndimage.convolve1d(mag, kern, axis=0, mode="nearest") + eps
    flatness = np.clip(geo / ari, 0.0, 1.0)

    tonality = 1.0 - flatness
    score = profile[:, None] * (0.5 + 0.5 * tonality)

    return {
        "score": np.clip(score, 0.0, 1.0).astype(np.float32),
        "flatness": flatness.astype(np.float32),
        "profile": profile.astype(np.float32),
    }
