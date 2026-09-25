"""Debug and measurement mode.

The purpose of this module is to make separation failures DIAGNOSABLE rather
than merely audible. Listening tells you a stem sounds wrong; it does not tell
you whether the cause was a bad spatial estimate, an over-aggressive mask, a
reconstruction fault or a clipped input. Each measurement below is chosen
because it distinguishes between causes that sound similar.
"""
from __future__ import annotations

import numpy as np
import librosa

from .config import SeparationConfig
from . import stft as S
from . import midside as MS


def signal_stats(y: np.ndarray, sr: int) -> dict:
    """Per-signal measurements. `y` may be (n,) or (2, n).

    WHAT EACH ONE DIAGNOSES

    rms / peak / crest factor
        Crest = peak/RMS. A stem whose crest factor collapses toward 1 has
        been flattened by limiting or clipping; one that rises sharply after
        masking usually means the mask punched holes, leaving isolated peaks
        over near-silence -- the signature of musical noise.

    spectral centroid
        The energy-weighted mean frequency,

            centroid = SUM_k f_k |X_k| / SUM_k |X_k|

        i.e. the spectrum's "centre of mass", correlating with perceived
        brightness. If a karaoke stem's centroid drops far below the mix's,
        the mask has removed high-frequency content wholesale -- the usual
        cause of a dull, muffled instrumental.

    spectral bandwidth
        The spread about the centroid. A sharp narrowing indicates the
        telephone-band artefact: energy squeezed into a narrow mid region,
        which is what makes an over-processed vocal sound boxy.

    band energies
        Energy in six bands. Reading these across mix and stems shows exactly
        WHERE energy went. The classic karaoke failure -- a missing bass line
        -- appears here as a collapsed sub/bass figure long before it is
        consciously noticed by ear.

    clipping fraction
        Samples at or beyond full scale. Clipping generates broadband
        harmonic distortion that is easily mistaken for separation artefacts,
        so it must be ruled out before blaming the algorithm. Checked on the
        INPUT as well, since a clipped source cannot be separated cleanly.
    """
    y = np.asarray(y, dtype=np.float64)
    mono = y.mean(axis=0) if y.ndim > 1 else y
    if mono.size == 0:
        return {}

    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(mono**2)))

    D = np.abs(librosa.stft(mono.astype(np.float32), n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    tot = np.sum(D, axis=0) + 1e-12
    centroid = float(np.mean(np.sum(freqs[:, None] * D, axis=0) / tot))
    bandwidth = float(np.mean(
        np.sqrt(np.sum(((freqs[:, None] - centroid) ** 2) * D, axis=0) / tot)))

    bands = {"sub_20_60": (20, 60), "bass_60_250": (60, 250),
             "lowmid_250_1k": (250, 1000), "mid_1k_4k": (1000, 4000),
             "highmid_4k_8k": (4000, 8000), "high_8k_20k": (8000, 20000)}
    total_e = float(np.sum(D**2)) + 1e-12
    band_energy = {
        name: float(np.sum(D[(freqs >= lo) & (freqs < hi), :] ** 2) / total_e)
        for name, (lo, hi) in bands.items()
    }

    out = {
        "rms": rms,
        "rmsDb": float(20 * np.log10(rms + 1e-12)),
        "peak": peak,
        "peakDb": float(20 * np.log10(peak + 1e-12)),
        "crestFactor": float(peak / (rms + 1e-12)),
        "spectralCentroidHz": centroid,
        "spectralBandwidthHz": bandwidth,
        "bandEnergyFraction": band_energy,
        "clippingFraction": float(np.mean(np.abs(y) >= 0.999)),
    }

    if y.ndim > 1 and y.shape[0] == 2:
        L, R = y[0], y[1]
        if np.std(L) > 1e-12 and np.std(R) > 1e-12:
            out["stereoCorrelation"] = float(np.corrcoef(L, R)[0, 1])
        else:
            out["stereoCorrelation"] = 1.0 if np.allclose(L, R) else 0.0
        M, Sd = MS.to_mid_side(y.astype(np.float32))
        out["sideToMidDb"] = float(10 * np.log10(
            (np.sum(Sd.astype(np.float64) ** 2) + 1e-20)
            / (np.sum(M.astype(np.float64) ** 2) + 1e-20)))
    return out


def mask_stats(M: np.ndarray, name: str) -> dict:
    """Summarise a mask. Reveals the two commonest mask pathologies.

    A mask whose mean is near 1 is not separating at all -- the "keep
    everything" degenerate solution that an uncalibrated confidence produces.
    A mask with a large fraction of near-zero bins is over-aggressive and
    will sound like musical noise, because isolated surviving bins amid
    silenced neighbours are exactly what the ear hears as chirping.
    """
    M = np.asarray(M, dtype=np.float64)
    return {
        "name": name,
        "mean": float(M.mean()),
        "std": float(M.std()),
        "min": float(M.min()),
        "max": float(M.max()),
        "fractionNearZero": float(np.mean(M < 0.05)),
        "fractionNearOne": float(np.mean(M > 0.95)),
    }


def debug_report(result, mixture: np.ndarray, cfg: SeparationConfig) -> dict:
    """Full diagnostic bundle for one separation result.

    Returns statistics plus (when the result carries debug arrays) the
    spectrograms and masks needed to see WHY a decision was made:
    original / Mid / Side spectrograms, the vocal and instrumental masks,
    the harmonic and percussive masks, and the final masked spectrograms.

    Arrays are returned in dB and downsampled for display. Downsampling is
    for plotting only and never feeds back into the audio path.
    """
    sr = result.sr
    report: dict = {
        "method": result.method,
        "methodName": result.method_name,
        "assumptions": result.assumptions,
        "stereoReport": result.stereo_report,
        "reconstruction": result.reconstruction,
        "config": cfg.to_dict(),
        "stats": {
            "mixture": signal_stats(mixture, sr),
            "instrumental": signal_stats(result.instrumental, sr),
            "vocals": signal_stats(result.vocals, sr),
        },
    }

    M, Sd = MS.to_mid_side(np.asarray(mixture, dtype=np.float32))
    report["stats"]["mid"] = signal_stats(M, sr)
    report["stats"]["side"] = signal_stats(Sd, sr)

    d = result.debug or {}
    if not d:
        return report

    for key, label in (("mask_vocal", "vocalMask"),
                       ("mask_instrumental", "instrumentalMask")):
        if key in d:
            report.setdefault("maskStats", []).append(mask_stats(d[key], label))
    if "hpss" in d:
        report.setdefault("maskStats", []).append(mask_stats(d["hpss"]["harmonic"], "harmonicMask"))
        report.setdefault("maskStats", []).append(mask_stats(d["hpss"]["percussive"], "percussiveMask"))

    grids: dict = {}

    def add_db(name, mag):
        grids[name] = _downsample_db(mag)

    def add_lin(name, arr):
        grids[name] = _downsample(np.asarray(arr, dtype=np.float32)).tolist()

    X_M = S.stft(M, cfg.stft)
    X_S = S.stft(Sd, cfg.stft)
    add_db("midSpectrogram", np.abs(X_M))
    add_db("sideSpectrogram", np.abs(X_S))
    add_db("originalSpectrogram", np.abs(S.stft(
        np.asarray(mixture, dtype=np.float32).mean(axis=0), cfg.stft)))

    if "mask_vocal" in d:
        add_lin("vocalMask", d["mask_vocal"])
        add_db("finalVocalSpectrogram", np.abs(X_M) * d["mask_vocal"])
    if "mask_instrumental" in d:
        add_lin("instrumentalMask", d["mask_instrumental"])
        add_db("finalInstrumentalSpectrogram", np.abs(X_M) * d["mask_instrumental"])
    if "hpss" in d:
        add_lin("harmonicMask", d["hpss"]["harmonic"])
        add_lin("percussiveMask", d["hpss"]["percussive"])
    if "C_v" in d:
        add_lin("vocalConfidence", d["C_v"])
    if "spatial" in d:
        add_lin("centreDominance", d["spatial"]["centre_dominance"])
        add_lin("coherence", d["spatial"]["coherence"])

    report["grids"] = grids
    report["waveforms"] = {
        "original": _envelope(np.asarray(mixture).mean(axis=0)),
        "mid": _envelope(M),
        "side": _envelope(Sd),
        "instrumental": _envelope(np.asarray(result.instrumental).mean(axis=0)),
        "vocals": _envelope(np.asarray(result.vocals).mean(axis=0)),
    }
    return report


def _downsample(A: np.ndarray, max_f: int = 128, max_t: int = 256) -> np.ndarray:
    """Block-average a spectrogram down to a displayable size."""
    fs = max(1, A.shape[0] // max_f)
    ts = max(1, A.shape[1] // max_t)
    A = A[: (A.shape[0] // fs) * fs, : (A.shape[1] // ts) * ts]
    return A.reshape(A.shape[0] // fs, fs, A.shape[1] // ts, ts).mean(axis=(1, 3))


def _downsample_db(mag: np.ndarray) -> list:
    d = _downsample(np.asarray(mag, dtype=np.float32))
    return librosa.amplitude_to_db(d, ref=np.max).round(2).tolist()


def _envelope(y: np.ndarray, points: int = 1000) -> list:
    """Peak envelope for waveform display (min/max preserved per block).

    Plain decimation would alias and hide clipping; block peak preserves the
    extremes, so a clipped or hole-punched region stays visible in the plot.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        return []
    block = max(1, y.size // points)
    y = y[: (y.size // block) * block].reshape(-1, block)
    return np.abs(y).max(axis=1).round(5).tolist()
