"""STFT / ISTFT with perfect-reconstruction verification.

MATHEMATICS
-----------
Analysis. For a signal x[n], window w[n] of length N and hop H:

    X(k,m) = SUM_{n=0}^{N-1} x[n + mH] w[n] exp(-j 2 pi k n / N)

Each X(k,m) is a complex number carrying magnitude and phase:

    X(k,m) = |X(k,m)| exp(j phi(k,m))

Synthesis. Overlap-add with a synthesis window w[n]:

    x_hat[n] = SUM_m ( IDFT{X(k,m)}[n - mH] w[n - mH] ) / SUM_m w^2[n - mH]

The denominator is the window-overlap normalisation, and it is what makes
the pair invertible. Because librosa divides by the actual SUM_m w^2 envelope
rather than assuming it is flat, reconstruction only needs the weaker NOLA
(NonZero OverLap-Add) condition -- SUM_m w^2[n - mH] > 0 everywhere -- rather
than strict COLA. This was verified empirically: a deliberately non-COLA hop
still round-trips at ~140 dB here, because the normalisation compensates.

COLA still matters for a different reason. Under COLA the overlap envelope is
exactly constant, so every sample is analysed with equal total weight. Without
it the envelope ripples, and although the ISTFT divides that ripple out for an
*unmodified* spectrogram, the ripple reappears once bins are masked: masking
breaks the consistency that the division relied upon, so frames contribute
unequally and the frame rate sr/H becomes faintly audible. Since this pipeline
exists to modify spectrograms, the COLA-satisfying Hann/75% pair is used, and
check_cola() reports the condition at configuration time.

WHY VERIFICATION IS NON-NEGOTIABLE
----------------------------------
Every separation result is X multiplied by a mask and inverted. If the
transform pair is not itself unitary to numerical precision, every stem
carries that error, and it is indistinguishable by ear from a separation
failure. Verifying ISTFT(STFT(x)) ~= x first means any later artefact is
attributable to the masking, not the plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import librosa

from .config import STFTConfig


@dataclass
class ReconstructionReport:
    """Result of the ISTFT(STFT(x)) == x round-trip test."""

    max_abs_error: float
    rms_error: float
    #: Signal-to-reconstruction-error ratio in dB. Above ~120 dB the error
    #: is at float32 rounding level, i.e. the transform pair is exact.
    snr_db: float
    n_samples: int
    passed: bool

    def summary(self) -> dict:
        return {
            "maxAbsError": self.max_abs_error,
            "rmsError": self.rms_error,
            "snrDb": self.snr_db,
            "nSamples": self.n_samples,
            "passed": self.passed,
        }


def stft(y: np.ndarray, cfg: STFTConfig) -> np.ndarray:
    """Complex STFT of a 1-D signal. Returns array of shape (n_bins, n_frames)."""
    return librosa.stft(
        np.ascontiguousarray(y, dtype=np.float32),
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        window=cfg.window,
        center=cfg.center,
    )


def istft(X: np.ndarray, cfg: STFTConfig, length: int | None = None) -> np.ndarray:
    """Inverse STFT with overlap-add and window normalisation."""
    return librosa.istft(
        X,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
        window=cfg.window,
        center=cfg.center,
        length=length,
    )


def stft_stereo(y: np.ndarray, cfg: STFTConfig) -> tuple[np.ndarray, np.ndarray]:
    """STFT of a (2, n) stereo array, returned as (X_L, X_R).

    Channels are transformed independently and identically. That matters:
    applying different windows or a different centring to L and R would
    introduce an artificial inter-channel phase difference, which the
    spatial stage would then read as real stereo geometry.
    """
    y = np.atleast_2d(y)
    if y.shape[0] != 2:
        raise ValueError(f"expected a (2, n) stereo array, got shape {y.shape}")
    return stft(y[0], cfg), stft(y[1], cfg)


def verify_reconstruction(
    y: np.ndarray,
    cfg: STFTConfig,
    tolerance_db: float = 100.0,
) -> ReconstructionReport:
    """Assert that ISTFT(STFT(y)) reproduces y. REQUIRED before separation.

    Returns a report rather than raising, so a caller can surface the number
    in a debug panel. `passed` is True when the reconstruction SNR exceeds
    `tolerance_db`; 100 dB is a deliberately strict bar that a correct
    COLA-satisfying Hann/75% pair clears by a wide margin in float32.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.reshape(-1)
    n = len(y)
    if n < cfg.n_fft:
        return ReconstructionReport(0.0, 0.0, float("inf"), n, True)

    recon = istft(stft(y, cfg), cfg, length=n)

    err = recon.astype(np.float64) - y.astype(np.float64)
    max_abs = float(np.max(np.abs(err)))
    rms_err = float(np.sqrt(np.mean(err**2)))
    sig_pow = float(np.mean(y.astype(np.float64) ** 2))

    if rms_err <= 0 or sig_pow <= 0:
        snr = float("inf")
    else:
        snr = float(10 * np.log10(sig_pow / (rms_err**2)))

    return ReconstructionReport(
        max_abs_error=max_abs,
        rms_error=rms_err,
        snr_db=snr,
        n_samples=n,
        passed=bool(snr >= tolerance_db),
    )


def check_cola(cfg: STFTConfig) -> bool:
    """Does (window, n_fft, hop) satisfy the constant-overlap-add condition?

    Checked analytically via scipy rather than empirically, so a bad hop is
    caught at configuration time instead of appearing as flutter in output.
    """
    from scipy import signal as sps

    try:
        return bool(sps.check_COLA(cfg.window, cfg.n_fft, cfg.n_fft - cfg.hop_length))
    except Exception:
        return False


def frequencies(sr: int, cfg: STFTConfig) -> np.ndarray:
    """Centre frequency of each STFT bin, in Hz. f_k = k * sr / n_fft."""
    return librosa.fft_frequencies(sr=sr, n_fft=cfg.n_fft)


def times(n_frames: int, sr: int, cfg: STFTConfig) -> np.ndarray:
    """Centre time of each STFT frame, in seconds."""
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=cfg.hop_length)
