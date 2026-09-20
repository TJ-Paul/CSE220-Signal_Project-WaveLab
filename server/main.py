"""Signal Lab API — HTTP surface over the audio_toolkit DSP modules.

Everything here is a thin transport layer: it loads/stores waveforms and
returns render-ready JSON. All actual signal processing stays in
audio_toolkit so the DSP and the UI evolve independently.
"""
from __future__ import annotations

import base64
import io
import os
import re
import sys

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_toolkit import (  # noqa: E402
    demo_signals,
    filters,
    io_utils,
    metrics,
    noise_reduction,
    sampling,
    separation,
    spectral,
    vad,
)
from server.store import Signal, store  # noqa: E402

app = FastAPI(title="Signal Lab API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def require(signal_id: str) -> Signal:
    sig = store.get(signal_id)
    if sig is None:
        raise HTTPException(status_code=404, detail="Signal not found")
    return sig


def peak_envelope(y: np.ndarray, buckets: int) -> dict:
    """Min/max envelope per bucket — how DAWs draw a waveform.

    Averaging would hide transients; min/max preserves the true extent of
    the signal at any zoom level.
    """
    n = len(y)
    if n == 0:
        return {"min": [], "max": []}
    buckets = max(1, min(buckets, n))
    edges = np.linspace(0, n, buckets + 1).astype(int)
    mins = np.empty(buckets, dtype=np.float32)
    maxs = np.empty(buckets, dtype=np.float32)
    for i in range(buckets):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        chunk = y[lo:hi]
        mins[i] = chunk.min()
        maxs[i] = chunk.max()
    return {"min": mins.round(5).tolist(), "max": maxs.round(5).tolist()}


def downsample_curve(x: np.ndarray, y: np.ndarray, points: int) -> dict:
    """Peak-preserving decimation for line curves (spectra, responses)."""
    n = len(y)
    if n <= points:
        return {"x": x.round(3).tolist(), "y": y.round(5).tolist()}
    edges = np.linspace(0, n, points + 1).astype(int)
    xs, ys = [], []
    for i in range(points):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        seg = y[lo:hi]
        idx = lo + int(np.argmax(np.abs(seg)))
        xs.append(float(x[idx]))
        ys.append(float(y[idx]))
    return {"x": np.round(xs, 3).tolist(), "y": np.round(ys, 5).tolist()}


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Signal lifecycle
# ---------------------------------------------------------------------------

@app.get("/api/signals")
def list_signals():
    return {"signals": [s.summary() for s in store.list()]}


@app.post("/api/signals/upload")
async def upload_signal(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        y, sr = io_utils.load_audio(io.BytesIO(raw), sr=None, mono=True)
        info = io_utils.get_audio_info(io.BytesIO(raw), filename=file.filename,
                                       file_size_bytes=len(raw))
    except Exception as exc:  # noqa: BLE001 — surface the real decode error
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    sig = store.add(
        file.filename or "uploaded audio", y, sr, origin="upload",
        format=info.subtype, channels=info.channels, bitDepth=info.bit_depth,
        bitrateKbps=round(info.bitrate_kbps, 1), isLossy=info.is_lossy,
        fileSizeBytes=len(raw),
    )
    return sig.summary()


@app.post("/api/signals/demo/{kind}")
def load_demo(kind: str):
    """Load a demo signal, or hand back the one already in the session.

    Demos are deterministic, so generating a second identical copy would
    only clutter the signal list.
    """
    existing = next(
        (s for s in store.list() if s.origin == "demo" and s.meta.get("demoKind") == kind),
        None,
    )
    if existing is not None:
        return existing.summary()

    if kind == "speech":
        y, sr = demo_signals.generate_speech_like_demo()
        sig = store.add("Speech-like demo", y, sr, origin="demo",
                        format="Synthetic", channels=1, demoKind=kind)
    elif kind == "song":
        mix, fg, bg, sr = demo_signals.generate_song_like_demo()
        sig = store.add("Song-like demo", mix, sr, origin="demo",
                        format="Synthetic", channels=1, demoKind=kind)
        store.add("Song demo · true melody", fg, sr, origin="derived",
                  source_id=sig.id, format="Synthetic", channels=1, groundTruth="foreground")
        store.add("Song demo · true accompaniment", bg, sr, origin="derived",
                  source_id=sig.id, format="Synthetic", channels=1, groundTruth="background")
    elif kind == "noisy":
        clean, noisy, sr = demo_signals.generate_noisy_tone_demo()
        sig = store.add("Noisy tone demo", noisy, sr, origin="demo",
                        format="Synthetic", channels=1, demoKind=kind)
        store.add("Noisy tone · clean reference", clean, sr, origin="derived",
                  source_id=sig.id, format="Synthetic", channels=1, groundTruth="clean")
    else:
        raise HTTPException(status_code=404, detail=f"Unknown demo '{kind}'")
    return sig.summary()


@app.delete("/api/signals/{signal_id}")
def delete_signal(signal_id: str):
    require(signal_id)
    store.remove(signal_id)
    _WAV_CACHE.pop(signal_id, None)
    return {"ok": True}


@app.post("/api/signals/clear")
def clear_signals():
    store.clear()
    _WAV_CACHE.clear()
    return {"ok": True}


@app.get("/api/signals/{signal_id}")
def get_signal(signal_id: str):
    return require(signal_id).summary()


_WAV_CACHE: dict[str, bytes] = {}


def encoded_wav(sig: Signal) -> bytes:
    """Encode once and reuse — range requests hit this repeatedly."""
    cached = _WAV_CACHE.get(sig.id)
    if cached is None:
        cached = wav_bytes(sig.y, sig.sr)
        _WAV_CACHE[sig.id] = cached
    return cached


@app.get("/api/signals/{signal_id}/audio")
def get_audio(signal_id: str, request: Request):
    """Serve the waveform as WAV, honouring HTTP Range.

    Range support is what makes the element seekable: without a 206 reply
    browsers clamp every seek back to zero, so the playhead never moves.
    """
    sig = require(signal_id)
    data = encoded_wav(sig)
    total = len(data)
    base_headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}

    range_header = request.headers.get("range")
    match = re.match(r"bytes=(\d*)-(\d*)", range_header or "")
    if match and (match.group(1) or match.group(2)):
        if match.group(1):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else total - 1
        else:  # suffix form: bytes=-N (final N bytes)
            start = max(0, total - int(match.group(2)))
            end = total - 1
        if start >= total:
            return Response(
                status_code=416,
                headers={**base_headers, "Content-Range": f"bytes */{total}"},
            )
        end = min(end, total - 1)
        chunk = data[start : end + 1]
        return Response(
            content=chunk,
            status_code=206,
            media_type="audio/wav",
            headers={
                **base_headers,
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Content-Length": str(len(chunk)),
            },
        )

    return Response(
        content=data,
        media_type="audio/wav",
        headers={**base_headers, "Content-Length": str(total)},
    )


# ---------------------------------------------------------------------------
# Time domain
# ---------------------------------------------------------------------------

@app.get("/api/signals/{signal_id}/waveform")
def get_waveform(signal_id: str, points: int = 2000, start: float = 0.0, end: float = -1.0):
    sig = require(signal_id)
    lo = max(0, int(start * sig.sr))
    hi = len(sig.y) if end < 0 else min(len(sig.y), int(end * sig.sr))
    seg = sig.y[lo:hi] if hi > lo else sig.y
    env = peak_envelope(seg, points)
    seg64 = seg.astype(np.float64)
    peak = float(np.max(np.abs(seg64))) if len(seg64) else 0.0
    rms = float(np.sqrt(np.mean(seg64**2))) if len(seg64) else 0.0
    return {
        **env,
        "startTime": lo / sig.sr,
        "endTime": hi / sig.sr,
        "duration": sig.duration,
        "stats": {
            "peak": peak,
            "rms": rms,
            "peakDb": float(20 * np.log10(peak + 1e-12)),
            "rmsDb": float(20 * np.log10(rms + 1e-12)),
            "crestFactor": float(peak / (rms + 1e-12)),
            "zeroCrossings": int(np.count_nonzero(np.diff(np.signbit(seg)))) if len(seg) > 1 else 0,
        },
    }


# ---------------------------------------------------------------------------
# Frequency domain
# ---------------------------------------------------------------------------

@app.get("/api/signals/{signal_id}/fft")
def get_fft(signal_id: str, points: int = 1400, peaks: int = 5,
            frame_ms: float = 0.0, center: float = 0.0):
    sig = require(signal_id)
    if frame_ms > 0:
        frame_len = max(8, int(sig.sr * frame_ms / 1000))
        start = max(0, int(center * sig.sr) - frame_len // 2)
        frame = sig.y[start:start + frame_len]
        if len(frame) < 8:
            frame = sig.y[:frame_len]
        target = frame * np.hamming(len(frame))
    else:
        target = sig.y

    freqs, mag = spectral.compute_fft(target, sig.sr)
    peak_idx, _ = find_peaks(mag, distance=max(1, len(mag) // 200))
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(mag))])
    order = peak_idx[np.argsort(mag[peak_idx])[::-1]][:peaks]

    return {
        **downsample_curve(freqs, mag, points),
        "nyquist": sig.sr / 2,
        "peakFrequency": float(freqs[int(np.argmax(mag))]),
        "spectralEnergy": float(np.sum(mag.astype(np.float64) ** 2)),
        "dominant": [
            {"frequency": float(freqs[i]), "magnitude": float(mag[i])} for i in order
        ],
    }


@app.get("/api/signals/{signal_id}/spectrogram")
def get_spectrogram(signal_id: str, n_fft: int = 2048, hop: int = 512,
                    max_frames: int = 1200, max_bins: int = 512, range_db: float = 70.0,
                    db_min: float | None = None, db_max: float | None = None):
    """dB magnitudes quantised to uint8 + base64 — a fraction of JSON's size.

    The frontend maps these bytes straight into canvas ImageData through a
    colormap, so a full spectrogram paints in one pass.

    db_min/db_max pin the scale explicitly, which is what makes two
    spectrograms (e.g. noisy vs denoised) actually comparable.
    """
    sig = require(signal_id)
    freqs, times, mag_db = spectral.spectrogram_db(sig.y, sig.sr, n_fft=n_fft, hop_length=hop)

    if mag_db.shape[1] > max_frames:
        idx = np.linspace(0, mag_db.shape[1] - 1, max_frames).astype(int)
        mag_db, times = mag_db[:, idx], times[idx]

    if mag_db.shape[0] > max_bins:
        # Max-pool frequency bins: preserves narrowband peaks that averaging
        # would smear away.
        groups = np.array_split(mag_db, max_bins, axis=0)
        mag_db = np.stack([g.max(axis=0) for g in groups])

    vmax = float(db_max) if db_max is not None else float(mag_db.max())
    vmin = float(db_min) if db_min is not None else float(max(mag_db.min(), vmax - range_db))
    norm = np.clip((mag_db - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    quantised = (norm * 255).astype(np.uint8)

    return {
        "width": int(quantised.shape[1]),
        "height": int(quantised.shape[0]),
        "data": base64.b64encode(quantised.tobytes()).decode("ascii"),
        "dbMin": vmin,
        "dbMax": vmax,
        "maxFrequency": float(freqs[-1]),
        "duration": float(times[-1]) if len(times) else sig.duration,
        "freqResolution": sig.sr / n_fft,
        "timeResolution": hop / sig.sr * 1000,
    }


# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------

class VadRequest(BaseModel):
    frameMs: float = 25.0
    hopMs: float = 10.0
    energyPercentile: float = 40.0
    bandRatioThresh: float = 0.35


@app.post("/api/signals/{signal_id}/vad")
def run_vad(signal_id: str, req: VadRequest):
    sig = require(signal_id)
    result = vad.detect_speech(
        sig.y, sig.sr, frame_ms=req.frameMs, hop_ms=req.hopMs,
        energy_percentile=req.energyPercentile, band_ratio_thresh=req.bandRatioThresh,
    )
    speech_time = sum(e - s for s, e in result.speech_segments)
    total = sig.duration
    return {
        "frameTimes": np.round(result.frame_times, 4).tolist(),
        "energyDb": np.round(result.energy_db, 2).tolist(),
        "bandRatio": np.round(result.speech_band_ratio, 4).tolist(),
        "isSpeech": result.is_speech.astype(bool).tolist(),
        "segments": [{"start": round(s, 3), "end": round(e, 3)} for s, e in result.speech_segments],
        "speechDuration": speech_time,
        "silenceDuration": total - speech_time,
        "speechRatio": (speech_time / total) if total else 0.0,
        "segmentCount": len(result.speech_segments),
    }


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class FilterRequest(BaseModel):
    kind: str = "lowpass"
    cutoff: float | None = None
    band: list[float] | None = None
    order: int = 6
    apply: bool = False


@app.post("/api/signals/{signal_id}/filter")
def run_filter(signal_id: str, req: FilterRequest):
    sig = require(signal_id)
    nyquist = sig.sr / 2
    if req.kind in ("bandpass", "bandstop"):
        if not req.band or len(req.band) != 2:
            raise HTTPException(status_code=400, detail="band [low, high] required for this filter")
        cutoff: float | tuple[float, float] = (req.band[0], req.band[1])
        label = f"{req.band[0]:.0f}–{req.band[1]:.0f} Hz"
    else:
        if req.cutoff is None:
            raise HTTPException(status_code=400, detail="cutoff required for this filter")
        cutoff = req.cutoff
        label = f"{req.cutoff:.0f} Hz"

    try:
        sos = filters.design_filter(req.kind, cutoff, sig.sr, order=req.order)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    freqs, mag_db = filters.frequency_response(sos, sig.sr)
    payload = {
        "response": downsample_curve(freqs, mag_db, 900),
        "cutoffs": [cutoff] if isinstance(cutoff, (int, float)) else list(cutoff),
        "nyquist": nyquist,
        "label": f"{req.kind.title()} · {label} · order {req.order}",
    }

    if req.apply:
        filtered = filters.apply_filter(sig.y, sos)
        new = store.add(f"{sig.name} · {req.kind} {label}", filtered, sig.sr,
                        origin="derived", source_id=sig.id, format="Processed", channels=1)
        payload["result"] = new.summary()
    return payload


# ---------------------------------------------------------------------------
# Noise reduction / separation / resampling
# ---------------------------------------------------------------------------

class DenoiseRequest(BaseModel):
    noiseDurationS: float = 0.5
    alpha: float = 2.0
    beta: float = 0.05


@app.post("/api/signals/{signal_id}/denoise")
def run_denoise(signal_id: str, req: DenoiseRequest):
    sig = require(signal_id)
    denoised = noise_reduction.spectral_subtraction(
        sig.y, sig.sr, noise_duration_s=req.noiseDurationS, alpha=req.alpha, beta=req.beta,
    )
    new = store.add(f"{sig.name} · denoised", denoised, sig.sr, origin="derived",
                    source_id=sig.id, format="Processed", channels=1)

    payload = {"result": new.summary()}
    reference = next((s for s in store.list()
                      if s.source_id == sig.id and s.meta.get("groundTruth") == "clean"), None)
    if reference is not None:
        payload["snrBefore"] = float(metrics.snr_db(reference.y, sig.y))
        payload["snrAfter"] = float(metrics.snr_db(reference.y, denoised))
        payload["correlationAfter"] = float(metrics.correlation(reference.y, denoised))
    return payload


class SeparateRequest(BaseModel):
    marginBackground: float = 2.0
    marginForeground: float = 10.0


@app.post("/api/signals/{signal_id}/separate")
def run_separate(signal_id: str, req: SeparateRequest):
    sig = require(signal_id)
    fg, bg = separation.separate_vocals_instrumental(
        sig.y, sig.sr,
        margin_background=req.marginBackground, margin_foreground=req.marginForeground,
    )
    fg_sig = store.add(f"{sig.name} · vocals", fg, sig.sr, origin="derived",
                       source_id=sig.id, format="Processed", channels=1)
    bg_sig = store.add(f"{sig.name} · instrumental", bg, sig.sr, origin="derived",
                       source_id=sig.id, format="Processed", channels=1)

    payload = {"foreground": fg_sig.summary(), "background": bg_sig.summary()}
    truth_fg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "foreground"), None)
    truth_bg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "background"), None)
    if truth_fg is not None and truth_bg is not None:
        payload["truth"] = {
            "foregroundCorrelation": float(metrics.correlation(truth_fg.y, fg)),
            "backgroundCorrelation": float(metrics.correlation(truth_bg.y, bg)),
        }
    return payload


class ResampleRequest(BaseModel):
    targetSr: int


@app.post("/api/signals/{signal_id}/resample")
def run_resample(signal_id: str, req: ResampleRequest):
    sig = require(signal_id)
    down = sampling.resample_audio(sig.y, sig.sr, req.targetSr)
    back = sampling.resample_audio(down, req.targetSr, sig.sr)
    new = store.add(f"{sig.name} · via {req.targetSr} Hz", back, sig.sr, origin="derived",
                    source_id=sig.id, format="Processed", channels=1)
    return {
        "result": new.summary(),
        "mse": float(metrics.mse(sig.y, back)),
        "snr": float(metrics.snr_db(sig.y, back)),
        "correlation": float(metrics.correlation(sig.y, back)),
    }


# ---------------------------------------------------------------------------
# Sampling / aliasing teaching demo (no loaded signal required)
# ---------------------------------------------------------------------------

@app.get("/api/sampling/demo")
def sampling_demo(freq: float = 500.0, fs: int = 800, duration: float = 0.05):
    t_hi, x_hi = sampling.generate_tone(freq, duration, fs_reference=44100)
    t_samples, x_samples = sampling.sample_signal(freq, duration, fs)
    x_recon = sampling.reconstruct_signal(t_samples, x_samples, t_hi)
    nyquist = fs / 2
    aliased = sampling.aliased_frequency(freq, fs) if freq > nyquist else None

    return {
        "continuous": {"t": np.round(t_hi, 6).tolist(), "x": np.round(x_hi, 5).tolist()},
        "reconstructed": {"t": np.round(t_hi, 6).tolist(), "x": np.round(x_recon, 5).tolist()},
        "samples": {"t": np.round(t_samples, 6).tolist(), "x": np.round(x_samples, 5).tolist()},
        "nyquist": nyquist,
        "aliasing": freq > nyquist,
        "aliasedFrequency": aliased,
        "mse": float(metrics.mse(x_hi, x_recon)),
        "correlation": float(metrics.correlation(x_hi, x_recon)),
        "snr": float(metrics.snr_db(x_hi, x_recon)),
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@app.get("/api/compare")
def compare(a: str, b: str, points: int = 1400):
    sig_a, sig_b = require(a), require(b)
    y_b, sr_b = sig_b.y, sig_b.sr
    resampled = False
    if sr_b != sig_a.sr:
        y_b = sampling.resample_audio(y_b, sr_b, sig_a.sr)
        resampled = True

    n = min(len(sig_a.y), len(y_b))
    a64, b64 = sig_a.y[:n].astype(np.float64), y_b[:n].astype(np.float64)
    rms_a = float(np.sqrt(np.mean(a64**2))) if n else 0.0
    rms_b = float(np.sqrt(np.mean(b64**2))) if n else 0.0
    peak_a = float(np.max(np.abs(a64))) if n else 0.0
    peak_b = float(np.max(np.abs(b64))) if n else 0.0

    freqs_a, mag_a = spectral.compute_fft(sig_a.y, sig_a.sr)
    freqs_b, mag_b = spectral.compute_fft(y_b, sig_a.sr)

    return {
        "a": sig_a.summary(),
        "b": sig_b.summary(),
        "resampled": resampled,
        "metrics": {
            "mse": float(metrics.mse(sig_a.y, y_b)),
            "snr": float(metrics.snr_db(sig_a.y, y_b)),
            "correlation": float(metrics.correlation(sig_a.y, y_b)),
            "rmsDifference": abs(rms_a - rms_b),
            "peakDifference": abs(peak_a - peak_b),
        },
        "waveformA": peak_envelope(sig_a.y, points),
        "waveformB": peak_envelope(y_b, points),
        "spectrumA": downsample_curve(freqs_a, mag_a, points),
        "spectrumB": downsample_curve(freqs_b, mag_b, points),
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "signals": len(store.list())}
