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
from PIL import Image as PILImage
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_toolkit import (  # noqa: E402
    demo_signals,
    editing,
    filters,
    io_utils,
    metrics,
    noise_reduction,
    sampling,
    separation,
    silence,
    spectral,
    timescale,
    vad,
    vocals,
)
from audio_toolkit import stereo_sep
import vault  # noqa: E402
from vault import stego as vault_stego  # noqa: E402
from vault.container import HEADER_SIZE as VAULT_HEADER_SIZE  # noqa: E402
from server.store import Signal, store  # noqa: E402
from server.vault_store import vault_store  # noqa: E402

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


def decimate_mean(x: np.ndarray, y: np.ndarray, points: int) -> dict:
    """Bucket-average decimation, for envelopes rather than peaks.

    `downsample_curve` keeps the largest-magnitude sample per bucket, which
    is right for spectra but wrong for a dB level curve: there the largest
    magnitude is the *most negative* value, so a silence detector's trace
    would be pulled toward the floor everywhere. Averaging shows the level
    the ear would actually hear over each bucket.
    """
    n = len(y)
    if n <= points:
        return {"x": np.round(x, 4).tolist(), "y": np.round(y, 2).tolist()}
    edges = np.linspace(0, n, points + 1).astype(int)
    xs, ys = [], []
    for i in range(points):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        xs.append(float(x[lo]))
        ys.append(float(np.mean(y[lo:hi])))
    return {"x": np.round(xs, 4).tolist(), "y": np.round(ys, 2).tolist()}


def wav_bytes(y: np.ndarray, sr: int) -> bytes:
    """Encode a waveform as 16-bit PCM WAV.

    Accepts mono (n,) or stereo (2, n); a stereo array is transposed to the
    interleaved (n, 2) layout libsndfile expects. Processing stays in float
    throughout and is converted to integer PCM only here, at the very last
    step, so no intermediate stage is quantised.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2 and y.shape[0] == 2:
        y = y.T
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
        # Also keep the stereo field. The mono downmix above computes
        # (L+R)/2 = Mid and destroys Side = (L-R)/2, which cannot be
        # recovered afterwards. For a centre-panned vocal Side is provably
        # vocal-free, so discarding it removes the strongest evidence a
        # stereo recording carries and makes stereo separation impossible.
        # Every existing mono analysis path still reads `y`.
        y_stereo = None
        if info.channels >= 2:
            try:
                y_stereo, _ = stereo_sep.load_stereo(io.BytesIO(raw), sr=sr)
            except Exception:  # noqa: BLE001 — stereo is optional, mono still works
                y_stereo = None
    except Exception as exc:  # noqa: BLE001 — surface the real decode error
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    sig = store.add(
        file.filename or "uploaded audio", y, sr, origin="upload",
        y_stereo=y_stereo,
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
        # Serve stereo when the signal has it. The separation stems are
        # genuinely two-channel -- the instrumental keeps the mix's full
        # stereo image because the Side channel passes through unmasked --
        # so serving the mono downmix here would discard the width the
        # pipeline worked to preserve.
        source = sig.y_stereo if sig.has_stereo else sig.y
        cached = wav_bytes(source, sig.sr)
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


# ---------------------------------------------------------------------------
# Classical stereo source separation (karaoke / vocal focus)
# ---------------------------------------------------------------------------

class StereoSeparateRequest(BaseModel):
    """Request for the stereo DSP separation pipeline.

    `config` is a partial, nested override of SeparationConfig; anything
    omitted keeps its documented default.
    """
    method: str = "m5"
    debug: bool = False
    config: dict | None = None


@app.get("/api/separation/methods")
def separation_methods():
    """The A/B method catalogue, with the assumptions each one relies on."""
    return {
        "methods": [{"id": k, "name": v} for k, v in stereo_sep.METHOD_NAMES.items()],
        "defaultConfig": stereo_sep.SeparationConfig().to_dict(),
        "listeningGuide": stereo_sep.LISTENING_GUIDE,
    }


@app.post("/api/signals/{signal_id}/stereo-separate")
def run_stereo_separate(signal_id: str, req: StereoSeparateRequest):
    """Split a stereo mixture into karaoke and vocal-focus stems.

    Requires a stereo source. This is a hard requirement, not a limitation
    that can be engineered around: the pipeline's strongest evidence is the
    Side signal S = (L-R)/2, in which a centre-panned vocal cancels exactly.
    A mono file has no Side channel, so that evidence does not exist and the
    request is refused rather than silently returning a worse result from a
    method the caller did not ask for.
    """
    sig = require(signal_id)
    if req.method not in stereo_sep.METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"method must be one of {list(stereo_sep.METHODS)}")

    if not sig.has_stereo:
        raise HTTPException(
            status_code=400,
            detail="This signal is mono. Stereo separation needs both channels: "
                   "the Side signal (L-R)/2, where a centred vocal cancels, is the "
                   "pipeline's primary evidence and does not exist in a mono file. "
                   "Upload the original stereo file.")

    cfg = stereo_sep.SeparationConfig.from_dict(req.config or {})
    cfg.debug = bool(req.debug)

    y = sig.y_stereo
    report = stereo_sep.analyse_stereo(y, sig.sr)
    result = stereo_sep.separate(y, sig.sr, method=req.method, cfg=cfg)

    karaoke = store.add(
        f"{sig.name} · karaoke [{req.method}]", result.instrumental.mean(axis=0),
        sig.sr, origin="derived", source_id=sig.id, y_stereo=result.instrumental,
        format="Karaoke", channels=2, stem="instrumental", method=req.method)
    vocal = store.add(
        f"{sig.name} · vocal focus [{req.method}]", result.vocals.mean(axis=0),
        sig.sr, origin="derived", source_id=sig.id, y_stereo=result.vocals,
        format="Vocal focus", channels=2, stem="vocals", method=req.method)

    payload: dict = {
        "method": result.method,
        "methodName": result.method_name,
        "assumptions": result.assumptions,
        "stereoReport": report.summary(),
        "reconstruction": result.reconstruction,
        "karaoke": karaoke.summary(),
        "vocalFocus": vocal.summary(),
        "proxyMetrics": stereo_sep.proxy_metrics(
            y, result.instrumental, result.vocals, sig.sr),
    }

    # A near-mono input is reported rather than hidden. The spatial features
    # carry no information there, so the caller should know the result rests
    # entirely on the non-spatial evidence.
    if report.is_effectively_mono:
        payload["warning"] = (
            "Channels are nearly identical (correlation "
            f"{report.lr_correlation:.3f}). There is almost no Side signal, so "
            "spatial separation has little to work with and the result depends "
            "on repetition and harmonic/percussive evidence alone.")

    if req.debug:
        payload["debug"] = stereo_sep.debug_report(result, y, cfg)

    truth_fg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "foreground"), None)
    truth_bg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "background"), None)
    if truth_fg is not None and truth_bg is not None:
        payload["bss"] = {
            "instrumental": stereo_sep.bss_eval(result.instrumental, truth_bg.y, truth_fg.y),
            "vocals": stereo_sep.bss_eval(result.vocals, truth_fg.y, truth_bg.y),
        }
    return payload


@app.post("/api/signals/{signal_id}/stereo-compare")
def run_stereo_compare(signal_id: str, req: StereoSeparateRequest):
    """Run every method on one signal so the extra mathematics can be judged.

    Returns a do-nothing reference row when ground truth is available. That
    row is the important one: a method scoring below the untouched mixture is
    actively harmful, and during development two methods did exactly that.
    """
    sig = require(signal_id)
    if not sig.has_stereo:
        raise HTTPException(status_code=400, detail="Stereo source required.")

    from audio_toolkit.stereo_sep.compare import compare_methods, format_table

    cfg = stereo_sep.SeparationConfig.from_dict(req.config or {})
    truth_fg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "foreground"), None)
    truth_bg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "background"), None)

    comparison = compare_methods(
        sig.y_stereo, sig.sr, cfg=cfg,
        true_instrumental=truth_bg.y if truth_bg is not None else None,
        true_vocals=truth_fg.y if truth_fg is not None else None,
    )
    comparison["table"] = format_table(comparison)
    return comparison


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
# Editing — trim, cut, fade, merge
# ---------------------------------------------------------------------------

class TrimRequest(BaseModel):
    start: float
    end: float
    mode: str = "keep"  # "keep" trims to the selection, "cut" deletes it


@app.post("/api/signals/{signal_id}/trim")
def run_trim(signal_id: str, req: TrimRequest):
    sig = require(signal_id)
    if req.mode not in ("keep", "cut"):
        raise HTTPException(status_code=400, detail="mode must be 'keep' or 'cut'")

    span = f"{req.start:.2f}–{req.end:.2f} s"
    if req.mode == "keep":
        edited = editing.trim(sig.y, sig.sr, req.start, req.end)
        label = f"trim {span}"
    else:
        edited = editing.cut(sig.y, sig.sr, req.start, req.end)
        label = f"cut {span}"

    new = store.add(f"{sig.name} · {label}", edited, sig.sr, origin="derived",
                    source_id=sig.id, format="Edited", channels=1)
    return {
        "result": new.summary(),
        "sourceDuration": sig.duration,
        "selectionDuration": max(0.0, req.end - req.start),
        "removedDuration": sig.duration - new.duration,
    }


class FadeRequest(BaseModel):
    fadeInS: float = 0.0
    fadeOutS: float = 0.0
    shape: str = "scurve"
    normalize: bool = False


@app.post("/api/signals/{signal_id}/fade")
def run_fade(signal_id: str, req: FadeRequest):
    sig = require(signal_id)
    if req.shape not in editing.FADE_SHAPES:
        raise HTTPException(status_code=400, detail=f"shape must be one of {editing.FADE_SHAPES}")

    faded = editing.apply_fades(sig.y, sig.sr, req.fadeInS, req.fadeOutS, req.shape)
    if req.normalize:
        faded = editing.peak_normalize(faded)

    parts = []
    if req.fadeInS > 0:
        parts.append(f"in {req.fadeInS:.2f}s")
    if req.fadeOutS > 0:
        parts.append(f"out {req.fadeOutS:.2f}s")
    label = f"fade {' / '.join(parts) or 'none'}"

    new = store.add(f"{sig.name} · {label}", faded, sig.sr, origin="derived",
                    source_id=sig.id, format="Edited", channels=1)
    return {"result": new.summary(), "shape": req.shape}


@app.get("/api/fades/shapes")
def fade_shapes(points: int = 160):
    """Every fade curve, for plotting the choice before committing to it."""
    t = np.linspace(0.0, 1.0, points)
    return {
        "t": np.round(t, 4).tolist(),
        "curves": {
            shape: np.round(editing.fade_ramp(points, shape), 4).tolist()
            for shape in editing.FADE_SHAPES
        },
    }


class MergeRequest(BaseModel):
    ids: list[str]
    crossfadeS: float = 0.0
    gapS: float = 0.0
    law: str = "equal_power"
    normalize: bool = True
    name: str | None = None


@app.post("/api/merge")
def run_merge(req: MergeRequest):
    """Join several signals end to end.

    Clips recorded at different rates are resampled to the first clip's rate
    before joining: concatenating buffers of different sample rates would
    play the later ones at the wrong speed and pitch.
    """
    if len(req.ids) < 2:
        raise HTTPException(status_code=400, detail="Select at least two signals to merge")
    if req.law not in editing.CROSSFADE_LAWS:
        raise HTTPException(status_code=400, detail=f"law must be one of {editing.CROSSFADE_LAWS}")

    sources = [require(i) for i in req.ids]
    target_sr = sources[0].sr

    clips, manifest = [], []
    for sig in sources:
        y = sig.y if sig.sr == target_sr else sampling.resample_audio(sig.y, sig.sr, target_sr)
        clips.append(y)
        manifest.append({
            "id": sig.id,
            "name": sig.name,
            "duration": len(y) / target_sr,
            "resampled": sig.sr != target_sr,
            "sourceSampleRate": sig.sr,
        })

    merged = editing.splice(clips, target_sr, req.crossfadeS, req.law, req.gapS)
    if req.normalize:
        merged = editing.peak_normalize(merged)

    # Where each clip starts in the output, so the UI can mark the seams.
    # A crossfade overlaps neighbours, so each join pulls the timeline back
    # by the overlap; a gap pushes it forward.
    boundaries, cursor = [], 0.0
    for i, clip in enumerate(clips):
        boundaries.append(cursor)
        cursor += len(clip) / target_sr
        if i < len(clips) - 1:
            cursor += req.gapS - (req.crossfadeS if req.gapS <= 0 else 0.0)

    new = store.add(
        req.name or f"Merge of {len(clips)} signals", merged, target_sr,
        origin="derived", source_id=sources[0].id, format="Edited", channels=1,
    )
    return {
        "result": new.summary(),
        "sampleRate": target_sr,
        "clips": manifest,
        "boundaries": [round(b, 4) for b in boundaries[1:]],
        "totalSourceDuration": sum(c["duration"] for c in manifest),
    }


# ---------------------------------------------------------------------------
# Silence detection and removal
# ---------------------------------------------------------------------------

class SilenceRequest(BaseModel):
    #: None asks the server to pick a threshold from the level distribution.
    thresholdDb: float | None = None
    minSilenceS: float = 0.35
    padS: float = 0.08
    hysteresisDb: float = 6.0
    apply: bool = False


@app.post("/api/signals/{signal_id}/silence")
def run_silence(signal_id: str, req: SilenceRequest):
    """Analyse (and optionally remove) the silent stretches.

    Mirrors the filter endpoint: the same call previews and commits, so the
    regions drawn on screen are exactly the ones `apply` will delete.
    """
    sig = require(signal_id)
    suggested = silence.suggest_threshold_db(sig.y, sig.sr)
    threshold = suggested if req.thresholdDb is None else req.thresholdDb
    analysis = silence.analyze_silence(
        sig.y, sig.sr,
        threshold_db=threshold,
        min_silence_s=req.minSilenceS,
        pad_s=req.padS,
        hysteresis_db=req.hysteresisDb,
    )

    payload = {
        "levels": decimate_mean(analysis.frame_times, analysis.level_db, 1200),
        "relativeThresholdDb": threshold,
        "suggestedThresholdDb": suggested,
        "thresholdDb": analysis.threshold_db,
        "peakDb": analysis.peak_db,
        "regions": [{"start": round(s, 3), "end": round(e, 3)} for s, e in analysis.silent_regions],
        "keptRegions": [{"start": round(s, 3), "end": round(e, 3)} for s, e in analysis.kept_regions],
        "removedDuration": analysis.removed_duration,
        "keptDuration": analysis.kept_duration,
        "totalDuration": analysis.total_duration,
        "regionCount": len(analysis.silent_regions),
    }

    if req.apply:
        trimmed = silence.remove_silence(sig.y, sig.sr, analysis)
        new = store.add(f"{sig.name} · silence removed", trimmed, sig.sr, origin="derived",
                        source_id=sig.id, format="Edited", channels=1)
        payload["result"] = new.summary()
    return payload


# ---------------------------------------------------------------------------
# Time stretching and pitch shifting
# ---------------------------------------------------------------------------

def verify_transform(source: np.ndarray, result: np.ndarray, sr: int, expected_cents: float) -> dict:
    """Measure what actually happened to the pitch, and score it.

    The transposition estimate is the primary evidence — it needs no note
    to be present, so it is defined for music, speech and noise alike. F0
    is reported alongside when the material has a stable enough pitch for
    the number to mean anything.
    """
    shift = timescale.estimate_transposition_cents(source, result, sr)
    before = timescale.estimate_f0(source, sr)
    after = timescale.estimate_f0(result, sr)

    return {
        "expectedCents": expected_cents,
        "measuredCents": shift.cents,
        "errorCents": None if shift.cents is None else shift.cents - expected_cents,
        "confidence": shift.confidence,
        "sourceF0": before.f0_hz if before.stable else None,
        "resultF0": after.f0_hz if after.stable else None,
        "f0Stable": before.stable and after.stable,
        "f0SpreadCents": before.spread_cents,
    }


class SpeedRequest(BaseModel):
    rate: float = 1.0
    preservePitch: bool = True


@app.post("/api/signals/{signal_id}/speed")
def run_speed(signal_id: str, req: SpeedRequest):
    sig = require(signal_id)
    rate = float(np.clip(req.rate, 0.25, 4.0))
    changed = timescale.change_speed(sig.y, sig.sr, rate, preserve_pitch=req.preservePitch)

    method = "phase vocoder" if req.preservePitch else "resampled"
    new = store.add(f"{sig.name} · {rate:.2f}× ({method})", changed, sig.sr, origin="derived",
                    source_id=sig.id, format="Processed", channels=1)

    # A phase vocoder should move the pitch nowhere; resampling should move
    # it by exactly the speed ratio. Measuring which actually happened is
    # what separates a claim from a demonstration.
    expected = 0.0 if req.preservePitch else float(1200 * np.log2(rate))
    return {
        "result": new.summary(),
        "rate": rate,
        "preservePitch": req.preservePitch,
        "method": method,
        "sourceDuration": sig.duration,
        "resultDuration": new.duration,
        "verification": verify_transform(sig.y, changed, sig.sr, expected),
    }


class PitchRequest(BaseModel):
    semitones: float = 0.0


@app.post("/api/signals/{signal_id}/pitch")
def run_pitch(signal_id: str, req: PitchRequest):
    sig = require(signal_id)
    semitones = float(np.clip(req.semitones, -24.0, 24.0))
    shifted = timescale.shift_pitch(sig.y, sig.sr, semitones)

    sign = "+" if semitones >= 0 else "−"
    new = store.add(f"{sig.name} · pitch {sign}{abs(semitones):g} st", shifted, sig.sr,
                    origin="derived", source_id=sig.id, format="Processed", channels=1)

    return {
        "result": new.summary(),
        "semitones": semitones,
        "ratio": timescale.semitone_ratio(semitones),
        "sourceDuration": sig.duration,
        "resultDuration": new.duration,
        "verification": verify_transform(sig.y, shifted, sig.sr, semitones * 100.0),
    }


# ---------------------------------------------------------------------------
# Karaoke / vocal production
# ---------------------------------------------------------------------------

class VocalsRequest(BaseModel):
    preset: str = "balanced"
    outputs: str = "both"  # "karaoke" | "acapella" | "both"
    levelMatch: bool = True


@app.post("/api/signals/{signal_id}/vocals")
def run_vocals(signal_id: str, req: VocalsRequest):
    sig = require(signal_id)
    if req.preset not in vocals.PRESETS:
        raise HTTPException(status_code=400, detail=f"preset must be one of {list(vocals.PRESETS)}")
    if req.outputs not in ("karaoke", "acapella", "both"):
        raise HTTPException(status_code=400, detail="outputs must be 'karaoke', 'acapella' or 'both'")

    stems = vocals.make_stems(sig.y, sig.sr, preset=req.preset, level_match=req.levelMatch)

    payload: dict = {
        "preset": stems.preset,
        "levelMatched": req.levelMatch,
        "metrics": {
            "vocalSuppressionDb": stems.vocal_suppression_db,
            "stemCorrelation": stems.stem_correlation,
            "vocalEnergyShare": stems.vocal_energy_share,
        },
    }

    if req.outputs in ("karaoke", "both"):
        karaoke = store.add(f"{sig.name} · karaoke", stems.instrumental, sig.sr, origin="derived",
                            source_id=sig.id, format="Karaoke", channels=1, stem="instrumental")
        payload["karaoke"] = karaoke.summary()
    if req.outputs in ("acapella", "both"):
        acapella = store.add(f"{sig.name} · a cappella", stems.vocals, sig.sr, origin="derived",
                             source_id=sig.id, format="A cappella", channels=1, stem="vocals")
        payload["acapella"] = acapella.summary()

    # Ground truth exists only for the synthetic song demo, where the true
    # melody and accompaniment were generated separately before mixing.
    truth_fg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "foreground"), None)
    truth_bg = next((s for s in store.list()
                     if s.source_id == sig.id and s.meta.get("groundTruth") == "background"), None)
    if truth_fg is not None and truth_bg is not None:
        payload["truth"] = {
            "vocalsCorrelation": float(metrics.correlation(truth_fg.y, stems.vocals)),
            "instrumentalCorrelation": float(metrics.correlation(truth_bg.y, stems.instrumental)),
        }
    return payload


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


# ---------------------------------------------------------------------------
# Secure vault — audio ⇄ encrypted PNG
# ---------------------------------------------------------------------------

#: How many pixels of the cover and the stego image are retained for the
#: LSB teaching view. A few thousand is plenty to browse and costs ~24 KB,
#: versus tens of megabytes to keep both images whole.
PIXEL_WINDOW = 4096


def ingest_audio(raw: bytes, filename: str, origin: str, **meta) -> tuple[Signal | None, dict]:
    """Decode audio for *visualisation only*, alongside the raw bytes.

    Decoding is a convenience: it lets the vault show a waveform and a
    spectrum using the same machinery as the rest of the app. It is
    deliberately allowed to fail — encryption operates on the original
    bytes and does not care whether this app can parse the format, so an
    exotic codec should still encrypt rather than be rejected.
    """
    try:
        y, sr = io_utils.load_audio(io.BytesIO(raw), sr=None, mono=True)
        info = io_utils.get_audio_info(io.BytesIO(raw), filename=filename,
                                       file_size_bytes=len(raw))
    except Exception:  # noqa: BLE001 — any decode failure is non-fatal here
        return None, {}

    audio_info = {
        "sampleRate": int(sr),
        "channels": info.channels,
        "durationSeconds": round(len(y) / sr, 4) if sr else 0.0,
        "format": info.subtype,
        "bitDepth": info.bit_depth,
        "bitrateKbps": round(info.bitrate_kbps, 1),
        "isLossy": info.is_lossy,
    }
    sig = store.add(
        filename or "vault audio", y, sr, origin=origin,
        format=info.subtype, channels=info.channels, bitDepth=info.bit_depth,
        bitrateKbps=round(info.bitrate_kbps, 1), isLossy=info.is_lossy,
        fileSizeBytes=len(raw), **meta,
    )
    return sig, audio_info


def head_bytes(pixels: np.ndarray) -> bytes:
    return pixels.reshape(-1, 3)[:PIXEL_WINDOW].tobytes()


@app.post("/api/vault/encode")
async def vault_encode(
    file: UploadFile = File(...),
    password: str = Form(...),
    bitsPerChannel: int = Form(1),
    cover: UploadFile | None = File(None),
):
    """Encrypt an audio file and hide the ciphertext in a PNG."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if not password:
        raise HTTPException(status_code=400, detail="A password is required")
    if not 1 <= bitsPerChannel <= vault_stego.MAX_BITS_PER_CHANNEL:
        raise HTTPException(status_code=400,
                            detail=f"bitsPerChannel must be 1–{vault_stego.MAX_BITS_PER_CHANNEL}")

    filename = file.filename or "audio.bin"
    signal, audio_info = ingest_audio(raw, filename, origin="upload", vaultRole="source")

    cover_png = await cover.read() if cover is not None else None
    try:
        result = vault.encode_audio(
            raw, filename, password,
            audio_info=audio_info,
            bits_per_channel=bitsPerChannel,
            cover_png=cover_png or None,
        )
    except vault.VaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    artifact = vault_store.add(
        "image", result.png, f"{stem}.vault.png", "image/png",
        bitsPerChannel=bitsPerChannel,
        coverHead=head_bytes(result.cover),
        stegoHead=head_bytes(result.pixels),
        width=int(result.pixels.shape[1]),
        sourceHistogram=vault_stego.byte_histogram(raw),
    )

    return {
        "imageId": artifact.id,
        "imageName": artifact.filename,
        "signalId": signal.id if signal else None,
        "source": {
            "filename": filename,
            "size": len(raw),
            "sha256": result.metadata.sha256,
            "audio": audio_info,
            "histogram": vault_stego.byte_histogram(raw),
            "entropy": vault_stego.shannon_entropy(vault_stego.byte_histogram(raw)),
        },
        "usedCover": cover_png is not None,
        **result.metrics(),
    }


@app.get("/api/vault/image/{artifact_id}")
def vault_image(artifact_id: str):
    """The stego PNG itself — the file that must be kept, bit-exact."""
    artifact = vault_store.get(artifact_id)
    if artifact is None or artifact.kind != "image":
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(
        content=artifact.data,
        media_type="image/png",
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "Content-Length": str(len(artifact.data)),
        },
    )


@app.get("/api/vault/preview/{artifact_id}")
def vault_preview(artifact_id: str, max_side: int = 420):
    """A downscaled preview, for showing the image in the page.

    Resampling averages neighbouring pixels and therefore destroys the
    low-bit plane — which is exactly why this is a *preview* and the
    download above is the artefact. A screenshot of this image contains
    no recoverable payload, and the UI says so.

    Served as JPEG for the same reason: the payload is already gone at
    this point, so there is nothing left for lossy compression to
    damage, and a thumbnail of incompressible noise is an order of
    magnitude smaller as JPEG than as PNG.
    """
    artifact = vault_store.get(artifact_id)
    if artifact is None or artifact.kind != "image":
        raise HTTPException(status_code=404, detail="Image not found")

    with PILImage.open(io.BytesIO(artifact.data)) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), PILImage.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-cache"})


@app.get("/api/vault/pixels/{artifact_id}")
def vault_pixels(artifact_id: str, offset: int = 0, count: int = 8):
    """Before/after pixel values, for the LSB demonstration."""
    artifact = vault_store.get(artifact_id)
    if artifact is None or artifact.kind != "image":
        raise HTTPException(status_code=404, detail="Image not found")

    width = int(artifact.meta.get("width", 1))
    cover = np.frombuffer(artifact.meta["coverHead"], dtype=np.uint8).reshape(-1, 3)
    steg = np.frombuffer(artifact.meta["stegoHead"], dtype=np.uint8).reshape(-1, 3)
    count = int(np.clip(count, 1, 64))

    rows = vault_stego.pixel_comparison(
        cover.reshape(1, -1, 3), steg.reshape(1, -1, 3), count=count, offset=offset
    )
    for row in rows:  # the reshape above flattened the real geometry back in
        row["x"] = row["index"] % width
        row["y"] = row["index"] // width
    return {"rows": rows, "windowSize": len(cover), "bitsPerChannel": artifact.meta["bitsPerChannel"]}


@app.post("/api/vault/inspect")
async def vault_inspect(image: UploadFile = File(...), bitsPerChannel: int = Form(1)):
    """Read the cleartext header — what is knowable without the password."""
    data = await image.read()
    try:
        return vault.inspect_png(data, bits_per_channel=bitsPerChannel)
    except vault.VaultError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/vault/decode")
async def vault_decode(
    image: UploadFile = File(...),
    password: str = Form(...),
    bitsPerChannel: int = Form(1),
):
    """Recover the original audio file from a stego PNG."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty")

    try:
        result = vault.decode_png(data, password, bits_per_channel=bitsPerChannel)
    except vault.VaultError as exc:
        # 422 rather than 400: the request was well-formed, the *content*
        # failed to authenticate. The frontend keys its "authentication
        # failed" state off this.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    artifact = vault_store.add(
        "file", result.file_bytes, result.metadata.filename,
        "application/octet-stream", sha256=result.recovered_sha256,
    )
    signal, _ = ingest_audio(result.file_bytes, result.metadata.filename,
                             origin="upload", vaultRole="recovered")

    return {
        "fileId": artifact.id,
        "signalId": signal.id if signal else None,
        **result.metrics(),
    }


@app.get("/api/vault/file/{artifact_id}")
def vault_file(artifact_id: str):
    """Download the recovered file under its original name."""
    artifact = vault_store.get(artifact_id)
    if artifact is None or artifact.kind != "file":
        raise HTTPException(status_code=404, detail="File not found")
    return Response(
        content=artifact.data,
        media_type=artifact.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{artifact.filename}"',
            "Content-Length": str(len(artifact.data)),
        },
    )


class TamperRequest(BaseModel):
    imageId: str
    pixels: int = 12


@app.post("/api/vault/tamper")
def vault_tamper(req: TamperRequest):
    """Corrupt a few pixels, to demonstrate that corruption is detected.

    The edits land past the header so the container still *parses* — the
    point is that parsing succeeds and authentication then fails, which
    is a far more convincing demonstration than an image that is
    rejected as malformed before decryption is even attempted.
    """
    artifact = vault_store.get(req.imageId)
    if artifact is None or artifact.kind != "image":
        raise HTTPException(status_code=404, detail="Image not found")

    pixels = vault_stego.from_png(artifact.data)
    flat = pixels.reshape(-1).copy()
    count = int(np.clip(req.pixels, 1, 1000))

    start = VAULT_HEADER_SIZE * 8  # first channel after the header
    if start >= flat.size:
        raise HTTPException(status_code=400, detail="Image is too small to tamper with")
    rng = np.random.default_rng()
    targets = rng.choice(np.arange(start, flat.size), size=min(count, flat.size - start),
                         replace=False)
    flat[targets] ^= 1  # flip the low bit, the smallest possible edit

    tampered = flat.reshape(pixels.shape)
    new_artifact = vault_store.add(
        "image", vault_stego.to_png(tampered), f"tampered-{artifact.filename}", "image/png",
        bitsPerChannel=artifact.meta["bitsPerChannel"],
        coverHead=artifact.meta["coverHead"],
        stegoHead=head_bytes(tampered),
        width=artifact.meta["width"],
    )
    return {
        "imageId": new_artifact.id,
        "imageName": new_artifact.filename,
        "flippedBits": int(len(targets)),
        "totalChannels": int(flat.size),
        "fractionChanged": float(len(targets) / flat.size),
    }


@app.post("/api/vault/clear")
def vault_clear():
    vault_store.clear()
    return {"ok": True}


@app.get("/api/health")
def health():
    return {"status": "ok", "signals": len(store.list())}
