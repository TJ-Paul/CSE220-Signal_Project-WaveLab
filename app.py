"""Audio Signal Processing Toolkit — Streamlit UI.

Ties together the audio_toolkit DSP modules into one interactive app
that walks through the full pipeline:

    Input -> Sampling -> Time-Domain -> Framing/VAD -> FFT/STFT ->
    Filtering/Separation/Denoising -> Reconstruction -> Comparison

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy.signal import find_peaks

from audio_toolkit import (
    demo_signals,
    filters,
    io_utils,
    metrics,
    noise_reduction,
    sampling,
    separation,
    spectral,
    vad,
    visualization as viz,
)
from audio_toolkit.framing import frame_signal

st.set_page_config(page_title="Audio Signal Processing Toolkit", layout="wide", page_icon="🎧")

# --------------------------------------------------------------------------
# Theme — a restrained "signal-processing lab" visual system.
# Structural layout only; nothing below touches the DSP logic.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --navy:        #16232E;
        --navy-2:      #1F3B4D;
        --ink:         #1B2733;
        --muted:       #64748B;
        --bg:          #F4F7F9;
        --surface:     #FFFFFF;
        --border:      #DCE3E8;
        --accent:      #0E7C9B;
        --accent-tint: #E4F2F5;
        --amber:       #C9791B;
        --amber-tint:  #FBF0DF;
        --green:       #1E8F53;
        --green-tint:  #E4F5EB;
        --red:         #C93B3B;
    }

    html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    code, .mono, div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }

    .stApp { background-color: var(--bg); }
    .block-container { padding-top: 1.6rem; padding-bottom: 6.5rem; max-width: 1400px; }
    h1, h2, h3, .app-header-title { color: var(--navy); }
    p, li, span, label { color: var(--ink); }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background-color: var(--navy);
        border-right: 1px solid #0E1A22;
    }
    section[data-testid="stSidebar"] * { color: #D7E2EA; }
    .brand { display:flex; align-items:center; gap:10px; padding: 4px 2px 14px 2px; border-bottom: 1px solid rgba(255,255,255,0.09); margin-bottom: 10px; }
    .brand-icon { font-size: 1.5rem; }
    .brand-title { font-weight: 700; font-size: 1.02rem; color: #FFFFFF; line-height:1.15; }
    .brand-sub { font-size: 0.72rem; color: #8DA1B3; letter-spacing: 0.02em; }
    .nav-group-label {
        font-size: 0.68rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase;
        color: #64809A; margin: 14px 4px 4px 4px;
    }
    .nav-item {
        display:flex; align-items:center; gap:9px; padding: 8px 12px; border-radius: 7px;
        font-size: 0.87rem; font-weight: 600; margin-bottom: 2px;
        background: var(--accent); color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stButton { margin-bottom: 1px; }
    section[data-testid="stSidebar"] .stButton > button {
        background: transparent; border: none; color: #C3D2DE !important; font-weight: 500;
        font-size: 0.87rem; text-align: left; justify-content: flex-start; padding: 8px 12px;
        border-radius: 7px; width: 100%;
    }
    section[data-testid="stSidebar"] .stButton > button:hover { background: rgba(255,255,255,0.08); color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stButton > button p { color: inherit !important; text-align: left; }
    section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; margin: 10px 0; }

    .signal-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 9px; padding: 10px 12px; margin-top: 6px;
    }
    .signal-card-name { font-weight: 600; font-size: 0.82rem; color: #FFFFFF !important; word-break: break-word; }
    .signal-card-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #93A9BC !important; margin-top: 3px; }
    .signal-card-empty { font-size: 0.78rem; color: #6E859A !important; }

    /* ---------- Header ---------- */
    .st-key-app_header {
        background: linear-gradient(120deg, var(--navy) 0%, var(--navy-2) 100%);
        border-radius: 12px; padding: 14px 22px; margin-bottom: 12px;
    }
    .st-key-app_header p, .st-key-app_header span, .st-key-app_header div { color: #FFFFFF; }
    .app-header-title { font-size: 1.35rem; font-weight: 700; margin: 0; color: #FFFFFF !important; }
    .app-header-sub { font-size: 0.82rem; color: #A9BBC9 !important; margin-top: 2px; }
    .st-key-app_header .stButton > button,
    .st-key-app_header [data-testid="stPopoverButton"] {
        background: rgba(255,255,255,0.08) !important; color: #fff !important;
        border: 1px solid rgba(255,255,255,0.25) !important;
        border-radius: 7px !important; font-size: 0.78rem !important; padding: 4px 10px !important;
        float: right; margin-top: 2px;
    }
    .st-key-app_header [data-testid="stPopoverButton"]:hover { background: rgba(255,255,255,0.18) !important; }
    .st-key-app_header div[data-testid="stPopover"] { width: auto !important; display: flex; justify-content: flex-end; }
    .status-pill {
        display:inline-flex; align-items:center; gap:6px; padding: 5px 12px; border-radius: 20px;
        font-size: 0.76rem; font-weight: 600; float:right; margin-top: 2px;
    }
    .status-dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
    .status-neutral { background: rgba(255,255,255,0.1); color:#D7E2EA !important; }
    .status-neutral .status-dot { background:#8DA1B3; }
    .status-success { background: rgba(30,143,83,0.22); color:#8CE0B4 !important; }
    .status-success .status-dot { background:#3DDB8B; box-shadow:0 0 6px #3DDB8B; }

    /* ---------- Pipeline strip ---------- */
    .pipeline-row {
        display:flex; align-items:center; gap:4px; overflow-x:auto; padding: 10px 4px 14px 4px;
        margin-bottom: 6px;
    }
    .chip {
        flex: 0 0 auto; font-size: 0.72rem; font-weight: 600; color: var(--muted);
        background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
        padding: 5px 12px; white-space: nowrap;
    }
    .chip-active { background: var(--accent); border-color: var(--accent); color: #FFFFFF; }
    .chip-arrow { color: #B7C2CB; font-size: 0.8rem; flex: 0 0 auto; }

    /* ---------- Cards / metrics ---------- */
    div[data-testid="stMetric"] {
        background-color: var(--surface); border: 1px solid var(--border); border-radius: 8px;
        padding: 0.8rem 1rem;
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.78rem; }
    div[data-testid="stMetricValue"] { color: var(--navy); }

    .section-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); margin: 14px 0 6px 0; }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        border-radius: 7px; border: 1px solid var(--accent); color: var(--accent);
        background-color: var(--surface); font-weight: 600; transition: all 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { background-color: var(--accent-tint); }
    .stButton > button[kind="primary"] { background-color: var(--accent); color: #FFFFFF; }
    .stButton > button[kind="primary"]:hover { background-color: #0B6883; }

    /* ---------- Expanders / dataframe ---------- */
    details[data-testid="stExpander"] { border: 1px solid var(--border); border-radius: 8px; background-color: var(--surface); }
    div[data-testid="stAlert"] { border-radius: 8px; }

    /* ---------- Loaded-signal card (Audio Input page) ---------- */
    .loaded-badge { display:inline-block; font-size: 0.72rem; font-weight:700; letter-spacing:0.05em; color: var(--green); background: var(--green-tint); padding: 3px 10px; border-radius: 12px; }
    .loaded-name { font-size: 1.05rem; font-weight: 700; color: var(--navy); margin: 8px 0 10px 0; word-break: break-word; }

    /* ---------- Checklist ---------- */
    .checklist-item { font-size: 0.86rem; padding: 4px 0; }
    .check-done { color: var(--green); font-weight: 600; }
    .check-pending { color: var(--muted); }

    /* ---------- Legend chips (VAD) ---------- */
    .legend-chip { font-size: 0.78rem; font-weight: 600; padding: 3px 4px; }
    .legend-speech { color: var(--green); }
    .legend-silence { color: var(--muted); }

    /* ---------- Empty state ---------- */
    .empty-state { text-align:center; padding: 40px 20px 18px 20px; }
    .empty-state-icon { font-size: 1.8rem; color: var(--accent); letter-spacing: 3px; opacity: 0.6; }
    .empty-state-title { font-size: 1.1rem; font-weight: 700; color: var(--navy); margin-top: 10px; }
    .empty-state-sub { font-size: 0.85rem; color: var(--muted); margin-top: 4px; max-width: 420px; margin-left:auto; margin-right:auto; }
    .empty-state-tags { text-align:center; font-size: 0.74rem; color: var(--muted); margin-top: 18px; letter-spacing: 0.01em; }

    /* ---------- Persistent mini player ---------- */
    .st-key-mini_player {
        position: fixed; left: 0; right: 0; bottom: 0; z-index: 999;
        background: var(--navy); padding: 10px 26px 6px 26px; box-shadow: 0 -3px 14px rgba(0,0,0,0.18);
        border-top: 1px solid rgba(255,255,255,0.08);
    }
    .st-key-mini_player * { color: #E4ECF1; }
    .mini-player-name { font-size: 0.82rem; font-weight: 600; margin-bottom: 2px; }
    .mini-player-meta { font-family:'JetBrains Mono',monospace; font-size: 0.75rem; color: #93A9BC; text-align:right; padding-top: 10px; }
    .st-key-mini_player audio { height: 32px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def to_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(y, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def audio_player(y: np.ndarray, sr: int):
    st.audio(to_wav_bytes(y, sr), format="audio/wav")


def download_button(y: np.ndarray, sr: int, label: str, filename: str, key: str):
    st.download_button(label, data=to_wav_bytes(y, sr), file_name=filename, mime="audio/wav", key=key)


def set_current_audio(y: np.ndarray, sr: int, name: str):
    st.session_state.audio_y = y.astype(np.float32)
    st.session_state.audio_sr = sr
    st.session_state.audio_name = name
    st.session_state.results = {"Original": (y.astype(np.float32), sr)}
    st.session_state.stage_done = {
        "sampling": True, "framed": False, "freq": False, "processed": False, "reconstructed": False,
    }
    # Drop cached results tied to whatever audio was loaded previously —
    # they have their own sample count and no longer match this signal.
    for key in ("_nr_result", "_sep_result", "_filt_result", "song_truth", "noise_truth", "_last_upload_size"):
        st.session_state.pop(key, None)


def store_result(name: str, y: np.ndarray, sr: int):
    st.session_state.setdefault("results", {})[name] = (y.astype(np.float32), sr)
    st.session_state.setdefault("stage_done", {}).update({"processed": True, "reconstructed": True})


def has_audio() -> bool:
    return st.session_state.get("audio_y") is not None


def load_demo(kind: str):
    if kind == "speech":
        y, sr = demo_signals.generate_speech_like_demo()
        set_current_audio(y, sr, "synthetic: speech-like demo")
    elif kind == "song":
        mix, fg, bg, sr = demo_signals.generate_song_like_demo()
        set_current_audio(mix, sr, "synthetic: song-like demo")
        st.session_state["song_truth"] = (fg, bg, sr)
    elif kind == "noisy":
        clean, noisy, sr = demo_signals.generate_noisy_tone_demo()
        set_current_audio(noisy, sr, "synthetic: noisy tone demo")
        st.session_state["noise_truth"] = (clean, sr)
    st.session_state.pop("_last_upload_info", None)
    st.session_state["_show_uploader"] = False


def empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-state-icon">∿ ∿ ∿</div>
            <div class="empty-state-title">No signal loaded</div>
            <div class="empty-state-sub">Upload an audio file or select a demonstration signal to begin analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    e1, e2, e3 = st.columns(3)
    with e1:
        if st.button("Go to Audio Input", width="stretch", key=f"empty_goto_{st.session_state.nav}"):
            st.session_state.nav = "input"
            st.rerun()
    with e2:
        if st.button("Try speech demo", width="stretch", key=f"empty_speech_{st.session_state.nav}"):
            load_demo("speech")
            st.rerun()
    with e3:
        if st.button("Try song demo", width="stretch", key=f"empty_song_{st.session_state.nav}"):
            load_demo("song")
            st.rerun()
    st.markdown(
        '<div class="empty-state-tags">Supported analysis: Waveform · FFT · STFT · Spectrogram · '
        'Filtering · VAD · Noise Reduction · Signal Separation</div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Session defaults
# --------------------------------------------------------------------------
st.session_state.setdefault("nav", "input")
st.session_state.setdefault(
    "stage_done", {"sampling": False, "framed": False, "freq": False, "processed": False, "reconstructed": False}
)

NAV_GROUPS = [
    ("INPUT", [("input", "Audio Input", "🎙️")]),
    ("ANALYSIS", [
        ("file_info", "File Information", "📄"),
        ("waveform", "Waveform", "〰️"),
        ("sampling", "Sampling & Aliasing", "📶"),
        ("vad", "Voice Activity Detection", "🗣️"),
        ("fft", "FFT Spectrum", "📊"),
        ("spectrogram", "Spectrogram", "🌈"),
    ]),
    ("PROCESSING", [
        ("filtering", "Filtering", "🎚️"),
        ("separation", "Vocal / Instrumental Separation", "🎼"),
        ("noise", "Noise Reduction", "🧹"),
    ]),
    ("COMPARISON", [("comparison", "Signal Comparison", "⚖️")]),
]
PAGE_TITLES = {key: label for _, items in NAV_GROUPS for key, label, _ in items}
PAGE_STAGE = {
    "input": "Input", "file_info": "Input", "waveform": "Time Domain", "sampling": "Sampling",
    "vad": "Energy", "fft": "Frequency Domain", "spectrogram": "FFT / STFT",
    "filtering": "Filtering", "separation": "Filtering", "noise": "Filtering", "comparison": "Output",
}
STAGES = ["Input", "Sampling", "Time Domain", "Framing", "Energy", "FFT / STFT",
          "Frequency Domain", "Filtering", "Reconstruction", "Output"]

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-icon">〰️</div>'
        '<div><div class="brand-title">Audio Signal Lab</div>'
        '<div class="brand-sub">Digital Signal Processing Toolkit</div></div></div>',
        unsafe_allow_html=True,
    )

    for group_name, items in NAV_GROUPS:
        st.markdown(f'<div class="nav-group-label">{group_name}</div>', unsafe_allow_html=True)
        for key, label, icon in items:
            if st.session_state.nav == key:
                st.markdown(f'<div class="nav-item">{icon} {label}</div>', unsafe_allow_html=True)
            else:
                if st.button(f"{icon}  {label}", key=f"nav_{key}", width="stretch"):
                    st.session_state.nav = key
                    st.rerun()

    st.divider()
    st.markdown('<div class="nav-group-label">Current Signal</div>', unsafe_allow_html=True)
    if has_audio():
        y0, sr0 = st.session_state.audio_y, st.session_state.audio_sr
        if st.session_state.audio_name.startswith("synthetic:"):
            fmt = "Synthetic"
        else:
            fmt = os.path.splitext(st.session_state.audio_name)[1].lstrip(".").upper() or "Audio"
        st.markdown(
            f'<div class="signal-card">'
            f'<div class="signal-card-name">{st.session_state.audio_name}</div>'
            f'<div class="signal-card-meta">{fmt} • {sr0/1000:.1f} kHz • {len(y0)/sr0:.1f}s</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="signal-card"><span class="signal-card-empty">No signal loaded</span></div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
status_label = "Signal Loaded" if has_audio() else "Ready"
status_class = "status-success" if has_audio() else "status-neutral"

with st.container(key="app_header"):
    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.markdown('<div class="app-header-title">Audio Signal Processing Toolkit</div>', unsafe_allow_html=True)
        st.markdown('<div class="app-header-sub">Analyze, transform, visualize, and compare audio signals.</div>', unsafe_allow_html=True)
    with hcol2:
        st.markdown(f'<div class="status-pill {status_class}"><span class="status-dot"></span>{status_label}</div>', unsafe_allow_html=True)
        with st.popover("Help"):
            st.markdown(
                "**About this tool**\n\n"
                "Walks an audio signal through the full DSP pipeline — input, sampling, "
                "time/frequency-domain analysis, filtering, separation, noise reduction, and "
                "reconstruction — so each stage's effect can be inspected numerically and visually.\n\n"
                "Use the sidebar to move between stages; the currently loaded signal stays available everywhere."
            )

# --------------------------------------------------------------------------
# Pipeline strip
# --------------------------------------------------------------------------
active_stage = PAGE_STAGE.get(st.session_state.nav)
chip_html = '<div class="pipeline-row">'
for i, s in enumerate(STAGES):
    if i:
        chip_html += '<span class="chip-arrow">→</span>'
    cls = "chip chip-active" if s == active_stage else "chip"
    chip_html += f'<span class="{cls}">{s.upper()}</span>'
chip_html += "</div>"
st.markdown(chip_html, unsafe_allow_html=True)

page = st.session_state.nav

# --------------------------------------------------------------------------
# Page — Audio Input
# --------------------------------------------------------------------------
if page == "input":
    st.subheader("Audio Input")

    if not has_audio() or st.session_state.get("_show_uploader"):
        st.markdown('<div class="section-label">Upload an audio file</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"], label_visibility="collapsed")
        if uploaded is not None and st.session_state.get("_last_upload_id") != uploaded.file_id:
            try:
                y, sr = io_utils.load_audio(uploaded, sr=None, mono=True)
                set_current_audio(y, sr, uploaded.name)
                st.session_state["_last_upload_info"] = io_utils.get_audio_info(
                    uploaded, filename=uploaded.name, file_size_bytes=uploaded.size
                )
                st.session_state["_last_upload_size"] = uploaded.size
                st.session_state["_last_upload_id"] = uploaded.file_id
                st.session_state["_show_uploader"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Could not load file: {e}")

        st.markdown('<div class="section-label">Or try a demo signal</div>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            if st.button("🗣️ Speech-like", width="stretch", key="demo_speech"):
                load_demo("speech")
                st.rerun()
        with d2:
            if st.button("🎵 Song-like", width="stretch", key="demo_song"):
                load_demo("song")
                st.rerun()
        with d3:
            if st.button("📡 Noisy tone", width="stretch", key="demo_noisy"):
                load_demo("noisy")
                st.rerun()

    if has_audio():
        y0, sr0 = st.session_state.audio_y, st.session_state.audio_sr
        info = st.session_state.get("_last_upload_info")
        duration = len(y0) / sr0
        mins, secs = divmod(duration, 60)

        with st.container(border=True):
            st.markdown('<div class="loaded-badge">✓ AUDIO LOADED</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="loaded-name">{st.session_state.audio_name}</div>', unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Format", info.subtype.split("_")[0] if info else "Synthetic")
            m2.metric("Sample rate", f"{sr0/1000:.1f} kHz")
            m3.metric("Channels (source)", info.channels if info else 1)
            m4.metric("Duration", f"{int(mins):02d}:{secs:05.2f}")
            audio_player(y0, sr0)
            if not st.session_state.get("_show_uploader") and st.button("Replace audio", key="replace_audio_btn"):
                st.session_state["_show_uploader"] = True
                st.rerun()

# --------------------------------------------------------------------------
# Page — File Information
# --------------------------------------------------------------------------
elif page == "file_info":
    st.subheader("Signal Overview")
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        info = st.session_state.get("_last_upload_info")
        upload_size = st.session_state.get("_last_upload_size")
        duration = len(y) / sr
        mins, secs = divmod(duration, 60)

        r1 = st.columns(4)
        r1[0].metric("Duration", f"{int(mins):02d}:{secs:05.2f}")
        r1[1].metric("Sample rate", f"{sr:,} Hz")
        r1[2].metric("Channels (source)", info.channels if info else 1)
        r1[3].metric("Bit depth", f"{info.bit_depth}-bit" if info and info.bit_depth else "N/A (compressed)")

        r2 = st.columns(4)
        r2[0].metric("Samples", f"{len(y):,}")
        r2[1].metric("Format", info.subtype if info else "Synthetic (float32)")
        r2[2].metric("Est. bitrate", f"{info.bitrate_kbps:.0f} kbps" if info else "—")
        r2[3].metric("File size", f"{upload_size/1e6:.1f} MB" if upload_size else "—")

        st.caption(
            "Note: processing throughout this app is done on a mono, floating-point "
            "([-1, 1]) version of the signal — standard practice for DSP analysis; "
            "channel count/bit depth above describe the original file."
        )

        st.markdown('<div class="section-label">Waveform preview</div>', unsafe_allow_html=True)
        st.pyplot(viz.plot_waveform(y, sr, title="Waveform preview"))
        audio_player(y, sr)

        st.markdown('<div class="section-label">Signal pipeline status</div>', unsafe_allow_html=True)
        sd = st.session_state.get("stage_done", {})
        checklist = [
            ("Audio loaded", True),
            ("Sampling information extracted", sd.get("sampling", False)),
            ("Signal framed (visit the VAD page)", sd.get("framed", False)),
            ("Frequency analysis run (FFT or Spectrogram page)", sd.get("freq", False)),
            ("Filtering / processing applied", sd.get("processed", False)),
            ("Signal reconstructed", sd.get("reconstructed", False)),
        ]
        for label, done in checklist:
            mark = "✓" if done else "○"
            cls = "check-done" if done else "check-pending"
            st.markdown(f'<div class="checklist-item {cls}">{mark} {label}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Page — Waveform
# --------------------------------------------------------------------------
elif page == "waveform":
    st.subheader("Waveform")
    st.caption("Time-domain representation")
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        duration = len(y) / sr
        zc1, zc2 = st.columns([3, 1])
        with zc1:
            zoom = st.slider("Zoom / time range (s)", 0.0, float(duration), (0.0, float(duration)), key="wave_zoom")
        with zc2:
            st.caption("Playback")
            audio_player(y, sr)

        start_i, end_i = int(zoom[0] * sr), int(zoom[1] * sr)
        seg = y[start_i:end_i] if end_i > start_i else y
        fig = viz.plot_waveform(y, sr, title=f"Waveform — {st.session_state.audio_name}", xlim=zoom, figsize=(11, 3.6))
        st.pyplot(fig)

        peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
        rms_val = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))) if len(seg) else 0.0
        peak_db = 20 * np.log10(peak + 1e-12)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak amplitude", f"{peak:.3f}")
        m2.metric("RMS level", f"{rms_val:.3f}")
        m3.metric("Peak level", f"{peak_db:.1f} dBFS")
        m4.metric("Zoomed duration", f"{zoom[1]-zoom[0]:.2f} s")

# --------------------------------------------------------------------------
# Page — Sampling & Aliasing
# --------------------------------------------------------------------------
elif page == "sampling":
    st.subheader("Sampling, Aliasing & Reconstruction")
    st.markdown(
        "The **Nyquist–Shannon theorem** says a signal band-limited to $f_{max}$ can be "
        "reconstructed perfectly only if sampled at $f_s \\ge 2f_{max}$. Sample below that "
        "rate and the true frequency **aliases** — appears, after reconstruction, as a lower "
        "'ghost' frequency that cannot be told apart from the real thing."
    )
    colA, colB, colC = st.columns(3)
    tone_freq = colA.slider("Tone frequency (Hz)", 20, 4000, 500, key="samp_freq")
    fs_demo = colB.slider("Sampling rate — Fs (Hz)", 100, 8000, 800, step=50, key="samp_fs")
    dur_demo = colC.slider("Duration (s)", 0.02, 0.2, 0.05, key="samp_dur")

    nyquist = fs_demo / 2
    t_hi, x_hi = sampling.generate_tone(tone_freq, dur_demo, fs_reference=44100)
    t_samples, x_samples = sampling.sample_signal(tone_freq, dur_demo, fs_demo)
    t_recon = t_hi
    x_recon = sampling.reconstruct_signal(t_samples, x_samples, t_recon)

    sc1, sc2 = st.columns(2)
    sc1.metric("Sampling rate — Fs", f"{fs_demo:,} Hz")
    sc2.metric("Nyquist frequency", f"{nyquist:,.0f} Hz")

    if tone_freq > nyquist:
        alias_f = sampling.aliased_frequency(tone_freq, fs_demo)
        st.warning(
            f"⚠ **Fs < 2·Fmax — aliasing likely.** Nyquist frequency is {nyquist:.0f} Hz but the tone is "
            f"{tone_freq} Hz. The reconstructed signal will look like a **{alias_f:.1f} Hz** alias, "
            f"not the true {tone_freq} Hz tone."
        )
    else:
        st.success(f"✓ **Fs > 2·Fmax — no aliasing expected.** Sampled above Nyquist ({nyquist:.0f} Hz); reconstruction should match the original closely.")

    fig = viz.plot_sampling_demo(t_hi, x_hi, t_samples, x_samples, t_recon, x_recon)
    st.pyplot(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("MSE (orig vs reconstructed)", f"{metrics.mse(x_hi, x_recon):.4f}")
    m2.metric("Correlation", f"{metrics.correlation(x_hi, x_recon):.3f}")
    m3.metric("SNR (dB)", f"{metrics.snr_db(x_hi, x_recon):.1f}")

    st.divider()
    st.markdown("**Apply the same idea to loaded audio** — downsample then upsample back:")
    if has_audio():
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        target_sr = st.select_slider("Downsample to (Hz)", options=[2000, 4000, 8000, 11025, 16000, 22050, sr], value=8000, key="audio_ds_sr")
        if st.button("Run downsample → upsample demo", key="run_resample_demo", type="primary"):
            down = sampling.resample_audio(y, sr, target_sr)
            back = sampling.resample_audio(down, target_sr, sr)
            store_result(f"Resampled {sr}→{target_sr}→{sr} Hz", back, sr)
            c1, c2 = st.columns(2)
            with c1:
                st.write("Original")
                audio_player(y, sr)
            with c2:
                st.write(f"Round-tripped through {target_sr} Hz")
                audio_player(back, sr)
            st.pyplot(viz.plot_before_after(y, sr, "Original", back, sr, f"Down/upsampled via {target_sr} Hz"))
            st.write(f"MSE: {metrics.mse(y, back):.6f} | SNR: {metrics.snr_db(y, back):.1f} dB | Correlation: {metrics.correlation(y, back):.3f}")
            st.caption("Saved as a result — compare it precisely on the Signal Comparison page.")
    else:
        st.info("Load audio from the Audio Input page to try this on a real signal.")

# --------------------------------------------------------------------------
# Page — VAD
# --------------------------------------------------------------------------
elif page == "vad":
    st.subheader("Voice Activity Detection")
    st.caption("Frames are classified using short-time signal energy and a speech-band frequency ratio.")
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2, c3 = st.columns(3)
        frame_ms = c1.slider("Frame size (ms)", 10, 40, 25, key="vad_frame_ms")
        hop_ms = c2.slider("Hop size (ms)", 5, 30, 10, key="vad_hop_ms")
        energy_pct = c3.slider("Energy threshold (percentile)", 10, 90, 40, key="vad_energy_pct")
        band_ratio_thresh = st.slider("Speech-band ratio threshold", 0.0, 1.0, 0.35, key="vad_band_thresh")

        result = vad.detect_speech(
            y, sr, frame_ms=frame_ms, hop_ms=hop_ms,
            energy_percentile=energy_pct, band_ratio_thresh=band_ratio_thresh,
        )
        st.session_state.stage_done["framed"] = True
        st.pyplot(viz.plot_vad(y, sr, result))

        speech_time = sum(e - s for s, e in result.speech_segments)
        total = len(y) / sr
        silence_time = total - speech_time
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Speech duration", f"{speech_time:.2f} s")
        m2.metric("Silence duration", f"{silence_time:.2f} s")
        m3.metric("Speech ratio", f"{(speech_time/total*100 if total else 0):.0f}%")
        m4.metric("Speech segments", len(result.speech_segments))

        lg1, lg2, _ = st.columns([1, 1, 4])
        lg1.markdown('<span class="legend-chip legend-speech">■ Speech</span>', unsafe_allow_html=True)
        lg2.markdown('<span class="legend-chip legend-silence">■ Silence</span>', unsafe_allow_html=True)

        with st.expander("Detected segments (s)"):
            st.write([(round(s, 2), round(e, 2)) for s, e in result.speech_segments])

        st.caption(
            "A frame is classified Speech only if BOTH its short-time energy exceeds an "
            "adaptive percentile threshold AND its FFT-derived speech-band (300–3400 Hz) "
            "energy ratio exceeds its threshold — combining a time-domain and a "
            "frequency-domain feature rejects loud broadband noise that energy alone would "
            "misclassify as speech."
        )

# --------------------------------------------------------------------------
# Page — FFT Spectrum
# --------------------------------------------------------------------------
elif page == "fft":
    st.subheader("FFT Spectrum")
    st.caption("Frequency-domain representation of the signal.")
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        duration = len(y) / sr
        mode = st.radio("Analyze:", ["Whole signal", "A single frame (20-30 ms)"], horizontal=True, key="fft_mode")

        if mode == "Whole signal":
            freqs, mag = spectral.compute_fft(y, sr)
            st.pyplot(viz.plot_time_freq_pair(y, sr))
        else:
            frame_ms = st.slider("Frame length (ms)", 10, 40, 25, key="fft_frame_ms")
            center_s = st.slider("Frame center (s)", 0.0, float(max(duration - 0.001, 0)), min(0.5, duration / 2), key="fft_frame_center")
            frame_len = int(sr * frame_ms / 1000)
            start = max(0, int(center_s * sr) - frame_len // 2)
            frame = y[start:start + frame_len]
            windowed = frame * np.hamming(len(frame))
            st.pyplot(viz.plot_time_freq_pair(windowed, sr, title_prefix=f"{frame_ms} ms frame @ {center_s:.2f}s — "))
            freqs, mag = spectral.compute_fft(windowed, sr)

        st.session_state.stage_done["freq"] = True

        ctrl1, ctrl2 = st.columns(2)
        log_x = ctrl1.checkbox("Log frequency axis", value=False, key="fft_log_x")
        n_show = ctrl2.slider("Peaks to list", 3, 10, 5, key="fft_npeaks")

        spec_col, table_col = st.columns([2, 1])
        with spec_col:
            st.pyplot(viz.plot_fft_spectrum(freqs, mag, log_x=log_x, figsize=(8, 3.6)))
        with table_col:
            st.markdown('<div class="section-label">Dominant frequencies</div>', unsafe_allow_html=True)
            peak_idx, _ = find_peaks(mag, distance=max(1, len(mag) // 200))
            if len(peak_idx) == 0:
                peak_idx = np.array([int(np.argmax(mag))])
            order = peak_idx[np.argsort(mag[peak_idx])[::-1]][:n_show]
            table = pd.DataFrame({
                "Frequency (Hz)": [f"{freqs[i]:.1f}" for i in order],
                "Magnitude": [f"{mag[i]:.3f}" for i in order],
            })
            st.dataframe(table, hide_index=True, width="stretch")

        dominant = freqs[np.argmax(mag)]
        spectral_energy = float(np.sum(mag.astype(np.float64) ** 2))
        m1, m2 = st.columns(2)
        m1.metric("Peak frequency", f"{dominant:.1f} Hz")
        m2.metric("Spectral energy", f"{spectral_energy:.3g}")

# --------------------------------------------------------------------------
# Page — Spectrogram
# --------------------------------------------------------------------------
elif page == "spectrogram":
    st.subheader("Spectrogram")
    st.caption("Short-Time Fourier Transform — how frequency content evolves over time.")
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2 = st.columns(2)
        n_fft = c1.select_slider("FFT size (frequency resolution)", options=[256, 512, 1024, 2048, 4096], value=2048, key="spec_nfft")
        hop = c2.select_slider("Hop length (time resolution)", options=[64, 128, 256, 512, 1024], value=512, key="spec_hop")
        freqs, times, mag_db = spectral.spectrogram_db(y, sr, n_fft=n_fft, hop_length=hop)
        st.session_state.stage_done["freq"] = True
        st.pyplot(viz.plot_spectrogram(freqs, times, mag_db, figsize=(11, 4.2)))

        r1, r2 = st.columns(2)
        r1.metric("Frequency resolution (Δf)", f"{sr/n_fft:.1f} Hz")
        r2.metric("Time resolution", f"{hop/sr*1000:.1f} ms")
        st.caption(
            "Larger n_fft sharpens frequency detail but blurs timing (and vice-versa) — "
            "the STFT's fundamental time/frequency trade-off. Color scale (right) shows magnitude in dB."
        )

# --------------------------------------------------------------------------
# Page — Filtering
# --------------------------------------------------------------------------
elif page == "filtering":
    st.subheader("Filtering")
    st.caption("Design and apply a Butterworth filter, then inspect its effect.")
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        nyquist = sr / 2
        c1, c2 = st.columns(2)
        kind = c1.selectbox("Filter type", ["lowpass", "highpass", "bandpass", "bandstop"], key="filt_kind")
        order = c2.slider("Filter order", 2, 12, 6, key="filt_order")

        if kind in ("lowpass", "highpass"):
            cutoff = st.slider("Cutoff frequency (Hz)", 20, int(nyquist) - 20, min(1000, int(nyquist) - 20), key="filt_cutoff")
            band_label = f"Cutoff: {cutoff} Hz"
        else:
            low, high = st.slider("Cutoff band (Hz)", 20, int(nyquist) - 20, (300, 3400), key="filt_band")
            cutoff = (low, high)
            band_label = f"Band: {low}–{high} Hz"

        sos = filters.design_filter(kind, cutoff, sr, order=order)
        freqs_resp, mag_db = filters.frequency_response(sos, sr)
        cutoffs_plot = [cutoff] if isinstance(cutoff, (int, float)) else list(cutoff)
        st.pyplot(viz.plot_filter_response(freqs_resp, mag_db, cutoffs=cutoffs_plot, title=f"{kind.title()} Filter — Magnitude Response (order {order})"))
        st.caption(f"{kind.title()} filter · {band_label} · order {order}")

        if st.button("Apply filter", key="apply_filter_btn", type="primary"):
            with st.spinner("Applying filter..."):
                filtered = filters.apply_filter(y, sos)
            store_result(f"Filtered ({kind}, {cutoff})", filtered, sr)
            st.session_state["_filt_result"] = (filtered, sr)

        if "_filt_result" in st.session_state:
            filtered, f_sr = st.session_state["_filt_result"]
            st.success("✓ Filter applied successfully")
            fc1, fc2 = st.columns(2)
            with fc1:
                st.write("Before")
                audio_player(y, sr)
            with fc2:
                st.write("After")
                audio_player(filtered, f_sr)
            st.pyplot(viz.plot_before_after(y, sr, "Before (original)", filtered, f_sr, "After (filtered)", title="Filtering — Before / After"))
            download_button(filtered, f_sr, "Download filtered audio (WAV)", "filtered.wav", "dl_filtered")
            st.caption("Saved as a result — compare it precisely on the Signal Comparison page.")

# --------------------------------------------------------------------------
# Page — Separation
# --------------------------------------------------------------------------
elif page == "separation":
    st.subheader("Vocal / Instrumental Separation")
    st.caption("Separate an audio mixture into vocal and accompaniment components.")
    with st.expander("Method & limitations (read this)", expanded=False):
        st.markdown(separation.__doc__.replace("METHOD\n------", "**Method**").replace(
            "LIMITATIONS (important — read before trusting the output)\n-----------------------------------------------------------",
            "**Limitations**"
        ))
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2 = st.columns(2)
        margin_bg = c1.slider("Background margin (aggressiveness)", 1.0, 20.0, 2.0, key="sep_margin_bg")
        margin_fg = c2.slider("Foreground margin (aggressiveness)", 1.0, 20.0, 10.0, key="sep_margin_fg")

        st.markdown('<div class="section-label">Original mixture</div>', unsafe_allow_html=True)
        st.pyplot(viz.plot_waveform(y, sr, title="Original — mixture"))
        audio_player(y, sr)

        if st.button("Run separation", key="run_sep_btn", type="primary"):
            with st.spinner("Decomposing signal into vocal and instrumental components..."):
                fg, bg = separation.separate_vocals_instrumental(y, sr, margin_background=margin_bg, margin_foreground=margin_fg)
            store_result("Separated: foreground (vocal-like)", fg, sr)
            store_result("Separated: background (instrumental-like)", bg, sr)
            st.session_state["_sep_result"] = (fg, bg, sr)

        if "_sep_result" in st.session_state:
            fg, bg, sep_sr = st.session_state["_sep_result"]
            st.success("✓ Separation complete")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Vocals** (foreground)")
                st.pyplot(viz.plot_waveform(fg, sep_sr, title="Foreground — vocal-like"))
                audio_player(fg, sep_sr)
                download_button(fg, sep_sr, "Download vocals", "foreground.wav", "dl_fg")
            with c2:
                st.markdown("**Instrumental** (background)")
                st.pyplot(viz.plot_waveform(bg, sep_sr, title="Background — instrumental-like"))
                audio_player(bg, sep_sr)
                download_button(bg, sep_sr, "Download instrumental", "background.wav", "dl_bg")

            if "song_truth" in st.session_state:
                true_fg, true_bg, _ = st.session_state["song_truth"]
                st.markdown("**Ground-truth check** (only available for the synthetic song demo):")
                m1, m2 = st.columns(2)
                m1.metric("Foreground correlation vs. true melody", f"{metrics.correlation(true_fg, fg):.3f}")
                m2.metric("Background correlation vs. true chord loop", f"{metrics.correlation(true_bg, bg):.3f}")
            st.caption("Saved as results — compare precisely on the Signal Comparison page.")

# --------------------------------------------------------------------------
# Page — Noise reduction
# --------------------------------------------------------------------------
elif page == "noise":
    st.subheader("Noise Reduction")
    st.caption("Spectral subtraction — estimate the noise floor from a quiet segment, subtract it from every frame.")
    with st.expander("Method & limitations (read this)", expanded=False):
        st.markdown(noise_reduction.__doc__.replace("METHOD\n------", "**Method**").replace(
            "LIMITATIONS\n-----------", "**Limitations**"
        ))
    if not has_audio():
        empty_state()
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2, c3 = st.columns(3)
        noise_dur = c1.slider("Noise sample duration (s, from start)", 0.1, min(2.0, len(y) / sr), 0.5, key="nr_noise_dur")
        alpha = c2.slider("Noise reduction strength — α", 0.5, 5.0, 2.0, key="nr_alpha")
        beta = c3.slider("Spectral floor — β", 0.0, 0.3, 0.05, key="nr_beta")
        st.caption("Noise profile: auto-detected from the first noise-sample-duration seconds of the signal.")

        if st.button("Apply noise reduction", key="run_nr_btn", type="primary"):
            with st.spinner("Estimating noise profile and subtracting..."):
                denoised = noise_reduction.spectral_subtraction(y, sr, noise_duration_s=noise_dur, alpha=alpha, beta=beta)
            store_result("Denoised (spectral subtraction)", denoised, sr)
            st.session_state["_nr_result"] = denoised

        if "_nr_result" in st.session_state:
            denoised = st.session_state["_nr_result"]
            st.success("✓ Noise reduction applied")
            left, right = st.columns(2)
            with left:
                st.markdown("**Noisy signal**")
                audio_player(y, sr)
                st.pyplot(viz.plot_waveform(y, sr, title="Noisy — waveform"))
            with right:
                st.markdown("**Cleaned signal**")
                audio_player(denoised, sr)
                st.pyplot(viz.plot_waveform(denoised, sr, title="Denoised — waveform"))

            freqs, times, mag_db_noisy = spectral.spectrogram_db(y, sr)
            _, _, mag_db_clean = spectral.spectrogram_db(denoised, sr)
            sc1, sc2 = st.columns(2)
            sc1.pyplot(viz.plot_spectrogram(freqs, times, mag_db_noisy, title="Noisy — Spectrogram"))
            sc2.pyplot(viz.plot_spectrogram(freqs, times, mag_db_clean, title="Denoised — Spectrogram"))

            if "noise_truth" in st.session_state:
                clean_ref, _ = st.session_state["noise_truth"]
                st.markdown("**Estimated SNR** (ground truth available for the synthetic noisy-tone demo):")
                m1, m2, m3 = st.columns(3)
                m1.metric("SNR before", f"{metrics.snr_db(clean_ref, y):.1f} dB")
                m2.metric("SNR after", f"{metrics.snr_db(clean_ref, denoised):.1f} dB")
                m3.metric("Correlation after", f"{metrics.correlation(clean_ref, denoised):.3f}")
            download_button(denoised, sr, "Download denoised audio (WAV)", "denoised.wav", "dl_denoised")
            st.caption("Saved as a result — compare it precisely on the Signal Comparison page.")

# --------------------------------------------------------------------------
# Page — Signal comparison
# --------------------------------------------------------------------------
elif page == "comparison":
    st.subheader("Signal Comparison")
    st.caption("Quantitatively compare any two stored signals from earlier stages.")
    results = st.session_state.get("results", {})
    if len(results) < 1:
        empty_state()
    else:
        names = list(results.keys())
        c1, c2 = st.columns(2)
        name_a = c1.selectbox("Signal A (reference)", names, index=0, key="cmp_a")
        name_b = c2.selectbox("Signal B (test)", names, index=min(1, len(names) - 1), key="cmp_b")
        y_a, sr_a = results[name_a]
        y_b, sr_b = results[name_b]

        if sr_a != sr_b:
            y_b = sampling.resample_audio(y_b, sr_b, sr_a)
            sr_b = sr_a
            st.info(f"Signal B was resampled to {sr_a} Hz to match Signal A for comparison.")

        n = min(len(y_a), len(y_b))
        rms_a = float(np.sqrt(np.mean(y_a[:n].astype(np.float64) ** 2))) if n else 0.0
        rms_b = float(np.sqrt(np.mean(y_b[:n].astype(np.float64) ** 2))) if n else 0.0
        peak_a = float(np.max(np.abs(y_a[:n]))) if n else 0.0
        peak_b = float(np.max(np.abs(y_b[:n]))) if n else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("MSE", f"{metrics.mse(y_a, y_b):.6f}")
        m2.metric("SNR (dB)", f"{metrics.snr_db(y_a, y_b):.1f}")
        m3.metric("Correlation", f"{metrics.correlation(y_a, y_b):.3f}")
        m4, m5 = st.columns(2)
        m4.metric("RMS difference", f"{abs(rms_a - rms_b):.4f}")
        m5.metric("Peak difference", f"{abs(peak_a - peak_b):.4f}")

        st.markdown('<div class="section-label">Waveform comparison</div>', unsafe_allow_html=True)
        st.pyplot(viz.plot_before_after(y_a, sr_a, f"A: {name_a}", y_b, sr_b, f"B: {name_b}"))

        st.markdown('<div class="section-label">Frequency spectrum comparison</div>', unsafe_allow_html=True)
        freqs_a, mag_a = spectral.compute_fft(y_a, sr_a)
        freqs_b, mag_b = spectral.compute_fft(y_b, sr_b)
        fc1, fc2 = st.columns(2)
        fc1.pyplot(viz.plot_fft_spectrum(freqs_a, mag_a, title=f"A: {name_a} — Spectrum"))
        fc2.pyplot(viz.plot_fft_spectrum(freqs_b, mag_b, title=f"B: {name_b} — Spectrum"))

        c1, c2 = st.columns(2)
        with c1:
            audio_player(y_a, sr_a)
        with c2:
            audio_player(y_b, sr_b)

        st.caption(
            "MSE: lower = more similar. SNR: treats A as ground truth and B's deviation from it "
            "as 'noise' — higher dB = closer match. Correlation: shape similarity independent of "
            "amplitude, in [-1, 1]. RMS/Peak difference: absolute change in loudness/peak level."
        )

# --------------------------------------------------------------------------
# Persistent mini audio player
# --------------------------------------------------------------------------
if has_audio():
    y0, sr0 = st.session_state.audio_y, st.session_state.audio_sr
    with st.container(key="mini_player"):
        mp1, mp2 = st.columns([5, 2])
        with mp1:
            st.markdown(f'<div class="mini-player-name">🎧 {st.session_state.audio_name}</div>', unsafe_allow_html=True)
            audio_player(y0, sr0)
        with mp2:
            st.markdown(f'<div class="mini-player-meta">{sr0/1000:.1f} kHz · {len(y0)/sr0:.1f} s</div>', unsafe_allow_html=True)
