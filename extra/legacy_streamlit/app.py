"""Audio Signal Processing Toolkit — Streamlit UI.

Ties together the audio_toolkit DSP modules into one interactive app
that walks through the full pipeline:

    Input -> Sampling -> Time-Domain -> Framing/VAD -> FFT/STFT ->
    Filtering/Separation/Denoising -> Reconstruction -> Comparison

Run with:  cd extra/legacy_streamlit && streamlit run app.py
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
from scipy.signal import find_peaks

# audio_toolkit lives at the repo root, two levels up from this file.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import visualization as viz  # noqa: E402  (sits next to this file)
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
from audio_toolkit.framing import frame_signal  # noqa: E402

st.set_page_config(page_title="Signal Lab — Audio Signal Processing Toolkit", layout="wide", page_icon="∿")

# --------------------------------------------------------------------------
# Theme — "Teal Lab": a minimal, professional signal-processing visual
# system. Structural layout and design tokens only; nothing below touches
# the DSP logic.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --navy:         #0F1B24;
        --navy-2:       #16303D;
        --ink:          #1C2B36;
        --muted:        #5B7184;
        --muted-soft:   #8598A6;
        --bg:           #EFF2F5;
        --surface:      #FFFFFF;
        --border:       #DEE5E9;
        --border-soft:  #EAEFF1;
        --accent:       #0891B2;
        --accent-dark:  #076C82;
        --accent-tint:  #E1F3F7;
        --amber:        #C2760F;
        --amber-tint:   #FBF1DF;
        --green:        #15803D;
        --green-tint:   #E3F5EA;
        --red:          #C0392B;
        --red-tint:     #FBEAE8;
        --radius-sm:    9px;
        --radius-md:    14px;
        --radius-lg:    22px;
        --shadow-sm:    0 2px 8px rgba(15,27,36,0.06);
        --shadow-md:    0 12px 32px rgba(15,27,36,0.09);
    }

    html { font-size: 17.5px; }
    html, body, .stApp, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
    code, .mono, div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }
    .stButton > button, .stDownloadButton > button, .stSelectbox, .stTextInput, .stSlider label,
    .stCheckbox label, .stRadio label { font-size: 0.98rem; }

    .stApp { background-color: var(--bg); }
    .block-container { padding-top: 2.6rem; padding-bottom: 6.5rem; max-width: 1360px; }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.01em; }
    h3 { font-weight: 700; }
    p, li, span, label { color: var(--ink); }
    [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

    /* ---------- Sidebar: icon rail + processing pipeline ---------- */
    section[data-testid="stSidebar"] {
        background-color: var(--bg);
        border-right: 1px solid var(--border-soft);
    }
    section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
    section[data-testid="stSidebar"] * { color: var(--ink); }
    section[data-testid="stSidebar"] hr { border-color: var(--border-soft) !important; margin: 14px 0; }

    [class*="st-key-sidebar_"] {
        background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius-lg);
        padding: 14px; margin-bottom: 14px; box-shadow: var(--shadow-sm);
    }
    .brand { display:flex; align-items:center; gap:10px; }
    .brand-mark { flex: 0 0 auto; width: 32px; height: 32px; border-radius: 9px; background: var(--accent-tint); display:flex; align-items:center; justify-content:center; }
    .brand-title { font-weight: 700; font-size: 1.02rem; color: var(--navy); line-height:1.2; letter-spacing: -0.01em; }
    .brand-sub { font-size: 0.72rem; color: var(--muted); letter-spacing: 0.03em; margin-top: 1px; }

    /* icon-only group nav (help= adds a tooltip wrapper, so use a descendant match) */
    [class*="st-key-railbtn_"] .stButton button {
        border-radius: var(--radius-md); padding: 10px 0; box-shadow: none; border: 1px solid transparent;
        background: transparent;
    }
    [class*="st-key-railbtn_"] .stButton button p { display: none; }
    [class*="st-key-railbtn_"] .stButton button [data-testid="stIconMaterial"] { font-size: 1.4rem; color: var(--muted); }
    [class*="st-key-railbtn_"] .stButton button:hover { background: var(--accent-tint); }
    [class*="st-key-railbtn_"] .stButton button:hover [data-testid="stIconMaterial"] { color: var(--accent-dark); }
    [class*="st-key-railbtn_"] .stButton button[kind="primary"] { background: var(--accent); box-shadow: var(--shadow-sm); }
    [class*="st-key-railbtn_"] .stButton button[kind="primary"] [data-testid="stIconMaterial"] { color: #FFFFFF; }

    /* vertical processing-pipeline timeline */
    .timeline-heading { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: var(--muted-soft); margin: 2px 2px 8px 2px; }
    [class*="st-key-pipeline_track"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
    [class*="st-key-pl_"] .stButton > button {
        border: none; border-left: 2px solid var(--border); border-radius: 0; background: transparent;
        text-align: left; justify-content: flex-start; padding: 9px 8px 9px 16px; font-size: 0.92rem;
        font-weight: 600; color: var(--muted); box-shadow: none; width: 100%; position: relative;
    }
    [class*="st-key-pl_"] .stButton > button::before {
        content: ''; position: absolute; left: -4px; top: 50%; transform: translateY(-50%);
        width: 7px; height: 7px; border-radius: 50%; background: var(--surface); border: 2px solid var(--border);
    }
    [class*="st-key-pl_"] .stButton > button:hover { color: var(--accent-dark); background: var(--accent-tint); }
    [class*="st-key-pl_"] .stButton > button[kind="primary"] {
        border-left: 2px solid var(--accent); background: var(--accent-tint); color: var(--accent-dark); box-shadow: none;
    }
    [class*="st-key-pl_"] .stButton > button[kind="primary"]::before { background: var(--accent); border-color: var(--accent); }

    .signal-card { background: var(--accent-tint); border: 1px solid var(--border-soft); border-radius: var(--radius-md); padding: 11px 13px; }
    .signal-card-name { font-weight: 600; font-size: 0.9rem; color: var(--navy); word-break: break-word; }
    .signal-card-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.76rem; color: var(--muted); margin-top: 4px; letter-spacing: 0.01em; }
    .signal-card-empty { font-size: 0.86rem; color: var(--muted-soft); }

    /* ---------- Top-right status indicator (all pages) ---------- */
    .topbar-status-row { display:flex; justify-content:flex-end; margin-bottom: 14px; }
    .topbar-status {
        display:inline-flex; align-items:center; gap:7px; padding: 8px 18px; border-radius: 20px;
        font-size: 0.88rem; font-weight: 700; white-space: nowrap;
    }
    .status-dot { width:6px; height:6px; border-radius:50%; display:inline-block; }
    .topbar-status.neutral { background: var(--surface); border: 1px solid var(--border); color: var(--muted); }
    .topbar-status.neutral .status-dot { background: var(--muted-soft); }
    .topbar-status.active { background: var(--accent); color: #FFFFFF; box-shadow: var(--shadow-sm); }
    .topbar-status.active .status-dot { background: #FFFFFF; }

    /* ---------- Dashboard hero banner (Audio Input page) ---------- */
    .st-key-hero_banner {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%);
        border-radius: var(--radius-lg); padding: 26px 28px; margin-bottom: 0;
        box-shadow: var(--shadow-md); position: relative; overflow: hidden; min-height: 176px;
    }
    .st-key-hero_banner p, .st-key-hero_banner span, .st-key-hero_banner div { color: #FFFFFF; }
    .hero-eyebrow { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #5FC3DB !important; position: relative; z-index: 1; }
    .hero-title { font-size: 1.7rem; font-weight: 700; margin: 4px 0 0 0; color: #FFFFFF !important; line-height: 1.28; letter-spacing: -0.015em; max-width: 78%; position: relative; z-index: 1; }
    .hero-sub { font-size: 0.95rem; color: #93AABA !important; margin-top: 6px; max-width: 70%; position: relative; z-index: 1; }
    .hero-illustration { position: absolute; right: -6px; bottom: -8px; opacity: 0.9; z-index: 0; }

    /* ---------- Upload widget card (Audio Input page) ---------- */
    .st-key-upload_card {
        background: var(--surface); border: 1.5px dashed var(--border); border-radius: var(--radius-lg);
        padding: 18px 20px 8px 20px; box-shadow: var(--shadow-sm); min-height: 176px;
        display:flex; flex-direction:column; justify-content:center;
    }
    .upload-title { font-weight: 700; font-size: 1.05rem; color: var(--navy); text-align:center; }
    .upload-sub { font-size: 0.82rem; color: var(--muted); text-align:center; margin-top: 2px; margin-bottom: 8px; }
    .st-key-upload_card [data-testid="stFileUploaderDropzone"] { border: none; background: transparent; padding: 0; }

    /* ---------- Demo signal cards ---------- */
    .demo-card-icon { width: 38px; height: 38px; border-radius: 11px; background: var(--accent-tint); display:flex; align-items:center; justify-content:center; margin: 0 auto 8px auto; }
    .demo-card-name { font-weight: 700; font-size: 1rem; color: var(--navy); text-align:center; }
    .demo-card-sub { font-size: 0.8rem; color: var(--muted); text-align:center; margin-top: 2px; margin-bottom: 10px; }
    [class*="st-key-demo_card_"] {
        background: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius-lg);
        padding: 18px 14px 14px 14px; box-shadow: var(--shadow-sm);
    }

    /* ---------- Signal overview (thin progress bars) ---------- */
    .overview-label { font-size: 0.94rem; font-weight: 600; color: var(--ink); padding-top: 6px; }
    .overview-value { font-size: 0.9rem; font-weight: 700; color: var(--navy); text-align: right; padding-top: 6px; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stProgress"] { padding-top: 10px; }
    div[data-testid="stProgress"] > div > div { background: var(--border-soft) !important; border-radius: 10px; height: 7px !important; }
    div[data-testid="stProgress"] > div > div > div { background: var(--accent) !important; border-radius: 10px; }

    /* ---------- Cards / metrics ---------- */
    div[data-testid="stMetric"] {
        background-color: var(--surface); border: 1px solid var(--border-soft); border-radius: var(--radius-md);
        padding: 0.85rem 1rem; box-shadow: var(--shadow-sm);
    }
    div[data-testid="stMetricLabel"] { color: var(--muted); font-size: 0.85rem; }
    div[data-testid="stMetricValue"] {
        color: var(--navy); font-size: 1.7rem; line-height: 1.25;
        white-space: normal; overflow-wrap: break-word;
    }

    .section-label { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); margin: 20px 0 10px 0; }

    /* ---------- Buttons ---------- */
    .stButton > button, .stDownloadButton > button {
        border-radius: var(--radius-sm); border: 1px solid var(--border); color: var(--ink);
        background-color: var(--surface); font-weight: 600; transition: all 0.14s ease; box-shadow: var(--shadow-sm);
    }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent-dark); background-color: var(--accent-tint); }
    .stButton > button[kind="primary"] { background-color: var(--accent); color: #FFFFFF; border-color: var(--accent); }
    .stButton > button[kind="primary"]:hover { background-color: var(--accent-dark); border-color: var(--accent-dark); color: #FFFFFF; }

    /* ---------- Expanders / dataframe ---------- */
    details[data-testid="stExpander"] { border: 1px solid var(--border-soft); border-radius: var(--radius-md); background-color: var(--surface); box-shadow: var(--shadow-sm); }
    div[data-testid="stAlert"] { border-radius: var(--radius-md); }

    /* ---------- Loaded-signal card (Audio Input page) ---------- */
    .loaded-badge { display:inline-block; font-size: 0.78rem; font-weight:700; letter-spacing:0.06em; text-transform: uppercase; color: var(--green); background: var(--green-tint); padding: 3px 10px; border-radius: 12px; }
    .loaded-name { font-size: 1.2rem; font-weight: 700; color: var(--navy); margin: 9px 0 10px 0; word-break: break-word; }

    /* ---------- Checklist ---------- */
    .checklist-item { font-size: 0.95rem; padding: 4px 0; }
    .check-done { color: var(--green); font-weight: 600; }
    .check-pending { color: var(--muted); }

    /* ---------- Legend chips (VAD) ---------- */
    .legend-chip { font-size: 0.88rem; font-weight: 600; padding: 3px 4px; }
    .legend-speech { color: var(--green); }
    .legend-silence { color: var(--muted); }

    /* ---------- Empty state ---------- */
    .empty-state { text-align:center; padding: 44px 20px 18px 20px; }
    .empty-state-icon { color: var(--accent); opacity: 0.65; margin-bottom: 4px; }
    .empty-state-title { font-size: 1.2rem; font-weight: 700; color: var(--navy); margin-top: 12px; }
    .empty-state-sub { font-size: 0.95rem; color: var(--muted); margin-top: 4px; max-width: 420px; margin-left:auto; margin-right:auto; }
    .empty-state-tags { text-align:center; font-size: 0.8rem; color: var(--muted-soft); margin-top: 20px; letter-spacing: 0.01em; }

    /* ---------- Persistent mini player ---------- */
    .st-key-mini_player {
        position: fixed; left: 0; right: 0; bottom: 0; z-index: 999;
        background: var(--navy); padding: 10px 26px 6px 26px; box-shadow: 0 -4px 18px rgba(0,0,0,0.2);
        border-top: 1px solid rgba(255,255,255,0.07);
    }
    .st-key-mini_player * { color: #DCE6EC; }
    .mini-player-name { font-size: 0.92rem; font-weight: 600; margin-bottom: 2px; letter-spacing: -0.005em; }
    .mini-player-meta { font-family:'JetBrains Mono',monospace; font-size: 0.82rem; color: #7C93A3; text-align:right; padding-top: 10px; }
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


def wave_icon(color: str = "#0891B2", size: int = 16) -> str:
    """A small inline waveform glyph, reused for the brand mark and demo cards."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M0.5 9 L3.2 9 L4.6 4 L7 12.5 L9 6.5 L10.5 9 L12 9 L13.2 6 L15.5 9" '
        f'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>'
    )


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


def empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <svg class="empty-state-icon" width="34" height="34" viewBox="0 0 34 34" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M1 20 L7 20 L10 9 L15 27 L19 14 L22 20 L26 20 L29 15 L33 20"
                      stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
            </svg>
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
    ("Input", "mic", [("input", "Audio Input")]),
    ("Analysis", "monitoring", [
        ("file_info", "File Information"),
        ("waveform", "Waveform"),
        ("sampling", "Sampling & Aliasing"),
        ("vad", "Voice Activity Detection"),
        ("fft", "FFT Spectrum"),
        ("spectrogram", "Spectrogram"),
    ]),
    ("Processing", "tune", [
        ("filtering", "Filtering"),
        ("separation", "Vocal / Instrumental Separation"),
        ("noise", "Noise Reduction"),
    ]),
    ("Comparison", "balance", [("comparison", "Signal Comparison")]),
]
PAGE_TITLES = {key: label for _, _, items in NAV_GROUPS for key, label in items}
PAGE_TO_GROUP = {key: group_name for group_name, _, items in NAV_GROUPS for key, _ in items}

# --------------------------------------------------------------------------
# Left column — icon rail (top-level categories) + vertical processing
# pipeline (pages within the active category). Replaces the old bulky
# text navigation panel with a minimal, icon-driven nav.
# --------------------------------------------------------------------------
with st.sidebar:
    with st.container(key="sidebar_brand"):
        st.markdown(
            f'<div class="brand"><div class="brand-mark">{wave_icon("#0891B2", 16)}</div>'
            '<div><div class="brand-title">Signal Lab</div>'
            '<div class="brand-sub">DSP Toolkit</div></div></div>',
            unsafe_allow_html=True,
        )

    active_group = PAGE_TO_GROUP[st.session_state.nav]

    with st.container(key="sidebar_rail"):
        rail_cols = st.columns(len(NAV_GROUPS))
        for col, (group_name, icon, items) in zip(rail_cols, NAV_GROUPS):
            with col:
                with st.container(key=f"railbtn_{group_name}"):
                    is_active = group_name == active_group
                    if st.button(
                        group_name, key=f"grp_{group_name}", icon=f":material/{icon}:",
                        help=group_name, type="primary" if is_active else "secondary", width="stretch",
                    ):
                        st.session_state.nav = items[0][0]
                        st.rerun()

    with st.container(key="sidebar_pipeline"):
        st.markdown(f'<div class="timeline-heading">{active_group} pipeline</div>', unsafe_allow_html=True)
        active_items = next(items for group_name, _, items in NAV_GROUPS if group_name == active_group)
        with st.container(key="pipeline_track"):
            for key, label in active_items:
                is_active = st.session_state.nav == key
                with st.container(key=f"pl_{key}"):
                    if st.button(label, key=f"navbtn_{key}", type="primary" if is_active else "secondary", width="stretch"):
                        st.session_state.nav = key
                        st.rerun()

    with st.container(key="sidebar_signal"):
        st.markdown('<div class="timeline-heading">Current signal</div>', unsafe_allow_html=True)
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
        if has_audio() and st.button("Clear signal", key="reset_signal_btn", icon=":material/restart_alt:", help="Clear the loaded signal and results", width="stretch"):
            for k in ("audio_y", "audio_sr", "audio_name", "results", "_nr_result", "_sep_result", "_filt_result", "song_truth", "noise_truth", "_last_upload_size", "_last_upload_info", "_last_upload_id"):
                st.session_state.pop(k, None)
            st.session_state.nav = "input"
            st.rerun()

# --------------------------------------------------------------------------
# Top-right status indicator (every page)
# --------------------------------------------------------------------------
status_label = "Signal Loaded" if has_audio() else "Ready"
status_class = "active" if has_audio() else "neutral"
st.markdown(
    f'<div class="topbar-status-row"><div class="topbar-status {status_class}">'
    f'<span class="status-dot"></span>{status_label}</div></div>',
    unsafe_allow_html=True,
)

page = st.session_state.nav

# --------------------------------------------------------------------------
# Page — Audio Input
# --------------------------------------------------------------------------
if page == "input":
    # ---------------------------------------------------------------
    # Top row: compact upload widget (left) + hero banner (right)
    # ---------------------------------------------------------------
    up_col, hero_col = st.columns([1, 1.5], gap="medium")

    with up_col:
        with st.container(key="upload_card"):
            st.markdown('<div class="upload-title">Upload audio</div>', unsafe_allow_html=True)
            st.markdown('<div class="upload-sub">Drag &amp; drop or browse a file</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Upload a WAV or MP3 file", type=["wav", "mp3"], label_visibility="collapsed", key="dashboard_uploader",
            )
            if uploaded is not None and st.session_state.get("_last_upload_id") != uploaded.file_id:
                try:
                    y, sr = io_utils.load_audio(uploaded, sr=None, mono=True)
                    set_current_audio(y, sr, uploaded.name)
                    st.session_state["_last_upload_info"] = io_utils.get_audio_info(
                        uploaded, filename=uploaded.name, file_size_bytes=uploaded.size
                    )
                    st.session_state["_last_upload_size"] = uploaded.size
                    st.session_state["_last_upload_id"] = uploaded.file_id
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not load file: {e}")

    with hero_col:
        with st.container(key="hero_banner"):
            st.markdown('<div class="hero-eyebrow">Signal Lab</div>', unsafe_allow_html=True)
            st.markdown('<div class="hero-title">Audio Signal Processing Toolkit</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="hero-sub">Analyze, transform, visualize, and compare audio signals.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="hero-illustration">'
                f'<svg width="170" height="100" viewBox="0 0 170 100" fill="none" xmlns="http://www.w3.org/2000/svg">'
                f'<path d="M0 55 L14 55 L20 30 L28 78 L36 15 L44 65 L52 45 L60 55 L150 55" '
                f'stroke="#5FC3DB" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.35"/>'
                f'<path d="M0 70 L20 70 L26 50 L34 88 L42 40 L50 75 L58 60 L66 70 L170 70" '
                f'stroke="#0891B2" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none" opacity="0.55"/>'
                f'</svg></div>',
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------------
    # Demo signal cards
    # ---------------------------------------------------------------
    st.markdown('<div class="section-label">Demo signals</div>', unsafe_allow_html=True)
    demo_defs = [
        ("speech", "Speech-like", "Synthetic voice · 6.0s"),
        ("song", "Song-like", "Melody + chords · 8.0s"),
        ("noisy", "Noisy tone", "Tone + noise floor · 4.0s"),
    ]
    current_name = st.session_state.get("audio_name", "")
    d_cols = st.columns(3)
    for col, (kind, label, sub) in zip(d_cols, demo_defs):
        is_current = current_name == f"synthetic: {label.lower()} demo"
        with col:
            with st.container(key=f"demo_card_{kind}"):
                st.markdown(f'<div class="demo-card-icon">{wave_icon("#0891B2", 16)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="demo-card-name">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="demo-card-sub">{sub}</div>', unsafe_allow_html=True)
                btn_label = "Loaded" if is_current else "Load demo"
                if st.button(btn_label, key=f"demo_btn_{kind}", width="stretch", type="primary" if is_current else "secondary"):
                    load_demo(kind)
                    st.rerun()

    # ---------------------------------------------------------------
    # Signal overview — quick metrics as thin progress bars
    # ---------------------------------------------------------------
    st.markdown('<div class="section-label">Signal overview</div>', unsafe_allow_html=True)
    if not has_audio():
        empty_state()
    else:
        y0, sr0 = st.session_state.audio_y, st.session_state.audio_sr
        info = st.session_state.get("_last_upload_info")
        duration = len(y0) / sr0
        mins, secs = divmod(duration, 60)

        with st.container(border=True):
            top1, top2 = st.columns([3, 1.3], vertical_alignment="center")
            with top1:
                st.markdown('<div class="loaded-badge">✓ AUDIO LOADED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="loaded-name">{st.session_state.audio_name}</div>', unsafe_allow_html=True)
            with top2:
                st.metric("Duration", f"{int(mins):02d}:{secs:05.2f}")

            audio_player(y0, sr0)

            peak = float(np.max(np.abs(y0))) if len(y0) else 0.0
            rms_val = float(np.sqrt(np.mean(y0.astype(np.float64) ** 2))) if len(y0) else 0.0
            energy_pct = min(rms_val / 0.3, 1.0)
            noise_ref = st.session_state.get("noise_truth")
            snr_val = metrics.snr_db(noise_ref[0], y0) if noise_ref else None

            rows = [
                ("Peak amplitude", peak, f"{peak:.3f}"),
                ("Energy level", energy_pct, f"{rms_val:.3f} RMS"),
            ]
            l1, b1, v1 = st.columns([2, 5, 1.4])
            l1.markdown(f'<div class="overview-label">{rows[0][0]}</div>', unsafe_allow_html=True)
            b1.progress(min(rows[0][1], 1.0))
            v1.markdown(f'<div class="overview-value">{rows[0][2]}</div>', unsafe_allow_html=True)

            l2, b2, v2 = st.columns([2, 5, 1.4])
            l2.markdown(f'<div class="overview-label">{rows[1][0]}</div>', unsafe_allow_html=True)
            b2.progress(rows[1][1])
            v2.markdown(f'<div class="overview-value">{rows[1][2]}</div>', unsafe_allow_html=True)

            l3, b3, v3 = st.columns([2, 5, 1.4])
            l3.markdown('<div class="overview-label">Signal-to-noise ratio</div>', unsafe_allow_html=True)
            if snr_val is not None:
                b3.progress(min(max(snr_val, 0) / 40, 1.0))
                v3.markdown(f'<div class="overview-value">{snr_val:.1f} dB</div>', unsafe_allow_html=True)
            else:
                b3.progress(0)
                v3.markdown('<div class="overview-value">—</div>', unsafe_allow_html=True)
            if snr_val is None:
                st.caption("SNR needs a ground-truth reference — available for the synthetic noisy-tone demo, or after Noise Reduction.")

            info_fmt = info.subtype.split("_")[0] if info else "Synthetic"
            st.caption(f"{info_fmt} · {sr0/1000:.1f} kHz · {info.channels if info else 1} channel(s) source")

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
            st.markdown(f'<div class="mini-player-name">{st.session_state.audio_name}</div>', unsafe_allow_html=True)
            audio_player(y0, sr0)
        with mp2:
            st.markdown(f'<div class="mini-player-meta">{sr0/1000:.1f} kHz · {len(y0)/sr0:.1f} s</div>', unsafe_allow_html=True)
