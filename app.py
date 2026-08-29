"""Audio Signal Processing Toolkit — Streamlit UI.

Ties together the audio_toolkit DSP modules into one interactive app
that walks through the full pipeline:

    Input -> Sampling -> Time-Domain -> Framing/VAD -> FFT/STFT ->
    Filtering/Separation/Denoising -> Reconstruction -> Comparison

Run with:  streamlit run app.py
"""
from __future__ import annotations

import io

import numpy as np
import soundfile as sf
import streamlit as st

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


def store_result(name: str, y: np.ndarray, sr: int):
    st.session_state.setdefault("results", {})[name] = (y.astype(np.float32), sr)


def has_audio() -> bool:
    return st.session_state.get("audio_y") is not None


# --------------------------------------------------------------------------
# Sidebar: audio input (shared across every tab)
# --------------------------------------------------------------------------

st.sidebar.title("🎧 Audio Input")

uploaded = st.sidebar.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])
if uploaded is not None and st.session_state.get("_last_upload_id") != uploaded.file_id:
    try:
        y, sr = io_utils.load_audio(uploaded, sr=None, mono=True)
        set_current_audio(y, sr, uploaded.name)
        st.session_state["_last_upload_info"] = io_utils.get_audio_info(
            uploaded, filename=uploaded.name, file_size_bytes=uploaded.size
        )
        st.session_state["_last_upload_id"] = uploaded.file_id
    except Exception as e:
        st.sidebar.error(f"Could not load file: {e}")

st.sidebar.caption("or load a synthetic demo signal (no file needed):")
c1, c2 = st.sidebar.columns(2)
if c1.button("Speech-like demo", use_container_width=True):
    y, sr = demo_signals.generate_speech_like_demo()
    set_current_audio(y, sr, "synthetic: speech-like demo")
    st.session_state.pop("_last_upload_info", None)
if c2.button("Song-like demo", use_container_width=True):
    mix, fg, bg, sr = demo_signals.generate_song_like_demo()
    set_current_audio(mix, sr, "synthetic: song-like demo")
    st.session_state["song_truth"] = (fg, bg, sr)
    st.session_state.pop("_last_upload_info", None)
if st.sidebar.button("Noisy-tone demo", use_container_width=True):
    clean, noisy, sr = demo_signals.generate_noisy_tone_demo()
    set_current_audio(noisy, sr, "synthetic: noisy tone demo")
    st.session_state["noise_truth"] = (clean, sr)
    st.session_state.pop("_last_upload_info", None)

st.sidebar.divider()

if has_audio():
    y0, sr0 = st.session_state.audio_y, st.session_state.audio_sr
    st.sidebar.success(f"Loaded: {st.session_state.audio_name}")
    st.sidebar.write(f"Duration: {len(y0)/sr0:.2f} s  |  Sample rate: {sr0} Hz")
    audio_player(y0, sr0)
    info = st.session_state.get("_last_upload_info")
    if info:
        st.sidebar.caption(
            f"Channels (source): {info.channels} | Subtype: {info.subtype} | "
            f"~Bitrate: {info.bitrate_kbps:.0f} kbps"
            + (" (estimated, lossy)" if info.is_lossy else "")
        )
else:
    st.sidebar.info("No audio loaded yet — upload a file or pick a demo above.")

st.title("Audio Signal Processing Toolkit")
st.caption(
    "Input → Sampling → Time-Domain Analysis → Framing → Energy Analysis → "
    "FFT/STFT → Frequency-Domain Analysis → Filtering/Processing → Reconstruction → Output"
)

tabs = st.tabs([
    "1 · File Info",
    "2 · Waveform",
    "3 · Sampling & Aliasing",
    "4 · VAD",
    "5 · FFT Spectrum",
    "6 · Spectrogram",
    "7 · Filtering",
    "8 · Vocal/Instrumental Separation",
    "9 · Noise Reduction",
    "10 · Signal Comparison",
])

# --------------------------------------------------------------------------
# Tab 1 — File info
# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Audio File Information")
    if not has_audio():
        st.warning("Load a file or demo signal from the sidebar to begin.")
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        info = st.session_state.get("_last_upload_info")
        cols = st.columns(4)
        cols[0].metric("Duration", f"{len(y)/sr:.2f} s")
        cols[1].metric("Sample rate", f"{sr} Hz")
        cols[2].metric("Samples", f"{len(y):,}")
        if info:
            cols[3].metric("Channels (source)", info.channels)
            st.write(
                f"**Subtype:** {info.subtype}  |  "
                f"**Bit depth:** {info.bit_depth or 'N/A (compressed)'}  |  "
                f"**Estimated bitrate:** {info.bitrate_kbps:.0f} kbps"
                + (" (estimated from file size — lossy format)" if info.is_lossy else " (exact, PCM)")
            )
        else:
            cols[3].metric("Channels", "1 (mono, synthetic)")
        st.caption(
            "Note: processing throughout this app is done on a mono, floating-point "
            "([-1, 1]) version of the signal — standard practice for DSP analysis; "
            "channel count/bit depth above describe the original file."
        )
        audio_player(y, sr)

# --------------------------------------------------------------------------
# Tab 2 — Waveform
# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Time-Domain Waveform")
    if not has_audio():
        st.warning("Load audio first.")
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        duration = len(y) / sr
        st.write("Zoom into a region of the signal:")
        zoom = st.slider("Time range (s)", 0.0, float(duration), (0.0, float(duration)), key="wave_zoom")
        start_i, end_i = int(zoom[0] * sr), int(zoom[1] * sr)
        fig = viz.plot_waveform(y, sr, title=f"Waveform — {st.session_state.audio_name}", xlim=zoom)
        st.pyplot(fig)
        if end_i > start_i:
            st.caption(f"Zoomed segment: {zoom[1]-zoom[0]:.2f} s selected")

# --------------------------------------------------------------------------
# Tab 3 — Sampling & Aliasing
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Sampling, Aliasing, and Reconstruction")
    st.markdown(
        "The **Nyquist-Shannon theorem** says a signal band-limited to $f_{max}$ can be "
        "reconstructed perfectly only if sampled at $f_s \\ge 2f_{max}$. Sample below that "
        "rate and the true frequency **aliases** — appears, after reconstruction, as a lower "
        "'ghost' frequency that cannot be told apart from the real thing."
    )
    colA, colB, colC = st.columns(3)
    tone_freq = colA.slider("Tone frequency (Hz)", 20, 4000, 500, key="samp_freq")
    fs_demo = colB.slider("Sampling rate (Hz)", 100, 8000, 800, step=50, key="samp_fs")
    dur_demo = colC.slider("Duration (s)", 0.02, 0.2, 0.05, key="samp_dur")

    nyquist = fs_demo / 2
    t_hi, x_hi = sampling.generate_tone(tone_freq, dur_demo, fs_reference=44100)
    t_samples, x_samples = sampling.sample_signal(tone_freq, dur_demo, fs_demo)
    t_recon = t_hi
    x_recon = sampling.reconstruct_signal(t_samples, x_samples, t_recon)

    if tone_freq > nyquist:
        alias_f = sampling.aliased_frequency(tone_freq, fs_demo)
        st.error(
            f"⚠️ Undersampling! Nyquist frequency is {nyquist:.0f} Hz but the tone is "
            f"{tone_freq} Hz. The reconstructed signal will look like a **{alias_f:.1f} Hz** alias, "
            f"not the true {tone_freq} Hz tone."
        )
    else:
        st.success(f"✅ Sampled above Nyquist ({nyquist:.0f} Hz) — reconstruction should match the original closely.")

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
        if st.button("Run downsample → upsample demo", key="run_resample_demo"):
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
            st.caption("Saved as a result — compare it precisely in tab 10.")
    else:
        st.info("Load audio from the sidebar to try this on a real signal.")

# --------------------------------------------------------------------------
# Tab 4 — VAD
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Voice Activity Detection (Speech vs. Silence)")
    if not has_audio():
        st.warning("Load audio first — the 'Speech-like demo' in the sidebar works well here.")
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2, c3 = st.columns(3)
        frame_ms = c1.slider("Frame length (ms)", 10, 40, 25, key="vad_frame_ms")
        hop_ms = c2.slider("Hop length (ms)", 5, 30, 10, key="vad_hop_ms")
        energy_pct = c3.slider("Energy threshold (percentile)", 10, 90, 40, key="vad_energy_pct")
        band_ratio_thresh = st.slider("Speech-band ratio threshold", 0.0, 1.0, 0.35, key="vad_band_thresh")

        result = vad.detect_speech(
            y, sr, frame_ms=frame_ms, hop_ms=hop_ms,
            energy_percentile=energy_pct, band_ratio_thresh=band_ratio_thresh,
        )
        st.pyplot(viz.plot_vad(y, sr, result))

        speech_time = sum(e - s for s, e in result.speech_segments)
        c1, c2 = st.columns(2)
        c1.metric("Speech detected", f"{speech_time:.2f} s / {len(y)/sr:.2f} s")
        c2.metric("Speech segments", len(result.speech_segments))
        with st.expander("Detected segments (s)"):
            st.write([(round(s, 2), round(e, 2)) for s, e in result.speech_segments])

        st.caption(
            "A frame is classified Speech only if BOTH its short-time energy exceeds an "
            "adaptive percentile threshold AND its FFT-derived speech-band (300-3400 Hz) "
            "energy ratio exceeds its threshold — combining a time-domain and a "
            "frequency-domain feature rejects loud broadband noise that energy alone would "
            "misclassify as speech."
        )

# --------------------------------------------------------------------------
# Tab 5 — FFT spectrum
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("FFT and Frequency Spectrum")
    if not has_audio():
        st.warning("Load audio first.")
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

        log_x = st.checkbox("Log frequency axis", value=False, key="fft_log_x")
        st.pyplot(viz.plot_fft_spectrum(freqs, mag, log_x=log_x))
        dominant = freqs[np.argmax(mag)]
        st.metric("Dominant frequency", f"{dominant:.1f} Hz")

# --------------------------------------------------------------------------
# Tab 6 — Spectrogram
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Spectrogram (Short-Time Fourier Transform)")
    if not has_audio():
        st.warning("Load audio first.")
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2 = st.columns(2)
        n_fft = c1.select_slider("FFT size (frequency resolution)", options=[256, 512, 1024, 2048, 4096], value=2048, key="spec_nfft")
        hop = c2.select_slider("Hop length (time resolution)", options=[64, 128, 256, 512, 1024], value=512, key="spec_hop")
        freqs, times, mag_db = spectral.spectrogram_db(y, sr, n_fft=n_fft, hop_length=hop)
        st.pyplot(viz.plot_spectrogram(freqs, times, mag_db))
        st.caption(
            f"Frequency resolution Δf = sr/n_fft = {sr/n_fft:.1f} Hz. Time resolution ≈ hop/sr = "
            f"{hop/sr*1000:.1f} ms. Larger n_fft sharpens frequency detail but blurs timing "
            "(and vice-versa) — the STFT's fundamental time/frequency trade-off."
        )

# --------------------------------------------------------------------------
# Tab 7 — Filtering
# --------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Audio Filtering (Low/High/Band-Pass/Band-Stop)")
    if not has_audio():
        st.warning("Load audio first.")
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        nyquist = sr / 2
        kind = st.selectbox("Filter type", ["lowpass", "highpass", "bandpass", "bandstop"], key="filt_kind")
        order = st.slider("Filter order", 2, 12, 6, key="filt_order")

        if kind in ("lowpass", "highpass"):
            cutoff = st.slider("Cutoff frequency (Hz)", 20, int(nyquist) - 20, min(1000, int(nyquist) - 20), key="filt_cutoff")
        else:
            low, high = st.slider(
                "Cutoff band (Hz)", 20, int(nyquist) - 20, (300, 3400), key="filt_band"
            )
            cutoff = (low, high)

        sos = filters.design_filter(kind, cutoff, sr, order=order)
        freqs_resp, mag_db = filters.frequency_response(sos, sr)
        cutoffs_plot = [cutoff] if isinstance(cutoff, (int, float)) else list(cutoff)
        st.pyplot(viz.plot_filter_response(freqs_resp, mag_db, cutoffs=cutoffs_plot, title=f"{kind.title()} Filter Response (order {order})"))

        if st.button("Apply filter", key="apply_filter_btn"):
            filtered = filters.apply_filter(y, sos)
            store_result(f"Filtered ({kind}, {cutoff})", filtered, sr)
            c1, c2 = st.columns(2)
            with c1:
                st.write("Original")
                audio_player(y, sr)
            with c2:
                st.write("Filtered")
                audio_player(filtered, sr)
            st.pyplot(viz.plot_before_after(y, sr, "Original", filtered, sr, "Filtered", title="Filtering — Before / After"))
            download_button(filtered, sr, "Download filtered audio (WAV)", "filtered.wav", "dl_filtered")
            st.caption("Saved as a result — compare it precisely in tab 10.")

# --------------------------------------------------------------------------
# Tab 8 — Separation
# --------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Vocal / Instrumental Separation")
    with st.expander("Method & limitations (read this)", expanded=False):
        st.markdown(separation.__doc__.replace("METHOD\n------", "**Method**").replace(
            "LIMITATIONS (important — read before trusting the output)\n-----------------------------------------------------------",
            "**Limitations**"
        ))
    if not has_audio():
        st.warning("Load audio first — the 'Song-like demo' in the sidebar is built to showcase this well.")
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2 = st.columns(2)
        margin_bg = c1.slider("Background margin (aggressiveness)", 1.0, 20.0, 2.0, key="sep_margin_bg")
        margin_fg = c2.slider("Foreground margin (aggressiveness)", 1.0, 20.0, 10.0, key="sep_margin_fg")

        if st.button("Run separation", key="run_sep_btn"):
            fg, bg = separation.separate_vocals_instrumental(y, sr, margin_background=margin_bg, margin_foreground=margin_fg)
            store_result("Separated: foreground (vocal-like)", fg, sr)
            store_result("Separated: background (instrumental-like)", bg, sr)
            st.session_state["_sep_result"] = (fg, bg, sr)

        if "_sep_result" in st.session_state:
            fg, bg, sr = st.session_state["_sep_result"]
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Foreground** (vocal-like)")
                audio_player(fg, sr)
                download_button(fg, sr, "Download foreground", "foreground.wav", "dl_fg")
            with c2:
                st.write("**Background** (instrumental-like)")
                audio_player(bg, sr)
                download_button(bg, sr, "Download background", "background.wav", "dl_bg")
            st.pyplot(viz.plot_before_after(fg, sr, "Foreground (vocal-like)", bg, sr, "Background (instrumental-like)"))

            if "song_truth" in st.session_state:
                true_fg, true_bg, _ = st.session_state["song_truth"]
                st.markdown("**Ground-truth check** (only available for the synthetic song demo):")
                m1, m2 = st.columns(2)
                m1.metric("Foreground correlation vs. true melody", f"{metrics.correlation(true_fg, fg):.3f}")
                m2.metric("Background correlation vs. true chord loop", f"{metrics.correlation(true_bg, bg):.3f}")
            st.caption("Saved as results — compare precisely in tab 10.")

# --------------------------------------------------------------------------
# Tab 9 — Noise reduction
# --------------------------------------------------------------------------
with tabs[8]:
    st.subheader("Noise Reduction (Spectral Subtraction)")
    with st.expander("Method & limitations (read this)", expanded=False):
        st.markdown(noise_reduction.__doc__.replace("METHOD\n------", "**Method**").replace(
            "LIMITATIONS\n-----------", "**Limitations**"
        ))
    if not has_audio():
        st.warning("Load audio first — the 'Noisy-tone demo' in the sidebar has a known clean reference.")
    else:
        y, sr = st.session_state.audio_y, st.session_state.audio_sr
        c1, c2, c3 = st.columns(3)
        noise_dur = c1.slider("Noise sample duration (s, from start)", 0.1, min(2.0, len(y) / sr), 0.5, key="nr_noise_dur")
        alpha = c2.slider("Oversubtraction factor (α)", 0.5, 5.0, 2.0, key="nr_alpha")
        beta = c3.slider("Spectral floor (β)", 0.0, 0.3, 0.05, key="nr_beta")

        if st.button("Run noise reduction", key="run_nr_btn"):
            denoised = noise_reduction.spectral_subtraction(y, sr, noise_duration_s=noise_dur, alpha=alpha, beta=beta)
            store_result("Denoised (spectral subtraction)", denoised, sr)
            st.session_state["_nr_result"] = denoised

        if "_nr_result" in st.session_state:
            denoised = st.session_state["_nr_result"]
            c1, c2 = st.columns(2)
            with c1:
                st.write("Noisy (original)")
                audio_player(y, sr)
            with c2:
                st.write("Denoised")
                audio_player(denoised, sr)
            st.pyplot(viz.plot_before_after(y, sr, "Noisy", denoised, sr, "Denoised"))

            freqs, times, mag_db_noisy = spectral.spectrogram_db(y, sr)
            _, _, mag_db_clean = spectral.spectrogram_db(denoised, sr)
            c1, c2 = st.columns(2)
            c1.pyplot(viz.plot_spectrogram(freqs, times, mag_db_noisy, title="Noisy — Spectrogram"))
            c2.pyplot(viz.plot_spectrogram(freqs, times, mag_db_clean, title="Denoised — Spectrogram"))

            if "noise_truth" in st.session_state:
                clean_ref, _ = st.session_state["noise_truth"]
                st.markdown("**Ground-truth comparison** (only available for the synthetic noisy-tone demo):")
                m1, m2, m3 = st.columns(3)
                m1.metric("SNR before", f"{metrics.snr_db(clean_ref, y):.1f} dB")
                m2.metric("SNR after", f"{metrics.snr_db(clean_ref, denoised):.1f} dB")
                m3.metric("Correlation after", f"{metrics.correlation(clean_ref, denoised):.3f}")
            download_button(denoised, sr, "Download denoised audio (WAV)", "denoised.wav", "dl_denoised")
            st.caption("Saved as a result — compare it precisely in tab 10.")

# --------------------------------------------------------------------------
# Tab 10 — Signal comparison
# --------------------------------------------------------------------------
with tabs[9]:
    st.subheader("Signal Comparison")
    results = st.session_state.get("results", {})
    if len(results) < 1:
        st.warning("Load audio first, then run a processing step (filter/denoise/separate/resample) to have something to compare.")
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

        m1, m2, m3 = st.columns(3)
        m1.metric("MSE", f"{metrics.mse(y_a, y_b):.6f}")
        m2.metric("SNR (dB)", f"{metrics.snr_db(y_a, y_b):.1f}")
        m3.metric("Correlation", f"{metrics.correlation(y_a, y_b):.3f}")

        st.pyplot(viz.plot_before_after(y_a, sr_a, f"A: {name_a}", y_b, sr_b, f"B: {name_b}"))
        c1, c2 = st.columns(2)
        with c1:
            audio_player(y_a, sr_a)
        with c2:
            audio_player(y_b, sr_b)

        st.caption(
            "MSE: lower = more similar. SNR: treats A as ground truth and B's deviation from it "
            "as 'noise' — higher dB = closer match. Correlation: shape similarity independent of "
            "amplitude, in [-1, 1]."
        )
