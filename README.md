# Audio Signal Processing Toolkit

An interactive Python toolkit demonstrating the complete audio DSP
pipeline on real (or synthetically generated) audio:

```
Input → Sampling → Time-Domain Analysis → Framing → Energy Analysis →
FFT/STFT → Frequency-Domain Analysis → Filtering/Processing →
Reconstruction → Output
```

Built for a university-style live demo: every tab pairs a short
explanation of the underlying math with an interactive control and a
real visualization, and every synthetic-data tab includes a
one-click "demo" button so it works without needing to hunt down an
audio file beforehand (real WAV/MP3 upload is fully supported too).

## Architecture

| Path             | Role                                                             |
| ---------------- | ---------------------------------------------------------------- |
| `audio_toolkit/` | All DSP — pure functions, no UI dependencies                      |
| `server/`        | FastAPI layer exposing `audio_toolkit` as JSON over HTTP          |
| `web/`           | React + TypeScript + Tailwind frontend (canvas-rendered charts)   |
| `app.py`         | Original Streamlit UI — superseded by `web/`, kept for reference  |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
./run.sh
```

Starts the API on `http://localhost:8000` and the web app on
`http://localhost:5173`; Ctrl+C stops both. On first run it installs the
frontend's npm dependencies.

To run the two halves separately:

```bash
python -m uvicorn server.main:app --port 8000 --reload   # API
cd web && npm run dev                                    # frontend
```

The legacy Streamlit UI still runs with `streamlit run app.py`.

This opens the app in your browser (default: http://localhost:8501).

## Project layout

```
app.py                          Streamlit UI — 10 tabs, one per DSP concept
audio_toolkit/
    io_utils.py                 Load/save audio, file metadata (duration, sr, channels, bitrate)
    framing.py                  Split a signal into short overlapping frames + windowing
    vad.py                      Voice Activity Detection (short-time energy + FFT speech-band feature)
    spectral.py                 FFT, STFT, and spectrogram computation
    filters.py                  Butterworth low/high/band-pass/band-stop filter design + application
    separation.py               Vocal/instrumental separation (nearest-neighbor spectral filtering)
    noise_reduction.py          Spectral-subtraction denoising
    sampling.py                 Sampling-rate, aliasing, and reconstruction demonstrations
    metrics.py                  MSE, SNR, correlation between two signals
    visualization.py            Matplotlib figure builders shared across tabs
    demo_signals.py             Synthetic demo signals (speech-like, noisy tone, song-like)
```

Each module is self-contained and independently testable — none of
them import Streamlit, so the DSP logic can be reused (or unit
tested) outside the UI.

## What each tab demonstrates

1. **File Info** — duration, sample rate, channel count, and (for PCM)
   exact bit depth/bitrate, or an estimated average bitrate for lossy
   formats like MP3 (libsndfile doesn't expose the original encoder
   bitrate, so it's derived from file size ÷ duration).
2. **Waveform** — the time-domain signal, zoomable to any sub-range.
3. **Sampling & Aliasing** — pick a tone frequency and a sampling
   rate below/above Nyquist and watch reconstruction either match the
   original or fold into an aliased ghost frequency; also lets you
   downsample/upsample a loaded file and measure what's lost.
4. **VAD** — frames the signal (20-30 ms), computes per-frame
   short-time energy *and* an FFT-derived speech-band energy ratio,
   and classifies each frame Speech/Silence by thresholding both.
5. **FFT Spectrum** — frequency content of the whole signal or of a
   single selected frame, shown side-by-side with its time-domain view.
6. **Spectrogram** — STFT magnitude over time, with adjustable FFT
   size/hop length to see the time/frequency resolution trade-off.
7. **Filtering** — design and apply Butterworth low/high/band-pass/
   band-stop filters with a live frequency-response plot, then play
   and download the filtered audio.
8. **Vocal/Instrumental Separation** — a classical (non-ML)
   nearest-neighbor spectral filtering method (related to REPET-SIM).
   **Read the in-app "Method & limitations" panel** — this is not a
   substitute for deep-learning separators like Demucs/Spleeter, and
   is documented as such.
9. **Noise Reduction** — spectral subtraction using a noise profile
   estimated from a user-selected noise-only region, with adjustable
   oversubtraction/floor parameters and a spectrogram before/after view.
10. **Signal Comparison** — pick any two signals produced during the
    session (original, filtered, denoised, separated, resampled) and
    compute MSE, SNR, and correlation between them.

## Sample audio

`sample_data/` has pre-rendered WAV versions of the three synthetic
demos (speech-like, noisy tone + its clean reference, song-like
mixture) — generated by `audio_toolkit/demo_signals.py`. Use these to
test the file-upload path itself, or regenerate them anytime with:

```bash
python3 -c "
from audio_toolkit import demo_signals, io_utils
y, sr = demo_signals.generate_speech_like_demo(duration=6.0)
io_utils.save_audio('sample_data/speech_like_demo.wav', y, sr)
"
```

## Notes and limitations

- Vocal separation and noise reduction are implemented as classical
  signal-processing techniques (documented in each module's
  docstring), not machine-learning models — this keeps the toolkit's
  dependencies to NumPy/SciPy/librosa/Matplotlib/Streamlit/SoundFile,
  runnable anywhere without GPU or large model downloads, at the cost
  of separation/denoising quality compared to trained neural models.
- MP3 reading uses libsndfile (via `soundfile`)/`audioread`; exporting
  processed audio always writes WAV (lossless, no external encoder
  dependency required).
