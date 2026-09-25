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
| `vault/`         | Encryption + PNG steganography — independent of the DSP package   |
| `server/`        | FastAPI layer exposing both as JSON over HTTP                     |
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

    editing.py                  Trim, cut, splice, merge, fades, crossfade laws, level matching
    silence.py                  Silence detection (Schmitt trigger + adaptive threshold) and removal
    timescale.py                Time stretching, pitch shifting, and transposition measurement
    vocals.py                   Karaoke / a cappella stems with separation metrics

vault/
    container.py                On-image format: header layout, metadata block, SHA-256
    crypto.py                   scrypt key derivation + AES-256-GCM authenticated encryption
    stego.py                    Bit packing, capacity maths, LSB embed/extract, cover generation
    pipeline.py                 The two end-to-end directions, plus timing/metrics
    selftest.py                 25 security and integrity checks (`python -m vault.selftest`)
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

## Editing and production

Every result below is added to the session as a new signal, so steps
chain: trim a clip, fade the trimmed result, merge that with another.

- **Trim, Cut & Fade** — drag across the waveform to select a region,
  then keep it or delete it. Deleting splices together material that was
  seconds apart, which is a step discontinuity and therefore a click, so
  every interior seam is crossfaded. Fades offer five gain curves
  (linear, exponential, logarithmic, S-curve, equal-power) with the
  chosen curve plotted before it is applied.
- **Silence Remover** — reduces the signal to a frame-rate RMS envelope
  and gates it with a Schmitt trigger (separate on/off thresholds, so
  the detector cannot chatter around the boundary). The threshold is
  derived from the recording's own level distribution rather than fixed,
  because a clean studio take and a noisy phone recording put their
  noise floors 40 dB apart. Two guards prevent over-editing: a minimum
  silence length, and padding kept at each end of a cut.
- **Merge** — joins clips in a chosen order with a crossfade or a gap,
  and exposes both crossfade laws: equal-power for unrelated sources
  (whose powers add, so a linear fade would dip ~3 dB mid-join) and
  linear for two pieces of the same take. Mixed sample rates are
  resampled to the first clip's rate before joining.
- **Speed & Pitch** — retime with a phase vocoder (duration changes,
  pitch does not) or by resampling (varispeed: both scale together), and
  transpose in semitones while holding duration fixed. **The effect on
  pitch is then measured, not asserted** — see below.
- **Karaoke & Vocals** — the application layer over `separation.py`,
  producing a backing track, an isolated vocal, or both, level-matched
  to the source so an A/B compares the processing rather than the gain.
  Reports vocal-band suppression, inter-stem correlation and energy
  share, plus correlation against ground truth on the song demo.

### Verifying a pitch transform

Claims about pitch are checked against the audio. A transposition
multiplies every frequency by the same ratio, and multiplication is
addition on a logarithmic axis — so transposing slides the whole
spectrum sideways along a log-frequency axis without changing its shape.
Resampling the averaged spectrum onto a grid spaced in cents and
cross-correlating source against result recovers the shift as the lag of
the correlation peak, refined by parabolic interpolation.

Because it matches the spectral envelope rather than any single partial,
it needs no note to be present and works on polyphonic music, speech and
noise alike. Measured against known ground truth on the bundled demos it
lands within ~2 cents: a 1.5× phase-vocoder stretch reports +0.1 cents
(pitch preserved) while the same stretch by resampling reports +700.2
against an ideal +702.

Fundamental frequency (YIN) is reported alongside, but only when the
material has a stable enough pitch for the number to mean anything — the
F0 track's interquartile spread is checked first, and a chord or dense
mix is reported as "no stable pitch" rather than given a median of
several different notes.

## Secure Vault — audio ⇄ encrypted PNG

A separate subsystem (`vault/`) that encrypts an audio file and hides the
ciphertext in the low bits of a PNG, then recovers the original file
**byte for byte**.

```
audio file → SHA-256 → +metadata → AES-256-GCM → header+ciphertext
           → bitstream → LSB into RGB pixels → PNG

PNG → LSB extraction → header → AES-256-GCM (tag verified)
    → metadata+file → SHA-256 compared → the original file
```

### The three pieces, and what each one actually provides

They are routinely conflated, so it is worth being precise:

- **Encryption** (AES-256-GCM) provides *confidentiality*. Without it,
  hiding data in an image is obscurity: anyone who suspects the
  technique reads the payload straight out.
- **Steganography** (LSB) provides *concealment*, not confidentiality.
  It hides that a message exists, and makes the container a PNG.
- **Authentication** (the 128-bit GCM tag) and the **SHA-256** digest
  provide *integrity*. Without them a flipped bit yields plausible
  garbage that the decoder would hand over as if it were audio.

Remove any one and the system fails at something the other two cannot
cover for.

### Order matters: encrypt first, then embed

Embedding a recognisable file and calling it secure is the classic
mistake. Encrypting first means an attacker who knows exactly where to
look, and extracts every low bit perfectly, is left with ciphertext.

The byte-value histogram in the UI is the visible proof: the source WAV
measures about 6.7 bits of entropy per byte with heavy spikes at 0x00
and 0xFF (PCM samples cluster near zero), while the ciphertext measures
7.999 of a possible 8 and sits flat on the uniform line. No waveform, no
format signature, no structure survives.

### Container format

A 48-byte cleartext header — magic `SGVL`, version, scrypt parameters,
salt, nonce, payload length — followed by the AES-GCM ciphertext.

The filename, original size and SHA-256 digest are **not** in the
header; they live inside the ciphertext, because each one leaks. A
filename is often the most sensitive part of a file, and a cleartext
digest turns the container into an oracle: anyone who guesses the
plaintext can hash their guess and confirm it without ever attacking the
password.

The header is nonetheless passed to AES-GCM as **associated data** —
authenticated but not encrypted — so editing the salt, nonce or length
to steer the decoder produces an authentication failure rather than a
decoder that quietly complies.

### Key derivation

scrypt, not a bare SHA-256 of the password and not PBKDF2. A password is
short and structured where AES wants 32 uniform bytes, so it has to be
stretched; PBKDF2 is only computation-hard, which is exactly the
workload custom hardware parallelises best. scrypt is additionally
*memory*-hard — at N = 2¹⁵, r = 8 each guess costs roughly 32 MB and tens
of milliseconds, which is unnoticeable once and ruinous a billion times
over. Salt and nonce are drawn fresh from the OS CSPRNG per encryption.

### Capacity

```
capacity_bits  = width × height × 3 channels × bits_per_channel
capacity_bytes = capacity_bits / 8
```

The image is sized to the payload and made square-ish, since for a fixed
area a square has the smallest maximum dimension. A 192 KB WAV lands in a
717 × 716 image at 99.9% utilisation.

The generated cover is uniform random noise, and that is a deliberate
choice rather than a lazy one: the low bits of a smooth image are *not*
random, so replacing them with ciphertext leaves the statistical
signature that classical LSB attacks (chi-squared, RS analysis) look
for. Noise has LSBs that are already uniform, so embedding changes the
image's statistics not at all. The cost is that noise is incompressible,
so the PNG runs about 8× the audio size — a real trade the UI reports
rather than hides.

### Running the self-test

```bash
python -m vault.selftest
```

25 checks covering bit-for-bit round trip, wrong password, a single
flipped bit, an edited header, an image with no container, user-supplied
covers, capacity arithmetic at 1 and 2 bits per channel, ciphertext
entropy, and key derivation. All of them are also runnable from the UI's
**Security & integrity tests** panel.

### Notes

- PNG is mandatory, JPEG is fatal. JPEG quantises in the frequency
  domain and will change a pixel by a step or two — invisible to a
  viewer, and total destruction for a payload in the lowest bit. The
  in-app image preview is deliberately downscaled and served as JPEG for
  exactly this reason: it is a picture, not the artefact, and the UI says
  so.
- The payload is **not** compressed before encryption. Every accepted
  format is already compressed or noise-like, so DEFLATE would save a
  fraction of a percent — and compressing before encrypting makes
  ciphertext length depend on plaintext content, which is the ingredient
  behind CRIME/BREACH-style attacks.
- A failed decryption returns nothing. There is no partial output and no
  "best effort" audio.

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

- Everything here is classical signal processing and conventional
  cryptography (documented in each module's docstring), not machine
  learning — separation, denoising, the phase vocoder, pitch shifting and
  the vault all included. There are no models, no training and no
  inference anywhere. Dependencies stay at
  NumPy/SciPy/librosa/SoundFile/Pillow/cryptography, runnable anywhere
  without a GPU or large model downloads, at the cost of
  separation/denoising quality compared to trained neural models.
- No cryptographic primitive is homemade. AES-256-GCM comes from the
  `cryptography` library and scrypt/SHA-256 from the Python standard
  library; `vault/` composes them and explains the composition.
- Time stretching and pitch shifting inherit the phase vocoder's
  artefacts: transients smear across the frames they were stretched
  over, and harmonics can lose phase alignment, heard as faint
  chorusing. Both grow with the stretch factor, so 0.8×–1.25× is
  near-transparent while 0.5× or 2× announces itself. Resampling has no
  such artefacts because nothing is estimated — it just moves the pitch
  too.
- MP3 reading uses libsndfile (via `soundfile`)/`audioread`; exporting
  processed audio always writes WAV (lossless, no external encoder
  dependency required).
