# Signal Lab — Audio Signal Processing Toolkit

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=flat-square&logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-6-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?style=flat-square&logo=vite&logoColor=white)
![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=flat-square)

An interactive web app that walks through the **complete audio DSP pipeline**, from loading a file to filtering, separating and reconstructing it, with a live visualization at every step.

```
Input → Sampling → Time-Domain Analysis → Framing → Energy Analysis →
FFT/STFT → Frequency-Domain Analysis → Filtering/Processing →
Reconstruction → Output
```

It was built for a live university demo. Each tab pairs a short explanation of the math with interactive controls and a real chart. Every tab has a one-click **demo** button, so you don't need your own audio file (WAV/MP3 upload works too).

> [!NOTE]
> **Two pretrained models, everything else classical.** Vocal separation can use Meta AI's pretrained **Demucs** network, and lyrics transcription uses OpenAI's pretrained **Whisper**, both as ready-made tools (no training happens in this project). Everything else is classical signal processing and standard cryptography: NumPy, SciPy, librosa. It runs on a laptop; Apple-Silicon Macs use the GPU (MPS) automatically.

---

## Contents

- [What's inside](#whats-inside)
- [Quick start](#quick-start)
- [Features](#features)
  - [Analysis tabs](#analysis-tabs)
  - [Editing and production](#editing-and-production)
  - [ML vocal separation (Demucs)](#ml-vocal-separation-demucs)
  - [Lyrics & Transcription (Demucs → Whisper)](#lyrics--transcription-demucs--whisper)
  - [Secure Vault: audio ⇄ encrypted PNG](#secure-vault-audio--encrypted-png)
  - [Hidden Note: encrypted text inside a song](#hidden-note-encrypted-text-inside-a-song)
- [Live demo on your own Wi-Fi](#live-demo-on-your-own-wi-fi)
- [Project layout](#project-layout)
- [Sample audio](#sample-audio)
- [Notes and limitations](#notes-and-limitations)
- [Complete setup guide (Windows · macOS · Linux)](#complete-setup-guide)

---

## What's inside

| Part             | What it does                                                   | Built with                  |
| ---------------- | -------------------------------------------------------------- | --------------------------- |
| `audio_toolkit/` | All the DSP: pure functions, no UI code                        | NumPy, SciPy, librosa, Demucs |
| `vault/`         | Audio encryption + hiding it inside a PNG, and encrypted text notes hidden inside a song (steganography) | `cryptography`, Pillow      |
| `server/`        | HTTP API that exposes both of the above as JSON                | FastAPI + Uvicorn           |
| `web/`           | The user interface, with canvas-rendered charts                | React, TypeScript, Tailwind |
| `sample_data/`   | Ready-made demo WAV files                                      | —                           |
| `run.sh`         | Starts the API and the web app together (macOS/Linux)          | Bash                        |

The app has two halves that run at the same time:

- **API** at `http://localhost:8000` (Python)
- **Web app** at `http://localhost:5173` (Node). **This is the one you open in your browser.**

---

## Quick start

Already have Python 3.10+ and Node.js 20.19+ installed? Then:

**macOS / Linux**

```bash
git clone https://github.com/TJ-Paul/CSE220-Signal_Project-WaveLab.git
cd CSE220-Signal_Project-WaveLab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./run.sh
```

**Windows (PowerShell)**: use two terminals. See [Windows setup](#windows).

Then open **http://localhost:5173**.

New to any of this? Jump to the **[complete setup guide](#complete-setup-guide)** at the bottom.

---

## Features

### Analysis tabs

| #  | Tab                                | What you'll see                                                                                                                                                                                   |
| -- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1  | **File Info**                      | Duration, sample rate, channels, bit depth and bitrate. For MP3 the bitrate is estimated as file size ÷ duration, because libsndfile doesn't expose the encoder's bitrate.                        |
| 2  | **Waveform**                       | The time-domain signal. You can zoom into any range.                                                                                                                                              |
| 3  | **Sampling & Aliasing**            | Pick a tone and a sample rate on either side of Nyquist, then watch the reconstruction match the original or fold into an aliased "ghost" frequency. You can also down/upsample a file and measure what's lost. |
| 4  | **VAD** (Voice Activity Detection) | Splits the audio into 20–30 ms frames and labels each one *Speech* or *Silence* using short-time energy plus an FFT speech-band energy ratio.                                                      |
| 5  | **FFT Spectrum**                   | Frequency content of the whole signal or a single frame, shown next to its waveform.                                                                                                              |
| 6  | **Spectrogram**                    | STFT magnitude over time. Adjust the FFT size and hop length to see the time/frequency resolution trade-off.                                                                                      |
| 7  | **Filtering**                      | Butterworth low/high/band-pass/band-stop filters with a live frequency-response plot. Play or download the result.                                                                                |
| 8  | **Vocal / Instrumental Separation** | Classical nearest-neighbour spectral filtering (related to REPET-SIM), shown as an analysis of the method. For the best separation use **Karaoke & Vocals** with the ML engine.               |
| 9  | **Noise Reduction**                | Spectral subtraction. You select a noise-only region, tune oversubtraction and floor, and compare spectrograms before and after.                                                                  |
| 10 | **Signal Comparison**              | Pick any two signals from the session and compute MSE, SNR and correlation between them.                                                                                                          |

### Editing and production

Every result becomes a new signal in the session, so steps **chain**: trim a clip, fade it, then merge it with another.

- **Trim, Cut & Fade.** Drag across the waveform to keep or delete a region. Deleted gaps are crossfaded so the splice doesn't click. You can choose from five fade curves (linear, exponential, logarithmic, S-curve, equal-power), and each is plotted before it's applied.
- **Silence Remover.** Detects silence with a Schmitt trigger (separate on/off thresholds, so it can't flicker at the boundary). The threshold adapts to the recording's own noise floor. A minimum silence length and edge padding keep it from over-cutting.
- **Merge.** Joins clips in order with a crossfade or a gap. *Equal-power* crossfades suit unrelated sources, where a linear fade dips about 3 dB; *linear* suits two parts of the same take. Mixed sample rates are resampled automatically.
- **Speed & Pitch.** A phase vocoder changes duration without changing pitch. Resampling (varispeed) changes both together. You can also transpose by semitones at a fixed duration. **The resulting pitch shift is measured, not assumed** (see below).
- **Karaoke & Vocals.** Produces a backing track and/or an isolated vocal, level-matched to the source for fair A/B listening. Choose the engine: **ML (Demucs)**, the default, or **Classical DSP** (repetition masking). It reports vocal-band suppression, correlation between stems and energy share, plus SI-SDR when ground truth is available. See [ML vocal separation](#ml-vocal-separation-demucs).

<details>
<summary><b>How the pitch shift is verified</b></summary>

Transposing multiplies every frequency by the same ratio, which on a **log-frequency axis** is just a sideways slide. The app resamples the averaged spectrum onto a grid spaced in cents and cross-correlates the source against the result. The lag of the correlation peak, refined by parabolic interpolation, is the measured shift.

Because this matches the whole spectral envelope, it works on polyphonic music, speech and noise alike. On the bundled demos it's accurate to about 2 cents. For example, a 1.5× phase-vocoder stretch reports **+0.1 cents** (pitch preserved), while the same stretch by resampling reports **+700.2** (ideal: +702).

Fundamental frequency (YIN) is shown too, but only when the pitch is stable enough to mean something. Chords and dense mixes report *"no stable pitch"* instead of a misleading number.

</details>

### ML vocal separation (Demucs)

The classical method separates *repeating* from *non-repeating* content, which only matches "instruments vs voice" on loop-based music. For real songs, the **Karaoke & Vocals** tab can instead call **Demucs** (`htdemucs`, Meta AI), a published, pretrained source-separation network.

```
mix (any rate, mono/stereo) → resample to 44.1 kHz, stereo
      → pretrained Demucs → drums, bass, other, vocals
      → vocals               = a cappella stem
        drums + bass + other = karaoke stem
      → resample back → level match → metrics
```

**We use the model; we don't build it.** It's the same pattern as calling an FFT instead of deriving the DFT, or `scipy.signal.butter` instead of designing the filter by hand. No architecture design, dataset or training happens here. The whole ML part is one call in [`audio_toolkit/ml_separation.py`](audio_toolkit/ml_separation.py), equivalent to the CLI `demucs --two-stems=vocals song.wav`.

<details>
<summary><b>How it works, and how we measure it</b></summary>

**The model, in one paragraph.** Demucs v4 is a *hybrid* model. One branch reads the raw waveform, the other reads its **STFT spectrogram**. Each branch is a U-Net-style encoder–decoder, and a transformer in the middle lets them share information. It was trained on songs whose true stems were known, so it has learned what a voice looks like in time *and* frequency. The spectrogram branch runs on the same STFT/ISTFT covered in the Spectrogram tab.

**Evaluation: SI-SDR.** When the true stems are known, the app reports the **scale-invariant signal-to-distortion ratio**, the standard separation score (`metrics.si_sdr_db`):

```
alpha  = <estimate, true> / ||true||²      (best gain, so loudness doesn't count)
SI-SDR = 10 · log10( ||alpha·true||² / ||estimate − alpha·true||² )   dB, higher is better
```

**Measured result.** A spoken voice mixed over the demo accompaniment, with both stems known:

| Engine        | Vocals SI-SDR | Backing SI-SDR |
| ------------- | ------------- | -------------- |
| Classical DSP | −0.4 dB       | 14.8 dB        |
| ML (Demucs)   | **15.1 dB**   | **20.7 dB**    |

**When the classical method wins.** The built-in *song demo*'s "melody" is a synthesized tone, not a voice. Demucs was trained on real singing, so it correctly leaves that tone in the backing track, and the classical method scores higher on that one demo. Use a real song to demo the ML engine.

**Cost.** About 600 MB for PyTorch, plus about 85 MB of model weights downloaded on the first run and cached afterwards. It runs on the Apple-Silicon GPU (MPS) when available: on an M5, a 60 s song separates in about 4 s, against about 15 s on the CPU, with the same quality. Other machines use an NVIDIA GPU or the CPU automatically.

</details>

### Lyrics & Transcription (Demucs → Whisper)

Turns a song's vocals, or a plain voice recording, into **timestamped lyrics** in **English, Hindi or Bangla**. The lyric sheet follows playback word by word, any line can be clicked to jump there, and the result exports as **SRT** (subtitles), **LRC** (music players) or **TXT**.

**Already have the lyrics?** Choose **Use my lyrics** and paste them, or open a `.txt`/`.lrc` file. Your exact text is kept, and every word is timed against the audio (*forced alignment*); see [Syncing your own lyrics](#syncing-your-own-lyrics) below.

```
song → Demucs: isolate vocal stem ──┐        (skipped for "Voice recording")
                                    ▼
      resample to 16 kHz → Whisper (language fixed, word timestamps)
      → second pass on voiced stretches with no text   (short-time energy)
      → split long segments at the singer's breaths    (short-time energy)
      → drop lines over a near-silent vocal            (energy gate)
      → timestamped lines + per-word times
```

**Why separate first?** Whisper was trained mostly on speech, so drums and chords look like noise to it. On the isolated vocal it hears only the singer, and the words come out cleaner.

<details>
<summary><b>What's ours vs. what's the model, speed, and design choices</b></summary>

**The models (used, not built).** Whisper (`turbo` = large-v3-turbo) is an encoder–decoder transformer that reads an 80-band **log-mel spectrogram** in 30-second windows and writes text with timestamps. The code is [`audio_toolkit/transcription.py`](audio_toolkit/transcription.py).

**The signal processing around it (ours).** All three steps use the vocal stem's **short-time RMS energy**, the same idea as the Voice Activity tab:

- **Recovering skipped lines.** Whisper tends to merge repeated lines (a chorus sung twice in one window comes out once) and then skip ahead. Stretches where the stem is clearly voiced for 1 s or more, but no line covers them, are transcribed again on their own.
- **Line breaks at breaths.** Whisper's word times are contiguous: a breath gets absorbed into the neighbouring word. So each long segment is split where the stem is quietest between words, with punctuation (`.` `,` `।`) as a tie-breaker.
- **Energy gate.** Given near-silence, Whisper sometimes invents text ("Thank you."). A line whose vocal is more than 35 dB below the loud singing is dropped.

**Choosing the language** skips Whisper's language-detection pass and stops it guessing wrong on a sung intro.

**Speed on an Apple M5 (MPS).** A 60 s stereo song takes about 12 s in total (Demucs 4 s, Whisper 8 s), roughly 5× faster than real time. Loading both models takes a few seconds, once per server start. Transcribing the same song again in another language reuses the vocal stem and skips Demucs.

**Design choices that matter:**

| Setting | Why |
| --- | --- |
| `turbo`, not `small`/`medium` | `small` writes Bangla in the wrong script. `turbo` is near large-v3 quality at a fraction of its decoding cost. |
| `temperature=0` (greedy) | Whisper's default retries "too repetitive" windows with added randomness. Choruses repeat, so songs took 4× longer and gave different lyrics each run. |
| `condition_on_previous_text=False` | Stops the "repeat the last line forever" loop on music. |
| `word_timestamps=True` | Segment-level times are coarse (a line at 2.0 s was reported at 0.0 s). Word alignment brings lines to within about 0.3 s, and enables the word-by-word highlight. |
| Audio passed as an array | No ffmpeg and no temporary files: the app already has the decoded samples. |

**MPS note.** Whisper's word alignment converts one small matrix to float64, which MPS doesn't support. That single matrix is moved to the CPU before the step; the network stays on the GPU.

</details>

#### Syncing your own lyrics

Transcription asks Whisper *what* is sung, and inherits its spelling mistakes. With known lyrics the question becomes *when* each word is sung, which is much easier to answer precisely. Code: [`audio_toolkit/lyrics_align.py`](audio_toolkit/lyrics_align.py).

```
your lyrics ─┐
             ├─ 1. rough placement: match your words to Whisper's (like `diff`)
Whisper's ───┘       matched → Whisper's time · misheard/missed → share the gap
words              ↓
             2. forced alignment, ≤30 s windows: Whisper gets the audio AND your
                exact text; cross-attention + DTW give every word a time
                   ↓
             3. cross-check: words Whisper heard reliably must agree within 1 s
                   ↓
             4. trim each word to where the vocal actually sounds (short-time energy)
```

**Measured accuracy.** Test songs with lines placed at known times over music, in all three languages:

| Language | Whisper heard | Line-start error (mean / worst) | Confidence |
| -------- | ------------- | ------------------------------- | ---------- |
| English  | 100% of words | 23 ms / 45 ms                   | 0.97       |
| Hindi    | 70%           | 16 ms / 34 ms                   | 0.96       |
| Bangla   | 19%           | 15 ms / 31 ms                   | 0.86       |

Bangla shows why this matters: Whisper recognised only a fifth of the words, yet every line is timed to within about 30 ms, because alignment only needs to find *where* the known words are.

**Step 4 is DSP, and it's what makes it precise.** Alignment paths are continuous, so the silence before a line gets absorbed into its first word, and lines came out about 1 s early. Within each word's time slot the app finds the stretches where the vocal stem is sounding (short-time RMS). A line's first word is taken as the *last* sounding stretch in its slot, its last word as the *first*, and any other word as the *longest*. That brought the error from about 1 s down to tens of milliseconds, and it improved plain transcription timing too.

**Confidence** is the mean probability the model gives your words given the audio. Correct lyrics scored 0.81–0.97, and lyrics from another song 0.08–0.63, so below 0.7 the app warns that the lyrics may not match.

**Tips for best results:**

- Paste lines **in singing order, including every repeat** of the chorus. `[Chorus]`-style headers, blank lines and LRC timestamps are ignored.
- A sung line missing from your lyrics can pull one neighbouring line off by a few seconds. For English the cross-check usually catches this; for Hindi and Bangla, where Whisper hears fewer words, it may not.

### Secure Vault: audio ⇄ encrypted PNG

A separate subsystem (`vault/`) that **encrypts an audio file, hides it inside a PNG image**, and later recovers the original **byte for byte**.

```
Encode:  audio → SHA-256 → +metadata → AES-256-GCM → header+ciphertext
               → bitstream → LSB into RGB pixels → PNG

Decode:  PNG → LSB extraction → header → AES-256-GCM (tag verified)
             → metadata+file → SHA-256 compared → original file
```

**Three layers, three jobs:**

| Layer                                    | Provides            | Without it…                                                       |
| ---------------------------------------- | ------------------- | ----------------------------------------------------------------- |
| **Encryption** (AES-256-GCM)             | Confidentiality     | Anyone who knows the trick reads the hidden audio directly.       |
| **Steganography** (LSB)                  | Concealment         | The data is obviously a secret file, not an innocent image.       |
| **Authentication** (GCM tag + SHA-256)   | Integrity           | A flipped bit gives plausible-looking garbage instead of an error. |

<details>
<summary><b>Design details: encryption order, container, key derivation, capacity</b></summary>

**Encrypt first, then embed.** Even an attacker who extracts every hidden bit perfectly only gets ciphertext. The UI's byte histogram shows this: the source WAV has about 6.7 bits of entropy per byte with spikes at 0x00/0xFF, while the ciphertext sits at 7.999 out of 8, completely flat.

**Container format.** There is a 48-byte cleartext header (magic `SGVL`, version, scrypt params, salt, nonce, length) followed by the ciphertext. The filename, size and SHA-256 are kept *inside* the ciphertext, because each of them leaks information: a cleartext hash would let anyone confirm a guessed plaintext. The header is still authenticated as AES-GCM *associated data*, so tampering with it causes a clean failure.

**Key derivation.** It uses scrypt (memory-hard), not plain SHA-256 or PBKDF2. At N = 2¹⁵, r = 8 each password guess costs about 32 MB and tens of milliseconds, which is unnoticeable once and ruinous a billion times over. A fresh salt and nonce are drawn for every encryption.

**Capacity.**

```
capacity_bits  = width × height × 3 channels × bits_per_channel
capacity_bytes = capacity_bits / 8
```

The image is sized to fit the payload in a near-square shape. A 192 KB WAV becomes a 717 × 716 image at 99.9% utilisation. The generated cover is random noise on purpose: its low bits are already uniform, so embedding leaves no statistical trace for chi-squared or RS analysis to find. The trade-off is that the PNG is about 8× the audio size, and the UI reports this.

**Things to know:**

- **Use PNG, never JPEG.** JPEG compression changes pixel values slightly, which destroys a payload hidden in the lowest bit. (The in-app preview *is* a downscaled JPEG. It's for display only, and the UI says so.)
- The payload is **not compressed** before encryption. That would save almost nothing, and it would open the door to CRIME/BREACH-style length leaks.
- A failed decryption returns **nothing**: no partial output, no "best effort" audio.

</details>

**Self-test:** 25 checks (round trip, wrong password, flipped bit, edited header, capacity maths, entropy, key derivation). You can run them from the UI's **Security & integrity tests** panel or from the terminal:

```bash
python -m vault.selftest
```

### Hidden Note: encrypted text inside a song

The reverse of the vault: instead of hiding audio in an image, the **Hidden Note** tab hides an **encrypted text note inside the samples of a song**. The result is an ordinary 16-bit WAV that sounds identical to the original. Only someone with the password can tell a note is there, or read it. Notes can be written in English, Bangla or Hindi. Code: [`vault/audio_stego.py`](vault/audio_stego.py).

```
Hide:    text → AES-256-GCM (password) → header + ciphertext → bits
              → password-chosen sample positions → LSB matching (±1) → WAV

Reveal:  WAV → password-chosen positions → bits → header + ciphertext
             → AES-256-GCM (tag verified) → text
```

**Using the tab.** Load a song, then choose **Hide a note**: type the note and a password, and a live meter shows how much of the song's capacity it uses. The result plays next to the original, can be downloaded as a WAV, and shows a table of the samples that changed. **Read a note** takes that WAV (or the file just made in this session, with no download needed) plus the password, and prints the note. Reading needs no loaded signal.

**How it differs from the Secure Vault:**

|                        | Secure Vault                                  | Hidden Note                                                  |
| ---------------------- | --------------------------------------------- | ------------------------------------------------------------ |
| Hides                  | an audio file                                 | a text note                                                  |
| Inside                 | the low bits of a PNG's pixels                | the low bits of a song's 16-bit samples                      |
| Where the bits go      | from the first pixel, after a readable `SGVL` header | scattered over the whole song, at positions only the password reproduces |
| How a bit is written   | LSB replacement (overwrite the bit)           | LSB matching (nudge the sample by ±1 at random)               |
| Without the password   | you can see a container exists                | there is nothing to find                                     |

<details>
<summary><b>Design details: scattering, LSB matching, capacity, deniability</b></summary>

**Encryption.** The same `vault.crypto` code as the image vault: AES-256-GCM under a scrypt-stretched key, with a fresh salt and nonce for every note. A 32-byte header (salt, nonce, length) goes in front of the ciphertext and is authenticated as associated data.

**Scattering.** A second scrypt derivation of the password seeds a random number generator, which picks *which* samples carry the bits. The positions must be known before anything is read, so this derivation uses a fixed public salt; the encryption key still uses the random per-note salt. The sequence is prefix-stable, so the decoder reads the header's positions first, learns the length, then continues the same sequence for the ciphertext.

**LSB matching.** A 16-bit sample's lowest bit is 1/32768 of full scale, about **−90 dB**, far below hearing. Only the sample's *parity* carries the bit. About half the chosen samples already have the right parity and are left alone; the rest move by +1 or −1 at random. Simply overwriting the bit (LSB replacement) pairs up neighbouring values in the histogram, a known fingerprint that matching avoids. Samples at the ±32767 rails are always nudged inward, so nothing wraps.

**Capacity.** At most one sample in 8 carries a bit, so the changes stay sparse:

```
capacity_bytes = (total_samples / 8 − 384 header+tag bits) / 8      (capped at 32 KB)
```

An English letter is 1 byte in UTF-8, a Bangla or Hindi letter 3. The 6 s song demo (mono, 22.05 kHz) holds 2,019 bytes; a minute of CD-quality stereo reaches the 32 KB cap.

**Measured.** A 67-byte mixed English/Bangla note in the song demo changed **446 of 132,300 samples (0.34%)**, each by exactly 1 step, for a PSNR of **115 dB**.

**Deniability.** A wrong password, a file with no note and a damaged file all return the same answer: *"No hidden note was found with this password."* A wrong password reads bits from the wrong positions, and the GCM tag rejects them, so the decoder never admits a note exists unless it can also open it.

</details>

**What destroys the note:** anything that changes sample values, such as MP3/AAC encoding, resampling, normalising, trimming or editing. Lossless copies (the WAV itself, or a FLAC made from it) keep it. An MP3 is fine as the *starting* song, because the output is always a lossless WAV.

---

## Live demo on your own Wi-Fi

Let everyone in the room open the app on their phone, download an encrypted file you share, and decrypt it themselves. No internet is needed: the Mac is the server and a router with nothing plugged into its internet port is the network.

```bash
./run.sh --demo
```

This builds the web app, serves it on every network interface, keeps the Mac awake, and opens a **join page** with two QR codes (join the Wi-Fi, open the app) to put on the projector. For the Wi-Fi QR, fill in `.env.demo` (git-ignored) with the router's details:

```
WIFI_SSID=SignalLab
WIFI_PASSWORD=your-password
```

**During the demo:** encrypt a song in **Secure Vault** (or hide a note in **Hidden Note**) and press **Share with room**. The file appears within seconds on every phone under **Security → Shared Files**. Guests download it, then open it in Secure Vault → *Recover* or Hidden Note → *Read a note* with the password.

**Karaoke and Lyrics from phones:** the models run on the Mac's GPU; a phone only starts the job and waits for the result, so it doesn't matter how fast the phone is. Jobs take turns on the GPU, and each phone shows how many are ahead of it. Guests can lock the screen or switch apps and come back for the result. If several phones ask for the same song with the same settings, it is processed once and everyone gets that result, so later requests come back instantly.

**What guests can and can't do:** everyone can browse every tab, run tools and share files. Only the presenter's Mac can clear the session or remove shared files; the API itself listens on `127.0.0.1` and is reachable only through the web app. Plain `./run.sh` stays on this computer only, so development on a café or campus network is never exposed.

**Before the day**

- Run **Karaoke & Vocals** (ML) and **Lyrics & Transcription** once with internet, so both models are cached. Demo mode runs them offline, and the launcher warns if one is missing.
- If the macOS firewall is on, click **Allow** when asked about `node` (the launcher prints a command that approves it in advance).
- Rehearse once: join the router with the Mac and a phone, run `./run.sh --demo`, and do the full share → download → decrypt round trip on the phone.
- Router: DHCP on, **AP/client isolation off**. Keep the lid open (a closed lid sleeps the Mac).

**If a phone can't open the page:** it may be using mobile data because the Wi-Fi has no internet. On Android choose "Stay connected" when asked, or turn mobile data off briefly. On phones, pick downloaded files with *Browse → Choose File*, not from Photos: re-compressing an image or audio file erases the hidden bits.

---

## Project layout

```
audio_toolkit/
    io_utils.py          Load/save audio, file metadata
    framing.py           Split a signal into overlapping frames + windowing
    vad.py               Voice Activity Detection
    spectral.py          FFT, STFT, spectrogram
    filters.py           Butterworth filter design + application
    separation.py        Vocal/instrumental separation (classical)
    ml_separation.py     Vocal/instrumental separation (pretrained Demucs)
    transcription.py     Lyrics / voice transcription (Demucs → Whisper)
    lyrics_align.py      Sync known lyrics to the audio (forced alignment)
    noise_reduction.py   Spectral-subtraction denoising
    sampling.py          Sampling, aliasing, reconstruction demos
    metrics.py           MSE, SNR, correlation
    demo_signals.py      Synthetic demo signals
    editing.py           Trim, cut, splice, merge, fades, crossfades
    silence.py           Silence detection and removal
    timescale.py         Time stretch, pitch shift, pitch measurement
    vocals.py            Karaoke / a cappella stems (ML or classical engine)
    stereo_sep/          Classical stereo source separation (has its own README)

vault/
    container.py         On-image format: header, metadata, SHA-256
    crypto.py            scrypt key derivation + AES-256-GCM
    stego.py             Bit packing, capacity, LSB embed/extract
    pipeline.py          End-to-end encode/decode + metrics
    audio_stego.py       Encrypted text notes hidden in a song's samples
    selftest.py          25 security and integrity checks

server/
    main.py              FastAPI routes
    store.py             In-memory store for session signals
    vault_store.py       In-memory store for vault files
    shared_store.py      Files shared with the room (on disk, in shared_files/)
    lan.py               Tells the presenter's Mac from guests on the network
    jobs.py              Background queue for Karaoke and Lyrics: one GPU lane, results reused

scripts/
    lan_demo.py          Demo helper: finds the LAN address, QR codes, join page

web/                     React + Vite frontend (src/views, src/components, src/lib)
sample_data/             Demo WAV files
requirements.txt         Python dependencies
run.sh                   One-command launcher (macOS/Linux); --demo for the LAN demo
```

None of the Python modules import a UI framework, so the DSP code can be reused or unit-tested on its own.

---

## Sample audio

`sample_data/` contains four ready-made WAV files generated by `audio_toolkit/demo_signals.py`:

- `speech_like_demo.wav`
- `noisy_tone_demo.wav` + `noisy_tone_demo_clean_reference.wav`
- `song_like_demo_mixture.wav`

Use them to try the file-upload path. To regenerate one:

```bash
python -c "
from audio_toolkit import demo_signals, io_utils
y, sr = demo_signals.generate_speech_like_demo(duration=6.0)
io_utils.save_audio('sample_data/speech_like_demo.wav', y, sr)
"
```

---

## Notes and limitations

- **One pretrained model, no training.** Only the ML engine in Karaoke & Vocals uses a neural network, and only for inference. Denoising, the phase vocoder, the classical separation and the vault use no models. Denoising quality is below modern neural tools.
- **Transcription accuracy depends on the singing.** Clear lead vocals transcribe well. Heavy effects, rap at high speed, or dense backing vocals cause mistakes. Whisper's Bangla is noticeably weaker than its English or Hindi.
- **Demucs is optional.** If `pip install demucs` fails (for example on an unusual Python version), everything else still works and Karaoke & Vocals can still use the Classical DSP engine.
- **No homemade crypto.** AES-256-GCM comes from the `cryptography` library, and scrypt/SHA-256 from Python's standard library.
- **Phase vocoder artefacts.** Big stretches smear transients and can add faint chorusing. 0.8×–1.25× sounds nearly transparent, while 0.5× or 2× is audible. Resampling has no such artefacts, but it changes pitch too.
- **Export is always WAV.** MP3 files can be read (via `soundfile`/`audioread`), but processed audio is saved as lossless WAV, so no external encoder is needed.

---

## Complete setup guide

Step-by-step instructions for a fresh machine, from cloning the repo to seeing the app in your browser.

### What you need

| Tool        | Minimum version | Check with          |
| ----------- | --------------- | ------------------- |
| **Git**     | any recent      | `git --version`     |
| **Python**  | **3.10** or newer | `python --version` (Windows) / `python3 --version` (macOS/Linux) |
| **Node.js** | **20.19** or newer (22 LTS recommended) | `node --version` |
| **npm**     | comes with Node | `npm --version`     |

You'll run these commands in a terminal: **PowerShell** on Windows, **Terminal** on macOS, or any terminal on Linux.

---

### Windows

#### Step 1: Install the tools

1. **Git**: download from <https://git-scm.com/download/win> and run the installer with the default options.
2. **Python**: download from <https://www.python.org/downloads/>.

   > **Important:** on the first installer screen, **tick "Add python.exe to PATH"** before clicking *Install Now*.
3. **Node.js**: download the **LTS** installer from <https://nodejs.org/> and run it with the default options.
4. **Close and reopen PowerShell** so it picks up the new tools, then check them:

   ```powershell
   git --version
   python --version
   node --version
   npm --version
   ```

   All four should print a version number.

#### Step 2: Clone the repository

```powershell
cd $HOME\Documents
git clone https://github.com/TJ-Paul/CSE220-Signal_Project-WaveLab.git
cd CSE220-Signal_Project-WaveLab
```

#### Step 3: Create a Python virtual environment

A virtual environment keeps this project's packages separate from the rest of your system.

```powershell
python -m venv .venv
```

#### Step 4: Activate it

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

> [!WARNING]
> **Got a red "running scripts is disabled on this system" error?** Run this once, answer **Y**, then try activating again:
>
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
>
> Using **Command Prompt** (cmd) instead of PowerShell? Activate with `.venv\Scripts\activate.bat`.

#### Step 5: Install the Python packages

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This takes a few minutes: it includes PyTorch (about 600 MB) for the ML vocal separation and transcription.

#### Step 6: Install the web app's packages

```powershell
cd web
npm install
cd ..
```

#### Step 7: Start the app (two terminals)

`run.sh` is a Bash script and won't run in PowerShell, so start the two halves separately.

**Terminal 1: API.** From the project folder, with `(.venv)` active:

```powershell
python -m uvicorn server.main:app --port 8000 --reload
```

Wait for `Application startup complete.`

**Terminal 2: web app.** Open a **new** PowerShell window:

```powershell
cd $HOME\Documents\CSE220-Signal_Project-WaveLab\web
npm run dev
```

#### Step 8: Open it

Go to **<http://localhost:5173>** in your browser.

To stop the app, press **Ctrl + C** in both terminals.

> [!TIP]
> If you have **Git Bash** (installed with Git) or **WSL**, `./run.sh` works there too. In Git Bash, activate with `source .venv/Scripts/activate`.

---

### macOS

#### Step 1: Install the tools

The easiest way is with [Homebrew](https://brew.sh/).

1. Install Homebrew if you don't have it (paste into Terminal and follow the prompts):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Git, Python and Node:

   ```bash
   brew install git python node
   ```

   *(Prefer installers? Get Python from <https://www.python.org/downloads/> and Node LTS from <https://nodejs.org/> instead.)*

3. Check them:

   ```bash
   git --version
   python3 --version
   node --version
   npm --version
   ```

#### Step 2: Clone the repository

```bash
cd ~/Documents
git clone https://github.com/TJ-Paul/CSE220-Signal_Project-WaveLab.git
cd CSE220-Signal_Project-WaveLab
```

#### Step 3: Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Your prompt should now start with `(.venv)`.

#### Step 4: Install the Python packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 5: Start the app

```bash
chmod +x run.sh     # only needed the first time
./run.sh
```

On the first run it installs the web app's npm packages automatically, which takes a minute. When you see the **Signal Lab** banner, it's ready.

#### Step 6: Open it

Go to **<http://localhost:5173>**.

Press **Ctrl + C** once to stop both halves.

---

### Linux

These commands are for **Ubuntu/Debian**. Other distros are listed after Step 1.

#### Step 1: Install the tools

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip curl lsof
```

Ubuntu's own `nodejs` package is often too old for this project, so install Node LTS from NodeSource:

```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
```

*(Or use [nvm](https://github.com/nvm-sh/nvm): `nvm install --lts`.)*

<details>
<summary>Fedora / Arch</summary>

```bash
# Fedora
sudo dnf install -y git python3 python3-pip nodejs npm lsof

# Arch
sudo pacman -S --needed git python python-pip nodejs npm lsof
```

</details>

Check everything:

```bash
git --version
python3 --version
node --version
npm --version
```

#### Step 2: Clone the repository

```bash
cd ~
git clone https://github.com/TJ-Paul/CSE220-Signal_Project-WaveLab.git
cd CSE220-Signal_Project-WaveLab
```

#### Step 3: Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 4: Install the Python packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 5: Start the app

```bash
chmod +x run.sh     # only needed the first time
./run.sh
```

#### Step 6: Open it

Go to **<http://localhost:5173>**.

Press **Ctrl + C** to stop.

---

### Running it again later

You only need to install once. Next time:

| OS                | Commands (from the project folder)                                                                                                             |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **macOS / Linux** | `source .venv/bin/activate` then `./run.sh`                                                                                                     |
| **Windows**       | Terminal 1: `.venv\Scripts\Activate.ps1` then `python -m uvicorn server.main:app --port 8000 --reload`<br>Terminal 2: `cd web` then `npm run dev` |

### Check that it's working

- **<http://localhost:8000/api/health>** should respond, which means the API is up.
- **<http://localhost:5173>** should show the app. Click any tab's **demo** button.
- Optional: `python -m vault.selftest` should report all 25 checks passing.

### Troubleshooting

| Problem                                                                   | Fix                                                                                                                                                                         |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python` / `python3` / `node` **not found**                               | The tool isn't installed or isn't on PATH. Reinstall it (on Windows, tick *Add to PATH*), then **open a new terminal**.                                                  |
| `error: .venv not found` from `run.sh`                                    | You skipped creating the virtual environment. Run Step 3, then try again.                                                                                                   |
| `ModuleNotFoundError: No module named 'fastapi'` (or numpy, librosa…)     | The virtual environment isn't active. Activate it (`(.venv)` should appear in your prompt) and re-run `pip install -r requirements.txt`.                                  |
| **The page loads but every action fails** / "network error"               | The API isn't running. Make sure the terminal with `uvicorn` is still open and shows no errors.                                                                             |
| `Address already in use` on port 8000 or 5173                             | Something else is using the port. On macOS/Linux, `run.sh` frees it automatically. On Windows, close the old terminal, or find the process with `netstat -ano \| findstr :8000` and end it in Task Manager. |
| `Permission denied: ./run.sh`                                             | Run `chmod +x run.sh` once.                                                                                                                                                 |
| `npm install` fails or Vite complains about the Node version              | Your Node.js is too old. Install the current **LTS** (20.19+ required).                                                                                                    |
| `ensurepip is not available` (Linux)                                      | Install the venv package: `sudo apt install python3-venv`, then delete `.venv` and create it again.                                                                         |
| `OSError: sndfile library not found` (Linux, uncommon)                    | `sudo apt install libsndfile1`                                                                                                                                              |
| Lyrics & Transcription says **"Transcription is not installed"**         | Run `pip install openai-whisper demucs` with `(.venv)` active, then restart the API. The first run downloads the Whisper `turbo` model (~1.6 GB). |
| Model download fails with **`CERTIFICATE_VERIFY_FAILED`** (macOS)        | Python can't verify HTTPS certificates. With the python.org installer, run *Install Certificates.command* from `/Applications/Python 3.x/`. Or download the model file yourself: `curl -L -o ~/.cache/whisper/large-v3-turbo.pt <URL>`, where the URL comes from `python -c "import whisper; print(whisper._MODELS['turbo'])"`. |
| Karaoke & Vocals says **"The ML engine needs Demucs"**                   | Run `pip install demucs` with `(.venv)` active, then restart the API. The first ML run also needs internet to download the model (~85 MB). |
| Installing a package fails with a compiler error                          | Your Python is probably too new or too old for a prebuilt wheel. Use Python **3.11–3.13**, recreate `.venv`, and reinstall.                                               |
