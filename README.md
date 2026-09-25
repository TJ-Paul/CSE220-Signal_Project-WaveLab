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
> **No machine learning anywhere.** Everything is classical signal processing and standard cryptography: NumPy, SciPy, librosa. It runs on any laptop with no GPU and no model downloads.

---

## Contents

- [What's inside](#whats-inside)
- [Quick start](#quick-start)
- [Features](#features)
  - [Analysis tabs](#analysis-tabs)
  - [Editing and production](#editing-and-production)
  - [Secure Vault: audio ⇄ encrypted PNG](#secure-vault-audio--encrypted-png)
- [Project layout](#project-layout)
- [Sample audio](#sample-audio)
- [Notes and limitations](#notes-and-limitations)
- [Complete setup guide (Windows · macOS · Linux)](#complete-setup-guide)

---

## What's inside

| Part             | What it does                                                   | Built with                  |
| ---------------- | -------------------------------------------------------------- | --------------------------- |
| `audio_toolkit/` | All the DSP: pure functions, no UI code                        | NumPy, SciPy, librosa       |
| `vault/`         | Audio encryption + hiding it inside a PNG (steganography)      | `cryptography`, Pillow      |
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
| 8  | **Vocal / Instrumental Separation** | Classical nearest-neighbour spectral filtering (related to REPET-SIM). It won't match deep-learning tools like Demucs, and the in-app **"Method & limitations"** panel explains why.               |
| 9  | **Noise Reduction**                | Spectral subtraction. You select a noise-only region, tune oversubtraction and floor, and compare spectrograms before and after.                                                                  |
| 10 | **Signal Comparison**              | Pick any two signals from the session and compute MSE, SNR and correlation between them.                                                                                                          |

### Editing and production

Every result becomes a new signal in the session, so steps **chain**: trim a clip, fade it, then merge it with another.

- **Trim, Cut & Fade.** Drag across the waveform to keep or delete a region. Deleted gaps are crossfaded so the splice doesn't click. You can choose from five fade curves (linear, exponential, logarithmic, S-curve, equal-power), and each is plotted before it's applied.
- **Silence Remover.** Detects silence with a Schmitt trigger (separate on/off thresholds, so it can't flicker at the boundary). The threshold adapts to the recording's own noise floor. A minimum silence length and edge padding keep it from over-cutting.
- **Merge.** Joins clips in order with a crossfade or a gap. *Equal-power* crossfades suit unrelated sources, where a linear fade dips about 3 dB; *linear* suits two parts of the same take. Mixed sample rates are resampled automatically.
- **Speed & Pitch.** A phase vocoder changes duration without changing pitch. Resampling (varispeed) changes both together. You can also transpose by semitones at a fixed duration. **The resulting pitch shift is measured, not assumed** (see below).
- **Karaoke & Vocals.** Produces a backing track and/or an isolated vocal, level-matched to the source for fair A/B listening. It reports vocal-band suppression, correlation between stems and energy share.

<details>
<summary><b>How the pitch shift is verified</b></summary>

Transposing multiplies every frequency by the same ratio, which on a **log-frequency axis** is just a sideways slide. The app resamples the averaged spectrum onto a grid spaced in cents and cross-correlates the source against the result. The lag of the correlation peak, refined by parabolic interpolation, is the measured shift.

Because this matches the whole spectral envelope, it works on polyphonic music, speech and noise alike. On the bundled demos it's accurate to about 2 cents. For example, a 1.5× phase-vocoder stretch reports **+0.1 cents** (pitch preserved), while the same stretch by resampling reports **+700.2** (ideal: +702).

Fundamental frequency (YIN) is shown too, but only when the pitch is stable enough to mean something. Chords and dense mixes report *"no stable pitch"* instead of a misleading number.

</details>

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

---

## Project layout

```
audio_toolkit/
    io_utils.py          Load/save audio, file metadata
    framing.py           Split a signal into overlapping frames + windowing
    vad.py               Voice Activity Detection
    spectral.py          FFT, STFT, spectrogram
    filters.py           Butterworth filter design + application
    separation.py        Vocal/instrumental separation
    noise_reduction.py   Spectral-subtraction denoising
    sampling.py          Sampling, aliasing, reconstruction demos
    metrics.py           MSE, SNR, correlation
    demo_signals.py      Synthetic demo signals
    editing.py           Trim, cut, splice, merge, fades, crossfades
    silence.py           Silence detection and removal
    timescale.py         Time stretch, pitch shift, pitch measurement
    vocals.py            Karaoke / a cappella stems
    stereo_sep/          Classical stereo source separation (has its own README)

vault/
    container.py         On-image format: header, metadata, SHA-256
    crypto.py            scrypt key derivation + AES-256-GCM
    stego.py             Bit packing, capacity, LSB embed/extract
    pipeline.py          End-to-end encode/decode + metrics
    selftest.py          25 security and integrity checks

server/
    main.py              FastAPI routes
    store.py             In-memory store for session signals
    vault_store.py       In-memory store for vault files

web/                     React + Vite frontend (src/views, src/components, src/lib)
sample_data/             Demo WAV files
requirements.txt         Python dependencies
run.sh                   One-command launcher (macOS/Linux)
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

- **Classical methods only.** Separation, denoising, the phase vocoder and the vault use no models, training or inference. It runs anywhere, but separation and denoising quality is below modern neural tools.
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

This takes a minute or two.

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
| Installing a package fails with a compiler error                          | Your Python is probably too new or too old for a prebuilt wheel. Use Python **3.11–3.13**, recreate `.venv`, and reinstall.                                               |
