# Classical Stereo Source Separation

Splits a stereo mixture into a **karaoke/instrumental** estimate and a
**vocal-focus** estimate using only mathematics and signal processing.
No neural networks, no learned models, no training data, no pretrained
weights, no cloud APIs. Dependencies are NumPy, SciPy and librosa, and
librosa is used only for its FFT framing, median filtering and
nearest-neighbour utilities — not for any learned component.

---

## 0. The problem is ill-posed, and that is not a detail

Per time-frequency bin the mixture gives

$$X(k,m) = V(k,m) + I(k,m)$$

— **one complex equation in two complex unknowns.** The system is
underdetermined. $V$ and $I$ are *not* determined by $X$, and no algorithm
recovers them exactly from an already-mixed recording. Anything that claims
otherwise is either using the unmixed multitracks or overstating.

Every method here therefore adds **assumptions** that constrain the
solution. The assumptions are the algorithm. They are listed per method and
returned in the API response, because a result is only meaningful alongside
the conditions under which it holds.

Accordingly, this codebase never says "this bin is vocal". It says a bin has
higher estimated vocal energy *under the stated assumptions*.

---

## 1. Mixture model

$$L(t) = V_L(t) + I_L(t), \qquad R(t) = V_R(t) + I_R(t)$$

$V$ is the vocal contribution, $I$ the instrumental. These overlap in both
time and frequency — a singer and a guitar occupy the same bins — which is
precisely why filtering cannot separate them and why masking is a matter of
apportioning shared bins rather than routing them.

## 2. Mid/Side

$$M(t) = \tfrac{1}{2}\big(L(t)+R(t)\big), \qquad S(t) = \tfrac{1}{2}\big(L(t)-R(t)\big)$$

with exact inverse $L = M+S$, $R = M-S$. This is a 45° rotation of the
$(L,R)$ plane — lossless and perfectly invertible.

For a centre-panned source ($V_L = V_R = V$):

$$M = V + \tfrac{1}{2}(I_L+I_R), \qquad S = \tfrac{1}{2}(I_L-I_R)$$

**The vocal cancels exactly in $S$.** Not approximately — by algebra. So $S$
is a vocal-free reference obtained with no estimation at all, which is an
unusually strong starting point.

But note carefully:

| Claim | Status |
|---|---|
| $M$ = vocal | **False.** $M$ holds every centred source: kick, bass, snare, often lead guitar. |
| $S$ = instrumental | **False.** $S$ holds only *panned* content; it has no centred accompaniment and sounds thin and bass-less alone. |

Both are spatial **estimates**, used as evidence.

## 3. STFT

$$X(k,m)=\sum_{n=0}^{N-1} x[n+mH]\,w[n]\,e^{-j2\pi kn/N}$$

Defaults: $N = 4096$, $H = 1024$ (75 % overlap), Hann.

**Why $N=4096$.** At 44.1 kHz that is $\Delta f = 10.8$ Hz and a 93 ms
window. Separation decisions need individual harmonics resolved: a singer at
$f_0 = 110$ Hz has partials 110 Hz apart, so the 21.5 Hz bins of an $N=2048$
analysis smear a low male voice's first partials into the bass. The cost is
93 ms of transient smearing, which is why **percussive evidence is computed
at a separate, shorter window** ($N=1024$) — a snare needs time resolution,
and it needs no pitch resolution at all.

This is the Gabor limit: $\Delta t \cdot \Delta f \ge 1/4\pi$. You cannot
have both, so the pipeline uses two analyses rather than compromising on one.

**Reconstruction is verified before any separation runs** — `verify_reconstruction()`
measures ISTFT(STFT(x)) against x and reports the SNR. It reaches **141 dB**
(float32 rounding). This is required, not decorative: if the transform pair
were inexact, every stem would carry that error and it would be
indistinguishable by ear from a separation failure.

## 4. Evidence features

Each returns a score in $[0,1]$ per bin. **None is a vocal detector.** Each
tests one narrow hypothesis and each is wrong in a *different* way, which is
what makes combining them worthwhile.

| Feature | Equation | Assumes | Fails when |
|---|---|---|---|
| **Spatial** | $C=\frac{\lvert M\rvert}{\lvert M\rvert+\lvert S\rvert}$ | lead is centred | bass/kick/snare are centred too; vocal is widened or doubled; mono source |
| **Coherence** | $\gamma=\frac{\lvert\langle L\bar R\rangle\rvert}{\sqrt{\langle\lvert L\rvert^2\rangle\langle\lvert R\rvert^2\rangle}}$ | dry centred sources are coherent; reverb is not | heavily processed or mono-summed reverb |
| **HPSS** | median filter along time → $H$; along frequency → $P$ | vocals harmonic, drums percussive | guitar/piano/strings are harmonic too; consonants are percussive |
| **Harmonic comb** | $\mathrm{HSS}(f_0)=\sum_h \lvert X(hf_0)\rvert/h$ | voiced singing has a resolvable $f_0$ | unvoiced consonants have none; octave errors; polyphony |
| **Repetition** | $B=\mathrm{median}_{j\in N(m)}\lvert X(k,j)\rvert$ | accompaniment repeats, lead does not | rubato/live/ambient backing; repeated choruses |
| **Spectral** | formant-region profile × tonality | voices concentrate near F1≈500 Hz, F2≈1500 Hz | guitars and synths live there too |

**Coherence needs smoothing to mean anything.** A single-frame coherence is
identically 1 by construction — any two complex numbers are "perfectly
coherent" alone — so it is averaged over 9 frames (~0.2 s).

**The bass guard.** Below 120 Hz the spatial score is forced to a neutral
0.5. Mixes are mono in the low end by engineering convention, so "centred"
there carries no information — it just means "bass". Treating centred low
frequencies as vocal is the single most common way a karaoke track loses its
bass line.

## 5. Power estimation and the Wiener partition

$$A_i = \text{accompaniment magnitude estimate (REPET background, primary)}$$
$$A_v = \underbrace{\lambda}_{\text{margin}} \cdot \max(\lvert M\rvert - A_i,\ 0)$$
$$M_v = \frac{A_v^{\,p}}{A_v^{\,p}+A_i^{\,p}+\varepsilon}, \qquad M_i = 1 - M_v$$

With $p=2$ this is the classical Wiener filter — the MMSE estimator under
the assumption that $V$ and $I$ are uncorrelated within a bin. The masks
**partition** the mixture ($M_v + M_i = 1$ exactly), so no energy is lost or
duplicated.

**Why the margin $\lambda$ exists.** The residual $\lvert M\rvert - A_i$
systematically *understates* vocal energy, because magnitudes add
vectorially:

$$\lvert X\rvert = \sqrt{\lvert I\rvert^2+\lvert V\rvert^2+2\lvert I\rvert\lvert V\rvert\cos\theta} \;\le\; \lvert I\rvert+\lvert V\rvert$$

with equality only when exactly in phase. So subtracting an accompaniment
magnitude leaves less than the true vocal magnitude almost everywhere.
$\lambda$ is a **bias correction**, not a free gain. Omitting it was a
measured defect — the mask degenerated to "keep everything".

$\lambda$ is scaled down when no repetition estimate exists, because the
margin corrects the bias of a *subtraction*, and where nothing was
subtracted there is no bias to correct.

## 6. Two pipelines, deliberately asymmetric

The two outputs are **not complements of each other**, because their failure
modes are not symmetric.

**Karaoke** — fails safe toward keeping music:
$$M_i = \max\Big(1 - \sigma_v\,(1-M_i^{\text{base}}),\ \ \mu,\ \ \text{drum guard}\Big)$$
A residual vocal is mildly annoying; a missing snare or bass ruins the
track, because there is nothing left to sing to. `music_preservation` ($\mu$)
is a hard floor no evidence can override.

**Vocal focus** — fails safe toward keeping voice:
$$M_v = \alpha\,M_v^{\text{base}} + (1-\alpha), \quad \text{lifted by harmonic and consonant guards}$$
Accompaniment bleed is tolerable; a thin, telephone-like, consonant-less
voice is useless.

**Both guards are gated by vocal confidence**, and that gate is not optional.
Harmonicity alone does not indicate voice — an ungated harmonic guard puts a
protective floor under every pitched instrument and the "vocal" stem becomes
the whole song at reduced level. Likewise an ungated transient guard
protects every drum hit.

## 7. Reconstruction

**Instrumental:**
$$L_{\text{out}} = M_i\!\cdot\! \text{Mid} + \text{Side}, \qquad R_{\text{out}} = M_i\!\cdot\!\text{Mid} - \text{Side}$$

Side passes through **unattenuated** — the most important structural choice
here. Side is provably vocal-free, so attenuating it can only remove
accompaniment. It is pure loss. This also preserves the stereo image, since
Side carries the width.

**Vocal:** $L_{\text{out}} = R_{\text{out}} = M_v\cdot\text{Mid}$, emitted
dual-mono. A source estimated purely from the centre *has* no width;
synthesising some would be inventing information the mixture does not contain.

Mixture phase is preserved throughout: $Y = M\!\cdot\!X = M\lvert X\rvert e^{j\phi}$.
The true phase of an isolated source is unknowable from the mixture, and the
mixture phase is its MMSE estimate wherever one source dominates — which is
exactly where the mask is near 1 and phase matters most.

Processing stays in float; conversion to 16-bit PCM happens only at file
write.

## 8. Mask smoothing

Applied across frequency (2 bins) and time (3 frames). Encodes two physical
priors: a source does not appear and vanish within 20 ms, and a harmonic
partial spans several adjacent bins.

**Measured cost: a (3, 5) kernel cost 2.1 dB of instrumental SDR** — the
largest single loss in the pipeline. It is kept anyway, at reduced size,
because this is a case where the metric and the ear disagree: SDR penalises
the blur, while the flicker it removes is heard as musical noise. *Do not
tune this parameter on SDR alone.*

A mask floor of 0.03 (−30 dB) prevents spectral holes. A bin of total
silence surrounded by energy is detected by the ear as a *hole*; heavily
attenuated content merely sounds quiet.

## 9. Stereo reverb — the honest limit

A lead vocal may be centred while its reverb is stereo. Then $V_L \neq V_R$,
cancellation in $S$ is only partial, and the residual $(V_L-V_R)/2$ remains.

**No L/R method can remove it.** The characteristic artefact is a karaoke
track with no dry voice but an audible ghost singing the melody. Doubled and
harmonised vocals behave the same way. This is a property of the mixture, not
a bug in the implementation, and no parameter fixes it.

---

## Measured results

Two synthetic stereo mixes with exact ground truth. **These are synthetic
signals, not real music** — treat them as a regression harness and a sanity
check, not as a prediction of performance on commercial recordings.

**Mix A** — loop-based backing, centred vocal (*assumptions hold*):

| method | iSDR | iSIR | iSAR | vSDR | vSIR | vSAR |
|---|---|---|---|---|---|---|
| do nothing | 6.53 | 6.53 | — | −6.47 | −6.47 | — |
| old `separation.py` (mono) | **14.05** | **20.73** | 15.14 | −2.34 | 7.68 | −1.20 |
| m1 L−R cancellation | −11.00 | 34.17 | −11.00 | −6.47 | −6.47 | — |
| m2 Mid/Side | 11.97 | 22.20 | 12.43 | 3.18 | 7.00 | 6.29 |
| m3 + STFT masking | 11.70 | 12.28 | **20.97** | 0.53 | 0.79 | 15.45 |
| m4 + HPSS | 10.57 | 11.57 | 17.75 | −2.99 | −2.88 | **17.90** |
| **m5 full pipeline** | 13.02 | 14.69 | 18.11 | **6.05** | **9.76** | 8.89 |

**Mix B** — through-composed backing, doubled/wide vocal (*assumptions break*):

| method | iSDR | vSDR |
|---|---|---|
| do nothing | 11.35 | −11.35 |
| old `separation.py` (mono) | **17.00** | −3.01 |
| m2 Mid/Side | 15.89 | 2.78 |
| **m5 full pipeline** | 11.77 | −6.53 |

### Reading these honestly

- **m5 wins decisively on the vocal stem** where its assumptions hold
  (6.05 dB vs the old code's −2.34, and vSAR 8.89 vs −1.20 — the old vocal
  stem is artefact-dominated, which is the "warbly a cappella" symptom).
- **The old mono REPET implementation is still competitive, and better on
  instrumental SIR.** It removes more vocal; m5 introduces fewer artefacts
  (higher SAR) and preserves stereo. Neither dominates. m5's advantage is
  the vocal stem and the stereo image, not universal superiority.
- **m2 is a strong, very cheap karaoke baseline** and beats m5 on Mix B.
  Do not skip it.
- **On Mix B every method degrades**, because the vocal is doubled (breaking
  the centre assumption) *and* the backing is through-composed (breaking the
  repetition assumption). Both pillars removed at once. This is the expected
  and documented behaviour, not a failure to fix.

`m1` behaves exactly as the textbook predicts: superb interference rejection
(iSIR 34 dB — it really does cancel the vocal) and catastrophic everything
else (iSAR −11 dB), because it also cancels the centred bass and drums and
collapses the output to mono. It is retained precisely to demonstrate that.

---

## Usage

```python
from audio_toolkit import stereo_sep

y, sr = stereo_sep.load_stereo("song.wav")          # keeps L/R — do not downmix
report = stereo_sep.analyse_stereo(y, sr)           # check before trusting spatial features
if report.is_effectively_mono:
    print("No Side signal — spatial evidence unavailable.")

cfg = stereo_sep.SeparationConfig()
cfg.output.vocal_suppression = 0.85                 # karaoke aggressiveness
cfg.output.music_preservation = 0.10                # hard floor on the backing
cfg.debug = True

result = stereo_sep.separate(y, sr, method="m5", cfg=cfg)
print(result.reconstruction)                        # round-trip SNR, verified
print(result.assumptions)                           # what this result rests on

debug = stereo_sep.debug_report(result, y, cfg)     # spectrograms, masks, measurements
```

A/B every method:

```python
from audio_toolkit.stereo_sep.compare import compare_methods, format_table
print(format_table(compare_methods(y, sr, true_instrumental=inst, true_vocals=voc)))
```

### HTTP

| endpoint | purpose |
|---|---|
| `GET /api/separation/methods` | method catalogue, default config, listening guide |
| `POST /api/signals/{id}/stereo-separate` | run one method; `{method, debug, config}` |
| `POST /api/signals/{id}/stereo-compare` | run all five for A/B |

Mono signals are **refused** with an explanation rather than silently given a
worse result from a method the caller did not request.

## Parameters that matter most

| parameter | default | raising it |
|---|---|---|
| `vocal_suppression` | 0.85 | removes more voice **and** more centred instruments |
| `music_preservation` | 0.10 | hard floor on the backing; raise if bass/snare vanish |
| `accompaniment_suppression` | 0.92 | cleaner vocal, more artefacts |
| `vocal_preservation` | 0.15 | protects consonants and partials; raise if vocals sound mumbled |
| `vocal_margin` | 4.0 | bias correction, useful range 2–6 |
| `mask_power` | 2.0 | sharper decisions, more musical noise |
| `smooth_time_frames` | 3 | less musical noise, smeared consonants and pre-echo |
| `bass_protect_hz` | 120 | protects more low end from vocal attribution |
| `n_fft` | 4096 | finer harmonics, smeared transients |

## Judging the output

Numbers cannot settle this; two files with identical SDR can sound
completely different. `stereo_sep.LISTENING_GUIDE` gives the full protocol.
The short version:

- **Karaoke:** is the bass still there? does the snare still crack? Warbling
  between notes means the masks are too aggressive.
- **Vocal:** are consonants intact ("t", "k", "s")? Mumbling means the mask
  is deleting broadband transients — raise `vocal_preservation`.
- Compare m1 against m5 **at matched level**. If m5 is not clearly better,
  the material likely breaks its assumptions, and the honest answer is that
  classical DSP has little to work with on that track.
