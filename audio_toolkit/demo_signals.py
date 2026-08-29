"""Synthetic demo signals so every tab can be demonstrated instantly,
without requiring the presenter to have a real audio file on hand.
Real WAV/MP3 upload is still fully supported everywhere in the app.
"""
from __future__ import annotations

import numpy as np


def generate_speech_like_demo(sr: int = 16000, duration: float = 6.0, seed: int = 0):
    """Alternating "voiced" bursts (sum of formant-like tones, amplitude
    modulated) and silence, plus a low background noise floor — built to
    exercise the VAD tab in a visually obvious way."""
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(n) / sr
    y = np.zeros(n)

    # Speech-like bursts roughly every ~1s, each ~0.4-0.7s long.
    pos = 0.3
    formants = [270, 870, 2400]  # rough vowel-like formant frequencies
    while pos < duration - 0.3:
        seg_len = rng.uniform(0.4, 0.7)
        start_i, end_i = int(pos * sr), int(min(duration, pos + seg_len) * sr)
        seg_t = t[start_i:end_i]
        envelope = np.hanning(len(seg_t))
        tone = sum(np.sin(2 * np.pi * f * seg_t) for f in formants)
        y[start_i:end_i] += 0.25 * envelope * tone / len(formants)
        pos += seg_len + rng.uniform(0.4, 0.9)

    y += rng.normal(0, 0.01, n)  # background noise floor
    return y.astype(np.float32), sr


def generate_noisy_tone_demo(sr: int = 16000, duration: float = 3.0, freq: float = 440.0,
                              noise_level: float = 0.15, silence_lead_s: float = 0.6, seed: int = 0):
    """A clean tone (preceded by true silence) plus stationary white noise —
    a controlled case for the noise-reduction tab, where the "clean" ground
    truth is known exactly. The silent lead-in matters: spectral subtraction
    estimates the noise profile from a noise-only segment, so the demo must
    actually contain one (a tone from sample 0 would contaminate that estimate).
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    lead_n = min(n, int(sr * silence_lead_s))
    t = np.arange(n) / sr

    clean = np.zeros(n)
    tone_t = t[lead_n:] - t[lead_n]
    envelope = np.hanning(2 * len(tone_t))[:len(tone_t)]  # fade in, no fade out
    clean[lead_n:] = 0.5 * np.sin(2 * np.pi * freq * tone_t) * envelope

    noise = rng.normal(0, noise_level, n)
    return clean.astype(np.float32), (clean + noise).astype(np.float32), sr


def generate_song_like_demo(sr: int = 22050, duration: float = 6.0, seed: int = 0):
    """A synthetic "song": a repeating chord loop (stand-in for a
    repetitive instrumental) mixed with a non-repeating melodic line
    (stand-in for a lead vocal). Returns the mixture plus the ground-truth
    foreground/background so separation quality can be checked directly.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Repeating background: a 1-second chord loop tiled across the clip.
    loop_s = 1.0
    loop_n = int(loop_s * sr)
    t_loop = np.arange(loop_n) / sr
    chord_freqs = [220.0, 277.18, 329.63]  # A3-C#4-E4 triad
    loop = sum(0.15 * np.sin(2 * np.pi * f * t_loop) for f in chord_freqs)
    reps = int(np.ceil(n / loop_n))
    background = np.tile(loop, reps)[:n]

    # Non-repeating foreground: a slowly, non-periodically varying melody.
    melody_freq = 440 + 80 * np.sin(2 * np.pi * 0.13 * t) + 20 * np.sin(2 * np.pi * 0.29 * t)
    phase = 2 * np.pi * np.cumsum(melody_freq) / sr
    foreground = 0.2 * np.sin(phase) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t) ** 2)

    mixture = (foreground + background + rng.normal(0, 0.005, n)).astype(np.float32)
    return mixture, foreground.astype(np.float32), background.astype(np.float32), sr
