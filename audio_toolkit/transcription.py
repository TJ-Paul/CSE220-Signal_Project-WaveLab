"""Lyrics extraction and voice transcription: Demucs, then Whisper.

PIPELINE
--------
    song / recording
      → [songs only] Demucs isolates the vocal stem   (ml_separation)
      → resample to 16 kHz mono                         (librosa)
      → Whisper speech-to-text, language fixed          (openai-whisper)
      → second pass on voiced stretches with no text    (short-time energy)
      → trim word times to where the vocal sounds       (short-time energy)
      → split long segments at the singer's breaths     (short-time energy)
      → energy gate: drop lines over a silent stem      (plain DSP, below)
      → timestamped lines: [(start, end, text), ...]

Why separate first: Whisper was trained mostly on speech. Drums, bass and
chords in a full mix look like noise to it and cause missed or invented
words. On the isolated vocal stem it hears (almost) only the singer. For a
plain voice recording there is nothing to remove, so that step is skipped.

Like Demucs, Whisper (OpenAI) is a published, pretrained model used here as
a tool: an encoder-decoder transformer that reads an 80-band log-mel
spectrogram of 30-second windows and writes out text with timestamps. No
training happens in this project.

EFFICIENCY
----------
- Both networks run on the Apple-Silicon GPU (PyTorch "mps") when present,
  about 3-4x faster than CPU here; CUDA or CPU otherwise.
- The user picks the language (English / Hindi / Bangla), so Whisper skips
  its language-detection pass and cannot guess wrong on a sung intro.
- Audio is handed to Whisper as an in-memory array, so no ffmpeg decode
  and no temporary files.
- Each model loads once and stays in memory.
- `condition_on_previous_text=False`: each window is decoded on its own,
  which stops the repeat-the-last-line loops Whisper is prone to on music.
- `temperature=0.0`: one greedy decode per window, no retries. Whisper's
  default retries a window with more and more randomness when its text is
  "too repetitive" (gzip compression ratio > 2.4). Choruses legitimately
  repeat, so on songs that check fired constantly: a 60 s track took 4x
  longer and gave different lyrics on every run. Greedy is fast and
  deterministic; the energy gate below handles the hallucination case.
- Word timestamps are on. Segment-level times alone are coarse (a line
  that starts at 2.0 s is often reported at 0.0 s); aligning each word
  pulls line boundaries to within ~0.2-0.5 s, which lyric sync needs, for
  about one extra second per song-minute.

THE ENERGY GATE (a DSP guard against hallucination)
---------------------------------------------------
Given near-silence, Whisper sometimes still "hears" words. After separation
we have the vocal stem itself, so we can check: if the stem's RMS during a
line is more than GATE_DB below the loudest part of the vocal, nobody was
singing there and the line is dropped.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import librosa

from . import lyrics_align, ml_separation
from .ml_separation import best_device

#: Languages offered in the UI, as Whisper language codes.
LANGUAGES: dict[str, str] = {"en": "English", "hi": "Hindi", "bn": "Bangla"}

#: large-v3-turbo: large-v3's encoder with a 4-layer decoder. Near large-v3
#: accuracy, several times faster, and the smallest model that handles
#: Bangla well ("small" produces the wrong script on it).
MODEL_NAME = "turbo"

WHISPER_SR = 16_000
GATE_DB = 35.0

#: Lyric-line shaping. Whisper segments can run several sentences long, and
#: its word times are contiguous (a breath is absorbed into the neighbouring
#: words), so pauses are found in the vocal's short-time energy instead: a
#: frame is "quiet" when its RMS is QUIET_DB below the loud vocal level. A
#: segment is split at its longest quiet run (sentence punctuation adds an
#: extra second) when that run is at least BREAK_PAUSE_S or the segment has
#: more than MAX_WORDS words. No line gets fewer than MIN_WORDS.
QUIET_DB = 30.0
ENVELOPE_HOP_S = 0.01
BREAK_PAUSE_S = 0.2
MAX_WORDS = 10
MIN_WORDS = 3
_SENTENCE_END = (".", "!", "?", ",", "।", "॥")

#: Second pass. Whisper can collapse repeated lines (a chorus sung twice in
#: one 30 s window comes out once) and then skip ahead. Stretches of the
#: vocal that are clearly voiced for at least RECOVER_MIN_S but covered by
#: no line are transcribed again on their own.
RECOVER_MIN_S = 1.0
WORD_GAP_S = 0.12  # quiet gaps shorter than this stay inside one word
_PHRASE_BRIDGE_S = 0.3  # quiet gaps shorter than this stay inside one phrase

_model = None
_model_device: str | None = None
_lock = threading.Lock()  # one GPU job at a time; the model is shared


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Line:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Transcript:
    language: str
    mode: str  # "song" | "voice"
    lines: list[Line]
    device: str
    model: str
    timings: dict[str, float] = field(default_factory=dict)
    #: Isolated vocal stem at the input rate (songs only), for playback.
    vocals: np.ndarray | None = None
    dropped_silent: int = 0
    recovered: int = 0
    #: True when the lines are the user's own lyrics, synced to the audio.
    aligned: bool = False
    #: Share of the user's words that Whisper also heard (aligned mode only).
    match_rate: float | None = None
    #: Mean probability of the user's words given the audio (aligned mode only).
    confidence: float | None = None

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def is_available() -> bool:
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model():
    """Load Whisper once, on the GPU if possible, falling back to CPU."""
    global _model, _model_device
    if _model is None:
        import whisper
        device = best_device()
        try:
            _model = whisper.load_model(MODEL_NAME, device=device)
        except (RuntimeError, NotImplementedError):
            device = "cpu"
            _model = whisper.load_model(MODEL_NAME, device=device)
        _model_device = device
        if device == "mps":
            _patch_word_alignment_for_mps()
    return _model, _model_device


def _patch_word_alignment_for_mps() -> None:
    """Word timing ends in a DTW over a small alignment matrix, which whisper
    converts to float64 first. MPS has no float64, so move just that matrix
    to the CPU; the network itself stays on the GPU."""
    import whisper.timing as timing
    original = timing.dtw
    timing.dtw = lambda x: original(x.cpu())


def _quiet_frames(y: np.ndarray, sr: int) -> np.ndarray:
    """Per-10 ms frame: is the vocal near-silent here? (short-time RMS)"""
    hop = max(1, int(ENVELOPE_HOP_S * sr))
    rms = librosa.feature.rms(y=y, frame_length=2 * hop, hop_length=hop)[0]
    reference = np.percentile(rms, 95) + 1e-12
    return rms < reference * 10 ** (-QUIET_DB / 20)


def _longest_quiet(quiet: np.ndarray, a: float, b: float) -> float:
    """Longest run of quiet frames between times a and b, in seconds."""
    run = best = 0
    for q in quiet[int(a / ENVELOPE_HOP_S):int(b / ENVELOPE_HOP_S)]:
        run = run + 1 if q else 0
        best = max(best, run)
    return best * ENVELOPE_HOP_S


def _split_words(words: list[Word], quiet: np.ndarray) -> list[list[Word]]:
    """Recursively cut a run of words at its biggest breath."""
    if len(words) < 2 * MIN_WORDS:
        return [words]

    def pause_before(k: int) -> float:
        # Search from mid-word to mid-word: the breath hides inside whichever
        # neighbour Whisper stretched to cover it.
        a = (words[k - 1].start + words[k - 1].end) / 2
        b = (words[k].start + words[k].end) / 2
        bonus = 1.0 if words[k - 1].text.strip().endswith(_SENTENCE_END) else 0.0
        return _longest_quiet(quiet, a, b) + bonus

    k = max(range(MIN_WORDS, len(words) - MIN_WORDS + 1), key=pause_before)
    if len(words) > MAX_WORDS or pause_before(k) >= BREAK_PAUSE_S:
        return _split_words(words[:k], quiet) + _split_words(words[k:], quiet)
    return [words]


def _split_into_lyric_lines(lines: list[Line], quiet: np.ndarray) -> list[Line]:
    """Break long segments into singable lines at the singer's breaths."""
    out: list[Line] = []
    for line in lines:
        if not line.words:
            out.append(line)
            continue
        for part in _split_words(line.words, quiet):
            out.append(Line(part[0].start, part[-1].end,
                            "".join(w.text for w in part).strip(), part))
    return out


def _snap_to_voice(lines: list[Line], quiet: np.ndarray) -> list[Line]:
    """Trim each word to where the vocal is actually sounding.

    Alignment paths are continuous, so the silence before a line gets
    absorbed into its first word (a line starting at 2.0 s is reported as
    starting when the previous line ended). Within each word's time slot
    the sounding stretches are found (quiet gaps shorter than WORD_GAP_S,
    such as stop consonants, are bridged) and the word is taken to be:
      - a line's first word: the last stretch (silence before it absorbed),
      - a line's last word: the first stretch (silence after it absorbed),
      - any other word: the longest stretch.
    """
    bridge = int(WORD_GAP_S / ENVELOPE_HOP_S)
    for line in lines:
        last = len(line.words) - 1
        for k, w in enumerate(line.words):
            i0 = int(w.start / ENVELOPE_HOP_S)
            i1 = max(int(np.ceil(w.end / ENVELOPE_HOP_S)), i0 + 1)
            voiced = np.flatnonzero(~quiet[i0:i1])
            if not voiced.size:
                continue
            # Split the voiced frames into stretches wherever a long gap occurs.
            cuts = np.flatnonzero(np.diff(voiced) > bridge + 1) + 1
            stretches = np.split(voiced, cuts)
            if k == 0 and last > 0:
                best = stretches[-1]
            elif k == last and last > 0:
                best = stretches[0]
            else:
                best = max(stretches, key=lambda r: r[-1] - r[0])
            w.start = (i0 + best[0]) * ENVELOPE_HOP_S
            w.end = (i0 + best[-1] + 1) * ENVELOPE_HOP_S
        if line.words:
            line.start, line.end = line.words[0].start, line.words[-1].end
    return lines


def _to_lines(result: dict, offset: float = 0.0) -> list[Line]:
    """Whisper segments → Lines, shifted by `offset` seconds."""
    return [
        Line(float(s["start"]) + offset, float(s["end"]) + offset, s["text"].strip(),
             # Raw text keeps Whisper's own leading spaces, so "pre" + "-trained"
             # joins back into "pre-trained" rather than "pre -trained".
             [Word(float(w["start"]) + offset, float(w["end"]) + offset, w["word"])
              for w in s.get("words", []) if w["word"].strip()])
        for s in result["segments"] if s["text"].strip()
    ]


def _missed_speech(quiet: np.ndarray, lines: list[Line]) -> list[tuple[float, float]]:
    """Voiced stretches of the vocal that no line covers."""
    covered = np.zeros(len(quiet), dtype=bool)
    for line in lines:
        covered[int(line.start / ENVELOPE_HOP_S):int(line.end / ENVELOPE_HOP_S) + 1] = True
    voiced = ~quiet

    # Bridge short quiet gaps (breaths inside a phrase) so a phrase is one span.
    bridge = int(_PHRASE_BRIDGE_S / ENVELOPE_HOP_S)
    missed = voiced & ~covered
    spans, start, last = [], None, -bridge - 1
    for i in np.flatnonzero(missed):
        if start is None:
            start = i
        elif i - last > bridge:
            spans.append((start, last + 1))
            start = i
        last = i
    if start is not None:
        spans.append((start, last + 1))
    return [(a * ENVELOPE_HOP_S, b * ENVELOPE_HOP_S) for a, b in spans
            if (b - a) * ENVELOPE_HOP_S >= RECOVER_MIN_S]


def _energy_gate(lines: list[Line], y: np.ndarray, sr: int) -> tuple[list[Line], int]:
    """Drop lines whose audio is GATE_DB below the loudest vocal passage."""
    frame = int(0.05 * sr)
    if len(y) < frame * 4:
        return lines, 0
    frame_rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=frame)[0]
    reference = np.percentile(frame_rms, 95) + 1e-12
    floor = reference * 10 ** (-GATE_DB / 20)

    kept = []
    for line in lines:
        segment = y[int(line.start * sr):max(int(line.end * sr), int(line.start * sr) + 1)]
        if np.sqrt(np.mean(segment.astype(np.float64) ** 2)) >= floor:
            kept.append(line)
    return kept, len(lines) - len(kept)


def transcribe(
    y: np.ndarray,
    sr: int,
    language: str,
    mode: str = "song",
    y_stereo: np.ndarray | None = None,
    vocals: np.ndarray | None = None,
    lyrics: list[str] | None = None,
) -> Transcript:
    """Timestamped transcript of a song's lyrics or a voice recording.

    mode="song" isolates the vocals with Demucs first; mode="voice" sends
    the audio straight to Whisper. Pass `vocals` to reuse a stem that was
    already separated (skips Demucs).

    Pass `lyrics` (one string per line) to sync known lyrics instead of
    trusting Whisper's text: see `lyrics_align`.
    """
    if language not in LANGUAGES:
        raise ValueError(f"language must be one of {list(LANGUAGES)}")
    timings: dict[str, float] = {}

    with _lock:
        # Model loading is a one-off per server start; time it separately so
        # the reported processing time describes the audio, not the startup.
        t = time.perf_counter()
        first_load = _model is None or (mode == "song" and vocals is None
                                        and best_device() not in ml_separation._separators)
        if mode == "song" and vocals is None:
            ml_separation._get_separator(best_device())
        model, device = _get_model()
        if first_load:
            timings["loadModels"] = time.perf_counter() - t

        if mode == "song":
            if vocals is None:
                t = time.perf_counter()
                vocals, _ = ml_separation.separate_vocals_instrumental(
                    y_stereo if y_stereo is not None else y, sr)
                if vocals.ndim == 2:
                    vocals = vocals.mean(axis=0)
                timings["separate"] = time.perf_counter() - t
            speech = vocals
        else:
            speech = y

        t = time.perf_counter()
        speech = np.asarray(speech, dtype=np.float32)
        audio = librosa.resample(speech, orig_sr=sr, target_sr=WHISPER_SR).astype(np.float32)
        options = dict(
            language=language,
            task="transcribe",
            fp16=device != "cpu",
            condition_on_previous_text=False,
            temperature=0.0,
            word_timestamps=True,
            verbose=None,
        )
        lines = _to_lines(model.transcribe(audio, **options))

        # Second pass: re-transcribe voiced stretches the first pass skipped.
        quiet = _quiet_frames(speech, sr)
        recovered = 0
        for a, b in _missed_speech(quiet, lines):
            lo, hi = max(0.0, a - 0.2), b + 0.2
            extra = _to_lines(model.transcribe(
                audio[int(lo * WHISPER_SR):int(hi * WHISPER_SR)], **options), offset=lo)
            recovered += len(extra)
            lines += extra
        lines.sort(key=lambda line: line.start)
        timings["transcribe"] = time.perf_counter() - t

        match_rate = confidence = None
        if lyrics:
            # The user's text is the truth; Whisper's words only anchor the timing.
            t = time.perf_counter()
            heard = [(w.start, w.end, w.text.strip()) for line in lines for w in line.words]
            synced, match_rate, confidence = lyrics_align.align(
                model, language, audio, lyrics, heard, fp16=device != "cpu")
            timings["align"] = time.perf_counter() - t

    if lyrics:
        lines = [Line(words[0][0], words[-1][1], text, [Word(s, e, w) for s, e, w in words])
                 for text, words in synced]
        lines = _snap_to_voice(lines, quiet)
        dropped = 0
    else:
        lines = _split_into_lyric_lines(_snap_to_voice(lines, quiet), quiet)
        lines, dropped = _energy_gate(lines, speech, sr)

    return Transcript(
        language=language,
        mode=mode,
        lines=lines,
        device=device,
        model=MODEL_NAME,
        timings=timings,
        vocals=vocals if mode == "song" else None,
        dropped_silent=dropped,
        recovered=recovered,
        aligned=bool(lyrics),
        match_rate=match_rate,
        confidence=confidence,
    )
