"""Sync lyrics you already have to the song: forced alignment.

Transcription asks Whisper "what words are sung here?" and inherits its
mistakes. When the correct lyrics are known, the question becomes "*when*
is each of these words sung?", which is far easier to answer well. The
text stays exactly as provided; only the timing comes from the audio.

TWO STAGES
----------
1. Rough placement: text alignment (no extra model work).
   Whisper has already transcribed the vocal with word times. The user's
   words and Whisper's words are matched as two sequences, the same idea
   as `diff` on two files (difflib's longest-matching-block algorithm):
     - equal       → the user's word takes Whisper's time
     - replaced    → Whisper misheard it; share the misheard words' time
     - inserted    → Whisper skipped it; place it in the gap between its
                     timed neighbours, in proportion to its length
   The share of words that matched exactly is reported as the match rate.
   It says how well Whisper *heard* the song, not whether the lyrics are
   right: correct Bangla lyrics can match only ~20%, and forced alignment
   (stage 2) still times them to within tens of milliseconds.

2. Precise timing: forced alignment on the audio.
   Lines are grouped into windows of at most ~30 s around their rough
   position. For each window Whisper is given the audio *and* the exact
   text; its cross-attention shows which audio frames each text token
   "looks at", and dynamic time warping (DTW) turns that into a monotonic
   word-by-word path through time. Every word gets a time from the audio,
   including the ones Whisper originally misheard. The mean probability of
   the user's words under the audio is the confidence score; it collapses
   when the lyrics don't belong to the song.

   Cross-check: a word Whisper heard reliably (inside a run of matching
   words) must land within MAX_DISAGREE_S of where Whisper heard it;
   otherwise (e.g. a sung line missing from the pasted lyrics pulled it off
   course) Whisper's timing is kept for that word.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

import numpy as np

WHISPER_SR = 16_000
WINDOW_S = 29.0  # Whisper sees 30 s at a time; keep a margin
PAD_S = 1.0  # audio kept either side of a window's rough span
MAX_TOKENS = 380  # Whisper's decoder holds 448 tokens, minus the prompt
MAX_DISAGREE_S = 1.0  # forced vs. heard timing gap that marks a word as misplaced
MIN_ANCHOR_RUN = 4  # matched words count as reliable only in runs this long

_LRC_TAG = re.compile(r"^\s*(\[\d{1,2}:\d{2}(?:[.:]\d{1,3})?\])+")
_SECTION = re.compile(r"^\s*\[[^\]]*\]\s*$")  # [Chorus], [Verse 2], [ar:Artist]


def parse_lyrics(text: str) -> list[str]:
    """Lyric lines to sync: blank lines, section headers and LRC time tags removed."""
    lines = []
    for raw in text.splitlines():
        line = _LRC_TAG.sub("", raw).strip()
        if line and not _SECTION.match(line):
            lines.append(line)
    return lines


def _normalise(word: str) -> str:
    """Compare words ignoring case, punctuation and nukta dots (ज़ vs ज)."""
    word = unicodedata.normalize("NFD", word.lower())
    return "".join(c for c in word if unicodedata.category(c)[0] not in "PS" and c != "़")


# ---------------------------------------------------------------------------
# Stage 1: rough placement by matching the two word sequences
# ---------------------------------------------------------------------------

def rough_times(user_words: list[str], heard: list[tuple[float, float, str]],
                duration: float) -> tuple[list[tuple[float, float]], list[bool], float]:
    """(start, end) for every user word, which words are reliable anchors,
    and the share of words Whisper heard exactly.

    An anchor is a word inside a run of at least MIN_ANCHOR_RUN exactly
    matching words: short runs, such as a two-word refrain, can match the
    wrong repeat of that refrain by coincidence."""
    n = len(user_words)
    times: list[tuple[float, float] | None] = [None] * n
    anchor = [False] * n
    matched = 0
    a = [_normalise(w) for w in user_words]
    b = [_normalise(w) for _, _, w in heard]

    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for k in range(i2 - i1):
                times[i1 + k] = heard[j1 + k][:2]
                anchor[i1 + k] = i2 - i1 >= MIN_ANCHOR_RUN
            matched += i2 - i1
        elif op == "replace":
            _spread(times, user_words, i1, i2, heard[j1][0], heard[j2 - 1][1])

    # Skipped words go into the gap between their nearest timed neighbours.
    i = 0
    while i < n:
        if times[i] is not None:
            i += 1
            continue
        j = i
        while j < n and times[j] is None:
            j += 1
        lo = times[i - 1][1] if i > 0 else (heard[0][0] if heard else 0.0)
        hi = times[j][0] if j < n else (heard[-1][1] if heard else duration)
        if hi <= lo:
            hi = min(duration, lo + 0.3 * (j - i))
        _spread(times, user_words, i, j, lo, hi)
        i = j

    return [t for t in times], anchor, matched / max(n, 1)


def _spread(times, words, i1, i2, start, end):
    """Share [start, end] between words[i1:i2] in proportion to their length."""
    lengths = np.array([max(len(w), 1) for w in words[i1:i2]], dtype=float)
    edges = start + (end - start) * np.concatenate([[0], np.cumsum(lengths) / lengths.sum()])
    for k in range(i2 - i1):
        times[i1 + k] = (float(edges[k]), float(edges[k + 1]))


# ---------------------------------------------------------------------------
# Stage 2: forced alignment of the exact text, window by window
# ---------------------------------------------------------------------------

def _windows(line_spans: list[tuple[float, float]]) -> list[tuple[int, int]]:
    """Group consecutive lines into windows that fit in Whisper's 30 s view."""
    groups, i = [], 0
    while i < len(line_spans):
        start = line_spans[i][0] - PAD_S
        j = i + 1
        while j < len(line_spans) and line_spans[j][1] + PAD_S - start <= WINDOW_S:
            j += 1
        groups.append((i, j))
        i = j
    return groups


def _force_align(model, tokenizer, audio: np.ndarray, t0: float, words: list[str],
                 fp16: bool) -> tuple[list[tuple[float, float]], list[float]] | None:
    """Word (start, end) times for `words`, sung somewhere in audio[t0:t0+30 s],
    and the model's probability for each word's tokens given that audio."""
    import torch
    import whisper
    from whisper.audio import HOP_LENGTH, N_FRAMES, N_SAMPLES
    from whisper.timing import find_alignment

    text = "".join(" " + w for w in words)
    tokens = tokenizer.encode(text)
    if len(tokens) > MAX_TOKENS:
        return None

    lo = int(max(t0, 0.0) * WHISPER_SR)
    chunk = audio[lo:lo + N_SAMPLES]
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(chunk), model.dims.n_mels)
    mel = mel.to(model.device, dtype=torch.float16 if fp16 else torch.float32)
    num_frames = min(N_FRAMES, len(chunk) // HOP_LENGTH)
    timings = find_alignment(model, tokenizer, tokens, mel, num_frames)

    # Whisper splits text into its own "words" (punctuation can split off), so
    # map each piece back to the user's word by character position.
    owner = np.concatenate([np.full(len(w) + 1, k) for k, w in enumerate(words)])
    spans: list[list[float]] = [[np.inf, -np.inf] for _ in words]
    probs: list[list[float]] = [[] for _ in words]
    cursor = 0
    for t in timings:
        piece = t.word
        stripped = len(piece) - len(piece.lstrip())
        pos = min(cursor + stripped, len(owner) - 1)
        k = int(owner[pos])
        spans[k][0] = min(spans[k][0], t.start)
        spans[k][1] = max(spans[k][1], t.end)
        probs[k].append(float(t.probability))
        cursor += len(piece)
    if any(not np.isfinite(s) for s, _ in spans):
        return None
    offset = lo / WHISPER_SR
    return ([(s + offset, max(e, s + 0.05) + offset) for s, e in spans],
            [float(np.mean(p)) for p in probs])


def align(model, language: str, audio: np.ndarray, lyric_lines: list[str],
          heard: list[tuple[float, float, str]], fp16: bool):
    """Timestamp each lyric line and word. Returns (lines, match_rate, confidence).

    `lines` is a list of (text, [(start, end, word), ...]); `heard` is
    Whisper's own transcription as (start, end, word) triples. `confidence`
    is the mean probability the model gives the user's words given the
    audio: it drops sharply when the lyrics don't belong to the song.
    """
    from whisper.tokenizer import get_tokenizer

    per_line = [line.split() for line in lyric_lines]
    flat = [w for ws in per_line for w in ws]
    duration = len(audio) / WHISPER_SR
    rough, anchor, match_rate = rough_times(flat, heard, duration)

    bounds = np.cumsum([0] + [len(ws) for ws in per_line])
    spans = [(rough[bounds[i]][0], rough[bounds[i + 1] - 1][1]) for i in range(len(per_line))]

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                              language=language, task="transcribe")
    final = list(rough)
    probabilities: list[float] = []
    for i, j in _windows(spans):
        words = flat[bounds[i]:bounds[j]]
        result = _force_align(model, tokenizer, audio, spans[i][0] - PAD_S, words, fp16)
        if result is not None:  # otherwise the rough times stand
            final[bounds[i]:bounds[j]], probs = result
            probabilities += probs

    # Cross-check: where Whisper reliably heard a word, the two timings should
    # agree. The forced path must use up every audio frame, so a sung line
    # missing from the lyrics gets absorbed into its neighbours' words; any
    # anchor word it moved more than MAX_DISAGREE_S keeps Whisper's timing.
    for k in range(len(flat)):
        if anchor[k] and abs(final[k][0] - rough[k][0]) > MAX_DISAGREE_S:
            final[k] = rough[k]

    out = []
    for i, text in enumerate(lyric_lines):
        words = [(s, e, (" " if k else "") + w) for k, ((s, e), w) in
                 enumerate(zip(final[bounds[i]:bounds[i + 1]], per_line[i]))]
        out.append((text, words))
    confidence = float(np.mean(probabilities)) if probabilities else 0.0
    return out, match_rate, confidence
