"""Hide an encrypted text note inside a song — audio steganography.

The output is an ordinary 16-bit WAV that plays exactly like the original.
Only someone with the password can tell a note is present, or read it.

    hide:    text → AES-256-GCM (password) → header + ciphertext → bits
             → password-chosen sample positions → LSB matching (±1) → WAV
    reveal:  WAV → password-chosen positions → bits → header + ciphertext
             → AES-256-GCM verify → text

THREE LAYERS
------------
1. Encryption (confidentiality + integrity). The note is sealed with
   AES-256-GCM under a scrypt-stretched key — the same `vault.crypto`
   code the image vault uses. The GCM tag authenticates everything, so a
   wrong password or an altered file fails cleanly instead of printing
   garbage.

2. Scattering (concealment). The image vault writes its container at the
   start of the image behind a readable "SGVL" marker. Here nothing is
   readable without the password: a second scrypt derivation of the
   password seeds a random generator that picks *which* samples carry the
   bits, scattered across the whole song. Without the password there is
   no marker to find and no known place to look.

3. LSB matching (statistical stealth). A 16-bit sample is an integer in
   [-32768, 32767]; its lowest bit is 1/32768 of full scale, about -90 dB,
   far below audibility. Rather than *overwriting* that bit (LSB
   replacement, which leaves a known fingerprint — values pair up), a
   sample whose parity is wrong is nudged by +1 or -1 at random. The
   parity still carries the bit, but the histogram stays natural.

DENIABILITY
-----------
A wrong password, a file with no note, and a damaged file all give the
same answer: "no hidden note found". The decoder never admits a note
exists unless it can also open it.

WHAT DESTROYS THE NOTE
----------------------
Anything that changes sample values: MP3/AAC encoding, resampling,
normalising, trimming or editing. Lossless copies (the WAV itself, or a
FLAC made from it) keep it.
"""
from __future__ import annotations

import hashlib
import io
import struct
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

from . import crypto
from .container import NONCE_SIZE, SALT_SIZE, TAG_SIZE

#: salt | nonce | ciphertext length — also the GCM associated data.
HEADER = struct.Struct(f">{SALT_SIZE}s{NONCE_SIZE}sI")
HEADER_BITS = HEADER.size * 8

MAX_MESSAGE_BYTES = 32_768
#: At most one sample in STEALTH_RATIO carries a bit, keeping the changes
#: sparse: a short note touches a tiny fraction of the song.
STEALTH_RATIO = 8

#: Fixed, public salt for the *position* key. Positions must be derivable
#: before anything else is read, so they cannot use the random per-message
#: salt; the encryption key still does.
_POSITION_SALT = b"signal-lab/audio-stego/positions/v1"
_POSITION_LOG2_N = 14
_BATCH = 8192


class StegoError(Exception):
    """Hiding failed (an input problem), with a message fit for the user."""


class RevealError(Exception):
    """No note could be opened. Deliberately one error for every cause."""


NOT_FOUND = ("No hidden note was found with this password. Either the password is wrong, "
             "there is no note, or the file was re-encoded or edited.")


@dataclass
class HideResult:
    wav: bytes
    pcm_original: np.ndarray  # (frames, channels) int16
    pcm_stego: np.ndarray
    sr: int
    stats: dict = field(default_factory=dict)
    changed_rows: list[dict] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass
class RevealResult:
    message: str
    stats: dict = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sample plumbing
# ---------------------------------------------------------------------------

def to_pcm16(y: np.ndarray) -> np.ndarray:
    """Float audio, (n,) or (channels, n), → int16 PCM shaped (frames, channels)."""
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if y.shape[0] > y.shape[1]:  # already (frames, channels)
        y = y.T
    return np.clip(np.round(y.T * 32768.0), -32768, 32767).astype(np.int16)


def wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def read_pcm16(data: bytes) -> tuple[np.ndarray, int, dict]:
    """Exact integer samples of a lossless file, shaped (frames, channels)."""
    try:
        info = sf.info(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise RevealError("This file could not be read as audio.") from exc
    lossless = info.subtype.startswith("PCM") or info.format in ("WAV", "FLAC", "AIFF")
    if not lossless:
        raise RevealError(
            f"This is a {info.format}/{info.subtype} file. Lossy formats rewrite every sample, "
            "so they cannot carry a hidden note; use the original WAV.")
    pcm, sr = sf.read(io.BytesIO(data), dtype="int16", always_2d=True)
    return pcm, sr, {"format": info.format, "subtype": info.subtype, "channels": info.channels}


# ---------------------------------------------------------------------------
# Password-chosen positions
# ---------------------------------------------------------------------------

def _position_rng(password: str) -> np.random.Generator:
    seed = hashlib.scrypt(password.encode("utf-8"), salt=_POSITION_SALT,
                          n=2**_POSITION_LOG2_N, r=8, p=1, dklen=32)
    return np.random.default_rng(np.frombuffer(seed, dtype=np.uint64))


def positions(password: str, total: int, count: int) -> np.ndarray:
    """The first `count` distinct sample indices of the password's sequence.

    Drawn in fixed-size batches and de-duplicated in order, so the start of
    the sequence never changes with `count`: the decoder reads the header
    positions first, then extends the same sequence for the ciphertext.
    """
    if count > total:
        raise StegoError("The note does not fit in this audio.")
    rng = _position_rng(password)
    seen = np.zeros(total, dtype=bool)
    out = np.empty(count, dtype=np.int64)
    filled = 0
    while filled < count:
        batch = rng.integers(0, total, size=_BATCH)
        # First occurrence of each index within the batch, in draw order...
        _, first = np.unique(batch, return_index=True)
        fresh = batch[np.sort(first)]
        # ...minus any already taken by earlier batches.
        take = fresh[~seen[fresh]][:count - filled]
        seen[take] = True
        out[filled:filled + take.size] = take
        filled += take.size
    return out


def _bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def capacity_bytes(total_samples: int) -> int:
    """Largest note (UTF-8 bytes) this audio can carry within the stealth budget."""
    usable_bits = total_samples // STEALTH_RATIO - HEADER_BITS - TAG_SIZE * 8
    return max(0, min(MAX_MESSAGE_BYTES, usable_bits // 8))


# ---------------------------------------------------------------------------
# LSB matching
# ---------------------------------------------------------------------------

def _embed_bits(flat: np.ndarray, idx: np.ndarray, bits: np.ndarray) -> int:
    """Set sample parities at `idx` to `bits`, nudging ±1. Returns the count changed."""
    samples = flat[idx]
    wrong = (samples & 1) != bits
    changed = idx[wrong]
    vals = flat[changed].astype(np.int32)
    # ±1 at random (fresh OS entropy every time, so no two files share a
    # pattern), forced inward at the 16-bit rails so a sample never wraps.
    step = np.where(np.random.default_rng().integers(0, 2, size=vals.size) == 0, -1, 1)
    step[vals <= -32768] = 1
    step[vals >= 32767] = -1
    flat[changed] = (vals + step).astype(np.int16)
    return int(wrong.sum())


def _read_bits(flat: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return (flat[idx] & 1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Hide / reveal
# ---------------------------------------------------------------------------

def hide(y: np.ndarray, sr: int, message: str, password: str) -> HideResult:
    t0 = [time.perf_counter()]
    timings: dict[str, float] = {}

    def lap(name: str) -> None:
        now = time.perf_counter()
        timings[name] = (now - t0[0]) * 1000
        t0[0] = now

    if not password:
        raise StegoError("A password is required.")
    plaintext = message.encode("utf-8")
    if not plaintext:
        raise StegoError("The note is empty.")
    if len(plaintext) > MAX_MESSAGE_BYTES:
        raise StegoError(f"The note is too long (limit {MAX_MESSAGE_BYTES} bytes).")

    pcm = to_pcm16(y)
    flat = pcm.reshape(-1).copy()  # interleaved samples
    total = flat.size
    cap = capacity_bytes(total)
    if len(plaintext) > cap:
        raise StegoError(
            f"This audio can hold {cap} bytes of text; the note is {len(plaintext)} bytes "
            "(an English letter is 1 byte, a Bangla or Hindi letter 3). "
            "Use a longer track or a shorter note.")

    # Encrypt (salt/nonce fresh per note). GCM output is always plaintext + a
    # 16-byte tag, so the header's length is known up front and the exact
    # header bytes can be authenticated as associated data.
    salt, nonce = crypto.random_salt(), crypto.random_nonce()
    key = crypto.derive_key(password, salt)
    lap("keyDerivation")
    header = HEADER.pack(salt, nonce, len(plaintext) + TAG_SIZE)
    ciphertext = crypto.encrypt(key, nonce, plaintext, header)
    bitstream = np.concatenate([_bits(header), _bits(ciphertext)])
    lap("encrypt")

    idx = positions(password, total, bitstream.size)
    changed = _embed_bits(flat, idx, bitstream)
    lap("embed")

    stego_pcm = flat.reshape(pcm.shape)
    wav = wav_bytes(stego_pcm, sr)
    lap("writeWav")

    # A small before/after sample table for the teaching view.
    changed_idx = idx[(pcm.reshape(-1)[idx] & 1) != bitstream][:8]
    rows = [{"index": int(i),
             "before": int(pcm.reshape(-1)[i]),
             "after": int(stego_pcm.reshape(-1)[i])}
            for i in changed_idx]

    stats = {
        "messageBytes": len(plaintext),
        "capacityBytes": cap,
        "totalSamples": int(total),
        "samplesUsed": int(idx.size),
        "samplesChanged": changed,
        "changeFraction": changed / total,
        "maxAmplitudeChange": 1,
        "psnrDb": _psnr(pcm.reshape(-1), stego_pcm.reshape(-1)),
        "channels": int(pcm.shape[1]),
        "sampleRate": int(sr),
    }
    return HideResult(wav=wav, pcm_original=pcm, pcm_stego=stego_pcm, sr=sr,
                      stats=stats, changed_rows=rows, timings_ms=timings)


def reveal(data: bytes, password: str) -> RevealResult:
    t0 = [time.perf_counter()]
    timings: dict[str, float] = {}

    def lap(name: str) -> None:
        now = time.perf_counter()
        timings[name] = (now - t0[0]) * 1000
        t0[0] = now

    if not password:
        raise RevealError("A password is required.")
    pcm, sr, fmt = read_pcm16(data)
    flat = pcm.reshape(-1)
    total = flat.size

    # Read the header first. The position sequence is prefix-stable, so asking
    # for more positions afterwards continues the same sequence.
    if total // STEALTH_RATIO < HEADER_BITS + TAG_SIZE * 8:
        raise RevealError(NOT_FOUND)
    header_idx = positions(password, total, HEADER_BITS)
    header_bytes = np.packbits(_read_bits(flat, header_idx)).tobytes()
    salt, nonce, clen = HEADER.unpack(header_bytes)
    # With a wrong password these fields are noise; reject impossible lengths
    # before reading anything more. Same answer as every other failure.
    if clen < TAG_SIZE or clen > MAX_MESSAGE_BYTES + TAG_SIZE:
        raise RevealError(NOT_FOUND)
    need = HEADER_BITS + clen * 8
    if need > total // STEALTH_RATIO:
        raise RevealError(NOT_FOUND)
    idx = positions(password, total, need)
    lap("positions")

    ciphertext = np.packbits(_read_bits(flat, idx[HEADER_BITS:])).tobytes()
    lap("extract")

    key = crypto.derive_key(password, salt)
    lap("keyDerivation")
    try:
        plaintext = crypto.decrypt(key, nonce, ciphertext, header_bytes)
    except crypto.DecryptionError as exc:
        raise RevealError(NOT_FOUND) from exc
    lap("decrypt")

    try:
        message = plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RevealError(NOT_FOUND) from exc

    return RevealResult(
        message=message,
        stats={"messageBytes": len(plaintext), "channels": int(pcm.shape[1]),
               "sampleRate": int(sr), "format": fmt.get("format")},
        timings_ms=timings,
    )


def _psnr(original: np.ndarray, stego: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB between the two sample streams."""
    diff = original.astype(np.float64) - stego.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse <= 0:
        return float("inf")
    return float(10 * np.log10((32767.0**2) / mse))
