"""End-to-end checks for the vault. Run with `python -m vault.selftest`.

These are the claims the project makes, each reduced to something that
either passes or fails:

    1. round trip      the recovered file is byte-for-byte the original
    2. wrong password  fails, and returns nothing
    3. tampered image  fails, even with the correct password
    4. foreign image   is rejected before any password is asked for
    5. cover image     a user-supplied carrier works and stays intact
    6. capacity        the arithmetic agrees with what actually fits
    7. ciphertext      looks like noise, not like an audio file
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

from . import crypto, stego
from .container import HEADER_SIZE, sha256_hex
from .pipeline import VaultError, decode_png, encode_audio, inspect_png

PASSWORD = "correct horse battery staple"


def _wav(seconds: float = 0.5, sr: int = 22050) -> bytes:
    """A real WAV file, built without touching the DSP package."""
    import struct
    import wave

    t = np.arange(int(sr * seconds)) / sr
    samples = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buf.getvalue()


class Report:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")


def main() -> int:
    report = Report()
    audio = _wav()
    print(f"\nSource: {len(audio):,}-byte WAV, sha256 {sha256_hex(audio)[:16]}…\n")

    # 1 ── round trip -------------------------------------------------
    print("1. Round trip")
    encoded = encode_audio(audio, "tone.wav", PASSWORD, audio_info={"sampleRate": 22050})
    decoded = decode_png(encoded.png, PASSWORD)
    report.check("recovered bytes are identical", decoded.file_bytes == audio,
                 f"{len(decoded.file_bytes):,} bytes")
    report.check("SHA-256 matches", decoded.recovered_sha256 == sha256_hex(audio))
    report.check("integrity flag set", decoded.integrity_ok)
    report.check("filename preserved", decoded.metadata.filename == "tone.wav")
    report.check("audio metadata preserved", decoded.metadata.audio.get("sampleRate") == 22050)
    differences = sum(a != b for a, b in zip(audio, decoded.file_bytes))
    report.check("byte-by-byte difference count is 0", differences == 0, f"{differences} differing")

    # 2 ── wrong password ---------------------------------------------
    print("\n2. Wrong password")
    try:
        decode_png(encoded.png, "hunter2")
        report.check("rejected", False, "decoding succeeded, which must not happen")
    except VaultError as exc:
        report.check("rejected with an authentication failure", "Authentication failed" in str(exc))

    # 3 ── tampered image ---------------------------------------------
    print("\n3. Tampered image")
    tampered = encoded.pixels.copy()
    flat = tampered.reshape(-1)
    # Flip the low bit of a byte well inside the ciphertext.
    target = (HEADER_SIZE + 64) * 8
    flat[target] ^= 1
    tampered_png = stego.to_png(flat.reshape(tampered.shape))
    try:
        decode_png(tampered_png, PASSWORD)
        report.check("rejected", False, "a modified container decoded successfully")
    except VaultError as exc:
        report.check("one flipped bit is detected", "Authentication failed" in str(exc))

    # A header edit must fail too — the header is associated data.
    header_tampered = encoded.pixels.copy().reshape(-1)
    header_tampered[8 * 8] ^= 1  # inside the salt field
    try:
        decode_png(stego.to_png(header_tampered.reshape(encoded.pixels.shape)), PASSWORD)
        report.check("header edit rejected", False, "an edited header was accepted")
    except VaultError:
        report.check("header edit is detected (authenticated as associated data)", True)

    # 4 ── a plain image ----------------------------------------------
    print("\n4. Image with no container")
    plain = stego.to_png(stego.generate_cover(64, 64, seed=7))
    try:
        inspect_png(plain)
        report.check("rejected", False, "an ordinary image was accepted as a container")
    except VaultError as exc:
        report.check("rejected on the magic number, before any password", "magic number" in str(exc))

    # 5 ── user-supplied cover ----------------------------------------
    print("\n5. User-supplied cover image")
    side = 1200
    cover_arr = np.zeros((side, side, 3), dtype=np.uint8)
    cover_arr[..., 0] = np.linspace(0, 255, side, dtype=np.uint8)[None, :]
    cover_arr[..., 1] = np.linspace(0, 255, side, dtype=np.uint8)[:, None]
    cover_png = stego.to_png(cover_arr)
    with_cover = encode_audio(audio, "tone.wav", PASSWORD, cover_png=cover_png)
    back = decode_png(with_cover.png, PASSWORD)
    report.check("round trip through a supplied cover", back.file_bytes == audio)
    delta = np.abs(with_cover.pixels.astype(int) - cover_arr.astype(int))
    report.check("no channel moves by more than 1 step", delta.max() <= 1,
                 f"max delta {delta.max()}, mean {delta.mean():.3f}")

    # Too-small cover must be refused rather than silently truncating.
    try:
        encode_audio(audio, "tone.wav", PASSWORD, cover_png=stego.to_png(stego.generate_cover(32, 32)))
        report.check("undersized cover refused", False, "it was accepted")
    except VaultError as exc:
        report.check("undersized cover refused with a clear message", "holds" in str(exc))

    # 6 ── capacity arithmetic ----------------------------------------
    print("\n6. Capacity arithmetic")
    for payload in (1, 1000, 1_000_000):
        plan = stego.plan_image(payload)
        report.check(f"{payload:>9,} bytes fits the planned {plan.width}×{plan.height}",
                     plan.total_bytes >= payload,
                     f"capacity {plan.total_bytes:,}")
    cap = encoded.capacity
    report.check("reported utilisation is consistent",
                 abs(cap.describe(len(encoded.container))["utilization"]
                     - len(encoded.container) / cap.total_bytes) < 1e-9)

    # 2 bits per channel should also round-trip.
    two_bit = encode_audio(audio, "tone.wav", PASSWORD, bits_per_channel=2)
    report.check("2 bits per channel round-trips",
                 decode_png(two_bit.png, PASSWORD, bits_per_channel=2).file_bytes == audio,
                 f"{two_bit.capacity.width}×{two_bit.capacity.height} vs "
                 f"{encoded.capacity.width}×{encoded.capacity.height} at 1 bit")

    # 7 ── ciphertext statistics --------------------------------------
    print("\n7. Ciphertext looks like noise")
    cipher_hist = stego.byte_histogram(encoded.container[HEADER_SIZE:])
    plain_hist = stego.byte_histogram(audio)
    cipher_entropy = stego.shannon_entropy(cipher_hist)
    plain_entropy = stego.shannon_entropy(plain_hist)
    report.check("ciphertext entropy is near the 8.0 bits/byte ceiling",
                 cipher_entropy > 7.9, f"{cipher_entropy:.4f} bits/byte")
    report.check("and higher than the source WAV's",
                 cipher_entropy > plain_entropy,
                 f"WAV {plain_entropy:.4f} vs cipher {cipher_entropy:.4f}")
    report.check("the RIFF signature is gone from the ciphertext",
                 b"RIFF" not in encoded.container[HEADER_SIZE:])

    # 8 ── key derivation ---------------------------------------------
    print("\n8. Key derivation")
    salt = crypto.random_salt()
    k1 = crypto.derive_key(PASSWORD, salt)
    k2 = crypto.derive_key(PASSWORD, salt)
    k3 = crypto.derive_key(PASSWORD, crypto.random_salt())
    report.check("deterministic for one salt", k1 == k2, f"{len(k1) * 8}-bit key")
    report.check("a different salt gives an unrelated key", k1 != k3)
    report.check("empty password refused", _raises(lambda: crypto.derive_key("", salt)))

    print(f"\n{'ALL CHECKS PASSED' if not report.failures else f'{report.failures} CHECK(S) FAILED'}\n")
    return 1 if report.failures else 0


def _raises(fn) -> bool:
    try:
        fn()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    sys.exit(main())
