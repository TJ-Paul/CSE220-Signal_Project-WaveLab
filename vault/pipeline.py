"""The two end-to-end directions, and the measurements the UI reports.

    encode_audio:  file bytes → metadata+file → AES-256-GCM → header+ciphertext
                   → bitstream → LSB into RGB pixels → PNG
    decode_png:    PNG → pixels → header → ciphertext → AES-256-GCM
                   → metadata+file → SHA-256 check → the original bytes

The ordering is load-bearing and worth stating once, because getting it
backwards is the classic mistake: **encrypt first, then embed.**

Embedding before encrypting would mean hiding a recognisable file inside
an image — obscurity, not security, and the moment anyone suspects the
technique the payload reads straight out. Encrypting first means that
even an adversary who knows exactly where to look, and extracts every
low bit perfectly, is left with ciphertext.

Compression is deliberately *not* applied to the payload. Audio files in
every format the vault accepts are already compressed (MP3, AAC, OGG,
FLAC) or incompressible noise-like PCM, so DEFLATE would typically save
a fraction of a percent. More importantly, compressing before encrypting
makes ciphertext length depend on plaintext content, which is the
ingredient behind compression side-channel attacks such as CRIME and
BREACH. Skipping it costs almost nothing and removes the whole class.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import crypto, stego
from .container import (
    HEADER_SIZE,
    TAG_SIZE,
    ContainerError,
    Header,
    Metadata,
    build_plaintext,
    sha256_hex,
    split_plaintext,
)


class VaultError(Exception):
    """Anything the user should see as a failed encode or decode."""


@dataclass
class EncodeResult:
    png: bytes
    pixels: np.ndarray
    cover: np.ndarray
    container: bytes
    header: Header
    metadata: Metadata
    capacity: stego.Capacity
    timings_ms: dict[str, float] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        histogram = stego.byte_histogram(self.container[HEADER_SIZE:])
        return {
            "originalSize": self.metadata.size,
            "containerSize": len(self.container),
            "ciphertextSize": self.header.payload_len,
            "headerSize": HEADER_SIZE,
            "tagSize": TAG_SIZE,
            "pngSize": len(self.png),
            "overheadBytes": len(self.container) - self.metadata.size,
            "sha256": self.metadata.sha256,
            "capacity": self.capacity.describe(len(self.container)),
            "timingsMs": self.timings_ms,
            "header": self.header.describe(),
            "cipherHistogram": histogram,
            "cipherEntropy": stego.shannon_entropy(histogram),
        }


@dataclass
class DecodeResult:
    file_bytes: bytes
    metadata: Metadata
    header: Header
    integrity_ok: bool
    recovered_sha256: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    def metrics(self) -> dict[str, Any]:
        return {
            "filename": self.metadata.filename,
            "extension": self.metadata.extension,
            "declaredSize": self.metadata.size,
            "recoveredSize": len(self.file_bytes),
            "expectedSha256": self.metadata.sha256,
            "recoveredSha256": self.recovered_sha256,
            "integrityVerified": self.integrity_ok,
            "sizeMatches": len(self.file_bytes) == self.metadata.size,
            "audio": self.metadata.audio,
            "timingsMs": self.timings_ms,
            "header": self.header.describe(),
        }


class _Clock:
    """Wall-clock timing, so the UI can report where the work went."""

    def __init__(self) -> None:
        self.marks: dict[str, float] = {}
        self._start = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round((now - self._start) * 1000, 2)
        self._start = now


def encode_audio(
    file_bytes: bytes,
    filename: str,
    password: str,
    audio_info: dict[str, Any] | None = None,
    bits_per_channel: int = 1,
    cover_png: bytes | None = None,
) -> EncodeResult:
    """Encrypt a file and hide the result in a PNG."""
    if not file_bytes:
        raise VaultError("The file is empty")
    if not password:
        raise VaultError("A password is required")

    clock = _Clock()

    extension = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    metadata = Metadata(
        filename=filename or "recovered.bin",
        extension=extension,
        size=len(file_bytes),
        sha256=sha256_hex(file_bytes),
        audio=audio_info or {},
    )
    plaintext = build_plaintext(metadata, file_bytes)
    clock.lap("hash")

    salt, nonce = crypto.random_salt(), crypto.random_nonce()
    key = crypto.derive_key(password, salt)
    clock.lap("keyDerivation")

    # The header is built before encryption because it is the associated
    # data: its final bytes, including the payload length, must be fixed
    # and authenticated. Hence computing the length up front rather than
    # after the fact.
    header = Header(salt=salt, nonce=nonce, payload_len=len(plaintext) + TAG_SIZE)
    header_bytes = header.pack()
    ciphertext = crypto.encrypt(key, nonce, plaintext, header_bytes)
    container = header_bytes + ciphertext
    clock.lap("encryption")

    if cover_png is not None:
        cover = stego.load_cover(cover_png)
        capacity = stego.Capacity(cover.shape[1], cover.shape[0], bits_per_channel)
        if capacity.total_bytes < len(container):
            raise VaultError(
                f"Cover image holds {capacity.total_bytes:,} bytes but the container "
                f"needs {len(container):,}. Use a larger image, raise bits per channel, "
                f"or switch to a generated cover."
            )
    else:
        capacity = stego.plan_image(len(container), bits_per_channel)
        cover = stego.generate_cover(capacity.width, capacity.height)
    clock.lap("coverPreparation")

    try:
        pixels = stego.embed(cover, container, bits_per_channel)
    except stego.CapacityError as exc:
        raise VaultError(str(exc)) from exc
    clock.lap("embedding")

    png = stego.to_png(pixels)
    clock.lap("pngEncode")

    return EncodeResult(
        png=png, pixels=pixels, cover=cover, container=container,
        header=header, metadata=metadata, capacity=capacity,
        timings_ms=clock.marks,
    )


def read_header(png_bytes: bytes, bits_per_channel: int = 1) -> tuple[Header, np.ndarray]:
    """Pull just the cleartext header out of a PNG. No password needed."""
    try:
        pixels = stego.from_png(png_bytes)
    except Exception as exc:  # noqa: BLE001 — Pillow raises a wide variety
        raise VaultError(f"Could not read this file as an image: {exc}") from exc

    try:
        raw = stego.extract(pixels, HEADER_SIZE, bits_per_channel)
        return Header.unpack(raw), pixels
    except (ContainerError, stego.CapacityError) as exc:
        raise VaultError(str(exc)) from exc


def inspect_png(png_bytes: bytes, bits_per_channel: int = 1) -> dict[str, Any]:
    """What can be learned about a container without the password."""
    header, pixels = read_header(png_bytes, bits_per_channel)
    capacity = stego.Capacity(pixels.shape[1], pixels.shape[0], bits_per_channel)
    return {
        "valid": True,
        "header": header.describe(),
        "capacity": capacity.describe(HEADER_SIZE + header.payload_len),
        "pngSize": len(png_bytes),
    }


def decode_png(png_bytes: bytes, password: str, bits_per_channel: int = 1) -> DecodeResult:
    """Recover the original file from a PNG, or fail loudly."""
    if not password:
        raise VaultError("A password is required")

    clock = _Clock()
    header, pixels = read_header(png_bytes, bits_per_channel)
    header_bytes = header.pack()

    capacity = stego.Capacity(pixels.shape[1], pixels.shape[0], bits_per_channel)
    total_needed = HEADER_SIZE + header.payload_len
    if total_needed > capacity.total_bytes:
        raise VaultError(
            "This container declares more data than the image can hold — "
            "the image has probably been cropped or re-encoded."
        )

    try:
        blob = stego.extract(pixels, total_needed, bits_per_channel)
    except stego.CapacityError as exc:
        raise VaultError(str(exc)) from exc
    ciphertext = blob[HEADER_SIZE:]
    clock.lap("extraction")

    key = crypto.derive_key(password, header.salt, header.log2_n, header.r, header.p)
    clock.lap("keyDerivation")

    try:
        plaintext = crypto.decrypt(key, header.nonce, ciphertext, header_bytes)
    except crypto.DecryptionError as exc:
        raise VaultError(str(exc)) from exc
    clock.lap("decryption")

    try:
        metadata, file_bytes = split_plaintext(plaintext)
    except ContainerError as exc:
        raise VaultError(str(exc)) from exc

    # The GCM tag has already proven the bytes are unaltered. This second,
    # independent check catches a different class of problem — a bug in
    # this code's own framing, where authentic bytes are reassembled
    # wrongly — and is what the UI reports as bit-for-bit verification.
    recovered_sha = sha256_hex(file_bytes)
    integrity_ok = recovered_sha == metadata.sha256 and len(file_bytes) == metadata.size
    clock.lap("verification")

    return DecodeResult(
        file_bytes=file_bytes, metadata=metadata, header=header,
        integrity_ok=integrity_ok, recovered_sha256=recovered_sha,
        timings_ms=clock.marks,
    )
