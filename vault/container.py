"""The on-image container format.

A stego image is not a pile of bits; it is a structured document, and
the structure has to be readable *before* anything is decrypted. The
decoder must be able to answer "is this even one of mine, and how long
is the payload?" while still knowing nothing about the password.

LAYOUT
------
    ┌──────────────────────── 48-byte header (cleartext) ─────────────┐
      0    4   magic        b"SGVL"
      4    1   version      format version (1)
      5    1   kdf_id       1 = scrypt
      6    1   log2_n       scrypt cost: N = 2**log2_n
      7    1   r            scrypt block size
      8    1   p            scrypt parallelism
      9    3   reserved     zeros, for future flags
     12   16   salt         random, per encryption
     28   12   nonce        random, per encryption (96-bit, GCM's native)
     40    8   payload_len  length of ciphertext+tag, uint64 big-endian
    └─────────────────────────────────────────────────────────────────┘
    ┌──────── ciphertext ────────┐
     payload_len bytes: AES-256-GCM output, ending in its 16-byte tag
    └────────────────────────────┘

The plaintext inside the ciphertext is itself structured:

     0    4   metadata_len, uint32 big-endian
     4    n   metadata, UTF-8 JSON
     4+n  ..  the original file, byte for byte

WHY THE FILENAME AND CHECKSUM ARE *INSIDE* THE CIPHERTEXT
----------------------------------------------------------
It would be simpler to put the filename, the original size and the
SHA-256 digest in the cleartext header, and plenty of file formats do.
This one does not, because each of those leaks:

- A filename is often the most sensitive part of a file. "interview
  with source.wav" gives away more than the audio usually does.
- A cleartext SHA-256 turns the container into an oracle. An attacker
  who guesses the plaintext can hash their guess and compare — so for
  any file drawn from a small or public set, confidentiality is gone
  without the password ever being attacked.
- An exact original size narrows the candidate set considerably.

So the header carries only what is genuinely needed to *attempt*
decryption — the KDF parameters, the salt, the nonce, and how many bytes
to read — and everything descriptive lives under encryption.

WHY THE HEADER IS AUTHENTICATED ANYWAY
---------------------------------------
Cleartext does not mean unprotected. The whole 48-byte header is passed
to AES-GCM as *associated data*: not encrypted, but covered by the
authentication tag. An attacker who edits the salt, the nonce or the
length to steer the decoder gets an authentication failure rather than a
decoder that quietly does what they asked. Binding a message to its own
header this way is the standard defence against such splicing attacks.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any

MAGIC = b"SGVL"
VERSION = 1
KDF_SCRYPT = 1

HEADER_SIZE = 48
SALT_SIZE = 16
#: 96 bits is GCM's native nonce width — the one size the standard uses
#: without an extra hashing step, and the only one with the usual
#: security proof.
NONCE_SIZE = 12
#: AES-GCM appends a 128-bit authentication tag to every ciphertext.
TAG_SIZE = 16

_HEADER_STRUCT = struct.Struct(">4sBBBBB3x16s12sQ")
assert _HEADER_STRUCT.size == HEADER_SIZE


class ContainerError(Exception):
    """The bytes are not a valid vault container."""


@dataclass
class Header:
    """The cleartext preamble — everything needed to attempt decryption."""

    salt: bytes
    nonce: bytes
    payload_len: int
    log2_n: int = 15
    r: int = 8
    p: int = 1
    version: int = VERSION
    kdf_id: int = KDF_SCRYPT

    def pack(self) -> bytes:
        return _HEADER_STRUCT.pack(
            MAGIC, self.version, self.kdf_id, self.log2_n, self.r, self.p,
            self.salt, self.nonce, self.payload_len,
        )

    @classmethod
    def unpack(cls, raw: bytes) -> "Header":
        if len(raw) < HEADER_SIZE:
            raise ContainerError("Truncated header — this image is too small to hold a container")
        magic, version, kdf_id, log2_n, r, p, salt, nonce, payload_len = _HEADER_STRUCT.unpack(
            raw[:HEADER_SIZE]
        )
        if magic != MAGIC:
            raise ContainerError(
                "No vault container found in this image — the magic number does not match"
            )
        if version != VERSION:
            raise ContainerError(f"Unsupported container version {version} (this build reads v{VERSION})")
        if kdf_id != KDF_SCRYPT:
            raise ContainerError(f"Unknown key derivation function id {kdf_id}")
        # A corrupt length field would otherwise drive a huge allocation
        # before the tag ever gets a chance to reject the image.
        if payload_len < TAG_SIZE:
            raise ContainerError("Declared payload is shorter than an authentication tag")
        return cls(salt=salt, nonce=nonce, payload_len=payload_len,
                   log2_n=log2_n, r=r, p=p, version=version, kdf_id=kdf_id)

    def describe(self) -> dict[str, Any]:
        """Everything a reader may know without the password."""
        return {
            "magic": MAGIC.decode(),
            "version": self.version,
            "kdf": "scrypt",
            "scryptN": 2**self.log2_n,
            "scryptR": self.r,
            "scryptP": self.p,
            "saltHex": self.salt.hex(),
            "nonceHex": self.nonce.hex(),
            "payloadLength": self.payload_len,
            "headerSize": HEADER_SIZE,
            "tagSize": TAG_SIZE,
        }


@dataclass
class Metadata:
    """The descriptive block, carried inside the encrypted payload."""

    filename: str
    extension: str
    size: int
    sha256: str
    audio: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps({
            "filename": self.filename,
            "extension": self.extension,
            "size": self.size,
            "sha256": self.sha256,
            "audio": self.audio,
        }, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "Metadata":
        try:
            data = json.loads(raw.decode("utf-8"))
            return cls(
                filename=str(data["filename"]),
                extension=str(data["extension"]),
                size=int(data["size"]),
                sha256=str(data["sha256"]),
                audio=data.get("audio") or {},
            )
        except (ValueError, KeyError, UnicodeDecodeError) as exc:
            raise ContainerError(f"Metadata block is malformed: {exc}") from exc


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_plaintext(metadata: Metadata, file_bytes: bytes) -> bytes:
    """Metadata and file body, as the single buffer that gets encrypted."""
    blob = metadata.to_json()
    return struct.pack(">I", len(blob)) + blob + file_bytes


def split_plaintext(plaintext: bytes) -> tuple[Metadata, bytes]:
    """Inverse of `build_plaintext`.

    Only ever called on plaintext that GCM has already authenticated, so
    these checks guard against a *bug*, not an attacker — by this point
    an attacker's edits have already been rejected by the tag.
    """
    if len(plaintext) < 4:
        raise ContainerError("Decrypted payload is too short to contain a metadata block")
    (blob_len,) = struct.unpack(">I", plaintext[:4])
    if 4 + blob_len > len(plaintext):
        raise ContainerError("Metadata block length runs past the end of the payload")
    metadata = Metadata.from_json(plaintext[4 : 4 + blob_len])
    return metadata, plaintext[4 + blob_len :]
