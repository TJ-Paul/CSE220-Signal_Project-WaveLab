"""Secure audio vault — encrypt an audio file and hide it inside a PNG.

    audio file → AES-256-GCM → ciphertext → LSB steganography → PNG
    PNG → LSB extraction → ciphertext → AES-256-GCM → the same audio file

The point of the exercise is the word *same*: what comes out is the
original file byte for byte, not an audio file that merely sounds like
it. Nothing here resamples, re-encodes or normalises anything — the
audio is treated as opaque binary from the moment it is read.

    container   the on-the-wire format: header layout, metadata block,
                SHA-256 integrity, parse and validation
    crypto      key derivation (scrypt) and authenticated encryption
    stego       bit packing, capacity arithmetic, LSB embed/extract,
                cover-image generation
    pipeline    the two end-to-end directions, and the measurements the
                UI reports

Each module is independent of the others' internals and of any UI, so
each can be tested on its own — `python -m vault.selftest` exercises all
of them together.

WHY THESE THREE PIECES, AND NOT ONE
------------------------------------
They are often conflated, so it is worth stating what each contributes:

- **Encryption** provides confidentiality. Without it, hiding data in an
  image is obscurity, not security: anyone who suspects the technique
  reads the payload straight out.
- **Steganography** provides concealment, not confidentiality. It hides
  *that* there is a message, and it is what makes the container a PNG.
- **Authentication** (the GCM tag) and the SHA-256 digest provide
  integrity. Without them a flipped bit yields plausible-looking
  garbage, and the decoder would hand it over as if it were audio.

Remove any one and the system fails at something the other two cannot
cover for.
"""
from .pipeline import (  # noqa: F401
    DecodeResult,
    EncodeResult,
    VaultError,
    decode_png,
    encode_audio,
    inspect_png,
)
