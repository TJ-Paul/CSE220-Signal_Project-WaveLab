"""Key derivation and authenticated encryption.

Nothing in this file is invented. Every primitive is a standard one from
a maintained library, which is the only defensible choice: cryptography
is the one part of a system where a clever original idea is almost
always a defect. What *is* worth explaining is why each standard piece
is here.

PASSWORDS ARE NOT KEYS
-----------------------
AES-256 wants 32 bytes of uniformly random key. A password is neither 32
bytes nor uniform — it is short, structured, and drawn from a
distribution an attacker knows well. Hashing it once with SHA-256 would
produce the right *shape* and none of the strength: an attacker able to
try billions of SHA-256 guesses a second on a GPU would walk through the
entire plausible password space.

A password-based KDF fixes this by making each guess expensive. scrypt
is used here rather than PBKDF2 because PBKDF2 is only
*computation*-hard: it is a long chain of hashes, which is exactly the
workload custom hardware parallelises best, so an attacker with an ASIC
gains orders of magnitude over the defender's CPU. scrypt is
additionally *memory*-hard — it forces a large random-access buffer per
guess, and memory is the one resource an attacker cannot make
arbitrarily cheap. At N = 2^15, r = 8 each guess costs about 32 MB and
tens of milliseconds, which is unnoticeable once and ruinous a billion
times over.

The salt is random per encryption, which is what stops one precomputed
table from covering every file ever produced: the same password used
twice yields two unrelated keys.

WHY AUTHENTICATED ENCRYPTION, AND NOT JUST ENCRYPTION
------------------------------------------------------
Plain AES-CTR (or CBC) hides content but says nothing about whether the
ciphertext is the one that was written. Both are malleable: flipping a
bit of CTR ciphertext flips exactly that bit of the plaintext, so an
attacker who cannot read the payload can still edit it, predictably. The
decoder would then emit an audio file that was silently altered and look
entirely successful doing it.

AES-GCM runs counter-mode encryption and a GHASH authenticator over the
same pass, producing a 128-bit tag. Decryption verifies the tag before
returning anything, so a modified container fails loudly instead of
yielding plausible garbage. This is what makes "wrong password" and
"tampered image" the same kind of event here: both are simply a tag that
does not verify.

NONCE REUSE IS THE ONE UNFORGIVING RULE
----------------------------------------
GCM fails catastrophically if a key and nonce pair is ever reused — it
leaks the XOR of the two plaintexts and, worse, the authentication
subkey, which lets an attacker forge tags at will. Here every encryption
draws a fresh random salt *and* a fresh random nonce from the OS CSPRNG,
so the key differs per file even before the nonce does.
"""
from __future__ import annotations

import hashlib
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .container import NONCE_SIZE, SALT_SIZE

KEY_SIZE = 32  # AES-256

#: scrypt cost parameters. N is the work factor (memory ≈ 128·N·r bytes,
#: so ~32 MB here); r tunes the block size; p the parallelism. These are
#: the interactive-login defaults from the scrypt paper, and are stored
#: in the header so a future build can raise them without breaking old
#: images.
SCRYPT_LOG2_N = 15
SCRYPT_R = 8
SCRYPT_P = 1
#: Without a ceiling, a hostile header could ask for 2^40 and exhaust
#: memory before the tag has a chance to reject the image.
MAX_SCRYPT_LOG2_N = 20


class DecryptionError(Exception):
    """The tag did not verify: wrong password, or the container was altered.

    Deliberately a single error for both causes. Distinguishing them
    would tell an attacker whether a guessed password was right for a
    container they had modified, which is a free oracle.
    """


def random_salt() -> bytes:
    return os.urandom(SALT_SIZE)


def random_nonce() -> bytes:
    return os.urandom(NONCE_SIZE)


def derive_key(password: str, salt: bytes, log2_n: int = SCRYPT_LOG2_N,
               r: int = SCRYPT_R, p: int = SCRYPT_P) -> bytes:
    """Stretch a password into a 256-bit key with scrypt."""
    if not password:
        raise ValueError("A password is required")
    if not 10 <= log2_n <= MAX_SCRYPT_LOG2_N:
        raise ValueError(f"scrypt cost out of range (log2 N = {log2_n})")
    return hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**log2_n,
        r=r,
        p=p,
        maxmem=256 * 1024 * 1024,
        dklen=KEY_SIZE,
    )


def encrypt(key: bytes, nonce: bytes, plaintext: bytes, header: bytes) -> bytes:
    """AES-256-GCM. Returns ciphertext with the 16-byte tag appended.

    `header` is associated data: authenticated, not encrypted, so the
    container's own preamble cannot be edited without detection.
    """
    return AESGCM(key).encrypt(nonce, plaintext, header)


def decrypt(key: bytes, nonce: bytes, payload: bytes, header: bytes) -> bytes:
    """Verify the tag and return the plaintext, or raise DecryptionError.

    The library checks the tag before releasing any plaintext, so a
    failure here means nothing decrypted was ever exposed.
    """
    try:
        return AESGCM(key).decrypt(nonce, payload, header)
    except InvalidTag as exc:
        raise DecryptionError(
            "Authentication failed — the password is wrong, or the image has been modified."
        ) from exc
