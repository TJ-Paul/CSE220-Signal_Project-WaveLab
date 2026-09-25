"""LSB steganography: storing a bitstream in the low bits of pixels.

An 8-bit colour channel is a number from 0 to 255. Changing its lowest
bit moves it by exactly one step — 183 becomes 182 or stays 183. That is
1/256 of the channel's range, far below the roughly 1% contrast
difference the eye can resolve, so the image looks unchanged while every
channel quietly carries one bit of someone else's data.

CAPACITY
--------
With one bit per channel in an RGB image:

    capacity_bits  = width × height × 3
    capacity_bytes = width × height × 3 / 8

Audio is big, so this matters: a three-megabyte file needs 24 million
bits and therefore 8 million pixels — a 2829 × 2829 image. The arithmetic
runs in both directions here, sizing an image to fit a payload and
checking a payload against an existing image.

Raising `bits_per_channel` to 2 doubles capacity and quarters the image
area, at the cost of moving each channel by up to 3 steps instead of 1.
On a noise cover that is still invisible; on a photograph it starts to
show in smooth gradients like skies.

WHY THE GENERATED COVER IS NOISE
---------------------------------
It would be prettier to generate a gradient. Noise is the right choice
for a reason beyond looks: the low bits of a smooth image are *not*
random. In a gradient or a photograph, neighbouring pixels are strongly
correlated, and that correlation reaches down into the LSB plane.
Replacing those bits with ciphertext — which is statistically random —
leaves a detectable signature, and the classical attacks on LSB
steganography (chi-squared, RS analysis) look for exactly that change in
the low-bit statistics.

Uniform random noise has LSBs that are already independent and
uniformly distributed, which is indistinguishable from what ciphertext
looks like. Embedding into noise therefore changes the image's
statistics not at all: there is no anomaly left to detect. The cost is
that noise is incompressible, so the PNG is large — a real trade of file
size for undetectability, and one the UI reports rather than hides.

BIT ORDER
---------
Bits are written most-significant first within each byte, and channels
are visited in row-major R, G, B order. Any consistent convention works;
what matters is that the writer and reader agree, so both directions go
through the two functions below rather than re-deriving the order.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

CHANNELS = 3  # RGB
MAX_BITS_PER_CHANNEL = 4


class CapacityError(Exception):
    """The payload does not fit in the carrier image."""


@dataclass
class Capacity:
    width: int
    height: int
    bits_per_channel: int

    @property
    def total_bits(self) -> int:
        return self.width * self.height * CHANNELS * self.bits_per_channel

    @property
    def total_bytes(self) -> int:
        return self.total_bits // 8

    def describe(self, used_bytes: int) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "pixels": self.width * self.height,
            "bitsPerChannel": self.bits_per_channel,
            "bitsPerPixel": CHANNELS * self.bits_per_channel,
            "capacityBits": self.total_bits,
            "capacityBytes": self.total_bytes,
            "usedBytes": used_bytes,
            "remainingBytes": max(0, self.total_bytes - used_bytes),
            "utilization": (used_bytes / self.total_bytes) if self.total_bytes else 0.0,
        }


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Expand bytes into a flat array of 0/1, most significant bit first."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Inverse of `bytes_to_bits`. Length must be a multiple of 8."""
    return np.packbits(bits.astype(np.uint8)).tobytes()


def plan_image(payload_bytes: int, bits_per_channel: int = 1,
               aspect: float = 1.0) -> Capacity:
    """Smallest image that holds `payload_bytes`, near the given aspect ratio.

    Sized deliberately square-ish: for a fixed area a square has the
    smallest maximum dimension, which keeps very large payloads inside
    the pixel limits that image viewers and browsers impose.
    """
    if bits_per_channel < 1 or bits_per_channel > MAX_BITS_PER_CHANNEL:
        raise CapacityError(f"bits_per_channel must be 1–{MAX_BITS_PER_CHANNEL}")

    needed_bits = payload_bytes * 8
    needed_pixels = math.ceil(needed_bits / (CHANNELS * bits_per_channel))
    width = max(1, math.ceil(math.sqrt(needed_pixels * aspect)))
    height = max(1, math.ceil(needed_pixels / width))

    capacity = Capacity(width, height, bits_per_channel)
    if capacity.total_bits < needed_bits:  # rounding guard
        capacity = Capacity(width, height + 1, bits_per_channel)
    return capacity


def generate_cover(width: int, height: int, seed: int | None = None) -> np.ndarray:
    """A uniform-random RGB cover, as an (h, w, 3) uint8 array.

    `seed` exists so tests can reproduce an image; leaving it None uses
    fresh entropy, which is what real use wants — a cover that is
    predictable from a seed is a cover an adversary can subtract.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, CHANNELS), dtype=np.uint8)


def load_cover(data: bytes) -> np.ndarray:
    """Read a carrier image and force it to 8-bit RGB.

    Palette and greyscale images are converted rather than rejected;
    anything with an alpha channel loses it, because a PNG's alpha is
    where naive viewers and optimisers are most likely to meddle.
    """
    with Image.open(io.BytesIO(data)) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def embed(cover: np.ndarray, payload: bytes, bits_per_channel: int = 1) -> np.ndarray:
    """Write `payload` into the low bits of `cover`. Returns a new array.

    Implemented as whole-array numpy operations rather than a Python
    loop over pixels: a few megabytes of audio is tens of millions of
    bits, and a per-bit loop would take minutes where this takes
    milliseconds.
    """
    flat = cover.reshape(-1).copy()
    bits = bytes_to_bits(payload)

    slots = flat.size * bits_per_channel
    if bits.size > slots:
        raise CapacityError(
            f"Payload needs {bits.size:,} bits but the image holds {slots:,}"
        )

    # Pad to a whole number of channels so the reshape below is exact.
    # The padding is random rather than zeros: a long run of zeros in the
    # LSB plane past the end of the payload is precisely the kind of
    # anomaly that makes an embedding detectable.
    if bits.size < slots:
        rng = np.random.default_rng()
        padding = rng.integers(0, 2, size=slots - bits.size, dtype=np.uint8)
        bits = np.concatenate([bits, padding])

    # Group the bitstream so each channel receives `bits_per_channel` of
    # them, most significant of the group first.
    grouped = bits.reshape(-1, bits_per_channel)
    weights = (1 << np.arange(bits_per_channel - 1, -1, -1)).astype(np.uint8)
    values = (grouped * weights).sum(axis=1).astype(np.uint8)

    mask = np.uint8(0xFF ^ ((1 << bits_per_channel) - 1))
    flat = (flat & mask) | values
    return flat.reshape(cover.shape)


def extract(stego: np.ndarray, byte_count: int, bits_per_channel: int = 1) -> bytes:
    """Read `byte_count` bytes back out of the low bits."""
    needed_bits = byte_count * 8
    channels_needed = math.ceil(needed_bits / bits_per_channel)

    flat = stego.reshape(-1)
    if channels_needed > flat.size:
        raise CapacityError("Image is too small to contain the declared payload")

    values = flat[:channels_needed].astype(np.uint8)
    shifts = np.arange(bits_per_channel - 1, -1, -1, dtype=np.uint8)
    bits = ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return bits_to_bytes(bits[:needed_bits])


def to_png(pixels: np.ndarray) -> bytes:
    """Encode to PNG.

    PNG and not JPEG, and the reason is the whole reason this works:
    JPEG is lossy. It quantises in the frequency domain and will happily
    change a pixel by a step or two, which is invisible to a viewer and
    completely destroys a payload living in the lowest bit. PNG's
    DEFLATE is lossless, so every pixel survives exactly.
    """
    buf = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def from_png(data: bytes) -> np.ndarray:
    return load_cover(data)


def pixel_comparison(cover: np.ndarray, stego: np.ndarray, count: int = 8,
                     offset: int = 0) -> list[dict]:
    """Before/after for the first few pixels, for the teaching view."""
    flat_cover = cover.reshape(-1, CHANNELS)
    flat_stego = stego.reshape(-1, CHANNELS)
    total = min(len(flat_cover), len(flat_stego))
    start = max(0, min(offset, max(0, total - count)))

    rows = []
    for i in range(start, min(start + count, total)):
        before = flat_cover[i]
        after = flat_stego[i]
        rows.append({
            "index": int(i),
            "x": int(i % cover.shape[1]),
            "y": int(i // cover.shape[1]),
            "before": [int(v) for v in before],
            "after": [int(v) for v in after],
            "beforeBits": [format(int(v), "08b") for v in before],
            "afterBits": [format(int(v), "08b") for v in after],
            "embedded": [int(v) & 1 for v in after],
            "changed": [bool(int(a) != int(b)) for a, b in zip(before, after)],
        })
    return rows


def byte_histogram(data: bytes, sample_limit: int = 4_000_000) -> list[int]:
    """Counts of each byte value 0–255.

    Ciphertext should look flat here — that flatness is the visible
    consequence of encryption having destroyed every structure the audio
    container had. Large inputs are sampled from the front; the
    distribution of a few megabytes is already the distribution.
    """
    view = np.frombuffer(data[:sample_limit], dtype=np.uint8)
    return np.bincount(view, minlength=256).tolist()


def shannon_entropy(histogram: list[int]) -> float:
    """Entropy in bits per byte. Random data approaches the 8.0 ceiling."""
    counts = np.array(histogram, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum())
