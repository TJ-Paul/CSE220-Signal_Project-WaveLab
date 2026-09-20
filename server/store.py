"""In-memory signal store.

Signals live for the lifetime of the server process. Each entry keeps the
float32 mono waveform plus provenance (how it was derived) so the frontend
can offer any pair of them for comparison.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Signal:
    id: str
    name: str
    y: np.ndarray
    sr: int
    origin: str  # "upload" | "demo" | "derived"
    source_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.y) / self.sr

    def summary(self) -> dict[str, Any]:
        y64 = self.y.astype(np.float64)
        peak = float(np.max(np.abs(y64))) if len(y64) else 0.0
        rms = float(np.sqrt(np.mean(y64**2))) if len(y64) else 0.0
        return {
            "id": self.id,
            "name": self.name,
            "sampleRate": self.sr,
            "samples": int(len(self.y)),
            "duration": self.duration,
            "origin": self.origin,
            "sourceId": self.source_id,
            "peak": peak,
            "rms": rms,
            "peakDb": float(20 * np.log10(peak + 1e-12)),
            "crestFactor": float(peak / (rms + 1e-12)),
            **self.meta,
        }


class SignalStore:
    def __init__(self) -> None:
        self._signals: dict[str, Signal] = {}
        self._lock = threading.Lock()

    def add(self, name: str, y: np.ndarray, sr: int, origin: str,
            source_id: str | None = None, **meta: Any) -> Signal:
        sig = Signal(
            id=uuid.uuid4().hex[:12],
            name=name,
            y=np.ascontiguousarray(y, dtype=np.float32),
            sr=int(sr),
            origin=origin,
            source_id=source_id,
            meta=meta,
        )
        with self._lock:
            self._signals[sig.id] = sig
        return sig

    def get(self, signal_id: str) -> Signal | None:
        return self._signals.get(signal_id)

    def list(self) -> list[Signal]:
        return list(self._signals.values())

    def remove(self, signal_id: str) -> None:
        with self._lock:
            self._signals.pop(signal_id, None)

    def clear(self) -> None:
        with self._lock:
            self._signals.clear()


store = SignalStore()
