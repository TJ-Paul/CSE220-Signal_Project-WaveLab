"""In-memory holding area for vault artefacts.

Stego PNGs and recovered files live only for the lifetime of the server
process, exactly like the signal store. They are held in memory so the
browser can preview an image and download a file without either ever
touching disk — a vault that scattered plaintext copies of decrypted
audio through a temp directory would undo much of the point.

Because a stego PNG is roughly the size of the audio it carries (noise
does not compress), the store is bounded by total bytes rather than by
item count, and evicts oldest-first when it would exceed the budget.
"""
from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

#: Enough for a handful of songs in a demo session, small enough that a
#: long-running server cannot quietly consume the machine.
MAX_TOTAL_BYTES = 768 * 1024 * 1024


@dataclass
class Artifact:
    id: str
    kind: str  # "image" | "file"
    data: bytes
    filename: str
    content_type: str
    meta: dict[str, Any] = field(default_factory=dict)


class VaultStore:
    def __init__(self, max_total_bytes: int = MAX_TOTAL_BYTES) -> None:
        self._items: OrderedDict[str, Artifact] = OrderedDict()
        self._lock = threading.Lock()
        self._max_total = max_total_bytes

    def add(self, kind: str, data: bytes, filename: str, content_type: str,
            **meta: Any) -> Artifact:
        artifact = Artifact(
            id=uuid.uuid4().hex[:12], kind=kind, data=data,
            filename=filename, content_type=content_type, meta=meta,
        )
        with self._lock:
            self._items[artifact.id] = artifact
            self._evict_locked()
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self._items.get(artifact_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def total_bytes(self) -> int:
        return sum(len(a.data) for a in self._items.values())

    def _evict_locked(self) -> None:
        total = sum(len(a.data) for a in self._items.values())
        while total > self._max_total and len(self._items) > 1:
            _, dropped = self._items.popitem(last=False)
            total -= len(dropped.data)


vault_store = VaultStore()
