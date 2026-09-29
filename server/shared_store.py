"""Files published to everyone on the network — the room's drop box.

Unlike the vault store these live on disk, under ``shared_files/``, so files
prepared before a demo survive a server restart. Each file sits in its own
directory named by a random id; the id is the only thing a client can name,
so no request can reach a path outside the store.
"""
from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "shared_files"
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

_ID = re.compile(r"^[0-9a-f]{12}$")
_KINDS = {".png": "image", ".wav": "audio", ".flac": "audio", ".mp3": "audio",
          ".ogg": "audio", ".m4a": "audio", ".aac": "audio"}


class SharedStoreError(Exception):
    pass


def _safe_name(name: str) -> str:
    name = re.split(r"[\\/]", name)[-1]
    name = "".join(c for c in name if c.isprintable()).strip(" .")
    return name[:120] or "file"


class SharedStore:
    def __init__(self, root: Path = ROOT) -> None:
        self._root = root
        self._lock = threading.RLock()

    def _entry(self, file_id: str) -> Path | None:
        if not _ID.match(file_id):
            return None
        folder = self._root / file_id
        try:
            # Dotfiles are in-progress writes; _safe_name never produces one.
            files = [p for p in folder.iterdir() if p.is_file() and not p.name.startswith(".")]
        except OSError:
            return None
        return files[0] if len(files) == 1 else None

    def _describe(self, path: Path) -> dict:
        stat = path.stat()
        return {
            "id": path.parent.name,
            "name": path.name,
            "size": stat.st_size,
            "sharedAt": stat.st_mtime,
            "kind": _KINDS.get(path.suffix.lower(), "file"),
        }

    def list(self) -> list[dict]:
        with self._lock:
            if not self._root.is_dir():
                return []
            entries = [self._entry(d.name) for d in self._root.iterdir() if d.is_dir()]
            described = [self._describe(p) for p in entries if p is not None]
        return sorted(described, key=lambda f: f["sharedAt"], reverse=True)

    def total_bytes(self) -> int:
        return sum(f["size"] for f in self.list())

    def add(self, name: str, data: bytes) -> dict:
        if not data:
            raise SharedStoreError("The file is empty.")
        if len(data) > MAX_FILE_BYTES:
            raise SharedStoreError(f"Files up to {MAX_FILE_BYTES // (1024 * 1024)} MB can be shared.")
        with self._lock:
            if self.total_bytes() + len(data) > MAX_TOTAL_BYTES:
                raise SharedStoreError("The shared folder is full — remove a file first.")
            folder = self._root / uuid.uuid4().hex[:12]
            folder.mkdir(parents=True)
            target = folder / _safe_name(name)
            partial = folder / ".partial"
            partial.write_bytes(data)
            os.replace(partial, target)  # never list a half-written file
            now = time.time()
            os.utime(target, (now, now))
            return self._describe(target)

    def path(self, file_id: str) -> Path | None:
        with self._lock:
            return self._entry(file_id)

    def remove(self, file_id: str) -> bool:
        with self._lock:
            if self._entry(file_id) is None:
                return False
            shutil.rmtree(self._root / file_id)
            return True


shared_store = SharedStore()
