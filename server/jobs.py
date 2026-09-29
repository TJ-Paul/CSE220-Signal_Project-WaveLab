"""Background jobs for the ML tools (Demucs, Whisper).

A phone that asks for karaoke or lyrics should not have to hold one HTTP
request open for up to a minute: a locked screen or a switched app drops
it, and the guest loses the result even though the Mac finishes the work.
So the request only *starts* a job and returns its id at once; the page
then asks for the job's state every second and picks up the result
whenever it is ready.

Two more things fall out of this:

- One GPU lane. The models share one GPU, so ML jobs run one at a time in
  the order they were asked for, and each guest is told how many are ahead.
- Same request, same job. Jobs are keyed by the audio's content and the
  settings. If ten phones ask for karaoke of the same song, the first one
  starts the job and the other nine join it; once it is done, anyone who
  asks again (after a page reload, say) gets the result back instantly.

Jobs that don't touch the GPU (the classical DSP engine) start at once on
a thread of their own, so they never wait behind a minute of Whisper. They
can still take half a minute on a full song, so they get the same polling
and keying as the ML jobs.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(eq=False)
class Job:
    id: str
    key: str
    fn: Callable[[], dict] | None
    status: str = "queued"  # "queued" | "running" | "done" | "error"
    result: dict | None = None
    error: str | None = None
    started: float | None = None
    finished: float | None = None


class JobQueue:
    def __init__(self, still_valid: Callable[[dict], bool]) -> None:
        #: Whether a finished result can still be handed out, e.g. that the
        #: signals it points at have not been deleted since.
        self._still_valid = still_valid
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, Job] = {}
        self._pending: deque[Job] = deque()
        self._running: Job | None = None  # the GPU lane's current job
        self._cv = threading.Condition()
        self._worker: threading.Thread | None = None

    def submit(self, key: str, fn: Callable[[], dict], gpu: bool = True) -> tuple[Job, bool]:
        """Start the job for `key`, or join the one already there.

        Returns (job, reused); reused is True when an earlier request had
        already started or finished the same work.
        """
        with self._cv:
            job = self._by_key.get(key)
            if job is not None and (
                job.status in ("queued", "running")
                or (job.status == "done" and self._still_valid(job.result))
            ):
                return job, True
            job = Job(id=uuid.uuid4().hex[:12], key=key, fn=fn)
            self._jobs[job.id] = job
            self._by_key[key] = job
            if gpu:
                self._pending.append(job)
                if self._worker is None or not self._worker.is_alive():
                    self._worker = threading.Thread(target=self._loop, name="ml-jobs", daemon=True)
                    self._worker.start()
                self._cv.notify()
            else:
                job.status = "running"
                job.started = time.monotonic()
                threading.Thread(target=self._run, args=(job,), name="dsp-job", daemon=True).start()
        return job, False

    def snapshot(self, job_id: str) -> dict[str, Any] | None:
        """The job's state as JSON, or None if the server no longer has it."""
        with self._cv:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            ahead = None
            if job.status == "queued":
                index = next(i for i, j in enumerate(self._pending) if j is job)
                ahead = index + (self._running is not None)
            elapsed = None
            if job.started is not None:
                elapsed = (job.finished or time.monotonic()) - job.started
            return {
                "id": job.id,
                "status": job.status,
                "ahead": ahead,
                "elapsedS": elapsed,
                "result": job.result,
                "error": job.error,
            }

    def forget_finished(self) -> None:
        """Drop finished jobs and their results (when the session is cleared)."""
        with self._cv:
            for job in list(self._jobs.values()):
                if job.status in ("done", "error"):
                    del self._jobs[job.id]
                    if self._by_key.get(job.key) is job:
                        del self._by_key[job.key]

    def _loop(self) -> None:
        while True:
            with self._cv:
                while not self._pending:
                    self._cv.wait()
                job = self._running = self._pending.popleft()
            try:
                self._run(job)
            finally:
                with self._cv:
                    self._running = None

    def _run(self, job: Job) -> None:
        with self._cv:
            job.status = "running"
            job.started = time.monotonic()
        try:
            result = job.fn()
        except Exception as exc:  # noqa: BLE001 — reported to the page, not raised
            traceback.print_exc()
            message = getattr(exc, "detail", None) or str(exc) or type(exc).__name__
            with self._cv:
                job.status, job.error = "error", str(message)
        else:
            with self._cv:
                job.status, job.result = "done", result
        finally:
            with self._cv:
                job.finished = time.monotonic()
                job.fn = None  # release the audio the closure holds
