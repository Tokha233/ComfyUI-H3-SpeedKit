# SPDX-License-Identifier: Apache-2.0
"""Optional bounded CPU exporter for a service which owns its GPU request loop."""
from concurrent.futures import ThreadPoolExecutor
from threading import Condition
from .export import export_mp4


class ExportQueue:
    """Submit completed CPU frames; count/byte backpressure bounds retained outputs.

    Inputs are transferred, not copied: never mutate a submitted tensor until
    its Future completes. Success means the final MP4 has been atomically saved.
    Call close() to drain and surface failures. No CUDA calls or hidden upload.
    """
    def __init__(self, *, max_jobs=2, max_bytes=2 * 1024**3, exporter=export_mp4):
        if max_jobs < 1 or max_bytes < 1:
            raise ValueError("Queue limits must be positive")
        self.max_jobs, self.max_bytes = max_jobs, max_bytes
        self.exporter = exporter
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-export")
        self.condition = Condition()
        self.jobs = self.bytes = 0
        self.futures = []
        self.closed = False

    def submit(self, frames, path, audio=None, **kwargs):
        if frames.pixels.device.type != "cpu":
            raise ValueError("Finish GPU transfer before queueing")
        tensors = [frames.pixels]
        if audio is not None:
            if audio["waveform"].device.type != "cpu":
                raise ValueError("Move PCM to CPU before queueing")
            tensors.append(audio["waveform"])
        size = sum(t.untyped_storage().nbytes() for t in tensors)
        if size > self.max_bytes:
            raise ValueError("One output exceeds the queue byte budget")
        with self.condition:
            self.condition.wait_for(lambda: self.closed or
                self.jobs < self.max_jobs and self.bytes + size <= self.max_bytes)
            if self.closed:
                raise RuntimeError("Export queue is closed")
            self.jobs += 1
            self.bytes += size
            try:
                future = self.executor.submit(self.exporter, frames, path, audio, **kwargs)
            except BaseException:
                self.jobs -= 1
                self.bytes -= size
                self.condition.notify_all()
                raise
            self.futures.append(future)
        def finished(_):
            with self.condition:
                self.jobs -= 1
                self.bytes -= size
                self.condition.notify_all()
        future.add_done_callback(finished)
        return future

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.executor.shutdown(wait=True, cancel_futures=False)
        failures = [f.exception() for f in self.futures if not f.cancelled() and f.exception() is not None]
        if failures:
            raise RuntimeError(f"{len(failures)} export job(s) failed") from failures[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
