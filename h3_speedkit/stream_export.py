# SPDX-License-Identifier: Apache-2.0
"""Request-owned, bounded CPU encoding of finalized RGB8 chunks."""
from queue import Empty, Full, Queue
from threading import Event, Thread

from .export import export_chunks


class StreamExport:
    """Transfer immutable CPU chunk views until finish/abort joins the worker.

    One producer owns this object. A view's backing memory must remain unchanged
    until completion. Codec failures reach the producer even with a full queue.
    """

    def __init__(self, shape, fps, path, audio=None, *, crf=23, threads=4,
                 exporter=export_chunks):
        if audio is not None and audio["waveform"].device.type != "cpu":
            raise ValueError("Move PCM to CPU before starting the encoder")
        self.queue = Queue(maxsize=2)
        self.cancelled = Event()
        self.error = None
        self.result = None
        self.closed = False

        def run():
            try:
                self.result = exporter(self.items(), shape, fps, path, audio,
                                       crf=crf, threads=threads)
            except BaseException as error:
                self.error = error

        self.thread = Thread(target=run, name="h3-stream-export")
        self.thread.start()

    def items(self):
        while True:
            if self.cancelled.is_set():
                raise RuntimeError("Video export cancelled")
            try:
                item = self.queue.get(timeout=0.05)
            except Empty:
                continue
            if item is None:
                return
            yield item

    def put(self, item):
        while True:
            if self.closed:
                raise RuntimeError("Video export is closed")
            if self.error is not None:
                raise self.error
            if not self.thread.is_alive():
                raise RuntimeError("Encoder stopped before receiving all frames")
            try:
                self.queue.put(item, timeout=0.05)
                return
            except Full:
                continue

    def write(self, start, rgb):
        self.put((start, rgb))

    def finish(self):
        if not self.closed:
            try:
                self.put(None)
            except BaseException:
                self.abort()
                raise
            self.thread.join()
            self.closed = True
            self.clear()
        if self.error is not None:
            raise self.error
        if self.cancelled.is_set():
            raise RuntimeError("Video export cancelled")
        return self.result

    def clear(self):
        while True:
            try:
                self.queue.get_nowait()
            except Empty:
                return

    def abort(self):
        if self.closed:
            return
        self.cancelled.set()
        self.thread.join()
        self.closed = True
        self.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        else:
            self.finish()
