import threading
import unittest
from types import SimpleNamespace
from h3_speedkit.service import ExportQueue


def frames(size=16):
    tensor = SimpleNamespace(device=SimpleNamespace(type='cpu'),
                             untyped_storage=lambda: SimpleNamespace(nbytes=lambda: size))
    return SimpleNamespace(pixels=tensor)


class ServiceTest(unittest.TestCase):
    def test_backpressure_and_drain(self):
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        def encode(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return 'done'
        queue = ExportQueue(max_jobs=1, max_bytes=16, exporter=encode)
        first = queue.submit(frames(), 'a')
        self.assertTrue(entered.wait(3))
        def submit():
            queue.submit(frames(), 'b')
            second.set()
        thread = threading.Thread(target=submit)
        thread.start()
        self.assertFalse(second.wait(.05))
        release.set()
        thread.join(3)
        self.assertTrue(second.is_set())
        queue.close()
        self.assertEqual(first.result(), 'done')
        self.assertEqual((queue.jobs, queue.bytes), (0, 0))

    def test_failure_surfaced_and_budget_released(self):
        def encode(*args, **kwargs):
            raise OSError('codec failure')
        queue = ExportQueue(exporter=encode)
        future = queue.submit(frames(), 'a')
        with self.assertRaises(OSError):
            future.result()
        with self.assertRaises(RuntimeError):
            queue.close()
        self.assertEqual(queue.bytes, 0)
        with self.assertRaises(RuntimeError):
            queue.submit(frames(), 'b')

    def test_oversized_job_rejected_before_submission(self):
        with ExportQueue(max_bytes=8) as queue:
            with self.assertRaises(ValueError):
                queue.submit(frames(16), 'a')
