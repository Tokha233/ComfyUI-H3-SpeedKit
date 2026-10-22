import threading
import unittest

from h3_speedkit.stream_export import StreamExport


class StreamExportTest(unittest.TestCase):
    def test_ordered_bounded_queue_and_finish(self):
        entered, release, sent = threading.Event(), threading.Event(), threading.Event()
        seen = []

        def encode(items, *args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            seen.extend(items)
            return "finished.mp4"

        export = StreamExport((4, 2, 2, 3), 24, "unused", exporter=encode)
        self.assertTrue(entered.wait(3))
        export.write(0, "a")
        export.write(1, "b")

        def submit():
            export.write(2, "c")
            sent.set()

        producer = threading.Thread(target=submit)
        producer.start()
        self.assertFalse(sent.wait(.05))
        release.set()
        producer.join(3)
        self.assertTrue(sent.is_set())
        self.assertEqual(export.finish(), "finished.mp4")
        self.assertEqual(export.finish(), "finished.mp4")
        self.assertEqual(seen, [(0, "a"), (1, "b"), (2, "c")])
        self.assertFalse(export.thread.is_alive())

    def test_codec_failure_unblocks_full_queue(self):
        entered, fail, done = threading.Event(), threading.Event(), threading.Event()
        errors = []

        def encode(*args, **kwargs):
            entered.set()
            self.assertTrue(fail.wait(3))
            raise OSError("disk full")

        export = StreamExport((4, 2, 2, 3), 24, "unused", exporter=encode)
        self.assertTrue(entered.wait(3))
        export.write(0, "a")
        export.write(1, "b")

        def submit():
            try:
                export.write(2, "c")
            except OSError as error:
                errors.append(str(error))
            finally:
                done.set()

        producer = threading.Thread(target=submit)
        producer.start()
        self.assertFalse(done.wait(.05))
        fail.set()
        producer.join(3)
        self.assertTrue(done.is_set())
        self.assertEqual(errors, ["disk full"])
        with self.assertRaisesRegex(OSError, "disk full"):
            export.finish()
        self.assertFalse(export.thread.is_alive())
        self.assertTrue(export.queue.empty())

    def test_decode_error_cancels_waiting_encoder(self):
        seen = threading.Event()

        def encode(items, *args, **kwargs):
            for item in items:
                seen.set()

        export = StreamExport((4, 2, 2, 3), 24, "unused", exporter=encode)
        with self.assertRaisesRegex(ValueError, "decode failed"):
            with export:
                export.write(0, "a")
                self.assertTrue(seen.wait(3))
                raise ValueError("decode failed")
        self.assertFalse(export.thread.is_alive())
        self.assertTrue(export.queue.empty())
        export.abort()
        with self.assertRaises(RuntimeError):
            export.write(1, "b")


if __name__ == "__main__":
    unittest.main()
