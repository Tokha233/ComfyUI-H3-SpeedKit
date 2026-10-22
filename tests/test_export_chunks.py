import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from h3_speedkit.export import export_chunks, export_mp4
from h3_speedkit.stream_export import StreamExport


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("torch", "av", "numpy")),
                     "Optional CPU codec dependencies are not installed")
class ExportChunksTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        generator = torch.Generator().manual_seed(85)
        self.rgb = torch.randint(0, 256, (19, 32, 48, 3), dtype=torch.uint8, generator=generator)
        self.audio = {"waveform": torch.randn(1, 2, 32000, generator=generator) * .05,
                      "sample_rate": 32000}

    def test_byte_identical_uneven_chunks_with_and_without_audio(self):
        for audio in (None, self.audio):
            with self.subTest(audio=audio is not None):
                whole, streamed = self.root / "whole.mp4", self.root / "stream.mp4"
                export_mp4(SimpleNamespace(pixels=self.rgb, fps=24), whole, audio)
                with StreamExport(tuple(self.rgb.shape), 24, streamed, audio) as encoder:
                    encoder.write(0, self.rgb[:7])
                    encoder.write(7, self.rgb[7:18])
                    encoder.write(18, self.rgb[18:])
                self.assertEqual(whole.read_bytes(), streamed.read_bytes())

    def test_incomplete_or_out_of_order_never_replaces_existing_file(self):
        output = self.root / "previous.mp4"
        for chunks in ([(0, self.rgb[:7])], [(1, self.rgb)]):
            output.write_bytes(b"previous completed output")
            with self.assertRaises(ValueError):
                export_chunks(chunks, tuple(self.rgb.shape), 24, output)
            self.assertEqual(output.read_bytes(), b"previous completed output")
            self.assertFalse(list(self.root.glob("*.partial.mp4")))

    def test_producer_failure_removes_partial_and_preserves_final(self):
        output = self.root / "previous.mp4"
        output.write_bytes(b"previous completed output")
        with self.assertRaisesRegex(RuntimeError, "decode interrupted"):
            with StreamExport(tuple(self.rgb.shape), 24, output) as encoder:
                encoder.write(0, self.rgb[:7])
                raise RuntimeError("decode interrupted")
        self.assertEqual(output.read_bytes(), b"previous completed output")
        self.assertFalse(list(self.root.glob("*.partial.mp4")))
        self.assertFalse(encoder.thread.is_alive())
