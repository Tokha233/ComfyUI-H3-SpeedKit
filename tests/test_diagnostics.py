"""Check that the lightweight report never claims to enable a GPU backend."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from h3_speedkit.diagnostics import collect_environment


class DiagnosticsTest(unittest.TestCase):
    def test_default_does_not_import_torch(self):
        import builtins
        original_import = builtins.__import__

        def guarded(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("Default report must not initialize torch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded):
            report = collect_environment()
        self.assertEqual(report["cuda_probe"], "not_requested")
        self.assertFalse(report["acceleration_enabled"])

    def test_failed_probe_does_not_expose_raw_error(self):
        def fail():
            raise RuntimeError("private-host/path/key")

        fake = SimpleNamespace(version=SimpleNamespace(cuda="13.0"),
                               cuda=SimpleNamespace(is_available=fail))
        with patch.dict(sys.modules, {"torch": fake}):
            report = collect_environment(probe_cuda=True)
        self.assertEqual(report["cuda_probe_error_type"], "RuntimeError")
        self.assertNotIn("private-host", str(report))

    def test_arch_match_is_not_a_support_claim(self):
        device = SimpleNamespace(name="test SM120", major=12, minor=0, total_memory=24 << 30)
        fake = SimpleNamespace(version=SimpleNamespace(cuda="13.0"), cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_properties=lambda index: device))
        with patch.dict(sys.modules, {"torch": fake}):
            report = collect_environment(probe_cuda=True)
        self.assertTrue(report["devices"][0]["sm120_candidate"])
        self.assertFalse(report["devices"][0]["validated_by_this_package"])
        self.assertFalse(report["acceleration_enabled"])


if __name__ == "__main__":
    unittest.main()
