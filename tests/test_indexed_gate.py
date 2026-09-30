"""CPU contract checks; the opt-in GPU benchmark checks real output parity."""
import unittest
import copy
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from h3_speedkit import indexed_gate
from h3_speedkit.indexed_gate import Unsupported, segment_key


class SegmentTest(unittest.TestCase):
    def test_ref2va_and_empty_spans(self):
        self.assertEqual(segment_key([(0, 128, 0), (128, 128, 1), (128, 14850, 2)], 14850, 3),
                         ((0, 128, 0), (128, 128, 1), (128, 14850, 2)))

    def test_reject_before_gpu_work(self):
        for segments in ([(0, 4, 0), (5, 8, 1)], [(0, 5, 0), (4, 8, 1)],
                         [(0, 8, -1)], [(0, 8, 3)], [(0, 9, 0)],
                         [(0, 7, 0)], [(0, 8, [0])], [(0, 8, True)]):
            with self.subTest(segments=segments), self.assertRaises(Unsupported):
                segment_key(segments, 8, 3)


class CloneTest(unittest.TestCase):
    def test_wrapper_survives_comfy_option_replacement_and_original_is_untouched(self):
        class FakeModel:
            def __init__(self):
                self.model_options = {"transformer_options": {"keep": "original"}}
                self.model = SimpleNamespace(diffusion_model=object())

            def clone(self):
                cloned = copy.copy(self)
                def copy_options(obj):
                    if isinstance(obj, dict):
                        return {key: copy_options(value) for key, value in obj.items()}
                    if isinstance(obj, list):
                        return [copy_options(value) for value in obj]
                    return obj
                cloned.model_options = copy_options(self.model_options)
                return cloned

            def set_model_patch_replace(self, replacement, name, block, index):
                # Comfy replaces this options dictionary; references captured
                # before all calls cannot be used to install the wrapper.
                options = copy.deepcopy(self.model_options['transformer_options'])
                options.setdefault('patches_replace', {}).setdefault(name, {})[(block, index)] = replacement
                self.model_options['transformer_options'] = options

        modules = {name: ModuleType(name) for name in (
            'comfy', 'comfy.patcher_extension', 'comfy.ldm', 'comfy.ldm.minimax',
            'comfy.ldm.minimax.model', 'comfy_kitchen')}
        modules['comfy'].patcher_extension = modules['comfy.patcher_extension']
        modules['comfy.ldm.minimax'].model = modules['comfy.ldm.minimax.model']
        modules['comfy.ldm.minimax.model'].MiniMaxH3Model = FakeModel
        modules['comfy_kitchen'].int8_gemm_indexed_gate = object()
        pe = modules['comfy.patcher_extension']
        pe.WrappersMP = SimpleNamespace(DIFFUSION_MODEL='diffusion')
        pe.add_wrapper_with_key = lambda kind, key, wrapper, options: options.setdefault('wrappers', {}).update({key: wrapper})
        model = FakeModel()
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        with patch.dict(sys.modules, modules), patch.object(indexed_gate, 'MODEL_SHA', sha), \
                patch.object(indexed_gate.inspect, 'getsourcefile', return_value=__file__):
            fast = indexed_gate.patch_model(model)
            self.assertEqual(model.model_options, {'transformer_options': {'keep': 'original'}})
            options = fast.model_options['transformer_options']
            self.assertEqual(len(options['patches_replace']['dit']), 50)
            self.assertIs(options['wrappers'][indexed_gate.KEY].__self__, fast._h3_indexed_gate)
            with self.assertRaises(Unsupported):
                indexed_gate.patch_model(fast)

    def test_fallback_keeps_executor_and_cleans_context_on_exception(self):
        consumer = indexed_gate.IndexedGatePatch(object())
        def unsupported(_):
            raise Unsupported('unsupported input')
        consumer.preflight = unsupported
        def executor(*args, **kwargs):
            self.assertEqual(consumer.context.get()['fallback'], 'unsupported input')
            return 'original result'
        with self.assertLogs(indexed_gate.LOG, level='WARNING'):
            self.assertEqual(consumer.wrapper(executor), 'original result')
        self.assertIsNone(consumer.context.get())
        self.assertEqual(consumer.last_report['fallback'], 'unsupported input')
        def failed(*args, **kwargs):
            raise OSError('test executor failure')
        with self.assertRaises(OSError):
            consumer.wrapper(failed)
        self.assertIsNone(consumer.context.get())
