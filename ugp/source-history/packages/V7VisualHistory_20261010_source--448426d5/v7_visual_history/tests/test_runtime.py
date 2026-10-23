import importlib
import tempfile
import unittest
from pathlib import Path

TEMP_ROOT = Path(__file__).resolve().parents[2]/'_visual_history_test_tmp'


def test_temp_directory():
    TEMP_ROOT.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEMP_ROOT)

def runtime_module(test):
    try:
        return importlib.import_module('v7_visual_history.runtime')
    except ModuleNotFoundError:
        test.fail('Separate visual-history runtime/checkpoint namespace is not implemented')


class RuntimeTests(unittest.TestCase):
    def test_resume_moves_rng_payloads_to_cpu_without_mutating_checkpoint(self):
        m = runtime_module(self)
        self.assertTrue(hasattr(m, 'normalize_rng_state'), 'RNG resume normalization is missing')

        class DeviceMappedState:
            def __init__(self, cpu_value):
                self.cpu_value = cpu_value

            def cpu(self):
                return self.cpu_value

        cpu_state, cuda_state = object(), object()
        saved = dict(python='python-state', numpy='numpy-state',
                     torch_cpu=DeviceMappedState(cpu_state),
                     torch_cuda=[DeviceMappedState(cuda_state)])
        normalized = m.normalize_rng_state(saved)
        self.assertIs(normalized['torch_cpu'], cpu_state)
        self.assertIs(normalized['torch_cuda'][0], cuda_state)
        self.assertEqual(normalized['python'], saved['python'])
        self.assertEqual(normalized['numpy'], saved['numpy'])
        self.assertIsInstance(saved['torch_cpu'], DeviceMappedState)
        self.assertIsInstance(saved['torch_cuda'][0], DeviceMappedState)
        self.assertIsNone(m.normalize_rng_state(None))

    def test_rejects_writes_into_original_foundation(self):
        m = runtime_module(self)
        with test_temp_directory() as temp:
            root = Path(temp)/'foundation'; root.mkdir()
            for output in [root, root/'storage/results', root.parent]:
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    m.check_output_directory(root, output)
            m.check_output_directory(root, Path(temp)/'separate-run')

    def test_manifest_must_match_before_reusing_run(self):
        m = runtime_module(self)
        with test_temp_directory() as temp:
            m.record_manifest(Path(temp), {'schema': 'test', 'frames': 6})
            m.record_manifest(Path(temp), {'schema': 'test', 'frames': 6})
            with self.assertRaisesRegex(ValueError, 'manifest'):
                m.record_manifest(Path(temp), {'schema': 'test', 'frames': 5})


if __name__ == '__main__':
    unittest.main()
