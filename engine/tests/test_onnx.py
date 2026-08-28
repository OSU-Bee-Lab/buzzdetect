"""src/inference/onnx.py: which execution provider and which graph get used.

Provider selection is the part of the engine that is quiet when it goes wrong
-- a run that fell back to the CPU looks exactly like a run that meant to --
so the fallback rules are pinned down here without needing the hardware.
"""

import os
import unittest
from unittest import mock

import tests._context as ctx  # noqa: F401

import src.inference.onnx as onnx

CPU = 'CPUExecutionProvider'
CUDA = 'CUDAExecutionProvider'
ROCM = 'ROCMExecutionProvider'
COREML = 'CoreMLExecutionProvider'


def available(*names):
    return mock.patch.object(onnx.ort, 'get_available_providers', lambda: list(names))


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        # _warn_once keeps a module-level set so a long run doesn't repeat
        # itself; each test wants a clean slate.
        onnx._warned.clear()
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop(onnx.ENV_ALLOW_FP16, None)
        self.stderr = mock.patch('sys.stderr')
        self.stderr.start()

    def tearDown(self):
        self.stderr.stop()
        self.env.stop()
        onnx._warned.clear()


class TestProvidersFor(ProviderTestCase):
    def test_a_cpu_worker_never_asks_for_a_gpu(self):
        with available(CUDA, CPU):
            self.assertEqual(onnx.providers_for('CPU'), ([CPU], [{}]))

    def test_cpu_is_always_appended_as_the_backstop(self):
        with available(CUDA, CPU):
            providers, _ = onnx.providers_for('GPU')
            self.assertEqual(providers, [CUDA, CPU])

    def test_the_preference_order_is_the_declared_one(self):
        with available(ROCM, CUDA, CPU):
            providers, _ = onnx.providers_for('GPU')
            self.assertEqual(providers[0], CUDA)

    def test_coreml_is_pinned_to_mlprogram(self):
        with available(COREML, CPU):
            _, options = onnx.providers_for('GPU')
            self.assertEqual(options[0]['ModelFormat'], 'MLProgram')

    def test_a_gpu_request_with_no_gpu_provider_falls_back_and_says_so(self):
        with available(CPU):
            providers, options = onnx.providers_for('GPU')
        self.assertEqual(providers, [CPU])
        self.assertEqual(options, [{}])
        self.assertIn('no-gpu-provider', onnx._warned)

    def test_the_fallback_warning_is_only_printed_once(self):
        with available(CPU), mock.patch('builtins.print') as printed:
            onnx.providers_for('GPU')
            onnx.providers_for('GPU')
        self.assertEqual(printed.call_count, 1)

    def test_provider_options_are_not_shared_between_calls(self):
        # Each call gets its own dict; mutating one must not edit the module's
        # GPU_PROVIDERS table.
        with available(COREML, CPU):
            _, first = onnx.providers_for('GPU')
            first[0]['ModelFormat'] = 'mutated'
            _, second = onnx.providers_for('GPU')
        self.assertEqual(second[0]['ModelFormat'], 'MLProgram')


class TestFp16(ProviderTestCase):
    def test_off_unless_the_environment_asks(self):
        self.assertFalse(onnx.allow_fp16())
        os.environ[onnx.ENV_ALLOW_FP16] = '1'
        self.assertTrue(onnx.allow_fp16())
        os.environ[onnx.ENV_ALLOW_FP16] = '0'
        self.assertFalse(onnx.allow_fp16())

    def test_only_coreml_acts_on_it(self):
        self.assertTrue(onnx.fp16_supported(COREML))
        self.assertFalse(onnx.fp16_supported(CUDA))
        self.assertFalse(onnx.fp16_supported(CPU))

    def test_coreml_reaches_the_neural_engine_when_asked(self):
        os.environ[onnx.ENV_ALLOW_FP16] = '1'
        with available(COREML, CPU):
            _, options = onnx.providers_for('GPU')
        self.assertEqual(options[0], onnx.COREML_FP16)

    def test_cuda_ignores_it(self):
        os.environ[onnx.ENV_ALLOW_FP16] = '1'
        with available(CUDA, CPU):
            _, options = onnx.providers_for('GPU')
        self.assertEqual(options[0], {})


class TestPathFor(ProviderTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = self.enterContext(__import__('tempfile').TemporaryDirectory())
        self.fp32 = os.path.join(self.tmp, 'model.onnx')
        open(self.fp32, 'w').close()

    def write_fp16(self):
        path = os.path.join(self.tmp, onnx.FNAME_FP16)
        open(path, 'w').close()
        return path

    def test_a_cpu_worker_always_gets_the_full_precision_graph(self):
        os.environ[onnx.ENV_ALLOW_FP16] = '1'
        self.write_fp16()
        with available(COREML, CPU):
            self.assertEqual(onnx.path_for(self.fp32, 'CPU'), self.fp32)

    def test_unasked_for_means_full_precision(self):
        self.write_fp16()
        with available(COREML, CPU):
            self.assertEqual(onnx.path_for(self.fp32, 'GPU'), self.fp32)

    def test_asked_for_on_coreml_with_a_sibling_graph(self):
        os.environ[onnx.ENV_ALLOW_FP16] = '1'
        expected = self.write_fp16()
        with available(COREML, CPU):
            self.assertEqual(onnx.path_for(self.fp32, 'GPU'), expected)
        self.assertIn('fp16', onnx._warned)

    def test_asked_for_but_the_model_has_no_fp16_graph(self):
        os.environ[onnx.ENV_ALLOW_FP16] = '1'
        with available(COREML, CPU):
            self.assertEqual(onnx.path_for(self.fp32, 'GPU'), self.fp32)
        self.assertIn('no-fp16-graph', onnx._warned)

    def test_asked_for_on_a_provider_that_cannot_use_it(self):
        os.environ[onnx.ENV_ALLOW_FP16] = '1'
        self.write_fp16()
        with available(CUDA, CPU):
            self.assertEqual(onnx.path_for(self.fp32, 'GPU'), self.fp32)


class TestGpuProvidersAvailable(ProviderTestCase):
    def test_reports_only_gpu_providers(self):
        with available(CUDA, CPU):
            self.assertEqual(onnx.gpu_providers_available(), [CUDA])

    def test_empty_on_a_cpu_only_build(self):
        with available(CPU):
            self.assertEqual(onnx.gpu_providers_available(), [])


class TestProbeGpu(ProviderTestCase):
    """The probe has to build a session AND run it; either failing drops the
    provider. See the cuDNN mismatch this was written for."""

    def fake_session(self, providers, run=None):
        session = mock.Mock()
        session.get_providers.return_value = providers
        session.get_inputs.return_value = [mock.Mock(name='in', shape=[onnx.PROBE_SAMPLES])]
        session.get_inputs.return_value[0].name = 'waveform'
        session.run.side_effect = run
        return session

    def test_a_build_only_installation_reports_nothing_usable(self):
        with available(CUDA, CPU), \
                mock.patch.object(onnx.ort, 'InferenceSession', side_effect=RuntimeError('no driver')):
            self.assertEqual(onnx.probe_gpu(), [])

    def test_a_provider_that_builds_and_runs_is_offered(self):
        session = self.fake_session([CUDA, CPU])
        with available(CUDA, CPU), mock.patch.object(onnx.ort, 'InferenceSession', return_value=session):
            self.assertEqual(onnx.probe_gpu(), [CUDA])
        self.assertTrue(session.run.called)

    def test_a_provider_that_builds_but_cannot_run_is_not_offered(self):
        session = self.fake_session([CUDA, CPU], run=RuntimeError('CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH'))
        with available(CUDA, CPU), mock.patch.object(onnx.ort, 'InferenceSession', return_value=session):
            self.assertEqual(onnx.probe_gpu(), [])

    def test_a_provider_that_silently_dropped_to_cpu_is_not_offered(self):
        session = self.fake_session([CPU])
        with available(CUDA, CPU), mock.patch.object(onnx.ort, 'InferenceSession', return_value=session):
            self.assertEqual(onnx.probe_gpu(), [])

    def test_a_cpu_only_build_never_builds_a_session(self):
        with available(CPU), mock.patch.object(onnx.ort, 'InferenceSession') as built:
            self.assertEqual(onnx.probe_gpu(), [])
        built.assert_not_called()

    def test_the_probe_uses_a_real_model_from_the_model_directory(self):
        path = onnx._probe_model()
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))


class TestMakeSession(ProviderTestCase):
    def test_the_free_dimension_is_pinned_before_the_session_is_built(self):
        # onnxruntime's own SessionOptions won't say what it was told, so stand
        # in for it. The name has to be the one the export gives the input, or
        # the override silently does nothing and CoreML refuses the graph.
        class RecordingOptions:
            def __init__(self):
                self.pinned = {}

            def add_free_dimension_override_by_name(self, name, value):
                self.pinned[name] = value

        session = mock.Mock()
        session.get_providers.return_value = [CUDA, CPU]
        with available(CUDA, CPU), \
                mock.patch.object(onnx.ort, 'SessionOptions', RecordingOptions), \
                mock.patch.object(onnx.ort, 'InferenceSession', return_value=session) as built:
            onnx.make_session('/models/m/model.onnx', 'GPU', 12345)
        options = built.call_args[0][1]
        self.assertEqual(options.pinned, {'samples': 12345})

    def test_warns_when_the_requested_provider_did_not_load(self):
        session = mock.Mock()
        session.get_providers.return_value = [CPU]
        with available(CUDA, CPU), mock.patch.object(onnx.ort, 'InferenceSession', return_value=session):
            onnx.make_session('/models/m/model.onnx', 'GPU', 16000)
        self.assertIn('gpu-not-loaded', onnx._warned)

    def test_no_warning_when_it_did(self):
        session = mock.Mock()
        session.get_providers.return_value = [CUDA, CPU]
        with available(CUDA, CPU), mock.patch.object(onnx.ort, 'InferenceSession', return_value=session):
            onnx.make_session('/models/m/model.onnx', 'GPU', 16000)
        self.assertEqual(onnx._warned, set())


if __name__ == '__main__':
    unittest.main()
