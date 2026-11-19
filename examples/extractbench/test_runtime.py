"""Offline runtime tests: no model calls or Docker daemon required."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as N
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

RUNTIME = Path(__file__).parent / 'runtime/run_task.py'
spec = importlib.util.spec_from_file_location('extractbench_runtime', RUNTIME)
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class FakeClient:
    def __init__(self, reply=None):
        self.reply = reply
        self.calls, self.events, self.rewards = [], [], []
        self.chat = N(completions=N(create=self.create))
        self.trajectories = N(log_event=lambda *a, **k: self.events.append(k),
                             log_reward=lambda *a, **k: self.rewards.append(k))
    def __enter__(self): return self
    def __exit__(self, *a): pass
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception): raise self.reply
        return self.reply


class RuntimeTests(unittest.TestCase):
    def fixture(self, root, value=.75):
        paths = N(verifier_dir=root / 'verifier', artifacts_dir=root / 'artifacts')
        paths.verifier_dir.mkdir()
        paths.artifacts_dir.mkdir()
        (paths.verifier_dir / 'metrics.json').write_text(json.dumps({
            'status': 'scored', 'value_f1': value, 'perfect': int(value == 1),
            'metrics': {'extract_unified_value_f1': value}}))
        result = N(exception_info=None, verifier_result=N(rewards={'reward': value}),
                   verifier_environment_mode='separate',
                   agent_result=N(metadata={'termination_reason': 'ENV_DONE'}))
        return result, paths

    def test_native_grades_include_zero_partial_and_correct(self):
        for value in (0, .75, 1):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                result, paths = self.fixture(Path(directory), value)
                self.assertEqual(harness.native_grade(result, paths)[0], value)

    def test_missing_invalid_and_nonfinite_rewards_fail(self):
        for value in ({}, {'reward': 'bad'}, {'reward': float('nan')}, {'reward': 1.01}):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                result, paths = self.fixture(Path(directory))
                result.verifier_result.rewards = value
                with self.assertRaises((KeyError, TypeError, ValueError)):
                    harness.native_grade(result, paths)

    def test_metrics_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory))
            result.verifier_result.rewards = {'reward': .5}
            with self.assertRaisesRegex(ValueError, 'disagree'):
                harness.native_grade(result, paths)

    def test_error_and_shared_verifier_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory))
            result.exception_info = N(exception_type='VerifierError')
            with self.assertRaises(RuntimeError): harness.native_grade(result, paths)
            result.exception_info = None
            result.verifier_environment_mode = 'shared'
            with self.assertRaises(RuntimeError): harness.native_grade(result, paths)

    def test_invalid_submission_zero_is_a_native_grade(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory), 0)
            (paths.verifier_dir / 'metrics.json').write_text(json.dumps({
                'status': 'invalid_submission', 'value_f1': 0, 'perfect': 0}))
            self.assertEqual(harness.native_grade(result, paths)[0], 0)

    def test_native_reward_component_and_ending_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory))
            result.agent_result.metadata['termination_reason'] = 'TRUNCATION'
            client = FakeClient()
            with patch.object(harness, 'save_evidence'):
                self.assertEqual(harness.publish_result(client, 'fixture', result, paths), ('TRUNCATION', .75))
            self.assertEqual(client.rewards, [{'name': 'extract_unified_value_f1', 'value': .75}])
            self.assertEqual(client.events[0]['payload']['value_f1'], .75)

    def test_no_reward_published_for_verifier_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory))
            result.exception_info = N(exception_type='VerifierError')
            client = FakeClient()
            with self.assertRaises(RuntimeError):
                harness.publish_result(client, 'fixture', result, paths)
            self.assertFalse(client.rewards)

    def test_evidence_failure_does_not_change_native_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory))
            client = FakeClient()
            with patch.object(harness, 'save_evidence', side_effect=OSError('fixture')):
                harness.publish_result(client, 'fixture', result, paths)
            self.assertEqual(client.rewards[0]['value'], .75)
            self.assertEqual(client.events[-1]['name'], 'evidence_error')

    def run_actor(self, finish='stop', metadata=None, reply_error=None):
        reply = N(choices=[N(finish_reason=finish, message=N(content='done', tool_calls=[]))],
                  model_extra={'metadata': metadata} if metadata else {})
        client = FakeClient(reply_error or reply)
        calls = []
        async def execute(*args, **kwargs):
            calls.append((args, kwargs))
            return N(return_code=0, stdout='', stderr='')
        environment = N(exec=execute)
        actor = object.__new__(harness.TrajectoryAgent)
        actor.timeout = 30
        context = N(metadata={})
        with patch.object(harness, 'Client', return_value=client), patch.dict(os.environ, {'TRAJECTORY_TID':'fixture'}):
            asyncio.run(actor.run('OFFICIAL INSTRUCTION', environment, context))
        return client, calls, context

    def test_stop_and_truncation_endings(self):
        for finish, reason in [('stop', 'ENV_DONE'), ('length', 'TRUNCATION'), ('max_tokens', 'TRUNCATION')]:
            client, calls, context = self.run_actor(finish)
            self.assertEqual(context.metadata['termination_reason'], reason)
            self.assertFalse(calls)
            self.assertEqual(client.calls[0]['messages'][1]['content'], 'OFFICIAL INSTRUCTION')

    def test_cancelled_metadata_stops_before_tools(self):
        _, calls, context = self.run_actor(metadata={'trajectory_termination_reason':'trajectory_cancelled'})
        self.assertFalse(calls)
        self.assertEqual(context.metadata['termination_reason'], 'TRUNCATION')

    def test_cutoff_tool_call_is_never_dispatched(self):
        reply = N(choices=[N(finish_reason='length', message=N(content='', tool_calls=[N(
            id='tool1', function=N(name='bash', arguments='{"command":"cat /tests/reference.test.json"}')
        )]))], model_extra={})
        client = FakeClient(reply)
        async def denied(*args, **kwargs):
            self.fail('cutoff tool call was dispatched')
        actor = object.__new__(harness.TrajectoryAgent)
        actor.timeout = 30
        context = N(metadata={})
        with patch.object(harness, 'Client', return_value=client), patch.dict(os.environ, {'TRAJECTORY_TID':'fixture'}):
            asyncio.run(actor.run('OFFICIAL INSTRUCTION', N(exec=denied), context))
        self.assertEqual(context.metadata['termination_reason'], 'TRUNCATION')

    def test_setup_private_boundary_failure_stops_actor(self):
        async def failed(*args, **kwargs):
            return N(return_code=1, stdout='', stderr='fixture boundary error')
        actor = object.__new__(harness.TrajectoryAgent)
        with self.assertRaisesRegex(RuntimeError, 'private paths'):
            asyncio.run(actor.setup(N(exec=failed)))

    def test_transport_failure_is_not_a_normal_ending(self):
        with self.assertRaisesRegex(RuntimeError, 'transport'):
            self.run_actor(reply_error=RuntimeError('transport'))

    def test_policy_requires_separate_verifier_and_no_actor_egress(self):
        task = N(config=N(agent=N(network_mode='allowlist', allowed_hosts=[]),
                          verifier=N(environment_mode='separate', environment=N(network_mode='no-network'))))
        harness.validate_task_policy(task)
        task.config.agent.allowed_hosts = ['huggingface.co']
        with self.assertRaises(ValueError): harness.validate_task_policy(task)

    def trial_task(self):
        return N(config=N(agent=N(network_mode='allowlist', allowed_hosts=[], timeout_sec=1800),
                          verifier=N(environment_mode='separate', environment=N(network_mode='no-network'))))

    def test_verifier_smoke_selects_builtin_nop_without_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory), 0)
            result.agent_result = None
            trial = N(paths=paths, run=AsyncMock(return_value=result))
            create = AsyncMock(return_value=trial)
            with patch.object(harness, 'Task', return_value=self.trial_task()), \
                 patch.object(harness.Trial, 'create', create), \
                 patch.object(harness, 'Client', side_effect=AssertionError('unexpected SDK/model call')):
                got, _ = asyncio.run(harness.run_trial(Path(directory), None, verifier_smoke=True))
            config = create.call_args.args[0]
            self.assertEqual(config.agent.name, 'nop')
            self.assertIsNone(config.agent.import_path)
            self.assertEqual(got.agent_result.metadata['termination_reason'], 'ENV_DONE')
            self.assertTrue(got.agent_result.metadata['verifier_smoke'])

    def test_default_trial_keeps_managed_actor(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory))
            trial = N(paths=paths, run=AsyncMock(return_value=result))
            create = AsyncMock(return_value=trial)
            with patch.object(harness, 'Task', return_value=self.trial_task()), \
                 patch.object(harness.Trial, 'create', create):
                asyncio.run(harness.run_trial(Path(directory), None))
            config = create.call_args.args[0]
            self.assertEqual(config.agent.import_path, harness.TrajectoryAgent.import_path())
            self.assertEqual(config.agent.override_timeout_sec, 1800)
            self.assertEqual(config.agent.kwargs, {'timeout': 1800})

    def test_nop_invalid_submission_preserves_native_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory), 0)
            result.agent_result = N(metadata={'termination_reason':'ENV_DONE', 'verifier_smoke': True})
            (paths.verifier_dir / 'metrics.json').write_text(json.dumps({
                'status': 'invalid_submission', 'value_f1': 0, 'perfect': 0}))
            client = FakeClient()
            with patch.object(harness, 'save_evidence'):
                reason, reward = harness.publish_result(client, 'nop-fixture', result, paths)
            self.assertEqual((reason, reward), ('ENV_DONE', 0))
            self.assertEqual(client.rewards, [{'name': 'extract_unified_value_f1', 'value': 0}])

    def test_nop_exception_cannot_be_normalized_to_success(self):
        with tempfile.TemporaryDirectory() as directory:
            result, paths = self.fixture(Path(directory), 0)
            result.agent_result = None
            result.exception_info = N(exception_type='AgentSetupError')
            trial = N(paths=paths, run=AsyncMock(return_value=result))
            with patch.object(harness, 'Task', return_value=self.trial_task()), \
                 patch.object(harness.Trial, 'create', AsyncMock(return_value=trial)):
                got, _ = asyncio.run(harness.run_trial(Path(directory), None, verifier_smoke=True))
            self.assertIsNone(got.agent_result)
            with self.assertRaises(RuntimeError): harness.native_grade(got, paths)

    def test_runtime_image_keeps_tests_out_of_actor_build_context(self):
        docker = (RUNTIME.parent / 'Dockerfile').read_text()
        self.assertIn('harbor==0.25.0', docker)
        self.assertIn('COPY tasks.tar.gz /opt/extractbench/tasks.tar.gz', docker)
        # Inner actor receives environment/ only through Harbor, not outer COPY tasks.
        self.assertIn('environment=EnvironmentConfig(type="docker")', RUNTIME.read_text())
        self.assertNotIn('override_network_mode', RUNTIME.read_text())


if __name__ == '__main__': unittest.main()
