"""Offline checks for the frozen task corpus and one-reply scoring contract."""

from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx2
import numpy as np
from trajectory import APIConnectionError

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME))
harness = importlib.import_module("harness")
verify = importlib.import_module("verify")

TASKS = [
    json.loads(line) for line in (RUNTIME / "tasks.jsonl").read_text().splitlines()
]
ZERO_POLICY = "```edits\nb3[:] = 0\n```"


def make_client(answer=ZERO_POLICY, finish_reason="stop", capture_status="completed"):
    client = Mock()
    client.trajectories.create.return_value = SimpleNamespace(tid="test")
    client.trajectories.complete.return_value = SimpleNamespace(status=capture_status)
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason, message=SimpleNamespace(content=answer)
            )
        ]
    )
    return client


class RuntimeTest(unittest.TestCase):
    def test_frozen_roster_prompts_and_source_hashes(self):
        source = json.loads((RUNTIME / "source.json").read_text())
        self.assertEqual(
            source["source_revision"], "4038830175c4201ca10e004ed4441985f2acddf2"
        )
        self.assertEqual(
            source["tasks_sha256"],
            hashlib.sha256((RUNTIME / "tasks.jsonl").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            source["verify_sha256"],
            hashlib.sha256((RUNTIME / "verify.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            Counter(task["split"] for task in TASKS), {"train": 317, "test": 5}
        )
        self.assertEqual(len({task["name"] for task in TASKS}), 322)
        self.assertEqual({task["episodes"] for task in TASKS}, {1000})
        self.assertEqual(
            {task["system_prompt"] for task in TASKS},
            {
                "You directly program neural-network weights. Follow the task constraints exactly."
            },
        )
        self.assertTrue(all(task["prompt"] for task in TASKS))
        self.assertEqual(
            {task["game"] for task in TASKS if task["split"] == "test"},
            {"asterix", "breakout", "freeway", "seaquest", "space_invaders"},
        )
        expected = {
            f"minatar_synth/games/{task['game']}.py"
            for task in TASKS
            if task["source"] == "synth"
        }
        self.assertEqual(set(source["game_sha256"]), expected)
        self.assertEqual(
            {
                str(path.relative_to(RUNTIME))
                for path in (RUNTIME / "minatar_synth/games").glob("*.py")
            },
            expected,
        )
        self.assertNotIn("minatar_synth/games/pong.py", expected)
        for path, digest in source["game_sha256"].items():
            with self.subTest(path=path):
                self.assertEqual(
                    hashlib.sha256((RUNTIME / path).read_bytes()).hexdigest(), digest
                )

    def test_every_packaged_environment_imports_and_steps(self):
        for task in TASKS:
            with self.subTest(game=task["game"]):
                environment = verify.make_environment(task["game"], task["source"])
                environment.seed(1729)
                environment.reset()
                state = verify.observation(
                    task["game"], task["source"], environment.state()
                )
                self.assertEqual(state.shape, (600,))
                self.assertTrue(np.isfinite(state).all())
                self.assertGreater(verify.score_cap(task["game"], task["source"]), 0)
                reward, _ = environment.act(0)
                self.assertTrue(np.isfinite(reward))

    def test_parser_preserves_source_semantics_without_executing_code(self):
        weights, count = verify.apply_edits(
            "```edits\nb3[0] = 999\n```\n```EDITS\nb3[-1] = 2\nb3[-1] *= 3\n```"
        )
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(weights["b3"], [0, 0, 0, 0, 0, 6])
        # Empty edits are an all-zero network in the frozen source parser.
        _, count = verify.apply_edits("```edits\n\n```")
        self.assertEqual(count, 0)
        for answer in (
            "no weights",
            "```edits\nb3[0] = 1e309\n```",
            "```edits\nb3[0] = __import__('os').system('false')\n```",
            "```edits\nb3[0] = 1\nimport os\n```",
        ):
            with (
                self.subTest(answer=answer),
                self.assertRaises((ValueError, SyntaxError)),
            ):
                verify.apply_edits(answer)

    def test_fixed_seed_replay_for_each_family(self):
        weights, _ = verify.apply_edits(ZERO_POLICY)
        # Recorded from the original v4 verifier and the pinned source checkout.
        expected = {"alien": (0.0, 6.0), "balloon": (2.0, 54.0), "breakout": (0.0, 6.0)}
        for game, source in (
            ("alien", "synth"),
            ("balloon", "synth"),
            ("breakout", "original"),
        ):
            with self.subTest(game=game):
                result = verify.evaluate(game, source, weights, episodes=2)
                self.assertEqual(result, expected[game])
                self.assertEqual(
                    result, verify.evaluate(game, source, weights, episodes=2)
                )
                self.assertTrue(np.isfinite(result).all())

    def test_exact_messages_and_clipped_game_reward(self):
        task = TASKS[0]
        for raw, expected in ((-3.0, 0.0), (4.0, 0.4), (13.0, 1.0)):
            with self.subTest(raw=raw):
                client = make_client()
                with (
                    patch.object(
                        harness, "evaluate", return_value=(raw, 10.0)
                    ) as evaluate,
                    patch.object(harness, "score_cap", return_value=10.0),
                ):
                    result = harness.run(client, task)
                self.assertEqual(result["reward"], expected)
                self.assertEqual(
                    evaluate.call_args.args[0:2], (task["game"], task["source"])
                )
                self.assertEqual(evaluate.call_args.args[3], 1000)
                client.chat.completions.create.assert_called_once_with(
                    model="policy",
                    messages=[
                        {"role": "system", "content": task["system_prompt"]},
                        {"role": "user", "content": task["prompt"]},
                    ],
                    max_tokens=32_768,
                    x_trajectory_id="test",
                )
                client.trajectories.log_reward.assert_called_once_with(
                    "test", name="reward_normalized_minatar_score", value=expected
                )
                client.trajectories.complete.assert_called_once_with(
                    "test", termination_reason="ENV_DONE"
                )

    def test_invalid_and_truncated_answers_preserve_grading_behavior(self):
        for answer, reason, expected in (
            ("no weights", "stop", 0.0),
            ("no weights", "length", 0.0),
            (ZERO_POLICY, "length", 0.4),
            (ZERO_POLICY, "max_tokens", 0.4),
        ):
            with self.subTest(answer=answer, reason=reason):
                client = make_client(answer, reason)
                with (
                    patch.object(harness, "evaluate", return_value=(4.0, 10.0)),
                    patch.object(harness, "score_cap", return_value=10.0),
                ):
                    result = harness.run(client, TASKS[0])
                self.assertEqual(result["reward"], expected)
                client.trajectories.complete.assert_called_once_with(
                    "test",
                    termination_reason="ENV_DONE" if reason == "stop" else "TRUNCATION",
                )

    def test_api_and_simulator_failures_are_errors_without_reward(self):
        client = make_client()
        client.chat.completions.create.side_effect = APIConnectionError(
            httpx2.Request("POST", "https://example.invalid")
        )
        with self.assertRaises(APIConnectionError):
            harness.run(client, TASKS[0])
        client.trajectories.log_reward.assert_not_called()
        client.trajectories.complete.assert_called_once_with(
            "test", termination_reason="ERROR"
        )
        client = make_client()
        with patch.object(
            harness, "evaluate", side_effect=ValueError("broken simulator")
        ):
            with self.assertRaisesRegex(ValueError, "broken simulator"):
                harness.run(client, TASKS[0])
        client.trajectories.log_reward.assert_not_called()
        client.trajectories.complete.assert_called_once_with(
            "test", termination_reason="ERROR"
        )

    def test_finalizing_is_accepted_without_rewriting_the_ending(self):
        client = make_client("no weights", capture_status="finalizing")
        result = harness.run(client, TASKS[0])
        self.assertEqual(result["capture_status"], "finalizing")
        client.trajectories.complete.assert_called_once_with(
            "test", termination_reason="ENV_DONE"
        )
        client.trajectories.retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
