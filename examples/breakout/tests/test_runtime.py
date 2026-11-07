"""Offline checks for parsing, scoring, and capture semantics."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
sys.path.insert(0, str(RUNTIME))
import breakout_harness as harness
import scorer


def make_client(content="```edits\nb3[:] = 0\n```", finish_reason="stop"):
    client = Mock()
    client.trajectories.create.return_value = SimpleNamespace(tid="test")
    client.trajectories.complete.return_value = SimpleNamespace(status="completed")
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=finish_reason,
            message=SimpleNamespace(content=content),
        )]
    )
    return client


class RuntimeTest(unittest.TestCase):
    def test_numeric_parser_skips_bad_statements_and_uses_last_block(self):
        weights, audit = scorer.parse_weights(
            "```edits\nb1[0] = 999\n```\n"
            "```edits\nb1[0] = 2\nb1[0] *= 3\n"
            "b1[-1] = 5\nb1[1] = __import__('os').system('false')\n"
            "b1[1] = 1e309\nb1[1] = True\n"
            "W2[0:2, 0:2] = [[1,2],\n[3,4]]\n```"
        )
        self.assertEqual(audit["applied"], 3)
        self.assertEqual(len(audit["skipped"]), 4)
        self.assertEqual(weights["b1"][0], 6)
        self.assertEqual(weights["b1"][1], 0)
        np.testing.assert_array_equal(weights["W2"][:2, :2], [[1, 2], [3, 4]])
        for text in ("no weights", "```edits\nimport os\n```", "```edits\nb1[0] = 2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                scorer.parse_weights(text)

    def test_observation_mapping_and_deterministic_episode_sampling(self):
        state = np.zeros((10, 10, 4), dtype=bool)
        for channel in range(4):
            state[channel, 2, channel] = True
        observation = scorer.observation(state).reshape(10, 10, 6)
        for source, target in ((0, 0), (1, 3), (2, 4), (3, 2)):
            self.assertTrue(observation[source, 2, target])
        self.assertEqual(observation.sum(), 4)

        weights, _ = scorer.parse_weights("```edits\nb3[:] = 0\n```")
        first = scorer.evaluate(weights, [41, 42], max_steps=100)
        self.assertEqual(first, scorer.evaluate(weights, [41, 42], max_steps=100))
        single = scorer.evaluate(weights, [42], max_steps=100)
        self.assertEqual(first["returns"][1], single["returns"][0])
        self.assertEqual(first["lengths"][1], single["lengths"][0])

    def test_packaged_split_and_prompt_provenance(self):
        rows = [json.loads(line) for line in (RUNTIME / "tasks.jsonl").read_text().splitlines()]
        train = [row for row in rows if row["split"] == "train"]
        test = [row for row in rows if row["split"] == "test"]
        self.assertEqual((len(train), len(test)), (32, 1))
        self.assertTrue(all(len(row["seeds"]) == 64 for row in train))
        train_seeds = [seed for row in train for seed in row["seeds"]]
        self.assertEqual(train_seeds, list(range(1000, 3048)))
        self.assertEqual(test[0]["seeds"], list(range(100000, 100256)))
        self.assertTrue(set(train_seeds).isdisjoint(test[0]["seeds"]))
        source = json.loads((RUNTIME / "source.json").read_text())
        self.assertEqual(source["prompt_sha256"], hashlib.sha256((RUNTIME / "prompt.txt").read_bytes()).hexdigest())

    def test_one_reply_hides_seeds_and_records_raw_reward(self):
        client = make_client()
        with patch.object(harness, "evaluate", return_value={"mean_return": 10.0}):
            result = harness.run(client, "fixed prompt", [987654])
        self.assertTrue(result["valid"])
        self.assertEqual(result["reward"], 10.0)
        client.chat.completions.create.assert_called_once_with(
            model="policy",
            messages=[{"role": "user", "content": "fixed prompt"}],
            max_tokens=32_768,
            x_trajectory_id="test",
        )
        client.trajectories.log_reward.assert_called_once_with(
            "test", name="breakout_score", value=10.0, min_value=0.0, max_value=10_000.0
        )
        client.trajectories.complete.assert_called_once_with("test", termination_reason="ENV_DONE")

    def test_invalid_and_capped_replies_have_zero_reward_and_distinct_endings(self):
        cases = (
            ("no weights", "stop", "ENV_DONE"),
            ("```edits\nb3[:] = 0\n```", "length", "TRUNCATION"),
            ("```edits\nb3[:] = 0\n```", "max_tokens", "TRUNCATION"),
        )
        for content, finish_reason, termination in cases:
            with self.subTest(finish_reason=finish_reason):
                client = make_client(content, finish_reason)
                with patch.object(harness, "evaluate") as evaluate:
                    result = harness.run(client, "prompt", [41])
                self.assertFalse(result["valid"])
                self.assertEqual(result["reward"], 0.0)
                self.assertEqual(result["termination_reason"], termination)
                evaluate.assert_not_called()
                client.trajectories.complete.assert_called_once_with("test", termination_reason=termination)

    def test_simulator_failure_is_error_without_a_fabricated_reward(self):
        client = make_client()
        with patch.object(harness, "evaluate", side_effect=RuntimeError("simulator unavailable")):
            with self.assertRaisesRegex(RuntimeError, "simulator unavailable"):
                harness.run(client, "prompt", [41])
        client.trajectories.log_reward.assert_not_called()
        client.trajectories.complete.assert_called_once_with("test", termination_reason="ERROR")

    def test_waits_for_capture_finalization(self):
        client = make_client("no weights")
        client.trajectories.complete.return_value = SimpleNamespace(status="finalizing")
        client.trajectories.retrieve.return_value = SimpleNamespace(status="completed")
        with patch.object(harness.time, "sleep"):
            result = harness.run(client, "prompt", [41])
        self.assertEqual(result["termination_reason"], "ENV_DONE")
        client.trajectories.retrieve.assert_called_once_with("test", include_steps=False)


if __name__ == "__main__":
    unittest.main()
