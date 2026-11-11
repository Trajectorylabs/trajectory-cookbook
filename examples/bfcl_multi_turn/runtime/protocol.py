"""Pinned BFCL release, categories, and the scenario-level split."""

import random

BFCL_PACKAGE = "bfcl-eval==2026.3.23"
BFCL_REPOSITORY = "https://github.com/ShishirPatil/gorilla"
BFCL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"

# Each category has 200 entries; entry i of every category is a variant of the same scenario.
CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
)
SCENARIOS = 200
TEST_SCENARIOS = 40
SPLIT_SEED = 20251110

# The policy's name inside BFCL; it also namespaces BFCL's per-entry API instances.
POLICY_NAME = "trajectory-policy"


def test_scenarios() -> frozenset[int]:
    """Hold out whole scenarios so no variant of a test scenario is trained on."""
    return frozenset(random.Random(SPLIT_SEED).sample(range(SCENARIOS), TEST_SCENARIOS))


def split_of(scenario: int) -> str:
    return "test" if scenario in test_scenarios() else "train"
