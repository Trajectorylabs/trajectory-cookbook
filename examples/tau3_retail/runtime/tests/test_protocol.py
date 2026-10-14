"""Offline checks of the pinned split and source digest."""

import os
from pathlib import Path

import pytest
from tau3_retail.protocol import TASK_IDS, TEST_IDS, TRAIN_IDS, verify_domain_data


def test_official_split_is_disjoint_and_complete():
    assert (len(TRAIN_IDS), len(TEST_IDS)) == (74, 40)
    assert not set(TRAIN_IDS) & set(TEST_IDS)
    assert TASK_IDS == tuple(str(index) for index in range(114))


@pytest.mark.skipif(
    "TAU2_DATA_DIR" not in os.environ,
    reason="Set TAU2_DATA_DIR to a tau2-bench checkout's data directory",
)
def test_pinned_retail_data_matches_digest_and_split():
    verify_domain_data(Path(os.environ["TAU2_DATA_DIR"]) / "tau2" / "domains" / "retail")
