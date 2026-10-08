"""Check reward-reporting examples against the installed public SDK."""

import ast
import inspect
import re
from pathlib import Path

import pytest
from trajectory.resources.trajectories.trajectories import Trajectories

ROOT = Path(__file__).resolve().parents[1]


def reward_calls():
    for path in [ROOT / "README.md", *sorted((ROOT / "examples").rglob("*"))]:
        if path.suffix not in {".md", ".py"}:
            continue
        text = path.read_text()
        blocks = (
            re.findall(r"```python[^\n]*\n(.*?)```", text, re.S)
            if path.suffix == ".md"
            else [text]
        )
        for block in blocks:
            if ".trajectories.log_reward(" not in block:
                continue
            for node in ast.walk(ast.parse(block)):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "log_reward"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "trajectories"
                ):
                    yield pytest.param(node, id=f"{path.relative_to(ROOT)}:{node.lineno}")


@pytest.mark.parametrize("call", list(reward_calls()))
def test_log_reward_arguments_match_sdk(call):
    inspect.signature(Trajectories.log_reward).bind(
        None, *[None for _ in call.args], **{kw.arg: None for kw in call.keywords},
    )
