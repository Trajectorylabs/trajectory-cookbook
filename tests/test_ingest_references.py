import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest
from trajectory import Client


@pytest.fixture(params=["gsm8k", "t_factory"])
def uploader(request: pytest.FixtureRequest) -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / request.param / "ingest.py"
    spec = importlib.util.spec_from_file_location(f"ingest_{request.param}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "agent",
    [
        {"agent_id": "agt_fixture"},
        {"agent_name": "Fixture Agent"},
        {"agent_id": "agt_fixture", "agent_name": "Fixture Agent"},
    ],
)
def test_cli_sends_references_through_sdk(
    uploader: ModuleType,
    agent: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            assert request.url.path.endswith("/sessions")
            body = json.loads(request.content)
            assert body["bench_name"] == "Fixture v1.2_test"
            assert "name" not in body["metadata"]
            assert "tasks" not in body["metadata"]
            assert {key: body[key] for key in agent} == agent
            assert not {"agent", "benchmark", "bench_id", "append"} & body.keys()
            return httpx.Response(
                201, json={"session_id": "op_fixture", "accepted": True}
            )
        assert request.url.path.endswith("/operations/op_fixture")
        return httpx.Response(
            200,
            json={
                "operation_id": "op_fixture",
                "bench_id": "bm_fixture",
                "status": "succeeded",
                "stage": "registering",
                "attempt": 1,
                "max_attempts": 3,
                "build_images": False,
                "registered_tasks": 1,
            },
        )

    with Client(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handle))
    ) as client:
        monkeypatch.setattr(uploader, "Client", lambda: client)
        if uploader.__name__ == "ingest_gsm8k":
            monkeypatch.setattr(
                uploader,
                "_load_rows",
                lambda split, count: [{"question": "6 * 7?", "answer": "42"}],
            )
        args = ["ingest.py", "--name", "Fixture v1.2_test", "--skip-build"]
        for key, value in agent.items():
            args.extend([f"--{key.replace('_', '-')}", value])
        monkeypatch.setattr(sys, "argv", args)
        assert uploader.main() == 0
    assert len(requests) == 2


def test_cli_requires_an_agent(
    uploader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["ingest.py"])
    with pytest.raises(SystemExit) as error:
        uploader.main()
    assert error.value.code == 2


def test_cli_rejects_benchmark_id_before_ingestion(
    uploader: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest.py", "--agent-name", "Fixture Agent", "--bench-id", "bm_fixture"],
    )
    with pytest.raises(SystemExit) as error:
        uploader.main()
    assert error.value.code == 2
