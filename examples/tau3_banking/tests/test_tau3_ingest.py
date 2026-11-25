import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]


@pytest.fixture
def ingest(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location("tau3_ingest", EXAMPLE / "ingest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("selector", "project_id", "project_name"),
    [
        ("--project-id", "agt_test", None),
        ("--project-name", None, "test-project"),
    ],
)
@pytest.mark.parametrize("upload_only", [False, True])
def test_cli_uploads_and_optionally_builds(
    ingest, monkeypatch, capsys, selector, project_id, project_name, upload_only
):
    client_factory = create_autospec(ingest.Client)
    images = SimpleNamespace(
        build=Mock(), list=Mock(return_value=[SimpleNamespace(build_status="ready")])
    )
    client = SimpleNamespace(specs=SimpleNamespace(images=images))
    client_factory.return_value.__enter__.return_value = client
    push = create_autospec(
        ingest.push, return_value=SimpleNamespace(bench_id="bm_test")
    )
    monkeypatch.setattr(ingest, "Client", client_factory)
    monkeypatch.setattr(ingest, "push", push)
    args = ["ingest.py", selector, project_id or project_name]
    if upload_only:
        args.append("--upload-only")
    monkeypatch.setattr(sys, "argv", args)

    ingest.main()

    push.assert_called_once()
    assert push.call_args.args[0] is client
    assert push.call_args.kwargs["agent_id"] == project_id
    assert push.call_args.kwargs["agent_name"] == project_name
    assert push.call_args.kwargs["build_images"] is False
    assert capsys.readouterr().out == "bench_id=bm_test\n"
    if upload_only:
        images.build.assert_not_called()
        images.list.assert_not_called()
    else:
        images.build.assert_called_once_with("bm_test")
        images.list.assert_called_once_with("bm_test")


@pytest.mark.parametrize(
    "flags", [[], ["--project-id", "agt_test", "--project-name", "test-project"]]
)
def test_cli_requires_one_project_selector(ingest, monkeypatch, flags):
    client_factory = create_autospec(ingest.Client)
    monkeypatch.setattr(ingest, "Client", client_factory)
    monkeypatch.setattr(sys, "argv", ["ingest.py", *flags])

    with pytest.raises(SystemExit) as error:
        ingest.main()

    assert error.value.code == 2
    client_factory.assert_not_called()
