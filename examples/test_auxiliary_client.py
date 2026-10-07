"""Exercise the literal example over local HTTP; no credentials or inference."""

import asyncio
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "auxiliary_client", Path(__file__).with_name("auxiliary_client.py")
)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("provider", ["openai", "litellm"])
@pytest.mark.parametrize("response_case", ["success", "http_error", "no_usage", "no_choices"])
def test_native_client_lifecycle(monkeypatch, provider, response_case):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            calls.append((self.command, self.path, dict(self.headers), None))
            assert self.path == "/api/v1/deploy/dpy_fixed"
            self.respond(200, {
                "deployment_id": "dpy_fixed", "model_slug": "fixed-helper",
                "model_slug_id": "mls_fixed", "status": "DEPLOYED", "role": "test",
                "is_active": False, "checkpoint_id": "cpt_fixed",
                "base_model_slug": "base/model", "checkpoint_step": 0,
                "created_at": "2026-01-01T00:00:00Z",
            })

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.loads(raw) if raw else None
            calls.append((self.command, self.path, dict(self.headers), body))
            if self.path == "/api/v1/trajectories":
                self.respond(200, {"tid": "traj_aux"})
            elif self.path == "/api/v1/trajectories/traj_aux/complete":
                self.respond(200, {"trajectory_id": "traj_aux", "status": "completed"})
            elif self.path == "/api/v1/deploy/dpy_fixed/chat/completions":
                if response_case == "http_error":
                    self.respond(400, {"error": {"message": "fake failure", "type": "invalid_request_error"}})
                else:
                    payload = {
                        "id": "chat_fake", "object": "chat.completion", "created": 1,
                        "model": "fixed-helper", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                    }
                    if response_case == "no_usage":
                        payload.pop("usage")
                    elif response_case == "no_choices":
                        payload["choices"] = []
                    self.respond(200, payload)
            else:
                self.respond(404, {"error": "unexpected path"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("AUXILIARY_SDK_ORIGIN", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("AUXILIARY_API_KEY", "fake-organization-key")
    monkeypatch.setenv("AUXILIARY_DEPLOYMENT_ID", "dpy_fixed")
    monkeypatch.setenv("AUXILIARY_CLIENT", provider)
    monkeypatch.setenv("TRAJECTORY_API_KEY", "fake-managed-actor-key")
    monkeypatch.setenv("TRAJECTORY_BASE_URL", "http://actor.invalid")
    try:
        if response_case == "http_error":
            with pytest.raises(Exception, match="fake failure"):
                asyncio.run(example.main())
        elif response_case == "no_choices":
            with pytest.raises(ValueError, match="no choices"):
                asyncio.run(example.main())
        else:
            result = asyncio.run(example.main())
            assert result["trajectory_id"] == "traj_aux"
            if response_case == "no_usage" and provider == "openai":
                assert result["usage"] is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert [path for _, path, _, _ in calls] == [
        "/api/v1/deploy/dpy_fixed", "/api/v1/trajectories",
        "/api/v1/deploy/dpy_fixed/chat/completions",
        "/api/v1/trajectories/traj_aux/complete",
    ]
    for _, _, headers, _ in (calls[0], calls[1], calls[3]):
        assert {k.lower(): v for k, v in headers.items()}["x-api-key"] == "fake-organization-key"
    native_headers = {k.lower(): v for k, v in calls[2][2].items()}
    assert native_headers["authorization"] == "Bearer fake-organization-key"
    assert native_headers["x-trajectory-id"] == "traj_aux"
    assert calls[1][3] is None  # No actor/training/task association supplied.
    assert calls[2][3]["model"] == "fixed-helper"
    assert calls[2][3]["max_tokens"] == 128
    completion = calls[3][3]
    if response_case in {"http_error", "no_choices"}:
        assert completion["termination_reason"] == "ERROR"
    else:
        assert completion == {}
