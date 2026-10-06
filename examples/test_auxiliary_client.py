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
@pytest.mark.parametrize("inference_status", [200, 400])
def test_native_client_lifecycle(monkeypatch, provider, inference_status):
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
                if inference_status == 400:
                    self.respond(400, {"error": {"message": "fake failure", "type": "invalid_request_error"}})
                else:
                    self.respond(200, {
                        "id": "chat_fake", "object": "chat.completion", "created": 1,
                        "model": "fixed-helper", "choices": [{"index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                    })
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
        if inference_status == 400:
            with pytest.raises(Exception, match="fake failure"):
                asyncio.run(example.main())
        else:
            asyncio.run(example.main())
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
    if inference_status == 400:
        assert completion["termination_reason"] == "ERROR"
    else:
        assert completion == {}
