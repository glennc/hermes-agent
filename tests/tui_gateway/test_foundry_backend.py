import base64
import io
import json
import sys
import threading
import time
import types

import pytest


_original_stdout = sys.stdout


def _drain_subscribers(backend) -> None:
    """Signal cancel on every session subscriber and wait briefly for exit."""
    threads = []
    for session in list(backend._sessions.values()):
        cancel = session.get("subscriber_cancel")
        if cancel is not None:
            cancel.set()
        active = session.get("active_response")
        if active is not None:
            try:
                active.close()
            except Exception:
                pass
        thread = session.get("subscriber_thread")
        if thread is not None:
            threads.append(thread)
    for thread in threads:
        thread.join(timeout=1.0)


@pytest.fixture(autouse=True)
def foundry_backend(monkeypatch):
    from tui_gateway import foundry_backend as backend

    # Default: any subscriber the fixture spawns blocks on cancel rather than
    # hitting the network. Tests that exercise the subscriber loop override
    # this via monkeypatch.setattr(backend, "_post_invocation_events", ...).
    def noop_events(session, payload, cancel):
        cancel.wait(timeout=10.0)
        if False:  # pragma: no cover - generator marker
            yield {}

    real_start_catalog_prewarm = backend._start_catalog_prewarm
    backend._start_catalog_prewarm_for_tests = real_start_catalog_prewarm
    monkeypatch.setenv("HERMES_FOUNDRY_WORKSPACE_KEY", "ws-test")
    monkeypatch.setattr(backend, "_post_invocation_events", noop_events)
    monkeypatch.setattr(backend, "_start_catalog_prewarm", lambda sid: None)

    backend._sessions.clear()
    backend._pending_controls.clear()
    backend._cached_oid = None
    backend._invalidate_catalog()
    backend._shutdown_credential()
    yield backend
    _drain_subscribers(backend)
    backend._sessions.clear()
    backend._pending_controls.clear()
    backend._cached_oid = None
    backend._invalidate_catalog()
    backend._shutdown_credential()
    sys.stdout = _original_stdout


def _mock_session_create_response(monkeypatch, backend, sid: str = "remote-a") -> None:
    def fake_post_rpc(session, request):
        if request["method"] == "session.create":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "session_id": sid,
                    "info": {
                        "model": "foundry/test",
                        "tools": {},
                        "skills": {},
                        "cwd": "/workspace",
                        "lazy": True,
                    },
                },
            }
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"status": "ok"}}

    monkeypatch.setattr(backend, "_post_rpc", fake_post_rpc)


def test_session_create_returns_foundry_info(monkeypatch, foundry_backend):
    def fake_post_rpc(session, request):
        if request["method"] == "session.create":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "session_id": "remote-a",
                    "info": {
                        "model": "anthropic/claude-sonnet-4",
                        "tools": {},
                        "skills": {},
                        "cwd": "/workspace",
                        "lazy": True,
                    },
                },
            }
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"status": "ok"}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    resp = foundry_backend.dispatch({"id": "r1", "method": "session.create", "params": {}})

    assert resp["result"]["session_id"] == "remote-a"
    info = resp["result"]["info"]
    assert info["model"] == "anthropic/claude-sonnet-4"
    assert info["tools"] == {}
    assert info["lazy"] is True
    assert "remote-a" in foundry_backend._sessions


def test_setup_status_tunnels_to_child_gateway(monkeypatch, foundry_backend):
    requests: list[dict] = []

    def fake_post_rpc(_session, request):
        requests.append(request)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"provider_configured": False},
        }

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    resp = foundry_backend.dispatch({"id": "setup", "method": "setup.status", "params": {}})

    assert resp == {
        "jsonrpc": "2.0",
        "id": "setup",
        "result": {"provider_configured": False},
    }
    assert requests == [
        {"jsonrpc": "2.0", "id": "setup", "method": "setup.status", "params": {}}
    ]


def test_unknown_method_without_session_tunnels_to_workspace(monkeypatch, foundry_backend):
    requests: list[dict] = []

    def fake_post_rpc(_session, request):
        requests.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    resp = foundry_backend.dispatch(
        {"id": "unknown", "method": "model.options", "params": {"foo": "bar"}}
    )

    assert resp["result"] == {"ok": True}
    assert requests == [
        {
            "jsonrpc": "2.0",
            "id": "unknown",
            "method": "model.options",
            "params": {"foo": "bar"},
        }
    ]


def test_unknown_method_with_known_session_tunnels_to_session(monkeypatch, foundry_backend):
    requests: list[dict] = []
    foundry_backend._register_session("remote-a", {}, "workspace-a")

    def fake_post_rpc(_session, request):
        requests.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    resp = foundry_backend.dispatch(
        {"id": "unknown", "method": "rollback.list", "params": {"session_id": "remote-a"}}
    )

    assert resp["result"] == {"ok": True}
    assert requests == [
        {
            "jsonrpc": "2.0",
            "id": "unknown",
            "method": "rollback.list",
            "params": {"session_id": "remote-a"},
        }
    ]


@pytest.mark.parametrize(
    "method_name,params",
    [
        ("session.list", {}),
        ("complete.path", {"prefix": "/tmp"}),
    ],
)
def test_fast_handlers_tunnel_synchronously_to_child_gateway(
    monkeypatch, foundry_backend, method_name, params
):
    requests: list[dict] = []
    if params.get("session_id"):
        foundry_backend._register_session("remote-a", {}, "workspace-a")

    def fake_post_rpc(_session, request):
        requests.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    resp = foundry_backend.dispatch({"id": "call", "method": method_name, "params": params})

    assert resp["result"] == {"ok": True}
    assert requests == [
        {"jsonrpc": "2.0", "id": "call", "method": method_name, "params": params}
    ]


@pytest.mark.parametrize(
    "method_name,params",
    [
        ("command.dispatch", {"name": "status"}),
        ("shell.exec", {"session_id": "remote-a", "command": "pwd"}),
        ("slash.exec", {"session_id": "remote-a", "command": "/help"}),
        ("session.compress", {"session_id": "remote-a"}),
    ],
)
def test_long_handlers_dispatch_on_pool_and_write_via_transport(
    monkeypatch, foundry_backend, method_name, params
):
    written: list[dict] = []
    finished = threading.Event()

    class _Transport:
        def write(self, frame):
            written.append(frame)
            finished.set()
            return True

    requests: list[dict] = []
    if params.get("session_id"):
        foundry_backend._register_session("remote-a", {}, "workspace-a")

    def fake_post_rpc(_session, request):
        requests.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"ok": True}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    transport = _Transport()
    resp = foundry_backend.dispatch(
        {"id": "call", "method": method_name, "params": params},
        transport=transport,
    )

    # Slow methods are dispatched to the pool; dispatch() returns None and
    # the eventual response is written via the bound transport.
    assert resp is None
    assert finished.wait(timeout=2.0), f"async dispatch never wrote a response for {method_name}"
    assert written == [{"jsonrpc": "2.0", "id": "call", "result": {"ok": True}}]
    assert requests == [
        {"jsonrpc": "2.0", "id": "call", "method": method_name, "params": params}
    ]


def test_invocations_url_uses_foundry_protocol_path(monkeypatch, foundry_backend):
    monkeypatch.setenv("HERMES_FOUNDRY_ENDPOINT", "http://localhost:8088")
    monkeypatch.setenv("HERMES_FOUNDRY_AGENT_NAME", "agent/name")
    monkeypatch.setenv("HERMES_FOUNDRY_WORKSPACE_KEY", "workspace one")
    monkeypatch.delenv("HERMES_FOUNDRY_INVOCATIONS_PATH", raising=False)

    url = foundry_backend._invocations_url({"workspace": "workspace one"})

    assert url.startswith("http://localhost:8088/agents/agent%2Fname/endpoint/protocols/invocations?")
    assert "agent_session_id=workspace+one" in url
    assert "api-version=v1" in url


def test_endpoint_prefers_project_endpoint_for_agent_invocations(monkeypatch, foundry_backend):
    monkeypatch.delenv("HERMES_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("HERMES_FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("HERMES_FOUNDRY_LOCAL_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_AI_PROJECT_NAME", "project one")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://acct.openai.azure.com/")
    monkeypatch.setenv(
        "AZURE_AI_PROJECT_ENDPOINT",
        "https://acct.services.ai.azure.com/api/projects/project-one",
    )

    assert (
        foundry_backend._endpoint()
        == "https://acct.services.ai.azure.com/api/projects/project-one"
    )


def test_endpoint_normalizes_openai_base_url_suffix(monkeypatch, foundry_backend):
    monkeypatch.delenv("HERMES_FOUNDRY_ENDPOINT", raising=False)
    monkeypatch.delenv("HERMES_FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("HERMES_FOUNDRY_LOCAL_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_AI_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_AI_SERVICES_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.setenv("AZURE_AI_PROJECT_NAME", "project-one")
    monkeypatch.setenv(
        "AZURE_FOUNDRY_BASE_URL", "https://acct.openai.azure.com/openai/v1"
    )

    assert (
        foundry_backend._endpoint()
        == "https://acct.openai.azure.com/api/projects/project-one"
    )


def test_headers_include_hosted_agents_preview_feature(monkeypatch, foundry_backend):
    monkeypatch.setenv("HERMES_FOUNDRY_BEARER_TOKEN", "test-token")

    headers = foundry_backend._headers()

    assert headers["Foundry-Features"] == "HostedAgents=V1Preview"
    assert headers["Authorization"] == "Bearer test-token"


def test_invocations_url_uses_configured_local_path(monkeypatch, foundry_backend):
    monkeypatch.setenv("HERMES_FOUNDRY_ENDPOINT", "http://localhost:8088")
    monkeypatch.setenv("HERMES_FOUNDRY_INVOCATIONS_PATH", "/invocations")

    assert foundry_backend._invocations_url({}) == "http://localhost:8088/invocations"


def test_rpc_payload_includes_invocation_request_and_session(foundry_backend):
    session = {"workspace": "workspace-a"}
    request = foundry_backend._rpc_request(
        "prompt.submit",
        {"session_id": "sid-a", "text": "hello"},
        "rpc-a",
    )

    payload = foundry_backend._rpc_payload("sid-a", session, request, "invoke-a")

    assert payload == {
        "kind": "hermes.rpc",
        "invocation_id": "invoke-a",
        "request": {
            "jsonrpc": "2.0",
            "id": "rpc-a",
            "method": "prompt.submit",
            "params": {"session_id": "sid-a", "text": "hello"},
        },
        "session": {"id": "sid-a", "workspace": "workspace-a"},
        "tui": {"protocol_version": 1},
    }


def test_call_session_rpc_includes_session_context(monkeypatch, foundry_backend):
    calls: list[dict] = []
    session = {"sid": "sid-a", "workspace": "workspace-a"}

    def fake_post_rpc(_session, request):
        calls.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"status": "ok"}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    response = foundry_backend._call_session_rpc(
        session,
        "clarify.respond",
        {"request_id": "req-a", "answer": "yes"},
        "control-a",
    )

    assert response["result"]["status"] == "ok"
    assert calls == [
        {
            "jsonrpc": "2.0",
            "id": "control-a",
            "method": "clarify.respond",
            "params": {"request_id": "req-a", "answer": "yes", "session_id": "sid-a"},
        }
    ]


def test_sse_event_parser_handles_tui_events(foundry_backend):
    class _Response:
        def __iter__(self):
            yield b'data: {"type":"message.start","payload":{}}\n'
            yield b"\n"
            yield b'data: {"type":"message.delta","payload":{"text":"hi"}}\n'
            yield b"\n"
            yield b'data: {"type":"done"}\n'
            yield b"\n"

    events = list(foundry_backend._iter_sse_events(_Response(), threading.Event()))

    assert [event["type"] for event in events] == [
        "message.start",
        "message.delta",
        "done",
    ]


def test_clarify_respond_sends_control_without_session_id(monkeypatch, foundry_backend):
    requests: list[dict] = []

    def fake_post_rpc(session, request):
        if request["method"] == "session.create":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "session_id": "remote-a",
                    "info": {
                        "model": "foundry/test",
                        "tools": {},
                        "skills": {},
                        "cwd": "/workspace",
                        "lazy": True,
                    },
                },
            }
        # Background catalog pre-warm fires after session.create. Ignore
        # it here — this test only cares about the clarify.respond call.
        if request["method"] == "commands.catalog":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"pairs": [], "sub": {}, "canon": {}, "categories": [], "skill_count": 0, "warning": ""},
            }
        requests.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"status": "ok"}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    created = foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})
    sid = created["result"]["session_id"]
    foundry_backend._pending_controls["req-a"] = sid

    resp = foundry_backend.dispatch(
        {
            "id": "clarify",
            "method": "clarify.respond",
            "params": {"request_id": "req-a", "answer": "interrupt handling"},
        }
    )

    assert resp["result"]["status"] == "ok"
    assert requests == [
        {
            "jsonrpc": "2.0",
            "id": "clarify",
            "method": "clarify.respond",
            "params": {
                "request_id": "req-a",
                "answer": "interrupt handling",
                "session_id": sid,
            },
        }
    ]
    assert "req-a" not in foundry_backend._pending_controls


def test_session_interrupt_passes_through_and_leaves_stream_to_drain(monkeypatch, foundry_backend):
    rpc_calls: list[dict] = []
    _mock_session_create_response(monkeypatch, foundry_backend, "remote-a")

    class _Response:
        closed = False

        def close(self):
            self.closed = True

    created = foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})
    sid = created["result"]["session_id"]
    session = foundry_backend._sessions[sid]
    cancel_event = threading.Event()
    response = _Response()
    session["active_invocation_id"] = "invoke-a"
    session["active_response"] = response
    session["cancel_event"] = cancel_event
    session["running"] = True

    def fake_post_rpc(_session, request):
        rpc_calls.append(request)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"status": "interrupted"},
        }

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    resp = foundry_backend.dispatch(
        {"id": "interrupt", "method": "session.interrupt", "params": {"session_id": sid}}
    )

    assert resp["result"]["status"] == "interrupted"
    assert cancel_event.is_set() is False
    assert response.closed is False
    assert session["running"] is True
    assert rpc_calls == [
        {
            "jsonrpc": "2.0",
            "id": "interrupt",
            "method": "session.interrupt",
            "params": {"session_id": sid},
        }
    ]


def test_abort_interrupt_signals_subscriber_and_closes_active_stream(monkeypatch, foundry_backend):
    rpc_calls: list[tuple[str, dict]] = []
    _mock_session_create_response(monkeypatch, foundry_backend, "remote-a")

    class _Response:
        closed = False

        def close(self):
            self.closed = True

    created = foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})
    sid = created["result"]["session_id"]
    session = foundry_backend._sessions[sid]
    subscriber_cancel = session["subscriber_cancel"]
    response = _Response()
    session["active_response"] = response

    monkeypatch.setattr(
        foundry_backend,
        "_send_rpc_async",
        lambda _session, method, params, _rid=None: rpc_calls.append((method, params)),
    )

    foundry_backend._interrupt_session(session, "session closed", abort_stream=True)

    assert subscriber_cancel.is_set()
    assert response.closed is True
    assert rpc_calls == [
        ("session.interrupt", {"session_id": sid, "reason": "session closed"})
    ]


def test_prompt_submit_is_plain_rpc(monkeypatch, foundry_backend):
    """prompt.submit no longer streams. It returns whatever upstream returns,
    matching local Hermes (which immediately returns `{status: "streaming"}` and
    fires events asynchronously)."""

    captured_requests: list[dict] = []

    def fake_post_rpc(session, request):
        captured_requests.append(request)
        if request["method"] == "session.create":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "session_id": "remote-a",
                    "info": {
                        "model": "foundry/test",
                        "tools": {},
                        "skills": {},
                        "cwd": "/workspace",
                        "lazy": True,
                    },
                },
            }
        if request["method"] == "prompt.submit":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"status": "streaming"},
            }
        return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    created = foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})
    sid = created["result"]["session_id"]
    submitted = foundry_backend.dispatch(
        {
            "id": "submit",
            "method": "prompt.submit",
            "params": {"session_id": sid, "text": "hello"},
        }
    )

    assert submitted == {
        "jsonrpc": "2.0",
        "id": "submit",
        "result": {"status": "streaming"},
    }
    # commands.catalog is pre-warmed asynchronously after session.create; the
    # test only cares about the prompt.submit / session.create ordering.
    methods = [
        req["method"]
        for req in captured_requests
        if req["method"] != "commands.catalog"
    ]
    assert methods == ["session.create", "prompt.submit"]


def test_event_subscriber_forwards_events_to_tui(monkeypatch, foundry_backend):
    """Events emitted by the hosted gateway during a turn flow back to the TUI
    via the per-session subscriber thread (POSTing session.events), not via
    prompt.submit's response."""

    frames: list[dict] = []
    _mock_session_create_response(monkeypatch, foundry_backend, "remote-a")

    captured_payloads: list[dict] = []
    stream_started = threading.Event()

    def fake_events(session, payload, cancel):
        captured_payloads.append(payload)
        stream_started.set()
        yield {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"type": "message.start", "session_id": "remote-a", "seq": 0},
        }
        yield {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "message.delta",
                "session_id": "remote-a",
                "seq": 1,
                "payload": {"text": "hi"},
            },
        }
        yield {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "message.complete",
                "session_id": "remote-a",
                "seq": 2,
                "payload": {"text": "hi"},
            },
        }
        cancel.wait(timeout=2.0)

    def capture(obj: dict) -> bool:
        frames.append(json.loads(json.dumps(obj)))
        return True

    monkeypatch.setattr(foundry_backend, "_post_invocation_events", fake_events)
    monkeypatch.setattr(foundry_backend, "write_json", capture)

    foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})

    assert stream_started.wait(timeout=2.0)
    deadline = time.time() + 2.0
    while (
        len([f for f in frames if f.get("method") == "event"]) < 3
        and time.time() < deadline
    ):
        time.sleep(0.01)

    event_types = [
        frame["params"]["type"]
        for frame in frames
        if frame.get("method") == "event"
    ]
    assert event_types[:3] == ["message.start", "message.delta", "message.complete"]

    # Subscriber requested session.events with the right cursor.
    assert captured_payloads, "subscriber never posted session.events"
    request = captured_payloads[0]["request"]
    assert request["method"] == "session.events"
    assert request["params"]["session_id"] == "remote-a"
    assert request["params"]["since_seq"] == -1


def test_event_subscriber_dedupes_by_seq(monkeypatch, foundry_backend):
    """Events with seq <= last_seen_seq are dropped (idempotent reconnect)."""

    frames: list[dict] = []
    _mock_session_create_response(monkeypatch, foundry_backend, "remote-a")

    iteration_count = {"n": 0}
    stream_done = threading.Event()

    def fake_events(session, payload, cancel):
        iteration_count["n"] += 1
        # First connection: emit two events, then disconnect.
        # Second connection: replay the second event (with same seq) plus a new one.
        if iteration_count["n"] == 1:
            yield {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {"type": "message.start", "session_id": "remote-a", "seq": 0},
            }
            yield {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {"type": "message.delta", "session_id": "remote-a", "seq": 1, "payload": {"text": "hi"}},
            }
            return
        yield {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"type": "message.delta", "session_id": "remote-a", "seq": 1, "payload": {"text": "hi"}},
        }
        yield {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {"type": "message.complete", "session_id": "remote-a", "seq": 2, "payload": {"text": "hi"}},
        }
        stream_done.set()
        cancel.wait(timeout=2.0)

    def capture(obj: dict) -> bool:
        frames.append(json.loads(json.dumps(obj)))
        return True

    monkeypatch.setattr(foundry_backend, "_post_invocation_events", fake_events)
    monkeypatch.setattr(foundry_backend, "write_json", capture)
    monkeypatch.setattr(foundry_backend, "_SUBSCRIBER_INITIAL_BACKOFF_S", 0.0)

    foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})

    assert stream_done.wait(timeout=2.0)

    event_types = [
        frame["params"]["type"]
        for frame in frames
        if frame.get("method") == "event"
    ]
    # Each event should appear exactly once even though seq=1 was replayed.
    assert event_types.count("message.delta") == 1
    assert event_types == ["message.start", "message.delta", "message.complete"]


def test_event_subscriber_translates_replay_gap_to_warning(monkeypatch, foundry_backend):
    frames: list[dict] = []
    _mock_session_create_response(monkeypatch, foundry_backend, "remote-a")

    stream_done = threading.Event()

    def fake_events(session, payload, cancel):
        yield {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "replay.gap",
                "session_id": "remote-a",
                "seq": 41,
                "payload": {"missed_through": 41},
            },
        }
        yield {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "message.complete",
                "session_id": "remote-a",
                "seq": 42,
                "payload": {"text": "ok"},
            },
        }
        stream_done.set()
        cancel.wait(timeout=2.0)

    def capture(obj: dict) -> bool:
        frames.append(json.loads(json.dumps(obj)))
        return True

    monkeypatch.setattr(foundry_backend, "_post_invocation_events", fake_events)
    monkeypatch.setattr(foundry_backend, "write_json", capture)

    foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})

    assert stream_done.wait(timeout=2.0)
    deadline = time.time() + 2.0
    while (
        len([f for f in frames if f.get("method") == "event"]) < 3
        and time.time() < deadline
    ):
        time.sleep(0.01)

    event_types = [
        frame["params"]["type"]
        for frame in frames
        if frame.get("method") == "event"
    ]
    # The raw gap event is forwarded AND a status.update warning is emitted.
    assert event_types == ["replay.gap", "status.update", "message.complete"]
    warning = next(
        frame for frame in frames
        if frame.get("method") == "event"
        and frame["params"]["type"] == "status.update"
    )
    assert warning["params"]["payload"]["kind"] == "warning"


def test_session_close_stops_subscriber(monkeypatch, foundry_backend):
    _mock_session_create_response(monkeypatch, foundry_backend, "remote-a")

    started = threading.Event()
    exited = threading.Event()

    def fake_events(session, payload, cancel):
        started.set()
        cancel.wait(timeout=5.0)
        exited.set()
        if False:  # pragma: no cover - generator marker
            yield {}

    monkeypatch.setattr(foundry_backend, "_post_invocation_events", fake_events)

    created = foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})
    sid = created["result"]["session_id"]
    session = foundry_backend._sessions[sid]
    subscriber_cancel = session["subscriber_cancel"]
    subscriber_thread = session["subscriber_thread"]

    assert started.wait(timeout=2.0)

    foundry_backend.dispatch(
        {"id": "close", "method": "session.close", "params": {"session_id": sid}}
    )

    assert subscriber_cancel.is_set()
    subscriber_thread.join(timeout=2.0)
    assert not subscriber_thread.is_alive()
    assert sid not in foundry_backend._sessions


def test_write_json_uses_stdio_transport(monkeypatch, foundry_backend):
    buf = io.StringIO()
    monkeypatch.setattr(foundry_backend, "_real_stdout", buf)

    assert foundry_backend.write_json({"ok": True})
    assert json.loads(buf.getvalue()) == {"ok": True}


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a"
        "0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63600000000000010001"
        "5a4d2e1e0000000049454e44ae426082"
    )


def test_clipboard_paste_reads_local_clipboard_and_forwards_bytes(monkeypatch, foundry_backend, tmp_path):
    payload = _png_bytes()

    def fake_save_clipboard_image(dest):
        from pathlib import Path as _Path
        _Path(dest).write_bytes(payload)
        return True

    fake_clipboard = types.SimpleNamespace(
        save_clipboard_image=fake_save_clipboard_image,
        has_clipboard_image=lambda: True,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.clipboard", fake_clipboard)

    captured: list[dict] = []

    def fake_post_rpc(_session, request):
        captured.append(request)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "attached": True,
                "path": "/agent/.hermes/images/clip_xxx.png",
                "count": 1,
                "width": 1,
                "height": 1,
                "name": "clip_xxx.png",
            },
        }

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    foundry_backend._register_session("remote-a", {}, "ws-test")

    response = foundry_backend.dispatch(
        {"id": "paste", "method": "clipboard.paste", "params": {"session_id": "remote-a"}}
    )

    assert response["result"]["attached"] is True
    assert response["result"]["path"].endswith("clip_xxx.png")
    assert captured, "clipboard.paste did not forward upstream"
    forwarded = captured[0]
    assert forwarded["method"] == "image.attach"
    assert forwarded["params"]["session_id"] == "remote-a"
    bytes_b64 = forwarded["params"]["bytes_b64"]
    assert base64.b64decode(bytes_b64) == payload
    assert forwarded["params"]["filename"].startswith("clip_")
    assert forwarded["params"]["filename"].endswith(".png")


def test_clipboard_paste_reports_no_image_when_empty(monkeypatch, foundry_backend):
    fake_clipboard = types.SimpleNamespace(
        save_clipboard_image=lambda dest: False,
        has_clipboard_image=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "hermes_cli.clipboard", fake_clipboard)

    foundry_backend._register_session("remote-a", {}, "ws-test")
    response = foundry_backend.dispatch(
        {"id": "paste", "method": "clipboard.paste", "params": {"session_id": "remote-a"}}
    )

    assert response["result"]["attached"] is False
    assert "No image" in response["result"]["message"]


def test_image_attach_reads_local_path_and_forwards_bytes(monkeypatch, foundry_backend, tmp_path):
    src = tmp_path / "shot.png"
    payload = _png_bytes()
    src.write_bytes(payload)

    captured: list[dict] = []

    def fake_post_rpc(_session, request):
        captured.append(request)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"attached": True, "path": "/agent/.hermes/images/shot.png", "count": 1},
        }

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)
    foundry_backend._register_session("remote-a", {}, "ws-test")

    response = foundry_backend.dispatch(
        {
            "id": "attach",
            "method": "image.attach",
            "params": {"session_id": "remote-a", "path": str(src)},
        }
    )

    assert response["result"]["attached"] is True
    # The pre-warm thread fires commands.catalog asynchronously after
    # _register_session; filter it out of the captured stream.
    image_calls = [c for c in captured if c["method"] == "image.attach"]
    assert image_calls
    assert image_calls[0]["params"]["bytes_b64"]
    assert base64.b64decode(image_calls[0]["params"]["bytes_b64"]) == payload
    assert image_calls[0]["params"]["filename"] == "shot.png"


def test_image_attach_sandbox_only_path_falls_through(monkeypatch, foundry_backend):
    captured: list[dict] = []

    def fake_post_rpc(_session, request):
        captured.append(request)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"attached": True, "path": "/agent/workspace/screenshot.png", "count": 1},
        }

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)
    foundry_backend._register_session("remote-a", {}, "ws-test")

    # A path that doesn't resolve on the local FS — must be passthrough.
    response = foundry_backend.dispatch(
        {
            "id": "attach",
            "method": "image.attach",
            "params": {"session_id": "remote-a", "path": "/agent/workspace/screenshot.png"},
        }
    )

    assert response["result"]["attached"] is True
    assert captured
    assert "bytes_b64" not in captured[0]["params"]
    assert captured[0]["params"]["path"] == "/agent/workspace/screenshot.png"


def test_image_attach_passes_through_when_caller_already_supplied_bytes(monkeypatch, foundry_backend):
    captured: list[dict] = []

    def fake_post_rpc(_session, request):
        captured.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"attached": True}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)
    foundry_backend._register_session("remote-a", {}, "ws-test")

    encoded = base64.b64encode(b"hi").decode("ascii")
    foundry_backend.dispatch(
        {
            "id": "attach",
            "method": "image.attach",
            "params": {
                "session_id": "remote-a",
                "bytes_b64": encoded,
                "filename": "x.png",
            },
        }
    )

    assert captured[0]["params"]["bytes_b64"] == encoded


def test_input_detect_drop_image_uploads_bytes(monkeypatch, foundry_backend, tmp_path):
    src = tmp_path / "drag.png"
    src.write_bytes(_png_bytes())

    captured: list[dict] = []

    def fake_post_rpc(_session, request):
        captured.append(request)
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {
                "attached": True,
                "path": "/agent/.hermes/images/drag.png",
                "count": 1,
                "width": 1,
                "height": 1,
                "token_estimate": 85,
                "name": "drag.png",
                "text": "[User attached image: drag.png]",
            },
        }

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)
    foundry_backend._register_session("remote-a", {}, "ws-test")

    response = foundry_backend.dispatch(
        {
            "id": "drop",
            "method": "input.detect_drop",
            "params": {"session_id": "remote-a", "text": str(src)},
        }
    )

    assert response["result"]["matched"] is True
    assert response["result"]["is_image"] is True
    assert response["result"]["path"] == str(src)
    assert response["result"]["width"] == 1
    assert response["result"]["height"] == 1
    assert response["result"]["token_estimate"] == 85
    image_calls = [c for c in captured if c["method"] == "image.attach"]
    assert image_calls
    assert image_calls[0]["params"]["bytes_b64"]


def test_input_detect_drop_non_image_returns_marker_without_upload(monkeypatch, foundry_backend, tmp_path):
    src = tmp_path / "notes.txt"
    src.write_text("hello world")

    posted: list[dict] = []
    monkeypatch.setattr(
        foundry_backend, "_post_rpc",
        lambda _s, request: (posted.append(request), {"jsonrpc": "2.0", "id": request["id"], "result": {}})[1],
    )
    foundry_backend._register_session("remote-a", {}, "ws-test")

    response = foundry_backend.dispatch(
        {
            "id": "drop",
            "method": "input.detect_drop",
            "params": {"session_id": "remote-a", "text": str(src)},
        }
    )

    assert response["result"]["matched"] is True
    assert response["result"]["is_image"] is False
    assert response["result"]["name"] == "notes.txt"
    assert "[User attached file:" in response["result"]["text"]
    # No upstream RPC should have been made — non-image is local-only.
    assert posted == []


def test_input_detect_drop_no_match(monkeypatch, foundry_backend):
    foundry_backend._register_session("remote-a", {}, "ws-test")
    response = foundry_backend.dispatch(
        {
            "id": "drop",
            "method": "input.detect_drop",
            "params": {"session_id": "remote-a", "text": "just some plain text"},
        }
    )

    assert response["result"] == {"matched": False}


def _catalog_result(extra_pairs: list[list[str]] | None = None) -> dict:
    pairs = [
        ["/help", "Show help"],
        ["/quit", "Exit the TUI"],
        ["/model", "Switch active model"],
        ["/clear", "Start a fresh session"],
        ["/compact", "Toggle compact display mode"],
        ["/logs", "Show recent gateway log lines"],
        ["/mouse", "Toggle mouse/wheel tracking"],
    ]
    if extra_pairs:
        pairs.extend(extra_pairs)
    return {
        "pairs": pairs,
        "sub": {},
        "canon": {p[0].lower(): p[0] for p in pairs},
        "categories": [{"name": "All", "pairs": list(pairs)}],
        "skill_count": 0,
        "warning": "",
    }


def test_commands_catalog_second_call_returns_from_cache(monkeypatch, foundry_backend):
    upstream_calls: list[dict] = []

    def fake_post_rpc(_session, request):
        upstream_calls.append(request)
        return {"jsonrpc": "2.0", "id": request["id"], "result": _catalog_result()}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    first = foundry_backend.dispatch(
        {"id": "cat1", "method": "commands.catalog", "params": {}}
    )
    second = foundry_backend.dispatch(
        {"id": "cat2", "method": "commands.catalog", "params": {}}
    )

    assert first["result"]["pairs"] == second["result"]["pairs"]
    assert first["id"] == "cat1"
    assert second["id"] == "cat2"
    # Exactly one upstream RPC for the catalog, even though TUI asked twice.
    catalog_calls = [c for c in upstream_calls if c["method"] == "commands.catalog"]
    assert len(catalog_calls) == 1


def test_complete_slash_prefix_match_runs_locally(monkeypatch, foundry_backend):
    upstream_calls: list[dict] = []

    def fake_post_rpc(_session, request):
        upstream_calls.append(request)
        if request["method"] == "commands.catalog":
            return {"jsonrpc": "2.0", "id": request["id"], "result": _catalog_result()}
        return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    # Warm cache via commands.catalog.
    foundry_backend.dispatch({"id": "cat", "method": "commands.catalog", "params": {}})
    upstream_calls.clear()

    response = foundry_backend.dispatch(
        {"id": "comp", "method": "complete.slash", "params": {"text": "/h"}}
    )

    assert response["result"]["replace_from"] == 1
    items = response["result"]["items"]
    names = [item["text"] for item in items]
    assert "/help" in names
    # No additional upstream RPC fired — the filter ran locally.
    assert upstream_calls == []


def test_complete_slash_includes_hardcoded_tui_extras(monkeypatch, foundry_backend):
    def fake_post_rpc(_session, request):
        if request["method"] == "commands.catalog":
            # Catalog with NONE of the TUI extras; complete.slash must
            # synthesize them locally to match upstream behavior.
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "pairs": [["/help", "Show help"]],
                    "sub": {},
                    "canon": {"/help": "/help"},
                    "categories": [],
                    "skill_count": 0,
                    "warning": "",
                },
            }
        return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    foundry_backend.dispatch({"id": "cat", "method": "commands.catalog", "params": {}})

    # `/d` should match the /details extra even though it's not in the
    # catalog pairs.
    response = foundry_backend.dispatch(
        {"id": "comp", "method": "complete.slash", "params": {"text": "/d"}}
    )
    names = [item["text"] for item in response["result"]["items"]]
    assert names == ["/details"]


def test_complete_slash_sub_arg_falls_through_to_upstream(monkeypatch, foundry_backend):
    upstream_calls: list[dict] = []

    def fake_post_rpc(_session, request):
        upstream_calls.append(request)
        if request["method"] == "commands.catalog":
            return {"jsonrpc": "2.0", "id": request["id"], "result": _catalog_result()}
        if request["method"] == "complete.slash":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {
                    "items": [{"text": "gpt-4o", "display": "gpt-4o", "meta": "openai"}],
                    "replace_from": 7,
                },
            }
        return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    foundry_backend.dispatch({"id": "cat", "method": "commands.catalog", "params": {}})
    upstream_calls.clear()

    response = foundry_backend.dispatch(
        {"id": "comp", "method": "complete.slash", "params": {"text": "/model gpt"}}
    )

    # Sub-arg case must passthrough; result mirrors what upstream returned.
    assert response["result"]["items"] == [
        {"text": "gpt-4o", "display": "gpt-4o", "meta": "openai"}
    ]
    forwarded = [c for c in upstream_calls if c["method"] == "complete.slash"]
    assert len(forwarded) == 1


def test_complete_slash_cache_miss_falls_through(monkeypatch, foundry_backend):
    upstream_calls: list[dict] = []

    def fake_post_rpc(_session, request):
        upstream_calls.append(request)
        if request["method"] == "commands.catalog":
            return {"jsonrpc": "2.0", "id": request["id"], "error": {"code": 500, "message": "down"}}
        return {
            "jsonrpc": "2.0",
            "id": request["id"],
            "result": {"items": [{"text": "/help", "display": "/help", "meta": "Show help"}], "replace_from": 1},
        }

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    response = foundry_backend.dispatch(
        {"id": "comp", "method": "complete.slash", "params": {"text": "/h"}}
    )

    forwarded = [c for c in upstream_calls if c["method"] == "complete.slash"]
    assert len(forwarded) == 1
    assert response["result"]["items"][0]["text"] == "/help"


def test_skills_reload_invalidates_catalog(monkeypatch, foundry_backend):
    catalog_calls = 0

    def fake_post_rpc(_session, request):
        nonlocal catalog_calls
        if request["method"] == "commands.catalog":
            catalog_calls += 1
            return {"jsonrpc": "2.0", "id": request["id"], "result": _catalog_result()}
        return {"jsonrpc": "2.0", "id": request["id"], "result": {"reloaded": True}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)

    foundry_backend.dispatch({"id": "cat1", "method": "commands.catalog", "params": {}})
    foundry_backend.dispatch({"id": "reload", "method": "skills.reload", "params": {}})
    foundry_backend.dispatch({"id": "cat2", "method": "commands.catalog", "params": {}})

    assert catalog_calls == 2


def test_replay_gap_invalidates_catalog(monkeypatch, foundry_backend):
    catalog_calls = 0

    def fake_post_rpc(_session, request):
        nonlocal catalog_calls
        if request["method"] == "commands.catalog":
            catalog_calls += 1
            return {"jsonrpc": "2.0", "id": request["id"], "result": _catalog_result()}
        return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)
    foundry_backend._register_session("remote-a", {}, "ws-test")

    foundry_backend.dispatch({"id": "cat1", "method": "commands.catalog", "params": {}})

    foundry_backend._emit_upstream_event(
        "remote-a",
        {
            "jsonrpc": "2.0",
            "method": "event",
            "params": {
                "type": "replay.gap",
                "session_id": "remote-a",
                "seq": 42,
                "payload": {"missed_through": 42},
            },
        },
    )

    foundry_backend.dispatch({"id": "cat2", "method": "commands.catalog", "params": {}})

    assert catalog_calls == 2


def test_session_create_prewarms_catalog(monkeypatch, foundry_backend):
    catalog_event = threading.Event()

    def fake_post_rpc(_session, request):
        if request["method"] == "session.create":
            return {
                "jsonrpc": "2.0",
                "id": request["id"],
                "result": {"session_id": "remote-a", "info": {}},
            }
        if request["method"] == "commands.catalog":
            catalog_event.set()
            return {"jsonrpc": "2.0", "id": request["id"], "result": _catalog_result()}
        return {"jsonrpc": "2.0", "id": request["id"], "result": {}}

    monkeypatch.setattr(foundry_backend, "_post_rpc", fake_post_rpc)
    monkeypatch.setattr(
        foundry_backend,
        "_start_catalog_prewarm",
        foundry_backend._start_catalog_prewarm_for_tests,
    )

    foundry_backend.dispatch({"id": "create", "method": "session.create", "params": {}})

    assert catalog_event.wait(timeout=2.0), "session.create did not pre-warm the catalog"
