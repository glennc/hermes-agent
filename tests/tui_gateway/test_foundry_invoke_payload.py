import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


class _InvocationAgentServerHostStub:
    def invoke_handler(self, handler):
        return handler

    def run(self):
        raise AssertionError("agent host should not run during tests")


def _load_agent_main(monkeypatch):
    invocations = types.ModuleType("azure.ai.agentserver.invocations")
    invocations.InvocationAgentServerHost = _InvocationAgentServerHostStub
    monkeypatch.setitem(sys.modules, "azure", types.ModuleType("azure"))
    monkeypatch.setitem(sys.modules, "azure.ai", types.ModuleType("azure.ai"))
    monkeypatch.setitem(
        sys.modules, "azure.ai.agentserver", types.ModuleType("azure.ai.agentserver")
    )
    monkeypatch.setitem(sys.modules, "azure.ai.agentserver.invocations", invocations)
    routine_provisioner = types.ModuleType("routine_provisioner")
    routine_provisioner.schedule_maintenance_routine = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "routine_provisioner", routine_provisioner)
    telemetry = types.ModuleType("telemetry")
    telemetry.ensure_connection_string_env = lambda: None
    telemetry.record_maintenance = lambda _result: None
    monkeypatch.setitem(sys.modules, "telemetry", telemetry)

    repo_root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "agent" / "main.py").exists()
        ),
        None,
    )
    if repo_root is None:
        pytest.skip("hermes-foundry-tui agent/main.py is not available")
    spec = importlib.util.spec_from_file_location(
        "foundry_agent_main_under_test", repo_root / "agent" / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Request:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()
        self.state = types.SimpleNamespace()

    async def body(self):
        return self._body


async def _collect_sse(response):
    frames = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8")
        assert chunk.startswith("data: ")
        frames.append(json.loads(chunk[len("data: ") :].strip()))
    return frames


def test_routine_input_string_routes_to_maintenance(monkeypatch):
    agent_main = _load_agent_main(monkeypatch)
    captured = {}

    async def handle_maintenance(payload):
        captured["payload"] = payload
        return agent_main.JSONResponse({"ok": True, "jobs": payload["jobs"]})

    monkeypatch.setattr(agent_main, "_handle_maintenance", handle_maintenance)
    request = _Request(
        {
            "input": json.dumps(
                {
                    "kind": "hermes.maintenance",
                    "session_id": "tui-session",
                    "jobs": ["all"],
                    "timeout_seconds": 540,
                }
            )
        }
    )

    response = asyncio.run(agent_main.handle_invoke(request))

    assert response.status_code == 200
    assert captured["payload"] == {
        "kind": "hermes.maintenance",
        "session_id": "tui-session",
        "jobs": ["all"],
        "timeout_seconds": 540,
    }
    assert json.loads(response.body) == {"ok": True, "jobs": ["all"]}


def test_routine_input_object_routes_to_maintenance(monkeypatch):
    agent_main = _load_agent_main(monkeypatch)
    captured = {}

    async def handle_maintenance(payload):
        captured["payload"] = payload
        return agent_main.JSONResponse({"ok": True})

    monkeypatch.setattr(agent_main, "_handle_maintenance", handle_maintenance)
    request = _Request(
        {
            "input": {
                "kind": "hermes.maintenance",
                "jobs": ["default"],
            }
        }
    )

    response = asyncio.run(agent_main.handle_invoke(request))

    assert response.status_code == 200
    assert captured["payload"] == {"kind": "hermes.maintenance", "jobs": ["default"]}


def test_top_level_kind_takes_precedence_over_input_wrapper(monkeypatch):
    agent_main = _load_agent_main(monkeypatch)
    captured = {}

    async def handle_maintenance(payload):
        captured["payload"] = payload
        return agent_main.JSONResponse({"ok": True})

    monkeypatch.setattr(agent_main, "_handle_maintenance", handle_maintenance)
    request = _Request(
        {
            "kind": "hermes.maintenance",
            "jobs": ["default"],
            "input": json.dumps({"kind": "hermes.maintenance", "jobs": ["all"]}),
        }
    )

    response = asyncio.run(agent_main.handle_invoke(request))

    assert response.status_code == 200
    assert captured["payload"]["jobs"] == ["default"]


def test_plain_text_input_wrapper_remains_unsupported(monkeypatch):
    agent_main = _load_agent_main(monkeypatch)
    response = asyncio.run(agent_main.handle_invoke(_Request({"input": "hello"})))

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "unsupported_payload"


def test_session_events_replays_pending_maintenance_summary_once(monkeypatch, tmp_path):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_CHILD_HOME", str(home))
    agent_main = _load_agent_main(monkeypatch)

    history_path = home / "foundry-maintenance" / "history.jsonl"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "kind": "hermes.maintenance.result",
                "run_id": "run-1",
                "status": "completed",
                "started_at": "2026-05-31T18:00:00Z",
                "ended_at": "2026-05-31T18:01:00Z",
                "duration_seconds": 60,
                "history_path": str(history_path),
                "jobs": [{"name": "refresh", "status": "success", "duration_seconds": 5}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    async def empty_subscribe(_session_id, _since_seq):
        if False:  # pragma: no cover - async generator marker
            yield {}

    monkeypatch.setattr(agent_main._broker, "subscribe", empty_subscribe)

    request = {
        "request": {
            "jsonrpc": "2.0",
            "id": "events-1",
            "method": "session.events",
            "params": {"session_id": "tui-session", "since_seq": -1},
        }
    }
    first = asyncio.run(_collect_sse(asyncio.run(agent_main._handle_rpc(request))))
    second = asyncio.run(_collect_sse(asyncio.run(agent_main._handle_rpc(request))))

    assert first[0]["result"]["status"] == "subscribed"
    assert first[1]["params"]["type"] == "maintenance.summary"
    assert first[1]["params"]["session_id"] == "tui-session"
    assert first[1]["params"]["payload"]["run_id"] == "run-1"
    assert first[1]["params"]["payload"]["delivery_key"] == "run:run-1"
    assert first[2]["type"] == "done"

    assert all(
        frame.get("params", {}).get("type") != "maintenance.summary" for frame in second
    )
    cursor = json.loads(
        (home / "foundry-maintenance" / "delivered-summary.json").read_text(encoding="utf-8")
    )
    assert cursor["delivery_key"] == "run:run-1"
