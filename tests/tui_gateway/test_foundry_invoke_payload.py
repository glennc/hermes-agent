import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path


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

    repo_root = Path(__file__).resolve().parents[4]
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

    async def body(self):
        return self._body


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
