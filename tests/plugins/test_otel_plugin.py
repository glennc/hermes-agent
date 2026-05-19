"""Tests for the bundled observability/otel plugin."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "otel"


_OTEL_ENV = (
    "HERMES_OTEL_EXPORTER",
    "HERMES_OTEL_CONSOLE",
    "HERMES_OTEL_CAPTURE_CONTENT",
    "HERMES_OTEL_MAX_ATTR_CHARS",
    "HERMES_OTEL_DEBUG",
    "HERMES_OTEL_SERVICE_NAME",
    "HERMES_OTEL_SERVICE_NAMESPACE",
    "HERMES_OTEL_SERVICE_INSTANCE_ID",
    "HERMES_OTEL_RESOURCE_ATTRIBUTES",
    "HERMES_OTEL_EXPORTER_OTLP_ENDPOINT",
    "HERMES_OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "HERMES_OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_EXPORTER_OTLP_PROTOCOL",
    "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
    "OTEL_RESOURCE_ATTRIBUTES",
    "APPLICATIONINSIGHTS_CONNECTION_STRING",
    "FOUNDRY_HOSTING_ENVIRONMENT",
    "FOUNDRY_PROJECT_ENDPOINT",
    "FOUNDRY_PROJECT_ARM_ID",
    "FOUNDRY_AGENT_NAME",
    "FOUNDRY_AGENT_VERSION",
    "FOUNDRY_AGENT_SESSION_ID",
)


def _clear_env(monkeypatch):
    for key in _OTEL_ENV:
        monkeypatch.delenv(key, raising=False)


def _fresh_module():
    for name in list(sys.modules):
        if name == "plugins.observability.otel" or name.startswith("plugins.observability.otel."):
            sys.modules.pop(name, None)
    return importlib.import_module("plugins.observability.otel")


class TestManifest:
    def test_plugin_directory_exists(self):
        assert PLUGIN_DIR.is_dir()
        assert (PLUGIN_DIR / "plugin.yaml").exists()
        assert (PLUGIN_DIR / "__init__.py").exists()

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "otel"
        assert data["version"]
        assert set(data["hooks"]) == {
            "pre_api_request",
            "post_api_request",
            "pre_llm_call",
            "post_llm_call",
            "pre_tool_call",
            "post_tool_call",
            "on_session_finalize",
            "on_session_end",
            "on_session_reset",
        }


class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        from hermes_cli import plugins as plugins_mod

        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        loaded = manager._plugins.get("observability/otel")
        assert loaded is not None, "plugin not discovered"
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


class TestConfig:
    def test_auto_mode_is_disabled_without_export_target(self, monkeypatch):
        _clear_env(monkeypatch)
        from plugins.observability.otel.config import resolve_settings

        settings = resolve_settings()
        assert settings.enabled is False
        assert settings.exporter == "disabled"

    def test_otlp_endpoint_wins_over_foundry_appinsights(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://collector.example.com")
        monkeypatch.setenv(
            "APPLICATIONINSIGHTS_CONNECTION_STRING",
            "InstrumentationKey=secret;IngestionEndpoint=https://example.invalid",
        )
        monkeypatch.setenv("FOUNDRY_HOSTING_ENVIRONMENT", "hosted")
        monkeypatch.setenv("FOUNDRY_PROJECT_ARM_ID", "/subscriptions/s/resourceGroups/rg/providers/x")
        from plugins.observability.otel.config import resolve_settings

        settings = resolve_settings()
        assert settings.enabled is True
        assert settings.exporter == "otlp"
        assert settings.span_attributes["microsoft.foundry.detected"] is True
        assert settings.span_attributes["microsoft.foundry.project.id"].startswith("/subscriptions/")

    def test_auto_mode_uses_azure_monitor_when_only_appinsights_is_set(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv(
            "APPLICATIONINSIGHTS_CONNECTION_STRING",
            "InstrumentationKey=secret;IngestionEndpoint=https://example.invalid",
        )
        from plugins.observability.otel.config import resolve_settings

        settings = resolve_settings()
        assert settings.enabled is True
        assert settings.exporter == "azure_monitor"

    def test_foundry_attrs_do_not_include_internal_appinsights_routing(self, monkeypatch):
        _clear_env(monkeypatch)
        monkeypatch.setenv("FOUNDRY_AGENT_NAME", "HermesHosted")
        monkeypatch.setenv("FOUNDRY_AGENT_SESSION_ID", "session-123")
        monkeypatch.setenv("FOUNDRY_PROJECT_ARM_ID", "/subscriptions/s/resourceGroups/rg/providers/project")
        from plugins.observability.otel.foundry import detect_foundry_attributes

        attrs = detect_foundry_attributes()
        assert attrs["microsoft.foundry.agent.name"] == "HermesHosted"
        assert attrs["gen_ai.conversation.id"] == "session-123"
        assert "_msft_appinsights_connection" not in attrs


class TestLazyImports:
    def test_otlp_exporter_does_not_import_azure_monitor(self, monkeypatch):
        _clear_env(monkeypatch)
        for name in list(sys.modules):
            if name.startswith("azure.monitor"):
                sys.modules.pop(name, None)

        from plugins.observability.otel.exporters.otlp import build_span_processor

        settings = SimpleNamespace(
            otlp_protocol="http/protobuf",
            otlp_endpoint="http://localhost:4318",
            otlp_traces_endpoint="",
        )
        processor = build_span_processor(settings)
        try:
            assert not any(name.startswith("azure.monitor") for name in sys.modules)
        finally:
            processor.shutdown()


class FakeSpan:
    def __init__(self, name: str, attributes: dict, parent=None):
        self.name = name
        self.attributes = dict(attributes)
        self.parent = parent
        self.ended = False
        self.error = None

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def end(self):
        self.ended = True


class FakeBackend:
    def __init__(self, *, capture_content=False, span_attributes=None):
        from plugins.observability.otel.config import OTelSettings

        self.settings = OTelSettings(
            exporter="otlp",
            enabled=True,
            capture_content=capture_content,
            span_attributes=span_attributes or {},
        )
        self.spans: list[FakeSpan] = []
        self.flush_count = 0

    def start_span(self, name, *, attributes, parent=None):
        span = FakeSpan(name, attributes, parent)
        self.spans.append(span)
        return span

    def end_span(self, span, *, attributes=None, error=None):
        if attributes:
            span.attributes.update(attributes)
        span.error = error
        span.end()

    def flush(self):
        self.flush_count += 1


class TestRuntimeGate:
    def test_get_backend_caches_disabled_state(self, monkeypatch):
        _clear_env(monkeypatch)
        mod = _fresh_module()

        assert mod._get_backend() is None

        called = {"n": 0}
        real_get = os.environ.get

        def tracking_get(key, default=None):
            if key in _OTEL_ENV:
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)
        for _ in range(10):
            assert mod._get_backend() is None
        assert called["n"] == 0

    def test_hooks_noop_when_disabled(self, monkeypatch):
        _clear_env(monkeypatch)
        mod = _fresh_module()

        mod.on_pre_llm_request(task_id="t", session_id="s", request_messages=[])
        mod.on_post_llm_call(task_id="t", session_id="s")
        mod.on_pre_tool_call(tool_name="terminal", args={}, task_id="t", session_id="s")
        mod.on_post_tool_call(tool_name="terminal", result="ok", task_id="t", session_id="s")
        mod.on_session_finalize(session_id="s")


class TestHookSpans:
    def test_llm_turn_span_lifecycle_excludes_content_by_default(self, monkeypatch):
        _clear_env(monkeypatch)
        mod = _fresh_module()
        fake = FakeBackend(
            span_attributes={
                "microsoft.foundry.detected": True,
                "microsoft.foundry.agent.name": "HermesHosted",
            }
        )
        monkeypatch.setattr(mod, "_get_backend", lambda: fake)

        mod.on_pre_llm_request(
            task_id="task-1",
            session_id="session-1",
            platform="cli",
            provider="openai",
            model="gpt-test",
            api_mode="chat_completions",
            api_call_count=1,
            request_messages=[{"role": "user", "content": "secret prompt"}],
            message_count=2,
            tool_count=3,
            approx_input_tokens=42,
        )
        mod.on_post_llm_call(
            task_id="task-1",
            session_id="session-1",
            platform="cli",
            provider="openai",
            model="gpt-test",
            api_mode="chat_completions",
            api_call_count=1,
            assistant_response="final answer",
            assistant_content_chars=12,
            usage={"input_tokens": 10, "output_tokens": 5},
        )

        names = [span.name for span in fake.spans]
        assert names == ["Hermes turn", "LLM call 1"]
        assert all(span.ended for span in fake.spans)
        assert fake.flush_count == 1

        root, llm = fake.spans
        assert root.attributes["microsoft.foundry.agent.name"] == "HermesHosted"
        assert llm.attributes["gen_ai.usage.input_tokens"] == 10
        assert llm.attributes["gen_ai.usage.output_tokens"] == 5
        assert not any(key.endswith(".content") for span in fake.spans for key in span.attributes)

    def test_tool_span_records_duration_and_error(self, monkeypatch):
        _clear_env(monkeypatch)
        mod = _fresh_module()
        fake = FakeBackend()
        monkeypatch.setattr(mod, "_get_backend", lambda: fake)

        mod.on_pre_llm_request(
            task_id="task-1",
            session_id="session-1",
            provider="openai",
            model="gpt-test",
            api_call_count=1,
            request_messages=[],
        )
        mod.on_pre_tool_call(
            tool_name="terminal",
            args={"command": "false"},
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
        )
        mod.on_post_tool_call(
            tool_name="terminal",
            result='{"error": "failed"}',
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
            duration_ms=123,
        )

        tool_span = next(span for span in fake.spans if span.name == "Tool: terminal")
        assert tool_span.ended is True
        assert tool_span.error == "tool returned error"
        assert tool_span.attributes["hermes.tool.duration_ms"] == 123
        assert tool_span.parent is fake.spans[0]

    def test_session_finalize_ends_pending_spans(self, monkeypatch):
        _clear_env(monkeypatch)
        mod = _fresh_module()
        fake = FakeBackend()
        monkeypatch.setattr(mod, "_get_backend", lambda: fake)

        mod.on_pre_llm_request(
            task_id="task-1",
            session_id="session-1",
            provider="openai",
            model="gpt-test",
            api_call_count=1,
            request_messages=[],
        )
        mod.on_session_finalize(session_id="session-1")

        assert all(span.ended for span in fake.spans)
        assert all(span.error == "session finalized before trace completed" for span in fake.spans)
        assert fake.flush_count == 1

    def test_legacy_llm_hooks_create_and_finish_llm_span(self, monkeypatch):
        _clear_env(monkeypatch)
        mod = _fresh_module()
        fake = FakeBackend()
        monkeypatch.setattr(mod, "_get_backend", lambda: fake)

        mod.on_pre_llm_call(
            task_id="task-legacy",
            session_id="session-legacy",
            provider="openai",
            model="gpt-test",
            messages=[{"role": "user", "content": "hi"}],
        )
        mod.on_post_llm_call(
            task_id="task-legacy",
            session_id="session-legacy",
            provider="openai",
            model="gpt-test",
            assistant_response="hello",
        )

        assert [span.name for span in fake.spans] == ["Hermes turn", "LLM call 0"]
        assert all(span.ended for span in fake.spans)
