"""OpenTelemetry observability plugin for Hermes.

The plugin is generic by default: setting ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
enough to emit Hermes turn, LLM, and tool spans to any OTLP-compatible backend.
When running inside Microsoft Foundry hosted agents, the plugin also detects
Foundry's public environment variables and can export directly to Azure Monitor
Application Insights via ``APPLICATIONINSIGHTS_CONNECTION_STRING``.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import OTelSettings, resolve_settings

logger = logging.getLogger(__name__)


@dataclass
class TraceState:
    task_key: str
    session_id: str
    root_span: Any
    llm_spans: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    pending_tools_by_name: dict[str, list[Any]] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)


_STATE_LOCK = threading.Lock()
_TRACE_STATE: dict[str, TraceState] = {}
_BACKEND: Any = None
_SETTINGS: Optional[OTelSettings] = None
_INIT_FAILED = object()

_SECRET_PATTERNS = (
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE), "Bearer <redacted>"),
    (re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{8,}\b"), "sk-<redacted>"),
    (re.compile(r"\bpk-[A-Za-z0-9][A-Za-z0-9_-]{8,}\b"), "pk-<redacted>"),
    (re.compile(r"\b[A-Z0-9]{20}AWS[A-Z0-9]{16}\b"), "<redacted-aws-key>"),
    (re.compile(r"(api[_-]?key['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE), r"\1<redacted>"),
    (re.compile(r"(token['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE), r"\1<redacted>"),
)


def _debug(message: str) -> None:
    settings = _SETTINGS
    if settings and settings.debug:
        logger.info("OpenTelemetry tracing: %s", message)


def _get_backend():
    global _BACKEND, _SETTINGS
    if _BACKEND is _INIT_FAILED:
        return None
    if _BACKEND is not None:
        return _BACKEND

    settings = resolve_settings()
    _SETTINGS = settings
    if not settings.enabled:
        _BACKEND = _INIT_FAILED
        _debug(f"disabled: {settings.reason}")
        return None

    try:
        from .runtime import build_backend

        _BACKEND = build_backend(settings)
        _debug(f"initialized exporter {settings.exporter}")
        return _BACKEND
    except Exception as exc:  # pragma: no cover - fail-open
        logger.warning("OpenTelemetry plugin disabled: %s", exc)
        _BACKEND = _INIT_FAILED
        return None


def _trace_key(task_id: str, session_id: str) -> str:
    if task_id:
        return task_id
    if session_id:
        return f"session:{session_id}"
    return f"thread:{threading.get_ident()}"


def _request_key(api_call_count: Any) -> str:
    return str(api_call_count or 0)


def _redact_text(value: str) -> str:
    redacted = value
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"... [truncated {len(value) - max_chars} chars]"


def _safe_json(value: Any, *, max_chars: int, depth: int = 0) -> Any:
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _truncate(_redact_text(value), max_chars)
    if isinstance(value, bytes):
        return {"type": "bytes", "len": len(value)}
    if isinstance(value, dict):
        return {
            str(k): _safe_json(v, max_chars=max_chars, depth=depth + 1)
            for k, v in list(value.items())[:50]
        }
    if isinstance(value, (list, tuple, set)):
        return [
            _safe_json(v, max_chars=max_chars, depth=depth + 1)
            for v in list(value)[:50]
        ]
    if hasattr(value, "__dict__"):
        return _safe_json(vars(value), max_chars=max_chars, depth=depth + 1)
    return _truncate(_redact_text(repr(value)), max_chars)


def _content_attr(value: Any, settings: OTelSettings) -> str:
    safe = _safe_json(value, max_chars=settings.max_attr_chars)
    try:
        text = json.dumps(safe, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(safe)
    return _truncate(_redact_text(text), settings.max_attr_chars)


def _maybe_content_attrs(prefix: str, value: Any, settings: OTelSettings) -> dict[str, Any]:
    if not settings.capture_content:
        return {}
    return {f"{prefix}.content": _content_attr(value, settings)}


def _base_attrs(backend: Any, *, task_id: str = "", session_id: str = "", platform: str = "") -> dict[str, Any]:
    settings = backend.settings
    attrs: dict[str, Any] = {
        "hermes.session.id": session_id,
        "hermes.task.id": task_id,
        "hermes.platform": platform,
        "hermes.observability.exporter": settings.exporter,
    }
    attrs.update(settings.span_attributes)
    return attrs


def _start_root_trace(
    task_key: str,
    *,
    backend: Any,
    task_id: str,
    session_id: str,
    platform: str,
    provider: str,
    model: str,
    api_mode: str,
    messages: Any,
) -> TraceState:
    attrs = _base_attrs(backend, task_id=task_id, session_id=session_id, platform=platform)
    attrs.update(
        {
            "hermes.trace.type": "turn",
            "gen_ai.operation.name": "agent",
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "hermes.api_mode": api_mode,
        }
    )
    attrs.update(_maybe_content_attrs("hermes.request", _last_user_message(messages), backend.settings))
    root_span = backend.start_span("Hermes turn", attributes=attrs)
    return TraceState(task_key=task_key, session_id=session_id, root_span=root_span)


def _last_user_message(messages: Any) -> Any:
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return {"role": "user", "content": message.get("content")}
    return None


def _coerce_request_messages(
    *,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
) -> list[dict[str, Any]]:
    for candidate in (request_messages, messages, conversation_history):
        if isinstance(candidate, list):
            return candidate
    if user_message is None:
        return []
    return [{"role": "user", "content": user_message}]


def _get_or_start_state(
    *,
    backend: Any,
    task_id: str,
    session_id: str,
    platform: str,
    provider: str,
    model: str,
    api_mode: str,
    messages: Any,
) -> TraceState:
    task_key = _trace_key(task_id, session_id)
    state = _TRACE_STATE.get(task_key)
    if state is None:
        state = _start_root_trace(
            task_key,
            backend=backend,
            task_id=task_id,
            session_id=session_id,
            platform=platform,
            provider=provider,
            model=model,
            api_mode=api_mode,
            messages=messages,
        )
        _TRACE_STATE[task_key] = state
    state.last_updated_at = time.time()
    return state


def _usage_attrs(usage: Any) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    mapping = {
        "input_tokens": "gen_ai.usage.input_tokens",
        "prompt_tokens": "gen_ai.usage.input_tokens",
        "output_tokens": "gen_ai.usage.output_tokens",
        "completion_tokens": "gen_ai.usage.output_tokens",
        "cache_read_tokens": "hermes.usage.cache_read_tokens",
        "cache_write_tokens": "hermes.usage.cache_write_tokens",
        "reasoning_tokens": "hermes.usage.reasoning_tokens",
    }
    attrs: dict[str, Any] = {}
    for source_key, attr_key in mapping.items():
        value = usage.get(source_key)
        if isinstance(value, (int, float)) and value:
            attrs[attr_key] = value
    return attrs


def _assistant_has_tool_calls(message: Any, assistant_tool_call_count: int) -> bool:
    return bool(getattr(message, "tool_calls", None)) or assistant_tool_call_count > 0


def _assistant_output_value(assistant_message: Any, assistant_response: Any) -> Any:
    if assistant_message is not None:
        return {
            "content": getattr(assistant_message, "content", None),
            "reasoning": getattr(assistant_message, "reasoning", None),
            "tool_calls": getattr(assistant_message, "tool_calls", None),
        }
    if assistant_response is not None:
        return {"content": assistant_response}
    return None


def _result_has_error(result: Any) -> bool:
    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                result = json.loads(stripped)
            except Exception:
                return False
    return isinstance(result, dict) and bool(result.get("error"))


def _finish_trace(task_key: str, *, error: str | None = None, final_attrs: dict[str, Any] | None = None) -> None:
    backend = _get_backend()
    if backend is None:
        return
    with _STATE_LOCK:
        state = _TRACE_STATE.pop(task_key, None)
    if state is None:
        return

    for span in list(state.llm_spans.values()):
        backend.end_span(span, error=error)
    for span in list(state.tools.values()):
        backend.end_span(span, error=error)
    for queue in state.pending_tools_by_name.values():
        for span in queue:
            backend.end_span(span, error=error)
    backend.end_span(state.root_span, attributes=final_attrs or {}, error=error)
    backend.flush()


def on_pre_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    api_mode: str = "",
    messages: Any = None,
    **_: Any,
) -> None:
    if not isinstance(messages, list):
        return
    backend = _get_backend()
    if backend is None:
        return
    with _STATE_LOCK:
        state = _get_or_start_state(
            backend=backend,
            task_id=task_id,
            session_id=session_id,
            platform=platform,
            provider=provider,
            model=model,
            api_mode=api_mode,
            messages=messages,
        )
        req_key = _request_key(0)
        if req_key not in state.llm_spans:
            attrs = _base_attrs(backend, task_id=task_id, session_id=session_id, platform=platform)
            attrs.update(
                {
                    "hermes.trace.type": "llm",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": provider,
                    "gen_ai.request.model": model,
                    "hermes.api_mode": api_mode,
                    "hermes.api.call_index": 0,
                }
            )
            attrs.update(_maybe_content_attrs("hermes.llm.request", messages[-12:], backend.settings))
            state.llm_spans[req_key] = backend.start_span(
                "LLM call 0",
                attributes=attrs,
                parent=state.root_span,
            )


def on_pre_llm_request(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    model: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    api_call_count: int = 0,
    request_messages: Any = None,
    messages: Any = None,
    conversation_history: Any = None,
    user_message: Any = None,
    message_count: int = 0,
    tool_count: int = 0,
    approx_input_tokens: int = 0,
    request_char_count: int = 0,
    max_tokens: Any = None,
    **_: Any,
) -> None:
    backend = _get_backend()
    if backend is None:
        return
    input_messages = _coerce_request_messages(
        request_messages=request_messages,
        messages=messages,
        conversation_history=conversation_history,
        user_message=user_message,
    )
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _get_or_start_state(
            backend=backend,
            task_id=task_id,
            session_id=session_id,
            platform=platform,
            provider=provider,
            model=model,
            api_mode=api_mode,
            messages=input_messages,
        )
        previous = state.llm_spans.pop(req_key, None)
        if previous is not None:
            backend.end_span(previous)
        attrs = _base_attrs(backend, task_id=task_id, session_id=session_id, platform=platform)
        attrs.update(
            {
                "hermes.trace.type": "llm",
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": model,
                "hermes.api_mode": api_mode,
                "hermes.api.call_index": api_call_count,
                "hermes.api.base_url": base_url,
                "hermes.message_count": message_count,
                "hermes.tool_count": tool_count,
                "hermes.approx_input_tokens": approx_input_tokens,
                "hermes.request_char_count": request_char_count,
            }
        )
        if max_tokens is not None:
            attrs["gen_ai.request.max_tokens"] = max_tokens
        attrs.update(_maybe_content_attrs("hermes.llm.request", input_messages[-12:], backend.settings))
        state.llm_spans[req_key] = backend.start_span(
            f"LLM call {api_call_count}",
            attributes=attrs,
            parent=state.root_span,
        )


def on_post_llm_call(
    *,
    task_id: str = "",
    session_id: str = "",
    platform: str = "",
    provider: str = "",
    base_url: str = "",
    api_mode: str = "",
    model: str = "",
    api_call_count: int = 0,
    assistant_message: Any = None,
    response: Any = None,
    api_duration: float = 0.0,
    finish_reason: str = "",
    usage: Any = None,
    assistant_content_chars: int = 0,
    assistant_tool_call_count: int = 0,
    assistant_response: Any = None,
    response_model: Any = None,
    **_: Any,
) -> None:
    backend = _get_backend()
    if backend is None:
        return
    task_key = _trace_key(task_id, session_id)
    req_key = _request_key(api_call_count)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        llm_span = state.llm_spans.pop(req_key, None) if state else None
    if state is None or llm_span is None:
        return

    output_value = _assistant_output_value(assistant_message, assistant_response)
    attrs = _base_attrs(backend, task_id=task_id, session_id=session_id, platform=platform)
    attrs.update(
        {
            "gen_ai.provider.name": provider,
            "gen_ai.request.model": model,
            "hermes.api_mode": api_mode,
            "hermes.api.base_url": base_url,
            "hermes.assistant.content_chars": assistant_content_chars,
            "hermes.assistant.tool_call_count": assistant_tool_call_count,
        }
    )
    if api_duration and api_duration > 0:
        attrs["hermes.api.duration_ms"] = int(api_duration * 1000)
    if finish_reason:
        attrs["gen_ai.response.finish_reason"] = finish_reason
    response_id = getattr(response, "id", None)
    if response_id:
        attrs["gen_ai.response.id"] = response_id
    if response_model:
        attrs["gen_ai.response.model"] = response_model
    attrs.update(_usage_attrs(usage))
    attrs.update(_maybe_content_attrs("hermes.llm.response", output_value, backend.settings))
    backend.end_span(llm_span, attributes=attrs)

    has_tools = _assistant_has_tool_calls(assistant_message, assistant_tool_call_count)
    has_content = bool(
        (getattr(assistant_message, "content", None) if assistant_message is not None else None)
        or assistant_response
        or assistant_content_chars
    )
    if not has_tools and has_content:
        final_attrs = {
            "hermes.turn.duration_ms": int((time.time() - state.started_at) * 1000),
        }
        if output_value is not None:
            final_attrs.update(_maybe_content_attrs("hermes.response", output_value, backend.settings))
        _finish_trace(task_key, final_attrs=final_attrs)


def on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    **_: Any,
) -> None:
    backend = _get_backend()
    if backend is None:
        return
    task_key = _trace_key(task_id, session_id)

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        attrs = _base_attrs(backend, task_id=task_id, session_id=session_id)
        attrs.update(
            {
                "hermes.trace.type": "tool",
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": tool_name,
                "hermes.tool.name": tool_name,
                "hermes.tool.call.id": tool_call_id,
            }
        )
        attrs.update(_maybe_content_attrs("hermes.tool.args", args, backend.settings))
        span = backend.start_span(f"Tool: {tool_name}", attributes=attrs, parent=state.root_span)
        if tool_call_id:
            state.tools[tool_call_id] = span
        else:
            state.pending_tools_by_name.setdefault(tool_name, []).append(span)


def on_post_tool_call(
    *,
    tool_name: str = "",
    args: Any = None,
    result: Any = None,
    task_id: str = "",
    session_id: str = "",
    tool_call_id: str = "",
    duration_ms: int = 0,
    **_: Any,
) -> None:
    backend = _get_backend()
    if backend is None:
        return
    task_key = _trace_key(task_id, session_id)
    span = None

    with _STATE_LOCK:
        state = _TRACE_STATE.get(task_key)
        if state is None:
            return
        if tool_call_id:
            span = state.tools.pop(tool_call_id, None)
        if span is None:
            queue = state.pending_tools_by_name.get(tool_name)
            if queue:
                span = queue.pop(0)
                if not queue:
                    state.pending_tools_by_name.pop(tool_name, None)

    if span is None:
        return
    attrs = {
        "gen_ai.tool.name": tool_name,
        "hermes.tool.name": tool_name,
        "hermes.tool.call.id": tool_call_id,
        "hermes.tool.duration_ms": duration_ms,
    }
    attrs.update(_maybe_content_attrs("hermes.tool.result", result, backend.settings))
    error = "tool returned error" if _result_has_error(result) else None
    backend.end_span(span, attributes=attrs, error=error)


def on_session_finalize(*, session_id: str | None = None, **_: Any) -> None:
    backend = _get_backend()
    if backend is None:
        return
    with _STATE_LOCK:
        keys = [
            key
            for key, state in _TRACE_STATE.items()
            if not session_id or state.session_id == session_id or key == f"session:{session_id}"
        ]
    for key in keys:
        _finish_trace(key, error="session finalized before trace completed")


def register(ctx) -> None:
    ctx.register_hook("pre_api_request", on_pre_llm_request)
    ctx.register_hook("post_api_request", on_post_llm_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_hook("on_session_end", on_session_finalize)
    ctx.register_hook("on_session_reset", on_session_finalize)
