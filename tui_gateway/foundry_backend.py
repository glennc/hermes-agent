from __future__ import annotations

import atexit
import base64
import concurrent.futures
import contextvars
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from agent.azure_identity_adapter import SCOPE_AI_AZURE_DEFAULT
from hermes_constants import get_hermes_home
from tui_gateway.transport import (
    StdioTransport,
    Transport,
    bind_transport,
    current_transport,
    reset_transport,
)

_hermes_home = Path(get_hermes_home())
_CRASH_LOG = str(_hermes_home / "logs" / "tui_gateway_crash.log")

_sessions: dict[str, dict[str, Any]] = {}
_methods: dict[str, Any] = {}
_pending_controls: dict[str, str] = {}
_stdout_lock = threading.Lock()

# Lifetime cache for commands.catalog and the data behind the local-prefix
# branch of complete.slash. The cached value is the full result dict the
# upstream gateway returns (`pairs`, `categories`, `canon`, `sub`,
# `skill_count`, `warning`); rid is added at response time. No TTL —
# the cache is cleared explicitly when an RPC that can change the slash
# registry succeeds (skills.reload, skills.manage, reload.mcp,
# reload.env) and on replay.gap events from the hosted subscriber
# (which indicates the hosted agent may have restarted).
_catalog_cache: dict[str, Any] | None = None
_catalog_lock = threading.Lock()

# Hard-coded TUI extras that upstream's complete.slash injects on top
# of SlashCommandCompleter output. Mirrors the `extras` list in
# server.py so local autocomplete is byte-for-byte equivalent for the
# top-level prefix case.
_COMPLETE_SLASH_EXTRAS: tuple[dict[str, str], ...] = (
    {"text": "/compact", "display": "/compact", "meta": "Toggle compact display mode"},
    {"text": "/details", "display": "/details", "meta": "Control agent detail visibility"},
    {"text": "/logs", "display": "/logs", "meta": "Show recent gateway log lines"},
    {"text": "/mouse", "display": "/mouse", "meta": "Toggle mouse/wheel tracking [on|off|toggle]"},
)
_COMPLETE_SLASH_LIMIT = 30

_real_stdout = sys.stdout
sys.stdout = sys.stderr
_stdio_transport = StdioTransport(lambda: _real_stdout, _stdout_lock)

_DEFAULT_ENDPOINT = "http://127.0.0.1:8088"
_DEFAULT_AGENT_NAME = "hermes-foundry-agent"
_DEFAULT_API_VERSION = "v1"
_REQUEST_TIMEOUT_S = 120.0
_CONTROL_TIMEOUT_S = 15.0

# Async dispatch — mirrors the local server.py pattern. Slash workers, shell
# execs, session lifecycle, and skill ops can each block for many seconds
# (slash worker cold-start on the hosted side, network round-trips to
# Foundry, model calls). Running them inline on the stdio dispatcher loop
# blocks every other RPC from the TUI — including autocomplete that the
# composer fires while the user is still typing. Route only the known-slow
# methods onto a small thread pool; everything else stays on the main
# thread so response ordering is preserved for the fast path.
# write_json is _stdout_lock-guarded so concurrent response writes are safe.
_LONG_HANDLERS = frozenset(
    {
        "browser.manage",
        "cli.exec",
        "command.dispatch",
        "session.branch",
        "session.compress",
        "session.resume",
        "shell.exec",
        "skills.manage",
        "slash.exec",
    }
)

try:
    _rpc_pool_workers = max(
        2, int(os.environ.get("HERMES_TUI_RPC_POOL_WORKERS") or "4")
    )
except (ValueError, TypeError):
    _rpc_pool_workers = 4
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=_rpc_pool_workers,
    thread_name_prefix="foundry-rpc",
)
atexit.register(lambda: _pool.shutdown(wait=False, cancel_futures=True))

_config: dict[str, Any] = {
    "display": {
        "busy_input_mode": "interrupt",
        "details_mode": "collapsed",
        "inline_diffs": True,
        "mouse_tracking": True,
        "show_cost": False,
        "show_reasoning": False,
        "streaming": True,
        "tui_auto_resume_recent": False,
        "tui_compact": False,
        "tui_status_indicator": "unicode",
        "tui_statusbar": "top",
    },
    "voice": {"record_key": "ctrl+b"},
}


def write_json(obj: dict) -> bool:
    if obj.get("method") == "event":
        sid = ((obj.get("params") or {}).get("session_id")) or ""
        if sid and (transport := (_sessions.get(sid) or {}).get("transport")) is not None:
            return transport.write(obj)

    return (current_transport() or _stdio_transport).write(obj)


def _emit(event: str, sid: str, payload: dict | None = None) -> None:
    params: dict[str, Any] = {"type": event, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    write_json({"jsonrpc": "2.0", "method": "event", "params": params})


def _ok(rid: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid: Any, code: int, msg: str) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}


def method(name: str):
    def dec(fn):
        _methods[name] = fn
        return fn

    return dec


def _normalize_request(req: Any) -> tuple[Any, str, dict] | dict:
    if not isinstance(req, dict):
        return _err(None, -32600, "invalid request: expected an object")

    rid = req.get("id")
    method_name = req.get("method")
    if not isinstance(method_name, str) or not method_name:
        return _err(rid, -32600, "invalid request: method must be a non-empty string")

    params = req.get("params", {})
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        return _err(rid, -32602, "invalid params: expected an object")

    return rid, method_name, params


def handle_request(req: dict) -> dict | None:
    normalized = _normalize_request(req)
    if isinstance(normalized, dict):
        return normalized

    rid, method_name, params = normalized
    fn = _methods.get(method_name)
    if fn:
        return fn(rid, params)
    return _proxy_rpc(method_name, params, rid)


def dispatch(req: dict, transport: Optional[Transport] = None) -> dict | None:
    t = transport or _stdio_transport
    token = bind_transport(t)
    rid = req.get("id") if isinstance(req, dict) else None
    method_name = req.get("method") if isinstance(req, dict) else None
    try:
        if method_name not in _LONG_HANDLERS:
            try:
                return handle_request(req)
            except Exception as exc:
                print(
                    f"[foundry-backend] dispatch crashed for method={method_name!r}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                return _err(rid, 5000, f"foundry proxy error: {type(exc).__name__}: {exc}")

        ctx = contextvars.copy_context()

        def run() -> None:
            try:
                resp = handle_request(req)
            except Exception as exc:
                print(
                    f"[foundry-backend] async handler crashed for method={method_name!r}: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exc(file=sys.stderr)
                resp = _err(rid, 5000, f"foundry proxy error: {type(exc).__name__}: {exc}")
            if resp is not None:
                t.write(resp)

        _pool.submit(lambda: ctx.run(run))
        return None
    finally:
        reset_transport(token)


def _endpoint() -> str:
    return (
        os.environ.get("HERMES_FOUNDRY_ENDPOINT")
        or os.environ.get("HERMES_FOUNDRY_PROJECT_ENDPOINT")
        or os.environ.get("HERMES_FOUNDRY_LOCAL_ENDPOINT")
        or os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
        or _DEFAULT_ENDPOINT
    ).rstrip("/")


def _agent_name() -> str:
    return (os.environ.get("HERMES_FOUNDRY_AGENT_NAME") or _DEFAULT_AGENT_NAME).strip()


def _api_version() -> str:
    return (os.environ.get("HERMES_FOUNDRY_API_VERSION") or _DEFAULT_API_VERSION).strip()


_FOUNDRY_TOKEN_SCOPE = SCOPE_AI_AZURE_DEFAULT
_AZ_LOGIN_HINT = (
    "Run 'az login' or configure a service principal "
    "(DefaultAzureCredential) with access to the Foundry project. "
    "Set HERMES_FOUNDRY_BEARER_TOKEN to override for tests/CI."
)
# Credential lifecycle mirrors agent.azure_identity_adapter: instantiate
# DefaultAzureCredential once per process (the SDK README is explicit:
# "creating multiple instances may create unnecessary resource usage"
# and breaks token caching). get_bearer_token_provider wraps the credential
# with BearerTokenCredentialPolicy, which handles in-process caching and
# automatic refresh on expiry — without it every RPC would shell out to
# `az` and block the proxy's stdio loop.
_credential_lock = threading.Lock()
_credential: Any = None
_token_provider: Any = None
_credential_atexit_registered = False
_cached_oid: Optional[str] = None


def _shutdown_credential() -> None:
    global _credential, _token_provider
    cred = _credential
    _credential = None
    _token_provider = None
    if cred is None:
        return
    try:
        cred.close()
    except Exception:
        pass


def _ensure_token_provider() -> Any:
    global _credential, _token_provider, _credential_atexit_registered

    if _token_provider is not None:
        return _token_provider

    with _credential_lock:
        if _token_provider is not None:
            return _token_provider

        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as exc:
            raise RuntimeError(
                "Foundry token acquisition requires the 'azure-identity' package. "
                "Install it in the Hermes Python environment, or set "
                "HERMES_FOUNDRY_BEARER_TOKEN to a pre-acquired bearer token."
            ) from exc

        _credential = DefaultAzureCredential()
        _token_provider = get_bearer_token_provider(_credential, _FOUNDRY_TOKEN_SCOPE)
        if not _credential_atexit_registered:
            atexit.register(_shutdown_credential)
            _credential_atexit_registered = True
        return _token_provider


def _acquire_token() -> str:
    explicit = (os.environ.get("HERMES_FOUNDRY_BEARER_TOKEN") or "").strip()
    if explicit:
        return explicit

    provider = _ensure_token_provider()
    try:
        token = provider()
    except Exception as exc:
        raise RuntimeError(
            f"Foundry token acquisition failed: {exc}. {_AZ_LOGIN_HINT}"
        ) from exc

    if not isinstance(token, str) or not token.strip():
        raise RuntimeError(
            f"Foundry token acquisition returned an empty token. {_AZ_LOGIN_HINT}"
        )
    return token


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        raise RuntimeError(
            "Foundry bearer token is not a JWT; cannot derive the user identity. "
            f"{_AZ_LOGIN_HINT}"
        )

    payload_b64 = parts[1]
    padding = "=" * (-len(payload_b64) % 4)
    try:
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(payload_bytes)
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode the Foundry bearer token payload: {exc}. {_AZ_LOGIN_HINT}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            "Foundry bearer token payload was not a JSON object; cannot derive the user identity. "
            f"{_AZ_LOGIN_HINT}"
        )
    return payload


def _entra_oid() -> str:
    global _cached_oid

    if _cached_oid:
        return _cached_oid

    payload = _decode_jwt_payload(_acquire_token())
    oid = payload.get("oid") or payload.get("sub")
    if not isinstance(oid, str) or not oid.strip():
        raise RuntimeError(
            "Foundry bearer token has no 'oid' or 'sub' claim; cannot derive the workspace key. "
            f"{_AZ_LOGIN_HINT}"
        )

    _cached_oid = oid.strip()
    return _cached_oid


def _workspace_key() -> str:
    explicit = (os.environ.get("HERMES_FOUNDRY_WORKSPACE_KEY") or "").strip()
    if explicit:
        return explicit

    digest = hashlib.sha256(_entra_oid().encode("utf-8")).hexdigest()[:16]
    return f"tui-{digest}"


def _invocations_url(session: dict[str, Any], invocation_id: str | None = None, *, cancel: bool = False) -> str:
    format_args = {
        "agent": quote(_agent_name(), safe=""),
        "api_version": quote(_api_version(), safe=""),
        "workspace": quote(str(session.get("workspace") or _workspace_key()), safe=""),
    }
    template = (os.environ.get("HERMES_FOUNDRY_INVOCATIONS_URL") or "").strip()
    if template:
        return template.format(**format_args)

    path = (os.environ.get("HERMES_FOUNDRY_INVOCATIONS_PATH") or "").strip()
    if path:
        if not path.startswith("/"):
            path = f"/{path}"
        path = path.format(**format_args)
    else:
        agent = format_args["agent"]
        path = f"/agents/{agent}/endpoint/protocols/invocations"

        if invocation_id:
            path += f"/{quote(invocation_id, safe='')}"
        if cancel:
            path += "/cancel"

        query = urlencode(
            {
                "agent_session_id": session.get("workspace") or _workspace_key(),
                "api-version": _api_version(),
            }
        )
        return f"{_endpoint()}{path}?{query}"

    if invocation_id:
        path += f"/{quote(invocation_id, safe='')}"
    if cancel:
        path += "/cancel"
    return f"{_endpoint()}{path}"


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "text/event-stream, application/json",
        "Content-Type": "application/json",
    }
    token = _acquire_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _timeout_seconds() -> float:
    raw = (os.environ.get("HERMES_FOUNDRY_TIMEOUT_S") or "").strip()
    if not raw:
        return _REQUEST_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError:
        return _REQUEST_TIMEOUT_S
    return value if value > 0 else _REQUEST_TIMEOUT_S


def _read_http_error(exc: HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return f"Foundry invocation failed: HTTP {exc.code}{f' - {body}' if body else ''}"


def _session_payload(sid: str, session: dict[str, Any]) -> dict[str, str]:
    return {
        "id": sid,
        "workspace": str(session.get("workspace") or _workspace_key()),
    }


def _rpc_request(method_name: str, params: dict[str, Any], rid: Any | None = None) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": rid if rid is not None else uuid.uuid4().hex,
        "method": method_name,
        "params": params,
    }


def _rpc_payload(
    sid: str,
    session: dict[str, Any],
    request: dict[str, Any],
    invocation_id: str | None = None,
) -> dict[str, Any]:
    return {
        "kind": "hermes.rpc",
        "invocation_id": invocation_id or uuid.uuid4().hex,
        "request": request,
        "session": _session_payload(sid, session),
        "tui": {"protocol_version": 1},
    }


def _decode_event(data: str) -> dict[str, Any] | None:
    data = data.strip()
    if not data or data == "[DONE]":
        return None

    try:
        event = json.loads(data)
    except json.JSONDecodeError:
        return {"type": "message.delta", "payload": {"text": data}}

    if not isinstance(event, dict):
        return {"type": "message.delta", "payload": {"text": str(event)}}
    return event


def _iter_sse_events(response, cancel_event: threading.Event):
    data_lines: list[str] = []

    for raw in response:
        if cancel_event.is_set():
            return
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                event = _decode_event("\n".join(data_lines))
                data_lines = []
                if event is not None:
                    yield event
            continue
        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines and not cancel_event.is_set():
        event = _decode_event("\n".join(data_lines))
        if event is not None:
            yield event


def _events_from_json(value: Any):
    if isinstance(value, dict):
        if value.get("jsonrpc") == "2.0":
            yield value
            return
        events = value.get("events")
        if isinstance(events, list):
            for item in events:
                if isinstance(item, dict):
                    yield item
            return
        if isinstance(value.get("type"), str):
            yield value
            return
        text = value.get("text") or value.get("message") or value.get("output")
        if text:
            yield {"type": "message.start", "payload": {}}
            yield {"type": "message.complete", "payload": {"text": str(text), "status": "complete"}}
            return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                yield item
        return
    if value is not None:
        yield {"type": "message.start", "payload": {}}
        yield {"type": "message.complete", "payload": {"text": str(value), "status": "complete"}}


def _read_response_events(response, cancel_event: threading.Event):
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        yield from _iter_sse_events(response, cancel_event)
        return

    body = response.read().decode("utf-8", errors="replace").strip()
    if not body:
        return

    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        value = body
    yield from _events_from_json(value)


def _post_invocation_events(session: dict[str, Any], payload: dict[str, Any], cancel_event: threading.Event):
    if cancel_event.is_set():
        return

    request = Request(
        _invocations_url(session),
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )

    try:
        with urlopen(request, timeout=_timeout_seconds()) as response:
            with session["lock"]:
                session["active_response"] = response
            try:
                yield from _read_response_events(response, cancel_event)
            finally:
                with session["lock"]:
                    if session.get("active_response") is response:
                        session["active_response"] = None
    except HTTPError as exc:
        raise RuntimeError(_read_http_error(exc)) from exc
    except URLError as exc:
        raise RuntimeError(f"Foundry invocation failed: {exc.reason}") from exc


def _post_rpc(session: dict[str, Any], request_body: dict[str, Any]) -> dict[str, Any]:
    sid = str(session.get("sid") or "")
    payload = _rpc_payload(sid, session, request_body)
    request = Request(
        _invocations_url(session),
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )

    try:
        with urlopen(request, timeout=min(_timeout_seconds(), _CONTROL_TIMEOUT_S)) as response:
            body = response.read().decode("utf-8", errors="replace").strip()
    except HTTPError as exc:
        raise RuntimeError(_read_http_error(exc)) from exc
    except URLError as exc:
        raise RuntimeError(f"Foundry RPC failed: {exc.reason}") from exc

    if not body:
        return _err(request_body.get("id"), 5000, "Foundry RPC returned an empty response.")
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return _err(request_body.get("id"), 5000, body)
    if not isinstance(value, dict):
        return _err(request_body.get("id"), 5000, f"Foundry RPC returned non-object response: {value!r}")
    return value


def _call_session_rpc(
    session: dict[str, Any],
    method_name: str,
    params: dict[str, Any],
    rid: Any | None = None,
) -> dict[str, Any]:
    rpc_params = dict(params)
    rpc_params.setdefault("session_id", session.get("sid") or "")
    return _post_rpc(session, _rpc_request(method_name, rpc_params, rid))


def _call_workspace_rpc(method_name: str, params: dict[str, Any], rid: Any | None = None) -> dict[str, Any]:
    bootstrap_session = {
        "lock": threading.Lock(),
        "sid": "",
        "transport": current_transport() or _stdio_transport,
        "workspace": _workspace_key(),
    }
    return _post_rpc(bootstrap_session, _rpc_request(method_name, params, rid))


def _send_rpc_async(
    session: dict[str, Any],
    method_name: str,
    params: dict[str, Any],
    rid: Any | None = None,
) -> None:
    def run() -> None:
        try:
            _call_session_rpc(session, method_name, params, rid)
        except Exception as exc:
            print(f"[foundry-backend] async RPC {method_name} failed: {exc}", file=sys.stderr, flush=True)

    threading.Thread(target=run, daemon=True, name=f"foundry-rpc-{method_name}").start()


def _interrupt_session(session: dict[str, Any], reason: str, *, abort_stream: bool = False) -> None:
    sid = str(session.get("sid") or "")
    if sid:
        _send_rpc_async(
            session,
            "session.interrupt",
            {"session_id": sid, "reason": reason},
        )

    if not abort_stream:
        return

    cancel = session.get("subscriber_cancel")
    if cancel is not None:
        cancel.set()

    active_response = session.get("active_response")
    if active_response is not None:
        try:
            active_response.close()
        except Exception as exc:
            print(f"[foundry-backend] active response close failed: {exc}", file=sys.stderr, flush=True)


_DEFAULT_SUBSCRIBER_BACKOFF_MAX_S = 30.0
_SUBSCRIBER_INITIAL_BACKOFF_S = 1.0
_REPLAY_GAP_WARNING_PREFIX = (
    "Some Hermes events for this session were not delivered "
    "(events through seq "
)


def _subscriber_backoff_max() -> float:
    raw = (os.environ.get("HERMES_FOUNDRY_EVENT_SUBSCRIBER_BACKOFF_MAX_S") or "").strip()
    if not raw:
        return _DEFAULT_SUBSCRIBER_BACKOFF_MAX_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SUBSCRIBER_BACKOFF_MAX_S
    return value if value > 0 else _DEFAULT_SUBSCRIBER_BACKOFF_MAX_S


def _emit_replay_gap_warning(sid: str, missed_through: int) -> None:
    _emit(
        "status.update",
        sid,
        {
            "kind": "warning",
            "text": (
                f"{_REPLAY_GAP_WARNING_PREFIX}{missed_through} were dropped). "
                "Run /resume to resync the conversation."
            ),
        },
    )


def _event_subscriber_loop(sid: str) -> None:
    session = _sessions.get(sid)
    if session is None:
        return
    cancel = session.get("subscriber_cancel")
    if cancel is None:
        return
    backoff = _SUBSCRIBER_INITIAL_BACKOFF_S
    backoff_max = _subscriber_backoff_max()

    while not cancel.is_set():
        session = _sessions.get(sid)
        if session is None:
            return

        try:
            since_seq = int(session.get("last_seen_seq", -1))
        except (TypeError, ValueError):
            since_seq = -1

        rid = uuid.uuid4().hex
        rpc = _rpc_request(
            "session.events",
            {"session_id": sid, "since_seq": since_seq},
            rid,
        )
        payload = _rpc_payload(sid, session, rpc, uuid.uuid4().hex)

        try:
            for event in _post_invocation_events(session, payload, cancel):
                if cancel.is_set():
                    return
                event_type = _upstream_event_type(event)
                if event_type == "replay.gap":
                    params = event.get("params") if isinstance(event, dict) else None
                    missed = -1
                    if isinstance(params, dict):
                        payload_data = params.get("payload")
                        if isinstance(payload_data, dict):
                            raw_missed = payload_data.get("missed_through", -1)
                            try:
                                missed = int(raw_missed)
                            except (TypeError, ValueError):
                                missed = -1
                    if not _emit_upstream_event(sid, event):
                        break
                    _emit_replay_gap_warning(sid, missed)
                    continue

                if not _emit_upstream_event(sid, event):
                    break

            backoff = _SUBSCRIBER_INITIAL_BACKOFF_S
            if cancel.is_set():
                return
            if cancel.wait(timeout=backoff):
                return
        except RuntimeError as exc:
            if cancel.is_set():
                return
            print(
                f"[foundry-backend] event subscriber {sid} reconnecting after error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if cancel.wait(timeout=backoff):
                return
            backoff = min(backoff * 2, backoff_max)
        except Exception as exc:
            if cancel.is_set():
                return
            print(
                f"[foundry-backend] event subscriber {sid} unexpected error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if cancel.wait(timeout=backoff):
                return
            backoff = min(backoff * 2, backoff_max)


def _start_event_subscriber(sid: str) -> None:
    session = _sessions.get(sid)
    if session is None:
        return
    if session.get("subscriber_thread") is not None:
        return
    thread = threading.Thread(
        target=_event_subscriber_loop,
        args=(sid,),
        daemon=True,
        name=f"foundry-events-{sid}",
    )
    session["subscriber_thread"] = thread
    thread.start()


def _stop_event_subscriber(session: dict[str, Any]) -> None:
    cancel = session.get("subscriber_cancel")
    if cancel is not None:
        cancel.set()


def _remember_pending_control(sid: str, event_type: str, payload: dict[str, Any]) -> None:
    if event_type not in {"approval.request", "clarify.request", "sudo.request", "secret.request"}:
        return
    request_id = str(payload.get("request_id") or "").strip()
    if request_id:
        _pending_controls[request_id] = sid


def _clear_pending_for_sid(sid: str) -> None:
    for request_id, owner_sid in list(_pending_controls.items()):
        if owner_sid == sid:
            _pending_controls.pop(request_id, None)


def _rpc_error_message(frame: dict[str, Any]) -> str:
    error = frame.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            return str(message)
    return "Foundry RPC failed."


def _upstream_event_type(event: dict[str, Any]) -> str:
    if event.get("jsonrpc") == "2.0" and event.get("method") == "event":
        params = event.get("params")
        if isinstance(params, dict):
            return str(params.get("type") or "").strip()
        return ""
    return str(event.get("type") or "").strip()


def _upstream_event_payload(event: dict[str, Any]) -> Any:
    if event.get("jsonrpc") == "2.0" and event.get("method") == "event":
        params = event.get("params")
        if isinstance(params, dict):
            return params.get("payload")
        return None
    return event.get("payload")


def _emit_upstream_event(sid: str, event: dict[str, Any]) -> bool:
    if event.get("jsonrpc") == "2.0":
        if event.get("method") != "event":
            if event.get("error"):
                _emit("error", sid, {"message": _rpc_error_message(event)})
            return True

        event_type = _upstream_event_type(event)
        payload = _upstream_event_payload(event)
        if _is_duplicate_event(sid, event):
            return True
    else:
        event_type = _upstream_event_type(event)
        payload = _upstream_event_payload(event)

    if not event_type:
        return True
    if event_type == "done":
        return False
    if event_type == "replay.gap":
        # Hosted side may have restarted; the slash registry could have
        # changed. Drop the cache so the next slash press refetches.
        _invalidate_catalog()

    if payload is None:
        payload = {}
    elif not isinstance(payload, dict):
        payload = {"text": str(payload)}
    _remember_pending_control(sid, event_type, payload)
    _emit(event_type, sid, payload)
    return True


def _is_duplicate_event(sid: str, event: dict[str, Any]) -> bool:
    params = event.get("params")
    if not isinstance(params, dict):
        return False
    seq_val = params.get("seq")
    if not isinstance(seq_val, int):
        return False
    session = _sessions.get(sid)
    if session is None:
        return False
    lock = session.get("lock")
    if lock is None:
        return False
    with lock:
        last_seen = session.get("last_seen_seq", -1)
        if not isinstance(last_seen, int):
            last_seen = -1
        if seq_val <= last_seen:
            return True
        session["last_seen_seq"] = seq_val
    return False


def resolve_skin() -> dict:
    return {
        "name": "foundry",
        "branding": {
            "agent_name": "Hermes Foundry",
            "help_header": "Hermes Foundry",
            "prompt_symbol": ">",
            "response_label": " Foundry ",
            "welcome": "Remote Hermes via Azure AI Foundry",
        },
        "colors": {
            "banner_accent": "cyan",
            "banner_border": "blue",
            "banner_dim": "gray",
            "banner_text": "white",
            "banner_title": "cyan",
        },
        "tool_prefix": "|",
    }


def _sess(params: dict, rid: Any) -> tuple[dict[str, Any] | None, dict | None]:
    sid = str(params.get("session_id") or "")
    session = _sessions.get(sid)
    if session is None:
        return None, _err(rid, 4004, "unknown session")
    return session, None


def _invalidate_catalog() -> None:
    global _catalog_cache
    with _catalog_lock:
        _catalog_cache = None


def _get_catalog(force_refresh: bool = False) -> dict[str, Any] | None:
    """Return the cached commands.catalog result, fetching upstream once on miss.

    Returns the upstream `result` dict (pairs, sub, canon, categories,
    skill_count, warning), not the full JSON-RPC envelope. Callers add
    their own rid before returning to the TUI. Returns None when the
    upstream call fails so the caller can decide whether to fall through
    or surface the error.
    """
    global _catalog_cache

    if not force_refresh:
        cached = _catalog_cache
        if cached is not None:
            return cached

    with _catalog_lock:
        if not force_refresh and _catalog_cache is not None:
            return _catalog_cache

        try:
            response = _call_workspace_rpc(
                "commands.catalog",
                {},
                uuid.uuid4().hex,
            )
        except RuntimeError as exc:
            print(
                f"[foundry-backend] catalog fetch failed: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

        if response.get("error"):
            return None

        result = response.get("result")
        if not isinstance(result, dict):
            return None

        _catalog_cache = result
        return result


def _session_id_from_response(response: dict[str, Any]) -> str:
    result = response.get("result")
    if not isinstance(result, dict):
        return ""
    return str(result.get("session_id") or result.get("id") or "")


def _register_session(sid: str, params: dict[str, Any], workspace: str | None = None) -> dict[str, Any]:
    session = _sessions.get(sid)
    transport = current_transport() or _stdio_transport
    cols = int(params.get("cols", 80) or 80)
    if session is not None:
        session["cols"] = cols
        session["transport"] = transport
        return session

    session = {
        "active_response": None,
        "cols": cols,
        "created_at": time.time(),
        "last_seen_seq": -1,
        "lock": threading.Lock(),
        "sid": sid,
        "subscriber_cancel": threading.Event(),
        "subscriber_thread": None,
        "transport": transport,
        "workspace": workspace or _workspace_key(),
    }
    _sessions[sid] = session
    _start_event_subscriber(sid)
    _start_catalog_prewarm(sid)
    return session


def _start_catalog_prewarm(sid: str) -> None:
    """Kick off a background commands.catalog fetch so the first slash
    keystroke after session.create finds the cache already warm.

    No-op when the catalog is already cached.
    """
    if _catalog_cache is not None:
        return

    def run() -> None:
        try:
            _get_catalog()
        except Exception as exc:
            print(
                f"[foundry-backend] catalog prewarm failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    threading.Thread(
        target=run,
        daemon=True,
        name=f"foundry-catalog-prewarm-{sid}",
    ).start()


def _track_session_response(method_name: str, params: dict[str, Any], response: dict[str, Any]) -> None:
    if response.get("error"):
        return
    if method_name in {"session.create", "session.resume"}:
        sid = _session_id_from_response(response)
        if sid:
            _register_session(sid, params)
    elif method_name == "session.delete":
        sid = str(params.get("session_id") or "")
        if sid:
            session = _sessions.pop(sid, None)
            if session is not None:
                _stop_event_subscriber(session)
            _clear_pending_for_sid(sid)


def _proxy_rpc(method_name: str, params: dict[str, Any], rid: Any | None = None) -> dict[str, Any]:
    sid = str(params.get("session_id") or "")
    session = _sessions.get(sid) if sid else None
    try:
        if session is not None:
            response = _call_session_rpc(session, method_name, params, rid)
        else:
            response = _call_workspace_rpc(method_name, params, rid)
    except RuntimeError as exc:
        return _err(rid, 5000, str(exc))
    _track_session_response(method_name, params, response)
    return response


@method("commands.catalog")
def _(rid, params: dict) -> dict:
    catalog = _get_catalog()
    if catalog is None:
        return _proxy_rpc("commands.catalog", params, rid)
    return _ok(rid, catalog)


@method("complete.slash")
def _(rid, params: dict) -> dict:
    text = str(params.get("text") or "")
    if not text.startswith("/"):
        return _ok(rid, {"items": []})

    # Sub-arg matchers (`/model x`, `/skin x`, `/personality x`,
    # `/details thinking expand`, etc.) are bespoke per-command upstream;
    # fall through so the canonical logic runs.
    if " " in text or text.endswith("\t"):
        return _proxy_rpc("complete.slash", params, rid)

    catalog = _get_catalog()
    if catalog is None:
        return _proxy_rpc("complete.slash", params, rid)

    pairs = catalog.get("pairs")
    if not isinstance(pairs, list):
        return _proxy_rpc("complete.slash", params, rid)

    text_lower = text.lower()
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in pairs:
        if not isinstance(entry, list) or len(entry) < 2:
            continue
        name, desc = str(entry[0]), str(entry[1])
        if not name.startswith("/"):
            continue
        if name.lower().startswith(text_lower) and name not in seen:
            items.append({"text": name, "display": name, "meta": desc})
            seen.add(name)
            if len(items) >= _COMPLETE_SLASH_LIMIT:
                break

    if len(items) < _COMPLETE_SLASH_LIMIT:
        for extra in _COMPLETE_SLASH_EXTRAS:
            name = extra["text"]
            if name in seen:
                continue
            if name.lower().startswith(text_lower):
                items.append(dict(extra))
                seen.add(name)
                if len(items) >= _COMPLETE_SLASH_LIMIT:
                    break

    return _ok(rid, {"items": items, "replace_from": 1})


def _proxy_then_invalidate_catalog(method_name: str, params: dict, rid: Any) -> dict[str, Any]:
    response = _proxy_rpc(method_name, params, rid)
    if not response.get("error"):
        _invalidate_catalog()
    return response


@method("skills.reload")
def _(rid, params: dict) -> dict:
    return _proxy_then_invalidate_catalog("skills.reload", params, rid)


@method("skills.manage")
def _(rid, params: dict) -> dict:
    return _proxy_then_invalidate_catalog("skills.manage", params, rid)


@method("reload.mcp")
def _(rid, params: dict) -> dict:
    return _proxy_then_invalidate_catalog("reload.mcp", params, rid)


@method("reload.env")
def _(rid, params: dict) -> dict:
    return _proxy_then_invalidate_catalog("reload.env", params, rid)


@method("session.create")
def _(rid, params: dict) -> dict:
    response = _proxy_rpc("session.create", params, rid)
    if response.get("error"):
        return response
    sid = _session_id_from_response(response)
    if not sid:
        return _err(rid, 5000, "Hermes RPC session.create did not return a session_id.")
    return response


@method("session.close")
def _(rid, params: dict) -> dict:
    sid = str(params.get("session_id") or "")
    session = _sessions.pop(sid, None)
    if session is None:
        return _proxy_rpc("session.close", params, rid)

    _stop_event_subscriber(session)
    _interrupt_session(session, "session closed", abort_stream=True)
    _clear_pending_for_sid(sid)
    try:
        return _call_session_rpc(session, "session.close", params, rid)
    except RuntimeError as exc:
        return _err(rid, 5000, str(exc))


@method("terminal.resize")
def _(rid, params: dict) -> dict:
    sid = str(params.get("session_id") or "")
    if sid and sid in _sessions:
        _sessions[sid]["cols"] = int(params.get("cols", 80) or 80)
    return _proxy_rpc("terminal.resize", params, rid)


def _read_local_image_bytes(path: Path) -> tuple[bytes | None, dict | None]:
    try:
        return path.read_bytes(), None
    except Exception as exc:
        return None, _err(None, 5027, f"image read failed: {exc}")


@method("clipboard.paste")
def _(rid, params: dict) -> dict:
    sid = str(params.get("session_id") or "")
    try:
        from hermes_cli.clipboard import has_clipboard_image, save_clipboard_image
    except Exception as exc:
        return _err(rid, 5027, f"clipboard unavailable: {exc}")

    tmp_path = Path(tempfile.gettempdir()) / f"hermes_foundry_clip_{uuid.uuid4().hex}.png"
    try:
        try:
            saved = save_clipboard_image(tmp_path)
        except Exception as exc:
            return _err(rid, 5027, f"clipboard read failed: {exc}")
        if not saved:
            msg = (
                "Clipboard has image but extraction failed"
                if has_clipboard_image()
                else "No image found in clipboard"
            )
            return _ok(rid, {"attached": False, "message": msg})

        try:
            data = tmp_path.read_bytes()
        except Exception as exc:
            return _err(rid, 5027, f"clipboard read failed: {exc}")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass

    encoded = base64.b64encode(data).decode("ascii")
    filename = f"clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    return _proxy_rpc(
        "image.attach",
        {"session_id": sid, "bytes_b64": encoded, "filename": filename},
        rid,
    )


@method("image.attach")
def _(rid, params: dict) -> dict:
    sid = str(params.get("session_id") or "")
    bytes_b64 = params.get("bytes_b64")
    if isinstance(bytes_b64, str) and bytes_b64:
        return _proxy_rpc("image.attach", params, rid)

    raw = str(params.get("path", "") or "").strip()
    if not raw:
        return _proxy_rpc("image.attach", params, rid)

    try:
        from cli import _detect_file_drop, _resolve_attachment_path, _split_path_input
    except Exception:
        return _proxy_rpc("image.attach", params, rid)

    remainder = ""
    dropped = _detect_file_drop(raw)
    if dropped:
        image_path = dropped["path"]
        remainder = dropped["remainder"]
    else:
        path_token, remainder = _split_path_input(raw)
        image_path = _resolve_attachment_path(path_token)

    if image_path is None or not image_path.exists():
        return _proxy_rpc("image.attach", params, rid)

    data, error = _read_local_image_bytes(image_path)
    if error is not None:
        return _err(rid, error["error"]["code"], error["error"]["message"])

    encoded = base64.b64encode(data).decode("ascii")
    return _proxy_rpc(
        "image.attach",
        {
            "session_id": sid,
            "bytes_b64": encoded,
            "filename": image_path.name,
            "remainder": remainder,
        },
        rid,
    )


@method("input.detect_drop")
def _(rid, params: dict) -> dict:
    sid = str(params.get("session_id") or "")
    raw = str(params.get("text", "") or "")
    try:
        from cli import _detect_file_drop
    except Exception:
        return _proxy_rpc("input.detect_drop", params, rid)

    try:
        dropped = _detect_file_drop(raw)
    except Exception as exc:
        return _err(rid, 5027, str(exc))
    if not dropped:
        return _ok(rid, {"matched": False})

    drop_path = dropped["path"]
    remainder = dropped["remainder"]

    if dropped["is_image"]:
        data, error = _read_local_image_bytes(drop_path)
        if error is not None:
            return _err(rid, error["error"]["code"], error["error"]["message"])
        encoded = base64.b64encode(data).decode("ascii")
        upstream = _proxy_rpc(
            "image.attach",
            {
                "session_id": sid,
                "bytes_b64": encoded,
                "filename": drop_path.name,
                "remainder": remainder,
            },
            rid,
        )
        if upstream.get("error"):
            return upstream
        result = upstream.get("result") or {}
        response: dict[str, Any] = {
            "matched": True,
            "is_image": True,
            "path": str(drop_path),
            "count": result.get("count", 0),
            "text": result.get("text")
            or remainder
            or f"[User attached image: {drop_path.name}]",
        }
        for key in ("width", "height", "token_estimate"):
            if key in result:
                response[key] = result[key]
        return _ok(rid, response)

    text = f"[User attached file: {drop_path}]" + (
        f"\n{remainder}" if remainder else ""
    )
    return _ok(
        rid,
        {
            "matched": True,
            "is_image": False,
            "path": str(drop_path),
            "name": drop_path.name,
            "text": text,
        },
    )


@method("clarify.respond")
def _(rid, params: dict) -> dict:
    request_id = str(params.get("request_id") or "").strip()
    sid = str(params.get("session_id") or _pending_controls.get(request_id) or "")
    session = _sessions.get(sid)
    if session is None:
        return _proxy_rpc("clarify.respond", params, rid)
    try:
        rpc_params = dict(params)
        rpc_params["session_id"] = sid
        response = _call_session_rpc(
            session,
            "clarify.respond",
            rpc_params,
            rid,
        )
    except RuntimeError as exc:
        return _err(rid, 5000, str(exc))
    if request_id:
        _pending_controls.pop(request_id, None)
    return response


@method("sudo.respond")
def _(rid, params: dict) -> dict:
    request_id = str(params.get("request_id") or "").strip()
    sid = str(params.get("session_id") or _pending_controls.get(request_id) or "")
    session = _sessions.get(sid)
    if session is None:
        return _proxy_rpc("sudo.respond", params, rid)
    try:
        rpc_params = dict(params)
        rpc_params["session_id"] = sid
        response = _call_session_rpc(session, "sudo.respond", rpc_params, rid)
    except RuntimeError as exc:
        return _err(rid, 5000, str(exc))
    if request_id:
        _pending_controls.pop(request_id, None)
    return response


@method("secret.respond")
def _(rid, params: dict) -> dict:
    request_id = str(params.get("request_id") or "").strip()
    sid = str(params.get("session_id") or _pending_controls.get(request_id) or "")
    session = _sessions.get(sid)
    if session is None:
        return _proxy_rpc("secret.respond", params, rid)
    try:
        rpc_params = dict(params)
        rpc_params["session_id"] = sid
        response = _call_session_rpc(session, "secret.respond", rpc_params, rid)
    except RuntimeError as exc:
        return _err(rid, 5000, str(exc))
    if request_id:
        _pending_controls.pop(request_id, None)
    return response


@method("approval.respond")
def _(rid, params: dict) -> dict:
    sid = str(params.get("session_id") or "")
    if not sid:
        request_id = str(params.get("request_id") or "").strip()
        sid = str(_pending_controls.get(request_id) or "")
    session = _sessions.get(sid)
    if session is None:
        return _proxy_rpc("approval.respond", params, rid)
    try:
        rpc_params = dict(params)
        rpc_params["session_id"] = sid
        response = _call_session_rpc(session, "approval.respond", rpc_params, rid)
    except RuntimeError as exc:
        return _err(rid, 5000, str(exc))
    request_id = str(params.get("request_id") or "").strip()
    if request_id:
        _pending_controls.pop(request_id, None)
    return response


def _shutdown_sessions() -> None:
    for session in list(_sessions.values()):
        _stop_event_subscriber(session)
        _interrupt_session(session, "gateway shutdown", abort_stream=True)


atexit.register(_shutdown_sessions)
