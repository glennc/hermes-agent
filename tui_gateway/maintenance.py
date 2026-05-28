from __future__ import annotations

import importlib
import json
import os
import threading
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from hermes_constants import get_hermes_home

_DEFAULT_JOBS = ("cron_tick", "cache_cleanup", "paste_sweep", "session_prune")
_ALL_JOBS = (*_DEFAULT_JOBS, "curator")
_SUPPORTED_JOBS = set(_ALL_JOBS)
_maintenance_lock = threading.Lock()


class _MaintenanceAlreadyRunning(RuntimeError):
    pass


class _MaintenanceFileLock(AbstractContextManager):
    def __init__(self, path: Path, metadata: dict[str, Any], stale_after_seconds: float) -> None:
        self._path = path
        self._metadata = metadata
        self._stale_after_seconds = stale_after_seconds
        self._fd: int | None = None
        self._uses_fcntl = False
        self._created_exclusive = False

    def __enter__(self) -> "_MaintenanceFileLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl  # type: ignore[import-not-found]
        except ImportError:
            fcntl = None

        if fcntl is not None:
            fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                os.close(fd)
                raise _MaintenanceAlreadyRunning(self._locked_message()) from exc
            self._fd = fd
            self._uses_fcntl = True
            self._write_metadata()
            return self

        self._acquire_exclusive_file()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._fd is not None:
            if self._uses_fcntl:
                try:
                    import fcntl  # type: ignore[import-not-found]

                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                except (ImportError, OSError):
                    pass
            os.close(self._fd)
            self._fd = None
        if self._created_exclusive:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
        return False

    def _acquire_exclusive_file(self) -> None:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        for _ in range(2):
            try:
                self._fd = os.open(self._path, flags, 0o644)
                self._created_exclusive = True
                self._write_metadata()
                return
            except FileExistsError:
                if not self._is_stale_file():
                    raise _MaintenanceAlreadyRunning(self._locked_message()) from None
                self._path.unlink(missing_ok=True)
        raise _MaintenanceAlreadyRunning(self._locked_message()) from None

    def _is_stale_file(self) -> bool:
        try:
            stat = self._path.stat()
        except FileNotFoundError:
            return True
        return time.time() - stat.st_mtime > self._stale_after_seconds

    def _locked_message(self) -> str:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError:
            raw = ""
        try:
            metadata = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            metadata = {}
        run_id = metadata.get("run_id") if isinstance(metadata, dict) else None
        started_at = metadata.get("started_at") if isinstance(metadata, dict) else None
        if run_id or started_at:
            return f"maintenance is already running (run_id={run_id or 'unknown'}, started_at={started_at or 'unknown'})"
        return "maintenance is already running"

    def _write_metadata(self) -> None:
        if self._fd is None:
            return
        raw = json.dumps(self._metadata, ensure_ascii=False, sort_keys=True) + "\n"
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, raw.encode("utf-8"))
        os.fsync(self._fd)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _positive_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _positive_float(value: Any, default: float, *, minimum: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _selected_jobs(raw: Any) -> list[str]:
    if raw is None:
        requested: list[Any] = list(_DEFAULT_JOBS)
    elif isinstance(raw, str):
        requested = [part.strip() for part in raw.split(",") if part.strip()]
    elif isinstance(raw, (list, tuple)):
        requested = list(raw)
    else:
        requested = [raw]

    selected: list[str] = []
    for item in requested:
        name = str(item).strip()
        if not name:
            continue
        expanded = _ALL_JOBS if name == "all" else _DEFAULT_JOBS if name == "default" else (name,)
        for job in expanded:
            if job not in selected:
                selected.append(job)
    return selected or list(_DEFAULT_JOBS)


def _load_config() -> dict[str, Any]:
    from hermes_cli.config import load_config

    cfg = load_config()
    return cfg if isinstance(cfg, dict) else {}


def _skip(name: str, reason: str) -> dict[str, Any]:
    return {"name": name, "status": "skipped", "reason": reason}


def _assert_hermes_module(module: ModuleType, module_name: str) -> None:
    root = os.environ.get("HERMES_PYTHON_SRC_ROOT", "").strip()
    module_file = getattr(module, "__file__", None)
    if not root or not module_file:
        return
    root_path = Path(root).expanduser().resolve()
    module_path = Path(module_file).expanduser().resolve()
    if not module_path.is_relative_to(root_path):
        raise RuntimeError(
            f"{module_name} resolved to {module_path}, outside HERMES_PYTHON_SRC_ROOT={root_path}"
        )


def _import_hermes_module(module_name: str) -> ModuleType:
    module = importlib.import_module(module_name)
    _assert_hermes_module(module, module_name)
    return module


def _job_cron_tick(payload: dict[str, Any]) -> dict[str, Any]:
    if _bool(payload.get("dry_run")):
        return _skip("cron_tick", "dry_run")
    scheduler = _import_hermes_module("cron.scheduler")
    executed = scheduler.tick(verbose=False, adapters=None, loop=None)
    return {"name": "cron_tick", "status": "ran", "jobs_executed": executed}


def _job_cache_cleanup(payload: dict[str, Any]) -> dict[str, Any]:
    if _bool(payload.get("dry_run")):
        return _skip("cache_cleanup", "dry_run")
    platform_base = _import_hermes_module("gateway.platforms.base")
    max_age_hours = _positive_int(payload.get("cache_max_age_hours"), 24)
    images_removed = platform_base.cleanup_image_cache(max_age_hours=max_age_hours)
    documents_removed = platform_base.cleanup_document_cache(max_age_hours=max_age_hours)
    return {
        "name": "cache_cleanup",
        "status": "ran",
        "cache_max_age_hours": max_age_hours,
        "images_removed": images_removed,
        "documents_removed": documents_removed,
    }


def _job_paste_sweep(payload: dict[str, Any]) -> dict[str, Any]:
    if _bool(payload.get("dry_run")):
        return _skip("paste_sweep", "dry_run")
    debug = _import_hermes_module("hermes_cli.debug")
    deleted, remaining = debug._sweep_expired_pastes()
    return {
        "name": "paste_sweep",
        "status": "ran",
        "deleted": deleted,
        "remaining": remaining,
    }


def _job_session_prune(payload: dict[str, Any]) -> dict[str, Any]:
    if _bool(payload.get("dry_run")):
        return _skip("session_prune", "dry_run")

    cfg = _load_config()
    session_cfg = cfg.get("sessions", {}) if isinstance(cfg.get("sessions"), dict) else {}
    if not _bool(payload.get("force_session_prune")) and not session_cfg.get("auto_prune", False):
        return _skip("session_prune", "disabled")

    retention_days = _positive_int(
        payload.get("session_retention_days") or session_cfg.get("retention_days"),
        90,
    )
    min_interval_hours = _positive_int(
        payload.get("session_prune_min_interval_hours") or session_cfg.get("min_interval_hours"),
        24,
    )

    state = _import_hermes_module("hermes_state")
    db = state.SessionDB()
    try:
        result = db.maybe_auto_prune_and_vacuum(
            retention_days=retention_days,
            min_interval_hours=min_interval_hours,
            sessions_dir=get_hermes_home() / "sessions",
        )
    finally:
        db.close()

    out: dict[str, Any] = {
        "name": "session_prune",
        "status": "skipped" if result.get("skipped") else "ran",
        "retention_days": retention_days,
        "min_interval_hours": min_interval_hours,
        "result": result,
    }
    if result.get("skipped"):
        out["reason"] = "min_interval"
    return out


def _job_curator(payload: dict[str, Any]) -> dict[str, Any]:
    curator = _import_hermes_module("agent.curator")
    dry_run = _bool(payload.get("dry_run"))
    force = _bool(payload.get("force_curator"))
    if not dry_run and not force and not curator.should_run_now():
        return _skip("curator", "not_due")

    summaries: list[str] = []
    result = curator.run_curator_review(
        on_summary=summaries.append,
        synchronous=True,
        dry_run=dry_run,
    )
    return {
        "name": "curator",
        "status": "ran",
        "dry_run": dry_run,
        "result": result,
        "summaries": summaries,
    }


_JOB_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "cron_tick": _job_cron_tick,
    "cache_cleanup": _job_cache_cleanup,
    "paste_sweep": _job_paste_sweep,
    "session_prune": _job_session_prune,
    "curator": _job_curator,
}


def _run_job(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    start = time.perf_counter()
    if name not in _SUPPORTED_JOBS:
        result = {"name": name, "status": "error", "error": f"unknown maintenance job: {name}"}
    else:
        try:
            result = _JOB_HANDLERS[name](payload)
        except Exception as exc:
            result = {"name": name, "status": "error", "error": str(exc)}
    result.setdefault("name", name)
    result.setdefault("status", "ran")
    result["duration_seconds"] = round(time.perf_counter() - start, 3)
    return result


def _maintenance_dir() -> Path:
    return get_hermes_home() / "foundry-maintenance"


def _already_running_result(payload: dict[str, Any], started_at: str, message: str) -> dict[str, Any]:
    return {
        "kind": "hermes.maintenance.result",
        "run_id": payload.get("run_id"),
        "status": "skipped",
        "reason": "already_running",
        "message": message,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "dry_run": _bool(payload.get("dry_run")),
        "selected_jobs": _selected_jobs(payload.get("jobs")),
        "jobs": [],
    }


def _run_locked(payload: dict[str, Any], started_at: str) -> dict[str, Any]:
    start = time.perf_counter()
    selected = _selected_jobs(payload.get("jobs"))
    jobs = [_run_job(name, payload) for name in selected]

    if any(job.get("status") == "error" for job in jobs):
        status = "error"
    elif all(job.get("status") == "skipped" for job in jobs):
        status = "skipped"
    else:
        status = "ok"

    return {
        "kind": "hermes.maintenance.result",
        "run_id": payload.get("run_id"),
        "status": status,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "duration_seconds": round(time.perf_counter() - start, 3),
        "dry_run": _bool(payload.get("dry_run")),
        "selected_jobs": selected,
        "jobs": jobs,
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("maintenance payload must be a JSON object")

    started_at = str(payload.get("started_at") or _utc_now())
    selected_timeout = _positive_float(payload.get("timeout_seconds"), 9 * 60.0, minimum=5.0)
    stale_after_seconds = _positive_float(
        payload.get("stale_lock_seconds"),
        selected_timeout * 2,
        minimum=selected_timeout,
    )
    lock_path = _maintenance_dir() / "maintenance.lock"
    metadata = {
        "run_id": payload.get("run_id"),
        "started_at": started_at,
        "pid": os.getpid(),
        "timeout_seconds": selected_timeout,
    }

    if not _maintenance_lock.acquire(blocking=False):
        return _already_running_result(payload, started_at, "maintenance is already running")

    try:
        try:
            with _MaintenanceFileLock(lock_path, metadata, stale_after_seconds):
                result = _run_locked(payload, started_at)
        except _MaintenanceAlreadyRunning as exc:
            result = _already_running_result(payload, started_at, str(exc))
    finally:
        _maintenance_lock.release()

    result.setdefault("lock_path", str(lock_path))
    return result
