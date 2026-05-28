from tui_gateway import maintenance, server


def test_default_dry_run_skips_default_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = maintenance.run({"run_id": "dry-run", "dry_run": True})

    assert result["status"] == "skipped"
    assert result["run_id"] == "dry-run"
    assert result["selected_jobs"] == [
        "cron_tick",
        "cache_cleanup",
        "paste_sweep",
        "session_prune",
    ]
    assert {job["status"] for job in result["jobs"]} == {"skipped"}
    assert {job["reason"] for job in result["jobs"]} == {"dry_run"}


def test_all_profile_expands_curator_once():
    assert maintenance._selected_jobs(["default", "all", "curator"]) == [
        "cron_tick",
        "cache_cleanup",
        "paste_sweep",
        "session_prune",
        "curator",
    ]


def test_unknown_job_marks_run_error(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    result = maintenance.run({"run_id": "bad-job", "jobs": ["unknown_job"], "dry_run": True})

    assert result["status"] == "error"
    assert result["jobs"][0]["status"] == "error"
    assert result["jobs"][0]["error"] == "unknown maintenance job: unknown_job"


def test_in_process_lock_reports_already_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert maintenance._maintenance_lock.acquire(blocking=False)
    try:
        result = maintenance.run({"run_id": "contended", "dry_run": True})
    finally:
        maintenance._maintenance_lock.release()

    assert result["status"] == "skipped"
    assert result["reason"] == "already_running"
    assert result["jobs"] == []


def test_maintenance_run_is_registered_long_handler(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "maint",
            "method": "maintenance.run",
            "params": {"run_id": "registered", "dry_run": True},
        }
    )

    assert "maintenance.run" in server._LONG_HANDLERS
    assert response["result"]["run_id"] == "registered"
