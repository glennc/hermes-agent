"""Tests for the bytes_b64 branch of tui_gateway.server image.attach.

The foundry proxy reads the user's local clipboard or local file and
forwards the bytes upstream so they land in $HERMES_HOME/images/ on
the sandbox.
"""

import base64
import os
from pathlib import Path

import pytest


@pytest.fixture()
def server(monkeypatch):
    # Defer import so the autouse _isolate_hermes_home fixture has
    # already pointed HERMES_HOME at a per-test tempdir.
    from tui_gateway import server as srv

    # _hermes_home is cached at module import; re-pin it per test so
    # each test writes into its own isolated tempdir.
    monkeypatch.setattr(srv, "_hermes_home", Path(os.environ["HERMES_HOME"]))
    srv._sessions.clear()
    srv._sessions["s1"] = {
        "attached_images": [],
        "image_counter": 0,
    }
    return srv


def _png_bytes() -> bytes:
    return bytes.fromhex(
        "89504e470d0a1a0a"
        "0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c63600000000000010001"
        "5a4d2e1e0000000049454e44ae426082"
    )


def test_image_attach_bytes_writes_into_hermes_home(server):
    payload = _png_bytes()
    encoded = base64.b64encode(payload).decode("ascii")

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "image.attach",
            "params": {
                "session_id": "s1",
                "bytes_b64": encoded,
                "filename": "screenshot.png",
            },
        }
    )

    assert response["result"]["attached"] is True
    saved_path = Path(response["result"]["path"])
    hermes_home = Path(os.environ["HERMES_HOME"])
    assert saved_path.parent == hermes_home / "images"
    assert saved_path.exists()
    assert saved_path.read_bytes() == payload
    assert response["result"]["count"] == 1
    assert str(saved_path) in server._sessions["s1"]["attached_images"]


def test_image_attach_bytes_disambiguates_existing_filename(server):
    payload = _png_bytes()
    encoded = base64.b64encode(payload).decode("ascii")
    params = {
        "session_id": "s1",
        "bytes_b64": encoded,
        "filename": "shot.png",
    }

    first = server.handle_request(
        {"jsonrpc": "2.0", "id": "r1", "method": "image.attach", "params": params}
    )
    second = server.handle_request(
        {"jsonrpc": "2.0", "id": "r2", "method": "image.attach", "params": params}
    )

    first_path = Path(first["result"]["path"])
    second_path = Path(second["result"]["path"])
    assert first_path != second_path
    assert first_path.exists() and second_path.exists()


def test_image_attach_bytes_rejects_non_image_extension(server):
    encoded = base64.b64encode(b"not really a script").decode("ascii")

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "image.attach",
            "params": {
                "session_id": "s1",
                "bytes_b64": encoded,
                "filename": "evil.sh",
            },
        }
    )

    assert "error" in response
    assert "unsupported image" in response["error"]["message"]
    assert server._sessions["s1"]["attached_images"] == []


def test_image_attach_bytes_rejects_invalid_base64(server):
    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "image.attach",
            "params": {
                "session_id": "s1",
                "bytes_b64": "not_base64_$$$",
                "filename": "shot.png",
            },
        }
    )

    assert "error" in response
    assert "invalid bytes_b64" in response["error"]["message"]


def test_image_attach_bytes_rejects_oversize_payload(server):
    # Build a base64 payload that exceeds the 16 MiB limit. We can use
    # a string of valid-looking base64 characters without actually
    # allocating any binary data — the handler must reject before decode.
    oversize = "A" * (17 * 1024 * 1024)

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "image.attach",
            "params": {
                "session_id": "s1",
                "bytes_b64": oversize,
                "filename": "huge.png",
            },
        }
    )

    assert "error" in response
    assert "too large" in response["error"]["message"]
    # Nothing should have been written.
    images_dir = Path(os.environ["HERMES_HOME"]) / "images"
    assert not images_dir.exists() or not any(images_dir.iterdir())


def test_image_attach_bytes_sanitizes_filename_traversal(server):
    payload = _png_bytes()
    encoded = base64.b64encode(payload).decode("ascii")

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "image.attach",
            "params": {
                "session_id": "s1",
                "bytes_b64": encoded,
                "filename": "../../etc/passwd.png",
            },
        }
    )

    saved_path = Path(response["result"]["path"])
    hermes_home = Path(os.environ["HERMES_HOME"])
    # Path.name strips any traversal components.
    assert saved_path == hermes_home / "images" / "passwd.png"


def test_image_attach_path_path_still_works(server, tmp_path):
    # Existing path-based behavior must remain untouched.
    src = tmp_path / "local.png"
    src.write_bytes(_png_bytes())

    response = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": "r1",
            "method": "image.attach",
            "params": {
                "session_id": "s1",
                "path": str(src),
            },
        }
    )

    assert response["result"]["attached"] is True
    assert response["result"]["path"] == str(src)
