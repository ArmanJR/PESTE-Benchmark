"""Remote controller policy tests that do not require a Linux GPU container."""

import os
import socket
import stat
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from peste import remote


def test_seal_cache_removes_benchmark_user_write_permissions(
    monkeypatch: Any, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(mode=0o777)
    artifact = cache / "artifact"
    artifact.write_text("data", encoding="utf-8")
    artifact.chmod(0o666)
    ownership: list[Path] = []

    def chown(path: Path, uid: int, gid: int, **kwargs: Any) -> None:
        assert uid == 0 and gid == 0
        ownership.append(path)

    monkeypatch.setattr(remote.os, "chown", chown)
    remote._seal_cache(cache)

    assert ownership == [cache, artifact]
    assert stat.S_IMODE(cache.stat().st_mode) & 0o022 == 0
    assert stat.S_IMODE(artifact.stat().st_mode) & 0o022 == 0


def test_offline_probe_requires_network_and_cache_write_denial(
    monkeypatch: Any, tmp_path: Path
) -> None:
    cache_roots = (tmp_path / "dataset", tmp_path / "hf")
    for root in cache_roots:
        root.mkdir(mode=0o555)
    monkeypatch.setattr(remote, "CACHE_ROOTS", cache_roots)

    def denied_socket(*args: Any, **kwargs: Any) -> socket.socket:
        raise PermissionError("offline")

    monkeypatch.setattr(remote.socket, "socket", denied_socket)
    try:
        assert remote.offline_probe() == {
            "network_denied": True,
            "cache_writes_denied": True,
        }
    finally:
        for root in cache_roots:
            root.chmod(0o755)


def test_remote_action_rejects_unapproved_environment_before_execution(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(remote.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="unsupported environment keys"):
        remote.execute_action(
            "dataset",
            "modern",
            '{"environment":{"VAST_API_KEY":"must-not-cross-boundary"}}',
        )


def test_offline_action_drops_privileges_from_benchmark_users_home(
    monkeypatch: Any, tmp_path: Path
) -> None:
    benchmark_home = tmp_path / "peste"
    benchmark_home.mkdir()
    executable = tmp_path / "peste"
    executable.touch()
    offline_guard = tmp_path / "libpeste_offline.so"
    offline_guard.touch()
    cache_roots = (tmp_path / "dataset", tmp_path / "hf")
    for root in cache_roots:
        root.mkdir()
    account = SimpleNamespace(
        pw_uid=10001,
        pw_gid=10001,
        pw_name="peste",
        pw_dir=str(benchmark_home),
    )
    recorded: dict[str, Any] = {}

    def run(command: list[str], **options: Any) -> SimpleNamespace:
        recorded["command"] = command
        recorded.update(options)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(remote.os, "geteuid", lambda: 0)
    monkeypatch.setattr(remote.pwd, "getpwnam", lambda name: account)
    monkeypatch.setattr(remote, "OFFLINE_GUARD", offline_guard)
    monkeypatch.setattr(remote, "CACHE_ROOTS", cache_roots)
    monkeypatch.setattr(remote, "_assert_cache_is_read_only", lambda root: None)
    monkeypatch.setattr(remote, "_runtime_executable", lambda runtime: executable)
    monkeypatch.setattr(remote.subprocess, "run", run)

    result = remote.execute_action(
        "smoke",
        "nemo",
        '{"model":"nvidia-fastconformer-fa","hardware_profile_json":"{}"}',
    )

    assert result == 0
    assert recorded["cwd"] == str(benchmark_home)
    assert recorded["user"] == account.pw_uid
    assert recorded["group"] == account.pw_gid
    assert recorded["extra_groups"] == ()
    assert recorded["env"]["HOME"] == str(benchmark_home)
    assert recorded["env"]["USER"] == "peste"
    assert recorded["env"]["LOGNAME"] == "peste"


def test_image_metadata_selects_only_non_secret_provenance(monkeypatch: Any) -> None:
    monkeypatch.setenv("PESTE_IMAGE_REFERENCE", "image@sha256:digest")
    monkeypatch.setenv("PESTE_IMAGE_DIGEST", "sha256:digest")
    monkeypatch.setenv("PESTE_SOURCE_REVISION", "revision")
    monkeypatch.setenv("CONTAINER_ID", "123")
    monkeypatch.setenv("VAST_API_KEY", "secret")

    metadata = remote.image_metadata()
    assert metadata["image_reference"] == "image@sha256:digest"
    assert metadata["container_id"] == "123"
    assert "VAST_API_KEY" not in metadata


def test_child_environment_removes_inherited_credentials(monkeypatch: Any) -> None:
    monkeypatch.setenv("CONTAINER_API_KEY", "instance-secret")
    monkeypatch.setenv("VAST_API_KEY", "account-secret")
    monkeypatch.setenv("HF_TOKEN", "hub-secret")
    monkeypatch.setenv("SAFE_VALUE", "retained")

    environment = remote._child_environment()
    assert environment["SAFE_VALUE"] == "retained"
    assert "CONTAINER_API_KEY" not in environment
    assert "VAST_API_KEY" not in environment
    assert "HF_TOKEN" not in environment


def test_allocated_cpu_count_falls_back_when_affinity_is_unavailable(monkeypatch: Any) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(remote.os, "cpu_count", lambda: 12)
    assert remote._allocated_cpu_count() == 12
