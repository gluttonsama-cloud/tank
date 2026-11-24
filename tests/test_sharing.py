import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile

import pytest

from tank import eval_plan, evaluation, sync, versions
from tank.eval_store import add_manifest, database
from tank.workspace import read_json, write_json


def git(folder, *args):
    subprocess.run(["git", "-C", str(folder), *args], check=True, capture_output=True)


def test_git_publish_and_restore_on_another_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(sync, "PART_BYTES", 4096)
    repo = Path(__file__).resolve().parents[1]
    remote = tmp_path / "remote.git"
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    git(first, "init", "-b", "main")
    git(first, "config", "user.email", "testing@localhost")
    git(first, "config", "user.name", "Sharing test")
    git(first, "config", "core.autocrlf", "true")
    for folder in ("strategies", "suites"):
        shutil.copytree(repo / folder, first / folder)
    resource = first / "resource"
    resource.mkdir()
    checksum = repo / "resource/SHA256SUMS"
    shutil.copy(checksum, resource)
    git(first, "add", ".")
    git(first, "commit", "-m", "fixture")
    git(first, "remote", "add", "origin", str(remote))
    git(first, "push", "origin", "main")
    git(tmp_path, "clone", str(remote), str(second))
    git(second, "config", "core.autocrlf", "true")
    wheel = checksum.read_text().split()[1]
    for folder in (first, second):
        shutil.copy(repo / "resource" / wheel, folder / "resource" / wheel)
    v = versions.create(first, "starter")["id"]
    plan = evaluation.start(first, [v], "baseline", "quick", mode="direct", replay=True, prepare_only=True)
    suite = evaluation.resume(first, plan["run_id"], background=False)
    assert suite["status"] == "completed", suite
    run_id = suite["run_id"]
    expected = evaluation.compare(first, run_id)
    published = sync.push(first, run_id)
    assert sync.list_remote(second) == [run_id]
    assert published["state"] == "published"
    manifest = read_json(first / "published/runs" / run_id / "manifest.json")
    frozen = eval_plan.runtime_directory(first, manifest["engine"])
    changed_package = tmp_path / "new-workbench/tank"
    shutil.copytree(frozen / "tank", changed_package)
    (changed_package / "simulation.py").write_text("raise RuntimeError('new runner')\n", encoding="utf-8")
    monkeypatch.setattr(eval_plan, "__file__", str(changed_package / "eval_plan.py"))
    monkeypatch.setenv("PYTHONPATH", str(changed_package.parent))
    restored = eval_plan.runtime_directory(second, manifest["engine"])
    assert not restored.exists()
    assert sync.fetch(second, run_id)["completed"] == suite["completed"]
    runtime = second / published["runtime"]
    assert runtime.is_file()
    with ZipFile(runtime) as archive:
        assert set(archive.namelist()) == set(eval_plan.RUNTIME_FILES)
    parts = list((second / "published/runs" / run_id).glob("*.parts/*.bin"))
    assert len(parts) > 1 and max(part.stat().st_size for part in parts) <= 4096
    assert evaluation.compare(second, run_id) == expected
    case_id = evaluation.cases(second, run_id, limit=1)["rows"][0]["case_id"]

    def updated_identity(*args, **kwargs):
        pytest.fail("historical run should use its frozen runtime identity algorithm")

    monkeypatch.setattr(eval_plan, "_runtime_identity", updated_identity)
    replay = evaluation.reproduce(second, case_id)
    assert replay["matches_recorded_result"] and replay["matches_recorded_replay"]
    assert (restored / "tank/simulation.py").read_bytes() == (frozen / "tank/simulation.py").read_bytes()
    manifest_path = second / "published/runs" / run_id / "manifest.json"
    wrong = read_json(manifest_path)
    wrong["timeout"] += 1
    write_json(manifest_path, wrong)
    with pytest.raises(ValueError, match="与批次清单不匹配"):
        sync.restore(second, run_id)


def test_restore_legacy_publication_without_runner_bundle(tmp_path):
    run_id = "run-" + "1" * 32
    root = tmp_path / "first"
    repo = Path(__file__).resolve().parents[1]
    shutil.copytree(repo / "strategies", root / "strategies")
    shutil.copytree(repo / "suites", root / "suites")
    (root / "resource").mkdir()
    checksum = repo / "resource/SHA256SUMS"
    _, wheel = checksum.read_text(encoding="utf-8").split()
    shutil.copyfile(checksum, root / "resource/SHA256SUMS")
    shutil.copyfile(checksum.with_name(wheel), root / "resource" / wheel)
    version = versions.create(root, "starter")["id"]
    manifest = eval_plan.make_manifest(root, run_id, [version], "baseline", "quick", 1, "direct", 120, False)
    del manifest["engine"]["source_sha256"]
    with database(root) as db:
        add_manifest(db, manifest)
    assert sync.publish(root, run_id)["runtime"] is None

    restored = tmp_path / "second"
    shutil.copytree(root / "published", restored / "published")
    assert sync.restore(restored, run_id)["status"] == "pending"
