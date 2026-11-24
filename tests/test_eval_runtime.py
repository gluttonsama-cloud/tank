from pathlib import Path
import shutil

import pytest

from tank import eval_plan, evaluation, process_bot, versions


REPO = Path(__file__).resolve().parents[1]


def _workspace(root):
    (root / "resource").mkdir(parents=True)
    checksum = REPO / "resource/SHA256SUMS"
    _, wheel = checksum.read_text(encoding="utf-8").split()
    shutil.copyfile(checksum, root / "resource/SHA256SUMS")
    shutil.copyfile(checksum.with_name(wheel), root / "resource" / wheel)
    (root / "suites").mkdir()
    (root / "suites/one.json").write_text(
        '{"version":1,"opponents":["baseline"],"seeds":[42],"repeats":1}', encoding="utf-8",
    )
    shutil.copytree(REPO / "strategies/starter", root / "strategies/starter")
    return versions.create(root, "starter")["id"]


def test_running_plan_uses_frozen_runner_after_live_sources_change(tmp_path, monkeypatch):
    root = tmp_path / "project"
    candidate = _workspace(root)
    live = tmp_path / "live/tank"
    live.mkdir(parents=True)
    for name in eval_plan.RUNTIME_FILES:
        shutil.copyfile(Path(eval_plan.__file__).with_name(name), live / name)
    monkeypatch.setattr(eval_plan, "__file__", str(live / "eval_plan.py"))
    monkeypatch.setenv("PYTHONPATH", str(live.parent))
    plan = evaluation.start(root, [candidate], "baseline", "one", mode="direct", prepare_only=True)
    (live / "simulation.py").write_text("raise RuntimeError('live code selected')\n", encoding="utf-8")

    result = evaluation.resume(root, plan["run_id"], background=False)

    assert result["status"] == "completed"
    assert result["completed"] == 2
    new = evaluation.estimate(root, [candidate], "baseline", "one", mode="direct")
    assert new["cached"] == 0


def test_runner_rejects_sdk_import_from_another_installation(tmp_path, monkeypatch):
    root = tmp_path / "project"
    _workspace(root)
    monkeypatch.setattr(eval_plan.seedcup, "__file__", str(tmp_path / "other/seedcup/__init__.py"))

    with pytest.raises(ValueError, match="导入的 SDK 与官方安装包不一致"):
        eval_plan.runner_identity(root)


def test_strategy_checks_the_actual_copied_files_before_running(tmp_path):
    root = tmp_path / "project"
    candidate = _workspace(root)
    bot = versions.load_bot(root, candidate)
    (versions.snapshot_data(root, candidate) / "bot.py").write_text("raise RuntimeError()\n", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="对局策略快照内容校验失败"):
            bot.on_match_start({})
    finally:
        bot.close()


def test_cpp_cache_is_keyed_by_compiler_fingerprint(tmp_path, monkeypatch):
    source = tmp_path / "strategy"
    source.mkdir()
    (source / "main.cpp").write_text("int main() {}", encoding="utf-8")
    (source / "submission.json").write_text('{"runtime":"cpp","entrypoint":"main.cpp"}', encoding="utf-8")
    compiler_id = "first"
    used = []

    def compile_cpp(args, *, cwd, check, timeout, creationflags):
        assert cwd == source
        assert creationflags == process_bot.NO_WINDOW
        used.append(args)
        Path(args[-1]).write_bytes(b"binary")

    monkeypatch.setattr(process_bot, "compiler_fingerprint", lambda: ("g++", compiler_id))
    monkeypatch.setattr(process_bot.subprocess, "run", compile_cpp)
    bot = process_bot.ProcessBot(tmp_path, source, "version")

    original = bot.command(source)
    assert bot.command(source) == original
    compiler_id = "second"
    assert bot.command(source) != original
    assert len(used) == 2
