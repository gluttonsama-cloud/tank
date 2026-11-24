import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from tank import checks
from tank.checks import CheckError, _perform, add_arguments


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _args(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args(args)


def _repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Tank tests")
    _git(root, "config", "user.email", "tests@example.invalid")
    (root / "README.md").write_text("# Test\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "initial")


def test_staged_check_ignores_unstaged_edits_and_reuses_success(tmp_path):
    _repo(tmp_path)
    source = tmp_path / "代码.py"
    source.write_text("answer = 42\n", encoding="utf-8")
    _git(tmp_path, "add", source.name)
    source.write_text("answer =\n", encoding="utf-8")

    assert _perform(tmp_path, _args("--staged")) == {
        "ok": True,
        "checks": ["syntax"],
        "cached": False,
        "errors": [],
    }
    assert _perform(tmp_path, _args("--staged"))["cached"]
    assert not _perform(tmp_path, _args())["ok"]

    _git(tmp_path, "add", source.name)
    source.write_text("answer = 42\n", encoding="utf-8")
    failure = _perform(tmp_path, _args("--staged"))
    assert not failure["ok"]
    assert failure["errors"][0].startswith("代码.py:1:")


def test_ci_check_reads_the_committed_tree(tmp_path):
    _repo(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text("[Link](./missing.md)\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-m", "add link")
    (tmp_path / "missing.md").write_text("# Untracked\n", encoding="utf-8")

    failure = _perform(tmp_path, _args("--base", "HEAD^"))

    assert failure["checks"] == ["links"]
    assert failure["errors"] == ["README.md:1: 链接目标不存在 ./missing.md"]


def test_worktree_checks_links_after_a_target_is_deleted(tmp_path):
    _repo(tmp_path)
    target = tmp_path / "target.md"
    target.touch()
    (tmp_path / "README.md").write_text("[Target](target.md)\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "add linked document")
    target.unlink()

    assert _perform(tmp_path, _args())["errors"] == ["README.md:1: 链接目标不存在 target.md"]


def test_staged_check_makes_verified_ignored_wheel_available_to_tests(tmp_path):
    _repo(tmp_path)
    wheel = tmp_path / "resource/seedcup-example.whl"
    wheel.parent.mkdir()
    wheel.write_bytes(b"test dependency")
    checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
    (wheel.parent / "SHA256SUMS").write_text(f"{checksum}  {wheel.name}\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("/resource/*.whl\n/.local/\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_sdk.py").write_text(
        "from pathlib import Path\n"
        "\n"
        "def test_sdk():\n"
        "    assert Path('resource/seedcup-example.whl').read_bytes() == b'test dependency'\n",
        encoding="utf-8",
    )
    _git(tmp_path, "add", ".")

    result = _perform(tmp_path, _args("--staged"))

    assert result["ok"], result["errors"]
    assert _perform(tmp_path, _args("--staged"))["cached"]
    wheel.write_bytes(b"corrupt dependency")
    with pytest.raises(CheckError, match="SHA256SUMS"):
        _perform(tmp_path, _args("--staged"))


@pytest.mark.skipif(shutil.which("node") is None, reason="JavaScript syntax check requires Node.js")
def test_staged_javascript_syntax_ignores_worktree(tmp_path):
    _repo(tmp_path)
    script = tmp_path / "app.mjs"
    script.write_text("const = 1;\n", encoding="utf-8")
    _git(tmp_path, "add", script.name)
    script.write_text("const value = 42\n", encoding="utf-8")

    result = _perform(tmp_path, _args("--staged"))

    assert not result["ok"]
    assert result["checks"] == ["javascript"]
    assert "app.mjs:1" in result["errors"][0]


def _write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _initial_files(root, files):
    _repo(root)
    for name, content in files.items():
        _write(root, name, content)
    _git(root, "add", ".")
    _git(root, "commit", "-m", "test fixtures")


def test_staged_pytest_executes_only_affected_transitive_imports(tmp_path):
    _initial_files(tmp_path, {
        "src/sample/__init__.py": "",
        "src/sample/core.py": "VALUE = 1\n",
        "src/sample/facade.py": "from .core import VALUE\n",
        "tests/test_related.py": "from sample.facade import VALUE\n\ndef test_related():\n    assert VALUE == 2\n",
        "tests/test_unrelated.py": "def test_unrelated():\n    raise AssertionError('unrelated test reached')\n",
    })
    core = tmp_path / "src/sample/core.py"
    core.write_text("VALUE = 2\n", encoding="utf-8")
    _git(tmp_path, "add", "src/sample/core.py")
    core.write_text("VALUE = 3\n", encoding="utf-8")
    targeted = _perform(tmp_path, _args("--staged"))
    assert targeted["tests"] == ["tests/test_related.py"]
    assert targeted["ok"], targeted["errors"]
    assert _perform(tmp_path, _args("--staged"))["cached"]

    all_checks = _perform(tmp_path, _args("--all"))
    assert all_checks["tests"] == ["tests/test_related.py", "tests/test_unrelated.py"]
    assert not all_checks["ok"]
    assert "unrelated test reached" in "\n".join(all_checks["errors"])


@pytest.mark.parametrize(("changed", "expected"), [
    ("src/tank/_sdk_bot.py", ["tests/test_process.py", "tests/test_suite.py"]),
    ("src/tank/worker.py", ["tests/test_child.py", "tests/test_literal.py"]),
    ("extras/tool.py", ["tests/test_file_entry.py"]),
    ("strategies/team/bot.py", ["tests/test_process.py"]),
    ("training/fit.py", ["tests/test_training_job.py"]),
    ("suites/group.json", ["tests/test_suite.py"]),
])
def test_dynamic_entry_points_and_data_choose_consumer_tests(tmp_path, changed, expected):
    files = {
        "src/tank/_sdk_bot.py": "",
        "src/tank/process_bot.py": "",
        "src/tank/worker.py": "",
        "src/tank/dispatch.py": "import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'tank.worker'])\n",
        "src/tank/use_tool.py": "from subprocess import run as start\nstart(['python', 'extras/tool.py'])\n",
        "extras/tool.py": "",
        "src/tank/jobs.py": "",
        "src/tank/eval_plan.py": "",
        "strategies/team/bot.py": "",
        "training/fit.py": "",
        "suites/group.json": "{}",
        "tests/test_child.py": "from tank import dispatch\n",
        "tests/test_literal.py": "from importlib import import_module as load\nload('tank.worker')\n",
        "tests/test_file_entry.py": "from tank import use_tool\n",
        "tests/test_process.py": "from tank import process_bot\n",
        "tests/test_suite.py": "from tank import eval_plan\n",
        "tests/test_training_job.py": "from tank import jobs\n",
        "tests/test_other.py": "",
    }
    for name, content in files.items():
        _write(tmp_path, name, content)
    selected, dependencies = checks._test_paths(tmp_path, {changed}, set(files), False)
    assert selected == expected
    assert changed in dependencies
    assert "tests/test_other.py" not in dependencies


def test_uncertain_changes_and_base_config_trigger_full_suite(tmp_path):
    files = {
        "pyproject.toml": "",
        "src/lone.py": "",
        "tests/test_a.py": "",
        "tests/test_b.py": "",
    }
    for name, content in files.items():
        _write(tmp_path, name, content)
    for changed in ("pyproject.toml", "src/lone.py", "tests/conftest.py"):
        selected, _ = checks._test_paths(tmp_path, {changed}, set(files), False)
        assert selected == ["tests/test_a.py", "tests/test_b.py"]

    files["tests/test_a.py"] = "from importlib import import_module\nimport_module(target)\n"
    _write(tmp_path, "tests/test_a.py", files["tests/test_a.py"])
    selected, _ = checks._test_paths(tmp_path, {"tests/test_b.py"}, set(files), False)
    assert selected == ["tests/test_a.py", "tests/test_b.py"]


def test_cache_keys_distinguish_test_selection(tmp_path):
    args = (tmp_path, set(), ["pytest"], {"src/sample.py"}, b"git")
    assert checks._cache_key(*args, ["tests/test_a.py"]) != checks._cache_key(*args, ["tests/test_b.py"])
