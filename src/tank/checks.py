"""Select and cache the repository's routine checks."""

import argparse
import ast
import hashlib
import importlib.metadata
from importlib.util import resolve_name
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import unquote, urlsplit


NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
TEST_MODULES = {"src", "strategies", "projects", "training", "suites", "tests", "web", "resource"}
TEST_CONFIG = {
    ".gitattributes", ".gitignore", "pyproject.toml", "pytest.ini", "setup.cfg", "setup.py",
    "tox.ini", "uv.lock", "poetry.lock", "Pipfile", "Pipfile.lock", "resource/SHA256SUMS",
}
# The SDK wrapper is started by path. The runner freeze also covers modules loaded by its child.
RUNTIME_FILES = {
    "src/tank/process_bot.py": {"src/tank/_sdk_bot.py"},
    "src/tank/online.py": {"src/tank/_sdk_bot.py"},
    "src/tank/eval_plan.py": {"src/tank/eval_runner.py", "src/tank/_sdk_bot.py", "src/tank/simulation.py"},
}
ASSET_USERS = {
    "projects/": {"src/tank/catalog.py"},
    "strategies/": {"src/tank/versions.py", "src/tank/process_bot.py", "src/tank/online.py"},
    "training/": {"src/tank/jobs.py"},
    "suites/": {"src/tank/eval_plan.py"},
    "web/": {"src/tank/webapp.py"},
    "resource/": {"src/tank/eval_plan.py", "src/tank/online.py"},
}
DYNAMIC_IMPORTS = {"__import__", "import_module", "run_module", "run_path", "spec_from_file_location"}
EXTERNAL_COMMAND_USERS = {"src/tank/checks.py", "src/tank/eval_runner.py", "src/tank/jobs.py", "src/tank/process_bot.py"}
SUBPROCESS_CALLS = {"Popen", "run", "call", "check_call", "check_output"}
RUNTIME_SUFFIXES = {
    ".bat", ".c", ".cc", ".cfg", ".cmd", ".cpp", ".h", ".hpp", ".ini",
    ".json", ".ps1", ".py", ".sh", ".toml", ".yaml", ".yml",
}
LOCAL_LINK = re.compile(r"\]\(\s*(?:<([^>]+)>|([^\s)]+))")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--staged", action="store_true", help="检查 Git 暂存区")
    source.add_argument("--base", metavar="REV", help="检查 HEAD 与指定 Git 版本的差异")
    source.add_argument("--all", action="store_true", help="检查工作区全部未忽略的文件")
    parser.add_argument("--json", action="store_true", help="输出 JSON 检查结果")


def _git(root: Path, *args: str, env: dict[str, str] | None = None) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, creationflags=NO_WINDOW)
    if result.returncode:
        message = (result.stdout + result.stderr).decode("utf-8", errors="replace").strip()
        raise CheckError(message or f"git {args[0]} 失败")
    return result.stdout


class CheckError(Exception):
    pass


def _paths(output: bytes) -> set[str]:
    return {os.fsdecode(name) for name in output.split(b"\0") if name}


def _checkout(root: Path, tree: str, snapshot: Path, index: Path) -> None:
    # Use a private index so Git can materialize the selected tree without changing the real index.
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    _git(root, "read-tree", tree, env=env)
    _git(root, "checkout-index", "--all", f"--prefix={snapshot.as_posix()}/", env=env)


def _select(root: Path, args: argparse.Namespace) -> tuple[set[str], set[str], str | None]:
    if args.staged or args.base:
        if args.staged:
            tree = _git(root, "write-tree").decode().strip()
            current = _paths(_git(root, "ls-tree", "-r", "--name-only", "-z", tree))
            changed = _paths(_git(root, "diff", "--cached", "--no-renames", "--name-only", "-z"))
            _git(root, "diff", "--cached", "--check")
        else:
            base = _git(root, "rev-parse", "--verify", "--end-of-options", f"{args.base}^{{commit}}").decode().strip()
            tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
            current = _paths(_git(root, "ls-tree", "-r", "--name-only", "-z", tree))
            changed = _paths(_git(root, "diff", "--no-renames", "--name-only", "-z", base, tree))
            _git(root, "diff", "--check", base, tree)
        return changed, current, tree

    untracked = _paths(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    tracked = _paths(_git(root, "ls-files", "--cached", "-z"))
    current = {name for name in tracked if (root / name).exists()} | untracked
    if args.all:
        return current, current, None
    changed = _paths(_git(root, "diff", "HEAD", "--no-renames", "--name-only", "-z"))
    _git(root, "diff", "HEAD", "--check")
    return changed | untracked, current, None


def _test_config(name: str) -> bool:
    return (
        name in TEST_CONFIG or PurePosixPath(name).name == "conftest.py"
        or name.startswith((".githooks/", ".github/workflows/", "requirements/"))
        or (name.startswith("requirements") and name.endswith(".txt"))
    )


def _affects_tests(name: str) -> bool:
    return (
        _test_config(name) or PurePosixPath(name).suffix in RUNTIME_SUFFIXES
        or (name.split("/", 1)[0] in TEST_MODULES and not name.endswith((".md", "/.gitkeep")))
    )


def _test_file(name: str) -> bool:
    path = PurePosixPath(name)
    return name.startswith("tests/") and (path.match("test_*.py") or path.match("*_test.py"))


def _module_names(name: str) -> list[str]:
    levels = list(PurePosixPath(name).with_suffix("").parts)
    if len(levels) > 1 and levels[0] == "src":
        levels.pop(0)
    if levels[-1] == "__init__":
        levels.pop()
    canonical = ".".join(levels)
    if levels and levels[0] in {"tests", "training", "strategies"} and len(levels) > 1:
        return [canonical, levels[-1]]
    return [canonical]


def _module_tree(name: str, modules: dict[str, set[str]]) -> set[str]:
    levels = name.split(".")
    return {
        file for size in range(1, len(levels) + 1)
        for file in modules.get(".".join(levels[:size]), ())
    }


def _literal_text(node) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _command_imports(node, name, modules, files) -> set[str] | None:
    argv = node.args[0] if node.args else next((kw.value for kw in node.keywords if kw.arg == "args"), None)
    if not isinstance(argv, (ast.List, ast.Tuple)) or not argv.elts:
        return set() if name in EXTERNAL_COMMAND_USERS else None
    executable, *args = argv.elts
    command = _literal_text(executable)
    python = isinstance(executable, ast.Attribute) and executable.attr == "executable"
    python = python or command in {"python", "python3", "py", "python.exe"}
    if not python:
        return set() if command or name in EXTERNAL_COMMAND_USERS else None
    for option, value in zip(args, args[1:]):
        if _literal_text(option) == "-m":
            target = _literal_text(value)
            return _module_tree(target, modules) if target is not None else None
    for arg in args:
        target = _literal_text(arg)
        if target is not None and target.endswith(".py"):
            return files & {target, (PurePosixPath(name).parent / target).as_posix()}
    return set() if name in EXTERNAL_COMMAND_USERS else None


def _imports(
    root: Path, name: str, modules: dict[str, set[str]], files: set[str], assets: dict[str, set[str]],
) -> set[str] | None:
    try:
        parsed = ast.parse((root / name).read_bytes(), filename=name)
    except SyntaxError:
        return None
    module = _module_names(name)[0]
    in_package = module if name.endswith("/__init__.py") else module.rpartition(".")[0]
    imports = _module_tree(in_package, modules) if in_package else set()
    dynamic_names = {key: key for key in DYNAMIC_IMPORTS}
    subprocess_modules, subprocess_calls = {"subprocess"}, set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            subprocess_modules.update(alias.asname for alias in node.names if alias.name == "subprocess" and alias.asname)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in DYNAMIC_IMPORTS:
                    dynamic_names[alias.asname or alias.name] = alias.name
                if node.module == "subprocess" and alias.name in SUBPROCESS_CALLS:
                    subprocess_calls.add(alias.asname or alias.name)
    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.update(_module_tree(alias.name, modules))
        elif isinstance(node, ast.ImportFrom):
            package = node.module or ""
            if node.level:
                try:
                    package = resolve_name("." * node.level + package, in_package)
                except ImportError:
                    return None
            imports.update(_module_tree(package, modules))
            for alias in node.names:
                imports.update(_module_tree(f"{package}.{alias.name}", modules))
        elif isinstance(node, ast.Call):
            function = node.func
            external = (
                isinstance(function, ast.Name) and function.id in subprocess_calls
            ) or (
                isinstance(function, ast.Attribute) and function.attr in SUBPROCESS_CALLS
                and isinstance(function.value, ast.Name) and function.value.id in subprocess_modules
            )
            if external:
                command_deps = _command_imports(node, name, modules, files)
                if command_deps is None:
                    return None
                imports.update(command_deps)
                continue
            if isinstance(function, ast.Name):
                operation = dynamic_names.get(function.id)
            elif isinstance(function, ast.Attribute) and function.attr in DYNAMIC_IMPORTS:
                operation = function.attr
            else:
                operation = None
            if operation is None:
                continue
            position = 1 if operation == "spec_from_file_location" else 0
            target = _literal_text(node.args[position]) if len(node.args) > position else None
            if target is None:
                return None
            if operation in {"run_path", "spec_from_file_location"}:
                candidates = {target, (PurePosixPath(name).parent / target).as_posix()}
                imports.update(files & candidates)
            else:
                if target.startswith("."):
                    package = _literal_text(node.args[1]) if len(node.args) > 1 else in_package
                    try:
                        target = resolve_name(target, package)
                    except (ImportError, ValueError):
                        return None
                imports.update(_module_tree(target, modules))
        elif (target := _literal_text(node)) is not None:
            imports.update(assets.get(target, ()))
    return imports


def _test_paths(root: Path, changed: set[str], current: set[str], force_all: bool) -> tuple[list[str], set[str]]:
    tests = sorted(name for name in current if _test_file(name))
    changes = {name for name in changed if _affects_tests(name)}
    if not changes and not force_all:
        return [], set()
    inputs = {name for name in current if _affects_tests(name)}
    if not tests:
        return (["tests"], inputs) if any(name.startswith("tests/") and name.endswith(".py") for name in inputs) else ([], set())
    if force_all or any(_test_config(name) for name in changes):
        return tests, inputs

    files = current | changes
    modules: dict[str, set[str]] = {}
    assets: dict[str, set[str]] = {}
    graph: dict[str, set[str] | None] = {}
    for name in files:
        if name.endswith(".py"):
            for module in _module_names(name):
                modules.setdefault(module, set()).add(name)
        if _affects_tests(name) and not (name.endswith(".py") and name.startswith(("src/", "tests/"))):
            path = PurePosixPath(name)
            aliases = {name}
            if name.startswith(tuple(ASSET_USERS)):
                aliases.update(parent.as_posix() for parent in path.parents if len(parent.parts) > 1)
            for alias in aliases:
                assets.setdefault(alias, set()).add(name)
    for name in inputs:
        if name.endswith(".py"):
            edges = _imports(root, name, modules, files, assets)
            if edges is not None:
                edges.update(RUNTIME_FILES.get(name, ()))
            graph[name] = edges
    for prefix, users in ASSET_USERS.items():
        assets = {name for name in files if name.startswith(prefix) and _affects_tests(name)}
        for user in users:
            if user in graph and graph[user] is not None:
                graph[user].update(assets)
    selected, covered = [], set()
    test_inputs = {name for name in inputs if _test_config(name)}
    for test in tests:
        visited, pending = set(), [test]
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            edges = graph.get(name, set())
            if edges is None:
                return tests, inputs
            pending.extend(edges - visited)
        affected = changes & visited
        if affected:
            selected.append(test)
            covered.update(affected)
            test_inputs.update(visited & current)
    if changes - covered:
        return tests, inputs
    return selected, test_inputs


def _test_targets(test_paths: list[str], current: set[str]) -> list[str]:
    # Let pytest discover extra tests configured by its own collection settings for a full run.
    return ["tests"] if test_paths and len(test_paths) == sum(_test_file(name) for name in current) else test_paths


def _check_links(root: Path, documents: set[str]) -> list[str]:
    failures = []
    for name in sorted(documents):
        document = root / name
        for number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for match in LOCAL_LINK.finditer(line):
                target = match.group(1) or match.group(2)
                link = urlsplit(target)
                if link.scheme or link.netloc or not link.path:
                    continue
                target_path = unquote(link.path)
                path = root / target_path.lstrip("/") if target_path.startswith("/") else document.parent / target_path
                if not path.exists():
                    failures.append(f"{name}:{number}: 链接目标不存在 {target}")
    return failures


def _check_python(root: Path, changed: set[str]) -> list[str]:
    failures = []
    for name in sorted(changed):
        if name.endswith(".py"):
            try:
                compile((root / name).read_bytes(), name, "exec")
            except SyntaxError as error:
                failures.append(f"{name}:{error.lineno}: {error.msg}")
    return failures


def _check_javascript(root: Path, changed: set[str], node: str) -> list[str]:
    failures = []
    for name in sorted(changed):
        result = subprocess.run([node, "--check", "--", name], cwd=root, capture_output=True,
                                encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        if result.returncode:
            failures.append(result.stderr.strip().replace(str(root / name), name))
    return failures


def _sdk_wheel(root: Path, target: Path) -> str | None:
    checksum = target / "resource/SHA256SUMS"
    if not checksum.is_file():
        return None
    fields = checksum.read_text(encoding="utf-8").split()
    if len(fields) != 2 or not re.fullmatch(r"[a-f0-9]{64}", fields[0]):
        raise CheckError("resource/SHA256SUMS 必须包含一个 SHA-256 摘要和安装包名称。")
    expected, name = fields
    name = name.lstrip("*")
    if not re.fullmatch(r"seedcup-[A-Za-z0-9_.-]+\.whl", name):
        raise CheckError(f"官方 SDK 安装包名称不合法 {name}")
    source = root / "resource" / name
    destination = target / "resource" / name
    if not source.is_file():
        raise CheckError(f"缺少 resource/{name}，请按 README 下载官方 SDK。")
    if target != root:
        shutil.copyfile(source, destination)
    with destination.open("rb") as package:
        if hashlib.file_digest(package, "sha256").hexdigest() != expected:
            raise CheckError(f"resource/{name} 与当前检查版本的 SHA256SUMS 不一致。")
    return destination.relative_to(target).as_posix()


def _test(root: Path, git_env: list[str], test_paths: list[str]) -> list[str]:
    try:
        importlib.metadata.version("pytest")
    except importlib.metadata.PackageNotFoundError:
        return ["缺少 pytest。请运行 python -m pip install -e '.[dev]'。"]
    env = {
        **os.environ,
        "PYTHONPATH": str(root / "src"),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("PYTEST_ADDOPTS", None)
    for name in git_env:
        env.pop(name, None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", *test_paths],
        cwd=root,
        env=env,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        creationflags=NO_WINDOW,
    )
    if result.returncode:
        return [result.stdout.strip(), result.stderr.strip()]
    return []


def _cache_key(
    root: Path, inputs: set[str], selected: list[str], changed: set[str], tools: bytes, test_paths: list[str],
    test_targets: list[str] | None = None,
) -> str:
    versions = (sys.version, platform.platform(), tools.decode().strip())
    dependencies = sorted(
        (dist.metadata["Name"], dist.version, dist.read_text("direct_url.json") or "")
        for dist in importlib.metadata.distributions()
    ) if "pytest" in selected else []
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(json.dumps([versions, dependencies, selected, sorted(changed), test_paths, test_targets]).encode())
    for name in sorted(inputs):
        digest.update(os.fsencode(name) + b"\0")
        path = root / name
        if path.is_file():
            with path.open("rb") as file:
                digest.update(hashlib.file_digest(file, "sha256").digest())
        else:
            digest.update(b"missing\0")
    return digest.hexdigest()


def _perform(root: Path, args: argparse.Namespace) -> dict:
    changed, current, tree = _select(root, args)
    documents = {name for name in changed & current if name.endswith(".md")}
    if changed - current:
        documents = {name for name in current if name.endswith(".md")}
    python = {name for name in changed & current if name.endswith(".py")}
    javascript = {name for name in changed & current if name.endswith((".js", ".mjs", ".cjs"))}
    result = {"ok": True, "checks": [], "cached": False, "errors": []}
    if not (documents or python or javascript or args.all or any(_affects_tests(name) for name in changed)):
        return result

    with tempfile.TemporaryDirectory(prefix="tank-check-") as directory:
        target = root
        if tree:
            temp = Path(directory)
            target = temp / "tree"
            _checkout(root, tree, target, temp / "index")
        documents = {name for name in documents if (target / name).is_file()}
        python = {name for name in python if (target / name).is_file()}
        test_paths, test_inputs = _test_paths(target, changed, current, args.all)
        selected = [name for name, enabled in (("links", documents), ("syntax", python),
                    ("javascript", javascript), ("pytest", test_paths)) if enabled]
        result["checks"] = selected
        if not selected:
            return result
        if test_paths:
            result["tests"] = test_paths
        test_targets = _test_targets(test_paths, current)
        # Link validity depends on the existence of targets anywhere in the checkout.
        inputs = (current | test_inputs) if documents else test_inputs | python | javascript
        if test_paths:
            wheel = _sdk_wheel(root, target)
            if wheel is not None:
                inputs.add(wheel)
        tools = _git(root, "--version")
        node = ""
        if javascript:
            node = shutil.which("node")
            if node is None:
                raise CheckError("JavaScript 语法检查需要 Node.js。")
            tools += subprocess.check_output([node, "--version"], cwd=target, creationflags=NO_WINDOW)
        key = _cache_key(target, inputs, selected, documents | python | javascript, tools, test_paths, test_targets)
        cache = root / ".local" / "checks" / key
        if cache.is_file():
            result["cached"] = True
            return result
        failures = _check_links(target, documents) + _check_python(target, python)
        if javascript:
            failures.extend(_check_javascript(target, javascript, node))
        if test_paths and not failures:
            git_env = _git(root, "rev-parse", "--local-env-vars").decode().splitlines()
            failures.extend(_test(target, git_env, test_targets))
        result["errors"] = [message for message in failures if message]
        result["ok"] = not failures
        if result["ok"]:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.touch()
    return result


def run(args: argparse.Namespace) -> int:
    try:
        root = Path(os.fsdecode(_git(Path.cwd(), "rev-parse", "--show-toplevel").rstrip(b"\r\n")))
        result = _perform(root, args)
    except (CheckError, OSError, ValueError, subprocess.CalledProcessError) as error:
        result = {"ok": False, "checks": [], "cached": False, "errors": [str(error)]}
    if args.json:
        print(json.dumps(result, ensure_ascii=True))
    elif not result["ok"]:
        for message in result["errors"]:
            print(message, file=sys.stderr)
    return int(not result["ok"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="tank check")
    add_arguments(parser)
    sys.exit(run(parser.parse_args()))
