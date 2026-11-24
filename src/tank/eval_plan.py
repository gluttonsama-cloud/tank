"""Freeze evaluation inputs and expand red/blue pairings."""

from __future__ import annotations

import hashlib
import importlib.metadata
import itertools
import json
import platform
import re
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy
import seedcup
from seedcup.game.rules import RULES_VERSION

from .process_bot import compiler_fingerprint
from .versions import describe

FORMAT = 1
SCORE_ITEMS = ("kill", "damage", "survive", "time")
SUITE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")
VERSION_ID = re.compile(r"baseline|v-[0-9a-f]{64}")
HASH = re.compile(r"[0-9a-f]{64}")
RUN_ID = re.compile(r"run-[0-9a-f]{32}")
RUNTIME_FILES = (
    "__init__.py", "_sdk_bot.py", "eval_plan.py", "eval_runner.py", "process_bot.py", "simulation.py",
    "versions.py", "workspace.py",
)


def json_text(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def required(obj, key):
    if key not in obj:
        raise ValueError(f"缺少字段 {key}")
    return obj[key]


def _integer(obj, key, minimum):
    value = required(obj, key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} 必须是大于或等于 {minimum} 的整数")
    return value


def _versions(values, field):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"{field} 须为非空且无重复的版本 ID 列表")
    for version in values:
        if not isinstance(version, str) or not VERSION_ID.fullmatch(version):
            raise ValueError(f"{field} 包含无效的版本 ID {version!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} 须为非空且无重复的版本 ID 列表")
    return list(values)


def validate_suite(suite, name):
    if not isinstance(suite, dict):
        raise ValueError(f"方案 {name} 必须是 JSON 对象")
    version = _integer(suite, "version", 1)
    opponents = _versions(required(suite, "opponents"), "opponents")
    seeds = required(suite, "seeds")
    if (
        not isinstance(seeds, list) or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("seeds 须为非空且无重复的非负整数列表")
    repeats = _integer(suite, "repeats", 1)
    game_no = suite.get("game_no_start", 0)
    if type(game_no) is not int or game_no < 0:
        raise ValueError("game_no_start 必须是非负整数")
    rules = suite.get("rules_version", RULES_VERSION)
    sides = suite.get("sides", ["red", "blue"])
    if not isinstance(rules, str) or not rules:
        raise ValueError("rules_version 必须是规则版本号")
    if sides != ["red", "blue"]:
        raise ValueError("sides 须依次为 red、blue")
    return {
        "name": name, "version": version, "rules_version": rules, "sides": sides,
        "opponents": opponents, "seeds": seeds,
        "repeats": repeats, "game_no_start": game_no,
    }


def get_suite(root, name):
    if not isinstance(name, str) or not SUITE_NAME.fullmatch(name):
        raise ValueError("方案名只能包含小写字母、数字、连字符和下划线")
    path = root / "suites" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"找不到方案 {name}")
    suite = validate_suite(json.loads(path.read_text(encoding="utf-8")), name)
    return suite | {"suite_id": f"suite-{digest(suite)}"}


def list_suites(root):
    return [get_suite(root, path.stem) for path in sorted((root / "suites").glob("*.json"))]


def _sdk_identity(root):
    wheel_hash, wheel_name = (root / "resource" / "SHA256SUMS").read_text(encoding="utf-8").split()
    if not HASH.fullmatch(wheel_hash):
        raise ValueError("resource/SHA256SUMS 的 SHA-256 格式错误")
    wheel = root / "resource" / wheel_name.lstrip("*")
    if not wheel.is_file() or hashlib.sha256(wheel.read_bytes()).hexdigest() != wheel_hash:
        raise ValueError(f"官方 SDK 安装包缺失或摘要不符 {wheel.name}")
    installed = Path(seedcup.__file__).resolve().parent.parent
    with ZipFile(wheel) as archive:
        for name in archive.namelist():
            if not (name.startswith("seedcup/") and name.endswith(".py")):
                continue
            local = installed / name
            if not local.is_file() or local.read_bytes() != archive.read(name):
                raise ValueError(f"当前导入的 SDK 与官方安装包不一致 {name}")
    return wheel_hash


def _runtime_sources(folder):
    return {name: (folder / name).read_text(encoding="utf-8").encode("utf-8") for name in RUNTIME_FILES}


def _frozen_sources(folder):
    return {path.name: path.read_bytes() for path in sorted(folder.glob("*.py"))}


def _source_hashes(files):
    return {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}


def _runtime_identity(root, files, *, cpp):
    sdk_hash = _sdk_identity(root)
    dist = importlib.metadata.distribution("numpy")
    if (dist.version != numpy.__version__
            or Path(dist.locate_file("numpy/__init__.py")).resolve() != Path(numpy.__file__).resolve()):
        raise ValueError("当前导入的 NumPy 与安装记录不一致")
    identity = {
        "files": _source_hashes(files),
        "python": [sys.implementation.name, sys.version],
        "platform": [sys.platform, platform.machine()],
        "numpy": [numpy.__version__, dist.read_text("WHEEL"), digest(dist.read_text("RECORD"))],
    }
    engine = {"sdk_sha256": sdk_hash, "runner_sha256": digest(identity), "source_sha256": digest(identity["files"])}
    if cpp:
        engine["cpp_sha256"] = compiler_fingerprint()[1]
    return engine


def runner_identity(root, *, cpp=False):
    return _runtime_identity(root, _runtime_sources(Path(__file__).resolve().parent), cpp=cpp)


def engine_fields(spec):
    return {name: spec[name] for name in ("sdk_sha256", "runner_sha256", "source_sha256", "cpp_sha256") if name in spec}


def engine_matches(actual, expected):
    return all(actual.get(key) == value for key, value in expected.items())


def runtime_directory(root, engine):
    return root / ".local/eval/runtimes" / engine["runner_sha256"]


def shared_runtime(root, engine):
    return root / "published/runtimes" / (engine["source_sha256"] + ".zip")


def read_runtime_archive(path, engine):
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            names = {item.filename for item in entries}
            if (len(names) != len(entries) or not {"__init__.py", "eval_runner.py"} <= names
                    or any(not re.fullmatch(r"[a-zA-Z0-9_]+\.py", name) for name in names)
                    or sum(item.file_size for item in entries) > 1_000_000):
                raise ValueError(f"批次运行器文件无效 {path.name}")
            files = {name: archive.read(name) for name in names}
    except (BadZipFile, FileNotFoundError) as error:
        raise ValueError(f"批次运行器文件无法读取 {path.name}") from error
    if digest(_source_hashes(files)) != engine["source_sha256"]:
        raise ValueError(f"批次运行器源码摘要不一致 {path.name}")
    return files


def publish_runtime(root, engine):
    path = shared_runtime(root, engine)
    if path.is_file():
        read_runtime_archive(path, engine)
        return path
    runtime = freeze_runner(root, engine)
    files = _frozen_sources(runtime / "tank")
    if digest(_source_hashes(files)) != engine["source_sha256"]:
        raise ValueError("冻结的运行器源码已修改")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        draft = Path(directory) / path.name
        with ZipFile(draft, "w") as archive:
            for name, contents in sorted(files.items()):
                archive.writestr(ZipInfo(name), contents)
        draft.replace(path)
    return path


def freeze_runner(root, engine):
    destination = runtime_directory(root, engine)
    cpp = "cpp_sha256" in engine
    if destination.is_dir():
        files = _frozen_sources(destination / "tank")
        if "source_sha256" in engine:
            valid = digest(_source_hashes(files)) == engine["source_sha256"]
        else:
            valid = engine_matches(_runtime_identity(root, files, cpp=cpp), engine)
        if not valid:
            raise ValueError("当前 SDK、运行依赖或已冻结的运行器与批次清单不同")
        return destination
    shared = shared_runtime(root, engine) if "source_sha256" in engine else None
    if shared is not None and shared.is_file():
        files = read_runtime_archive(shared, engine)
    else:
        files = _runtime_sources(Path(__file__).resolve().parent)
        if not engine_matches(_runtime_identity(root, files, cpp=cpp), engine):
            raise ValueError("当前 SDK、运行依赖或运行器与批次清单不同")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        staged = Path(directory) / "runtime"
        package = staged / "tank"
        package.mkdir(parents=True)
        for name, contents in files.items():
            (package / name).write_bytes(contents)
        try:
            staged.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
            return freeze_runner(root, engine)
    return destination


def make_manifest(root, run_id, candidates, baseline, suite, shards, mode, timeout, replay):
    manifest = {
        "format": FORMAT, "run_id": run_id, "candidates": list(candidates),
        "baseline": baseline, "suite": get_suite(root, suite), "shards": shards,
        "mode": mode, "timeout": timeout, "replay": replay, "engine": runner_identity(root),
    }
    validate_manifest(manifest)
    if any(describe(root, version)["runtime"] == "cpp" for version in version_ids(manifest)):
        manifest["engine"]["cpp_sha256"] = compiler_fingerprint()[1]
    if manifest["suite"]["rules_version"] != RULES_VERSION:
        raise ValueError(f"当前 SDK 规则版本为 {RULES_VERSION}")
    return manifest


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("批次清单必须是 JSON 对象")
    if required(manifest, "format") != FORMAT:
        raise ValueError("不支持的批次清单格式")
    run_id = required(manifest, "run_id")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("批次 ID 格式错误")
    candidates = _versions(required(manifest, "candidates"), "candidates")
    baseline, = _versions([required(manifest, "baseline")], "baseline")
    if baseline in candidates:
        raise ValueError("候选版本与比较基准重复")
    suite = required(manifest, "suite")
    suite_name = required(suite, "name")
    if not isinstance(suite_name, str) or not SUITE_NAME.fullmatch(suite_name):
        raise ValueError("方案名格式错误")
    normalized = validate_suite(suite, suite_name)
    if suite != normalized | {"suite_id": f"suite-{digest(normalized)}"}:
        raise ValueError("测试方案内容与摘要不符")
    _integer(manifest, "shards", 1)
    if required(manifest, "mode") not in ("opponent", "direct"):
        raise ValueError("mode 只支持 opponent 或 direct")
    timeout = required(manifest, "timeout")
    if type(timeout) not in (int, float) or not 0 < timeout < float("inf"):
        raise ValueError("timeout 必须是有限正数")
    if type(required(manifest, "replay")) is not bool:
        raise ValueError("replay 必须是布尔值")
    engine = required(manifest, "engine")
    for key in ("sdk_sha256", "runner_sha256", *(key for key in ("source_sha256", "cpp_sha256") if key in engine)):
        value = required(engine, key)
        if not isinstance(value, str) or not HASH.fullmatch(value):
            raise ValueError(f"{key} 摘要格式错误")


def version_ids(manifest):
    versions = manifest["candidates"] + [manifest["baseline"]]
    if manifest["mode"] == "opponent":
        versions += manifest["suite"]["opponents"]
    return set(versions)


def _case(manifest, focal, opponent, side, seed, game_no):
    red, blue = (focal, opponent) if side == 0 else (opponent, focal)
    spec = manifest["engine"] | {
        "red": red, "blue": blue, "side": side, "seed": seed, "game_no": game_no,
        "strategy_seed": _strategy_seed(seed, game_no, side),
    }
    return "case-" + digest(spec), spec


@lru_cache(maxsize=4096)
def _strategy_seed(seed, game_no, side):
    return int(digest([seed, game_no, side])[:8], 16)


def _pairings(manifest):
    suite = manifest["suite"]
    direct = manifest["mode"] == "direct"
    opponents = [manifest["baseline"]] if direct else suite["opponents"]
    for seed, repeat, opponent in itertools.product(suite["seeds"], range(suite["repeats"]), opponents):
        game_no = suite["game_no_start"] + repeat
        baseline_cases = [
            _case(manifest, manifest["baseline"], opponent, side, seed, game_no) for side in (0, 1)
        ] if not direct else None
        for candidate in manifest["candidates"]:
            focal_cases = [_case(manifest, candidate, opponent, side, seed, game_no) for side in (0, 1)]
            yield candidate, opponent, seed, repeat, focal_cases, focal_cases if direct else baseline_cases


@lru_cache(maxsize=4)
def _compile_pairings(encoded):
    groups = list(_pairings(json.loads(encoded)))
    cases = {}
    for _, _, _, _, focal, baseline in groups:
        cases.update(focal)
        cases.update(baseline)
    return cases, groups


def _plan_key(manifest):
    return json_text({key: manifest[key] for key in ("candidates", "baseline", "suite", "mode", "engine")})


def pairings(manifest):
    return iter(_compile_pairings(_plan_key(manifest))[1])


def match_specs(manifest):
    return _compile_pairings(_plan_key(manifest))[0]


def shard_for(spec, count):
    # Keep all sides, candidates and their shared baseline for one map/repeat together.
    return _condition_shard(spec["seed"], spec["game_no"], count)


@lru_cache(maxsize=4096)
def _condition_shard(seed, game_no, count):
    return int(digest([seed, game_no])[:16], 16) % count
