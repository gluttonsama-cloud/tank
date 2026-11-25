"""Local submission preflight and explicit SeedCup team operations."""

import asyncio
import getpass
import gzip
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unicodedata
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import uuid4
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo
from zlib import error as ZlibError

import websockets
from seedcup.game import Game, generate_map, parse_action
from seedcup.game.bots import make_bot
from seedcup.game.replay import FORMAT_VERSION
from seedcup.game.runner import match_info
from seedcup.server.evaluation_manifest import manifest
from seedcup.server.hub import PRACTICE_NAMES

from . import versions
from .eval_plan import SCORE_ITEMS
from .process_bot import ProcessBot
from .workspace import read_json, write_json


SITE = "https://tank.dian.org.cn"
SOCKET = "wss://tank.dian.org.cn/ws/client"
SUBMISSIONS = "/api/team/submissions"
ROOMS = "/api/team/rooms"
RUNTIME_PACKAGES = {"seedcup": "0.4.3", "numpy": "2.2.6"}


class OnlineError(ValueError):
    pass


class RequestUncertain(OnlineError):
    pass


class RemoteError(OnlineError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def account_path(root):
    return root / ".local/online/account.json"


def account(root):
    path = account_path(root)
    if path.is_file():
        return read_json(path)
    username = os.environ.get("SEEDCUP_USERNAME")
    password = os.environ.get("SEEDCUP_PASSWORD")
    if username and password:
        return {"username": username, "password": password}
    return None


def save_account(root, username, password):
    if not isinstance(username, str) or not username.strip() or not isinstance(password, str) or not password:
        raise OnlineError("请填写账号和密码")
    credentials = {"username": username.strip(), "password": password}
    OnlineClient(**credentials)
    write_json(account_path(root), credentials)
    return {"username": credentials["username"]}


class OnlineClient:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.token = None
        self.opener = build_opener(_NoRedirect())
        login = self.request("POST", "/api/login", fields={"username": username, "password": password})
        self.token = login["token"]
        self.user = login["user"]
        if self.user["role"] != "team":
            raise OnlineError("请使用参赛队伍账号登录")

    @classmethod
    def login(cls, root):
        credentials = account(root)
        if not credentials:
            raise OnlineError("请在「策略 → 官网操作 → 比赛账号」填写账号")
        return cls(**credentials)

    def request(self, method: str, path: str, *, fields=None, body=None, content_type=None, compressed=False):
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if fields is not None:
            body = json.dumps(fields).encode()
            content_type = "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(SITE + path, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=15) as response:
                return response.read() if compressed else json.load(response)
        except HTTPError as error:
            if error.code >= 500:
                raise RequestUncertain(f"官网暂时不可用（HTTP {error.code}）") from None
            try:
                detail = json.load(error)["detail"]
            except (KeyError, UnicodeError, json.JSONDecodeError):
                detail = None
            if not isinstance(detail, str):
                detail = f"官网请求失败（HTTP {error.code}）"
            raise RemoteError(error.code, detail) from None
        except (URLError, OSError, IncompleteRead, UnicodeError, json.JSONDecodeError) as error:
            raise RequestUncertain(f"未收到完整官网响应（{type(error).__name__}）") from None


def _archive(source: Path, expected_files: dict) -> bytes:
    files = sorted(path for path in source.rglob("*") if path.is_file())
    if not 1 <= len(files) <= 128:
        raise OnlineError("ZIP 必须包含 1 至 128 个文件")
    if sum(path.stat().st_size for path in files) > 32 * 1024 * 1024:
        raise OnlineError("ZIP 解压后不得超过 32 MiB")
    seen = set()
    archived_files = {}
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        for path in files:
            name = path.relative_to(source).as_posix()
            key = unicodedata.normalize("NFC", name).casefold()
            if key in seen or len(name) > 240 or re.search(r"[\\:\x00-\x1f\x7f]", name):
                raise OnlineError(f"ZIP 路径无效或重名 {name}")
            seen.add(key)
            content = path.read_bytes()
            archived_files[name] = hashlib.sha256(content).hexdigest()
            if name.lower().endswith(".py"):
                if len(content) > 256 * 1024:
                    raise OnlineError(f"Python 单文件不得超过 256 KiB {name}")
                try:
                    compile(content, name, "exec")
                except SyntaxError as error:
                    raise OnlineError(f"{name}:{error.lineno} {error.msg}") from None
            member = ZipInfo(name)
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, content, compress_type=ZIP_DEFLATED, compresslevel=9)
    if archived_files != expected_files:
        raise OnlineError("打包时的文件与策略快照摘要不一致")
    payload = data.getvalue()
    if len(payload) >= 10_000_000:
        raise OnlineError("ZIP 必须小于 10,000,000 字节")
    return payload


_PYTHON_RUNNER = """
import importlib.machinery, json, runpy, sys
from pathlib import Path
packages, entry, adapter, name = sys.argv[1:]
packages = json.loads(packages)
class ContestPackages:
    def find_spec(self, name, path=None, target=None):
        if name in packages:
            return importlib.machinery.PathFinder.find_spec(name, [packages[name]])
sys.meta_path.append(ContestPackages())
sys.path.insert(0, str(Path.cwd()))
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
if name:
    sys.argv = [adapter, entry + ":" + name]
    runpy.run_path(adapter, run_name="__main__")
else:
    sys.argv = [entry]
    sys.path.insert(0, str(Path(entry).resolve().parent))
    runpy.run_path(entry, run_name="__main__")
"""


class _PreflightBot(ProcessBot):
    def __init__(self, root, folder, version_id, spec):
        super().__init__(root, folder, version_id)
        self.spec = spec

    def command(self, source):
        if self.spec["runtime"] == "cpp":
            return super().command(source)
        if sys.version_info[:2] != (3, 12):
            raise OnlineError("正式评测预检需要 Python 3.12")
        packages = {}
        for name, expected in RUNTIME_PACKAGES.items():
            try:
                distribution = importlib.metadata.distribution(name)
            except importlib.metadata.PackageNotFoundError:
                raise OnlineError(f"正式评测预检需要 {name}=={expected}") from None
            if distribution.version != expected:
                raise OnlineError(f"正式评测预检需要 {name}=={expected}")
            packages[name] = str(distribution.locate_file(""))
        return [
            sys.executable, "-I", "-S", "-B", "-u", "-c", _PYTHON_RUNNER,
            json.dumps(packages), self.spec["entrypoint"],
            str(Path(__file__).with_name("_sdk_bot.py")), self.spec["class"] or "",
        ]


def _cpu_seconds(process) -> float:
    if os.name == "nt":
        from ctypes import WinDLL, WinError, byref, wintypes

        clock = WinDLL("kernel32", use_last_error=True).GetProcessTimes
        clock.argtypes = [wintypes.HANDLE, *([wintypes.LPFILETIME] * 4)]
        clock.restype = wintypes.BOOL
        times = [wintypes.FILETIME() for _ in range(4)]
        if not clock(int(process._handle), *(byref(value) for value in times)):
            raise WinError()
        return sum((value.dwHighDateTime << 32) | value.dwLowDateTime for value in times[2:]) / 10_000_000
    if sys.platform.startswith("linux"):
        fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    raise OnlineError("正式评测 CPU 时间预检需要 Windows 或 Linux")


def _trial(root: Path, folder: Path, key: str, spec: dict) -> dict:
    game_map = generate_map(42)
    game = Game(game_map)
    opponent = make_bot("rookie", 0)
    opponent.on_match_start(match_info(game_map, 1, "参赛队伍"))
    info = match_info(game_map, 0, "官方人机")
    info["map"].pop("seed")
    bot = _PreflightBot(root, folder, key, spec)
    try:
        bot.on_match_start(info)
        cpu = 0.0
        while not game.over:
            raw = bot.decide(game.observe(0))
            cpu = _cpu_seconds(bot.process)
            if cpu > 10:
                raise OnlineError(f"策略第 {game.tick} 帧超过单局 10 秒 CPU 限制")
            action, invalid = parse_action(raw)
            if invalid:
                raise OnlineError(f"策略第 {game.tick} 帧动作字段类型或范围无效")
            rival, _ = parse_action(opponent.decide(game.observe(1)))
            game.step([action, rival])
        bot.on_match_end(game.result_summary() | {"side": info["side"]})
    except BaseException:
        bot.close(failed=True)
        raise
    return {"seed": 42, "frames": game.tick, "cpu_seconds": cpu}


def pack(root: Path, version_id: str) -> dict:
    if version_id == "baseline":
        raise OnlineError("请选择一个已保存的参赛策略版本")
    versions.verify(root, version_id)
    source = versions.snapshot_data(root, version_id)
    data = _archive(source, versions.describe(root, version_id)["files"])
    sha256 = hashlib.sha256(data).hexdigest()
    parent = root / ".local/online/packages"
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{sha256}.zip"
    with tempfile.TemporaryDirectory(prefix="seedcup-preflight-", dir=parent) as temp:
        staging = Path(temp)
        with ZipFile(io.BytesIO(data)) as archive:
            declared = archive.read("submission.json")
            if len(declared) > 8192:
                raise OnlineError("submission.json 不得超过 8 KiB")
            spec = manifest(declared, set(archive.namelist()))
            folder = staging / "source"
            archive.extractall(folder)
        trial = _trial(root, folder, "zip-" + sha256, spec)
        draft = staging / "package.zip"
        draft.write_bytes(data)
        os.replace(draft, path)
    return {"version_id": version_id, "path": str(path), "sha256": sha256, "bytes": len(data), "trial": trial}


def _matching(state: dict, version: str, digest: str):
    existing = next((item for item in state["submissions"] if item["version"] == version), None)
    if existing is not None and existing["sha256"] != digest:
        raise OnlineError(f"官网版本标签 {version} 已被不同的 ZIP 使用")
    return existing


def _store(root: Path, client: OnlineClient, submission: dict, package: dict) -> dict:
    if not submission["evaluation"]:
        raise OnlineError("官网记录没有计榜评测状态")
    record = {
        **submission, "version_id": package["version_id"], "package": package["path"],
        "team_id": client.user["id"], "operator": getpass.getuser(),
    }
    write_json(_submission_path(root, client, submission["id"]), record)
    return record


def _submission_path(root: Path, client: OnlineClient, submission_id: int) -> Path:
    return root / ".local/online/submissions" / f"{client.user['id']}-{submission_id}.json"


def _multipart(fields: dict, payload: bytes) -> tuple[bytes, str]:
    boundary = uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
             for name, value in fields.items()]
    parts.extend([
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="submission.zip"\r\n'
        'Content-Type: application/zip\r\n\r\n'.encode(),
        payload, f"\r\n--{boundary}--\r\n".encode(),
    ])
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def submit(root: Path, version_id: str, *, version=None, note="", client=None) -> dict:
    package = pack(root, version_id)
    payload = Path(package["path"]).read_bytes()
    digest = package["sha256"]
    if hashlib.sha256(payload).hexdigest() != digest:
        raise OnlineError("上传 ZIP 与预检摘要不一致")
    version = version.strip() if version is not None else "tank-" + digest[:32]
    note = note.strip()
    if not 1 <= len(version) <= 40 or any(ord(char) < 32 for char in version):
        raise OnlineError("官网版本标签需为 1 至 40 个可见字符")
    if len(note) > 2000:
        raise OnlineError("作品说明不能超过 2000 字")
    client = client if client is not None else OnlineClient.login(root)
    state = client.request("GET", SUBMISSIONS)
    existing = _matching(state, version, digest)
    if existing is not None:
        return _store(root, client, existing, package)
    if not state["window"]["automatic_evaluation"]:
        raise OnlineError("官网当前未启用计榜评测")
    body, content_type = _multipart({"version": version, "entrypoint": "", "note": note}, payload)
    for attempt in range(2):
        try:
            response = client.request("POST", SUBMISSIONS, body=body, content_type=content_type)
        except (RequestUncertain, RemoteError) as error:
            if isinstance(error, RemoteError) and error.status != 409:
                raise
            state = client.request("GET", SUBMISSIONS)
            existing = _matching(state, version, digest)
            if existing is not None:
                return _store(root, client, existing, package)
            if attempt or isinstance(error, RemoteError) or not state["window"]["automatic_evaluation"]:
                raise
        else:
            item = _matching({"submissions": [response["submission"]]}, version, digest)
            if item is None:
                raise OnlineError("官网返回的提交版本标签与请求不一致")
            return _store(root, client, item, package)
    raise AssertionError("upload attempts exhausted")


def status(root: Path, submission_id: int, *, client=None) -> dict:
    client = client if client is not None else OnlineClient.login(root)
    state = client.request("GET", SUBMISSIONS)
    item = next((row for row in state["submissions"] if row["id"] == submission_id), None)
    if item is None:
        raise OnlineError(f"当前队伍没有提交记录 {submission_id}")
    path = _submission_path(root, client, submission_id)
    if path.is_file():
        old = read_json(path)
        if item["sha256"] != old["sha256"] or item["version"] != old["version"]:
            raise OnlineError(f"提交记录 {submission_id} 与本地摘要不一致")
        item = old | item
        write_json(path, item)
    return item


def _score_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _complete_score(item):
    score = item.get("evaluation")
    if not isinstance(score, dict) or score.get("status") != "completed" or score.get("error"):
        return None
    total = score.get("score")
    breakdown = score.get("breakdown")
    if not _score_number(total) or not isinstance(breakdown, dict):
        return None
    parts = {name: breakdown.get(name) for name in SCORE_ITEMS}
    if not all(map(_score_number, parts.values())):
        return None
    if round(total * 100) != sum(round(value * 100) for value in parts.values()):
        return None
    return {"score": total, "breakdown": parts, "games": 40}


def _existing_version(root, version_id):
    if not isinstance(version_id, str) or not re.fullmatch(r"v-[0-9a-f]{64}", version_id):
        return False
    try:
        versions.verify(root, version_id)
    except (ValueError, FileNotFoundError, KeyError):
        return False
    return True


def _archive_version(root, digest):
    path = root / ".local/online/packages" / (digest + ".zip")
    if not path.is_file():
        return None
    try:
        with path.open("rb") as file:
            if (os.fstat(file.fileno()).st_size >= 10_000_000
                    or hashlib.file_digest(file, "sha256").hexdigest() != digest):
                return None
            with ZipFile(file) as archive:
                files = archive.infolist()
                names = {file.filename for file in files}
                if (len(names) != len(files) or not 1 <= len(files) <= 128 or "submission.json" not in names
                        or any(file.is_dir() for file in files)
                        or sum(file.file_size for file in files) > 32 * 1024 * 1024):
                    return None
                hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}
    except (BadZipFile, FileNotFoundError):
        return None
    version_id = versions._version_id(hashes)
    return (version_id, path) if _existing_version(root, version_id) else None


def _associated_version(root, client, item, archives):
    record_path = _submission_path(root, client, item["id"])
    if record_path.is_file():
        record = read_json(record_path)
        if any(record.get(field) != item[field] for field in ("id", "version", "sha256")):
            return None
        if record.get("team_id") != client.user["id"] or not _existing_version(root, record.get("version_id")):
            return None
        return record["version_id"]
    digest = item["sha256"]
    if digest not in archives:
        archives[digest] = _archive_version(root, digest)
    match = archives[digest]
    if match is None:
        return None
    version_id, package = match
    write_json(record_path, item | {
        "version_id": version_id, "package": str(package), "team_id": client.user["id"],
        "link_source": "archive_sha256",
    })
    return version_id


def refresh(root: Path, *, client=None) -> dict:
    """Refresh the team's current official scores using one submission listing request."""
    client = client if client is not None else OnlineClient.login(root)
    listing = client.request("GET", SUBMISSIONS)
    rows = []
    archives = {}
    for item in listing["submissions"]:
        score = _complete_score(item)
        if score is None:
            continue
        digest = item["sha256"]
        if (type(item["id"]) is not int or item["id"] < 1 or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest) or not _score_number(item["created_at"])):
            raise OnlineError("官网返回了无效的提交记录标识")
        rows.append({
            "id": item["id"], "version": item["version"], "sha256": digest, "created_at": item["created_at"],
            "version_id": _associated_version(root, client, item, archives),
        } | score)
    rows.sort(key=lambda item: (item["created_at"], item["id"]), reverse=True)
    best_id = listing["best_id"]
    snapshot = {
        "team_id": client.user["id"], "refreshed_at": time.time(), "best_id": best_id,
        "source_url": f"{SITE}/team#submissionPanel",
        "best": next((row for row in rows if row["id"] == best_id), None),
        "recent": rows[0] if rows else None, "submissions": rows,
    }
    write_json(root / ".local/online/team-scores.json", snapshot)
    return snapshot


def matches(root: Path, *, client=None) -> list[dict]:
    client = client if client is not None else OnlineClient.login(root)
    result = client.request("GET", "/api/team/matches")
    return [
        item | {"replay_url": f"{SITE}/replay/{item['id']}" if item["status"] == "finished" else None}
        for item in result["matches"] if item["kind"] in ("room", "ladder")
    ]


async def _receive(ws, timeout=20) -> dict:
    message = json.loads(await asyncio.wait_for(ws.recv(), timeout))
    if message["type"] == "error":
        if message["code"] == "replaced":
            raise OnlineError("连接被替换")
        raise OnlineError(message["message"])
    return message


PRACTICE_LEVELS = tuple(PRACTICE_NAMES)


def _practice_path(root: Path, record: dict) -> Path:
    return root / ".local/online/practice" / record["version_id"] / record["level"] / f"{record['id']}.json"


def practice_replay_path(root: Path, match_id: int) -> Path:
    return root / ".local/online/replays" / f"{match_id}.json.gz"


def _read_practice_record(path: Path) -> dict:
    match_id = int(path.stem)
    record = read_json(path)
    if (not isinstance(record, dict) or record.get("id") != match_id or record.get("level") != path.parent.name
            or record.get("version_id") != path.parent.parent.name or record["level"] not in PRACTICE_LEVELS):
        raise OnlineError(f"对局 {match_id} 的本机练习记录与快照或档位不一致")
    return record


def history(root: Path, version_id: str) -> list[dict]:
    """Return the ten most recent local practice matches for a snapshot."""
    if not re.fullmatch(r"v-[0-9a-f]{64}", version_id):
        raise OnlineError("策略快照 ID 无效")
    directory = root / ".local/online/practice" / version_id
    paths = (path for level in PRACTICE_LEVELS for path in (directory / level).glob("*.json"))
    recent = sorted(paths, key=lambda path: int(path.stem), reverse=True)[:10]
    matches = []
    for path in recent:
        record = _read_practice_record(path)
        match_id = record["id"]
        if not _score_number(record.get("score")):
            raise OnlineError(f"对局 {match_id} 的本机练习记录的 score 缺失或无效")
        saved = practice_replay_path(root, match_id).is_file()
        url = f"/replay?online={match_id}" if saved else f"{SITE}/replay/{match_id}"
        matches.append({"id": match_id, "level": record["level"], "score": record["score"],
                        "replay_saved": saved, "replay_url": url})
    return matches


def _archive_practice_replay(root: Path, client: OnlineClient, record: dict, *, retry_pending=False) -> dict:
    match_id = record["id"]
    side = record.get("side")
    if side not in ("red", "blue"):
        raise OnlineError("本机练习记录的 side 无效")
    team_id = client.user["id"]
    if record.get("team_id", team_id) != team_id:
        raise OnlineError(f"练习对局 {match_id} 属于另一支队伍")
    opposite = "red" if side == "blue" else "blue"
    endpoint = f"/api/replays/{match_id}"
    try:
        raw = client.request("GET", endpoint, compressed=True)
    except RemoteError as error:
        if not retry_pending or error.status != 404 or str(error) != "回放还在生成，稍后再试":
            raise
        # The server saves replays asynchronously after sending match_end.
        time.sleep(0.5)
        raw = client.request("GET", endpoint, compressed=True)
    try:
        data = json.loads(gzip.decompress(raw))
    except (OSError, EOFError, ZlibError, UnicodeError, json.JSONDecodeError):
        raise OnlineError(f"对局 {match_id} 的官网回放无法解压或解析") from None
    try:
        match = data["match"]
        own, opponent = match[side], match[opposite]
        room = match["room"]
        map_seed = data["map"]["seed"]
        if data["format"] != FORMAT_VERSION or not isinstance(data["frames"], list) or not data["frames"]:
            raise OnlineError(f"对局 {match_id} 的官网回放格式无效")
        if (match["id"] != match_id or match["kind"] != "room" or not isinstance(room, str)
                or not re.fullmatch(r"[A-Z0-9]{6}", room) or room != record.get("room", room)):
            raise OnlineError(f"官网回放与本机练习房对局 {match_id} 不一致")
        if own["id"] != team_id or own["bot"] is not False:
            raise OnlineError(f"官网回放中对局 {match_id} 的队伍或出生侧不匹配")
        if (opponent["id"] is not None or opponent["bot"] is not True
                or opponent["name"] != PRACTICE_NAMES[record["level"]]):
            raise OnlineError(f"官网回放中对局 {match_id} 的练习对手不匹配")
        seeds = (record["seed"], record.get("map_seed"))
        if type(map_seed) is not int or any(seed is not None and seed != map_seed for seed in seeds):
            raise OnlineError(f"官网回放中对局 {match_id} 的地图种子不匹配")
        result = data["result"]
        score = result["scores"][("red", "blue").index(side)]["total"]
        if result != record["result"] or score != record["score"]:
            raise OnlineError(f"官网回放中对局 {match_id} 的练习成绩不匹配")
    except KeyError as error:
        raise OnlineError(f"回放或本机练习记录缺少字段 {error.args[0]}") from None
    except (TypeError, IndexError):
        raise OnlineError(f"对局 {match_id} 的回放或本机练习记录格式无效") from None

    path = practice_replay_path(root, match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / path.name
        staged.write_bytes(raw)
        staged.replace(path)
    archived = record | {"team_id": team_id, "room": room, "map_seed": map_seed,
                         "opponent": opponent["name"], "replay_path": str(path)}
    write_json(_practice_path(root, archived), archived)
    return archived


def replay(root: Path, match_id: int, *, client=None) -> dict:
    """Download the official replay for a match recorded by local practice."""
    if type(match_id) is not int or match_id < 1:
        raise OnlineError("官网对局 ID 需为正整数")
    paths = list((root / ".local/online/practice").glob(f"*/*/{match_id}.json"))
    if len(paths) != 1:
        reason = "存在多条本机练习记录" if paths else "找不到本机练习记录"
        raise OnlineError(f"对局 {match_id} {reason}")
    record = _read_practice_record(paths[0])
    version_id = record["version_id"]
    if versions.describe(root, version_id).get("id") != version_id:
        raise OnlineError(f"对局 {match_id} 的本机练习记录与快照清单的 id 不一致")
    client = client if client is not None else OnlineClient.login(root)
    return _archive_practice_replay(root, client, record)


async def _room_matches(root, source, ws, games, practice_info, completed):
    bot = None
    match_id = None
    opponent = PRACTICE_NAMES[practice_info["level"]]
    try:
        while len(completed) < games:
            message = await _receive(ws)
            kind = message["type"]
            if kind == "match_start" and message["kind"] == "room":
                if message["opponent"] != opponent:
                    raise OnlineError("在线练习连接到了其他对手")
                bot = ProcessBot(root, source, practice_info["version_id"], match_seconds=None)
                match_id = message["match_id"]
                side = message["side"]
                map_seed = message["map"]["seed"]
                await asyncio.to_thread(bot.on_match_start, message)
            elif match_id is not None and message.get("match_id") == match_id:
                if kind == "obs":
                    raw = await asyncio.to_thread(bot.decide, message)
                    action, invalid = parse_action(raw)
                    if invalid:
                        raise OnlineError(f"策略第 {message['frame']} 帧动作字段类型或范围无效")
                    await ws.send(json.dumps({"type": "action", "match_id": match_id,
                                              "frame": message["frame"], **action}))
                elif kind == "match_end":
                    await asyncio.to_thread(bot.on_match_end, message["result"])
                    record = practice_info | {
                        "id": match_id, "result": message["result"], "side": side,
                        "score": message["result"]["scores"][("red", "blue").index(side)]["total"],
                        "map_seed": map_seed, "opponent": opponent, "replay_url": f"{SITE}/replay/{match_id}",
                    }
                    write_json(_practice_path(root, record), record)
                    completed.append(record)
                    bot = match_id = None
                elif kind == "match_abort":
                    raise OnlineError(message["reason"])
    finally:
        if bot is not None:
            bot.close()


async def _practice(root, version_id, client, games, seed, level, completed):
    source = versions.snapshot_data(root, version_id)
    async with websockets.connect(SOCKET, open_timeout=10, max_size=2**20) as ws:
        await ws.send(json.dumps({"type": "login", "username": client.username,
                                  "password": client.password, "sdk": "tank-workbench/0.1"}))
        welcome = await _receive(ws, 15)
        if welcome["type"] != "welcome" or welcome["team"]["id"] != client.user["id"]:
            raise OnlineError("在线练习连接的账号与选手中心不一致")
        room = await asyncio.to_thread(client.request, "POST", ROOMS, fields={
            "opponent": "baseline", "level": level, "games": games, "seed": seed, "public": True,
        })
        code = room["code"]
        if not re.fullmatch(r"[A-Z0-9]{6}", code):
            raise OnlineError("官网返回了无效的练习房间号")
        practice_info = {"version_id": version_id, "level": level, "seed": seed,
                         "room": code, "team_id": client.user["id"]}
        close_error = None
        try:
            await asyncio.to_thread(client.request, "POST", f"{ROOMS}/{code}/start")
            async with asyncio.timeout(games * 100 + 20):
                await _room_matches(root, source, ws, games, practice_info, completed)
        finally:
            try:
                await asyncio.to_thread(client.request, "POST", f"{ROOMS}/{code}/close")
            except OnlineError as error:
                close_error = str(error)
        result = {"room": code, "level": level, "version_id": version_id, "matches": completed}
        if close_error is not None:
            result["close_error"] = close_error
        return result


def practice(root: Path, version_id: str, *, client=None, games=1, seed=None, level="rookie") -> dict:
    if level not in PRACTICE_LEVELS:
        raise OnlineError(f"未知练习档位 {level}")
    if type(games) is not int or games not in (1, 2):
        raise OnlineError("在线练习支持 1 局或 2 局")
    if seed is not None and (type(seed) is not int or not 0 <= seed < 2**31):
        raise OnlineError("地图种子需为 0 到 2147483647 的整数")
    versions.verify(root, version_id)
    client = client if client is not None else OnlineClient.login(root)
    completed = []
    try:
        result = asyncio.run(_practice(root, version_id, client, games, seed, level, completed))
    except TimeoutError:
        raise OnlineError("在线练习超时") from None
    except websockets.exceptions.WebSocketException:
        raise OnlineError("练习连接已断开，可稍后重试") from None
    finally:
        for index, record in enumerate(completed):
            try:
                completed[index] = _archive_practice_replay(root, client, record, retry_pending=True)
            except (OnlineError, OSError) as error:
                record["replay_error"] = str(error)
    return result
