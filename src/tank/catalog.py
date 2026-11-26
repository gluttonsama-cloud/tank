"""Strategy projects and summaries of their recorded workbench results."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from collections import defaultdict
from pathlib import Path, PurePosixPath

from . import eval_plan, evaluation, jobs, versions
from .eval_plan import digest, required
from .eval_store import database
from .online import _complete_score
from .workspace import read_json

KINDS = {"rule", "planning", "rl", "imitation", "hybrid", "other"}


PURPOSES = {"pve", "pvp", "collection", "bot_sparring", "opponent_sparring", "other"}


def _purpose(value):
    if not isinstance(value, str) or value not in PURPOSES:
        raise ValueError(f"无效的项目用途 {value!r}")
    return value


def _project(project_id, name, kind="other", purpose="other"):
    return {
        "id": project_id, "name": name, "kind": kind, "purpose": _purpose(purpose),
        "sources": [], "training_scripts": [], "versions": [],
        "local": {"latest": None, "best_by_condition": []},
        "official": {"best": None, "latest": None},
        "training": {"latest": None, "active": 0},
    }


def _references(data, field, *, scripts=False):
    values = required(data, field)
    if not isinstance(values, list):
        raise ValueError(f"{field} 必须是列表")
    for value in values:
        if not isinstance(value, str) or not value or "\\" in value:
            raise ValueError(f"{field} 包含无效路径 {value!r}")
        path = PurePosixPath(value)
        if (
            not path.parts or path.is_absolute() or ".." in path.parts
            or path.as_posix() != value
            or (path.suffix != ".py" if scripts else len(path.parts) != 1)
        ):
            raise ValueError(f"{field} 包含无效路径 {value!r}")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} 包含重复路径")
    return values


def _assign(index, refs, project, field):
    for ref in refs:
        if ref in index:
            raise ValueError(f"{field} 中的 {ref} 同时属于 {index[ref]['id']} 和 {project['id']}")
        index[ref] = project


def _definitions(root):
    projects_by_id, sources, scripts = {}, {}, {}
    for path in sorted((root / "projects").glob("*.json")):
        data = read_json(path)
        name, kind = required(data, "name"), required(data, "kind")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path.name} 的 name 必须是非空文本")
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError(f"{path.name} 的 kind 无效 {kind!r}")
        project = _project(path.stem, name, kind, data.get("purpose", "other"))
        project["sources"] = _references(data, "sources")
        project["training_scripts"] = _references(data, "training", scripts=True)
        _assign(sources, project["sources"], project, "sources")
        _assign(scripts, project["training_scripts"], project, "training")
        projects_by_id[project["id"]] = project
    for path in sorted((root / "strategies").glob("*/submission.json")):
        name = path.parent.name
        if name not in sources:
            project = _project(f"source:{name}", name)
            project["sources"] = [name]
            sources[name] = projects_by_id[project["id"]] = project
    return projects_by_id, sources, scripts


def _training_project(projects_by_id, scripts, script):
    if script not in scripts:
        project_id = f"training:{script}"
        project = _project(project_id, f"训练 {PurePosixPath(script).stem}")
        project["training_scripts"] = [script]
        scripts[script] = projects_by_id[project_id] = project
    return scripts[script]


def _training(root, projects_by_id, scripts):
    job_projects = defaultdict(dict)
    for path in (root / ".local/training/jobs").glob("train-*/queued.json"):
        job = jobs.status(root, path.parent.name)
        script = job.get("script")
        if script is None:
            continue
        project = _training_project(projects_by_id, scripts, script)
        training = project["training"]
        latest = training["latest"]
        if latest is None or (job["created_at"], job["job_id"]) > (latest["created_at"], latest["job_id"]):
            fields = ("job_id", "script", "status", "created_at", "snapshot_id", "checkpoint_id", "error")
            training["latest"] = {key: job[key] for key in fields if key in job}
        if job["status"] in {"queued", "running"}:
            training["active"] += 1
        if job.get("snapshot_id") is not None:
            job_projects[job["snapshot_id"]][project["id"]] = project
    return job_projects


def _label_text(value):
    return isinstance(value, str) and bool(value.strip()) and all(ord(char) >= 32 and char != "\x7f" for char in value)


def validate_project_labels(payload, version_ids):
    """Validate display-only project metadata against the batch's snapshot set."""
    if not isinstance(payload, dict):
        raise ValueError("项目归属必须是快照 ID 到项目列表的映射")
    allowed = set(version_ids) - {"baseline"}
    for version_id, labels in payload.items():
        if version_id not in allowed:
            raise ValueError(f"项目归属包含批次外快照 {version_id}")
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"快照 {version_id} 的项目归属必须是非空列表")
        seen = set()
        for label in labels:
            if not isinstance(label, dict) or set(label) - {"number", "purpose"} != {"id", "name", "kind"}:
                raise ValueError(f"快照 {version_id} 的项目归属需要 id、name、kind")
            for field in ("id", "name"):
                if not _label_text(label[field]):
                    raise ValueError(f"快照 {version_id} 的项目 {field} 无效")
            if "number" in label and (type(label["number"]) is not int or not 1 <= label["number"] < 2**53):
                raise ValueError("共享版本号必须为正整数")
            _purpose(label.get("purpose", "other"))
            kind = label["kind"]
            if not isinstance(kind, str) or kind not in KINDS:
                raise ValueError(f"快照 {version_id} 的项目 kind 无效")
            if label["id"] in seen:
                raise ValueError(f"快照 {version_id} 的项目 ID 重复 {label['id']}")
            seen.add(label["id"])
    return payload


def _label_files(directory):
    legacy = directory / "projects.json"
    if legacy.is_file():
        yield legacy
    for path in sorted(directory.glob("*.projects.json")):
        fragment = path.name.removesuffix(".projects.json")
        if eval_plan.HASH.fullmatch(fragment) and (
            (directory / f"{fragment}.jsonl.gz").is_file() or (directory / f"{fragment}.parts").is_dir()
        ):
            yield path


def _published_labels(root, selected):
    if not selected:
        return {}
    labels_by_snapshot = defaultdict(dict)
    for directory in sorted((root / "published/runs").glob("run-*")):
        files = list(_label_files(directory))
        if not files:
            continue
        manifest = read_json(directory / "manifest.json")
        eval_plan.validate_manifest(manifest)
        if manifest["run_id"] != directory.name:
            raise ValueError(f"项目归属的批次目录与清单不匹配 {directory.name}")
        allowed = eval_plan.version_ids(manifest)
        if selected.isdisjoint(allowed):
            continue
        for path in files:
            payload = validate_project_labels(read_json(path), allowed)
            for version_id in selected & payload.keys():
                for label in payload[version_id]:
                    labels_by_snapshot[version_id].setdefault(label["id"], label)
    return labels_by_snapshot


def _snapshot_records(root, selected):
    if selected is None:
        return versions.list_versions(root)
    return [
        {key: value for key, value in versions.describe(root, version_id).items() if key != "files"}
        for version_id in sorted(selected)
    ]


def _snapshots(root, projects_by_id, sources, scripts, job_projects, selected):
    attached = defaultdict(list)
    version_records = {}
    by_training_name = defaultdict(dict)
    for script, project in scripts.items():
        by_training_name[f"训练 {PurePosixPath(script).stem}"][project["id"]] = project
    from .sampling_probes import catalog as probe_catalog
    probe_names = {"行为采样"} | {p["label"] for key, p in probe_catalog().items() if key != "lancer"}
    snapshots = _snapshot_records(root, selected)
    shared = _published_labels(root, {snapshot["id"] for snapshot in snapshots} - {"baseline"})
    for snapshot in snapshots:
        version_id, name = snapshot["id"], snapshot["name"]
        if version_id == "baseline":
            continue
        history = snapshot | {
            "local": {"latest": None, "by_condition": []},
            "official": {"best": None, "latest": None},
        }
        version_records[version_id] = history
        matches = dict(job_projects.get(version_id, ()))
        if name in sources:
            project = sources[name]
            matches[project["id"]] = project
        for label in shared.get(version_id, {}).values():
            project_id = label["id"]
            if project_id not in projects_by_id:
                projects_by_id[project_id] = _project(project_id, label["name"], label["kind"], label.get("purpose", "other"))
            matches[project_id] = projects_by_id[project_id]
        if not matches:
            training_matches = by_training_name[name]
            if len(training_matches) == 1:
                matches.update(training_matches)
            else:
                project_id = f"source:{name}"
                purpose = "collection" if name in probe_names else "other"
                project = projects_by_id.setdefault(project_id, _project(project_id, name, purpose=purpose))
                matches[project_id] = project
        for project in matches.values():
            preferred = shared.get(version_id, {}).get(project["id"], {}).get("number")
            project["versions"].append(history | {"project_id": project["id"], "number": preferred})
            attached[version_id].append(project)
    _number_versions(root, projects_by_id.values())
    return attached, version_records


def _number_versions(root, projects):
    if not any(p["versions"] for p in projects):
        return
    directory = root / ".local"
    directory.mkdir(exist_ok=True)
    with closing(sqlite3.connect(directory / "version-numbers.sqlite3", timeout=30)) as db, db:
        db.execute("""CREATE TABLE IF NOT EXISTS project_version_numbers (
    project TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    number INTEGER NOT NULL,
    PRIMARY KEY (project, snapshot),
    UNIQUE (project, number)
);""")
        db.execute("BEGIN IMMEDIATE")
        for project in projects:
            if not project["versions"]:
                continue
            assigned = dict(db.execute(
                "SELECT snapshot, number FROM project_version_numbers WHERE project=?", (project["id"],)
            ))
            used = set(assigned.values())
            next_number = max(used, default=0) + 1
            # Reserve known shared numbers before assigning numbers to older unlabelled snapshots.
            for version in sorted(project["versions"], key=lambda v: (v["number"] is None, v["display_time"], v["id"])):
                key = version["id"]
                if key not in assigned:
                    number = version["number"]
                    if number is None or number in used:
                        number = next_number
                    db.execute("INSERT INTO project_version_numbers VALUES (?, ?, ?)", (project["id"], key, number))
                    assigned[key] = number
                    used.add(number)
                    next_number = max(next_number, number + 1)
                version["number"] = assigned[key]


def _condition_id(manifest):
    scope = {key: manifest[key] for key in ("mode", "baseline", "engine")}
    scope["timeout"] = float(manifest["timeout"])
    return "condition-" + digest(scope | {"suite_id": manifest["suite"]["suite_id"]})


def _comparison_revision():
    files = (__file__, evaluation.__file__, eval_plan.__file__)
    return digest([hashlib.sha256(Path(filename).read_bytes()).hexdigest() for filename in files])


def _comparison_rows(root, run, revision):
    cache_key = digest([revision, run["manifest"]])
    if run["cache_key"] == cache_key:
        return json.loads(run["records"])
    fields = ("candidate", "status", "total_score", "scored_games", "scheduled_games")
    rows = [
        {field: row[field] for field in fields}
        for row in evaluation.compare(root, run["run_id"])["rows"]
    ]
    if rows and all(row["status"] == "completed" and row["scored_games"] == row["scheduled_games"] for row in rows):
        with database(root) as db:
            db.execute(
                """INSERT INTO catalog_comparisons VALUES (?, ?, ?)
                   ON CONFLICT(run_id) DO UPDATE SET cache_key = excluded.cache_key, records = excluded.records""",
                (run["run_id"], cache_key, json.dumps(rows, separators=(",", ":"), allow_nan=False)),
            )
    return rows


def _condition_quality(record):
    complete = record["status"] == "completed" and record["scored_games"] == record["scheduled_games"]
    return complete, record["scored_games"]


def _local_results(root, project_records, attached, version_records):
    if not attached or not (root / ".local/eval/results.sqlite3").is_file():
        return
    with database(root) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS catalog_comparisons (
                       run_id TEXT PRIMARY KEY, cache_key TEXT NOT NULL, records TEXT NOT NULL
                   )""")
        runs = db.execute(
            """SELECT r.run_id, r.manifest, r.created_at, c.cache_key, c.records FROM runs AS r
               LEFT JOIN catalog_comparisons AS c ON c.run_id = r.run_id WHERE r.attempted = 1
               ORDER BY r.created_at DESC, r.run_id DESC"""
        ).fetchall()
    revision = _comparison_revision()
    best = defaultdict(dict)
    versions_by_condition = defaultdict(dict)
    for run in runs:
        manifest = json.loads(run["manifest"])
        if attached.keys().isdisjoint(manifest["candidates"]):
            continue
        condition_id = _condition_id(manifest)
        for row in _comparison_rows(root, run, revision):
            version_id = row["candidate"]
            if version_id not in attached or not row["scored_games"]:
                continue
            record = {
                "version_id": version_id, "run_id": run["run_id"], "created_at": run["created_at"],
                "suite": manifest["suite"]["name"], "suite_id": manifest["suite"]["suite_id"],
                "rules_version": manifest["suite"]["rules_version"],
                "mode": manifest["mode"], "baseline": manifest["baseline"], "timeout": manifest["timeout"],
                "opponents": [manifest["baseline"]] if manifest["mode"] == "direct" else manifest["suite"]["opponents"],
                "condition_id": condition_id, "status": row["status"],
                "total_score": row["total_score"], "scored_games": row["scored_games"],
                "scheduled_games": row["scheduled_games"],
            }
            history = version_records[version_id]["local"]
            if history["latest"] is None:
                history["latest"] = record
            version_conditions = versions_by_condition[version_id]
            previous = version_conditions.get(condition_id)
            if previous is None or _condition_quality(record) > _condition_quality(previous):
                version_conditions[condition_id] = record
            for project in attached[version_id]:
                local = project["local"]
                if local["latest"] is None:
                    local["latest"] = record
                if row["status"] == "completed":
                    conditions = best[project["id"]]
                    current = conditions.get(condition_id)
                    if current is None or record["total_score"] > current["total_score"]:
                        conditions[condition_id] = record
    for project_id, conditions in best.items():
        project_records[project_id]["local"]["best_by_condition"] = list(conditions.values())
    for version_id, conditions in versions_by_condition.items():
        version_records[version_id]["local"]["by_condition"] = list(conditions.values())


def _raw_submission(path):
    data = read_json(path)
    evaluation = data.get("evaluation")
    status = evaluation["status"] if evaluation else "awaiting_evaluation"
    return {
        "id": data["id"], "version": data["version"], "version_id": data["version_id"],
        "team_id": data["team_id"], "created_at": data["created_at"], "sha256": data["sha256"],
        "status": status, "games": 40,
        "score": None, "breakdown": None,
    } | (_complete_score(data) or {})


def _consider_official(official, item):
    latest, best = official["latest"], official["best"]
    if latest is None or (item["created_at"], item["id"]) > (latest["created_at"], latest["id"]):
        official["latest"] = item
    if item["score"] is not None and (
        best is None or (item["score"], -item["id"]) > (best["score"], -best["id"])
    ):
        official["best"] = item


def _official_results(root, attached, version_records):
    if not attached:
        return
    folder = root / ".local/online"
    cached = folder / "team-scores.json"
    team = read_json(cached) if cached.is_file() else None
    linked = {}
    for path in (folder / "submissions").glob("*.json"):
        record = _raw_submission(path)
        linked[record["team_id"], record["id"]] = record
    if team is None:
        if not linked:
            return
        team_id = max(linked.values(), key=lambda row: (row["created_at"], row["id"]))["team_id"]
        records = {key: row for key, row in linked.items() if row["team_id"] == team_id}
    else:
        team_id = team["team_id"]
        records = {
            key: row for key, row in linked.items() if row["team_id"] == team_id and row["status"] != "completed"
        }
    if team is not None:
        for item in team["submissions"]:
            record = {
                **{key: item[key] for key in (
                    "id", "version", "version_id", "created_at", "sha256", "score", "breakdown", "games",
                )},
                "team_id": team_id, "status": "completed",
            }
            key = record["team_id"], record["id"]
            records[key] = record
    for item in records.values():
        version_id = item["version_id"]
        if version_id not in attached:
            continue
        _consider_official(version_records[version_id]["official"], item)
        for project in attached[version_id]:
            _consider_official(project["official"], item)


def _strategy_index(root, selected=None):
    projects_by_id, sources, scripts = _definitions(root)
    job_projects = _training(root, projects_by_id, scripts)
    attached, version_records = _snapshots(root, projects_by_id, sources, scripts, job_projects, selected)
    return projects_by_id, attached, version_records


def project_labels(root: Path, version_ids) -> dict:
    """Export display-only ownership for the snapshots used by a batch."""
    selected = set(version_ids) - {"baseline"}
    if not selected:
        return {}
    _, attached, _ = _strategy_index(root, selected)
    payload = {
        version_id: [
            {key: project[key] for key in ("id", "name", "kind", "purpose")} | {
                "number": next(v["number"] for v in project["versions"] if v["id"] == version_id)
            }
            for project in sorted(attached[version_id], key=lambda project: project["id"])
        ]
        for version_id in sorted(selected)
    }
    return validate_project_labels(payload, selected)


def projects(root: Path) -> list[dict]:
    """Read a project inventory from local records without starting work or contacting the contest website."""
    projects_by_id, attached, version_records = _strategy_index(root)
    _local_results(root, projects_by_id, attached, version_records)
    _official_results(root, attached, version_records)
    return sorted(projects_by_id.values(), key=lambda project: (project["name"], project["id"]))
