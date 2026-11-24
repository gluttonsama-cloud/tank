"""Strategy projects and summaries of their recorded workbench results."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

from . import eval_plan, evaluation, jobs, versions
from .eval_plan import digest, required
from .eval_store import database
from .online import _complete_score
from .workspace import read_json

KINDS = {"rule", "planning", "rl", "imitation", "hybrid", "other"}


def _project(project_id, name, kind="other"):
    return {
        "id": project_id, "name": name, "kind": kind,
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
        project = _project(path.stem, name, kind)
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


def _snapshots(root, projects_by_id, sources, scripts, job_projects):
    attached = defaultdict(list)
    by_training_name = defaultdict(dict)
    for script, project in scripts.items():
        by_training_name[f"训练 {PurePosixPath(script).stem}"][project["id"]] = project
    for snapshot in versions.list_versions(root):
        version_id, name = snapshot["id"], snapshot["name"]
        if version_id == "baseline":
            continue
        matches = dict(job_projects.get(version_id, ()))
        if name in sources:
            project = sources[name]
            matches[project["id"]] = project
        elif not matches:
            training_matches = by_training_name[name]
            if len(training_matches) == 1:
                matches.update(training_matches)
            else:
                project_id = f"source:{name}"
                project = projects_by_id.setdefault(project_id, _project(project_id, name))
                matches[project_id] = project
        for project in matches.values():
            project["versions"].append(snapshot | {"project_id": project["id"]})
            attached[version_id].append(project)
    return attached


def _condition_id(manifest):
    scope = {key: manifest[key] for key in ("mode", "baseline", "engine", "timeout")}
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


def _local_results(root, project_records, attached):
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
    for run in runs:
        manifest = json.loads(run["manifest"])
        if attached.keys().isdisjoint(manifest["candidates"]):
            continue
        condition_id = _condition_id(manifest)
        for row in _comparison_rows(root, run, revision):
            if not row["scored_games"]:
                continue
            record = {
                "version_id": row["candidate"], "run_id": run["run_id"], "created_at": run["created_at"],
                "suite": manifest["suite"]["name"], "suite_id": manifest["suite"]["suite_id"],
                "mode": manifest["mode"], "baseline": manifest["baseline"], "timeout": manifest["timeout"],
                "opponents": [manifest["baseline"]] if manifest["mode"] == "direct" else manifest["suite"]["opponents"],
                "condition_id": condition_id, "status": row["status"],
                "total_score": row["total_score"], "scored_games": row["scored_games"],
                "scheduled_games": row["scheduled_games"],
            }
            for project in attached.get(row["candidate"], ()):
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


def _raw_submission(path):
    data = read_json(path)
    status = data["evaluation"]["status"]
    return {
        "id": data["id"], "version": data["version"], "version_id": data["version_id"],
        "team_id": data["team_id"], "created_at": data["created_at"], "sha256": data["sha256"],
        "status": status, "games": 40,
        "score": None, "breakdown": None,
    } | (_complete_score(data) or {})


def _official_results(root, attached):
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
        for project in attached.get(item["version_id"], ()):
            official = project["official"]
            latest, best = official["latest"], official["best"]
            if latest is None or (item["created_at"], item["id"]) > (latest["created_at"], latest["id"]):
                official["latest"] = item
            if item["score"] is not None and (
                best is None or (item["score"], -item["id"]) > (best["score"], -best["id"])
            ):
                official["best"] = item


def projects(root: Path) -> list[dict]:
    """Read a project inventory from local records without starting work or contacting the contest website."""
    projects_by_id, sources, scripts = _definitions(root)
    job_projects = _training(root, projects_by_id, scripts)
    attached = _snapshots(root, projects_by_id, sources, scripts, job_projects)
    _local_results(root, projects_by_id, attached)
    _official_results(root, attached)
    return sorted(projects_by_id.values(), key=lambda project: (project["name"], project["id"]))
