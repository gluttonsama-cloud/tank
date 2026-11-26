"""Identify gameplay and scoring separately from parameter search and profiling."""

import ast
from pathlib import Path

from seedcup.game import rules

from tank.eval_plan import _runtime_identity, digest

SOURCE = Path(__file__).parent
LEGACY_FILES = (
    "cycle_train.py", "cycle_matches.py", "cycle_policy.py", "fit_boss_specialist.py",
    "evaluate_style_router.py", "boss_specialist_policy.py",
)
GAME_NODES = {
    "cycle_matches.py": {
        "GameCase", "Trial", "_map", "_bot", "_game", "series_verdict", "play",
    },
    "evaluate_style_router.py": {"LocalBot"},
    "boss_specialist_policy.py": {
        "_SnapshotSourceLoader", "_SnapshotFinder", "_SNAPSHOT_FINDER", "_snapshot_package", "load_strategy",
    },
}


def _node_names(node):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {target.id for target in node.targets if isinstance(target, ast.Name)}
    return set()


def _gameplay_sources(folder):
    sources = {}
    for name, needed in GAME_NODES.items():
        tree = ast.parse((folder / name).read_bytes())
        nodes = {key: ast.dump(node, include_attributes=False)
                 for node in tree.body for key in _node_names(node) & needed}
        if nodes.keys() != needed:
            raise ValueError(f"评测入口已改变：{name}")
        imports = sorted(ast.dump(node, include_attributes=False) for node in tree.body
                         if isinstance(node, (ast.Import, ast.ImportFrom)))
        sources[name] = digest([imports, nodes]).encode("ascii")
    return sources


def game_protocol(root):
    return _runtime_identity(root, _gameplay_sources(SOURCE), cpp=False) | {"rules": rules.RULES_VERSION, "format": 2}


def equivalent_gameplay(root, prior, current):
    if prior == current:
        return True
    if not prior or prior.get("format") or prior.get("rules") != current["rules"]:
        return False
    archived = root / ".local/cycle/protocols" / prior["runner_sha256"]
    if not all((archived / name).is_file() for name in LEGACY_FILES):
        return False
    full_sources = {name: (archived / name).read_bytes() for name in LEGACY_FILES}
    original = _runtime_identity(root, full_sources, cpp=False) | {"rules": current["rules"]}
    if prior != original:
        return False
    normalized = _runtime_identity(root, _gameplay_sources(archived), cpp=False)
    return normalized | {"rules": current["rules"], "format": 2} == current


def search_protocol():
    names = ("cycle_train.py", "cycle_policy.py", "fit_boss_specialist.py")
    return digest({name: ast.dump(ast.parse((SOURCE / name).read_bytes()), include_attributes=False) for name in names})
