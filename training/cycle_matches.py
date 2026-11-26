"""SeedCup match execution and paired local acceptance statistics."""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from statistics import fmean

import numpy as np
from seedcup.game import Game, generate_map, parse_action
from seedcup.game.bots import make_bot
from seedcup.game.runner import match_info
from seedcup.server.evaluation_scoring import summarize

from tank import versions
from tank.eval_plan import digest
from tank.workspace import read_json, write_json

from .boss_specialist_policy import load_strategy
from .evaluate_style_router import LocalBot


@dataclass(frozen=True)
class GameCase:
    map_seed: int
    side: int
    contestant_seed: int
    opponent_seed: int
    game_no: int


@dataclass(frozen=True)
class Trial:
    style: str
    contestant: str
    opponent: str
    games: tuple[GameCase, ...]


@cache
def _map(seed):
    return generate_map(seed)


def _bot(source, seed):
    if source.startswith("sdk:"):
        return LocalBot(seed, make_bot, source.removeprefix("sdk:"), seed)
    return LocalBot(seed, load_strategy(Path(source)))


def _game(contestant_path, opponent_path, case):
    game_map = _map(case.map_seed)
    game = Game(game_map)
    side = case.side
    bots = [_bot(contestant_path, case.contestant_seed), _bot(opponent_path, case.opponent_seed)]
    paths = [contestant_path, opponent_path]
    if side:
        bots.reverse()
        paths.reverse()
    for player, bot in enumerate(bots):
        opponent_name = "官方人机" if paths[1 - player].startswith("sdk:") else "参赛队伍"
        info = match_info(game_map, player, opponent_name, game=case.game_no)
        if not paths[player].startswith("sdk:"):
            del info["map"]["seed"]
        bot.call(bot.bot.on_match_start, {"type": "match_start", **info})
    while not game.over:
        actions = []
        for player, bot in enumerate(bots):
            obs = {"type": "obs", **game.observe(player)}
            action, bad = parse_action(bot.call(bot.bot.decide, obs))
            actions.append(action)
            game.violations[player] += bad
        game.step(actions)
    result = game.result_summary()
    for player, bot in enumerate(bots):
        bot.call(bot.bot.on_match_end, {"type": "match_end", "side": ("red", "blue")[player], **result})
    return asdict(case) | {"result": result, "invalid_actions": game.violations[side]}


def series_verdict(wins, damage):
    """Contestant series points, with uncapped shell damage breaking ties."""
    score, opponent = zip(wins, damage, strict=True)
    return 1.0 if score > opponent else 0.0 if score < opponent else .5


def play(trial):
    wins = [0, 0]
    damage = [0, 0]
    games = []
    for case in trial.games:
        row = _game(trial.contestant, trial.opponent, case)
        games.append(row)
        result = row["result"]
        if result["winner"] in (0, 1):
            wins[int(result["winner"] != case.side)] += 1
        damage[0] += result["stats"][case.side]["damage"]
        damage[1] += result["stats"][1 - case.side]["damage"]
        if max(wins) == 6:
            break
    return {"style": trial.style, "games": games, "series_points": series_verdict(wins, damage),
            "wins": wins, "actual_damage": damage}


def open_pool(max_workers):
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    return ProcessPoolExecutor(max_workers=max_workers)


def binding(trials, protocol):
    packages = {path for trial in trials for path in (trial.contestant, trial.opponent)}
    identities = {path: path if path.startswith("sdk:") else versions._version_id(versions._file_hashes(Path(path)))
                  for path in packages}
    return {"protocol": protocol,
            "contestants": sorted({identities[trial.contestant] for trial in trials}),
            "opponents": {trial.style: identities[trial.opponent] for trial in trials},
            "schedule_sha256": digest([(trial.style, [asdict(game) for game in trial.games]) for trial in trials]),
            "conditions_sha256": digest([(trial.style, identities[trial.contestant], identities[trial.opponent],
                                          [asdict(game) for game in trial.games]) for trial in trials])}


def run_trials(pool, trials, capacity, report_path=None, *, protocol=None):
    """Checkpoint at each bounded batch, preserving order across worker counts."""
    expected = binding(trials, protocol) if report_path else None
    results = []
    if report_path and report_path.exists():
        saved = read_json(report_path)
        if not isinstance(saved, dict) or saved["binding"] != expected:
            raise ValueError(f"逐局检查点的策略快照、协议或条件与本轮不符：{report_path}")
        results = saved["trials"]
    if len(results) > len(trials):
        raise ValueError(f"逐局检查点超出评测计划：{report_path}")
    for row, trial in zip(results, trials):
        games = row["games"]
        if (row["style"] != trial.style or not 0 < len(games) <= len(trial.games)
                or any(any(game[key] != value for key, value in asdict(case).items())
                       for game, case in zip(games, trial.games))):
            raise ValueError(f"逐局检查点与评测计划不符：{report_path}")
    while len(results) < len(trials):
        start = len(results)
        workers = capacity()
        results.extend(pool.map(play, trials[start:start + workers], chunksize=1))
        if report_path:
            write_json(report_path, {"binding": expected, "trials": results})
    return results


def _games(results):
    return [game for result in results for game in result["games"]]


def _damage_margin(row):
    result, side = row["result"], row["side"]
    own, opponent = result["stats"][side], result["stats"][1 - side]
    return (own["damage"] - opponent["damage"], own["hp"] - opponent["hp"])


def game_metrics(results):
    rows = _games(results)
    scores = [g["result"]["scores"][g["side"]]["total"] for g in rows]
    margins = [_damage_margin(row) for row in rows]
    totals = summarize(rows)
    return {"games": len(rows), "total": totals["score_cents"] / 100, "breakdown": totals["breakdown"],
            "minimum": min(scores), "mean": fmean(scores),
            "wins": sum(g["result"]["winner"] == g["side"] for g in rows),
            "draws": sum(g["result"]["winner"] == -1 for g in rows),
            "mean_damage_margin": fmean(m[0] for m in margins), "mean_hp_margin": fmean(m[1] for m in margins),
            "invalid_actions": sum(g["invalid_actions"] for g in rows)}


def search_utility(line, metrics):
    if metrics["invalid_actions"]:
        return -1e9
    if line == "pve":
        return metrics["total"]
    return ((metrics["wins"] + .5 * metrics["draws"]) / metrics["games"]
            + .0002 * metrics["mean_damage_margin"] + .0001 * metrics["mean_hp_margin"])


def paired_bootstrap(samples, seed):
    """Resample whole matched groups independently within each opponent style."""
    rng = np.random.default_rng(seed)
    counts = sum(map(len, samples.values()))
    draws = sum(rng.choice(values, size=(10_000, len(values))).sum(axis=1)
                for values in samples.values())
    return {"mean_delta": sum(map(sum, samples.values())) / counts,
            "ci95": np.quantile(draws / counts, (.025, .975)).tolist(), "group_deltas": samples}


def _check_pair(first, second, swap):
    if first["style"] != second["style"]:
        raise ValueError("配对对手不一致")
    for a, b in zip(first["games"], second["games"]):
        info = ("map_seed", "contestant_seed", "opponent_seed", "game_no")
        if any(a[key] != b[key] for key in info) or a["side"] != (b["side"] ^ swap):
            raise ValueError("配对地图、随机数、局号或出生侧不一致")


def _group_deltas(rows, deltas):
    if len(rows) % 2:
        raise ValueError("每个配对组必须有两个换边对照")
    groups = {}
    for index in range(0, len(rows), 2):
        first, second = rows[index:index + 2]
        _check_pair(first, second, 1)
        groups.setdefault(first["style"], []).append(fmean(deltas[index:index + 2]))
    return groups


def paired_pool_series(candidate, incumbent, seed):
    deltas = []
    for new, old in zip(candidate, incumbent, strict=True):
        _check_pair(new, old, 0)
        deltas.append(new["series_points"] - old["series_points"])
    series_groups = _group_deltas(candidate, deltas)
    return paired_bootstrap(series_groups, seed)


def paired_series(candidate, incumbent, direct, seed):
    direct_groups = _group_deltas(direct, [row["series_points"] - .5 for row in direct])
    return (paired_pool_series(candidate, incumbent, seed),
            paired_bootstrap(direct_groups, seed + 1))


def paired_games(candidate, incumbent, seed):
    deltas = []
    for new, old in zip(candidate, incumbent, strict=True):
        _check_pair(new, old, 0)
        a, = new["games"]
        b, = old["games"]
        deltas.append(a["result"]["scores"][a["side"]]["total"] - b["result"]["scores"][b["side"]]["total"])
    return paired_bootstrap(_group_deltas(candidate, deltas), seed)
