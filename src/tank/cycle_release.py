"""Official submission, activation and comparable-opponent monitoring."""

import time
from collections import defaultdict
from statistics import fmean

import numpy as np

from . import cycle_store as store
from . import online
from .eval_store import shard_lock
from .workspace import read_json


class _SubmissionPaused(online.OnlineError):
    def __init__(self, path):
        super().__init__("策略循环已暂停")
        prefix = online.SUBMISSIONS + "/"
        self.submission = int(path.removeprefix(prefix).removesuffix("/evaluate")) if path.endswith("/evaluate") else None


class _SubmissionClient:
    """Check the durable pause flag at each upload or queue request."""

    def __init__(self, client, root, line):
        self.client = client
        self.user = client.user
        self.root = root
        self.line = line

    def request(self, method, path, **kwargs):
        if method == "POST" and not store.enabled(self.root, self.line):
            raise _SubmissionPaused(path)
        return self.client.request(method, path, **kwargs)


def official_score(root, version, *, line):
    """Serialize both lanes' uploads and retain verified terminal results locally."""
    if not store.enabled(root, line):
        return {"state": "paused"}
    with shard_lock(store.folder(root), "submission") as acquired:
        if not acquired:
            return {"state": "waiting"}
        client = _SubmissionClient(online.OnlineClient.login(root), root, line)
        namespace = f"official-{client.user['id']}"
        saved = store.get(root, namespace, version, {})
        if saved.get("state") in ("completed", "failed"):
            return saved
        try:
            if "submission" in saved:
                item = online.status(root, saved["submission"], client=client)
                if not item.get("evaluation"):
                    # Resume an archived submission without creating another upload.
                    package = {"version_id": version, "path": item["package"]}
                    item = online._submit_evaluation(root, client, item, package,
                                                     client.request("GET", online.SUBMISSIONS))
            else:
                board = online.refresh(root, client=client)
                known = next((r for r in board["submissions"] if r["version_id"] == version), None)
                if known:
                    result = {"state": "completed", "submission": known["id"], "score": known["score"]}
                    store.put(root, namespace, **{version: result})
                    return result
                item = online.submit(root, version, client=client)
        except _SubmissionPaused as error:
            result = {"state": "paused"}
            if error.submission is not None:
                result["submission"] = error.submission
                store.put(root, namespace, **{version: result})
            return result

        score = online._complete_score(item)
        evaluation = item.get("evaluation") or {}
        result = {"submission": item["id"], "state": "waiting"}
        if score:
            result.update(state="completed", score=score["score"])
        elif evaluation.get("status") in ("completed", "failed", "cancelled", "error"):
            reason = "官网评测得分或明细校验失败" if evaluation["status"] == "completed" else "官网评测失败"
            result.update(state="failed", reason=evaluation.get("error") or reason)
        store.put(root, namespace, **{version: result})
        return result


def comparable_series(series, previous, current, since):
    by_opponent = defaultdict(lambda: [[], []])
    for s in series:
        if s["status"] != "finished" or s.get("outcome") not in (-1,0,1):
            continue
        version = s.get("version_id")
        if version == current and s["started_at"] >= since:
            arm = 1
        elif version == previous and s["ended_at"] <= since:
            arm = 0
        else:
            continue
        by_opponent[s["opponent_id"]][arm].append((s["ended_at"],(s["outcome"]+1)/2))
    paired = []
    for old,new in by_opponent.values():
        if old and new:
            paired.append(([v for _,v in sorted(old)[-max(3,len(new)):]], [v for _,v in new]))
    n = sum(len(new) for _,new in paired)
    result = {"comparable":n,"opponents":len(paired),"rollback":False}
    if n < 12 or len(paired) < 4:
        return result
    rng = np.random.default_rng(11264333)
    strata = []
    for old,new in paired:
        strata.append(rng.choice(new,(5000,len(new))).mean(axis=1)-rng.choice(old,(5000,len(old))).mean(axis=1))
    idx = rng.integers(len(strata),size=(5000,len(strata)))
    draws = np.array(strata)[idx,np.arange(5000)[:,None]].mean(axis=1)
    ci = np.quantile(draws,[.025,.975]).tolist()
    return result | {"delta":fmean(fmean(n)-fmean(o) for o,n in paired),"ci95":ci,"rollback":ci[1]<0}


def review_activation(root, live):
    from . import cycle_ladder
    review = store.get(root, "pvp", "review")
    if not review:
        return None
    if not live["consistent"] or live["active_version"] is None:
        return "等待核实官网生效快照"

    previous, current = review["previous"], review["current"]
    target = review["previous_submission"]
    def clear_review(**fields):
        if store.get(root, "pvp", "selection_target") == target:
            fields["selection_target"] = None
        store.put(root, "pvp", review=None, **fields)

    def rollback_complete():
        clear_review(last_change={"reason": "可比天梯战绩下降，已回滚", "version": previous, "at": time.time()})

    if live["active_version"] == previous and live["active_submission"] == target and review.get("rolling_back"):
        rollback_complete()
        return None
    if live["active_version"] != current or live["pending_submission"] not in (None, target):
        clear_review()
        return None

    archived = (read_json(path) for path in (store.folder(root) / "ladder/series").glob("*.json"))
    records = (r for r in archived if r["team_id"] == live["team_id"])
    stats = comparable_series(records, previous, current, review["since"])
    if stats["rollback"] or review.get("rolling_back"):
        if not store.enabled(root, "pvp"):
            return "回滚已暂停"
        store.put(root, "pvp", review=review | {"rolling_back": True}, monitor=stats, selection_target=target)
        actual = cycle_ladder.request_activation(root, previous, target, expected_active_version=current,
                                                 allow_activation=lambda: store.enabled(root, "pvp"))
        activation = actual["activation"]
        state = activation["status"]
        if state == "active":
            rollback_complete()
            return None
        if state in ("active_changed", "changed", "superseded"):
            clear_review()
            return None
        interrupted = state in ("series_void", "series_cancelled", "series_expired")
        if state in ("rejected", "cancelled", "series_not_activated") or interrupted and not activation["can_retry"]:
            clear_review(rollback_failure=activation)
            return "官网拒绝回滚" if state == "rejected" else "官网回滚未生效"
        if state == "paused":
            return "回滚已暂停"
        return "正在恢复上一主力"
    store.put(root, "pvp", monitor=stats)
    return None
