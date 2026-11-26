from unittest.mock import Mock

import pytest

from tank import cycle_ladder, cycle_release, cycle_store, online


def _enabled(root, line, enabled):
    with cycle_store.database(root) as db:
        db.execute("INSERT OR REPLACE INTO lines VALUES (?, ?)", (line, int(enabled)))


def _client(monkeypatch):
    client = Mock()
    client.user = {"id": 57}
    monkeypatch.setattr(online.OnlineClient, "login", Mock(return_value=client))
    monkeypatch.setattr(online, "refresh", Mock(return_value={"submissions": []}))
    return client


def test_two_lines_use_one_formal_evaluation_for_the_same_snapshot(tmp_path,monkeypatch):
    _client(monkeypatch)
    for line in ("pvp","pve"):
        _enabled(tmp_path,line,True)
    score={"status":"completed","score":8000,
           "breakdown":{"kill":2000,"damage":2000,"survive":2000,"time":2000}}
    submit=Mock(return_value={"id":12,"evaluation":score})
    monkeypatch.setattr(online,"submit",submit)
    first=cycle_release.official_score(tmp_path,"same",line="pve")
    assert first=={"state":"completed","submission":12,"score":8000}
    assert cycle_release.official_score(tmp_path,"same",line="pvp")==first
    submit.assert_called_once()


def test_paused_lane_preserves_archive_and_prevents_remote_evaluation(tmp_path, monkeypatch):
    client = _client(monkeypatch)
    evaluation_path = f"{online.SUBMISSIONS}/12/evaluate"

    def upload(root, version, *, client):
        client.request("POST", online.SUBMISSIONS)
        _enabled(root, "pve", False)
        client.request("POST", evaluation_path)
        pytest.fail("the evaluation POST should be stopped by pause")

    submit = Mock(side_effect=upload)
    monkeypatch.setattr(online, "submit", submit)
    _enabled(tmp_path, "pve", True)
    assert cycle_release.official_score(tmp_path, "version", line="pve") == {"state": "paused", "submission": 12}
    client.request.assert_called_once_with("POST", online.SUBMISSIONS)
    assert cycle_store.get(tmp_path, "official-57", "version") == {"state": "paused", "submission": 12}
    assert cycle_release.official_score(tmp_path, "version", line="pve") == {"state": "paused"}

    _enabled(tmp_path, "pve", True)
    monkeypatch.setattr(online, "status", Mock(return_value={"id": 12, "package": "file.zip", "evaluation": None}))

    def enqueue(root, client, item, package, state):
        assert package == {"version_id": "version", "path": "file.zip"}
        client.request("POST", evaluation_path)
        return item | {"evaluation": {"status": "queued"}}

    monkeypatch.setattr(online, "_submit_evaluation", enqueue)
    assert cycle_release.official_score(tmp_path, "version", line="pve") == {"state": "waiting", "submission": 12}
    assert submit.call_count == 1
    assert client.request.call_args.args == ("POST", evaluation_path)


def test_definitively_invalid_terminal_score_fails_once(tmp_path, monkeypatch):
    _client(monkeypatch)
    _enabled(tmp_path, "pvp", True)
    submit = Mock(return_value={"id": 20, "evaluation": {"status": "completed", "score": 9000}})
    monkeypatch.setattr(online, "submit", submit)
    result = cycle_release.official_score(tmp_path, "version", line="pvp")
    assert result == {"submission": 20, "state": "failed", "reason": "官网评测得分或明细校验失败"}
    assert cycle_release.official_score(tmp_path, "version", line="pvp") == result
    assert submit.call_count == 1


def _review():
    return {"previous": "old", "previous_submission": 11, "current": "new", "since": 100, "rolling_back": True}


def _live(**values):
    return {"consistent": True, "active_version": "new", "active_submission": 12, "pending_submission": None} | values


def test_unverified_or_paused_live_state_preserves_rollback(tmp_path, monkeypatch):
    store = cycle_store
    review = _review()
    store.put(tmp_path, "pvp", review=review, selection_target=11)
    activation = Mock()
    monkeypatch.setattr(cycle_ladder, "request_activation", activation)
    assert cycle_release.review_activation(tmp_path, _live(consistent=False, active_version=None)) == "等待核实官网生效快照"
    assert cycle_release.review_activation(tmp_path, _live()) == "回滚已暂停"
    assert store.get(tmp_path, "pvp", "review") == review
    assert store.get(tmp_path, "pvp", "selection_target") == 11
    activation.assert_not_called()


@pytest.mark.parametrize("finished", ["active", "active_changed", "rejected", "superseded",
                                      "series_not_activated", "series_cancelled"])
def test_rollback_tracks_its_selection_until_authoritative_completion(tmp_path, monkeypatch, finished):
    _enabled(tmp_path, "pvp", True)
    cycle_store.put(tmp_path, "pvp", review=_review())

    def activate(root, version, submission, *, expected_active_version, allow_activation):
        assert (version, submission, expected_active_version) == ("old", 11, "new")
        assert cycle_store.get(root, "pvp", "selection_target") == 11
        assert allow_activation()
        _enabled(root, "pvp", False)
        assert not allow_activation()
        return {"activation": {"status": finished, "can_retry": False}}

    monkeypatch.setattr(cycle_ladder, "request_activation", activate)
    outcome = cycle_release.review_activation(tmp_path, _live())
    assert cycle_store.get(tmp_path, "pvp", "review") is None
    assert cycle_store.get(tmp_path, "pvp", "selection_target") is None
    if finished == "active":
        assert cycle_store.get(tmp_path, "pvp", "last_change")["version"] == "old"
    if finished == "rejected":
        assert outcome == "官网拒绝回滚"
    if finished in ("series_not_activated", "series_cancelled"):
        assert outcome == "官网回滚未生效"
        assert cycle_store.get(tmp_path, "pvp", "rollback_failure")["status"] == finished


def test_interrupted_rollback_stays_selected_if_official_pending_matches(tmp_path, monkeypatch):
    _enabled(tmp_path, "pvp", True)
    cycle_store.put(tmp_path, "pvp", review=_review(), selection_target=11)
    monkeypatch.setattr(cycle_ladder, "request_activation",
                        Mock(return_value={"activation": {"status": "series_void", "can_retry": True}}))
    assert cycle_release.review_activation(tmp_path, _live(pending_submission=11)) == "正在恢复上一主力"
    assert cycle_store.get(tmp_path, "pvp", "selection_target") == 11


def test_comparable_series_requires_replicated_opponents_and_detects_decline():
    rows = []
    for opponent in range(4):
        for repeat in range(3):
            rows.extend([
                {"status": "finished", "opponent_id": opponent, "version_id": "old", "started_at": repeat,
                 "ended_at": repeat + 1, "outcome": 1},
                {"status": "finished", "opponent_id": opponent, "version_id": "new", "started_at": 101 + repeat,
                 "ended_at": 102 + repeat, "outcome": -1},
            ])
    smaller = cycle_release.comparable_series(rows[:-1], "old", "new", 100)
    assert smaller == {"comparable": 11, "opponents": 4, "rollback": False}
    result = cycle_release.comparable_series(rows, "old", "new", 100)
    assert result["comparable"] == 12
    assert result["ci95"] == [-1.0, -1.0]
    assert result["rollback"] is True
