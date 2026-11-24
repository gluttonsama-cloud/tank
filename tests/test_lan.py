import threading
import time
from contextlib import contextmanager

import pytest

from tank import evaluation, lan

from test_evaluation import fake_match, workspace


TOKEN = b"seedcup test only"


@contextmanager
def service(coordinator, port=0):
    with lan._server(coordinator, "127.0.0.1", port) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            thread.join()


def test_loopback_executes_official_shards_from_independent_workspaces(tmp_path):
    root, candidate = workspace(tmp_path / "coordinator")
    worker, _ = workspace(tmp_path / "worker")
    run_id = evaluation.start(
        root, [candidate], "baseline", "quick", shards=2, mode="direct", prepare_only=True,
    )["run_id"]
    coordinator = lan.Coordinator(root, run_id, TOKEN, 5)

    with service(coordinator) as url:
        with pytest.raises(ValueError, match="未通过认证"):
            lan.work(worker, url, name="wrong", token=b"wrong")
        receipt = lan.work(worker, url, name="worker", max_shards=2, token=TOKEN)

    assert {item["shard"] for item in receipt["receipts"]} == {0, 1}
    assert evaluation.status(root, run_id)["completed"] == 4
    assert evaluation.status(root, run_id)["status"] == "completed"
    assert evaluation.compare(root, run_id)["rows"][0]["completed_pairs"] == 2


def test_expired_worker_and_duplicate_upload_are_merged(tmp_path, monkeypatch):
    root, candidate = workspace(tmp_path / "coordinator")
    first, _ = workspace(tmp_path / "first")
    second, _ = workspace(tmp_path / "second")
    run_id = evaluation.start(root, [candidate], "baseline", "one", prepare_only=True)["run_id"]
    now = [time.time()]
    coordinator = lan.Coordinator(root, run_id, TOKEN, 5, clock=lambda: now[0])
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)

    with service(coordinator) as url:
        client = lan.Client(url, TOKEN)
        claim = client.request("/claim", {"worker": "first"})
        first_assignment = claim["assignment"]
        assert client.request("/claim", {"worker": "second"})["assignment"] is None
        now[0] += 6
        recovered = lan.work(second, url, name="second", token=TOKEN)
        assert recovered["status"] == "completed"
        assert recovered["receipts"][0]["shard"] == first_assignment["shard"]

        outbox = first / "outbox"
        outbox.mkdir()
        lan._process_assignment(first, client, "first", claim, outbox)
        late = lan._submit_outbox(first, client, "first", outbox)
        assert late[0]["status"]["completed"] == 4
        assert evaluation.status(root, run_id)["completed"] == 4


def test_completed_shard_survives_coordinator_disconnect(tmp_path, monkeypatch):
    root, candidate = workspace(tmp_path / "coordinator")
    worker, _ = workspace(tmp_path / "worker")
    run_id = evaluation.start(root, [candidate], "baseline", "one", prepare_only=True)["run_id"]
    coordinator = lan.Coordinator(root, run_id, TOKEN, 5)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    send = lan.Client.request

    def disconnect_before_upload(self, action, data):
        if action == "/submit":
            raise ConnectionError("模拟断线")
        return send(self, action, data)

    with service(coordinator) as url:
        monkeypatch.setattr(lan.Client, "request", disconnect_before_upload)
        with pytest.raises(ConnectionError, match="模拟断线"):
            lan.work(worker, url, name="worker", token=TOKEN)
        assert evaluation.status(worker, run_id)["completed"] == 4
        assert evaluation.status(root, run_id)["completed"] == 0
    restarted = lan.Coordinator(root, run_id, TOKEN, 5)
    monkeypatch.setattr(lan.Client, "request", send)
    with service(restarted, int(url.rsplit(":", 1)[1])) as new_url:
        recovered = lan.work(worker, new_url, name="worker", token=TOKEN)
        assert recovered["status"] == "completed"
        assert recovered["receipts"][0]["status"]["completed"] == 4
