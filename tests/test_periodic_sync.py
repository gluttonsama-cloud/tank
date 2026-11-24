import shutil
import threading

from tank import catalog, evaluation, sync, versions
from tank.workspace import write_json
from tank.webapp import WorkbenchHTTP
from test_auto_sharing import repository
from test_evaluation import fake_match
from test_sharing import git


def test_service_receives_missed_and_later_fragments_without_browser(tmp_path, monkeypatch):
    first, _, remote = repository(tmp_path)
    second = tmp_path / "teammate"
    git(tmp_path, "clone", str(remote), str(second))
    source = first / "strategies/route-v2"
    shutil.copytree(first / "strategies/starter", source)
    (source / "settings.json").write_text("{}")
    candidate = versions.create(first, source.name)["id"]
    write_json(first / "projects/route.json", {"name": "路线规划", "kind": "planning",
                                              "sources": [source.name], "training": []})
    run = evaluation.start(first, [candidate], "baseline", "quick", prepare_only=True)
    run_id = run["run_id"]
    sync.push(first, run_id)
    monkeypatch.setattr(evaluation, "run_isolated", fake_match)
    monkeypatch.delenv("SEEDCUP_USERNAME", raising=False)
    fetches = []
    original = sync._fetch_refs
    def record(root, run_id, refs):
        fetches.append(set(refs))
        return original(root, run_id, refs)
    monkeypatch.setattr(sync, "_fetch_refs", record)
    assert sync.receive_updates(second) == [run_id]
    assert evaluation.status(second, run_id)["completed"] == 0
    evaluation.resume(first, run_id, background=False)
    sync.push(first, run_id)

    # Run the same loop as the launcher, without issuing a single HTTP request.
    with WorkbenchHTTP(("127.0.0.1", 0), second) as server:
        received = threading.Event()
        receive = sync.receive_updates
        def observe(root):
            updated = receive(root)
            if not updated:
                received.set()
            return updated
        monkeypatch.setattr(sync, "receive_updates", observe)
        worker = threading.Thread(target=server.sync_loop, args=(0.02,), daemon=True)
        worker.start()
        try:
            assert received.wait(30)
            assert server.team_sync["error"] is None
        finally:
            server.stop_sync.set()
            worker.join(10)

    assert len(fetches) == 2
    assert len(fetches[0]) == len(fetches[1]) == 1 and fetches[0].isdisjoint(fetches[1])
    assert evaluation.compare(second, run_id) == evaluation.compare(first, run_id)
    project = next(item for item in catalog.projects(second) if item["id"] == "route")
    assert (project["name"], project["kind"], project["sources"]) == ("路线规划", "planning", [])
    assert project["versions"][0]["local"]["latest"]["run_id"] == run_id
    assert not (second / "projects/route.json").exists()
    git(second, "diff", "--exit-code", "HEAD", "--", "strategies", "src")
