import shutil
from pathlib import Path

import pytest

from tank import versions
from tank.webapp import overview
from tank.workspace import read_json, write_json


def test_snapshot_changes_and_reuse_integrity(tmp_path):
    source = tmp_path / "strategies/starter"
    shutil.copytree(Path(__file__).resolve().parents[1] / "strategies/starter", source)
    first = versions.create(tmp_path, "starter")
    (source / "settings.json").write_text("{}")
    second = versions.create(tmp_path, "starter")
    difference = versions.compare(tmp_path, first["id"], second["id"])
    assert difference["added"] == ["settings.json"]
    summary = overview(tmp_path)
    assert all("files" not in version for version in summary["versions"])
    assert "training" not in summary
    metadata = versions.snapshot_data(tmp_path, second["id"]).parent / "version.json"
    write_json(metadata, read_json(metadata) | {"entrypoint": "wrong.py"})
    with pytest.raises(ValueError, match="校验失败"):
        versions.create(tmp_path, "starter")
