import json
import os
import tempfile
from pathlib import Path


def project_root() -> Path:
    start = Path(os.environ.get("TANK_ROOT", ".")).resolve()
    for folder in (start, *start.parents):
        if (folder / "resource/SHA256SUMS").is_file():
            return folder
    raise ValueError("请在工作台目录运行，或设置 TANK_ROOT。")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8",
                                     delete=False) as temp:
        try:
            json.dump(value, temp, ensure_ascii=False, indent=2, allow_nan=False)
            temp.write("\n")
        except BaseException:
            temp.close()
            Path(temp.name).unlink()
            raise
    os.replace(temp.name, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))
