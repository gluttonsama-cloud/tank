from pathlib import Path
from runpy import run_path
from threading import Thread

import pytest

from tank.webapp import WorkbenchHTTP


def test_launcher_reuses_only_its_own_workspace(tmp_path):
    launcher = Path(__file__).resolve().parents[1] / "启动工作台.pyw"
    running = run_path(str(launcher))["running"]
    with WorkbenchHTTP(("127.0.0.1", 0), tmp_path) as server:
        worker = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            assert running(server.server_port, tmp_path)
            with pytest.raises(RuntimeError, match="另一个工作台"):
                running(server.server_port, tmp_path / "other")
        finally:
            server.shutdown()
            worker.join()
