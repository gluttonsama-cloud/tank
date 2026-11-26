"""Launch one resumable experiment without tying its lifetime to the CLI."""

import os
import sys
import time
from pathlib import Path

from . import cycle_store as store
from .eval_store import shard_lock
from .workspace import project_root, read_json, write_json


def work(root: Path, line: str, number: int, verify=False):
    from training.cycle_train import train

    out = store.round_dir(root, line, number)
    with shard_lock(out, "fit") as acquired:
        if not acquired or not store.enabled(root, line) or ((out / "result.json").exists() and not verify):
            return
        request = read_json(out / "request.json")
        try:
            write_json(out / "process.json", {"pid": os.getpid(), "at": time.time()})
            if verify:
                from training.cycle_train import revalidate

                revalidate(root, out, workers=read_json(out / "workers.json")["workers"])
                return
            train(
                root,
                line,
                request["incumbent"],
                number,
                read_json(out / "workers.json")["workers"],
                out,
                recipient_incumbent=request["recipient"],
            )
        except Exception as error:
            write_json(out / "failure.json", {"error": str(error), "at": time.time()})
            raise


if __name__ == "__main__":
    work(project_root(), sys.argv[1], int(sys.argv[2]), sys.argv[3:] == ["verify"])
