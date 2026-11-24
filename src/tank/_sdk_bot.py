"""Child entry point for class-based Python strategies."""
import json
import os
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path

from seedcup.client.loader import load_bot


def main():
    wire = sys.stdout
    with redirect_stdout(sys.stderr):
        if "TANK_STRATEGY_SEED" in os.environ:
            seed = int(os.environ["TANK_STRATEGY_SEED"])
            random.seed(seed)
            import numpy as np
            np.random.seed(seed % (2**32))
        sys.path.insert(0, str(Path.cwd()))
        bot = load_bot(sys.argv[1])()
        for line in sys.stdin:
            msg = json.loads(line)
            if msg["type"] == "match_start":
                bot.on_match_start(msg)
            elif msg["type"] == "obs":
                result = bot.decide(msg)
                print(json.dumps({**result, "type": "action", "frame": msg["frame"]},
                                 allow_nan=False), file=wire, flush=True)
            else:
                bot.on_match_end(msg["result"])
                break


if __name__ == "__main__":
    main()
