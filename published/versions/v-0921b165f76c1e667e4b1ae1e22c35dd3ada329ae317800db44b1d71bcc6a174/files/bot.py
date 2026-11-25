from pathlib import Path
import json
from boss_specialist_policy import load_strategy
from boss_local_policy import make_local_policy

ROOT = Path(__file__).resolve().parent
Specialist = make_local_policy(load_strategy(ROOT / "base"), load_strategy(ROOT / "replica"),
                               json.loads((ROOT / "weights.json").read_text(encoding="utf-8"))["params"])
