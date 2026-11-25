from pathlib import Path
import json

from boss_specialist_policy import load_strategy, make_policy

ROOT = Path(__file__).resolve().parent
PARAMS = json.loads((ROOT / "weights.json").read_text(encoding="utf-8"))["params"]
from boss_forecast import make_predictor

PREDICTOR = make_predictor(json.loads((ROOT / "forecast.json").read_text(encoding="utf-8")))
BossSpecialist = make_policy(load_strategy(ROOT / "stock"), PARAMS, predictor=PREDICTOR)
