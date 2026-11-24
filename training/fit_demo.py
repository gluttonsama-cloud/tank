"""Fit a small linear behavior cloning policy from captured demonstrations."""

import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np

from demo_policy import features


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def main(request_path):
    folder = request_path.parent
    request = json.loads(request_path.read_text(encoding="utf-8"))
    params = request["params"]
    unknown = params.keys() - {"ridge", "fire_threshold"}
    if unknown:
        raise ValueError(f"未知训练参数 {', '.join(sorted(unknown))}")
    ridge = params.get("ridge", 0.01)
    threshold = params.get("fire_threshold", 0.5)
    if type(ridge) not in (int, float) or not math.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge 必须为非负有限数值")
    if type(threshold) not in (int, float) or not 0 <= threshold <= 1:
        raise ValueError("fire_threshold 必须在 0 到 1 之间")

    observations, actions = [], []
    for source in request["inputs"]:
        demonstration = json.loads((folder / source["path"]).read_text(encoding="utf-8"))
        if demonstration["schema_version"] != 1:
            raise ValueError(f"不支持的示范格式 {source['source']}")
        teacher = request["config"]["teacher_snapshot"]
        if teacher and "snapshots" not in demonstration:
            raise ValueError("这份示范没有策略版本记录，无法核对指定教师")
        for sample in demonstration["samples"]:
            if teacher and demonstration["snapshots"][sample["side"]] != teacher:
                continue
            observations.append(features(sample["observation"]))
            move = sample["action"]
            actions.append((move["forward"], move["turn"], float(move["fire"])))
    if not observations:
        raise ValueError("输入中没有符合训练配置的示范样本")
    x, y = np.asarray(observations), np.asarray(actions)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("示范观测或动作包含非有限数值")
    regularizer = np.sqrt(ridge) * np.eye(x.shape[1])
    regularizer[0, 0] = 0  # Do not penalize the intercept.
    weights, _, _, _ = np.linalg.lstsq(
        np.vstack((x, regularizer)), np.vstack((y, np.zeros((x.shape[1], y.shape[1])))), rcond=None,
    )
    metrics = {"samples": len(x), "train_mse": float(np.mean((x @ weights - y) ** 2))}
    output = folder / request["output"]
    strategy = output / "strategy"
    strategy.mkdir()
    shutil.copyfile(Path(__file__).with_name("demo_policy.py"), strategy / "bot.py")
    np.save(strategy / "weights.npy", weights, allow_pickle=False)
    _write(strategy / "submission.json", {"runtime": "python", "entrypoint": "bot.py:DemoPolicy"})
    _write(strategy / "policy.json", {"fire_threshold": threshold})
    _write(output / "result.json", {"weights": "strategy/weights.npy", "step": 1,
                                    "strategy": "strategy", "metrics": metrics})
    print(f"示范样本 {metrics['samples']} 条，训练均方误差 {metrics['train_mse']:.6f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
