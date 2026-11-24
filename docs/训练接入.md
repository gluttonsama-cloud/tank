# 训练接入

## 跑一次示范训练

在已安装工作台的 PowerShell 中运行。`fit_demo.py` 读取示范并导出一个可评测的候选快照。

```powershell
$trainingSample = tank training collect --red baseline --json | ConvertFrom-Json
$trainingConfig = tank jobs configure --teacher baseline --opponent baseline=1 --seed 2026 --json | ConvertFrom-Json
$trainingJob = tank jobs start fit_demo.py --config $trainingConfig.config_id --input $trainingSample.path --json | ConvertFrom-Json
tank jobs status $trainingJob.job_id --json
```

在工作台的「策略」页查看训练状态。作业完成后，点击「测试此版本」可用产出的快照发起测试。

## 逐帧训练接口

训练脚本使用 `tank.simulation.FrameEnvironment`。本机示范、训练记录和中间权重放在项目的 `.local/training/`。

```python
from pathlib import Path

from tank.simulation import FrameEnvironment
from tank.training import TrainingStore
from tank.versions import load_bot

root = Path.cwd()
store = TrainingStore(root)
config_id = store.register_config(
    teacher_snapshot=None,
    opponent_pool={"baseline": 1},
    sampling_seed=2026,
)
env = FrameEnvironment()
frame, episode = store.start_episode(
    env, config_id, episode_index=0, seed=42, game_no=0, controlled_side="red",
)
opponent = load_bot(root, episode["opponent_snapshot"])
opponent.on_match_start(env.match_info("blue"))

while not frame.terminated and not frame.truncated:
    red_obs, blue_obs = frame.observations
    red_action = {"forward": 0.0, "turn": 0.0, "fire": False}  # 在此调用待训练策略
    frame = env.step(red_action, opponent.decide(blue_obs))

if frame.terminated:
    opponent.on_match_end(frame.result | {"side": "blue"})
    print(frame.match_id, frame.scores, frame.reason)
```

- `observations` 按红、蓝顺序提供 SDK 观测。地图网格在各侧的 `env.match_info(side)["map"]` 中。
- 比赛结算时，`frame.terminated` 为真，`frame` 包含终局观测、比分和官方结束原因。采样预算耗尽时调用 `env.truncate("sample_limit")` 交付截断帧，`frame.truncated` 为真，`frame.scores` 为 `None`。完成或截断后，再调用 `reset()` 或 `start_episode()` 开始新对局。
- 对手池的值是正整数权重。使用固定的 `sampling_seed` 和独立的 `episode_index` 可复现实际抽中的对手。自对弈时可将已有的本队策略快照加入对手池。
- `store.register_checkpoint(config_id, weights_path, step=100)` 登记权重路径和 SHA-256。参数 `snapshot_id` 可关联已导出的策略快照。通过 `read_checkpoint`、`read_config` 和 `read_episode` 按 ID 恢复检查点、教师和实际陪练记录。

## 采集示范

```python
from pathlib import Path

from seedcup.game.bots import BaselineBot

from tank.training import collect_demonstrations

collect_demonstrations(
    BaselineBot(), BaselineBot(), 42, 0,
    match_id="teacher-42-0", sides=("red",),
    output=Path(".local/training/demonstrations/teacher-42-0.json"),
)
```

可以采集 `red`、`blue` 或双方。每份文件保存各受控方的开局信息、逐帧 `samples` 和本局结算。样本用 `match_id`、`frame`、`side` 定位，`observation` 为实参，`action` 为 Bot 返回的三项指令。装填中发出的开火指令会记入示范。
