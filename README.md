# 种子杯策略实验工作台

三人团队共用的策略训练、批量评测和成绩比较工具。

- [开发文档](./工作台开发文档.md)包含模块、操作接口和开发顺序。
- [协作规则](./AGENTS.md)规定调研、代码、界面和 Git 工作方式。

## 官方 SDK

将比赛平台的 [Python 安装包](https://tank.dian.org.cn/sdk/seedcup-0.4.3-py3-none-any.whl) 下载到 `resource/`，按仓库里的 [SHA-256](./resource/SHA256SUMS) 核对版本。示例和手册另见[官方离线资料包](https://tank.dian.org.cn/sdk/seedcup-sdk-0.4.3.zip)，赛方可独立更新其中的文档。

## 首次启动（Windows）

在仓库根目录用 PowerShell 运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .\resource\seedcup-0.4.3-py3-none-any.whl
python -m pip install -e '.[dev]'
git config --local core.hooksPath .githooks
tank web --open
```

首页按项目展示策略类型、官网成绩和最近本地实测，历史版本可展开。从项目开始测试会固定当前版本。GPT 用 `tank project list --json` 查询项目，用 `tank eval start --strategy starter --suite full --json` 固定源码并开始完整测试。训练入口见[训练说明](./training/README.md)。

添加项目时，GPT 在 `projects/` 登记显示名、类型、策略目录和训练脚本。一个项目可对应多个版本；类型支持规则、规划、强化学习、模仿学习、混合及其他。可参考[规则策略](./projects/starter.json)和[模仿学习模型](./projects/demo-policy.json)。

## 官网练习

在启动工作台的 PowerShell 窗口设置队伍账号。密码通过系统凭据提示框输入：

```powershell
$teamLogin = Get-Credential -Message '比赛官网队伍账号'
$env:SEEDCUP_USERNAME = $teamLogin.UserName
$env:SEEDCUP_PASSWORD = $teamLogin.GetNetworkCredential().Password
tank web --open
```

在「策略」选择快照后点「在线练习」。

工作台读取本机队伍账号，打开首页时自动更新官网成绩，每五分钟至多请求一次。GPT 可用 `tank online refresh --json` 更新。官网显示历史最高的 40 局总分。本地显示实际计入的局数和总分，只比较同一方案的成绩。

## 分摊批量测试

协调机用 `tank eval start --candidates <快照ID> --suite full --shards 8 --prepare-only` 建批次，再用 `tank sync publish <批次ID> --push` 分享方案和快照。工作机用 `tank sync fetch <批次ID>` 取回。各机使用同一版工作台，设置相同的环境变量 `TANK_LAN_TOKEN`。

- 协调机运行 `tank lan serve <批次ID> --host 0.0.0.0 --port 8766`。
- 工作机运行 `tank lan work http://<协调机校园网IP>:8766 --max-shards 8` 领取分片并回传。
- 单机离线运行用 `tank eval resume <批次ID> --shard 0`。交接时由 GPT 执行 `tank sync publish <批次ID> --push`；队友的 GPT 用 `tank sync list --remote` 查找已共享批次，用 `tank sync fetch <批次ID>` 合并成绩。

## 自动检查

Git 提交前由本地钩子运行 `tank check --staged`。GitHub Actions 在 PR 和推送到 `main` 时运行同一套脚本。修改网页脚本时需要 Node.js 20 做语法检查。日常编辑和保存策略快照不会触发 CI。需要手动查当前改动时运行 `tank check --json`。
