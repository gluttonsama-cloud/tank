# 种子杯策略工作台

团队共用的策略训练、批量评测和成绩比较工具。Windows 安装完成后，双击 [启动工作台.pyw](./启动工作台.pyw)。重复启动会打开已有工作台。

- [使用手册](./使用手册.md) 分为队员和 AI 两部分。
- [AGENTS.md](./AGENTS.md) 规定 AI 的开发、检查和 Git 协作方式。
- [开发文档](./docs/工作台开发文档.md) 记录模块和评价口径。[训练接入](./docs/训练接入.md) 提供环境与模型示例。

## 首次安装（Windows）

安装 Python 3.12 和 Git，将[官方 SDK 安装包](https://tank.dian.org.cn/sdk/seedcup-0.4.3-py3-none-any.whl) 放入 `resource/`。在项目根目录用 PowerShell 运行：

```powershell
$wheel = Get-Item .\resource\seedcup-0.4.3-py3-none-any.whl
$expected = ((Get-Content .\resource\SHA256SUMS) -split '\s+')[0]
if ((Get-FileHash $wheel.FullName -Algorithm SHA256).Hash -ne $expected) { throw 'SDK 校验失败' }
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install $wheel.FullName
.\.venv\Scripts\python -m pip install -e '.[dev]'
git config --local core.hooksPath .githooks
```

修改网页脚本时，自动检查还需要 Node.js 20。[官方离线资料包](https://tank.dian.org.cn/sdk/seedcup-sdk-0.4.3.zip) 包含比赛示例和选手手册。
