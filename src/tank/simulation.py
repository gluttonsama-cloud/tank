"""本地整局评测和训练共用的官方仿真适配。"""

from dataclasses import dataclass
from uuid import uuid4

from seedcup import Bot
from seedcup.game import SIDES, Game, GameMap, generate_map, parse_action, rules
from seedcup.game.replay import ReplayRecorder
from seedcup.game.runner import match_info, play


_SIDE_NAMES = ("红方", "蓝方")


def _official_map(seed: int, game_no: int) -> GameMap:
    if type(seed) is not int:
        raise ValueError("seed 必须为整数")
    if type(game_no) is not int or game_no < 0:
        raise ValueError("game_no 必须为非负整数")
    if rules.VARIANT:
        raise ValueError(f"工作台使用正式规则，当前 SDK 规则变体为 {rules.VARIANT}")
    return generate_map(seed)


def _totals(result: dict) -> list[float]:
    return [score["total"] for score in result["scores"]]


def run_match(red: Bot, blue: Bot, seed: int, game_no: int, *, replay: bool = False) -> dict:
    """执行一局 SDK 本地比赛，返回红蓝总分、结算和可选的官方格式回放。"""
    game_map = _official_map(seed, game_no)
    recorder = None
    if replay:
        recorder = ReplayRecorder(game_map, {
            "id": f"local-{seed}-{game_no}",
            "kind": "local",
            "public": True,
            "red": {"name": _SIDE_NAMES[0]},
            "blue": {"name": _SIDE_NAMES[1]},
            "round": None,
            "game": game_no,
            "table": None,
            "room": None,
        })
    game = play(game_map, (red, blue), _SIDE_NAMES, recorder=recorder, game_no=game_no)
    result = game.result_summary()
    match = {"scores": _totals(result), "winner": result["winner"], "result": result}
    if recorder is not None:
        assert game.result is not None
        match["replay"] = recorder.to_dict(game.result.to_dict())
    return match


@dataclass(frozen=True)
class Frame:
    match_id: str
    observations: tuple[dict, dict]
    terminated: bool = False
    truncated: bool = False
    reason: str | None = None
    result: dict | None = None

    @property
    def scores(self) -> list[float] | None:
        return _totals(self.result) if self.result is not None else None


class FrameEnvironment:
    """一例一局。先交付终局或中断状态，再显式开启下一局。"""

    def __init__(self) -> None:
        self._game: Game | None = None
        self._interruption: str | None = None
        self._game_no: int
        self._match_id: str

    def reset(self, *, seed: int, game_no: int) -> Frame:
        if self._game is not None and not self._game.over and self._interruption is None:
            raise RuntimeError("当前对局仍在进行，请先完成对局或调用 truncate()")
        self._game = Game(_official_map(seed, game_no))
        self._game_no = game_no
        self._match_id = uuid4().hex
        self._interruption = None
        return self._frame()

    def match_info(self, side: str) -> dict:
        """返回指定一侧 Bot 的官方开局信息。"""
        game = self._require_game()
        index = SIDES.index(side)
        return match_info(game.map, index, _SIDE_NAMES[1 - index], game=self._game_no)

    def step(self, red_action: dict, blue_action: dict) -> Frame:
        game = self._require_active_game()
        actions = []
        for side, raw_action in enumerate((red_action, blue_action)):
            action, bad = parse_action(raw_action)
            game.violations[side] += bad
            actions.append(action)
        game.step(actions)
        return self._frame()

    def truncate(self, reason: str) -> Frame:
        self._require_active_game()
        if not reason.strip():
            raise ValueError("reason 不能为空")
        self._interruption = reason
        return self._frame()

    def _require_game(self) -> Game:
        if self._game is None:
            raise RuntimeError("请先调用 reset() 开局")
        return self._game

    def _require_active_game(self) -> Game:
        game = self._require_game()
        if game.over or self._interruption is not None:
            raise RuntimeError("当前对局已结束，请调用 reset() 开始下一局")
        return game

    def _frame(self) -> Frame:
        game = self._require_game()
        result = game.result_summary() if game.over else None
        return Frame(
            match_id=self._match_id,
            observations=(game.observe(0), game.observe(1)),
            terminated=game.over,
            truncated=self._interruption is not None,
            reason=result["reason"] if result is not None else self._interruption,
            result=result,
        )
