import pytest
from seedcup.game import generate_map
from seedcup.game.bots import BaselineBot
from seedcup.game.runner import play

from tank.simulation import FrameEnvironment, run_match


def test_run_match_uses_the_official_settlement_and_replay():
    official = play(generate_map(42), (BaselineBot(), BaselineBot()), game_no=0)
    match = run_match(BaselineBot(), BaselineBot(), 42, 0, replay=True)
    result = official.result_summary()

    assert match["result"] == result
    assert match["scores"] == [score["total"] for score in result["scores"]]
    assert match["winner"] == result["winner"]
    assert match["replay"]["result"] == official.result.to_dict()
    assert match["replay"]["frames"][0][0] == 0
    assert match["replay"]["frames"][-1][:-1] == official.snapshot()
    assert run_match(BaselineBot(), BaselineBot(), 42, 0) == {
        key: value for key, value in match.items() if key != "replay"
    }


def test_interleaved_environments_deliver_each_final_frame_before_reset():
    environments = [FrameEnvironment(), FrameEnvironment()]
    bots = [(BaselineBot(), BaselineBot()) for _ in environments]
    frames = [env.reset(seed=42, game_no=game_no) for game_no, env in enumerate(environments)]
    for env, (red, blue) in zip(environments, bots, strict=True):
        red.on_match_start(env.match_info("red"))
        blue.on_match_start(env.match_info("blue"))

    while not all(frame.terminated for frame in frames):
        for index, (env, (red, blue)) in enumerate(zip(environments, bots, strict=True)):
            frame = frames[index]
            if not frame.terminated:
                red_obs, blue_obs = frame.observations
                frames[index] = env.step(red.decide(red_obs), blue.decide(blue_obs))

    assert frames[0].result["ticks"] != frames[1].result["ticks"]
    for game_no, (env, frame) in enumerate(zip(environments, frames, strict=True)):
        official = play(generate_map(42), (BaselineBot(), BaselineBot()), game_no=game_no)
        assert frame.result == official.result_summary()
        assert frame.scores == [score["total"] for score in official.result.scores]
        assert frame.reason == official.result.reason
        assert frame.observations == (official.observe(0), official.observe(1))
        assert not frame.truncated
        opening = env.reset(seed=42, game_no=game_no)
        assert opening.match_id != frame.match_id
        assert opening.observations[0]["frame"] == 0
        assert frame.observations[0]["frame"] == official.tick


def test_interruption_preserves_current_observations_and_leaves_scores_unset():
    action = {"forward": 0, "turn": 0, "fire": False}
    env = FrameEnvironment()
    opening = env.reset(seed=42, game_no=0)
    advanced = env.step(action, action)
    with pytest.raises(RuntimeError, match="当前对局仍在进行"):
        env.reset(seed=42, game_no=0)
    ended = env.truncate("sample_limit")

    assert ended.match_id == opening.match_id
    assert ended.observations == advanced.observations
    assert ended.reason == "sample_limit"
    assert ended.truncated
    assert not ended.terminated
    assert ended.result is None
    assert ended.scores is None
    with pytest.raises(RuntimeError, match="当前对局已结束"):
        env.step(action, action)
    assert env.reset(seed=42, game_no=0).match_id != opening.match_id
