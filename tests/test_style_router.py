from seedcup.game import rules

from training.style_router import StyleDetector, StyleRouter, policy_match_info

CONFIG = {"latest_frame": 10, "consecutive_frames": 3, "min_drive": 0.18}


def observation(frame, x, *, toward_right=True):
    return {
        "frame": frame,
        "self": {"x": 16.0, "y": 12.0, "alive": True},
        "enemy": {"x": x, "y": 12.0, "heading": 0.0 if toward_right else 180.0, "alive": True},
        "zone": {"x": rules.ZONE_CX, "y": rules.ZONE_CY, "radius": rules.ZONE_R0},
    }


def test_requires_consecutive_forward_motion_and_switches_once():
    detector = StyleDetector(bytearray([1]) * (rules.W * rules.H), CONFIG)
    positions = [10.0, 10.2, 10.4, 10.4, 10.6, 10.8, 11.0, 10.0]
    assert [detector.observe(observation(i, x)) for i, x in enumerate(positions)] == [
        None, None, None, None, None, None, 6, 6,
    ]
    assert [record["frame"] for record in detector.trace if record["switch"]] == [6]


def test_includes_exact_threshold_in_both_directions():
    walls = bytearray([1]) * (rules.W * rules.H)
    for direction in (1, -1):
        detector = StyleDetector(walls, CONFIG)
        for frame in range(4):
            detector.observe(observation(frame, 10.0 + direction * 0.18 * frame, toward_right=direction == 1))
        assert detector.switch_frame == 3


def test_contact_window_starts_at_first_clear_frame_and_has_a_hard_deadline():
    walls = bytearray([1]) * (rules.W * rules.H)
    config = CONFIG | {"latest_frame": 20, "frames_from_first_clear": 3}

    def run(first_clear, positions):
        detector = StyleDetector(walls, config)
        for frame, x in enumerate(positions):
            walls[12 * rules.W + 13] = 2 if frame < first_clear else 1
            detector.observe(observation(frame, x))
        return detector.switch_frame

    assert run(12, [10.0] * 12 + [10.0, 10.2, 10.4, 10.6]) == 15
    assert run(12, [10.0] * 14 + [10.0, 10.2, 10.4, 10.6]) is None
    assert run(19, [10.0] * 19 + [10.0, 10.2, 10.4, 10.6]) is None


def test_ignores_wall_occlusion_disconnected_frames_and_late_evidence():
    walls = bytearray([1]) * (rules.W * rules.H)
    detector = StyleDetector(walls, CONFIG)
    for row in (11, 12):
        walls[row * rules.W + 13] = 2
    for frame in range(4):
        detector.observe(observation(frame, 10.0 + 0.2 * frame))
    assert detector.switch_frame is None
    assert all(not row["clear_lane"] for row in detector.trace)

    walls[:] = bytes([1]) * len(walls)
    for frame in (4, 5, 7, 8, 9, 11, 12):
        detector.observe(observation(frame, 10.0 + 0.2 * frame))
    assert detector.switch_frame is None
    assert detector.trace[-1]["frame"] == 9


def test_mirror_equivalence_and_metadata_removed_at_start():
    walls = bytearray([1]) * (rules.W * rules.H)
    left = StyleDetector(walls, CONFIG)
    right = StyleDetector(walls, CONFIG)
    for frame in range(4):
        x = 10.0 + 0.2 * frame
        left.observe(observation(frame, x))
        right.observe(observation(frame, rules.W - x, toward_right=False))
    assert left.trace == right.trace

    start = {"side": "red", "opponent": "boss", "tier": "boss", "game": 2, "spawn": {}, "rules": {},
             "map": {"width": rules.W, "height": rules.H, "grid": [], "seed": 123}}
    public = policy_match_info(start)
    assert set(public) == {"side", "map", "spawn", "rules"}
    assert set(public["map"]) == {"width", "height", "grid"}


def test_b_receives_the_prior_observations_without_running_inactive_search():
    class World:
        def __init__(self):
            self.frames = []

        def update(self, obs):
            self.frames.append(obs["frame"])

    class A:
        def decide(self, obs):
            return "A"

    class BaseB:
        def __init__(self):
            self.base_frames = []
            self._local_world = World()

        def decide(self, obs):
            self.base_frames.append(obs["frame"])

    class B(BaseB):
        def __init__(self):
            super().__init__()
            self.search_frames = []

        def decide(self, obs):
            super().decide(obs)
            self._local_world.update(obs)
            self.search_frames.append(obs["frame"])
            return "B"

    router = StyleRouter.__new__(StyleRouter)
    router.a, router.b = A(), B()
    router.active, router.config = "A", CONFIG
    router.detector = StyleDetector(bytearray([1]) * (rules.W * rules.H), CONFIG)
    positions = [10.0, 10.2, 10.4, 10.6, 10.6, 10.6]
    assert [router.decide(observation(i, x)) for i, x in enumerate(positions)] == ["A", "A", "A", "B", "B", "B"]
    assert router.b.search_frames == [3, 4, 5]
    assert router.b.base_frames == router.b._local_world.frames == list(range(6))
