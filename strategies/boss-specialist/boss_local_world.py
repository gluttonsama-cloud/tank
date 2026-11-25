"""Roll out the frozen v31 sparring model from public SDK observations."""

from copy import copy
from math import hypot

from seedcup.game import SIDES, Game, GameMap, rules
from seedcup.game.engine import STILL, Shell, Tank, ang_diff
from seedcup.game.maps import Spawn


def _fork_model(model):
    # The v31 model's weights and map data are shared. Its distance cache is a
    # memoization of the fixed wall graph. Only the two histories and evade
    # helper hold mutable per-rollout state below the bot itself.
    replica = copy(model)
    replica.features = copy(model.features)
    replica.features.history = model.features.history.copy()
    replica.features.motion = model.features.motion.copy()
    replica.evade = copy(model.evade)
    return replica


def _fork_game(game):
    replica = copy(game)
    replica.tanks = [copy(tank) for tank in game.tanks]
    replica.shells = [copy(shell) for shell in game.shells]
    replica.violations = game.violations.copy()
    replica.last_actions = [copy(action) for action in game.last_actions]
    replica.start_hp = game.start_hp.copy()
    return replica


def _tank(side, obs):
    return Tank(side, obs["x"], obs["y"], obs["heading"], obs["hp"], obs["alive"])


def _ticks(seconds):
    return round(seconds * rules.TICK_HZ)


class Branch:
    """A mutable roll-out with its own SDK game and v31 observation history."""

    def __init__(self, game, model, own_side, enemy_action):
        self.game = game
        self._model = model
        self.own_side = own_side
        self.enemy_side = 1 - own_side
        self._enemy_action = enemy_action
        self.events = []

    @property
    def enemy_action(self):
        if self._enemy_action is None:
            self._enemy_action = (
                STILL.copy() if self.game.over else self._model.decide(self.game.observe(self.enemy_side))
            )
        return self._enemy_action

    def step(self, own_action):
        enemy_action = self.enemy_action
        actions = [own_action, enemy_action] if self.own_side == 0 else [enemy_action, own_action]
        self.events = self.game.step(actions)
        self._enemy_action = None
        return self.game

    def observation(self):
        return self.game.observe(self.own_side)

    def fork(self):
        return Branch(
            _fork_game(self.game), _fork_model(self._model), self.own_side, copy(self._enemy_action)
        )


class World:
    """Refresh the state once per live observation before starting predictions.

    The factory receives public match info for the opposing side and returns
    an initialized private copy of the frozen MovementSparring model.
    """

    def __init__(self, info, opponent_factory):
        if info["rules"]["version"] != rules.RULES_VERSION or rules.VARIANT:
            raise ValueError(f"预测需要正式 SDK 规则 {rules.RULES_VERSION}")
        self.own_side = SIDES.index(info["side"])
        self.enemy_side = 1 - self.own_side
        raw_map = info["map"]
        grid = bytes(cell for row in raw_map["grid"] for cell in row)
        x, y, heading = (info["spawn"][key] for key in ("x", "y", "heading"))
        own_spawn = Spawn(x, y, heading)
        enemy_spawn = Spawn(raw_map["width"] - x, y, (180 - heading) % 360)
        spawns = (own_spawn, enemy_spawn) if self.own_side == 0 else (enemy_spawn, own_spawn)
        # The official start packet supplies the complete grid and may omit the seed.
        # Standard-rule rollouts have zero spread and consume no map RNG.
        self.map = GameMap(raw_map.get("seed", 0), grid, sum(grid) / len(grid), -1, spawns)
        enemy_info = info | {
            "side": SIDES[self.enemy_side],
            "opponent": "本地策略",
            "spawn": {"x": enemy_spawn.x, "y": enemy_spawn.y, "heading": enemy_spawn.heading},
        }
        self._model = opponent_factory(enemy_info)
        self.game = None
        self.enemy_action = None

    def update(self, obs):
        frame = obs["frame"]
        if self.game is not None and frame == self.game.tick:
            return self.game, self.enemy_action

        own = _tank(self.own_side, obs["self"])
        enemy = _tank(self.enemy_side, obs["enemy"])
        own.reload = _ticks(obs["self"]["reload"])
        own.still = _ticks(obs["self"]["still"])
        own.repairing = obs["self"]["repairing"]
        own.collided = obs["collision"]
        tanks = [own, enemy] if self.own_side == 0 else [enemy, own]
        shells = [
            Shell(
                index,
                self.own_side if shell["mine"] else self.enemy_side,
                shell["x"],
                shell["y"],
                shell["dx"],
                shell["dy"],
            )
            for index, shell in enumerate(obs["shells"], start=1)
        ]
        game = Game(self.map, tick=frame, tanks=tanks, shells=shells, _shell_id=len(shells) + 1)
        game.last_actions[self.own_side] = copy(obs["last_action"])
        previous = self.game
        self._restore_enemy(game, own, enemy, previous)
        game.over = not (own.alive and enemy.alive) or frame >= rules.GAME_TICKS
        self.game = game
        self.enemy_action = STILL.copy() if game.over else self._model.decide(game.observe(self.enemy_side))
        return game, self.enemy_action

    def _restore_enemy(self, game, own, enemy, previous):
        muzzle_distance = rules.RADIUS + rules.MUZZLE_EXTENSION + rules.SHELL_SPEED * rules.DT
        # A surviving new shell ends this frame one flight step from its muzzle.
        # Older shells gain more distance than their owner can move each tick.
        fired = any(
            shell.owner == self.enemy_side
            and abs(hypot(shell.x - enemy.x, shell.y - enemy.y) - muzzle_distance) <= 0.002
            for shell in game.shells
        )
        if previous is None:
            enemy.reload = rules.RELOAD_TICKS if fired else 0
            return

        before = previous.tanks[self.enemy_side]
        elapsed = game.tick - previous.tick
        enemy.reload = max(0, before.reload - elapsed)
        if not fired and elapsed == 1 and enemy.reload == 0 and enemy.alive:
            own_lost_hp = previous.tanks[self.own_side].hp - own.hp
            damaged = own_lost_hp >= rules.DAMAGE - 0.1
            predicted_fire = self.enemy_action["fire"]
            if damaged or predicted_fire:
                fired = self._vanished_shot(game, own, enemy, damaged, predicted_fire)
        if fired:
            enemy.reload = rules.RELOAD_TICKS

        moving = (
            hypot(enemy.x - before.x, enemy.y - before.y) > 0.01 * rules.SPEED * rules.DT + 0.001
            or abs(ang_diff(enemy.h, before.h)) > 0.01 * rules.TURN_RATE * rules.DT + 0.01
        )
        # A v31 drive command can leave the tank stationary when it meets a wall.
        blocked_drive = elapsed == 1 and abs(self.enemy_action["forward"]) > 0.01
        hit = before.hp - enemy.hp >= rules.DAMAGE - 0.1
        if not (moving or blocked_drive or fired or hit):
            enemy.still = before.still + elapsed
        enemy.repairing = enemy.alive and enemy.hp > before.hp

    def _vanished_shot(self, game, own, enemy, damaged, predicted_fire):
        # At point-blank range a shell can hit or strike a wall before it is
        # visible. Probe the public post-movement poses with the SDK itself.
        shooter = copy(enemy)
        shooter.reload = 0
        target = copy(own)
        target.hp = rules.HP
        target.alive = True
        tanks = [target, shooter] if self.own_side == 0 else [shooter, target]
        probe = Game(self.map, tick=game.tick - 1, tanks=tanks)
        actions = [STILL, STILL]
        actions[self.enemy_side] = STILL | {"fire": True}
        events = probe.step(actions)
        return any(
            (damaged and event["type"] == "hit") or (predicted_fire and event["type"] == "spark")
            for event in events
        )

    def branch(self):
        if self.game is None:
            raise RuntimeError("预测前需要先更新公开观测")
        return Branch(
            _fork_game(self.game), _fork_model(self._model), self.own_side, self.enemy_action.copy()
        )
