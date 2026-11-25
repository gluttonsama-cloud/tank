"""Grid navigation and short-horizon action selection for the SeedCup arcade arena."""
from collections import deque
from math import atan2, cos, degrees, hypot, radians, sin

from seedcup import Bot
from seedcup.game import rules
from seedcup.game.bots import clear_shot, wall_grid


ZONE_LOOKAHEAD = 30


def angle(dx, dy):
    return degrees(atan2(dy, dx))


def delta(target, current):
    return (target - current + 180) % 360 - 180


def clamp(value, limit):
    return max(-limit, min(limit, value))


class Lancer(Bot):
    def on_match_start(self, info):
        self.grid = info["map"]["grid"]
        self.walls = wall_grid(info)
        self.open = {(x, y) for y, row in enumerate(self.grid)
                     for x, wall in enumerate(row) if not wall}
        self.links = {p: [q for q in ((p[0]+1,p[1]), (p[0]-1,p[1]),
                                      (p[0],p[1]+1), (p[0],p[1]-1)) if q in self.open]
                      for p in self.open}
        self.previous = None
        self.route_goal = None
        self.distances = {}
        self.orbit = 1
        self.retreating = False

    def waypoint(self, x, y, tx, ty):
        goal = int(tx), int(ty)
        if goal not in self.open:
            goal = min(self.open, key=lambda p: (p[0]+0.5-tx)**2 + (p[1]+0.5-ty)**2)
        if goal != self.route_goal:
            queue = deque([goal])
            dist = {goal: 0}
            while queue:
                p = queue.popleft()
                for q in self.links[p]:
                    if q not in dist:
                        dist[q] = dist[p] + 1
                        queue.append(q)
            self.distances, self.route_goal = dist, goal
        p = int(x), int(y)
        if p == goal:
            return goal[0]+0.5, goal[1]+0.5
        q = min(self.links[p], key=self.distances.__getitem__)
        # In a one-cell corridor, line up with its centre before turning a corner.
        return q[0]+0.5, q[1]+0.5

    def free(self, x, y):
        for cx in (int(x-.38), int(x+.38)):
            for cy in (int(y-.38), int(y+.38)):
                if (cx, cy) not in self.open:
                    dx = max(cx-x, 0, x-cx-1)
                    dy = max(cy-y, 0, y-cy-1)
                    if dx*dx + dy*dy < .38**2:
                        return False
        return True

    def evade(self, x, y, heading, preferred, speed, shells, safe_center, frame):
        # Rank short movement arcs by clearance from the visible game projectiles.
        best = None
        cx, cy = safe_center
        radii = [rules.zone_radius(frame + tick) for tick in range(1, 9)]
        already_outside = hypot(x-cx, y-cy) > radii[0]
        future_radius = rules.zone_radius(frame + ZONE_LOOKAHEAD)
        for turn in (-1, -.5, 0, .5, 1, preferred):
            for forward in (-1, 1, 0):
                px, py, h = x, y, heading
                clearance = 3
                zone_cost = 0
                for tick, radius in enumerate(radii, 1):
                    h += turn*18
                    nx = px + forward*.25*cos(radians(h))
                    ny = py + forward*.25*sin(radians(h))
                    if self.free(nx, ny):
                        px, py = nx, ny
                    if hypot(px-cx, py-cy) > radius:
                        # Prefer an in-zone next step if one is possible.
                        zone_cost += 26 if tick == 1 and not already_outside else .8
                    for sx, sy, vx, vy in shells:
                        along = clamp((px-sx)*vx + (py-sy)*vy - (tick-.5)*1.2, .6)
                        t = (tick-.5)*1.2 + along
                        clearance = min(clearance, hypot(px-sx-vx*t, py-sy-vy*t))
                utility = (min(clearance, .95)*20 - abs(turn-preferred)*.24
                           - abs(forward-speed)*.2
                           - zone_cost - max(0, hypot(px-cx, py-cy)-future_radius)*2)
                if best is None or utility > best[0]:
                    best = utility, turn, forward
        return best[1:]

    def decide(self, obs):
        me, enemy = obs["self"], obs["enemy"]
        x, y, heading = me["x"], me["y"], me["heading"]
        ex, ey = enemy["x"], enemy["y"]
        vx, vy = (0, 0) if self.previous is None else (ex-self.previous[0], ey-self.previous[1])
        self.previous = ex, ey
        dist = hypot(ex-x, ey-y)
        flight = min(14, dist/1.2)
        ax, ay = ex+vx*flight, ey+vy*flight
        if not self.free(ax, ay):
            ax, ay = ex, ey
        bearing = angle(ax-x, ay-y)
        visible = clear_shot(self.walls, x, y, ax, ay)
        zone = obs["zone"]
        safe_center = zone["x"], zone["y"]
        safe_radius = rules.zone_radius(obs["frame"] + ZONE_LOOKAHEAD)
        margin = 1.1 if self.retreating else .6
        self.retreating = hypot(x-safe_center[0], y-safe_center[1]) > safe_radius-margin
        cooldown = me["reload"]
        if self.retreating or not visible:
            tx, ty = safe_center if self.retreating else (ex, ey)
            wx, wy = self.waypoint(x,y,tx,ty)
            target = angle(wx-x, wy-y)
            speed = min(1, hypot(wx-x,wy-y)*4)
        elif cooldown <= .55:
            target, speed = bearing, 0
        else:
            if obs["collision"]:
                self.orbit *= -1
            target = bearing + self.orbit*(70 if dist > 10 else 105)
            speed = 1
        diff = delta(target, heading)
        if speed and abs(diff) > 90:
            diff = delta(target+180, heading)
            speed = -speed
        turn = clamp(diff/18, 1)
        if abs(diff) > 55:
            speed *= .35
        incoming = []
        for s in obs["shells"]:
            if s["mine"]:
                continue
            dx, dy = x-s["x"], y-s["y"]
            along = dx*s["dx"]+dy*s["dy"]
            if 0 < along < 10 and abs(dx*s["dy"]-dy*s["dx"]) < 1.1:
                incoming.append((s["x"],s["y"],s["dx"],s["dy"]))
        if incoming:
            turn, speed = self.evade(x,y,heading,turn,speed,incoming,safe_center,obs["frame"])
        fire = visible and abs(delta(bearing,heading+turn*18))*max(dist,1) < 24
        return {"forward": speed, "turn": turn, "fire": fire}
