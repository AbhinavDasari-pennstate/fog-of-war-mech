"""Fight tactics for bot.py: disengage behind cover and regenerate, finish wounded hunters, keep threats in front.

All tunables are module constants so a training loop can set them (fight.RETREAT_HP = 0.5).
`Fight.plan(...)` returns None (normal v3 behaviour) or a dict: hide (no firing), move ([mx, my] or None),
reason. Stdlib only; ray tests go through the `clear(a, b)` function bot.py passes in (ray.clear).
"""
import math
import time

# ---- tunables ----
RETREAT_HP = 0.0         # hide below this HP fraction, one known hunter (0 = off: hiding cost kills in simulation)
MULTI_RETREAT_HP = 0.0   # hide below this with 2+ known hunters (0 = off)
REGEN_HP = 0.90          # stay hidden until this HP fraction
FINISH_HP = 0.25         # a hunter below this HP fraction within FINISH_RANGE keeps us fighting...
FINISH_SELF_HP = 0.30    # ...while we are above this
FINISH_RANGE = 200.0
SPLIT_DEG = 150.0        # two known hunters this far apart in bearing (we are sandwiched): break line of sight
SPLIT_MIN_HP = 0.0       # sandwich rule only below this HP fraction (0 = off)
THREAT_MEMORY_S = 12.0   # how long a hunter's last known position counts for hiding
HIDE_RADII = (10.0, 20.0, 32.0, 46.0, 62.0)
HIDE_BEARINGS = 16
CHASE_HP = 0.0           # walk at a known hunter below this HP fraction (0 = off; 1.01 = always close in)...
CHASE_RANGE = 160.0      # ...within this range...
CHASE_SELF_HP = 0.40     # ...while we are above this and it is the only hunter we know of
CHASE_S = 6.0            # keep going to its last known position this long after the track is lost
CHASE_NEEDS_LASER = 1    # chase only while we hold a laser (gun-only duels are lost)
WAY_R = 20.0            # corner waypoint ring radius for spots we cannot walk to straight
NO_COVER_S = 1.0         # after a failed cover search, fight on this long before searching again
SPOT_BUDGET_S = 0.015    # wall-clock cap of one cover search (counts as "no cover" when hit)
SPOT_RECHECK_S = 0.5     # re-test the chosen spot this often
EYE, CENTRE, TOP, KNEE = 8.0, 4.5, 9.0, 2.5


def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)


class Fight:
    def __init__(self):
        self.hiding = False
        self.threats = {}      # mech id -> (x, y, z, t last known)
        self.spot = None       # (x, y) we are walking to / holding
        self.spot_t = -99.0
        self.way = None        # corner waypoint on the way to the spot
        self.reason = ""
        self.prey = None       # (x, y, t) last known position of the wounded hunter we are chasing
        self.no_cover_t = -99.0  # last failed cover search

    # ---- geometry ----
    def _seen_from(self, clear, tx, ty, tz, x, y, z, scale=1.0):
        eye = (tx, ty, tz + EYE)
        return clear(eye, (x, y, z + CENTRE * scale)) or clear(eye, (x, y, z + TOP * scale))

    def exposed(self, clear, x, y, z, threats, scale=1.0):
        return [k for k, (tx, ty, tz, _) in threats.items() if self._seen_from(clear, tx, ty, tz, x, y, z, scale)]

    def _walk(self, clear, ax, ay, bx, by, z):
        """Straight walk a->b with no wall at knee or eye height (also False when b is inside a building)."""
        return clear((ax, ay, z + KNEE), (bx, by, z + KNEE), False) and clear((ax, ay, z + EYE), (bx, by, z + EYE), False)

    def find_spot(self, clear, x, y, z, threats, zone=None, scale=1.0):
        """(spot, waypoint): the nearest point hidden from every threat that we can walk to straight
        (waypoint None) or round one corner (via waypoint); None if there is none. zone: (cx, cy, r)."""
        away = [0.0, 0.0]
        for tx, ty, _, _ in threats.values():
            ux, uy = _unit(x - tx, y - ty)
            away[0] += ux
            away[1] += uy
        base = math.atan2(away[1], away[0]) if any(away) else 0.0
        step = 2 * math.pi / HIDE_BEARINGS
        # bearings ordered by closeness to "away from the threats"
        order = sorted(range(HIDE_BEARINGS), key=lambda i: abs(((i * step) + math.pi) % (2 * math.pi) - math.pi))
        ways = None  # corner points we can walk to straight: computed only if a hidden spot needs one
        stop = time.perf_counter() + SPOT_BUDGET_S
        for r in HIDE_RADII:
            for i in order:
                if time.perf_counter() > stop:
                    return None
                a = base + i * step
                px, py = x + r * math.cos(a), y + r * math.sin(a)
                if zone is not None and math.hypot(px - zone[0], py - zone[1]) > zone[2]:
                    continue
                if self.exposed(clear, px, py, z, threats, scale):
                    continue
                if self._walk(clear, x, y, px, py, z):
                    return (px, py), None
                if ways is None:
                    ways = [(x + WAY_R * math.cos(base + k * step), y + WAY_R * math.sin(base + k * step))
                            for k in range(HIDE_BEARINGS)]
                    ways = [w for w in ways if self._walk(clear, x, y, w[0], w[1], z)]
                for w in sorted(ways, key=lambda w: math.hypot(w[0] - px, w[1] - py)):
                    if time.perf_counter() > stop:
                        return None
                    if self._walk(clear, w[0], w[1], px, py, z):
                        return (px, py), w
        return None

    def _chase(self, t, x, y, hp, mechs, zone, me):
        """Walk at a wounded hunter (it needs 5 s clean to start regen), or to where it was last seen."""
        armed = not CHASE_NEEDS_LASER or any(isinstance(w, dict) and w.get("kind") == "laser"
                                             for w in (me.get("weapons") or []))
        if CHASE_HP <= 0 or hp <= CHASE_SELF_HP or len(self.threats) > 1 or not armed:
            self.prey = None
            return None
        for m in mechs:
            mh = m.get("hp")
            if (mh is not None and mh / max(m.get("max_hp") or 1400, 1) < CHASE_HP
                    and math.dist(m["pos"][:2], (x, y)) < CHASE_RANGE):
                self.prey = (m["pos"][0], m["pos"][1], t)
                break
        if self.prey is None or t - self.prey[2] > CHASE_S:
            self.prey = None
            return None
        px, py = self.prey[0], self.prey[1]
        if zone is not None and math.hypot(px - zone[0], py - zone[1]) > zone[2]:
            self.prey = None
            return None
        if math.hypot(px - x, py - y) < 8.0:
            if not mechs:  # reached its last position and it is gone
                self.prey = None
                return None
            self.reason = "chase"  # on top of it: hold here rather than let goals pull us off and back
            return {"hide": False, "move": [0.0, 0.0], "reason": "chase"}
        self.reason = "chase"
        return {"hide": False, "move": list(_unit(px - x, py - y)), "reason": "chase"}

    # ---- decision ----
    def plan(self, t, me, mechs, clear, zone=None, urgent=False, damaged_recent=False):
        """me: obs self; mechs: obs enemy mechs (kind mech); clear: ray.clear(a, b, terrain=True) -> bool."""
        x, y, z = me["pos"]
        scale = 1.0 + 0.05 * (me.get("cubes") or 0)
        for m in mechs:
            p = m.get("pos") or ()
            if m.get("id") is not None and len(p) >= 2:
                self.threats[m["id"]] = (p[0], p[1], p[2] if len(p) > 2 else z, t)
        self.threats = {k: v for k, v in self.threats.items() if t - v[3] <= THREAT_MEMORY_S}
        hp = me["hp"] / max(me.get("max_hp") or 1, 1)
        n_known = len(self.threats)

        # finish rule: a wounded hunter nearby is worth more than our regen
        finish = hp > FINISH_SELF_HP and any(
            m.get("hp") is not None and m["hp"] / max(m.get("max_hp") or 1400, 1) < FINISH_HP
            and math.dist(m["pos"][:2], (x, y)) < FINISH_RANGE for m in mechs)
        sandwich = False
        if hp < SPLIT_MIN_HP and len(mechs) >= 2:
            bs = [math.atan2(m["pos"][1] - y, m["pos"][0] - x) for m in mechs[:3]]
            sandwich = any(abs((a - b + math.pi) % (2 * math.pi) - math.pi) > math.radians(SPLIT_DEG)
                           for i, a in enumerate(bs) for b in bs[i + 1:])
        trigger = hp < (MULTI_RETREAT_HP if n_known >= 2 else RETREAT_HP) or sandwich
        if self.hiding:
            if hp >= REGEN_HP or urgent or finish or not self.threats:
                self.hiding, self.spot, self.way = False, None, None
        elif trigger and not urgent and not finish and self.threats and t - self.no_cover_t >= NO_COVER_S:
            self.hiding, self.spot, self.way = True, None, None
        if not self.hiding:
            self.reason = "finish" if finish else ""
            return self._chase(t, x, y, hp, mechs, zone, me)

        # hiding: hold still if nobody can see us, else walk to the nearest hidden spot
        if self.spot is not None and t - self.spot_t >= SPOT_RECHECK_S:
            self.spot_t = t
            if self.exposed(clear, self.spot[0], self.spot[1], z, self.threats, scale):
                self.spot = self.way = None
        if self.spot is None or (math.hypot(self.spot[0] - x, self.spot[1] - y) < 2.0 and damaged_recent):
            if not self.exposed(clear, x, y, z, self.threats, scale) and not damaged_recent:
                self.spot, self.way, self.spot_t = (x, y), None, t
            else:
                found = self.find_spot(clear, x, y, z, self.threats, zone, scale)
                self.spot, self.way = found if found else (None, None)
                self.spot_t = t
        if self.spot is None:  # no cover in reach: fleeing in the open loses, keep fighting
            self.hiding, self.way, self.no_cover_t = False, None, t
            self.reason = "no_cover"
            return None
        if self.way is not None and (math.hypot(self.way[0] - x, self.way[1] - y) < 3.0
                                     or math.hypot(self.spot[0] - x, self.spot[1] - y) < 2.0):
            self.way = None
        gx, gy = self.way if self.way is not None else self.spot
        dx, dy = gx - x, gy - y
        d = math.hypot(dx, dy)
        self.reason = "hide_hold" if d < 2.0 else "hide_go"
        return {"hide": True, "move": [0.0, 0.0] if d < 2.0 else list(_unit(dx, dy)), "reason": self.reason,
                "far": d}
