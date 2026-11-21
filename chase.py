"""Chase a wounded hunter that backs off to regenerate (it needs 5 s without damage, then heals 45 HP/s).

`Chase.plan(...)` returns None (normal behaviour) or a dict: move [mx, my], boost, aim_at [x, y, z] (where to
look when the track is gone), id, tracked, reason. Stdlib only; the line test is the `clear(a, b)` that bot.py
passes in (ray.clear). All thresholds are module constants so a tuner can set them (chase.CHASE_HP = 0.6).
"""
import math

CHASE_HP = 0.80            # chase a hunter below this HP fraction (house hunters break off at 0.6 / 0.8)...
CHASE_START_RANGE = 200.0  # ...that starts within this range (vision range); once chasing, any range
CHASE_SELF_HP = 0.45       # never chase below this HP fraction (bot.py flees under 0.45)
ENGAGE_S = 5.0             # "fighting it": it hit us or we hit it this recently
OTHER_MEMORY_S = 5.0       # another mech counts as present this long after it leaves the track list
CHASE_LOST_S = 4.0         # after the track drops, walk to its last known position this long
CHASE_HOLD_RANGE = 60.0    # this close with a clear line: stop advancing
CHASE_BOOST_RANGE = 120.0  # boost when farther than this...
DENY_S = 3.5               # ...or when our last hit on it is this old (regen starts at 5 s)
CHASE_BOOST_ENERGY = 30.0  # keep this much energy for jumps
CHASE_NEEDS_BACKOFF = 1    # 1: start only on a hunter moving away from us (0: any wounded hunter we fight)
ZONE_MARGIN = 40.0         # the prey must stay this far inside the zone circle...
ZONE_LOOK_S = 8.0          # ...now and this many seconds ahead
EYE, CENTRE = 8.0, 4.5


def _circle(zone, t):
    """Zone circle (cx, cy, r) at time t, interpolated like the server shrinks it."""
    s0, s1 = zone["shrink_start"], zone["shrink_end"]
    f = min(max((t - s0) / (s1 - s0), 0.0), 1.0) if s1 > s0 else 0.0
    (x0, y0), (x1, y1) = zone["center"], zone["next_center"]
    return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f, zone["radius"] + (zone["next_radius"] - zone["radius"]) * f


def _away(m, x, y):
    """Its velocity points away from us."""
    v = m.get("vel") or (0.0, 0.0)
    return v[0] * (m["pos"][0] - x) + v[1] * (m["pos"][1] - y) > 0


class Chase:
    def __init__(self):
        self.prey = None  # (id, x, y, z, t last tracked)
        self.seen = {}    # mech id -> last time it was on the track list

    def state(self):
        return self.prey, dict(self.seen)

    def restore(self, s):
        self.prey, self.seen = s[0], dict(s[1])

    def forget(self, mid):
        """We killed it: stop chasing it, and it no longer counts as another mech around."""
        self.seen.pop(mid, None)
        if self.prey is not None and self.prey[0] == mid:
            self.prey = None

    def observe(self, t, mechs):
        """Note who is on the track list; called every decision, also when plan() is skipped."""
        for m in mechs:
            self.seen[m.get("id")] = t
        self.seen = {k: v for k, v in self.seen.items() if t - v <= OTHER_MEMORY_S}

    def _drop(self):
        self.prey = None
        return None

    def plan(self, t, me, mechs, hit_on, hit_by, zone, clear):
        """me: obs self; mechs: enemy mechs on the track list; hit_on / hit_by: id -> time of the last hit
        we dealt / took; zone: obs zone; clear: ray.clear(a, b) -> bool."""
        x, y, z = me["pos"]
        self.observe(t, mechs)
        armed = any(isinstance(w, dict) and w.get("kind") == "laser" for w in (me.get("weapons") or []))
        if me["hp"] / max(me.get("max_hp") or 1, 1) < CHASE_SELF_HP or not armed:
            return self._drop()

        def wounded(m):
            return m.get("hp") is not None and m["hp"] / max(m.get("max_hp") or 1400, 1) < CHASE_HP

        track = None
        if self.prey is not None:
            track = next((m for m in mechs if m.get("id") == self.prey[0]), None)
            if track is not None and not wounded(track):
                return self._drop()  # healed: an ordinary fight again
        else:
            cand = [m for m in mechs if m.get("id") is not None and wounded(m) and math.dist(m["pos"][:2], (x, y)) < CHASE_START_RANGE
                    and min(t - hit_on.get(m.get("id"), -99.0), t - hit_by.get(m.get("id"), -99.0)) < ENGAGE_S
                    and (not CHASE_NEEDS_BACKOFF or _away(m, x, y))]
            track = min(cand, key=lambda m: math.dist(m["pos"][:2], (x, y))) if cand else None
            if track is None:
                return None
        if track is not None:
            p = track["pos"]
            self.prey = (track.get("id"), p[0], p[1], p[2] if len(p) > 2 else z, t)
        pid, px, py, pz, pt = self.prey
        if (t - pt > CHASE_LOST_S or any(k != pid for k in self.seen)
                or any(k != pid and t - v < OTHER_MEMORY_S for k, v in hit_by.items())):
            return self._drop()  # lost it, or another mech is around or shooting us: no tunnel vision
        for cx, cy, r in (_circle(zone, t), _circle(zone, t + ZONE_LOOK_S)):
            if math.hypot(px - cx, py - cy) > r - ZONE_MARGIN:
                return self._drop()  # the chase would leave (or end outside) the zone
        d = math.hypot(px - x, py - y)
        if track is None and d < 8.0:
            return self._drop()  # reached where it was and it is not here
        ux, uy = ((px - x) / d, (py - y) / d) if d > 1e-6 else (0.0, 0.0)
        if d < CHASE_HOLD_RANGE and track is not None and clear((x, y, z + EYE), (px, py, pz + CENTRE)):
            ux = uy = 0.0  # close and in sight: let the guns work
        last_hit = hit_on.get(pid, -99.0)
        boost = (d > CHASE_BOOST_RANGE or t - last_hit > DENY_S) and me.get("energy", 0) > CHASE_BOOST_ENERGY \
            and (ux or uy)
        return {"move": [ux, uy], "boost": bool(boost), "aim_at": [px, py, pz + CENTRE], "id": pid,
                "tracked": track is not None, "reason": "chase"}
