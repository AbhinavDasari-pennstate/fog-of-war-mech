"""Steer to a clear line when nothing is shootable: PEEK. Movement only: never aim, never fire. Stdlib only.

The caller runs step() only while the aim module has no shot (no fire flag set and no mech in a clear line);
any shot the aim module has, including one at a target's lead point while its torso is still behind cover, wins
(the -0.37 of peek+hold 3a5ab11 came from overriding those shots).

With a target we are fighting (it hit us or we hit it in the last 5 s) listed in obs.mechs but behind cover, and
a weapon ready: walk to the nearest spot 2-12 m sideways (1.5 m forward) with a clear line, reachable in
PEEK_ETA_S, with no line from another known hunter; the aim module shoots once we are there. When the line is
clear but nothing is shootable (weapons cooling or reloading) after a peek, step back to where we stood.
step() returns (move_x, move_y) or None (the bot's own movement). It raises on bad input; the hook counts it.
"""
import math

PEEK_SIDE = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)  # m sideways, both sides: 12 spots, nearest first
PEEK_FWD = 1.5          # m toward the target
PEEK_ETA_S = 1.0        # s: a spot must be this close (half walk speed while the laser charges)
PEEK_S = 1.5            # s to reach the spot and get a line before we give it up
BACK_S = 1.5            # s at most to step back to where we stood
RAY_CAP = 16            # ray tests per decision at most; out of budget = no line (spot) / seen (exposure)
BACK_MAX_M, STALE_S = 15.0, 10.0  # never "step back" further than this, or to a peek older than this


class ShotLine:
    def __init__(self):
        self.reset()

    def reset(self):
        self.charging = False
        self.mode = self.home = self.spot = self.tid = None
        self.t0 = 0.0

    def step(self, t, me, target, others, ray_clear, ground=None, engaged=True):
        left = [RAY_CAP]

        def ray(a, b, out_of_budget=False):
            if left[0] <= 0:
                return out_of_budget
            left[0] -= 1
            return ray_clear(a, b)

        x, y, z = me["pos"]
        st = me.get("stats") or {}
        mh, walk = st.get("muzzle_height", 7.0), st.get("walk_speed", 11.7)
        ws = [w for w in (me.get("weapons") or []) if w]
        las = next((w for w in ws if w.get("kind") == "laser"), None)
        gun = next((w for w in ws if w.get("kind") == "gun"), None)
        charge = (las.get("charge") or 0.0) if las else 0.0
        self.charging = charge > 0
        if charge > 0:
            walk *= 0.5                                         # charging halves move speed
        ready = (las is not None and (charge > 0 or (las.get("cooldown") or 0) <= 0)) or (
            gun is not None and (gun.get("ammo") or 0) > 0 and (gun.get("reload") or 0) <= 0)
        if self.mode == "back":
            if math.hypot(self.home[0] - x, self.home[1] - y) < 1.0 or t - self.t0 > BACK_S:
                self.reset()
                return None
            return self._toward(x, y, self.home)
        ok = target is not None and target.get("kind") == "mech" and target.get("id") is not None
        if not ok or (self.mode and target.get("id") != self.tid):
            return self._back(t, x, y)                          # target gone or switched: back to cover
        tx, ty, tz = target["pos"]
        tgt = (tx, ty, tz + 4.5)
        if ray((x, y, z + mh), tgt):
            # a clear line but nothing to shoot with (cooling, reloading): out on a peek, step back to cover
            return self._back(t, x, y) if self.mode == "peek" and not ready else None
        if self.mode == "peek":
            if t - self.t0 <= PEEK_S:
                return self._toward(x, y, self.spot)
            return self._back(t, x, y)                          # the spot never gave a line
        if not (ready and engaged):
            return None                                         # nothing to shoot with, or not a fight we are in
        spot = self._peek_spot(x, y, z, mh, walk, tgt, others, ray, ground)
        if spot is None:
            return None
        self.mode, self.home, self.spot, self.t0, self.tid = "peek", (x, y), spot, t, target.get("id")
        return self._toward(x, y, spot)

    def _back(self, t, x, y):
        if self.mode == "peek" and self.charging:
            return None                                         # never walk the beam off the line mid-charge
        if (self.mode == "peek" and self.home is not None and t - self.t0 < STALE_S
                and math.hypot(self.home[0] - x, self.home[1] - y) <= BACK_MAX_M):
            self.mode, self.t0 = "back", t
            return self._toward(x, y, self.home)
        self.reset()
        return None

    def _peek_spot(self, x, y, z, mh, walk, tgt, others, ray, ground):
        """Nearest sideways spot with a clear line to tgt, reachable in PEEK_ETA_S, with no line from other hunters."""
        n = math.hypot(tgt[0] - x, tgt[1] - y)
        if n < 1e-6:
            return None
        ux, uy = (tgt[0] - x) / n, (tgt[1] - y) / n
        cands = sorted((math.hypot(s, PEEK_FWD), x - uy * s * sd + ux * PEEK_FWD, y + ux * s * sd + uy * PEEK_FWD)
                       for s in PEEK_SIDE for sd in (1.0, -1.0))
        for dist, sx, sy in cands:
            if dist / max(walk, 0.1) > PEEK_ETA_S:
                break
            gz = ground(sx, sy) if ground is not None else z
            if abs(gz - z) > 1.2:
                continue                                        # a step or a drop, not a sideways step
            if not ray((sx, sy, gz + mh), tgt):
                continue
            if not (ray((x, y, z + 1.5), (sx, sy, gz + 1.5)) and ray((x, y, z + 6.0), (sx, sy, gz + 6.0))):
                continue                                        # a wall between us and the spot
            if any(ray((o["pos"][0], o["pos"][1], o["pos"][2] + 8.0), (sx, sy, gz + 4.5), True) for o in others):
                continue                                        # another hunter would see us there
            return (sx, sy)
        return None

    @staticmethod
    def _toward(x, y, p):
        dx, dy = p[0] - x, p[1] - y
        n = math.hypot(dx, dy)
        return (dx / n, dy / n) if n > 1e-6 else None
