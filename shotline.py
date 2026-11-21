"""Get the shot when a tracked target's line of fire is blocked: PEEK, else HOLD. Stdlib only.

A target listed in obs.mechs (the server tracks real positions for 5 s out of view) with no clear line from our
muzzle:
  peek  step to the nearest spot 2-12 m sideways (1.5 m forward) with a clear line, reachable in PEEK_ETA_S at
        walk speed and with no line from another known hunter; stand there while the aim module shoots; when
        the shot is done (laser fired, a gun burst, or PEEK_SHOT_S) step back to where we stood
  hold  no such spot: keep the bot's own move (move None), aim at its tracked position, gun quiet
step() returns a plan dict or None (line clear and nothing in progress: the bot's own logic).
"""
import math

PEEK_SIDE = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)  # m sideways, both sides: 12 spots, nearest first
PEEK_FWD = 1.5          # m toward the target
PEEK_ETA_S = 1.0        # s: a spot must be this close at walk speed
PEEK_SHOT_S = 1.5       # s at the spot with a clear line (or walking to one) before we count it done / give up
BACK_S = 1.5            # s at most to step back to where we stood
GUN_BURST = 3           # rounds that count as a gun burst
RAY_CAP = 16            # ray tests per decision at most; out of budget = no line (spot) / seen (exposure)
HOLD_RECHECK_S = 0.5    # s a hold is reused before the spots are searched again


class ShotLine:
    def __init__(self):
        self.prev_charge = 0.0
        self.reset()

    def reset(self):
        self.mode = self.home = self.spot = self.tid = self.ammo0 = self.t_clear = None
        self.t0 = 0.0
        self.fired = self.was_clear = False
        self.hold_until = -99.0

    def _abort(self, t, x, y, d, me):
        if self.home is not None and self.mode == "peek":  # leave a peek: step back to where we stood
            self.mode, self.t0 = "back", t
            return self._plan("peek", "back", self._toward(x, y, self.home), d, me)
        self.reset()
        return None

    def step(self, t, me, target, others, ray_clear, ground=None):
        left = [RAY_CAP]

        def ray(a, b, out_of_budget=False):
            if left[0] <= 0:
                return out_of_budget
            left[0] -= 1
            return ray_clear(a, b)
        return self._step(t, me, target, others, ray, ground)   # raises on bad input: the hook turns it off

    def _step(self, t, me, target, others, ray, ground):
        x, y, z = me["pos"]
        st = me.get("stats") or {}
        mh, walk = st.get("muzzle_height", 7.0), st.get("walk_speed", 11.7)
        ws = [w for w in (me.get("weapons") or []) if w]
        las = next((w for w in ws if w.get("kind") == "laser"), None)
        gun = next((w for w in ws if w.get("kind") == "gun"), None)
        charge = (las.get("charge") or 0.0) if las else 0.0
        if self.prev_charge > 0 and charge == 0 and self.was_clear:
            self.fired = True                                   # the laser went off on a clear line
        self.prev_charge = charge
        if gun and charge == 0 and self.ammo0 is not None and (gun.get("ammo") or 0) <= self.ammo0 - GUN_BURST:
            self.fired = True                                   # a gun burst (a charging laser outranks it)
        if charge > 0:
            walk *= 0.5                                         # charging halves move speed
        if self.mode == "back":
            if math.hypot(self.home[0] - x, self.home[1] - y) < 1.0 or t - self.t0 > BACK_S:
                self.reset()
                return None
            return self._plan("peek", "back", self._toward(x, y, self.home), 0.0, me)
        if (target is None or target.get("kind") != "mech" or target.get("id") is None
                or (self.mode and target.get("id") != self.tid)):
            return self._abort(t, x, y, 0.0, me)
        tx, ty, tz = target["pos"]
        tgt = (tx, ty, tz + 4.5)
        d = math.hypot(tx - x, ty - y)
        self.was_clear = ray((x, y, z + mh), tgt)
        if self.was_clear:                                      # a clear line now
            if self.mode != "peek":
                self.reset()
                return None
            if self.t_clear is None:
                self.t_clear = t
            if charge == 0 and (self.fired or t - self.t_clear > PEEK_SHOT_S):
                return self._abort(t, x, y, d, me)              # shot done (never mid-charge): back
            return self._plan("peek", "shooting", (0.0, 0.0), d, me)
        if self.mode == "peek":
            if t - self.t0 <= PEEK_SHOT_S:
                return self._plan("peek", "stepping out", self._toward(x, y, self.spot), d, me)
            return self._abort(t, x, y, d, me)                  # the spot never gave a line
        rng = max([(w.get("stats") or {}).get("range", 230.0) for w in ws] or [0.0])
        if d > rng:
            self.reset()
            return None                                         # nothing we hold reaches it
        ready = (las is not None and (charge > 0 or (las.get("cooldown") or 0) <= 0)) or (
            gun is not None and (gun.get("ammo") or 0) > 0 and (gun.get("reload") or 0) <= 0)
        if ready and not (t < self.hold_until and target.get("id") == self.tid):
            spot = self._peek_spot(x, y, z, mh, walk, tgt, others, ray, ground)
            if spot is not None:
                self.reset()
                self.mode, self.home, self.spot, self.t0, self.tid = "peek", (x, y), spot, t, target.get("id")
                self.ammo0 = gun.get("ammo") if gun else None
                return self._plan("peek", "spot %.1f m" % math.dist(spot, (x, y)), self._toward(x, y, spot), d, me)
            self.reset()
            self.hold_until, self.tid = t + HOLD_RECHECK_S, target.get("id")
        dx, dy, dz = tgt[0] - x, tgt[1] - y, tgt[2] - z - mh
        aim = [math.atan2(dy, dx), math.atan2(dz, max(math.hypot(dx, dy), 1e-6))]
        return self._plan("hold", "" if ready else "reloading", None, d, me, aim=aim, gun_quiet=True)

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
        return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)

    @staticmethod
    def _plan(opt, note, move, d, me, aim=None, gun_quiet=False):
        return {"opt": opt, "note": note, "move": move, "aim": aim, "gun_quiet": gun_quiet,
                "log": "shot %s d=%.0f e=%.0f%s" % (opt, d, me.get("energy") or 0.0, (" " + note) if note else "")}
