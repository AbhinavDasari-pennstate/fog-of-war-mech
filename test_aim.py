"""Tests for aim.py. Run: python test_aim.py

The recorded-track benchmark needs the kit's check/stream.jsonl.gz: set MECHFIGHT_STREAM or keep
the kit next to this repo (../kit/mechfight-kit/check/stream.jsonl.gz). Without it those tests skip.
"""
import gzip
import json
import math
import os
import random
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aim  # noqa: E402

wrap = aim.wrap
BEARINGS = [-math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4, 0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4]

GUN = {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "spread_deg": 0.6, "mag": 12, "reload": 2.4}
LASER = {"damage": 315, "charge": 0.75, "min_charge": 0.15, "min_frac": 0.35, "hold_s": 2, "cooldown": 1.65,
         "range": 280}
LIGHTNING = {"damage": 60, "cooldown": 0.25, "range": 45, "cone_deg": 30, "chain": 2, "energy": 3}
CONFIG = {
    "mech": {"radius": 2.5, "height": 9, "muzzle_height": 7, "turn_rate": 4, "pitch_rate": 3, "min_pitch": -0.7,
             "max_pitch": 1.3, "ground_accel": 58.5, "boost_speed": 19.5, "walk_speed": 11.7},
    "cubes": {"size_per_cube": 0.05}, "vision": {"memory_s": 5}, "npc": {"radius": 1.8, "height": 3.5},
    "crates": {"radius": 1.4, "height": 2.4},
    "weapons": {"gun": GUN, "laser": LASER, "lightning": LIGHTNING},
}


def make_me(pos=(0.0, 0.0, 0.0), aim_=(0.0, 0.0), vel=(0.0, 0.0, 0.0), cubes=0, weapons=None, energy=100.0, hp=1400):
    scale = 1.0 + 0.05 * cubes
    gun = {"kind": "gun", "cooldown": 0.0, "ammo": 12, "reload": 0.0, "stats": dict(GUN)}
    return {"id": 3, "pos": list(pos), "vel": list(vel), "aim": list(aim_), "hp": hp, "max_hp": 1400,
            "energy": energy, "cubes": cubes,
            "stats": {"muzzle_height": 7.0 * scale, "turn_rate": 4.0, "pitch_rate": 3.0, "boost_speed": 19.5},
            "weapons": weapons if weapons is not None else [gun, None]}


def laser_weapon(charge=0.0, cooldown=0.0):
    return {"kind": "laser", "cooldown": cooldown, "charge": charge, "stats": dict(LASER)}


def lightning_weapon():
    return {"kind": "lightning", "cooldown": 0.0, "stats": dict(LIGHTNING)}


def make_target(tid=10, pos=(80.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), cubes=0, hp=1400, max_hp=1400, aim_=(0.0, 0.0),
                charging=False, kind="mech"):
    return {"id": tid, "kind": kind, "pos": list(pos), "vel": list(vel), "aim": list(aim_), "hp": hp,
            "max_hp": max_hp, "cubes": cubes, "charging": charging}


def make_obs(me, mechs, t=1.0, crates=(), events=()):
    return {"type": "obs", "tick": int(t * 20), "t": t, "self": me, "mechs": list(mechs), "crates": list(crates),
            "events": list(events), "sounds": [], "projectiles": []}


def dir_to(src, dst):
    d = [dst[i] - src[i] for i in range(3)]
    n = math.sqrt(sum(c * c for c in d))
    return [c / n for c in d]


def aim_dir(yaw, pitch):
    return [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]


# ---------------------------------------------------------------------------- closed-loop duel

DT = 0.05


def strafer(bearing, seed):
    rng = random.Random(seed + 99)
    perp = bearing + math.pi / 2
    st = {"sign": 1, "speed": rng.uniform(5, 14), "next": 1.0}

    def f(t, pos, vel):
        while t >= st["next"]:
            st["sign"] = -st["sign"]
            st["speed"] = rng.uniform(5, 14)
            st["next"] += 1.0
        return math.cos(perp) * st["sign"] * st["speed"], math.sin(perp) * st["sign"] * st["speed"]
    return f


def accelerator(bearing, seed):
    perp = bearing + math.pi / 2

    def f(t, pos, vel):
        if t < 1.0:
            return 0.0, 0.0
        return math.cos(perp) * 11.7, math.sin(perp) * 11.7
    return f


def circler(bearing, seed, dist=100.0):
    perp = bearing + math.pi / 2
    start = (dist * math.cos(bearing), dist * math.sin(bearing))
    centre = (start[0] + 20 * math.cos(perp), start[1] + 20 * math.sin(perp))

    def f(t, pos, vel):
        phi = math.atan2(pos[1] - centre[1], pos[0] - centre[0])
        return -math.sin(phi) * 12.0, math.cos(phi) * 12.0
    return f


def straight_runner(bearing, seed):
    perp = bearing + math.pi / 2

    def f(t, pos, vel):
        return math.cos(perp) * 15.0, math.sin(perp) * 15.0
    return f


def duel(motion_factory, bearing, dist, method, seed=1, seconds=8.0, cubes=0, circ_dist=None):
    """Closed loop at 20 Hz physics, a decision every 2 ticks (like the server).

    We stand still at the origin, start looking 0.8 rad off the target, and use plan_shots. The torso
    slews at turn_rate and pitch_rate, the gun fires 252 m/s bullets along the ACTUAL aim with
    Gaussian spread, and the target follows its commanded velocity at ground_accel 58.5.
    Returns (shots, hits).
    """
    rng = random.Random(seed)
    motion = motion_factory(bearing, seed) if circ_dist is None else motion_factory(bearing, seed, circ_dist)
    tpos = [dist * math.cos(bearing), dist * math.sin(bearing), 0.0]
    tvel = [0.0, 0.0, 0.0]
    radius, height = 2.5 * (1 + 0.05 * cubes), 9.0 * (1 + 0.05 * cubes)
    yaw, pitch = wrap(bearing + 0.8), 0.0
    cmd_aim, cmd_fire = [yaw, pitch], [False, False]
    ammo, reload_t, cooldown = 12, 0.0, 0.0
    tracker = aim.Tracker(CONFIG, method=method)
    state = aim.TargetState()
    bullets = []
    shots = hits = 0
    for tick in range(int(seconds / DT)):
        t = tick * DT
        if tick % 2 == 0:
            gun = {"kind": "gun", "cooldown": max(cooldown, 0.0), "ammo": ammo, "reload": reload_t,
                   "stats": dict(GUN)}
            me = make_me(aim_=(yaw, pitch), weapons=[gun, None])
            tg = make_target(pos=tpos, vel=tvel, cubes=cubes)
            obs = make_obs(me, [tg], t)
            tracker.update(obs)
            target = aim.choose_target(obs, me, state, tracker)
            plan = aim.plan_shots(obs, me, target, tracker, CONFIG, None)
            if plan["aim"]:
                cmd_aim = plan["aim"]
            cmd_fire = plan["fire"]
        # our torso
        dy = wrap(cmd_aim[0] - yaw)
        yaw = wrap(yaw + max(-4.0 * DT, min(4.0 * DT, dy)))
        pitch += max(-3.0 * DT, min(3.0 * DT, cmd_aim[1] - pitch))
        pitch = max(-0.7, min(1.3, pitch))
        # gun
        cooldown -= DT
        if reload_t > 0:
            reload_t -= DT
            if reload_t <= 0:
                reload_t, ammo = 0.0, 12
        if cmd_fire[0] and cooldown <= 0 and ammo > 0 and reload_t <= 0:
            s = math.radians(0.6)
            d = aim_dir(yaw + rng.gauss(0, s), pitch + rng.gauss(0, s))
            org = [3.0 * d[0], 3.0 * d[1], 7.0 + 3.0 * d[2]]
            bullets.append([org, [252.0 * c for c in d]])
            shots += 1
            ammo -= 1
            cooldown = 0.24
            if ammo == 0:
                reload_t = 2.4
        # target
        cx, cy = motion(t, tpos, tvel)
        ddx, ddy = cx - tvel[0], cy - tvel[1]
        n = math.hypot(ddx, ddy)
        lim = 58.5 * DT
        if n > lim:
            ddx, ddy = ddx * lim / n, ddy * lim / n
        old = tpos[:]
        nvx, nvy = tvel[0] + ddx, tvel[1] + ddy
        tpos = [tpos[0] + 0.5 * (tvel[0] + nvx) * DT, tpos[1] + 0.5 * (tvel[1] + nvy) * DT, 0.0]
        tvel = [nvx, nvy, 0.0]
        # bullets, 10 substeps per tick against the target cylinder
        alive = []
        for b in bullets:
            hit = False
            for k in range(10):
                f = (k + 1) / 10.0
                for i in range(3):
                    b[0][i] += b[1][i] * DT / 10.0
                px = old[0] + (tpos[0] - old[0]) * f
                py = old[1] + (tpos[1] - old[1]) * f
                if math.hypot(b[0][0] - px, b[0][1] - py) <= radius and 0.0 <= b[0][2] <= height:
                    hit = True
                    break
            if hit:
                hits += 1
            elif math.hypot(b[0][0], b[0][1]) < 240.0:
                alive.append(b)
        bullets = alive
    return shots, hits


def duel_set(motion_factory, method, dists=(60.0, 110.0), seeds=(1, 2), cubes=0, circ=False):
    """Total (shots, hits) over 8 bearings x distances x seeds."""
    s = h = 0
    for b in BEARINGS:
        for d in dists:
            for sd in seeds:
                a, c = duel(motion_factory, b, d, method, sd, cubes=cubes, circ_dist=d if circ else None)
                s += a
                h += c
    return s, h


# ---------------------------------------------------------------------------- recorded-track benchmark


def find_stream():
    for c in (os.environ.get("MECHFIGHT_STREAM"),
              os.path.join(HERE, "..", "kit", "mechfight-kit", "check", "stream.jsonl.gz"),
              os.path.join(HERE, "stream.jsonl.gz")):
        if c and os.path.isfile(c):
            return c
    return None


def load_obs(path):
    return [o for o in (json.loads(line) for line in gzip.open(path, "rt")) if "self" in o]


def truth_at(series, t):
    """Interpolated (x, y) at time t, None unless t is bracketed by samples at most 0.25 s apart."""
    for k in range(len(series) - 1):
        a, b = series[k], series[k + 1]
        if a[0] <= t <= b[0]:
            if b[0] - a[0] > 0.25:
                return None
            f = (t - a[0]) / (b[0] - a[0]) if b[0] > a[0] else 0.0
            return a[1][0] + (b[1][0] - a[1][0]) * f, a[1][1] + (b[1][1] - a[1][1]) * f
    return None


def replay(obs_list, method, mech_id=10, speed=252.0):
    """[(range, t, miss_m)]: Tracker prediction at the bullet's flight time vs the real track."""
    series = [(o["t"], m["pos"]) for o in obs_list for m in o["mechs"] if m["id"] == mech_id]
    tr = aim.Tracker(CONFIG, method=method)
    out = []
    for o in obs_list:
        tr.update(o)
        m = next((m for m in o["mechs"] if m["id"] == mech_id), None)
        if m is None:
            continue
        d = math.hypot(m["pos"][0] - o["self"]["pos"][0], m["pos"][1] - o["self"]["pos"][1])
        tof = d / speed
        q = truth_at(series, o["t"] + tof)
        if q is None:
            continue
        p = tr.predict(mech_id, tof)
        out.append((d, o["t"], math.hypot(p[0] - q[0], p[1] - q[1])))
    return out


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


BASELINE_M = 1.94  # linear lead, h4.py, rounded truth: median miss on the same track


@unittest.skipIf(find_stream() is None, "recorded stream not found (set MECHFIGHT_STREAM)")
class TestRecordedBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs = load_obs(find_stream())

    def test_aim_benchmark_beats_linear_lead_on_recorded_tracks(self):
        lin = replay(self.obs, "linear")
        new = replay(self.obs, aim.DEFAULT_METHOD)
        self.assertGreaterEqual(len(new), 100)
        self.assertEqual(len(new), len(lin))
        print("\nrecorded-track benchmark: default predictor %s (damp %.2f) vs linear lead, gun speed 252 m/s"
              % (aim.DEFAULT_METHOD, aim.DEFAULT_DAMP))
        print("  bucket      n   linear median   default median")
        for lo, hi in ((0, 40), (40, 80), (80, 200)):
            el = [m for d, _, m in lin if lo <= d < hi]
            en = [m for d, _, m in new if lo <= d < hi]
            if el:
                print("  %3d-%3d m %4d   %10.3f m   %11.3f m" % (lo, hi, len(el), med(el), med(en)))
                if len(el) >= 20:
                    self.assertLess(med(en), med(el), "bucket %d-%d" % (lo, hi))
            else:
                print("  %3d-%3d m %4d   no recorded samples (the track covers %.0f-%.0f m)" % (
                    lo, hi, 0, min(d for d, _, _ in new), max(d for d, _, _ in new)))
        print("  all       %4d   %10.3f m   %11.3f m   (h4.py baseline %.2f m)" % (
            len(new), med([m for _, _, m in lin]), med([m for _, _, m in new]), BASELINE_M))
        self.assertLess(med([m for _, _, m in new]), BASELINE_M)
        self.assertLess(med([m for _, _, m in new]), med([m for _, _, m in lin]))

    # --- predictor selection on one half, score on the other
    def _series(self):
        out = []
        for o in self.obs:
            for m in o["mechs"]:
                if m["id"] == 10:
                    out.append((o["t"], m["pos"], m["vel"], o["self"]["pos"]))
        return out

    @staticmethod
    def _split(S):
        t0 = S[0][0]
        halves = ([], [])
        for i, s in enumerate(S):
            b = int((s[0] - t0) // 6.0)
            into = (s[0] - t0) - b * 6.0
            if into < 0.7 or 6.0 - into < 0.7:
                continue
            halves[b % 2].append(i)
        return halves

    @staticmethod
    def _eval(S, idx, method, kw, truth):
        errs = []
        for i in idx:
            t, p, v, me = S[i]
            d = math.hypot(p[0] - me[0], p[1] - me[1])
            tof = d / 252.0
            q = truth_at(truth, t + tof)
            if q is None:
                continue
            h = []
            k = i
            while k >= 0 and len(h) < 6 and (k == i or S[k + 1][0] - S[k][0] <= aim.MAX_GAP_S):
                h.insert(0, (S[k][0], S[k][1][0], S[k][1][1], S[k][1][2], S[k][2][0], S[k][2][1], S[k][2][2]))
                k -= 1
            pr = aim.extrapolate(h, tof, method, **kw)
            errs.append(math.hypot(pr[0] - q[0], pr[1] - q[1]))
        return errs

    CANDS = ([("linear", {})] + [("smoothed", {"n": n}) for n in (3, 4, 5)]
             + [("damped", {"damp": k}) for k in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)]
             + [("accel", {"n": n}) for n in (3, 4, 5, 6)] + [("circular", {"n": n}) for n in (4, 5, 6)])

    def _choose(self, S, train, truth):
        best, bm = None, 1e9
        for c in self.CANDS:
            m = med(self._eval(S, train, c[0], c[1], truth))
            if m < bm - 1e-9:
                best, bm = c, m
        return best

    def test_predictor_chosen_on_train_half_scored_on_held_out_half(self):
        S = self._series()
        truth = [(s[0], s[1]) for s in S]
        ha, hb = self._split(S)
        self.assertGreater(len(ha), 30)
        self.assertGreater(len(hb), 30)
        picks = []
        for name, train, test in (("A->B", ha, hb), ("B->A", hb, ha)):
            best = self._choose(S, train, truth)
            picks.append(best)
            lin = med(self._eval(S, test, "linear", {}, truth))
            new = med(self._eval(S, test, best[0], best[1], truth))
            print("held-out %s: chosen %s %s  linear %.3f m  chosen %.3f m (n=%d)" % (
                name, best[0], best[1], lin, new, len(self._eval(S, test, "linear", {}, truth))))
            self.assertLess(new, lin, "the predictor picked on the train half must beat linear on the held-out half")
            # no leakage: scrambling the held-out labels must not change the pick made on the train half
            held = set(test)
            scrambled = [(s[0], [s[1][0] + 500.0 * (i in held), s[1][1] - 300.0 * (i in held), s[1][2]])
                         for i, s in enumerate(S)]
            self.assertEqual(self._choose(S, train, scrambled), best)
        # the shipped default must be what the benchmark picks, in both directions
        for method, kw in picks:
            self.assertEqual(method, aim.DEFAULT_METHOD)
            if "damp" in kw:
                self.assertLessEqual(abs(kw["damp"] - aim.DEFAULT_DAMP), 0.15)

    def test_default_predictor_median_stable_across_either_half(self):
        S = self._series()
        truth = [(s[0], s[1]) for s in S]
        for idx in self._split(S):
            lin = med(self._eval(S, idx, "linear", {}, truth))
            new = med(self._eval(S, idx, aim.DEFAULT_METHOD, {}, truth))
            self.assertLess(new, lin)


# ---------------------------------------------------------------------------- closed-loop duels


class TestSimulatedTargets(unittest.TestCase):
    """8 bearings including the +/-pi wrap, 2 ranges, 2 seeds per motion. Hit rate = hits / shots fired."""

    def _run(self, name, factory, circ=False):
        s_new, h_new = duel_set(factory, aim.DEFAULT_METHOD, circ=circ)
        s_lin, h_lin = duel_set(factory, "linear", circ=circ)
        print("\n%-12s default %s: %d hits / %d shots = %.2f | linear lead: %d / %d = %.2f" % (
            name, aim.DEFAULT_METHOD, h_new, s_new, h_new / max(s_new, 1), h_lin, s_lin, h_lin / max(s_lin, 1)))
        return s_new, h_new, s_lin, h_lin

    def test_strafer_hit_rate(self):
        s, h, sl, hl = self._run("strafer", strafer)
        self.assertGreater(s, 400)
        self.assertGreaterEqual(h / s, 0.7)
        self.assertGreaterEqual(h, hl)  # flipping every 1 s at 5 to 14 m/s: no worse than linear lead

    def test_accelerating_target(self):
        s, h, sl, hl = self._run("accelerator", accelerator)
        self.assertGreater(s, 400)
        self.assertGreaterEqual(h / s, 0.8)
        self.assertGreaterEqual(h, 0.95 * hl)

    def test_circling_target(self):
        s, h, sl, hl = self._run("circler", circler, circ=True)
        self.assertGreater(s, 400)
        self.assertGreaterEqual(h / s, 0.9)
        self.assertGreaterEqual(h, 0.95 * hl)

    def test_straight_runner_not_hurt_by_lead_damping(self):
        s, h, sl, hl = self._run("runner 15 m/s", straight_runner)
        self.assertGreaterEqual(h / s, 0.8)
        self.assertGreaterEqual(h, 0.95 * hl)

    def test_cube_scaled_target_is_hit_more_often(self):
        a_s, a_h = duel_set(strafer, aim.DEFAULT_METHOD, dists=(110.0,), seeds=(1, 2, 3), cubes=0)
        b_s, b_h = duel_set(strafer, aim.DEFAULT_METHOD, dists=(110.0,), seeds=(1, 2, 3), cubes=10)
        print("\nstrafer at 110 m, 0 cubes %d/%d, 10 cubes %d/%d" % (a_h, a_s, b_h, b_s))
        self.assertGreater(b_h / b_s, a_h / a_s)


# ---------------------------------------------------------------------------- aim geometry


class TestAimGeometry(unittest.TestCase):
    def test_points_at_stationary_target_all_bearings_left_right_above_below(self):
        for b in BEARINGS:
            for rel in (1.2, -1.2):            # target to the left / right of where we look
                for dz in (-12.0, 0.0, 20.0):  # below / level / above
                    with self.subTest(bearing=b, rel=rel, dz=dz):
                        me = make_me(aim_=(wrap(b - rel), 0.0))
                        tgt = make_target(pos=(80 * math.cos(b), 80 * math.sin(b), dz))
                        sol = aim.solve(me, tgt, "gun", GUN, CONFIG)
                        self.assertLess(abs(wrap(sol.yaw - b)), 1e-6)
                        want = math.atan2(dz + 4.5 - 7.0, 80.0)
                        self.assertLess(abs(sol.pitch - want), 1e-6)
                        self.assertAlmostEqual(sol.t_on_target, max(abs(rel) / 4.0, abs(want) / 3.0), places=6)
                        y, p, t, hp = aim.aim_solution(me, tgt, "gun", GUN, CONFIG)
                        self.assertEqual((y, p, t, hp), sol.as_tuple())

    def test_yaw_wrap_takes_the_short_way_round(self):
        me = make_me(aim_=(math.pi - 0.1, 0.0))
        tgt = make_target(pos=(80 * math.cos(-math.pi + 0.1), 80 * math.sin(-math.pi + 0.1), 0.0))
        sol = aim.solve(me, tgt, "gun", GUN, CONFIG)
        self.assertAlmostEqual(sol.t_on_target, 0.2 / 4.0, places=3)

    def test_slew_uses_max_of_yaw_and_pitch_time_not_the_sum(self):
        # yaw 1.2 rad = 0.30 s; target 35 m up at 80 m = pitch 0.386 rad = 0.129 s
        me = make_me(aim_=(0.0, 0.0))
        up = make_target(pos=(80 * math.cos(1.2), 80 * math.sin(1.2), 35.0))
        sol = aim.solve(me, up, "gun", GUN, CONFIG)
        pitch_t = abs(sol.pitch) / 3.0
        self.assertAlmostEqual(sol.t_on_target, 1.2 / 4.0, places=6)
        self.assertLess(sol.t_on_target, 1.2 / 4.0 + pitch_t - 0.05)
        # pitch dominated: yaw 0.2 rad = 0.05 s, pitch time 0.129 s
        up2 = make_target(pos=(80 * math.cos(0.2), 80 * math.sin(0.2), 35.0))
        sol2 = aim.solve(me, up2, "gun", GUN, CONFIG)
        self.assertAlmostEqual(sol2.t_on_target, abs(sol2.pitch) / 3.0, places=6)
        self.assertLess(sol2.t_on_target, 0.2 / 4.0 + abs(sol2.pitch) / 3.0 - 0.03)

    def test_slew_rates_come_from_self_stats(self):
        me = make_me(aim_=(0.0, 0.0))
        me["stats"]["turn_rate"] = 5.2  # gyros
        tgt = make_target(pos=(80 * math.cos(1.2), 80 * math.sin(1.2), 0.0))
        sol = aim.solve(me, tgt, "gun", GUN, CONFIG)
        self.assertAlmostEqual(sol.t_on_target, 1.2 / 5.2, places=6)

    def test_slew_compensation_leads_a_moving_target(self):
        me = make_me(aim_=(1.0, 0.0))  # looking 1 rad away: 0.25 s of turning
        tgt = make_target(pos=(100.0, 0.0, 0.0), vel=(0.0, 10.0, 0.0))
        sol = aim.solve(me, tgt, "gun", GUN, CONFIG, method="linear")
        self.assertAlmostEqual(sol.t_on_target, 0.25 * (1 - 0.0), delta=0.04)
        extra = sol.yaw - sol.ideal_yaw   # lead beyond the fire-now solution: ~ v * t_slew / range
        self.assertGreater(extra, 0.7 * 10.0 * sol.t_on_target / 100.0)
        self.assertLess(extra, 1.3 * 10.0 * sol.t_on_target / 100.0)

    def test_gun_leads_by_flight_time_laser_does_not(self):
        me = make_me(aim_=(0.0, 0.0))
        tgt = make_target(pos=(200.0, 0.0, 0.0), vel=(0.0, 10.0, 0.0))
        g = aim.solve(me, tgt, "gun", GUN, CONFIG, method="linear")
        l = aim.solve(me, tgt, "laser", LASER, CONFIG, method="linear")
        self.assertGreater(g.flight, 0.7)
        self.assertEqual(l.flight, 0.0)
        # fire-now lead: gun sees (flight + latency), laser only the latency
        self.assertAlmostEqual(math.tan(g.ideal_yaw), 10.0 * (g.flight + aim.LATENCY_S) / 200.0, delta=0.002)
        self.assertAlmostEqual(math.tan(l.ideal_yaw), 10.0 * aim.LATENCY_S / 200.0, delta=0.001)

    def test_pitch_to_centre_mass_on_roof_target(self):
        me = make_me(aim_=(0.0, 0.0))
        roof = make_target(pos=(60.0, 0.0, 20.0))
        sol = aim.solve(me, roof, "gun", GUN, CONFIG)
        self.assertAlmostEqual(sol.pitch, math.atan2(20.0 + 4.5 - 7.0, 60.0), places=6)
        self.assertAlmostEqual(sol.aim_point[2], 24.5, places=6)

    def test_pitch_clamped_to_config_limits(self):
        me = make_me(aim_=(0.0, 0.0))
        up = aim.solve(me, make_target(pos=(5.0, 0.0, 60.0)), "gun", GUN, CONFIG)
        down = aim.solve(me, make_target(pos=(5.0, 0.0, -60.0)), "gun", GUN, CONFIG)
        self.assertAlmostEqual(up.pitch, 1.3, places=9)
        self.assertAlmostEqual(down.pitch, -0.7, places=9)

    def test_cube_scaled_target_size_and_muzzle(self):
        cfg = aim.Cfg(CONFIG)
        r0, h0 = aim.target_dims(make_target(cubes=0), cfg)
        r10, h10 = aim.target_dims(make_target(cubes=10), cfg)
        self.assertAlmostEqual(r10 / r0, 1.5)
        self.assertAlmostEqual(h10 / h0, 1.5)
        r30, _ = aim.target_dims(make_target(cubes=30), cfg)   # no cap
        self.assertAlmostEqual(r30 / r0, 2.5)
        # centre mass moves up with the scaled height
        me = make_me(aim_=(0.0, 0.0))
        sol = aim.solve(me, make_target(pos=(80.0, 0.0, 0.0), cubes=10), "gun", GUN, CONFIG)
        self.assertAlmostEqual(sol.aim_point[2], 6.75, places=6)
        # our own muzzle height comes from self.stats (scaled by our cubes)
        me10 = make_me(aim_=(0.0, 0.0), cubes=10)
        sol10 = aim.solve(me10, make_target(pos=(80.0, 0.0, 0.0)), "gun", GUN, CONFIG)
        self.assertAlmostEqual(sol10.pitch, math.atan2(4.5 - 10.5, 80.0), places=6)
        # a bigger target is hit with a bigger aim error: 0.03 rad at 100 m is 3 m off
        me_off = make_me(aim_=(0.03, 0.0))
        small = aim.solve(me_off, make_target(pos=(100.0, 0.0, 0.0), cubes=0), "gun", GUN, CONFIG)
        big = aim.solve(me_off, make_target(pos=(100.0, 0.0, 0.0), cubes=10), "gun", GUN, CONFIG)
        self.assertLess(small.hit_prob, 0.35)
        self.assertGreater(big.hit_prob, 0.35)
        self.assertFalse(aim.fire_decision(me_off, None, me_off["weapons"][0], small, CONFIG)[0])
        self.assertTrue(aim.fire_decision(me_off, None, me_off["weapons"][0], big, CONFIG)[0])

    def test_hit_probability_falls_with_range_and_aim_error(self):
        me = make_me(aim_=(0.0, 0.0))
        near = aim.solve(me, make_target(pos=(30.0, 0.0, 0.0)), "gun", GUN, CONFIG)
        far = aim.solve(me, make_target(pos=(200.0, 0.0, 0.0)), "gun", GUN, CONFIG)
        self.assertGreater(near.hit_prob, far.hit_prob)
        ps = [aim.solve(make_me(aim_=(e, 0.0)), make_target(pos=(100.0, 0.0, 0.0)), "gun", GUN, CONFIG).hit_prob
              for e in (0.0, 0.01, 0.02, 0.04, 0.08)]
        self.assertEqual(ps, sorted(ps, reverse=True))
        self.assertLess(ps[-1], 0.01)

    def test_rect_hit_prob_matches_known_values(self):
        # zero offset: erf(w / (s sqrt2)) * erf(h / (s sqrt2)); big offset: ~0; zero sigma: step
        e = math.erf(2.0 / (1.0 * math.sqrt(2.0)))
        self.assertAlmostEqual(aim.rect_hit_prob(0.0, 0.0, 1.0, 2.0, 2.0), e * e, places=9)
        self.assertLess(aim.rect_hit_prob(10.0, 0.0, 1.0, 2.0, 4.5), 1e-6)
        self.assertEqual(aim.rect_hit_prob(1.0, 4.0, 0.0, 2.0, 4.5), 1.0)
        self.assertEqual(aim.rect_hit_prob(3.0, 0.0, 0.0, 2.0, 4.5), 0.0)
        self.assertEqual(aim.rect_hit_prob(0.0, 0.0, 1.0, 0.0, 4.5), 0.0)
        # Monte Carlo cross-check of an offset case on a tall silhouette
        rng = random.Random(5)
        n, hit = 40000, 0
        for _ in range(n):
            if abs(1.5 + rng.gauss(0, 1.2)) < 2.5 and abs(3.0 + rng.gauss(0, 1.2)) < 4.5:
                hit += 1
        self.assertAlmostEqual(aim.rect_hit_prob(1.5, 3.0, 1.2, 2.5, 4.5), hit / n, delta=0.015)

    def test_rect_hit_prob_non_finite_input_is_zero(self):
        nan, inf = float("nan"), float("inf")
        for args in ((nan, 0.0, 1.0, 2.0, 4.5), (0.0, nan, 1.0, 2.0, 4.5), (0.0, 0.0, nan, 2.0, 4.5),
                     (0.0, 0.0, 1.0, nan, 4.5), (0.0, 0.0, 1.0, 2.0, inf), (inf, 0.0, 1.0, 2.0, 4.5),
                     (0.0, 0.0, inf, 2.0, 4.5), (None, 0.0, 1.0, 2.0, 4.5)):
            self.assertEqual(aim.rect_hit_prob(*args), 0.0, args)
        for args in ((nan, 1.0, 2.0), (0.0, nan, 2.0), (0.0, 1.0, nan), (inf, 1.0, 2.0), (0.0, inf, 2.0),
                     (0.0, 1.0, inf), (None, 1.0, 2.0)):
            self.assertEqual(aim._p1(*args), 0.0, args)

    def test_vertical_offset_inside_the_silhouette_still_hits(self):
        # level aim from the 7 m muzzle at a 9 m tall mech 100 m away crosses it 2.5 m above centre mass
        for b in BEARINGS:
            tgt = make_target(pos=(100.0 * math.cos(b), 100.0 * math.sin(b), 0.0))
            sol = aim.solve(make_me(aim_=(b, 0.0)), tgt, "gun", GUN, CONFIG)
            self.assertTrue(sol.inside, b)
            self.assertGreater(sol.hit_prob, 0.6, b)
            # a 5 m drop under the feet is outside the silhouette
            low = aim.solve(make_me(aim_=(b, -0.08)), tgt, "gun", GUN, CONFIG)
            self.assertFalse(low.inside, b)


# ---------------------------------------------------------------------------- target choice


class TestChooseTarget(unittest.TestCase):
    def test_prefers_wounded_charging_attacker_over_nearer_full_hp_mech(self):
        me = make_me()
        near = make_target(tid=1, pos=(30.0, 0.0, 0.0), aim_=(math.pi / 2, 0.0))
        # 120 m away, 30% HP, charging, aimed straight at us
        attacker = make_target(tid=2, pos=(0.0, 120.0, 0.0), hp=420, aim_=(-math.pi / 2, 0.0), charging=True)
        got = aim.choose_target(make_obs(me, [near, attacker]), me, aim.TargetState())
        self.assertEqual(got["id"], 2)

    def test_prefers_lower_hp_target_over_nearer_full_hp(self):
        me = make_me()
        near = make_target(tid=1, pos=(40.0, 0.0, 0.0))
        hurt = make_target(tid=2, pos=(0.0, 70.0, 0.0), hp=300)
        self.assertEqual(aim.choose_target(make_obs(me, [near, hurt]), me, aim.TargetState())["id"], 2)

    def test_aimed_at_us_and_recent_hit_raise_the_score(self):
        me = make_me()
        a = make_target(tid=1, pos=(60.0, 0.0, 0.0), aim_=(0.0, 0.0))                 # looking away
        b = make_target(tid=2, pos=(0.0, 70.0, 0.0), aim_=(-math.pi / 2 + 0.05, 0.0))  # looking at us
        self.assertEqual(aim.choose_target(make_obs(me, [a, b]), me, aim.TargetState())["id"], 2)
        st = aim.TargetState()
        ev = [{"type": "damaged", "amount": 90, "weapon": "gun", "bearing": 0.0, "by": 1}]
        self.assertEqual(aim.choose_target(make_obs(me, [a, b], t=5.0, events=ev), me, st)["id"], 1)
        # 3 s later the hit no longer counts
        self.assertEqual(aim.choose_target(make_obs(me, [a, b], t=8.5), me, aim.TargetState())["id"], 2)
        st2 = aim.TargetState()
        st2.last_hit[1] = 5.0
        self.assertEqual(aim.choose_target(make_obs(me, [a, b], t=7.9), me, st2)["id"], 1)

    def test_target_state_clears_when_time_goes_backwards(self):
        me = make_me()
        st = aim.TargetState()
        a = make_target(tid=4, pos=(60.0, 0.0, 0.0))
        b = make_target(tid=5, pos=(60.0, 5.0, 0.0))
        aim.choose_target(make_obs(me, [a, b], t=50.0, events=[{"type": "damaged", "amount": 45, "by": 5}]), me, st)
        self.assertIn(5, st.last_hit)
        self.assertEqual(st.current, 5)
        aim.choose_target(make_obs(me, [a], t=1.0), me, st)     # a new match: t restarts
        self.assertNotIn(5, st.last_hit)
        got = aim.choose_target(make_obs(me, [a, b], t=1.1), me, st)
        self.assertEqual(st.last_hit, {})
        self.assertIsNotNone(got)

    def test_sticky_target_no_thrash(self):
        me = make_me()
        st = aim.TargetState()
        picks = []
        for k in range(60):  # two similar mechs; their ranges wobble past each other every tick
            wob = 2.0 if k % 2 == 0 else -2.0
            a = make_target(tid=1, pos=(60.0 + wob, 0.0, 0.0))
            b = make_target(tid=2, pos=(0.0, 60.0 - wob, 0.0))
            picks.append(aim.choose_target(make_obs(me, [a, b], t=1.0 + 0.1 * k), me, st)["id"])
        self.assertEqual(len(set(picks)), 1, picks)
        # a clearly better target (30% HP) takes over: +25% is not enough to keep the old one
        a = make_target(tid=1, pos=(60.0, 0.0, 0.0))
        b = make_target(tid=2, pos=(0.0, 60.0, 0.0), hp=420)
        self.assertEqual(aim.choose_target(make_obs(me, [a, b], t=9.0), me, st)["id"], 2)

    def test_keeps_current_unless_rival_scores_25_percent_higher(self):
        me = make_me()
        st = aim.TargetState()
        a = make_target(tid=1, pos=(60.0, 0.0, 0.0))
        b = make_target(tid=2, pos=(0.0, 60.0, 0.0), hp=1330)   # 5% wounded: score +0.05 on ~0.7
        aim.choose_target(make_obs(me, [a]), me, st)
        self.assertEqual(st.current, 1)
        self.assertEqual(aim.choose_target(make_obs(me, [a, b]), me, st)["id"], 1)
        st.current = 2
        self.assertEqual(aim.choose_target(make_obs(me, [a, b]), me, st)["id"], 2)

    def test_ignores_dead_npcs_and_self_and_handles_empty(self):
        me = make_me()
        st = aim.TargetState()
        self.assertIsNone(aim.choose_target(make_obs(me, []), me, st))
        self.assertIsNone(aim.choose_target({"t": 1.0, "self": me}, me, st))
        dead = make_target(tid=1, hp=0)
        npc = make_target(tid=101, kind="npc", pos=(20.0, 0.0, 0.0), hp=350, max_hp=350)
        me_echo = make_target(tid=3, pos=(5.0, 0.0, 0.0))
        self.assertIsNone(aim.choose_target(make_obs(me, [dead, npc, me_echo]), me, st))
        self.assertEqual(aim.choose_target(make_obs(me, [dead, npc]), me, st, include_npcs=True)["id"], 101)
        nan = make_target(tid=4, pos=(float("nan"), 0.0, 0.0))
        self.assertIsNone(aim.choose_target(make_obs(me, [nan]), me, st))
        nohp = {"id": 9, "kind": "mech", "pos": [50.0, 0.0, 0.0], "vel": [0, 0, 0]}
        self.assertEqual(aim.choose_target(make_obs(me, [nohp]), me, st)["id"], 9)

    def test_memory_only_target_is_chosen_for_memory_s_then_dropped(self):
        me = make_me()
        tr = aim.Tracker(CONFIG)
        st = aim.TargetState()
        tg = make_target(tid=7, pos=(70.0, 0.0, 0.0), vel=(0.0, 8.0, 0.0))
        tr.update(make_obs(me, [tg], t=1.0))
        tr.update(make_obs(me, [], t=1.1))
        got = aim.choose_target(make_obs(me, [], t=3.0), me, st, tr)
        self.assertIsNotNone(got)
        self.assertEqual(got["id"], 7)
        self.assertTrue(got["stale"])
        self.assertTrue(all(math.isfinite(c) for c in got["pos"]))
        tr.update(make_obs(me, [], t=7.5))
        self.assertIsNone(aim.choose_target(make_obs(me, [], t=7.5), me, aim.TargetState(), tr))


# ---------------------------------------------------------------------------- fire gate


class TestFireGate(unittest.TestCase):
    def _gun(self, me, tgt, los=None, **kw):
        sol = aim.solve(me, tgt, "gun", GUN, CONFIG)
        return aim.fire_decision(me, tgt, me["weapons"][0], sol, CONFIG, los, **kw)

    def test_fire_gate_uses_hit_probability_not_fixed_tolerance(self):
        # big close target, 0.08 rad (0.8 m at 10 m) off: fire. small far one, 0.02 rad (3 m at 150 m): hold.
        close = self._gun(make_me(aim_=(0.08, 0.0)), make_target(pos=(10.0, 0.0, 0.0)))
        self.assertTrue(close[0], close)
        far = self._gun(make_me(aim_=(0.02, 0.0)), make_target(pos=(150.0, 0.0, 0.0), kind="npc"))
        self.assertFalse(far[0], far)
        self.assertEqual(far[1], "p")

    def test_no_fire_through_blocked_los_and_fire_when_clear(self):
        me = make_me(aim_=(0.0, 0.0))
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        calls = []

        def blocked(a, b):
            calls.append((a, b))
            return False

        self.assertEqual(self._gun(me, tgt, blocked), (False, "blocked"))
        self.assertEqual(self._gun(me, tgt, lambda a, b: True), (True, "fire"))
        self.assertEqual(self._gun(me, tgt, None), (True, "fire"))   # default: assume clear
        # the segment tested is muzzle (feet z + 7) to the aim point (centre mass)
        a, b = calls[0]
        self.assertAlmostEqual(a[2], 7.0)
        self.assertAlmostEqual(b[2], 4.5)

    def test_plan_shots_no_fire_through_blocked_los_for_every_weapon(self):
        me = make_me(aim_=(0.0, 0.0), weapons=[None, laser_weapon()])
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        out = aim.plan_shots(make_obs(me, [tgt]), me, tgt, None, CONFIG, lambda a, b: False)
        self.assertEqual(out["fire"], [False, False])
        out = aim.plan_shots(make_obs(me, [tgt]), me, tgt, None, CONFIG, lambda a, b: True)
        self.assertEqual(out["fire"], [False, True])   # laser starts charging

    def test_fire_gate_yields_to_exposure_deadline_and_hide(self):
        me = make_me(aim_=(0.0, 0.0))
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        sol = aim.solve(me, tgt, "gun", GUN, CONFIG)
        w = me["weapons"][0]
        self.assertTrue(aim.fire_decision(me, tgt, w, sol, CONFIG)[0])
        self.assertEqual(aim.fire_decision(me, tgt, w, sol, CONFIG, hide=True), (False, "hide"))
        self.assertEqual(aim.fire_decision(me, tgt, w, sol, CONFIG, deadline=10.0, now=10.0), (False, "deadline"))
        self.assertTrue(aim.fire_decision(me, tgt, w, sol, CONFIG, deadline=10.0, now=9.9)[0])
        out = aim.plan_shots(make_obs(me, [tgt], t=10.5), me, tgt, None, CONFIG, None, deadline=10.0)
        self.assertEqual(out["fire"], [False, False])
        out = aim.plan_shots(make_obs(me, [tgt]), me, tgt, None, CONFIG, None, hide=True)
        self.assertEqual(out["fire"], [False, False])

    def test_gun_holds_fire_with_no_ammo_or_out_of_range(self):
        me = make_me(aim_=(0.0, 0.0))
        me["weapons"][0]["ammo"] = 0
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        self.assertEqual(self._gun(me, tgt), (False, "ammo"))
        self.assertTrue(aim.needs_reload(me["weapons"][0]))
        me2 = make_me(aim_=(0.0, 0.0))
        self.assertEqual(self._gun(me2, make_target(pos=(300.0, 0.0, 0.0))), (False, "range"))

    def test_laser_releases_only_when_actual_aim_within_target_radius(self):
        tgt = make_target(pos=(100.0, 0.0, 0.0))
        # the COMMANDED aim is perfect in both cases; only self.aim differs
        off = make_me(aim_=(0.04, 0.0), weapons=[None, laser_weapon(charge=0.8)])    # 4 m off at 100 m
        on = make_me(aim_=(0.008, 0.0), weapons=[None, laser_weapon(charge=0.8)])    # 0.8 m off
        s_off = aim.solve(off, tgt, "laser", LASER, CONFIG)
        s_on = aim.solve(on, tgt, "laser", LASER, CONFIG)
        self.assertLess(abs(s_off.yaw), 0.001)       # commanded aim is on the target
        self.assertFalse(s_off.inside)
        self.assertTrue(s_on.inside)
        self.assertEqual(aim.fire_decision(off, tgt, off["weapons"][1], s_off, CONFIG), (True, "charging"))
        self.assertEqual(aim.fire_decision(on, tgt, on["weapons"][1], s_on, CONFIG), (False, "release"))

    def test_laser_charge_cycle_and_hold_limit(self):
        tgt = make_target(pos=(100.0, 0.0, 0.0))
        for charge, want in ((0.0, (True, "charge")), (0.1, (True, "charging")), (0.5, (True, "charging")),
                             (0.8, (False, "release")), (2.7, (False, "release_hold_limit"))):
            me = make_me(aim_=(0.0, 0.0), weapons=[None, laser_weapon(charge=charge)])
            sol = aim.solve(me, tgt, "laser", LASER, CONFIG)
            self.assertEqual(aim.fire_decision(me, tgt, me["weapons"][1], sol, CONFIG), want, charge)
        # off target at the hold limit: release anyway (the server fires it on its own)
        me = make_me(aim_=(0.2, 0.0), weapons=[None, laser_weapon(charge=2.7)])
        sol = aim.solve(me, tgt, "laser", LASER, CONFIG)
        self.assertEqual(aim.fire_decision(me, tgt, me["weapons"][1], sol, CONFIG), (False, "release_hold_limit"))
        # cooling down: do not start
        me = make_me(aim_=(0.0, 0.0), weapons=[None, laser_weapon(cooldown=1.0)])
        sol = aim.solve(me, tgt, "laser", LASER, CONFIG)
        self.assertEqual(aim.fire_decision(me, tgt, me["weapons"][1], sol, CONFIG), (False, "cooldown"))

    def test_laser_waits_for_full_charge_on_target(self):
        tgt = make_target(pos=(100.0, 0.0, 0.0))
        full = LASER.get("charge", 0.75)
        for charge, want in ((0.92 * full, (True, "charging")), (full, (False, "release"))):
            me = make_me(aim_=(0.0, 0.0), weapons=[None, laser_weapon(charge=charge)])
            sol = aim.solve(me, tgt, "laser", LASER, CONFIG)
            self.assertEqual(aim.fire_decision(me, tgt, me["weapons"][1], sol, CONFIG), want, charge)

    def test_laser_releases_across_the_pi_wrap_on_actual_aim(self):
        for b in BEARINGS:
            tgt = make_target(pos=(100.0 * math.cos(b), 100.0 * math.sin(b), 0.0))
            for d_aim, want in ((0.008, (False, "release")), (-0.008, (False, "release")),
                                (0.04, (True, "charging")), (-0.04, (True, "charging"))):
                me = make_me(aim_=(wrap(b + d_aim), 0.0), weapons=[None, laser_weapon(charge=0.8)])
                sol = aim.solve(me, tgt, "laser", LASER, CONFIG)
                self.assertEqual(aim.fire_decision(me, tgt, me["weapons"][1], sol, CONFIG), want, (b, d_aim))

    def test_decide_releases_before_auto_fire_when_charge_stays_at_full(self):
        # the reported charge may stop at `charge` (0.75 s): time the hold from when it got there
        aim.init({"config": CONFIG})
        tgt = make_target(pos=(100.0, 0.0, 0.0))
        rel_t = None
        for i in range(60):
            t = 1.0 + 0.1 * i
            me = make_me(aim_=(0.2, 0.0), weapons=[None, laser_weapon(charge=0.75)])   # off target
            _, f, _ = aim.decide(make_obs(me, [tgt], t=t), lambda a, b: True)
            if not f[1]:
                rel_t = t
                break
        self.assertIsNotNone(rel_t)
        self.assertEqual(aim.last_plan["reasons"][1], "release_hold_limit")
        self.assertGreaterEqual(rel_t, 1.0 + 2.0 - 0.2 - 1e-9)
        self.assertLess(rel_t, 1.0 + 2.0 + 1e-9)
        # a swap in that slot (or charge back to 0) clears the timer
        aim.decide(make_obs(make_me(aim_=(0.2, 0.0), weapons=[None, lightning_weapon()]), [tgt], t=7.0), None)
        me = make_me(aim_=(0.2, 0.0), weapons=[None, laser_weapon(charge=0.75)])
        _, f, _ = aim.decide(make_obs(me, [tgt], t=7.1), lambda a, b: True)
        self.assertTrue(f[1])
        self.assertEqual(aim.last_plan["reasons"][1], "charging")

    def test_charging_laser_keeps_holding_under_hide_deadline_and_no_target(self):
        tgt = make_target(pos=(100.0, 0.0, 0.0))
        cases = (("hide", dict(hide=True), [tgt], 1.0), ("deadline", dict(deadline=0.5), [tgt], 1.0),
                 ("no target", {}, [], 1.0))
        for name, kw, mechs, t0 in cases:
            with self.subTest(case=name):
                aim.init({"config": CONFIG})
                me = make_me(aim_=(0.2, 0.0), weapons=[make_me()["weapons"][0], laser_weapon(charge=0.5)])
                _, f, _ = aim.decide(make_obs(me, mechs, t=t0), lambda a, b: True, **kw)
                self.assertEqual(f, [False, True])            # gun quiet, laser held (no release)
                # held at full charge: the hold-limit logic still lets go before the auto-fire
                rel = None
                for i in range(1, 40):
                    me = make_me(aim_=(0.2, 0.0), weapons=[make_me()["weapons"][0], laser_weapon(charge=0.75)])
                    _, f, _ = aim.decide(make_obs(me, mechs, t=t0 + 0.1 * i), lambda a, b: True, **kw)
                    self.assertFalse(f[0])
                    if not f[1]:
                        rel = t0 + 0.1 * i
                        break
                self.assertIsNotNone(rel)
                self.assertEqual(aim.last_plan["reasons"][1], "release_hold_limit")
                # not charging: hide / deadline / no target never start a charge
                me = make_me(aim_=(0.0, 0.0), weapons=[None, laser_weapon(charge=0.0)])
                _, f, _ = aim.decide(make_obs(me, mechs, t=10.0), lambda a, b: True, **kw)
                self.assertEqual(f, [False, False])
        aim.init(None)

    def test_laser_blocked_never_starts_and_never_releases_on_a_hit_claim(self):
        tgt = make_target(pos=(100.0, 0.0, 0.0))
        me = make_me(aim_=(0.0, 0.0), weapons=[None, laser_weapon()])
        sol = aim.solve(me, tgt, "laser", LASER, CONFIG)
        self.assertEqual(aim.fire_decision(me, tgt, me["weapons"][1], sol, CONFIG, lambda a, b: False),
                         (False, "blocked"))
        me2 = make_me(aim_=(0.0, 0.0), weapons=[None, laser_weapon(charge=0.8)])
        sol2 = aim.solve(me2, tgt, "laser", LASER, CONFIG)
        self.assertEqual(aim.fire_decision(me2, tgt, me2["weapons"][1], sol2, CONFIG, lambda a, b: False),
                         (True, "charging"))

    def test_lightning_only_when_target_is_nearest_in_cone_and_energy_over_20(self):
        me = make_me(aim_=(0.0, 0.0), weapons=[None, lightning_weapon()])
        tgt = make_target(pos=(30.0, 0.0, 0.0))

        def lg(mechs=(), crates=(), energy=100.0, los=None, aim_yaw=0.0):
            me["energy"] = energy
            me["aim"] = [aim_yaw, 0.0]
            others = list(mechs) + list(crates) + [tgt]
            sol = aim.solve(me, tgt, "lightning", LIGHTNING, CONFIG, others=others, los_fn=los)
            return aim.fire_decision(me, tgt, me["weapons"][1], sol, CONFIG, los)

        self.assertEqual(lg(), (True, "fire"))
        self.assertEqual(lg(energy=20.0), (False, "energy"))
        self.assertTrue(lg(energy=20.5)[0])
        crate = {"id": 1007, "pos": [20.0, 3.0, 0.0], "hp": 300}
        self.assertEqual(lg(crates=[crate]), (False, "not_nearest_in_cone"))
        npc = make_target(tid=101, kind="npc", pos=(15.0, -4.0, 0.0))
        self.assertEqual(lg(mechs=[npc]), (False, "not_nearest_in_cone"))
        side = {"id": 1008, "pos": [0.0, 20.0, 0.0], "hp": 300}      # 90 deg off: outside the cone
        self.assertTrue(lg(crates=[side])[0])
        behind_wall = lg(crates=[crate], los=lambda a, b: not (b[0] < 25 and b[0] > 15))
        self.assertTrue(behind_wall[0])                               # the nearer crate has no line of sight
        self.assertFalse(lg(aim_yaw=0.6)[0])                          # target 34 deg off our aim: out of the cone
        far = make_target(tid=11, pos=(60.0, 0.0, 0.0))
        sol = aim.solve(me, far, "lightning", LIGHTNING, CONFIG, others=[far])
        self.assertEqual(sol.hit_prob, 0.0)                           # beyond 45 m

    def test_unsupported_weapon_never_fires(self):
        me = make_me(aim_=(0.0, 0.0), weapons=[None, {"kind": "mortar", "cooldown": 0.0, "stats": {"speed": 72}}])
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        out = aim.plan_shots(make_obs(me, [tgt]), me, tgt, None, CONFIG, None)
        self.assertEqual(out["fire"], [False, False])


# ---------------------------------------------------------------------------- tracker


class TestTracker(unittest.TestCase):
    def feed(self, tr, mechs_by_t):
        me = make_me()
        for t, mechs in mechs_by_t:
            tr.update(make_obs(me, mechs, t=t))

    def test_constant_velocity_predict_linear_and_damped(self):
        for method, k in (("linear", 1.0), ("damped", aim.DEFAULT_DAMP)):
            tr = aim.Tracker(CONFIG, method=method)
            self.feed(tr, [(0.1 * i, [make_target(pos=(50.0 + i, 0.0, 0.0), vel=(10.0, 0.0, 0.0))]) for i in range(6)])
            p = tr.predict(10, 0.5)
            self.assertAlmostEqual(p[0], 55.0 + k * 5.0, places=6)
            self.assertAlmostEqual(p[1], 0.0, places=6)
            self.assertEqual(tr.predict(10, 0.0)[:2], (55.0, 0.0))

    def test_smoothed_velocity_averages_last_obs(self):
        tr = aim.Tracker(CONFIG, smooth_n=4)
        vs = [(8.0, 0.0), (12.0, 0.0), (9.0, 0.0), (11.0, 0.0), (10.0, 0.0)]
        self.feed(tr, [(0.1 * i, [make_target(pos=(50.0, 0.0, 0.0), vel=(vx, vy, 0.0))]) for i, (vx, vy) in enumerate(vs)])
        self.assertAlmostEqual(tr.velocity(10)[0], (12.0 + 9.0 + 11.0 + 10.0) / 4.0)

    def test_acceleration_estimate_is_capped_at_ground_accel(self):
        tr = aim.Tracker(CONFIG)
        self.feed(tr, [(0.1 * i, [make_target(pos=(50.0, 0.0, 0.0), vel=(40.0 * i, 0.0, 0.0))]) for i in range(5)])
        ax, ay = tr.accel(10)
        self.assertLessEqual(math.hypot(ax, ay), 58.5 + 1e-9)
        self.assertGreater(ax, 50.0)
        # a real ramp at 30 m/s^2 is recovered
        tr2 = aim.Tracker(CONFIG)
        self.feed(tr2, [(0.1 * i, [make_target(pos=(50.0, 0.0, 0.0), vel=(3.0 * i, 0.0, 0.0))]) for i in range(5)])
        self.assertAlmostEqual(tr2.accel(10)[0], 30.0, places=6)

    def test_accel_predictor_does_not_run_backwards_when_braking(self):
        h = [(0.1 * i, 50.0, 0.0, 0.0, 10.0 - 2.4 * i, 0.0, 0.0) for i in range(5)]   # slowing to a stop (0.4 m/s)
        x, y, z = aim.extrapolate(h, 2.0, "accel")
        self.assertGreaterEqual(x, h[-1][1] - 1e-9)
        self.assertLess(x, h[-1][1] + 2.0)

    def test_circular_predictor_follows_an_arc(self):
        R, w, v = 20.0, 0.6, 12.0
        h = [(0.1 * i, R * math.cos(w * 0.1 * i), R * math.sin(w * 0.1 * i), 0.0,
              -R * w * math.sin(w * 0.1 * i), R * w * math.cos(w * 0.1 * i), 0.0) for i in range(8)]
        x, y, _ = aim.extrapolate(h, 0.8, "circular", n=6)
        t = 0.7 + 0.8
        self.assertLess(math.hypot(x - R * math.cos(w * t), y - R * math.sin(w * t)), 0.6)
        xl, yl, _ = aim.extrapolate(h, 0.8, "linear")
        self.assertGreater(math.hypot(xl - R * math.cos(w * t), yl - R * math.sin(w * t)), 0.9)

    def test_ids_appear_and_disappear(self):
        tr = aim.Tracker(CONFIG)
        me = make_me()
        a = make_target(tid=5, pos=(50.0, 0.0, 0.0), vel=(10.0, 0.0, 0.0))
        b = make_target(tid=6, pos=(0.0, 60.0, 0.0))
        tr.update(make_obs(me, [a, b], t=1.0))
        self.assertTrue(tr.has(5) and tr.has(6))
        tr.update(make_obs(me, [b], t=1.1))
        self.assertTrue(tr.get(5).stale)
        self.assertFalse(tr.get(6).stale)
        p = tr.predict(5, 0.5)                      # memory only: half-velocity dead reckoning, bounded
        self.assertTrue(all(math.isfinite(c) for c in p))
        self.assertLess(abs(p[0] - 50.0), 4.0)
        self.assertGreater(tr.age(5), 0.0)
        self.assertIsNone(tr.predict(999, 0.5))
        tr.update(make_obs(me, [a, b], t=1.4))      # reappears: live again, history continues
        self.assertFalse(tr.get(5).stale)
        tr.update(make_obs(me, [b], t=2.0))
        tr.update(make_obs(me, [b], t=6.3))         # last seen 1.4, 4.9 s unseen: still kept
        self.assertTrue(tr.has(5))
        tr.update(make_obs(me, [b], t=6.5))         # past memory_s (5 s) unseen: dropped
        self.assertFalse(tr.has(5))
        self.assertTrue(tr.has(6))

    def test_dt_zero_replaces_the_sample_and_stays_finite(self):
        tr = aim.Tracker(CONFIG)
        me = make_me()
        tr.update(make_obs(me, [make_target(pos=(50.0, 0.0, 0.0), vel=(10.0, 0.0, 0.0))], t=1.0))
        tr.update(make_obs(me, [make_target(pos=(50.5, 0.0, 0.0), vel=(10.0, 0.0, 0.0))], t=1.0))
        self.assertEqual(len(tr.get(10).hist), 1)
        for m in aim.METHODS:
            p = tr.predict(10, 0.4, m)
            self.assertTrue(all(math.isfinite(c) for c in p), m)
        self.assertEqual(tr.accel(10), (0.0, 0.0))

    def test_teleport_and_long_gap_restart_the_history(self):
        tr = aim.Tracker(CONFIG)
        me = make_me()
        for i in range(5):
            tr.update(make_obs(me, [make_target(pos=(50.0 + i, 0.0, 0.0), vel=(10.0, 0.0, 0.0))], t=0.1 * i))
        self.assertEqual(len(tr.get(10).hist), 5)
        tr.update(make_obs(me, [make_target(pos=(350.0, 200.0, 0.0), vel=(0.0, -8.0, 0.0))], t=0.5))  # respawn-like
        self.assertEqual(len(tr.get(10).hist), 1)
        p = tr.predict(10, 1.0, "linear")
        self.assertAlmostEqual(p[0], 350.0)
        self.assertAlmostEqual(p[1], 192.0)
        tr.update(make_obs(me, [make_target(pos=(350.0, 192.0, 0.0), vel=(0.0, -8.0, 0.0))], t=2.5))   # 2 s gap
        self.assertEqual(len(tr.get(10).hist), 1)
        tr.update(make_obs(me, [make_target(pos=(350.0, 192.0, 0.0))], t=0.2))                          # time went back
        self.assertEqual(len(tr.get(10).hist), 1)

    def test_garbage_observations_are_ignored(self):
        tr = aim.Tracker(CONFIG)
        me = make_me()
        bad = [make_target(tid=1, pos=(float("nan"), 0.0, 0.0)), {"id": None, "pos": [1, 2, 3]},
               {"id": 2, "pos": [1, 2]}, make_target(tid=3, vel=(float("inf"), 0.0, 0.0))]
        tr.update(make_obs(me, bad, t=1.0))
        self.assertFalse(tr.has(1))
        self.assertFalse(tr.has(2))
        self.assertTrue(tr.has(3))
        self.assertEqual(tr.get(3).hist[-1][4:], (0.0, 0.0, 0.0))
        tr.update({"t": float("nan"), "mechs": []})
        tr.update({"t": 2.0})
        for m in aim.METHODS:
            self.assertTrue(all(math.isfinite(c) for c in tr.predict(3, 0.3, m)))


# ---------------------------------------------------------------------------- degenerate inputs, fuzz, integration


class TestDegenerate(unittest.TestCase):
    def _finite_solution(self, sol):
        for v in (sol.yaw, sol.pitch, sol.t_on_target, sol.hit_prob, sol.p_aimed, sol.dist, sol.miss_m, sol.flight,
                  *sol.aim_point):
            self.assertTrue(math.isfinite(v), v)
        self.assertTrue(0.0 <= sol.hit_prob <= 1.0)

    def test_zero_distance(self):
        me = make_me(aim_=(1.0, 0.2))
        tgt = make_target(pos=(0.0, 0.0, 0.0))
        for kind, stats in (("gun", GUN), ("laser", LASER), ("lightning", LIGHTNING)):
            sol = aim.solve(me, tgt, kind, stats, CONFIG)
            self._finite_solution(sol)
            self.assertAlmostEqual(sol.yaw, 1.0, places=9)   # no bearing exists: keep the current yaw

    def test_target_directly_above_and_below(self):
        me = make_me(aim_=(0.5, 0.0))
        for z in (60.0, -60.0, 12.0):
            sol = aim.solve(me, make_target(pos=(0.0, 0.0, z)), "gun", GUN, CONFIG)
            self._finite_solution(sol)
            self.assertAlmostEqual(sol.yaw, 0.5, places=9)
            self.assertLessEqual(sol.pitch, 1.3 + 1e-12)
            self.assertGreaterEqual(sol.pitch, -0.7 - 1e-12)
        self.assertLess(aim.solve(me, make_target(pos=(0.0, 0.0, 60.0)), "gun", GUN, CONFIG).hit_prob, 0.01)

    def test_missing_fields_and_odd_weapon_stats(self):
        me = {"id": 3, "pos": [0, 0, 0], "aim": [0.0, 0.0]}
        tgt = {"id": 1, "pos": [50.0, 5.0, 0.0]}
        sol = aim.solve(me, tgt, "gun", None, CONFIG)
        self._finite_solution(sol)
        sol = aim.solve(me, tgt, "gun", {"speed": 0, "spread_deg": None}, None)
        self._finite_solution(sol)
        sol = aim.solve(me, tgt, "mortar", {"speed": 72}, CONFIG)
        self.assertEqual(sol.hit_prob, 0.0)

    def test_fuzz_never_raises_and_stays_finite(self):
        rng = random.Random(42)
        for _ in range(400):
            me = make_me(pos=(rng.uniform(-300, 300), rng.uniform(-300, 300), rng.uniform(-10, 40)),
                         aim_=(rng.uniform(-7, 7), rng.uniform(-0.7, 1.3)),
                         vel=(rng.uniform(-19, 19), rng.uniform(-19, 19), 0.0), cubes=rng.choice([0, 0, 3, 12]))
            tgt = make_target(pos=(me["pos"][0] + rng.choice([0, rng.uniform(-250, 250)]),
                                   me["pos"][1] + rng.choice([0, rng.uniform(-250, 250)]), rng.uniform(-10, 80)),
                              vel=(rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-11, 11)),
                              cubes=rng.choice([0, 5, 40]), kind=rng.choice(["mech", "npc"]))
            for kind, stats in (("gun", GUN), ("laser", LASER), ("lightning", LIGHTNING)):
                for method in (None, "linear", "damped", "smoothed", "accel", "circular"):
                    self._finite_solution(aim.solve(me, tgt, kind, stats, CONFIG, method=method))

    def test_decide_integration_and_never_raises(self):
        aim.init({"config": CONFIG})
        me = make_me(aim_=(0.3, 0.0))
        tgt = make_target(pos=(80.0, 10.0, 0.0), vel=(0.0, 9.0, 0.0))
        a, f, t = aim.decide(make_obs(me, [tgt]), lambda p, q: True)
        self.assertEqual(t["id"], 10)
        self.assertEqual(len(a), 2)
        self.assertTrue(all(math.isfinite(c) for c in a))
        self.assertEqual(len(f), 2)
        self.assertTrue(all(isinstance(x, bool) for x in f))
        self.assertNotIn("error", aim.last_plan)
        a, f, t = aim.decide(make_obs(me, []), None)
        self.assertEqual((a, f, t), (None, [False, False], None))
        a, f, t = aim.decide({"t": 1.0}, None)                       # garbage: no self
        self.assertEqual((a, f, t), (None, [False, False], None))
        self.assertIn("error", aim.last_plan)
        aim.init(None)
        a, f, t = aim.decide(make_obs(make_me(), [make_target()]), None)
        self.assertIsNotNone(t)

    def test_decide_hide_and_los_hooks(self):
        aim.init({"config": CONFIG})
        me = make_me(aim_=(0.0, 0.0))
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        self.assertEqual(aim.decide(make_obs(me, [tgt]), lambda a, b: True)[1], [True, False])
        self.assertEqual(aim.decide(make_obs(me, [tgt]), lambda a, b: False)[1], [False, False])
        self.assertEqual(aim.decide(make_obs(me, [tgt]), lambda a, b: True, hide=True)[1], [False, False])
        self.assertEqual(aim.decide(make_obs(me, [tgt], t=3.0), lambda a, b: True, deadline=2.5)[1], [False, False])

    def test_decide_drives_aim_with_laser_then_lightning_then_gun(self):
        aim.init({"config": CONFIG})
        tgt = make_target(pos=(100.0, 0.0, 0.0))
        me = make_me(aim_=(0.0, 0.0), weapons=[make_me()["weapons"][0], laser_weapon()])
        a, f, _ = aim.decide(make_obs(me, [tgt]), None)
        self.assertEqual(aim.last_plan["reasons"][1], "charge")
        self.assertTrue(f[1])
        # lightning in arc range takes the aim and fires; gun and lightning together
        near = make_target(pos=(25.0, 0.0, 0.0))
        me2 = make_me(aim_=(0.0, 0.0), weapons=[make_me()["weapons"][0], lightning_weapon()])
        a, f, _ = aim.decide(make_obs(me2, [near]), None)
        self.assertTrue(f[1])
        self.assertTrue(f[0])


try:
    import ray as _ray
    _REAL_MAP = _ray.Ray()
    _REAL_DIR = _ray._find_map_dir()
except Exception:  # no map files here
    _REAL_MAP = _REAL_DIR = None


class TestWithRealLineOfFire(unittest.TestCase):
    def _world(self, *prisms):
        import ray
        return ray.Ray(world={"prisms": list(prisms)})

    def test_no_fire_through_building_gun_and_laser(self):
        wall = {"kind": "building", "base": 0, "top": 25, "ring": [30, -20, 40, -20, 40, 20, 30, 20]}
        r = self._world(wall)
        me = make_me(aim_=(0.0, 0.0), weapons=[make_me()["weapons"][0], laser_weapon(charge=0.8)])
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        out = aim.plan_shots(make_obs(me, [tgt]), me, tgt, None, CONFIG, r.clear)
        self.assertEqual(out["fire"], [False, True])                # gun blocked; laser only keeps holding
        self.assertEqual(out["reasons"][0], "blocked")
        self.assertNotEqual(out["reasons"][1], "release")
        me2 = make_me(aim_=(0.0, 0.0), weapons=[make_me()["weapons"][0], laser_weapon(charge=0.0)])
        out2 = aim.plan_shots(make_obs(me2, [tgt]), me2, tgt, None, CONFIG, r.clear)
        self.assertEqual(out2["fire"], [False, False])              # no shots, no laser charge started
        clear_world = self._world()
        out3 = aim.plan_shots(make_obs(me2, [tgt]), me2, tgt, None, CONFIG, clear_world.clear)
        self.assertEqual(out3["fire"], [True, True])

    def test_muzzle_blocked_by_low_wall_we_see_over(self):
        wall = {"kind": "building", "base": 0, "top": 7.5, "ring": [30, -20, 31, -20, 31, 20, 30, 20]}
        r = self._world(wall)
        self.assertFalse(r.clear((0, 0, 7.0), (60, 0, 4.5)))       # muzzle 7 m: the wall is in the way
        self.assertTrue(r.clear((0, 0, 8.0), (60, 0, 8.0)))        # eye 8 m sees over it
        me = make_me(aim_=(0.0, 0.0))
        tgt = make_target(pos=(60.0, 0.0, 0.0))
        out = aim.plan_shots(make_obs(me, [tgt]), me, tgt, None, CONFIG, r.clear)
        self.assertEqual(out["fire"], [False, False])
        self.assertEqual(out["reasons"][0], "blocked")

    def test_fires_through_trees(self):
        r = self._world()   # trees are not prisms: they never reach the line-of-fire test
        import ray
        rt = ray.Ray(world={"prisms": [], "trees": {"fields": ["x", "y", "z", "height", "crownR", "crownBase", "trunkR"],
                                                    "data": [40, 0, 0, 20, 6, 8, 1.0]}})
        me = make_me(aim_=(0.0, 0.0))
        tgt = make_target(pos=(80.0, 0.0, 0.0))
        out = aim.plan_shots(make_obs(me, [tgt]), me, tgt, None, CONFIG, rt.clear)
        self.assertEqual(out["fire"], [True, False])
        self.assertTrue(r.clear((0, 0, 7), (80, 0, 4.5)))

    @unittest.skipIf(_REAL_MAP is None, "map files not found")
    def test_init_loads_the_map_from_hello_and_decide_uses_it_when_los_is_none(self):
        import ray
        if _REAL_MAP is None:
            self.skipTest("kit map not found")
        try:
            # level shot from inside a building's footprint height band: blocked on the real map
            src = None
            for p in _REAL_MAP.prisms:
                if p["kind"] == "building" and p["top"] - p["base"] > 15 and not p["holes"]:
                    x0, y0, x1, y1 = p["bbox"]
                    if x1 - x0 > 20:
                        src = (x0 - 30.0, 0.5 * (y0 + y1), p["base"], x1 + 30.0)
                        break
            self.assertIsNotNone(src)
            x, y, z, xb = src
            me = make_me(pos=(x, y, z), aim_=(0.0, 0.0))
            tgt = make_target(pos=(xb, y, z))
            if _REAL_MAP.clear((x, y, z + 7.0), (xb, y, z + 4.5)):
                self.skipTest("picked segment is clear")
            aim.init({"config": CONFIG})                        # no map dir: None means assume clear
            aim.decide(make_obs(me, [tgt]), None)
            self.assertNotEqual(aim.last_plan["reasons"][0], "blocked")
            aim.init({"config": CONFIG, "map": {"dir": _REAL_DIR}})
            aim.decide(make_obs(me, [tgt]), None)
            self.assertEqual(aim.last_plan["reasons"][0], "blocked")
            aim.init({"config": CONFIG, "map": {"dir": "/no/such/dir"}})   # a bad mount never raises
            a, f, t = aim.decide(make_obs(me, [tgt]), None)
            self.assertNotIn("error", aim.last_plan)
        finally:
            aim.init(None)
            ray.init()

    def test_decide_with_the_real_map_los(self):
        import ray
        if _REAL_MAP is None:
            self.skipTest("kit map not found")
        ray.init()
        aim.init({"config": CONFIG})
        me = make_me(pos=(0.0, 0.0, 40.0), aim_=(0.0, 0.0))
        tgt = make_target(pos=(60.0, 0.0, 40.0))
        a, f, t = aim.decide(make_obs(me, [tgt]), ray.clear)
        self.assertIsNotNone(t)
        self.assertNotIn("error", aim.last_plan)


# ---------------------------------------------------------------------------- cost


class TestPerformance(unittest.TestCase):
    def _avg_ms(self, fn, n=500):
        fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t0) / n * 1000.0

    def test_per_call_cost_under_one_millisecond(self):
        rng = random.Random(8)
        mechs = [make_target(tid=i, pos=(rng.uniform(-150, 150), rng.uniform(-150, 150), rng.uniform(0, 20)),
                             vel=(rng.uniform(-12, 12), rng.uniform(-12, 12), 0.0), hp=rng.randint(200, 1400),
                             aim_=(rng.uniform(-3, 3), 0.0)) for i in range(15)]
        me = make_me(aim_=(0.3, 0.0), weapons=[make_me()["weapons"][0], laser_weapon(charge=0.4)])
        obs = make_obs(me, mechs, t=2.0)
        tr = aim.Tracker(CONFIG)
        st = aim.TargetState()
        for k in range(8):
            tr.update(make_obs(me, mechs, t=1.0 + 0.1 * k))
        target = aim.choose_target(obs, me, st, tr)
        sol = aim.solve(me, target, "gun", GUN, CONFIG, tr)
        aim.init({"config": CONFIG})
        los = _REAL_MAP.clear if _REAL_MAP is not None else (lambda a, b: True)
        t = [2.0]

        def step():
            t[0] += 0.1
            aim.decide(make_obs(me, mechs, t=t[0]), los)

        rows = [
            ("aim_solution (gun, tracker)", lambda: aim.aim_solution(me, target, "gun", GUN, CONFIG, tr)),
            ("aim_solution (laser)", lambda: aim.aim_solution(me, target, "laser", LASER, CONFIG, tr)),
            ("choose_target (15 mechs)", lambda: aim.choose_target(obs, me, st, tr)),
            ("fire_decision", lambda: aim.fire_decision(me, target, me["weapons"][0], sol, CONFIG)),
            ("Tracker.update (15 mechs)", lambda: tr.update(obs)),
            ("decide (15 mechs, both weapons)", step),
        ]
        print()
        for name, fn in rows:
            ms = self._avg_ms(fn)
            print("  %-34s %.3f ms" % (name, ms))
            self.assertLess(ms, 1.0, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
