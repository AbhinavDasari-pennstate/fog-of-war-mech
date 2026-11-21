"""shotline.py: PEEK, else HOLD, when a tracked target's line of fire is blocked (box-world rays)."""
import math
import unittest

import shotline as S


class World:
    """Axis-aligned boxes (x0, y0, z0, x1, y1, z1); clear() is a slab test. Counts calls."""
    def __init__(self, *boxes):
        self.boxes = boxes
        self.calls = 0

    def clear(self, a, b):
        self.calls += 1
        for bx in self.boxes:
            t0, t1 = 0.0, 1.0
            for k in range(3):
                d = b[k] - a[k]
                lo, hi = bx[k], bx[k + 3]
                if abs(d) < 1e-12:
                    if not lo <= a[k] <= hi:
                        break
                    continue
                u, v = (lo - a[k]) / d, (hi - a[k]) / d
                if u > v:
                    u, v = v, u
                t0, t1 = max(t0, u), min(t1, v)
                if t0 > t1:
                    break
            else:
                return False
        return True


def gun(ammo=12):
    return {"kind": "gun", "ammo": ammo, "reload": 0.0, "cooldown": 0.0, "stats": {"range": 230}}


def laser(charge=0.0, cooldown=0.0):
    return {"kind": "laser", "charge": charge, "cooldown": cooldown, "stats": {"range": 280}}


def me(pos=(0.0, 0.0, 0.0), weapons=None):
    return {"pos": list(pos), "stats": {"muzzle_height": 7.0, "walk_speed": 11.7}, "energy": 100.0,
            "grounded": True, "weapons": weapons if weapons is not None else [gun(), None]}


def hunter(i=5, pos=(80.0, 0.0, 0.0)):
    return {"kind": "mech", "id": i, "pos": list(pos), "vel": [0.0, 0.0, 0.0], "hp": 1400, "max_hp": 1400}


PILLAR = World((10, -3, -1, 14, 3, 30))         # narrow: a sideways step clears it
WIDE = World((10, -60, -1, 14, 60, 30))         # wide: no spot within 1 s clears it


class TestShotLine(unittest.TestCase):
    def test_clear_line_is_left_to_the_bot(self):
        self.assertIsNone(S.ShotLine().step(1.0, me(), hunter(), [], World().clear))

    def test_peek_steps_out_shoots_and_steps_back(self):
        s = S.ShotLine()
        p = s.step(1.0, me(), hunter(), [], PILLAR.clear)
        self.assertEqual(p["opt"], "peek")
        self.assertGreater(abs(p["move"][1]), 0.8)                       # sideways
        spot = s.spot
        self.assertTrue(PILLAR.clear((spot[0], spot[1], 7.0), (80, 0, 4.5)))
        self.assertLessEqual(math.dist(spot, (0, 0)), S.PEEK_ETA_S * 11.7)
        p = s.step(1.2, me(pos=(spot[0] / 2, spot[1] / 2, 0)), hunter(), [], PILLAR.clear)
        self.assertEqual(p["note"], "stepping out")
        p = s.step(1.5, me(pos=(spot[0], spot[1], 0.0)), hunter(), [], PILLAR.clear)
        self.assertEqual((p["note"], p["move"], p["aim"]), ("shooting", (0.0, 0.0), None))  # the aim module shoots
        p = s.step(1.7, me(pos=(spot[0], spot[1], 0.0), weapons=[gun(ammo=9), None]), hunter(), [], PILLAR.clear)
        self.assertEqual(p["note"], "back")                              # a 3-round burst: back behind the corner
        self.assertLess(p["move"][1] * spot[1], 0.0)
        self.assertIsNone(s.step(1.8, me(pos=(0.2, 0.0, 0.0)), hunter(), [], PILLAR.clear))  # home: done

    def test_a_fired_laser_ends_the_peek(self):
        s = S.ShotLine()
        s.step(1.0, me(weapons=[gun(), laser()]), hunter(), [], PILLAR.clear)
        at = me(pos=(s.spot[0], s.spot[1], 0.0), weapons=[gun(), laser(charge=0.5)])
        self.assertEqual(s.step(1.5, at, hunter(), [], PILLAR.clear)["note"], "shooting")
        at = me(pos=(s.spot[0], s.spot[1], 0.0), weapons=[gun(), laser(cooldown=1.6)])
        self.assertEqual(s.step(1.6, at, hunter(), [], PILLAR.clear)["note"], "back")

    def test_no_step_back_while_the_laser_charges_and_a_release_into_a_wall_is_not_the_shot(self):
        # SL-H2: a 3-round gun burst while the laser charges must not end the peek (the laser would fire into the wall)
        s = S.ShotLine()
        s.step(1.0, me(weapons=[gun(), laser()]), hunter(), [], PILLAR.clear)
        at = lambda w: me(pos=(s.spot[0], s.spot[1], 0.0), weapons=w)
        self.assertEqual(s.step(1.5, at([gun(ammo=12), laser(charge=0.3)]), hunter(), [], PILLAR.clear)["note"], "shooting")
        p = s.step(1.7, at([gun(ammo=8), laser(charge=0.5)]), hunter(), [], PILLAR.clear)
        self.assertEqual(p["note"], "shooting")                          # burst done, laser still charging: stay
        p = s.step(3.4, at([gun(ammo=4), laser(charge=0.75)]), hunter(), [], PILLAR.clear)
        self.assertEqual(p["note"], "shooting")                          # past PEEK_SHOT_S, still charging: stay
        p = s.step(3.5, at([gun(ammo=4), laser(cooldown=1.6)]), hunter(), [], PILLAR.clear)
        self.assertEqual(p["note"], "back")                              # released on a clear line: done
        s2 = S.ShotLine()                                                # a release behind the wall is not the shot
        s2.step(1.0, me(weapons=[gun(), laser(charge=0.6)]), hunter(), [], WIDE.clear)
        s2.step(1.1, me(weapons=[gun(), laser(cooldown=1.6)]), hunter(), [], WIDE.clear)
        self.assertFalse(s2.fired)

    def test_charging_halves_the_reach_of_a_peek(self):
        far = World((10, -7, -1, 14, 7, 30))                              # needs about 10 m of side step
        self.assertEqual(S.ShotLine().step(1.0, me(), hunter(), [], far.clear)["opt"], "peek")
        p = S.ShotLine().step(1.0, me(weapons=[gun(), laser(charge=0.3)]), hunter(), [], far.clear)
        self.assertEqual(p["opt"], "hold")                                # at half speed it is over 1 s away

    def test_no_peek_with_nothing_ready_and_leaving_a_peek_steps_back(self):
        p = S.ShotLine().step(1.0, me(weapons=[{"kind": "gun", "ammo": 0, "reload": 1.2, "stats": {"range": 230}},
                                               laser(cooldown=1.0)]), hunter(), [], PILLAR.clear)
        self.assertEqual((p["opt"], p["note"]), ("hold", "reloading"))     # SL-M1: nothing to shoot: no exposure
        s = S.ShotLine()
        s.step(1.0, me(), hunter(), [], PILLAR.clear)
        p = s.step(1.2, me(pos=(s.spot[0], s.spot[1], 0.0)), hunter(6), [], PILLAR.clear)
        self.assertEqual(p["note"], "back")                              # SL-M2: target switched: back to cover
        s = S.ShotLine()
        s.step(1.0, me(), hunter(), [], PILLAR.clear)
        self.assertEqual(s.step(3.0, me(pos=(0.5, 0.5, 0.0)), hunter(), [], PILLAR.clear)["note"], "back")  # timed out

    def test_bot_hold_keeps_its_own_move_while_hit(self):
        # SL-H1: holding behind a wall must not pin the bot still while an unseen attacker hits it
        import bot as B
        from test_bot import HELLO, make_obs, mech as bmech

        class FakeRay:
            clear = staticmethod(WIDE.clear)
            ground = staticmethod(lambda x, y: 0.0)
        saved = B.raylib
        try:
            B.raylib = FakeRay
            b = B.Bot(HELLO)
            base = B.Bot(HELLO)
            base.shot = None
            hit = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 3.0, "by": None}]
            zone = {"center": [-300, 0], "radius": 400, "next_center": [-300, 0], "next_radius": 100,
                    "shrink_start": 60, "shrink_end": 110, "dps": 20}
            o = lambda k: make_obs(tick=k, t=30 + k / 10, mechs=[bmech(5, (80, 0, 0))], vel=(0, 5, 0), events=hit, zone=zone)
            for k in range(5):
                a, a0 = b.act(o(k)), base.act(o(k))
            self.assertEqual(b.shot_log[0][0], "hold")
            self.assertEqual(a["move"], a0["move"])                       # the same move as without shotline
            self.assertGreater(math.hypot(*a["move"]), 0.5)
            self.assertFalse(a["fire"][0])                                # gun quiet: it would hit the wall
        finally:
            B.raylib = saved

    def test_never_peeks_where_another_hunter_sees_us(self):
        watcher = hunter(6, (0.0, -60.0, 0.0))                           # open ground: it sees both sides
        self.assertEqual(S.ShotLine().step(1.0, me(), hunter(), [watcher], PILLAR.clear)["opt"], "hold")
        walled = World(*PILLAR.boxes, (-60, -40, -1, -56, 40, 40))        # a hunter behind a wall far west
        p = S.ShotLine().step(1.0, me(), hunter(), [hunter(6, (-100.0, 0.0, 0.0))], walled.clear)
        self.assertEqual(p["opt"], "peek")

    def test_hold_aims_at_it_and_keeps_the_gun_quiet(self):
        p = S.ShotLine().step(1.0, me(weapons=[gun(), laser()]), hunter(pos=(90, 0, 0)), [], WIDE.clear)
        self.assertEqual((p["opt"], p["move"], p["gun_quiet"]), ("hold", None, True))   # keeps the bot's move
        self.assertAlmostEqual(p["aim"][0], 0.0)

    def test_hold_is_reused_before_searching_again(self):
        s = S.ShotLine()
        s.step(1.0, me(), hunter(), [], WIDE.clear)
        WIDE.calls = 0
        self.assertEqual(s.step(1.2, me(), hunter(), [], WIDE.clear)["opt"], "hold")
        self.assertEqual(WIDE.calls, 1)                                   # only the line test
        s.step(1.6, me(), hunter(), [], WIDE.clear)
        self.assertGreater(WIDE.calls, 2)                                 # 0.5 s later: searched again

    def test_out_of_every_weapons_range_is_left_to_the_bot(self):
        light = {"kind": "lightning", "stats": {"range": 45}}
        self.assertIsNone(S.ShotLine().step(1.0, me(weapons=[light, None]), hunter(), [], PILLAR.clear))

    def test_ray_budget_and_exposure_out_of_budget_counts_as_seen(self):
        w = World(*PILLAR.boxes, (-60, -300, -1, -56, 300, 60))           # a wall hides the crowd from every spot
        crowd = [hunter(10 + k, (-100.0, 10.0 * k - 100.0, 0.0)) for k in range(20)]
        self.assertEqual(S.ShotLine().step(1.0, me(), hunter(), crowd[:2], w.clear)["opt"], "peek")   # control
        w.calls = 0
        p = S.ShotLine().step(1.0, me(), hunter(), crowd, w.clear)
        self.assertLessEqual(w.calls, S.RAY_CAP)
        self.assertEqual(p["opt"], "hold")                                # could not check them all: not safe

    def test_never_raises(self):
        with self.assertRaises(Exception):                               # bad input raises: the hook turns it off
            S.ShotLine().step(1.0, {"pos": None}, hunter(), [], PILLAR.clear)

    def test_bot_hook_peeks_and_logs_the_option(self):
        import bot as B
        import mechfight as mf
        from test_bot import HELLO, make_obs, mech as bmech

        class FakeRay:
            clear = staticmethod(PILLAR.clear)
            ground = staticmethod(lambda x, y: 0.0)
        saved, saved_log = B.raylib, mf.log
        lines = []
        try:
            B.raylib, mf.log = FakeRay, lines.append
            b = B.Bot(HELLO)
            a = b.act(make_obs(mechs=[bmech(5, (80, 0, 0))], vel=(0, 5, 0)))
            self.assertGreater(abs(a["move"][1]), 0.8)                  # sideways off the pillar
            self.assertTrue(any("shot peek" in ln for ln in lines), lines)
        finally:
            B.raylib, mf.log = saved, saved_log


if __name__ == "__main__":
    unittest.main()
