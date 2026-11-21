"""shotline.py: movement-only peek when nothing is shootable (box-world rays), and the bot hook never costs a shot."""
import copy
import math
import unittest

import shotline as S


class World:
    """Axis-aligned boxes (x0, y0, z0, x1, y1, z1); clear() is a slab test. Counts calls."""
    def __init__(self, *boxes):
        self.boxes = boxes
        self.calls = 0

    def clear(self, a, b, terrain=True):
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


def gun(ammo=12, reload=0.0):
    return {"kind": "gun", "ammo": ammo, "reload": reload, "cooldown": 0.0, "stats": {"range": 230}}


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
    def test_clear_line_or_no_fight_or_nothing_ready_leaves_movement_alone(self):
        self.assertIsNone(S.ShotLine().step(1.0, me(), hunter(), [], World().clear))
        self.assertIsNone(S.ShotLine().step(1.0, me(), hunter(), [], PILLAR.clear, engaged=False))
        empty = [gun(ammo=0, reload=1.0), laser(cooldown=1.0)]
        self.assertIsNone(S.ShotLine().step(1.0, me(weapons=empty), hunter(), [], PILLAR.clear))
        self.assertIsNone(S.ShotLine().step(1.0, me(), hunter(), [], WIDE.clear))   # no spot: bot's own move

    def test_peek_walks_to_a_spot_with_a_line_then_hands_over_then_steps_back_to_reload(self):
        s = S.ShotLine()
        mv = s.step(1.0, me(), hunter(), [], PILLAR.clear)
        self.assertGreater(abs(mv[1]), 0.8)                              # sideways
        spot = s.spot
        self.assertTrue(PILLAR.clear((spot[0], spot[1], 7.0), (80, 0, 4.5)))
        self.assertLessEqual(math.dist(spot, (0, 0)), S.PEEK_ETA_S * 11.7)
        self.assertIsNotNone(s.step(1.2, me(pos=(spot[0] / 2, spot[1] / 2, 0)), hunter(), [], PILLAR.clear))
        at = me(pos=(spot[0], spot[1], 0.0))
        self.assertIsNone(s.step(1.5, at, hunter(), [], PILLAR.clear))  # there, ready, line clear: the aim module's
        at = me(pos=(spot[0], spot[1], 0.0), weapons=[gun(ammo=0, reload=2.4), None])
        mv = s.step(2.5, at, hunter(), [], PILLAR.clear)                 # reloading in the open: back to cover
        self.assertLess(mv[1] * spot[1], 0.0)
        self.assertIsNone(s.step(3.0, me(pos=(0.3, 0.0, 0.0)), hunter(), [], PILLAR.clear))   # home: done
        self.assertIsNone(s.mode)

    def test_a_charging_laser_halves_the_reach(self):
        far = World((10, -7, -1, 14, 7, 30))                              # needs about 10 m of side step
        self.assertIsNotNone(S.ShotLine().step(1.0, me(), hunter(), [], far.clear))
        self.assertIsNone(S.ShotLine().step(1.0, me(weapons=[gun(), laser(charge=0.3)]), hunter(), [], far.clear))

    def test_never_peeks_where_another_hunter_sees_us_and_the_budget_counts_unchecked_as_seen(self):
        self.assertIsNone(S.ShotLine().step(1.0, me(), hunter(), [hunter(6, (0.0, -60.0, 0.0))], PILLAR.clear))
        walled = World(*PILLAR.boxes, (-60, -300, -1, -56, 300, 60))      # a wall hides these hunters from every spot
        crowd = [hunter(10 + k, (-100.0, 10.0 * k - 100.0, 0.0)) for k in range(20)]
        self.assertIsNotNone(S.ShotLine().step(1.0, me(), hunter(), crowd[:2], walled.clear))
        walled.calls = 0
        self.assertIsNone(S.ShotLine().step(1.0, me(), hunter(), crowd, walled.clear))
        self.assertLessEqual(walled.calls, S.RAY_CAP)

    def test_target_switch_or_loss_steps_back_but_never_far_or_stale(self):
        s = S.ShotLine()
        s.step(1.0, me(), hunter(), [], PILLAR.clear)
        mv = s.step(1.3, me(pos=(s.spot[0], s.spot[1], 0.0)), hunter(6), [], PILLAR.clear)
        self.assertLess(mv[1] * s.spot[1], 0.0)                           # another target: back to cover
        s = S.ShotLine()
        s.step(1.0, me(), hunter(), [], PILLAR.clear)
        self.assertIsNone(s.step(1.3, me(pos=(40.0, 30.0, 0.0)), None, [], PILLAR.clear))      # far from home: no
        s = S.ShotLine()
        s.step(1.0, me(), hunter(), [], PILLAR.clear)
        self.assertIsNone(s.step(20.0, me(pos=(s.spot[0], s.spot[1], 0.0)), None, [], PILLAR.clear))  # stale: no

    def test_no_step_back_while_the_laser_charges(self):
        s = S.ShotLine()
        s.step(1.0, me(weapons=[gun(), laser()]), hunter(), [], PILLAR.clear)
        at = me(pos=(s.spot[0], s.spot[1], 0.0), weapons=[gun(), laser(charge=0.4)])
        self.assertIsNone(s.step(1.4, at, hunter(6), [], PILLAR.clear))  # target switch mid-charge: stay put
        self.assertEqual(s.mode, "peek")

    def test_bad_input_raises_for_the_hook_to_count(self):
        with self.assertRaises(Exception):
            S.ShotLine().step(1.0, {"pos": None}, hunter(), [], PILLAR.clear)


class TestBotHook(unittest.TestCase):
    """The -0.37 of 3a5ab11: a target stepping out from cover (lead point clear, torso still blocked) must keep
    getting shot exactly as without shotline; shotline only moves us when nothing is shootable."""

    def _run(self, world, with_shot, ticks=12, others=()):
        import bot as B
        from test_bot import HELLO, make_obs, mech
        from test_bot import gun as bgun

        class FakeRay:
            clear = staticmethod(world.clear)
            ground = staticmethod(lambda x, y: 0.0)
        saved = B.raylib, (B.aimlib._state.get("los") if B.aimlib is not None else None)
        try:
            B.raylib = FakeRay
            if B.aimlib is not None:
                B.aimlib._state["los"] = world.clear
            b = B.Bot(copy.deepcopy(HELLO))
            if not with_shot:
                b.shot = None
            out, aim = [], (0.0, 0.0)
            for k in range(ticks):
                y = -3.0 + 0.8 * k
                e = mech(5, (80, y, 0), (0, 8, 0))
                hit = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}] if k == 0 else []
                o = make_obs(tick=k, t=30 + k / 10, mechs=[e] + list(others), weapons=[bgun(), None], aim=aim,
                             events=hit, vel=(0, 0, 0))
                a = b.act(o)
                aim = tuple(a["aim"])
                out.append(a)
            return b, out
        finally:
            B.raylib = saved[0]
            if B.aimlib is not None:
                B.aimlib._state["los"] = saved[1]

    def test_a_target_stepping_out_from_cover_keeps_getting_shot(self):
        from test_bot import mech
        corner = World((70, -80, -1, 74, 0, 30))          # the wall ends at y = 0 next to the target
        watcher = mech(6, (0, -150, 0))
        _, base = self._run(corner, False, others=[watcher])
        _, ours = self._run(corner, True, others=[watcher])
        self.assertGreaterEqual(sum(a["fire"][0] for a in base), 10)     # the base takes the corner shots
        self.assertEqual([a["fire"] for a in ours], [a["fire"] for a in base])
        self.assertEqual([a["aim"] for a in ours], [a["aim"] for a in base])

    def test_out_on_a_peek_with_a_clear_line_and_nothing_ready_the_bot_steps_back(self):
        # SL-PM-H1: the clear line makes the target "shootable", but with nothing firing step() must still run
        import bot as B
        from test_bot import HELLO, make_obs, mech
        open_ground = World()

        class FakeRay:
            clear = staticmethod(open_ground.clear)
            ground = staticmethod(lambda x, y: 0.0)
        saved = B.raylib
        try:
            B.raylib = FakeRay
            b = B.Bot(copy.deepcopy(HELLO))
            b.shot.mode, b.shot.home, b.shot.spot, b.shot.t0, b.shot.tid = "peek", (0.0, -8.0), (0.0, 0.0), 30.0, 5
            b.hit_by[5] = 30.0
            reloading = [{"kind": "gun", "ammo": 0, "reload": 1.5, "cooldown": 0.0, "stats": {"range": 230}}, None]
            a = b.act(make_obs(t=30.5, mechs=[mech(5, (80, 0, 0))], weapons=reloading, vel=(0, 0, 0)))
            self.assertFalse(any(a["fire"]))
            self.assertLess(a["move"][1], -0.9)                          # toward home (0, -8), behind cover
            self.assertFalse(a["boost"])
        finally:
            B.raylib = saved

    def test_shotline_only_moves_when_nothing_is_shootable(self):
        import bot as B
        b, ours = self._run(PILLAR, True, ticks=1)
        self.assertEqual(b.shot.mode, "peek")             # blocked, nothing shootable: it steps out
        self.assertGreater(abs(ours[0]["move"][1]), 0.8)
        _, base = self._run(PILLAR, False, ticks=1)
        self.assertEqual(ours[0]["aim"], base[0]["aim"])  # movement only: aim and fire untouched
        self.assertEqual(ours[0]["fire"], base[0]["fire"])


if __name__ == "__main__":
    unittest.main()
