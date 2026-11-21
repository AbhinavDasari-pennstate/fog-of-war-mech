"""Tests for fight.py and its hook in bot.py. Run: python -B -m unittest test_fight"""
import copy
import json
import math
import os
import shutil
import tempfile
import unittest

import fight
import ray
from bot import Bot
from test_bot import HELLO, make_obs, mech, gun, laser, problems

BOX = (20.0, -10.0, 40.0, 10.0)  # one 20 x 20 m building, 14 m tall, east of the origin


def box_world():
    b = BOX
    return {"prisms": [{"ring": [b[0], b[1], b[2], b[1], b[2], b[3], b[0], b[3]], "base": 0, "top": 14}],
            "bounds": {"minX": -500, "minY": -500, "maxX": 500, "maxY": 500}}


def hp_mech(i, pos, hp):
    m = mech(i, pos)
    m["hp"], m["max_hp"] = hp, 1400
    return m


class Tunables(unittest.TestCase):
    """Each test sets fight constants; restore them after."""
    KEYS = [k for k in dir(fight) if k.isupper()]

    def setUp(self):
        self.saved = {k: getattr(fight, k) for k in self.KEYS}
        fight.SPOT_BUDGET_S = 5.0  # tests check geometry, not the wall-clock cap (CPU load would flake them)

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(fight, k, v)


class TestDefaults(Tunables):
    def test_hide_and_chase_are_off_by_default(self):
        self.assertEqual(fight.RETREAT_HP, 0.0)
        self.assertEqual(fight.MULTI_RETREAT_HP, 0.0)
        self.assertEqual(fight.CHASE_HP, 0.0)

    def test_off_by_default_plan_is_none_at_low_hp(self):
        fight.CHASE_HP = 0.0
        f = fight.Fight()
        me = {"pos": [0, 0, 0], "hp": 100, "max_hp": 1400}
        self.assertIsNone(f.plan(10.0, me, [hp_mech(5, (80, 0, 0), 1400)], lambda a, b, terrain=True: True))


class TestHideGeometry(Tunables):
    def setUp(self):
        super().setUp()
        self.R = ray.Ray(world=box_world())

    def test_spot_behind_the_building_is_hidden(self):
        f = fight.Fight()
        threats = {5: (-60.0, 0.0, 0.0, 0.0)}  # west of us; the box is east
        found = f.find_spot(self.R.clear, 15.0, 0.0, 0.0, threats)
        self.assertIsNotNone(found)
        spot, way = found
        self.assertIsNotNone(way)  # right against the wall: round a corner
        self.assertFalse(BOX[0] < way[0] < BOX[2] and BOX[1] < way[1] < BOX[3])
        self.assertFalse(f.exposed(self.R.clear, spot[0], spot[1], 0.0, threats))
        self.assertGreater(spot[0], BOX[2])  # on the far side of the box
        self.assertFalse(BOX[0] < spot[0] < BOX[2] and BOX[1] < spot[1] < BOX[3])  # never inside it

    def test_open_ground_has_no_spot(self):
        f = fight.Fight()
        R = ray.Ray(world={"prisms": []})
        self.assertIsNone(f.find_spot(R.clear, 0.0, 0.0, 0.0, {5: (-60.0, 0.0, 0.0, 0.0)}))

    def test_spot_respects_the_zone(self):
        f = fight.Fight()
        threats = {5: (-60.0, 0.0, 0.0, 0.0)}
        self.assertIsNone(f.find_spot(self.R.clear, 15.0, 0.0, 0.0, threats, zone=(0.0, 0.0, 20.0)))


class TestPlan(Tunables):
    def setUp(self):
        super().setUp()
        self.R = ray.Ray(world=box_world())
        fight.RETREAT_HP = 0.45

    def me(self, hp, pos=(15.0, 0.0, 0.0)):
        return {"pos": list(pos), "hp": hp, "max_hp": 1400, "cubes": 0}

    def test_low_hp_walks_to_cover_then_holds_and_leaves_at_regen_hp(self):
        f = fight.Fight()
        e = hp_mech(5, (-60, 0, 0), 1400)
        p = f.plan(1.0, self.me(400), [e], self.R.clear)
        self.assertTrue(p["hide"])
        self.assertGreater(abs(p["move"][1]), 0.5)  # round the corner first (the box is right in front)
        sx, sy = f.spot
        p = f.plan(2.0, self.me(400, (sx, sy, 0)), [e], self.R.clear)
        self.assertEqual(p["move"], [0.0, 0.0])
        self.assertIsNone(f.plan(3.0, self.me(0.95 * 1400, (sx, sy, 0)), [e], self.R.clear))
        self.assertFalse(f.hiding)

    def test_finish_rule_beats_hiding(self):
        f = fight.Fight()
        self.assertIsNone(f.plan(1.0, self.me(500), [hp_mech(5, (-60, 0, 0), 200)], self.R.clear))
        self.assertEqual(f.reason, "finish")

    def test_no_cover_means_keep_fighting(self):
        f = fight.Fight()
        R = ray.Ray(world={"prisms": []})
        self.assertIsNone(f.plan(1.0, self.me(400, (0, 0, 0)), [hp_mech(5, (-60, 0, 0), 1400)], R.clear))
        self.assertEqual(f.reason, "no_cover")

    def test_no_cover_is_remembered(self):
        f = fight.Fight()
        R = ray.Ray(world={"prisms": []})
        calls = []
        orig = f.find_spot
        f.find_spot = lambda *a, **k: calls.append(1) or orig(*a, **k)
        e = [hp_mech(5, (-60, 0, 0), 1400)]
        for i in range(5):  # 0.4 s of decisions
            f.plan(1.0 + i / 10, self.me(400, (0, 0, 0)), e, R.clear)
        self.assertEqual(len(calls), 1)
        f.plan(1.0 + fight.NO_COVER_S + 0.1, self.me(400, (0, 0, 0)), e, R.clear)
        self.assertEqual(len(calls), 2)

    def test_cover_search_has_a_time_cap(self):
        fight.SPOT_BUDGET_S = -1.0
        self.assertIsNone(fight.Fight().find_spot(self.R.clear, 15.0, 0.0, 0.0, {5: (-60.0, 0.0, 0.0, 0.0)}))

    def test_urgent_zone_never_hides(self):
        f = fight.Fight()
        self.assertIsNone(f.plan(1.0, self.me(400), [hp_mech(5, (-60, 0, 0), 1400)], self.R.clear, urgent=True))

    def test_threats_expire(self):
        f = fight.Fight()
        f.plan(1.0, self.me(1400), [hp_mech(5, (-60, 0, 0), 1400)], self.R.clear)
        f.plan(1.0 + fight.THREAT_MEMORY_S + 1, self.me(1400), [], self.R.clear)
        self.assertEqual(f.threats, {})


class TestChase(Tunables):
    def setUp(self):
        super().setUp()
        fight.CHASE_HP = 0.4
        self.clear = lambda a, b, terrain=True: True

    def test_no_chase_without_a_laser(self):
        me = {"pos": [0, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": [gun(), None]}
        self.assertIsNone(fight.Fight().plan(1.0, me, [hp_mech(5, (100, 0, 0), 300)], self.clear))
        me["weapons"][1] = laser()
        self.assertEqual(fight.Fight().plan(1.0, me, [hp_mech(5, (100, 0, 0), 300)], self.clear)["reason"], "chase")

    def test_walks_at_a_wounded_hunter_then_its_last_position(self):
        f = fight.Fight()
        me = {"pos": [0, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": [gun(), laser()]}
        p = f.plan(1.0, me, [hp_mech(5, (100, 0, 0), 300)], self.clear)
        self.assertEqual(p["reason"], "chase")
        self.assertAlmostEqual(p["move"][0], 1.0)
        p = f.plan(4.0, me, [], self.clear)  # track lost: keep going to where it was
        self.assertEqual(p["reason"], "chase")
        self.assertIsNone(f.plan(4.0 + fight.CHASE_S + 1, me, [], self.clear))

    def test_no_chase_when_hurt_far_or_outnumbered(self):
        me_low = {"pos": [0, 0, 0], "hp": 300, "max_hp": 1400, "weapons": [gun(), laser()]}
        me = {"pos": [0, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": [gun(), laser()]}
        self.assertIsNone(fight.Fight().plan(1.0, me_low, [hp_mech(5, (100, 0, 0), 300)], self.clear))
        self.assertIsNone(fight.Fight().plan(1.0, me, [hp_mech(5, (fight.CHASE_RANGE + 10, 0, 0), 300)], self.clear))
        two = [hp_mech(5, (100, 0, 0), 300), hp_mech(6, (-100, 0, 0), 1400)]
        self.assertIsNone(fight.Fight().plan(1.0, me, two, self.clear))

    def test_holds_on_top_of_the_prey(self):
        me = {"pos": [0, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": [gun(), laser()]}
        p = fight.Fight().plan(1.0, me, [hp_mech(5, (7.9, 0, 0), 500)], self.clear)
        self.assertEqual(p["move"], [0.0, 0.0])

    def test_mech_without_id_or_with_2d_pos_is_ignored(self):
        me = {"pos": [0, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": [gun(), laser()]}
        m = hp_mech(5, (50, 0, 0), 1400)
        del m["id"]
        m2 = hp_mech(6, (60, 0, 0), 1400)
        m2["pos"] = [60, 0]
        f = fight.Fight()
        f.plan(1.0, me, [m, m2], self.clear)
        self.assertEqual(list(f.threats), [6])

    def test_never_chases_out_of_the_zone(self):
        me = {"pos": [0, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": [gun(), laser()]}
        self.assertIsNone(fight.Fight().plan(1.0, me, [hp_mech(5, (100, 0, 0), 300)], self.clear, zone=(0, 0, 50)))


class TestBotHook(Tunables):
    def bot_with_box(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "v2"))
        with open(os.path.join(d, "v2", "world.json"), "w") as fh:
            json.dump(box_world(), fh)
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = d
        return Bot(hello)

    def test_low_hp_keeps_fighting_by_default(self):
        bot = Bot(HELLO)
        e = mech(5, (100, 0, 0))
        a = bot.act(make_obs(hp=400, weapons=[gun(), laser(charge=0.3)], mechs=[e]))
        self.assertEqual(problems(make_obs(), a), [])
        self.assertTrue(a["fire"][1])  # the charge is kept, not dropped as v3's flee did
        self.assertGreaterEqual(a["move"][0], -0.01)  # no run straight away from it

    def test_hide_holds_fire_and_moves_to_cover(self):
        fight.RETREAT_HP = 0.45
        bot = self.bot_with_box()
        e = mech(5, (-60, 0, 0))
        a = bot.act(make_obs(pos=(15, 0, 0), hp=400, weapons=[gun(), laser()], mechs=[e]))
        self.assertEqual(bot.errors, 0)
        self.assertTrue(bot.fight.hiding)
        self.assertEqual(a["fire"], [False, False])
        self.assertGreater(math.hypot(*a["move"]), 0.3)

    def test_hide_keeps_a_charging_laser_held(self):
        fight.RETREAT_HP = 0.45
        for aim_ok in (True, False):  # aim module and v3 fallback paths
            bot = self.bot_with_box()
            bot.aim_ok = aim_ok
            a = bot.act(make_obs(pos=(15, 0, 0), hp=400, weapons=[gun(), laser(charge=0.3)], mechs=[mech(5, (-60, 0, 0))]))
            self.assertTrue(bot.fight.hiding)
            self.assertFalse(a["fire"][0])
            self.assertTrue(a["fire"][1], f"aim_ok={aim_ok}: releasing would fire the beam")

    def chase_obs(self, in_zone):
        e = mech(5, (100, 0, 0))
        e["hp"], e["max_hp"] = 300, 1400
        drop = {"id": 77, "pos": [0, -60, 0], "lands_in": 0, "radius": 8, "holder": None}
        zone = {"center": [0, 0], "radius": 900, "next_center": [0, -800], "next_radius": 600}
        return make_obs(weapons=[gun(), laser()], mechs=[e], zone=zone, in_zone=in_zone, drops=[drop])

    def test_chase_beats_goals_in_zone_but_not_the_zone_pull(self):
        fight.CHASE_HP = 0.8
        bot = Bot(HELLO)
        a = bot.act(self.chase_obs(True))
        self.assertEqual(bot.fight.reason, "chase")
        self.assertGreater(a["move"][0], 0.9)  # at the hunter, not toward the drop or the next zone centre
        bot = Bot(HELLO)
        a = bot.act(self.chase_obs(False))
        self.assertLess(a["move"][1], -0.5)  # outside the zone: back toward it first

    def test_two_near_mechs_get_no_move_toward_either(self):
        zone = {"center": [0, 0], "radius": 900, "next_center": [800 * math.cos(math.radians(50)),
                                                                 800 * math.sin(math.radians(50))], "next_radius": 300}
        a100 = math.radians(100)
        ms = [mech(5, (60, 0, 0)), mech(6, (60 * math.cos(a100), 60 * math.sin(a100), 0))]
        a = Bot(HELLO).act(make_obs(hp=500, zone=zone, mechs=ms))
        for m in ms:
            ux, uy = m["pos"][0] / 60, m["pos"][1] / 60
            self.assertLessEqual(a["move"][0] * ux + a["move"][1] * uy, 1e-6)

    def test_low_hp_never_closes_on_a_near_mech_in_zone(self):
        """MR-02 on the default (fight.py) path: no flee push, but the zone pull loses its part toward the mech."""
        zone = {"center": [300, 0], "radius": 560, "next_center": [420, 0], "next_radius": 140}
        e = mech(5, (70, 0, 0))
        e["aim"] = [math.pi, 0]
        a = Bot(HELLO).act(make_obs(hp=420, zone=zone, mechs=[e]))
        self.assertLessEqual(a["move"][0], 1e-9)

    def test_no_cube_drift_while_hiding(self):
        fight.RETREAT_HP = 0.45
        bot = self.bot_with_box()
        bot.dmg_t = -99.0
        bot.fight.hiding, bot.fight.spot, bot.fight.spot_t = True, (50.0, 0.0), 30.0
        bot.fight.threats = {5: (-60.0, 0.0, 0.0, 30.0)}
        a = bot.act(make_obs(pos=(50, 0, 0), hp=400, cubes=[{"id": 1, "pos": [70, 0, 0]}]))
        self.assertTrue(bot.fight.hiding)
        self.assertEqual(a["move"], [0.0, 0.0])

    def test_a_failing_plan_falls_back_to_v3_flee(self):
        bot = Bot(HELLO)

        def boom(*a, **k):
            raise RuntimeError("x")
        bot.fight.plan = boom
        e = mech(5, (70, 0, 0))
        a = bot.act(make_obs(hp=400, mechs=[e]))
        self.assertEqual(bot.fight_errors, 1)
        self.assertEqual(bot.errors, 0)
        self.assertLess(a["move"][0], 0.0)  # v3 flee: away from the hunter
        bot.act(make_obs(tick=2, hp=400, mechs=[e]))
        self.assertEqual(bot.fight_errors, 1)  # not called again

    def test_degraded_pass_restores_fight_state(self):
        bot = Bot(HELLO)
        bot.fight.threats = {9: (1.0, 2.0, 0.0, 29.0)}
        orig = bot._decide
        calls = []

        def flaky(obs):
            calls.append(1)
            if len(calls) == 1:
                bot.fight.threats[7] = (0, 0, 0, 30.0)
                raise RuntimeError("first pass")
            return orig(obs)
        bot._decide = flaky
        bot.act(make_obs())
        self.assertNotIn(7, bot.fight.threats)


if __name__ == "__main__":
    unittest.main()
