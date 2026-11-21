"""Tests for loot_mem.py and its hook in bot.py. Run: python -B -m unittest test_loot_mem"""
import math
import unittest

import bot as botmod
import loot_mem
from bot import Bot
from test_bot import HELLO, make_obs, mech, pickup, gun, lightning

ZONE = {"center": [0, 0], "radius": 900, "next_center": [0, 0], "next_radius": 800}


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.saved = loot_mem.LOOT_MEM_S

    def tearDown(self):
        loot_mem.LOOT_MEM_S = self.saved

    def me(self, x=0.0, y=0.0):
        return {"pos": [x, y, 0.0]}

    def test_keeps_a_pickup_that_left_view(self):
        m = loot_mem.LootMemory()
        m.update({"t": 1.0, "pickups": [pickup(7, "laser", (100, 0, 0))]}, self.me(), 45)
        m.update({"t": 2.0, "pickups": []}, self.me(), 45)
        self.assertEqual([p["id"] for p in m.pickups()], [7])

    def test_forgets_after_the_memory_time(self):
        m = loot_mem.LootMemory()
        m.update({"t": 1.0, "pickups": [pickup(7, "laser", (100, 0, 0))]}, self.me(), 45)
        m.update({"t": 1.0 + loot_mem.LOOT_MEM_S + 0.1, "pickups": []}, self.me(), 45)
        self.assertEqual(m.pickups(), [])

    def test_forgets_one_we_would_see_but_do_not(self):
        m = loot_mem.LootMemory()
        m.update({"t": 1.0, "pickups": [pickup(7, "laser", (100, 0, 0))]}, self.me(), 45)
        m.update({"t": 5.0, "pickups": []}, self.me(70, 0), 45)  # 30 m away, inside near range: gone
        self.assertEqual(m.pickups(), [])

    def test_forgets_one_whose_spot_we_see_empty(self):
        m = loot_mem.LootMemory()
        m.update({"t": 1.0, "pickups": [pickup(7, "laser", (100, 0, 0))]}, self.me(), 45)
        m.update({"t": 2.0, "pickups": []}, self.me(), 45, visible=lambda p: False)  # looking away: kept
        self.assertEqual(len(m.pickups()), 1)
        m.update({"t": 3.0, "pickups": []}, self.me(), 45, visible=lambda p: True)  # spot in view, empty: taken
        self.assertEqual(m.pickups(), [])

    def test_nan_time_is_ignored(self):
        m = loot_mem.LootMemory()
        m.update({"t": float("nan"), "pickups": [pickup(7, "laser", (100, 0, 0))]}, self.me(), 45)
        self.assertEqual(m.pickups(), [])

    def test_ignores_bad_pickups_and_time_going_back(self):
        m = loot_mem.LootMemory()
        bad = [{"id": None, "kind": "laser", "pos": [1, 2, 3]}, {"id": 2, "kind": "laser", "pos": [float("nan"), 0, 0]},
               {"id": 3, "kind": "laser", "pos": [1, 2]}]
        m.update({"t": 1.0, "pickups": bad + [pickup(7, "laser", (100, 0, 0))]}, self.me(), 45)
        self.assertEqual([p["id"] for p in m.pickups()], [7])
        m.update({"t": 0.5, "pickups": []}, self.me(), 45)  # a new match or a clock reset
        self.assertEqual(m.pickups(), [])


class TestBotHook(unittest.TestCase):
    def test_keeps_walking_to_a_laser_that_left_view_while_fighting(self):
        """Seen at spawn, then we turn to a hunter and the laser leaves view (sim seed 71056311)."""
        bot = Bot(HELLO)
        bot.act(make_obs(t=0.0, pos=(0, 0, 0), pickups=[pickup(1, "laser", (0, 80, 0))], zone=ZONE))
        a = bot.act(make_obs(t=3.0, tick=2, pos=(0, 30, 0), mechs=[mech(5, (150, 30, 0))], zone=ZONE))
        self.assertEqual(bot.errors, 0)
        self.assertGreater(a["move"][1], 0.9)

    def test_bot_forgets_a_laser_it_sees_is_gone(self):
        bot = Bot(HELLO)
        bot.act(make_obs(t=0.0, pickups=[pickup(1, "laser", (100, 0, 0))], zone=ZONE))
        a = bot.act(make_obs(t=6.0, tick=2, aim=(0.0, 0.0), zone=ZONE))  # aimed at the spot, nothing listed
        self.assertEqual(bot.loot.seen, {})
        self.assertLess(abs(a["move"][0]), 0.9)

    def test_economy_phase_keeps_obs_pickups(self):
        bot = Bot(HELLO)
        bot.act(make_obs(t=0.0, pickups=[pickup(1, "laser", (0, 80, 0))], aim=(1.57, 0.0), zone=ZONE))
        if bot.eco is None:
            self.skipTest("no economy module")
        bot.eco.phase, bot.eco.phase_t = "rest", 2.9
        a = bot.act(make_obs(t=3.0, tick=2, pos=(0, 30, 0), aim=(-1.57, 0.0), zone=ZONE))
        self.assertIn(1, bot.loot.seen)  # still remembered for later
        self.assertLess(a["move"][1], 0.5)  # but economy's phase decides now, as before

    def test_off_means_obs_only(self):
        loot_mem.LOOT_MEM_S, saved = 0.0, loot_mem.LOOT_MEM_S
        try:
            bot = Bot(HELLO)
            bot.act(make_obs(t=0.0, pickups=[pickup(1, "laser", (0, 80, 0))], zone=ZONE))
            a = bot.act(make_obs(t=3.0, tick=2, pos=(0, 30, 0), zone=ZONE))
            self.assertLess(a["move"][1], 0.5)
        finally:
            loot_mem.LOOT_MEM_S = saved

    def test_give_up_time_is_the_tunable(self):
        bot = Bot(HELLO)
        pk = [pickup(1, "laser", (0, 80, 0))]
        bot.act(make_obs(t=1.0, pickups=pk, zone=ZONE))
        a = bot.act(make_obs(t=1.0 + loot_mem.GRAB_GIVE_UP_S - 1, tick=2, pickups=pk, zone=ZONE))
        self.assertGreater(a["move"][1], 0.9)
        a = bot.act(make_obs(t=1.0 + loot_mem.GRAB_GIVE_UP_S + 1, tick=3, pickups=pk, zone=ZONE))
        self.assertLess(a["move"][1], 0.5)
        self.assertIn(1, bot.skip)

    def test_swap_drops_the_lightning_slot_where_it_is(self):
        bot = Bot(HELLO)
        a = bot.act(make_obs(t=3.0, weapons=[lightning(), gun()], pickups=[pickup(1, "laser", (2, 0, 0))]))
        self.assertEqual((a.get("take"), a.get("drop")), (1, 0))

    def test_a_failing_memory_falls_back_to_obs_pickups(self):
        bot = Bot(HELLO)

        def boom(*a, **k):
            raise TypeError("bad pickup")
        bot.loot.update = boom
        a = bot.act(make_obs(t=1.0, pickups=[pickup(1, "laser", (60, 0, 0))], zone=ZONE))
        self.assertEqual(bot.errors, 0)
        self.assertEqual(bot.loot_errors, 1)
        self.assertGreater(a["move"][0], 0.9)  # still goes for the laser it sees
        a = bot.act(make_obs(t=2.0, tick=2, pickups=[pickup(1, "laser", (60, 0, 0))], zone=ZONE))
        self.assertEqual(bot.loot_errors, 1)  # not tried again

    def test_without_the_module_obs_pickups_and_15_s(self):
        self.addCleanup(setattr, botmod, "lootlib", botmod.lootlib)
        botmod.lootlib = None
        bot = Bot(HELLO)
        self.assertIsNone(bot.loot)
        pk = [pickup(1, "laser", (0, 80, 0))]
        bot.act(make_obs(t=1.0, pickups=pk, zone=ZONE))
        a = bot.act(make_obs(t=3.0, tick=2, zone=ZONE))
        self.assertLess(a["move"][1], 0.5)  # out of view: forgotten
        a = bot.act(make_obs(t=17.0, tick=3, pickups=pk, zone=ZONE))
        self.assertIn(1, bot.skip)  # old 15 s give-up
        a = bot.act(make_obs(t=17.0, tick=4, weapons=[gun(), lightning()], pickups=[pickup(3, "laser", (2, 0, 0))]))
        self.assertNotIn("take", a)  # no swap without the module

    def test_timed_out_nearest_laser_falls_through_to_the_next(self):
        bot = Bot(HELLO)
        bot.act(make_obs(t=1.0, pickups=[pickup(1, "laser", (0, 40, 0))], zone=ZONE))
        t = 1.0 + loot_mem.GRAB_GIVE_UP_S + 1
        a = bot.act(make_obs(t=t, tick=2, pos=(0, 0, 0), zone=ZONE,
                             pickups=[pickup(1, "laser", (0, 40, 0)), pickup(2, "laser", (90, 0, 0))]))
        self.assertIn(1, bot.skip)
        self.assertGreater(a["move"][0], 0.9)  # same tick: on to laser 2

    def test_lite_pass_uses_obs_pickups_only(self):
        bot = Bot(HELLO)
        bot.loot.seen = {9: (pickup(9, "laser", (0, 80, 0)), 0.5)}
        bot.lite = True
        a = bot._decide(make_obs(t=1.0, zone=ZONE))
        self.assertLess(a["move"][1], 0.5)

    def test_degraded_pass_restores_the_memory(self):
        bot = Bot(HELLO)
        bot.loot.seen = {9: (pickup(9, "laser", (100, 0, 0)), 0.5)}
        orig = bot._decide
        calls = []

        def flaky(obs):
            calls.append(1)
            if len(calls) == 1:
                bot.loot.seen[8] = (pickup(8, "laser", (50, 0, 0)), 1.0)
                raise RuntimeError("first pass")
            return orig(obs)
        bot._decide = flaky
        bot.act(make_obs(t=1.0))
        self.assertNotIn(8, bot.loot.seen)
        self.assertIn(9, bot.loot.seen)


if __name__ == "__main__":
    unittest.main()
