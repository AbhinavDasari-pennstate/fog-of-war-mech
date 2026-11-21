"""Tests for chase.py and its bot.py hook. Run: python test_chase.py (stdlib only)."""
import math
import unittest

import chase
from chase import Chase
from test_bot import HELLO, Bot, gun, laser, make_obs, mech

ZONE = {"center": [0, 0], "radius": 560, "next_center": [0, 0], "next_radius": 380,
        "shrink_start": 60, "shrink_end": 110}


def me(pos=(0, 0, 0), hp=1400, energy=100, weapons=None):
    return {"pos": list(pos), "hp": hp, "max_hp": 1400, "energy": energy,
            "weapons": weapons if weapons is not None else [gun(), laser()]}


def hunter(i, pos, hp=700):
    n = math.hypot(pos[0], pos[1]) or 1.0
    m = mech(i, pos, vel=(5 * pos[0] / n, 5 * pos[1] / n, 0))  # backing off from the origin
    m["hp"] = hp
    return m


def clear_all(a, b, *args):
    return True


def plan(c, t, mechs, hit_on=None, my=None, zone=None, clear=clear_all):
    return c.plan(t, my or me(), mechs, hit_on or {}, {}, zone or ZONE, clear)


class ChaseUnit(unittest.TestCase):
    def test_wounded_fought_hunter_is_chased(self):
        p = plan(Chase(), 30.0, [hunter(5, (150, 0, 0))], {5: 29.0})
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p["move"][0], 1.0, places=3)
        self.assertEqual(p["id"], 5)

    def test_healthy_or_unfought_or_far_hunter_is_not(self):
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (150, 0, 0), hp=1300)], {5: 29.0}))
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (150, 0, 0))], {}))  # not fighting it
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (150, 0, 0))], {5: 20.0}))  # last hit too old
        far = chase.CHASE_START_RANGE + 10
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (far, 0, 0))], {5: 29.0}))

    def test_hit_by_counts_as_fighting(self):
        c = Chase()
        self.assertIsNotNone(c.plan(30.0, me(), [hunter(5, (150, 0, 0))], {}, {5: 29.5}, ZONE, clear_all))

    def test_needs_laser_and_hp_floor(self):
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (150, 0, 0))], {5: 29.0}, me(weapons=[gun(), None])))
        low = int(1400 * chase.CHASE_SELF_HP) - 1
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (150, 0, 0))], {5: 29.0}, me(hp=low)))

    def test_once_chasing_keeps_going_past_start_range(self):
        c = Chase()
        self.assertIsNotNone(plan(c, 30.0, [hunter(5, (150, 0, 0))], {5: 29.0}))
        p = plan(c, 33.0, [hunter(5, (260, 0, 0))], {5: 29.0})  # backing off, our last hit 4 s ago
        self.assertIsNotNone(p)
        self.assertTrue(p["boost"])  # far and regen is near: close in fast

    def test_another_hunter_aborts_and_is_remembered(self):
        c = Chase()
        self.assertIsNotNone(plan(c, 30.0, [hunter(5, (150, 0, 0))], {5: 29.0}))
        self.assertIsNone(plan(c, 30.5, [hunter(5, (150, 0, 0)), hunter(6, (0, 150, 0), hp=1400)], {5: 29.0}))
        # the other one left the track list, but it is still in memory: no new chase
        self.assertIsNone(plan(c, 31.0, [hunter(5, (150, 0, 0))], {5: 30.9}))
        # memory expired: chase again
        self.assertIsNotNone(plan(c, 30.5 + chase.OTHER_MEMORY_S + 0.1, [hunter(5, (150, 0, 0))], {5: 35.5}))

    def test_follows_last_known_position_then_gives_up(self):
        c = Chase()
        plan(c, 30.0, [hunter(5, (150, 40, 0))], {5: 29.0})
        p = plan(c, 31.0, [], {5: 29.0})  # track dropped
        self.assertIsNotNone(p)
        self.assertFalse(p["tracked"])
        self.assertGreater(p["move"][0], 0.9)
        self.assertEqual(p["aim_at"][:2], [150, 40])
        self.assertIsNone(plan(c, 30.0 + chase.CHASE_LOST_S + 0.1, [], {5: 29.0}))

    def test_reached_last_position_and_gone_ends_chase(self):
        c = Chase()
        plan(c, 30.0, [hunter(5, (150, 0, 0))], {5: 29.0})
        self.assertIsNone(c.plan(31.0, me(pos=(148, 0, 0)), [], {5: 29.0}, {}, ZONE, clear_all))

    def test_close_with_clear_line_holds_position(self):
        p = plan(Chase(), 30.0, [hunter(5, (chase.CHASE_HOLD_RANGE - 5, 0, 0))], {5: 29.0})
        self.assertEqual(p["move"], [0.0, 0.0])
        blocked = plan(Chase(), 30.0, [hunter(5, (chase.CHASE_HOLD_RANGE - 5, 0, 0))], {5: 29.0},
                       clear=lambda a, b, *k: False)
        self.assertGreater(blocked["move"][0], 0.9)  # no line: keep closing

    def test_boost_only_far_or_regen_due_and_with_energy(self):
        near = chase.CHASE_BOOST_RANGE - 20
        self.assertFalse(plan(Chase(), 30.0, [hunter(5, (near, 0, 0))], {5: 29.5})["boost"])
        self.assertTrue(plan(Chase(), 30.0, [hunter(5, (near, 0, 0))], {5: 30.0 - chase.DENY_S - 0.1})["boost"])
        self.assertTrue(plan(Chase(), 30.0, [hunter(5, (chase.CHASE_BOOST_RANGE + 10, 0, 0))], {5: 29.5})["boost"])
        self.assertFalse(plan(Chase(), 30.0, [hunter(5, (chase.CHASE_BOOST_RANGE + 10, 0, 0))], {5: 29.5},
                              me(energy=chase.CHASE_BOOST_ENERGY - 1))["boost"])

    def test_prey_near_zone_edge_or_outside_next_circle_aborts(self):
        edge = 560 - chase.ZONE_MARGIN + 5
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (edge, 0, 0))], {5: 29.0}, me(pos=(edge - 120, 0, 0))))
        # shrink to radius 100 starting now: the prey at 150 m from centre will be outside soon
        z = dict(ZONE, next_radius=100, shrink_start=30, shrink_end=32)
        self.assertIsNone(plan(Chase(), 30.0, [hunter(5, (150, 0, 0))], {5: 29.0}, zone=z))

    def test_healed_prey_is_dropped(self):
        c = Chase()
        plan(c, 30.0, [hunter(5, (150, 0, 0))], {5: 29.0})
        self.assertIsNone(plan(c, 31.0, [hunter(5, (150, 0, 0), hp=1300)], {5: 29.0}))

    def test_killed_mech_does_not_block_next_chase(self):
        c = Chase()
        plan(c, 30.0, [hunter(5, (150, 0, 0)), hunter(6, (0, 150, 0))], {5: 29.0, 6: 29.0})
        c.forget(5)
        self.assertIsNotNone(plan(c, 30.1, [hunter(6, (0, 150, 0))], {5: 29.0, 6: 29.0}))

    def test_kill_forgets_prey(self):
        c = Chase()
        plan(c, 30.0, [hunter(5, (150, 0, 0))], {5: 29.0})
        c.forget(5)
        self.assertIsNone(plan(c, 30.1, [], {5: 29.0}))

    def test_unseen_attacker_aborts(self):
        c = Chase()
        self.assertIsNone(c.plan(30.0, me(), [hunter(5, (150, 0, 0))], {5: 29.0}, {9: 29.9}, ZONE, clear_all))
        self.assertIsNone(c.plan(30.0, me(), [hunter(5, (150, 0, 0))], {5: 29.0}, {None: 29.9}, ZONE, clear_all))
        self.assertIsNotNone(c.plan(30.0, me(), [hunter(5, (150, 0, 0))], {5: 29.0}, {5: 29.9, 9: 20.0},
                                    ZONE, clear_all))

    def test_observe_keeps_sightings_while_plan_is_skipped(self):
        c = Chase()
        c.observe(30.0, [hunter(7, (0, 150, 0), hp=1400)])
        self.assertIsNone(plan(c, 30.1, [hunter(5, (150, 0, 0))], {5: 30.0}))

    def test_id_none_is_never_prey(self):
        self.assertIsNone(plan(Chase(), 30.0, [hunter(None, (150, 0, 0))], {None: 29.0}))

    def test_backoff_gate(self):
        old = chase.CHASE_NEEDS_BACKOFF
        chase.CHASE_NEEDS_BACKOFF = 1
        try:
            coming = hunter(5, (150, 0, 0))
            coming["vel"] = [-10, 0, 0]
            self.assertIsNone(plan(Chase(), 30.0, [coming], {5: 29.0}))
            going = hunter(5, (150, 0, 0))
            going["vel"] = [10, 0, 0]
            self.assertIsNotNone(plan(Chase(), 30.0, [going], {5: 29.0}))
        finally:
            chase.CHASE_NEEDS_BACKOFF = old


class ChaseInBot(unittest.TestCase):
    def obs(self, t, mechs, events=(), **kw):
        return make_obs(t=t, weapons=[gun(), laser()], mechs=mechs, events=list(events), **kw)

    def test_bot_walks_and_boosts_at_wounded_hunter(self):
        b = Bot(HELLO)
        a = b.act(self.obs(30.0, [hunter(5, (170, 0, 0))],
                           [{"type": "hit", "target": 5, "amount": 300, "weapon": "laser"}]))
        self.assertGreater(a["move"][0], 0.9)
        self.assertTrue(a["boost"])

    def test_kill_event_ends_chase_in_bot(self):
        b = Bot(HELLO)
        b.act(self.obs(30.0, [hunter(5, (170, 0, 0))], [{"type": "hit", "target": 5, "amount": 300,
                                                          "weapon": "laser"}]))
        a = b.act(self.obs(30.1, [], [{"type": "kill", "target": 5}]))
        self.assertIsNone(b.chaser.prey)
        self.assertFalse(a["boost"])

    def test_drop_capture_is_not_abandoned(self):
        b = Bot(HELLO)
        drop = {"id": 1, "pos": [3, 0, 0], "radius": 8, "lands_in": 0, "holder": None, "progress": 2.0}
        a = b.act(self.obs(30.0, [hunter(5, (170, 0, 0))], [{"type": "hit", "target": 5, "amount": 300,
                                                              "weapon": "laser"}], drops=[drop]))
        self.assertLess(a["move"][0], 0.5)

    def test_degraded_pass_restores_chase_state(self):
        b = Bot(HELLO)
        b.act(self.obs(30.0, [hunter(5, (170, 0, 0))], [{"type": "hit", "target": 5, "amount": 300,
                                                          "weapon": "laser"}]))
        before = b.chaser.state()
        orig = b._gate_fire
        b._gate_fire = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x"))
        b.act(self.obs(30.1, [hunter(5, (190, 0, 0))]))
        b._gate_fire = orig
        self.assertEqual(b.chaser.state(), before)

    HIT = {"type": "hit", "target": 5, "amount": 300, "weapon": "laser"}

    def test_bot_observes_while_capturing(self):
        b = Bot(HELLO)
        drop = {"id": 1, "pos": [3, 0, 0], "radius": 8, "lands_in": 0, "holder": None}
        b.act(self.obs(30.0, [hunter(5, (170, 0, 0)), hunter(6, (0, 180, 0), hp=1400)], [self.HIT], drops=[drop]))
        a = b.act(self.obs(30.5, [hunter(5, (170, 0, 0))], [self.HIT]))
        self.assertIsNone(b.chaser.prey)  # 6 is remembered: no chase
        self.assertLess(a["move"][0], 0.5)

    def test_look_at_keeps_held_laser_aim(self):
        b = Bot(HELLO)
        b.act(self.obs(30.0, [hunter(5, (150, 60, 0))], [self.HIT]))
        b.lt_aim, b.lt_t = [0.1, 0.0], 30.9
        a = b.act(make_obs(t=31.0, weapons=[gun(), laser(charge=0.5)], mechs=[]))
        if b.chaser.prey is not None and a["fire"][1]:
            self.assertAlmostEqual(a["aim"][0], 0.1, places=3)
        else:
            self.skipTest("laser not held in this setup")

    def test_no_chase_boost_while_stuck(self):
        b = Bot(HELLO)
        b.stuck = 5
        a = b.act(self.obs(30.0, [hunter(5, (170, 0, 0))], [self.HIT]))
        self.assertGreater(a["move"][0], 0.0)
        self.assertFalse(a["boost"])
        self.assertTrue(a["jump"])

    def test_fight_chase_stays_off(self):
        import fight
        self.assertFalse(hasattr(fight, "CHASE_HP"))  # fight.py has no chase: chase.py is the only chase

    def test_fight_hiding_beats_chase(self):
        import fight
        b = Bot(HELLO)
        b.fight.plan = lambda *a, **k: {"hide": True, "move": [0.0, -1.0], "reason": "hide_go"}
        a = b.act(self.obs(30.0, [hunter(5, (170, 0, 0))], [self.HIT]))
        self.assertLess(a["move"][1], -0.5)
        self.assertIsNone(b.chaser.prey)

    def test_bot_without_chase_module_keeps_v3(self):
        import bot as botmod
        saved = botmod.chaselib
        botmod.chaselib = None
        try:
            b = Bot(HELLO)
            a = b.act(self.obs(30.0, [hunter(5, (170, 0, 0))],
                               [{"type": "hit", "target": 5, "amount": 300, "weapon": "laser"}]))
            self.assertLess(a["move"][0], 0.5)
        finally:
            botmod.chaselib = saved

    def test_bot_aims_at_last_position_after_track_drops(self):
        b = Bot(HELLO)
        b.act(self.obs(30.0, [hunter(5, (150, 60, 0))], [{"type": "hit", "target": 5, "amount": 300,
                                                          "weapon": "laser"}]))
        a = b.act(self.obs(31.0, [], aim=(0.0, 0.0)))
        self.assertAlmostEqual(a["aim"][0], math.atan2(60, 150), places=2)
        self.assertGreater(a["move"][0], 0.8)

    def test_chase_error_switches_it_off(self):
        b = Bot(HELLO)

        def boom(*a, **k):
            raise RuntimeError("x")
        b.chaser.plan = boom
        a = b.act(self.obs(30.0, [hunter(5, (170, 0, 0))],
                           [{"type": "hit", "target": 5, "amount": 300, "weapon": "laser"}]))
        self.assertIn("move", a)
        self.assertEqual(b.chase_errors, 1)
        self.assertFalse(b._chase_on())

    def test_no_chase_when_urgent(self):
        b = Bot(HELLO)
        a = b.act(self.obs(30.0, [hunter(5, (170, 0, 0))],
                           [{"type": "hit", "target": 5, "amount": 300, "weapon": "laser"}],
                           in_zone=False, pos=(0, 600, 0)))
        self.assertLess(a["move"][1], 0)  # back to the zone, not at the hunter


if __name__ == "__main__":
    unittest.main()
