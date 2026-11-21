"""Scenario tests for Bot.act. Run: python test_bot.py (stdlib only)."""
import atexit
import copy
import gzip
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bot as botmod  # noqa: E402
from bot import Bot, ang_diff  # noqa: E402

GUN = {"damage": 45, "cooldown": 0.24, "speed": 252, "range": 230, "spread_deg": 0.6, "mag": 12, "reload": 2.4}
LASER = {"damage": 315, "charge": 0.75, "min_charge": 0.15, "min_frac": 0.35, "hold_s": 2, "cooldown": 1.65, "range": 280}
LIGHT = {"damage": 60, "cooldown": 0.25, "range": 45, "cone_deg": 30, "chain": 2, "energy": 3}
MORT = {"damage": 200, "splash": 160, "cooldown": 2.6, "speed": 72, "gravity": 22}
STATS = {"walk_speed": 11.7, "boost_speed": 19.5, "turn_rate": 4, "pitch_rate": 3, "min_pitch": -0.7,
         "max_pitch": 1.3, "muzzle_height": 7, "near_range_bonus": 0, "vision_bonus": 0}
# an empty world, so the shared ray test never reads the real campus map in unit tests
EMPTY_MAP = tempfile.mkdtemp(prefix="empty_map_")
os.makedirs(os.path.join(EMPTY_MAP, "v2"))
with open(os.path.join(EMPTY_MAP, "v2", "world.json"), "w") as _f:
    json.dump({"prisms": []}, _f)
atexit.register(shutil.rmtree, EMPTY_MAP, True)
HELLO = {"type": "hello", "protocol": "mechfight/1", "slot": 0, "seed": 1,
         "map": {"dir": EMPTY_MAP, "files": []},
         "config": {"loot": {"pickup_radius": 4}, "weapons": {"mortar": {"gravity": 22}},
                    "vision": {"range": 200, "fov_deg": 120, "near_range": 45, "memory_s": 5}}}
STREAM_CANDIDATES = [
    os.environ.get("MECHFIGHT_STREAM", ""),
    os.path.join(HERE, "..", "kit", "mechfight-kit", "check", "stream.jsonl.gz"),
]


def gun(ammo=12, reload=0.0):
    return {"kind": "gun", "cooldown": 0, "ammo": ammo, "reload": reload, "stats": dict(GUN)}


def laser(charge=0.0, cooldown=0.0, stats=None):
    return {"kind": "laser", "cooldown": cooldown, "charge": charge, "stats": dict(stats or LASER)}


def lightning():
    return {"kind": "lightning", "cooldown": 0, "stats": dict(LIGHT)}


def mortar():
    return {"kind": "mortar", "cooldown": 0, "stats": dict(MORT)}


def mech(i, pos, vel=(0, 0, 0), kind="mech"):
    return {"id": i, "kind": kind, "pos": list(pos), "vel": list(vel), "aim": [0, 0], "hp": 1400, "max_hp": 1400}


def pickup(i, kind, pos):
    return {"id": i, "kind": kind, "pos": list(pos)}


def make_obs(t=30.0, tick=1, pos=(0, 0, 0), aim=(0.0, 0.0), hp=1400, energy=100, grounded=True,
             weapons=None, in_zone=True, mechs=(), pickups=(), cubes=(), sounds=(), events=(),
             offer=None, zone=None, vel=(0, 0, 0), drops=(), projectiles=()):
    z = {"center": [0, 0], "radius": 560, "next_center": [0, 0], "next_radius": 380,
         "shrink_start": 60, "shrink_end": 110, "dps": 20}
    z.update(zone or {})
    return {"type": "obs", "tick": tick, "t": t, "alive": 10,
            "self": {"id": 0, "pos": list(pos), "vel": list(vel), "aim": list(aim), "hp": hp, "max_hp": 1400,
                     "energy": energy, "max_energy": 100, "grounded": grounded, "cubes": 0, "upgrades": [],
                     "in_zone": in_zone, "weapons": weapons if weapons is not None else [gun(), None],
                     "stats": dict(STATS)},
            "mechs": list(mechs), "projectiles": list(projectiles), "crates": [], "cubes": list(cubes),
            "pickups": list(pickups), "sounds": list(sounds), "beams": [], "events": list(events), "zone": z,
            "offer": offer, "drops": list(drops)}


def problems(obs, a):
    """Schema-level checks on an action (mirrors what the conformance check enforces)."""
    p = []
    if a.get("tick") != obs["tick"]:
        p.append("tick not echoed")
    mv = a.get("move")
    if mv is not None:
        if not (isinstance(mv, list) and len(mv) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in mv)):
            p.append(f"bad move {mv}")
        elif math.hypot(*mv) > 1 + 1e-9:
            p.append(f"move too long {mv}")
    for key in ("boost", "jump", "reload"):
        if key in a and not isinstance(a[key], bool):
            p.append(f"{key} not bool")
    f = a.get("fire")
    if f is not None and not (isinstance(f, list) and len(f) == 2 and all(isinstance(v, bool) for v in f)):
        p.append(f"bad fire {f}")
    ai = a.get("aim")
    if ai is not None and not (isinstance(ai, list) and len(ai) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in ai)):
        p.append(f"bad aim {ai}")
    if "pick" in a:
        off = obs.get("offer")
        if not (isinstance(a["pick"], int) and off and 0 <= a["pick"] < len(off["choices"])):
            p.append("bad pick")
    for key in ("take", "drop"):
        if key in a and not isinstance(a[key], int):
            p.append(f"{key} not int")
    if "drop" in a and a["drop"] not in (0, 1):
        p.append("drop slot")
    json.dumps(a)
    return p


class Base(unittest.TestCase):
    def setUp(self):
        self.bot = Bot(HELLO)

    def act(self, obs):
        a = self.bot.act(obs)
        self.assertEqual(problems(obs, a), [])
        self.assertEqual(self.bot.errors, 0, "act() fell back to the degraded path")
        return a

    def direct_aim(self, me_pos, tgt_pos, d_z=4.5, muzzle=7):
        x, y, z = me_pos
        tx, ty, tz = tgt_pos
        return [math.atan2(ty - y, tx - x), math.atan2(tz + d_z - (z + muzzle), math.hypot(tx - x, ty - y))]


class FallbackBase(Base):
    """Runs the v3 aim and fire logic (what the bot falls back to when the aim module fails)."""

    def setUp(self):
        super().setUp()
        self.bot.aim_ok = False


class TestWeaponGrab(Base):
    # zone pulls us toward -x (dz 300 > safe_r 289)
    ZONE = {"next_center": [-300, 0]}

    def test_a_far_laser_moves_with_boost(self):
        a = self.act(make_obs(t=3, pickups=[pickup(8001, "laser", (60, 0, 0))], zone=self.ZONE))
        self.assertGreater(a["move"][0], 0.9)
        self.assertAlmostEqual(math.hypot(*a["move"]), 1.0, places=6)
        self.assertTrue(a["boost"])
        self.assertNotIn("take", a)

    def test_a2_zone_pull_without_pickup_goes_the_other_way(self):
        a = self.act(make_obs(t=3, zone=self.ZONE))
        self.assertLess(a["move"][0], -0.3)

    def test_a3_no_calm_gate_with_enemy_and_sound(self):
        sounds = [{"kind": "gun", "bearing": 2.0, "dist": "near", "age": 0.1}]
        a = self.act(make_obs(t=3, pickups=[pickup(8001, "laser", (60, 0, 0))], zone=self.ZONE,
                              mechs=[mech(5, (90, 40, 0))], sounds=sounds))
        self.assertGreater(a["move"][0], 0.5)

    def test_b_laser_in_reach_is_taken(self):
        a = self.act(make_obs(t=3, pickups=[pickup(8001, "laser", (3, 0, 0))]))
        self.assertEqual(a["take"], 8001)
        self.assertNotIn("drop", a)

    def test_b2_lightning_in_reach_is_taken_and_laser_preferred(self):
        a = self.act(make_obs(t=3, pickups=[pickup(8002, "lightning", (1, 0, 0))]))
        self.assertEqual(a["take"], 8002)
        a = self.act(make_obs(t=3, pickups=[pickup(8002, "lightning", (1, 0, 0)), pickup(8001, "laser", (3, 1, 0))]))
        self.assertEqual(a["take"], 8001)

    def test_b3_pickup_out_of_reach_not_taken(self):
        a = self.act(make_obs(t=3, pickups=[pickup(8001, "laser", (4.5, 0, 0))]))
        self.assertNotIn("take", a)
        a = self.act(make_obs(t=3, pickups=[pickup(8001, "laser", (2, 0, 6))]))  # 6 m above
        self.assertNotIn("take", a)

    def test_c_mortar_only_is_ignored(self):
        a = self.act(make_obs(t=3, pickups=[pickup(8003, "mortar", (2, 0, 0))]))
        self.assertNotIn("take", a)
        a = self.act(make_obs(t=3, pickups=[pickup(8003, "mortar", (60, 0, 0))], zone=self.ZONE))
        self.assertLess(a["move"][0], 0)
        self.assertNotIn("take", a)

    def test_c2_prefers_laser_over_nearer_lightning_and_ignores_mortar(self):
        pk = [pickup(1, "lightning", (20, 0, 0)), pickup(2, "laser", (0, 70, 0)), pickup(3, "mortar", (5, 5, 0))]
        a = self.act(make_obs(t=3, pickups=pk, zone=self.ZONE))
        self.assertGreater(a["move"][1], 0.9)
        self.bot = Bot(HELLO)  # fresh: the first bot remembers the laser (loot_mem)
        a = self.act(make_obs(t=3, pickups=pk[:1] + pk[2:], zone=self.ZONE))
        self.assertGreater(a["move"][0], 0.9)

    def test_c3_laser_replaces_a_held_lightning_but_nothing_else(self):
        a = self.act(make_obs(t=3, weapons=[gun(), lightning()], pickups=[pickup(1, "laser", (60, 0, 0))], zone=self.ZONE))
        self.assertGreater(a["move"][0], 0.9)  # loot_mem.SWAP_LIGHTNING: go for the laser
        a = self.act(make_obs(t=3, weapons=[gun(), lightning()], pickups=[pickup(1, "laser", (2, 0, 0))]))
        self.assertEqual((a.get("take"), a.get("drop")), (1, 1))  # lightning (slot 1) out, laser in
        self.bot = Bot(HELLO)
        a = self.act(make_obs(t=3, weapons=[gun(), laser()], pickups=[pickup(1, "lightning", (2, 0, 0))]))
        self.assertNotIn("take", a)
        self.addCleanup(setattr, botmod.lootlib, "SWAP_LIGHTNING", botmod.lootlib.SWAP_LIGHTNING)
        botmod.lootlib.SWAP_LIGHTNING = 0
        self.bot = Bot(HELLO)
        a = self.act(make_obs(t=3, weapons=[gun(), lightning()], pickups=[pickup(1, "laser", (2, 0, 0))]))
        self.assertNotIn("take", a)

    def test_grab_ignores_pickup_outside_zone_circle_and_far_above(self):
        a = self.act(make_obs(t=3, pickups=[pickup(1, "laser", (-100, 600, 0))], zone=self.ZONE))
        self.assertLess(a["move"][0], 0)
        self.assertAlmostEqual(a["move"][1], 0.0)  # pure zone pull, no sideways grab component
        a = self.act(make_obs(t=3, pickups=[pickup(1, "laser", (60, 0, 40))], zone=self.ZONE))
        self.assertLess(a["move"][0], 0)
        self.assertAlmostEqual(a["move"][1], 0.0)

    def test_outside_zone_overrides_grab(self):
        a = self.act(make_obs(t=3, pos=(700, 0, 0), in_zone=False, pickups=[pickup(1, "laser", (760, 0, 0))],
                              zone={"next_center": [0, 0]}))
        self.assertLess(a["move"][0], -0.9)
        self.assertTrue(a["boost"])
        # pickup inside the circle but off to the side: still the zone first
        a = self.act(make_obs(t=3, tick=2, pos=(570, 0, 0), in_zone=False, pickups=[pickup(1, "laser", (500, 200, 0))],
                              zone={"next_center": [0, 0]}))
        self.assertLess(a["move"][0], -0.9)
        self.assertAlmostEqual(a["move"][1], 0.0)

    def test_gone_pickup_is_no_longer_chased(self):
        pk = [pickup(1, "laser", (60, 0, 0)), pickup(2, "lightning", (0, 50, 0))]
        a = self.act(make_obs(t=3, pickups=pk, zone=self.ZONE))
        self.assertGreater(a["move"][0], 0.9)
        ev = [{"type": "take_failed", "id": 1, "reason": "gone"}]
        a = self.act(make_obs(t=3.1, tick=2, pickups=pk, zone=self.ZONE, events=ev))
        self.assertGreater(a["move"][1], 0.9)

    def test_grab_gives_up_after_a_long_chase(self):
        botmod.lootlib.GRAB_GIVE_UP_S, saved = 15.0, botmod.lootlib.GRAB_GIVE_UP_S  # the timings below
        self.addCleanup(setattr, botmod.lootlib, "GRAB_GIVE_UP_S", saved)
        pk = [pickup(1, "laser", (60, 0, 0))]
        self.act(make_obs(t=3, pickups=pk, zone=self.ZONE))
        a = self.act(make_obs(t=20, tick=2, pickups=pk, zone=self.ZONE))
        self.assertLess(a["move"][0], 0)

    def test_chase_clock_survives_the_pickup_leaving_view(self):
        botmod.lootlib.GRAB_GIVE_UP_S, saved = 15.0, botmod.lootlib.GRAB_GIVE_UP_S  # the timings below
        self.addCleanup(setattr, botmod.lootlib, "GRAB_GIVE_UP_S", saved)
        pk = [pickup(1, "laser", (60, 0, 0))]
        self.act(make_obs(t=3, pickups=pk, zone=self.ZONE))
        a = self.act(make_obs(t=9, tick=2, pickups=[], aim=(math.pi, 0.0), zone=self.ZONE))  # looking away: remembered
        self.assertGreater(a["move"][0], 0.9)
        a = self.act(make_obs(t=12, tick=3, pickups=pk, zone=self.ZONE))  # back, 9 s in: still chased
        self.assertGreater(a["move"][0], 0.9)
        self.act(make_obs(t=17, tick=4, pickups=[], zone=self.ZONE))
        a = self.act(make_obs(t=19, tick=5, pickups=pk, zone=self.ZONE))  # 16 s since the first chase: give up
        self.assertLess(a["move"][0], 0)
        a = self.act(make_obs(t=25, tick=6, pickups=pk, zone=self.ZONE))
        self.assertLess(a["move"][0], 0)

    def test_chase_clock_is_per_pickup(self):
        botmod.lootlib.GRAB_GIVE_UP_S, saved = 15.0, botmod.lootlib.GRAB_GIVE_UP_S
        self.addCleanup(setattr, botmod.lootlib, "GRAB_GIVE_UP_S", saved)
        self.act(make_obs(t=3, pickups=[pickup(1, "laser", (60, 0, 0))], zone=self.ZONE))
        a = self.act(make_obs(t=30, tick=2, pickups=[pickup(2, "laser", (60, 0, 0))], zone=self.ZONE))
        self.assertGreater(a["move"][0], 0.9)

    def test_scan_when_early_slot_empty_and_nothing_to_grab(self):
        a = self.act(make_obs(t=5, aim=(0.3, 0.0)))
        self.assertAlmostEqual(ang_diff(a["aim"][0], 0.3), 1.0)
        a = self.act(make_obs(t=5, aim=(0.3, 0.0), pickups=[pickup(3, "mortar", (30, 30, 0))]))
        self.assertAlmostEqual(ang_diff(a["aim"][0], 0.3), 1.0)

    def test_scan_stops_at_20_s_or_with_a_weapon(self):
        a = self.act(make_obs(t=25, aim=(0.3, 0.0)))
        self.assertAlmostEqual(ang_diff(a["aim"][0], 0.3), 0.5)  # plain idle sweep
        a = self.act(make_obs(t=5, aim=(0.3, 0.0), weapons=[gun(), lightning()]))
        self.assertAlmostEqual(ang_diff(a["aim"][0], 0.3), 0.5)

    def test_no_scan_while_walking_to_a_pickup(self):
        a = self.act(make_obs(t=5, aim=(0.0, 0.0), pickups=[pickup(1, "laser", (0, 60, 0))]))
        self.assertAlmostEqual(a["aim"][0], math.pi / 2, places=3)


class TestMortar(Base):
    def test_f_holding_mortar_pitch_is_not_overridden(self):
        e = mech(5, (100, 0, 0))
        obs = make_obs(weapons=[gun(), mortar()], mechs=[e])
        a = self.act(obs)
        gun_aim, _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertAlmostEqual(a["aim"][1], gun_aim[1], places=6)
        self.assertLess(abs(a["aim"][1]), 0.1)  # a mortar lob would be about 0.22
        self.assertFalse(a["fire"][1])

    def test_mortar_dropped_when_calm(self):
        a = self.act(make_obs(weapons=[gun(), mortar()]))
        self.assertEqual(a.get("drop"), 1)
        self.assertNotIn("take", a)

    def test_mortar_kept_while_enemy_visible_or_just_hit(self):
        a = self.act(make_obs(weapons=[gun(), mortar()], mechs=[mech(5, (100, 0, 0))]))
        self.assertNotIn("drop", a)
        ev = [{"type": "damaged", "amount": 50, "weapon": "gun", "bearing": 1.0, "by": 5}]
        a = self.act(make_obs(t=31, tick=2, weapons=[gun(), mortar()], events=ev))
        self.assertNotIn("drop", a)

    def test_mortar_swapped_for_laser_in_reach_even_in_a_fight(self):
        a = self.act(make_obs(weapons=[gun(), mortar()], mechs=[mech(5, (100, 0, 0))],
                              pickups=[pickup(8001, "laser", (2, 1, 0)), pickup(8003, "mortar", (1, 0, 0))]))
        self.assertEqual((a.get("take"), a.get("drop")), (8001, 1))
        a = self.act(make_obs(tick=2, weapons=[gun(), mortar()], pickups=[pickup(8002, "lightning", (2, 1, 0))]))
        self.assertEqual((a.get("take"), a.get("drop")), (8002, 1))

    def test_holding_mortar_never_takes_another_mortar(self):
        a = self.act(make_obs(weapons=[gun(), mortar()], pickups=[pickup(8003, "mortar", (1, 0, 0))]))
        self.assertNotIn("take", a)

    def test_calm_mortar_holder_heads_for_a_laser(self):
        a = self.act(make_obs(t=3, weapons=[gun(), mortar()], pickups=[pickup(1, "laser", (60, 0, 0))],
                              zone={"next_center": [-300, 0]}))
        self.assertGreater(a["move"][0], 0.9)
        self.assertEqual(a.get("drop"), 1)


class TestLaser(FallbackBase):
    def enemy(self, pos=(150, 20, 0), vel=(0, 11.7, 0)):
        return mech(5, pos, vel)

    def test_d_zero_lead_aim_and_charge(self):
        e = self.enemy()
        obs = make_obs(weapons=[gun(), laser()], mechs=[e])
        a = self.act(obs)
        want = self.direct_aim((0, 0, 0), e["pos"])
        self.assertAlmostEqual(a["aim"][0], want[0], places=6)
        self.assertAlmostEqual(a["aim"][1], want[1], places=6)
        lead, _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertGreater(abs(lead[0] - a["aim"][0]), 0.03)  # a lead aim would be visibly different
        self.assertTrue(a["fire"][1])

    def test_e_full_charge_on_target_releases(self):
        e = self.enemy()
        want = self.direct_aim((0, 0, 0), e["pos"])
        a = self.act(make_obs(aim=want, weapons=[gun(), laser(charge=0.75)], mechs=[e]))
        self.assertFalse(a["fire"][1])

    def test_e2_partial_charge_keeps_holding(self):
        e = self.enemy()
        want = self.direct_aim((0, 0, 0), e["pos"])
        a = self.act(make_obs(aim=want, weapons=[gun(), laser(charge=0.4)], mechs=[e]))
        self.assertTrue(a["fire"][1])

    def test_e3_full_charge_off_target_holds_then_gives_up(self):
        e = self.enemy()
        want = self.direct_aim((0, 0, 0), e["pos"])
        off = [want[0] + 0.2, want[1]]
        # the server caps the reported charge at full: hold at exactly 0.75 while time passes
        released = None
        for i in range(0, 16):
            a = self.act(make_obs(t=30 + i / 10, tick=i, aim=off, weapons=[gun(), laser(charge=0.75)], mechs=[e]))
            if not a["fire"][1]:
                released = i / 10
                break
        self.assertIsNotNone(released)
        self.assertGreater(released, 0.5)   # held for 0.5 s first
        self.assertLessEqual(released, 0.7)  # and let go well before hold_s (2 s) fires it blind

    def test_e3b_overdue_timer_restarts_after_the_shot(self):
        e = self.enemy()
        want = self.direct_aim((0, 0, 0), e["pos"])
        off = [want[0] + 0.2, want[1]]
        for i in range(8):  # first hold, released at 0.6 s
            self.act(make_obs(t=30 + i / 10, tick=i, aim=off, weapons=[gun(), laser(charge=0.75)], mechs=[e]))
        a = self.act(make_obs(t=31, tick=20, aim=off, weapons=[gun(), laser(charge=0.0, cooldown=1.6)], mechs=[e]))
        self.assertTrue(a["fire"][1])  # cooldown: charge 0 resets the timer, trigger held for the next one
        a = self.act(make_obs(t=33, tick=21, aim=off, weapons=[gun(), laser(charge=0.75)], mechs=[e]))
        self.assertTrue(a["fire"][1])  # a fresh full charge gets its own 0.5 s
        a = self.act(make_obs(t=33.6, tick=22, aim=off, weapons=[gun(), laser(charge=0.75)], mechs=[e]))
        self.assertFalse(a["fire"][1])

    def test_e4_full_charge_comes_from_weapon_stats(self):
        e = self.enemy()
        want = self.direct_aim((0, 0, 0), e["pos"])
        fast = dict(LASER, charge=0.6)
        a = self.act(make_obs(aim=want, weapons=[gun(), laser(charge=0.6, stats=fast)], mechs=[e]))
        self.assertFalse(a["fire"][1])

    def test_e5_releases_after_the_shot_then_holds_for_the_next(self):
        e = self.enemy()
        want = self.direct_aim((0, 0, 0), e["pos"])
        a = self.act(make_obs(aim=want, weapons=[gun(), laser(charge=0.0, cooldown=1.6)], mechs=[e]))
        self.assertTrue(a["fire"][1])

    def test_laser_charging_while_fleeing_is_released(self):
        self.bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        e = self.enemy((100, 0, 0))
        a = self.act(make_obs(hp=400, weapons=[gun(), laser(charge=0.3)], mechs=[e]))
        self.assertFalse(a["fire"][1])
        a = self.act(make_obs(tick=2, hp=400, weapons=[gun(), laser(charge=0.0)], mechs=[e]))
        self.assertFalse(a["fire"][1])

    def test_target_only_in_memory_is_not_charged(self):
        behind = mech(5, (-120, 0, 0))  # outside the 120 degree wedge, beyond proximity range
        a = self.act(make_obs(weapons=[gun(), laser()], mechs=[behind]))
        self.assertFalse(a["fire"][1])
        self.assertAlmostEqual(abs(a["aim"][0]), math.pi, places=2)  # still turns to look
        a = self.act(make_obs(tick=2, weapons=[gun(), laser(charge=0.4)], mechs=[behind]))
        self.assertFalse(a["fire"][1])

    def test_laser_range_is_capped_at_200_even_with_sensor_upgrade(self):
        obs = make_obs(weapons=[gun(), laser()], mechs=[mech(5, (215, 0, 0))])
        obs["self"]["stats"]["vision_bonus"] = 25
        self.assertFalse(self.act(obs)["fire"][1])
        obs = make_obs(tick=2, weapons=[gun(), laser()], mechs=[mech(5, (190, 0, 0))])
        obs["self"]["stats"]["vision_bonus"] = 25
        self.assertTrue(self.act(obs)["fire"][1])

    def test_target_beyond_200_is_not_charged(self):
        a = self.act(make_obs(weapons=[gun(), laser()], mechs=[mech(5, (230, 0, 0))]))
        self.assertFalse(a["fire"][1])

    def test_adjacent_target_outside_wedge_counts_as_in_view(self):
        a = self.act(make_obs(weapons=[gun(), laser()], mechs=[mech(5, (-30, 0, 0))]))
        self.assertTrue(a["fire"][1])

    def test_laser_is_not_charged_at_a_drone(self):
        a = self.act(make_obs(weapons=[gun(), laser()], mechs=[mech(104, (40, 0, 0), kind="npc")],
                              events=[{"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 0.0, "by": 104}]))
        self.assertFalse(a["fire"][1])

    def test_gun_waits_for_lead_aim_while_laser_aim_is_off_the_gun_line(self):
        e = self.enemy()
        want = self.direct_aim((0, 0, 0), e["pos"])
        obs = make_obs(aim=want, weapons=[gun(), laser(charge=0.3)], mechs=[e])
        a = self.act(obs)
        self.assertFalse(a["fire"][0])  # moving target, laser line is behind the gun lead
        still = mech(5, (150, 20, 0))
        a = self.act(make_obs(tick=2, aim=want, weapons=[gun(), laser(charge=0.3)], mechs=[still]))
        self.assertTrue(a["fire"][0])


class TestEngagement(Base):
    def on_target(self, pos, **kw):
        e = mech(5, pos)
        obs = make_obs(mechs=[e], **kw)
        aim, _ = self.bot.lead_aim(obs["self"], e, 252)
        obs["self"]["aim"] = aim
        return obs

    def test_gun_holds_fire_at_long_range_when_not_hit(self):
        self.assertFalse(self.act(self.on_target((150, 0, 0)))["fire"][0])

    def test_gun_fires_at_close_range(self):
        self.assertTrue(self.act(self.on_target((100, 0, 0)))["fire"][0])

    def test_gun_fires_at_long_range_after_taking_damage(self):
        ev = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]
        self.assertTrue(self.act(self.on_target((150, 0, 0), events=ev))["fire"][0])
        # still counts 2 s later, expires after 3 s
        self.assertTrue(self.act(self.on_target((150, 0, 0), t=32, tick=2))["fire"][0])
        self.assertFalse(self.act(self.on_target((150, 0, 0), t=34, tick=3))["fire"][0])

    def test_zone_damage_does_not_count_as_a_fight(self):
        ev = [{"type": "damaged", "amount": 20, "weapon": "zone", "bearing": None, "by": None}]
        self.assertFalse(self.act(self.on_target((150, 0, 0), events=ev))["fire"][0])

    def test_lone_attacker_at_full_hp_is_fought_not_fled(self):
        # one visible enemy, full health, no zone pull: stand and shoot, laser charges too
        e = mech(5, (100, 0, 0), (-8, 0, 0))
        e["aim"] = [math.pi, 0]
        obs = make_obs(weapons=[gun(), laser()], mechs=[e])
        a = self.act(obs)
        self.assertEqual(a["move"], [0.0, 0.0])
        self.assertTrue(a["fire"][1])
        self.assertAlmostEqual(a["aim"][0], 0.0, places=6)
        obs = self.on_target((100, 0, 0), hp=900)  # 64 percent: still no retreat
        a = self.act(obs)
        self.assertTrue(a["fire"][0])
        self.assertEqual(a["move"], [0.0, 0.0])

    def test_gun_returns_fire_at_long_range_when_a_visible_mech_aims_at_us(self):
        e = mech(5, (170, 0, 0))
        e["aim"] = [math.pi, 0.0]  # pointing straight back at us
        obs = make_obs(mechs=[e])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertTrue(self.act(obs)["fire"][0])
        e["aim"] = [math.pi - 0.5, 0.0]  # looking elsewhere
        self.assertFalse(self.act(obs)["fire"][0])

    def test_gun_fires_at_long_range_when_holding_a_laser(self):
        obs = self.on_target((150, 0, 0), weapons=[gun(), laser(cooldown=1.0)])
        self.assertTrue(self.act(obs)["fire"][0])

    def test_gun_never_fires_empty_or_out_of_range(self):
        self.assertFalse(self.act(self.on_target((100, 0, 0), weapons=[gun(ammo=0), None]))["fire"][0])
        ev = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]
        self.assertFalse(self.act(self.on_target((260, 0, 0), events=ev))["fire"][0])

    def test_lightning_fires_in_cone_and_range(self):
        obs = self.on_target((30, 0, 0), weapons=[gun(), lightning()])
        self.assertTrue(self.act(obs)["fire"][1])
        obs = self.on_target((90, 0, 0), weapons=[gun(), lightning()])
        self.assertFalse(self.act(obs)["fire"][1])


class TestReload(Base):
    def test_reload_when_low_and_no_target(self):
        self.assertTrue(self.act(make_obs(weapons=[gun(ammo=3), None])).get("reload"))
        self.assertTrue(self.act(make_obs(weapons=[gun(ammo=4), None])).get("reload"))

    def test_no_reload_when_ammo_above_four(self):
        self.assertFalse(self.act(make_obs(weapons=[gun(ammo=5), None])).get("reload", False))
        self.assertFalse(self.act(make_obs(weapons=[gun(ammo=12), None])).get("reload", False))

    def test_no_reload_with_a_target_visible_while_rounds_remain(self):
        a = self.act(make_obs(weapons=[gun(ammo=3), None], mechs=[mech(5, (100, 0, 0))]))
        self.assertFalse(a.get("reload", False))

    def test_empty_gun_reloads_even_with_a_target(self):
        a = self.act(make_obs(weapons=[gun(ammo=0), None], mechs=[mech(5, (100, 0, 0))]))
        self.assertTrue(a.get("reload"))
        a = self.act(make_obs(tick=2, weapons=[gun(ammo=0, reload=1.2), None], mechs=[mech(5, (100, 0, 0))]))
        self.assertFalse(a.get("reload", False))


class TestThreatBearing(Base):
    LIGHT_SLOT = None

    def obs(self, **kw):
        kw.setdefault("weapons", [gun(), lightning()])  # slot 1 full: no scanning, no weapon grab
        return make_obs(**kw)

    def hit(self, bearing=1.0):
        return [{"type": "damaged", "amount": 50, "weapon": "gun", "bearing": bearing, "by": 5}]

    def test_g_damage_bearing_survives_a_gun_sound(self):
        a = self.act(self.obs(t=30, events=self.hit(1.0)))
        self.assertAlmostEqual(a["aim"][0], 1.0)
        snd = [{"kind": "gun", "bearing": -2.0, "dist": "near", "age": 0.1}]
        a = self.act(self.obs(t=31, tick=2, sounds=snd))
        self.assertAlmostEqual(a["aim"][0], 1.0)
        a = self.act(self.obs(t=33.9, tick=3, sounds=snd))
        self.assertAlmostEqual(a["aim"][0], 1.0)
        self.assertAlmostEqual(self.bot.threat_yaw, 1.0)

    def test_g2_step_sound_never_sets_the_bearing(self):
        snd = [{"kind": "step", "bearing": -2.0, "dist": "near", "age": 0.1},
               {"kind": "crate", "bearing": 2.5, "dist": "near", "age": 0.1}]
        a = self.act(self.obs(t=30, aim=(0.3, 0), sounds=snd))
        self.assertIsNone(self.bot.threat_yaw)
        self.assertNotAlmostEqual(a["aim"][0], -2.0, places=1)

    def test_g3_sound_updates_once_damage_is_old(self):
        self.act(self.obs(t=30, events=self.hit(1.0)))
        snd = [{"kind": "laser_charge", "bearing": -2.0, "dist": "mid", "age": 0.1}]
        a = self.act(self.obs(t=34.5, tick=2, sounds=snd))
        self.assertAlmostEqual(a["aim"][0], -2.0)

    def test_g4_new_damage_replaces_old_bearing(self):
        self.act(self.obs(t=30, events=self.hit(1.0)))
        a = self.act(self.obs(t=31, tick=2, events=self.hit(-1.5)))
        self.assertAlmostEqual(a["aim"][0], -1.5)

    def test_g5_each_weapon_sound_kind_counts_and_far_ones_do_not(self):
        for kind in ("gun", "laser", "laser_charge", "lightning", "mortar"):
            bot = Bot(HELLO)
            snd = [{"kind": kind, "bearing": 2.0, "dist": "near", "age": 0.1}]
            a = bot.act(self.obs(t=30, sounds=snd))
            self.assertAlmostEqual(a["aim"][0], 2.0, msg=kind)
        bot = Bot(HELLO)
        snd = [{"kind": "gun", "bearing": 2.0, "dist": "far", "age": 0.1}]
        bot.act(self.obs(t=30, sounds=snd))
        self.assertIsNone(bot.threat_yaw)

    def test_g6_zone_damage_has_no_bearing_and_is_ignored(self):
        ev = [{"type": "damaged", "amount": 20, "weapon": "zone", "bearing": None, "by": None}]
        self.act(self.obs(t=30, events=ev))
        self.assertIsNone(self.bot.threat_yaw)

    def test_sounds_do_not_block_looting(self):
        snd = [{"kind": "gun", "bearing": 2.0, "dist": "near", "age": 0.1}]
        a = self.act(make_obs(t=30, weapons=[gun(), None], pickups=[pickup(1, "laser", (0, 60, 0))], sounds=snd))
        self.assertGreater(a["move"][1], 0.9)
        self.bot.loot.seen.clear()  # the laser is gone (taken): only the cube is left
        a = self.act(self.obs(t=31, tick=2, cubes=[{"id": 5001, "pos": [20, 0, 0]}], sounds=snd))
        self.assertGreater(a["move"][0], 0.3)  # quiet walk: 4.5 / 11.7 of full speed

    def test_recent_damage_blocks_cube_looting(self):
        a = self.act(self.obs(t=30, cubes=[{"id": 5001, "pos": [20, 0, 0]}], events=self.hit(1.0)))
        self.assertAlmostEqual(math.hypot(*a["move"]), 0.0, places=6)


class TestUpgrades(Base):
    def choices(self, *ids):
        return {"id": 1, "expires": 99, "choices": [{"id": i, "name": i, "desc": ""} for i in ids]}

    def test_priority_plating_laser_damage_armor_then_rest(self):
        pairs = [(("armor", "plating", "sensors"), 1), (("armor", "laser_damage", "plating"), 2),
                 (("armor", "laser_damage", "nanites"), 1), (("nanites", "armor", "servos"), 1),
                 (("sensors", "gun_rate", "nanites"), 2), (("gun_damage", "gun_rate", "servos"), 0),
                 (("laser_charge", "gun_damage", "jets"), 0), (("jets", "reactor", "capacitor"), 2),
                 (("gun_rate", "gun_damage", "gun_mag"), 2), (("armor", "gun_mag", "nanites"), 1),
                 (("gyros", "servos", "sensors"), 2),
                 (("mortar_splash", "mortar_reload", "jets"), 2), (("mortar_splash", "mortar_reload"), 0)]
        for ids, idx in pairs:
            a = self.act(make_obs(offer=self.choices(*ids)))
            self.assertEqual(a["pick"], idx, ids)

    def test_every_config_upgrade_is_ranked_or_a_mortar(self):
        from bot import UPGRADE_PRIO
        ids = ["plating", "armor", "servos", "capacitor", "reactor", "jets", "nanites", "gyros", "sensors",  # hello.config
               "gun_damage", "gun_rate", "gun_velocity", "gun_mag", "mortar_splash", "mortar_reload", "mortar_velocity",
               "laser_charge", "laser_damage", "laser_pierce", "lightning_chain", "lightning_range", "lightning_efficiency"]
        self.assertEqual(len(set(UPGRADE_PRIO)), len(UPGRADE_PRIO))
        for u in ids:
            self.assertTrue(u in UPGRADE_PRIO or u.startswith("mortar"), u)

    def test_empty_or_odd_offer_never_breaks_the_decision(self):
        for off in ({"id": 1, "expires": 5, "choices": []}, {"id": 1, "expires": 5},
                    {"id": 1, "expires": 5, "choices": [{"name": "x"}, {"id": "plating"}]}):
            a = self.act(make_obs(offer=off))
            self.assertEqual(problems(make_obs(offer=off), a), [])
        a = self.act(make_obs(offer={"id": 1, "expires": 5, "choices": [{"name": "x"}, {"id": "plating"}]}))
        self.assertEqual(a["pick"], 1)

    def test_no_offer_no_pick(self):
        self.assertNotIn("pick", self.act(make_obs()))



class TestKillLoot(Base):
    """economy.py hook: after a kill walk to its pile, then rest without firing."""

    def kill(self, pos=(0, 50, 0)):
        self.act(make_obs(t=30, tick=1, mechs=[mech(5, pos)]))
        return self.act(make_obs(t=30.1, tick=2, hp=900, events=[{"type": "kill", "target": 5}]))

    def test_walks_to_kill_spot_after_kill(self):
        a = self.kill()
        self.assertGreater(a["move"][1], 0.9)
        self.assertFalse(a["boost"])  # boosting is loud: walk to loot

    def test_victims_last_hit_does_not_cancel_loot(self):
        self.act(make_obs(t=29.5, tick=1, mechs=[mech(5, (0, 50, 0))],
                          events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 1.57, "by": 5}]))
        a = self.act(make_obs(t=30.1, tick=2, hp=900, events=[{"type": "kill", "target": 5}]))
        self.assertEqual(self.bot.eco.phase, "loot")
        self.assertGreater(a["move"][1], 0.9)

    def test_rest_keeps_a_charging_laser(self):
        self.kill(pos=(0, 2, 0))
        self.bot._aim_plan = lambda *a, **k: {"aim": [0.0, 0.0], "fire": [True, True], "target": None, "reload": False}
        self.bot._gate_fire = lambda raw, *a, **k: list(raw)
        a = self.act(make_obs(t=30.2, tick=3, hp=900, weapons=[gun(), laser(charge=0.4)]))
        self.assertEqual(self.bot.eco.phase, "rest")
        self.assertEqual(a["fire"], [False, True])  # gun held, charging laser keeps its trigger
        a = self.act(make_obs(t=30.3, tick=4, hp=900, weapons=[gun(), laser(charge=0.0)]))
        self.assertEqual(a["fire"], [False, False])  # no new charge while resting

    def test_contact_aborts_loot(self):
        self.kill()
        a = self.act(make_obs(t=30.2, tick=3, hp=900, aim=(-math.pi / 2, 0), mechs=[mech(6, (0, -100, 0))]))
        self.assertIsNone(self.bot.eco.phase)
        self.assertLess(a["move"][1], 0.5)

    def test_rest_holds_fire_and_a_hit_ends_it(self):
        self.kill(pos=(0, 2, 0))  # already at the spot: straight to rest
        drone = mech(104, (0, 30, 0), kind="npc")
        hit = [{"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 1.57, "by": 104}]
        self.act(make_obs(t=30.2, tick=3, hp=900, weapons=[gun(), None]))
        self.assertEqual(self.bot.eco.phase, "rest")
        a = self.act(make_obs(t=30.3, tick=4, hp=900, weapons=[gun(), None]))
        self.assertEqual(a["fire"], [False, False])
        a = self.act(make_obs(t=30.4, tick=5, hp=900, mechs=[drone], events=hit))  # hit: contact ends the rest
        self.assertIsNone(self.bot.eco.phase)

    def test_rest_is_capped_at_two_seconds(self):
        self.kill(pos=(0, 2, 0))  # at the spot: rest starts on the next decision
        self.act(make_obs(t=30.2, tick=3, hp=900))
        self.act(make_obs(t=31.7, tick=4, hp=900))
        self.assertEqual(self.bot.eco.phase, "rest")
        self.act(make_obs(t=32.4, tick=5, hp=900))
        self.assertIsNone(self.bot.eco.phase)

    def test_rest_ends_at_85_percent(self):
        self.kill(pos=(0, 2, 0))
        self.act(make_obs(t=30.2, tick=3, hp=900))
        self.act(make_obs(t=31, tick=4, hp=1200))
        self.assertIsNone(self.bot.eco.phase)

    def test_broken_economy_turns_looting_off(self):
        self.bot.eco.update = None  # calling it raises TypeError
        a = self.act(make_obs(t=30, cubes=[{"id": 5001, "pos": [0, 60, 0]}]))
        self.assertIsNone(self.bot.eco)
        self.assertEqual(problems(make_obs(t=30), a), [])


class TestEdgeCases(Base):
    def test_empty_everything(self):
        a = self.act(make_obs(weapons=[gun(), None]))
        self.assertEqual(a["fire"], [False, False])

    def test_no_gun_in_slot_zero(self):
        for ws in ([None, None], [None, laser()], [laser(), None]):
            self.act(make_obs(weapons=ws, mechs=[mech(5, (80, 0, 0))]))

    def test_slot_zero_gun_with_lightning_and_npc_only(self):
        self.act(make_obs(weapons=[gun(), lightning()], mechs=[mech(104, (20, 0, 0), kind="npc")]))

    def test_airborne_with_low_energy(self):
        self.act(make_obs(grounded=False, energy=5, vel=(0, 0, 3), pickups=[pickup(1, "laser", (60, 0, 0))]))

    def test_zone_outside_runs_for_the_circle(self):
        a = self.act(make_obs(pos=(800, 0, 0), in_zone=False, zone={"next_center": [0, 0]}))
        self.assertLess(a["move"][0], -0.9)
        self.assertTrue(a["boost"])

    def test_zone_instant_shrink_window(self):
        self.act(make_obs(zone={"shrink_start": 60, "shrink_end": 60}))

    def test_offer_present_with_enemies(self):
        offer = {"id": 1, "expires": 99, "choices": [{"id": "armor"}, {"id": "plating"}, {"id": "sensors"}]}
        a = self.act(make_obs(offer=offer, mechs=[mech(5, (100, 0, 0))]))
        self.assertEqual(a["pick"], 1)

    def test_stuck_triggers_detour_and_jump(self):
        bot = Bot(HELLO)
        for i in range(6):
            obs = make_obs(tick=i, t=3 + i * 0.1, pickups=[pickup(1, "laser", (60, 0, 0))], vel=(0, 0, 0))
            a = bot.act(obs)
            self.assertEqual(problems(obs, a), [])
        self.assertTrue(a["jump"] or bot.detour > 0)

    def test_a_failed_stuck_jump_still_leads_to_a_detour(self):
        """Pressed into a wall: the jump at stuck 3 lifts us off the ground; landing still slow must start the
        detour, not restart the count (sim seed 1648103790: 20 s jumping at one wall, laser never reached)."""
        bot = Bot(HELLO)
        pk = [pickup(1, "laser", (60, 0, 0))]
        tick = 0
        for _ in range(3):
            for grounded in (True, True, True, False, False, False):
                tick += 1
                obs = make_obs(tick=tick, t=3 + tick / 10, pickups=pk, vel=(0, 0, 0), grounded=grounded)
                a = bot.act(obs)
                self.assertEqual(problems(obs, a), [])
                if bot.detour > 0:
                    break
            if bot.detour > 0:
                break
        self.assertGreater(bot.detour, 0)
        self.assertLessEqual(tick, 7)  # 3 slow steps, the jump, landing: detour

    def test_a_detour_does_not_restart_in_mid_air(self):
        """A detour that runs out while we are airborne (stuck count frozen at 4+) must not start a new one until we land:
        restarting in the air kept jetting until energy hit 0 (red team M1, sim seed 700039)."""
        bot = Bot(HELLO)
        pk = [pickup(1, "laser", (60, 0, 0))]
        bot.stuck, bot.detour = 5, 0
        obs = make_obs(tick=1, t=40.0, pickups=pk, vel=(11, 0, 0), grounded=False)
        a = bot.act(obs)
        self.assertEqual(problems(obs, a), [])
        self.assertEqual(bot.detour, 0)
        self.assertFalse(a.get("boost"))
        obs = make_obs(tick=2, t=40.1, pickups=pk, vel=(0, 0, 0), grounded=True)  # landed and still slow: detour
        bot.act(obs)
        self.assertGreater(bot.detour, 0)

    def test_a_wide_detour_never_heads_away_from_the_zone(self):
        """Outside the zone a repeat detour keeps the first, narrow angle: at 2.4 rad it boosted away from the
        centre (red team on 470ca16). The move must still have a component toward the centre."""
        bot = Bot(HELLO)
        bot.detour, bot.detour_lvl, bot.detour_sign = 10, 3, 1
        obs = make_obs(tick=1, t=40.0, pos=(700, 0, 0), in_zone=False, vel=(0, 0, 0))
        a = bot.act(obs)
        self.assertEqual(problems(obs, a), [])
        self.assertLess(a["move"][0], 0)  # centre is at -x

    def test_missing_optional_keys_do_not_crash(self):
        obs = make_obs(mechs=[{"id": 5, "kind": "mech", "pos": [50, 0, 0]}])
        for k in ("projectiles", "crates", "beams", "drops"):
            obs.pop(k)
        self.act(obs)

    def test_debug_label_is_set(self):
        import mechfight as mf
        mf.draw.clear()
        self.act(make_obs())
        self.assertIsNotNone(mf.draw.take())


class TestSlotsAndHeldKinds(Base):
    def test_laser_in_slot_zero_is_fired_by_kind(self):
        e = mech(5, (150, 20, 0))
        a = self.act(make_obs(weapons=[laser(), None], mechs=[e]))
        self.assertEqual(a["fire"], [True, False])
        a = self.act(make_obs(tick=2, weapons=[laser(), gun()], mechs=[e]))
        self.assertTrue(a["fire"][0])

    def test_gun_in_slot_one_fires_from_slot_one(self):
        e = mech(5, (100, 0, 0))
        obs = make_obs(weapons=[None, gun()], mechs=[e])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertEqual(self.act(obs)["fire"], [False, True])

    def test_never_takes_a_kind_already_held(self):
        a = self.act(make_obs(t=3, weapons=[laser(), None], pickups=[pickup(1, "laser", (2, 0, 0))]))
        self.assertNotIn("take", a)
        pk = [pickup(1, "laser", (2, 0, 0)), pickup(2, "lightning", (3, 1, 0))]
        a = self.act(make_obs(t=3, tick=2, weapons=[laser(), None], pickups=pk))
        self.assertEqual(a["take"], 2)
        a = self.act(make_obs(t=3, tick=3, weapons=[gun(), laser()], pickups=[pickup(2, "lightning", (2, 0, 0))]))
        self.assertNotIn("take", a)  # both slots full
        a = self.act(make_obs(t=3, tick=4, weapons=[laser(), mortar()], pickups=[pickup(2, "lightning", (3, 1, 0))]))
        self.assertEqual((a.get("take"), a.get("drop")), (2, 1))
        a = self.act(make_obs(t=3, tick=5, weapons=[laser(), mortar()], pickups=[pickup(1, "laser", (2, 0, 0))]))
        self.assertNotIn("take", a)
        a = self.act(make_obs(t=3, tick=6, weapons=[mortar(), gun()], pickups=[pickup(1, "laser", (2, 0, 0))]))
        self.assertEqual((a.get("take"), a.get("drop")), (1, 0))

    def test_holding_laser_chases_a_lightning_not_a_second_laser(self):
        pk = [pickup(1, "laser", (0, 40, 0)), pickup(2, "lightning", (60, 0, 0))]
        a = self.act(make_obs(t=3, weapons=[laser(), None], pickups=pk, zone={"next_center": [-300, 0]}))
        self.assertGreater(a["move"][0], 0.9)

    def test_optional_weapon_fields_missing(self):
        e = mech(5, (100, 0, 0))
        g = gun()
        del g["ammo"], g["reload"]
        self.act(make_obs(weapons=[g, None], mechs=[e]))
        self.act(make_obs(tick=2, weapons=[g, None]))
        ls = laser()
        del ls["charge"], ls["cooldown"]
        self.act(make_obs(tick=3, weapons=[gun(), ls], mechs=[e]))
        self.act(make_obs(tick=4, weapons=[{"kind": "gun"}, {"kind": "laser"}], mechs=[e]))
        self.act(make_obs(tick=5, weapons=[{"kind": "gun"}, {"kind": "lightning"}], mechs=[mech(5, (20, 0, 0))]))

    def test_zero_hp_and_zero_max_hp(self):
        obs = make_obs(hp=0)
        obs["self"]["max_hp"] = 0
        self.act(obs)


class TestFleeAndDodge(Base):
    ZONE = {"center": [300, 0], "radius": 560, "next_center": [420, 0], "next_radius": 140}

    def test_low_hp_never_closes_on_the_attacker_while_in_zone(self):
        self.bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        e = mech(5, (70, 0, 0))
        e["aim"] = [math.pi, 0]
        ev = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]
        a = self.act(make_obs(hp=420, zone=self.ZONE, mechs=[e], events=ev))
        self.assertLessEqual(a["move"][0], 0.0)
        # outside the zone the circle still wins
        a = self.act(make_obs(tick=2, hp=420, in_zone=False, zone=self.ZONE, mechs=[e], events=ev))
        self.assertGreater(a["move"][0], 0.0)

    def test_low_hp_keeps_the_sideways_part_of_the_zone_pull(self):
        self.bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        e = mech(5, (0, 70, 0))
        zone = {"center": [300, 100], "radius": 560, "next_center": [420, 100], "next_radius": 140}
        a = self.act(make_obs(hp=420, zone=zone, mechs=[e]))
        self.assertGreater(a["move"][0], 0.3)
        self.assertLessEqual(a["move"][1], 0.0)

    def test_sidesteps_a_laser_charging_at_us(self):
        e = mech(5, (120, 0, 0))
        e["aim"] = [math.pi, 0.0]
        e["charging"] = True
        moves = []
        for i in range(5):
            a = self.act(make_obs(tick=i, t=50 + i / 10, mechs=[e], vel=(0, 5, 0)))  # moving: no stuck detour
            self.assertAlmostEqual(math.hypot(*a["move"]), 1.0)
            self.assertAlmostEqual(a["move"][0], 0.0)  # across the line, not along it
            self.assertFalse(a["boost"])  # shooter is 120 m away: no need to burn energy
            moves.append(a["move"][1])
        self.assertEqual(len(set(moves)), 1)  # same side for the whole charge

    def test_no_sidestep_when_the_charging_laser_points_elsewhere(self):
        e = mech(5, (120, 0, 0))
        e["aim"] = [math.pi - 1.0, 0.0]
        e["charging"] = True
        a = self.act(make_obs(mechs=[e]))
        self.assertEqual(a["move"], [0.0, 0.0])

    def shell(self, land, tf=1.0):
        start = (land[0] - 50, land[1], 20.0)
        vel = ((land[0] - start[0]) / tf, 0.0, (0 - start[2] + 0.5 * 22 * tf * tf) / tf)
        return {"id": 900, "kind": "mortar", "pos": list(start), "vel": list(vel)}

    def test_moves_away_from_an_incoming_mortar_shell(self):
        a = self.act(make_obs(projectiles=[self.shell((2, 0))]))
        self.assertLess(a["move"][0], -0.9)
        a = self.act(make_obs(tick=2, projectiles=[self.shell((0, -3))]))
        self.assertGreater(a["move"][1], 0.9)

    def test_sidesteps_a_shell_that_lands_on_us(self):
        a = self.act(make_obs(projectiles=[self.shell((0, 0))]))
        self.assertAlmostEqual(abs(a["move"][1]), 1.0)

    def test_ignores_shells_that_land_late(self):
        a = self.act(make_obs(projectiles=[self.shell((0, 0), tf=6.0)]))
        self.assertEqual(a["move"], [0.0, 0.0])

    def test_laser_sidestep_is_off_outside_the_zone(self):
        e = mech(5, (120, 0, 0))
        e["aim"] = [math.pi, 0.0]
        e["charging"] = True
        e["pos"] = [820, 0, 0]
        a = self.act(make_obs(pos=(700, 0, 0), in_zone=False, mechs=[e], zone={"next_center": [0, 0]}))
        self.assertLess(a["move"][0], -0.9)  # back to the circle, not sideways

    def test_ignores_far_shells_and_enemy_bullets(self):
        a = self.act(make_obs(projectiles=[self.shell((60, 0))]))
        self.assertEqual(a["move"], [0.0, 0.0])
        gun_shot = {"id": 901, "kind": "gun", "pos": [-50, 0, 7], "vel": [200, 0, 0]}
        a = self.act(make_obs(tick=2, projectiles=[gun_shot]))
        self.assertEqual(a["move"], [0.0, 0.0])


class TestGunGate(Base):
    def test_gun_fires_only_inside_the_targets_angular_size(self):
        e = mech(5, (100, 0, 0))
        base, _ = self.bot.lead_aim(make_obs()["self"], e, 252)
        for off, expect in ((0.01, True), (0.03, False)):
            a = self.act(make_obs(mechs=[e], aim=(base[0] + off, base[1])))
            self.assertEqual(a["fire"][0], expect, off)
        e2 = mech(5, (30, 0, 0))
        base2, _ = self.bot.lead_aim(make_obs()["self"], e2, 252)
        a = self.act(make_obs(tick=2, mechs=[e2], aim=(base2[0] + 0.04, base2[1])))
        self.assertTrue(a["fire"][0])  # 0.04 rad is only 1.2 m at 30 m

    HIT = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]

    def test_returns_fire_at_a_hitting_track_beyond_sight(self):
        e = mech(5, (215, 0, 0))
        obs = make_obs(mechs=[e], events=self.HIT)
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertTrue(self.act(obs)["fire"][0])

    def test_return_fire_window_is_three_seconds(self):
        e = mech(5, (215, 0, 0))
        obs = make_obs(t=30, mechs=[e], events=self.HIT)
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertTrue(self.act(obs)["fire"][0])
        self.assertTrue(self.act(dict(obs, tick=2, t=32.9, events=[]))["fire"][0])
        self.assertFalse(self.act(dict(obs, tick=3, t=33.5, events=[]))["fire"][0])

    def test_old_hits_do_not_make_an_unseen_track_a_target(self):
        e = mech(5, (215, 0, 0))
        obs = make_obs(t=30, weapons=[gun(), laser(cooldown=1.0)], mechs=[e], events=self.HIT)  # a laser keeps us engaged
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertTrue(self.act(obs)["fire"][0])
        self.assertFalse(self.act(dict(obs, tick=2, t=40, events=[]))["fire"][0])  # hit 10 s ago, still unseen

    def test_no_fire_at_an_unseen_track_that_is_only_aimed_at_us(self):
        e = mech(5, (215, 0, 0))
        e["aim"] = [math.pi, 0.0]
        obs = make_obs(mechs=[e])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertFalse(self.act(obs)["fire"][0])
        obs["self"]["stats"]["vision_bonus"] = 25  # sensors upgrade: it can be seen now
        obs["tick"] = 2
        self.assertTrue(self.act(obs)["fire"][0])


class TestLineOfFire(Base):
    BUILDING = {"kind": "building", "base": -10, "top": 30, "ring": [40, -20, 60, -20, 60, 20, 40, 20]}

    def bot_with(self, *prisms):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "v2"))
        with open(os.path.join(d, "v2", "world.json"), "w") as f:
            json.dump({"prisms": list(prisms)}, f)
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = d
        return Bot(hello)

    def shoot(self, bot, enemy_pos, weapons=None):
        e = mech(5, enemy_pos)
        obs = make_obs(weapons=weapons or [gun(), None], mechs=[e],
                       events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}])
        obs["self"]["aim"], _ = bot.lead_aim(obs["self"], e, 252)
        return bot.act(obs)

    def test_no_gun_or_laser_into_a_building(self):
        bot = self.bot_with(self.BUILDING)
        self.assertFalse(self.shoot(bot, (100, 0, 0))["fire"][0])
        a = self.shoot(bot, (100, 0, 0), [gun(), laser()])
        self.assertEqual(a["fire"], [False, False])
        a = self.shoot(bot, (30, 0, 0), [gun(), lightning()])  # same side of the wall: fine
        self.assertEqual(a["fire"], [True, True])

    def test_clear_lines_still_fire(self):
        bot = self.bot_with(self.BUILDING)
        self.assertTrue(self.shoot(bot, (100, 60, 0))["fire"][0])   # passes beside it
        self.assertTrue(self.shoot(bot, (-100, 0, 0))["fire"][0])   # other direction
        self.assertTrue(self.shoot(bot, (100, 0, 0), [gun(), laser()])["fire"][1] is False)  # blocked laser
        self.assertTrue(self.shoot(bot, (100, 60, 0), [gun(), laser()])["fire"][1])         # clear laser

    def retreat_obs(self, pos, hp, enemy=(100, 0, 0), **kw):
        return make_obs(pos=pos, hp=hp, weapons=[gun(), laser()], mechs=[mech(5, enemy)],
                        events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}], **kw)

    def test_hurt_walks_to_the_nearest_spot_the_shooter_cannot_see(self):
        bot = self.bot_with(self.BUILDING)  # we stand at (30, 40) in its view; behind the building's corner is hidden
        a = bot.act(self.retreat_obs((30, 40, 0), 300))
        self.assertTrue(bot.retreating)
        self.assertLess(a["move"][1], -0.5)  # down toward the building's shadow
        self.assertGreater(bot.act(self.retreat_obs((30, 40, 0), 1400, tick=2))["move"][1], -0.5)  # healthy: no

    def test_hidden_and_hurt_holds_fire_until_80_percent_unless_hit_or_finishing(self):
        def bot_and_obs(hp, enemy_hp=1400, events=()):
            e = dict(mech(5, (100, 0, 0)), hp=enemy_hp)
            o = make_obs(hp=hp, weapons=[gun(), None], mechs=[e], events=list(events))
            o["self"]["aim"], _ = bot.lead_aim(o["self"], e, 252)
            return o
        bot = self.bot_with()  # open ground: our gun has a clear line; cover_spot says "hidden here"
        bot.cover_spot = lambda t, x, y, *a: (x, y)
        a = bot.act(bot_and_obs(300))
        self.assertEqual(a["move"], [0.0, 0.0])
        self.assertEqual(a["fire"], [False, False])
        a = bot.act(dict(bot_and_obs(1000), tick=2))  # 71%: still resting
        self.assertEqual(a["fire"], [False, False])
        self.assertTrue(bot.retreating)
        hit = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 9}]
        self.assertTrue(bot.act(dict(bot_and_obs(1000, events=hit), tick=3))["fire"][0])  # hit while hiding: shoot
        bot.act(dict(bot_and_obs(1150), t=40.0, tick=4))  # 82%: fight again
        self.assertFalse(bot.retreating)
        bot = self.bot_with()
        bot.cover_spot = lambda t, x, y, *a: (x, y)
        self.assertTrue(bot.act(bot_and_obs(300, enemy_hp=150))["fire"][0])  # nearly dead and in line: finish it

    def test_a_wall_that_hides_our_centre_but_not_our_top_is_not_cover(self):
        low_wall = {"kind": "building", "base": -10, "top": 6, "ring": [33, -20, 36, -20, 36, 20, 33, 20]}
        threat = [mech(5, (100, 0, 0))]
        self.assertNotEqual(self.bot_with(low_wall).cover_spot(0.0, 30, 0, 0.0, threat, 0, 0, 1e9), (30, 0))
        self.assertEqual(self.bot_with(self.BUILDING).cover_spot(0.0, 30, 0, 0.0, threat, 0, 0, 1e9), (30, 0))

    def test_no_cover_spot_past_the_zone_edge_line(self):
        threat = [mech(5, (100, 0, 0))]
        self.assertIsNotNone(self.bot_with(self.BUILDING).cover_spot(0.0, 30, 40, 0.0, threat, 0, 0, 1e9))
        self.assertIsNone(self.bot_with(self.BUILDING).cover_spot(0.0, 30, 40, 0.0, threat, 30, 40, 28))

    def test_no_hiding_when_cornered_or_outside_the_zone(self):
        bot = self.bot_with(self.BUILDING)
        bot.act(self.retreat_obs((30, 40, 0), 300, enemy=(30, 55, 0)))  # 15 m away: fight back
        self.assertIsNone(bot.cover)
        bot = self.bot_with(self.BUILDING)
        bot.act(self.retreat_obs((30, 40, 0), 300, in_zone=False))
        self.assertFalse(bot.retreating)

    def test_no_cover_in_reach_fights_on_as_before(self):
        bot = self.bot_with()  # open ground
        bot.act(self.retreat_obs((30, 40, 0), 300))
        self.assertFalse(bot.retreating)

    def test_holes_and_low_roofs_do_not_block(self):
        holed = dict(self.BUILDING, holes=[[30, -5, 70, -5, 70, 5, 30, 5]])
        self.assertTrue(self.shoot(self.bot_with(holed), (100, 0, 0))["fire"][0])
        low = dict(self.BUILDING, base=-10, top=2)
        self.assertTrue(self.shoot(self.bot_with(low), (100, 0, 0))["fire"][0])
        bridge = dict(self.BUILDING, kind="bridge", base=12, top=14)
        self.assertTrue(self.shoot(self.bot_with(bridge), (100, 0, 0))["fire"][0])

    def test_line_rising_over_a_low_building_is_clear(self):
        low = dict(self.BUILDING, top=10)  # the line is at 18 to 23 m up over the building
        high = dict(self.BUILDING, top=25)
        self.assertTrue(self.shoot(self.bot_with(low), (100, 0, 30))["fire"][0])
        self.assertFalse(self.shoot(self.bot_with(high), (100, 0, 30))["fire"][0])

    def test_return_fire_beyond_sight_still_needs_a_clear_line(self):
        self.assertFalse(self.shoot(self.bot_with(self.BUILDING), (215, 0, 0))["fire"][0])
        self.assertTrue(self.shoot(self.bot_with(), (215, 0, 0))["fire"][0])

    def test_prism_sampling_still_guards_the_line_when_ray_is_missing(self):
        import unittest.mock as um
        bot = self.bot_with(self.BUILDING)
        bot.aim_ok = False
        with um.patch.object(botmod, "raylib", None), um.patch.object(botmod, "aimlib", None):
            self.assertFalse(self.shoot(bot, (100, 0, 0))["fire"][0])
            self.assertTrue(self.shoot(bot, (100, 60, 0))["fire"][0])
            self.assertTrue(self.shoot(bot, (30, 0, 0))["fire"][0])

    def test_unreadable_map_means_no_line_of_fire_check(self):
        import unittest.mock as um
        hello = copy.deepcopy(HELLO)
        del hello["map"]
        with um.patch.object(botmod.raylib, "clear", side_effect=OSError("gone")):
            self.assertTrue(self.shoot(Bot(hello), (100, 0, 0))["fire"][0])
            bot = Bot(hello)
            bot.aim_ok = False  # the v3 helper too
            self.assertTrue(self.shoot(bot, (100, 0, 0))["fire"][0])


class TestSupplyDrop(Base):
    def drop(self, pos=(35, -18, 0), **kw):
        d = {"id": 20000, "pos": list(pos), "radius": 8, "lands_in": 0, "capture_s": 4, "progress": 0, "holder": None}
        d.update(kw)
        return d

    def test_goes_for_a_landed_uncontested_drop(self):
        a = self.act(make_obs(t=100, hp=700, drops=[self.drop()]))
        want = math.atan2(-18, 35)
        self.assertAlmostEqual(math.atan2(a["move"][1], a["move"][0]), want, places=3)
        self.assertTrue(a["boost"])

    def test_holds_still_inside_the_ring_and_when_it_is_ours(self):
        a = self.act(make_obs(t=100, pos=(33, -17, 0), drops=[self.drop(holder=0)]))
        self.assertEqual(a["move"], [0.0, 0.0])

    def test_ignores_contested_far_early_or_outside_zone_drops(self):
        cases = [self.drop(holder=7), self.drop(pos=(300, 0, 0)), self.drop(lands_in=60)]
        for i, d in enumerate(cases):
            a = self.act(make_obs(tick=i, t=100, drops=[d]))
            self.assertEqual(a["move"], [0.0, 0.0], d)
        a = self.act(make_obs(tick=9, t=100, drops=[self.drop()], zone={"center": [-200, 0], "radius": 100,
                                                                         "next_center": [-200, 0], "next_radius": 60}))
        self.assertLess(a["move"][0], 0.0)  # the zone is the other way

    def test_ignores_a_drop_with_a_mech_near_it_or_us(self):
        a = self.act(make_obs(t=100, drops=[self.drop()], mechs=[mech(5, (60, -18, 0))]))
        self.assertAlmostEqual(math.hypot(*a["move"]), 0.0)
        a = self.act(make_obs(tick=2, t=100, drops=[self.drop()], mechs=[mech(5, (-40, 0, 0))]))
        self.assertAlmostEqual(math.hypot(*a["move"]), 0.0)

    def test_weapon_grab_comes_before_the_drop(self):
        a = self.act(make_obs(t=100, drops=[self.drop()], pickups=[pickup(1, "laser", (0, 50, 0))]))
        self.assertGreater(a["move"][1], 0.9)


class TestSideStep(Base):
    def shooter(self, pos):
        e = mech(5, pos)
        e["aim"] = [math.atan2(-pos[1], -pos[0]), 0.0]
        e["charging"] = True
        return e

    def test_boost_only_when_the_shooter_is_close(self):
        a = self.act(make_obs(mechs=[self.shooter((40, 0, 0))], vel=(0, 5, 0)))
        self.assertTrue(a["boost"])
        a = self.act(make_obs(tick=2, mechs=[self.shooter((120, 0, 0))], vel=(0, 5, 0)))
        self.assertFalse(a["boost"])

    def test_side_alternates_every_half_second(self):
        e = self.shooter((120, 0, 0))
        signs = []
        for i in range(12):
            a = self.act(make_obs(tick=i, t=50 + i / 10, mechs=[e], vel=(0, 5, 0)))
            signs.append(1 if a["move"][1] > 0 else -1)
        self.assertEqual(len(set(signs[:5])), 1)
        self.assertEqual(signs[5:10], [-signs[0]] * 5)
        self.assertEqual(signs[10], signs[0])

    def test_no_sidestep_while_fleeing(self):
        self.bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        a = self.act(make_obs(hp=400, mechs=[self.shooter((100, 0, 0))], vel=(0, 5, 0)))
        self.assertAlmostEqual(a["move"][1], 0.0)
        self.assertLess(a["move"][0], 0.0)

    def test_no_sidestep_when_the_zone_edge_is_close(self):
        a = self.act(make_obs(pos=(540, 0, 0), mechs=[self.shooter((640, 0, 0))], vel=(5, 0, 0)))
        self.assertLess(a["move"][0], -0.9)


class TestDropGuards(Base):
    def drop(self):
        return {"id": 20000, "pos": [35, -18, 0], "radius": 8, "lands_in": 0, "capture_s": 4, "progress": 0, "holder": None}

    def test_a_mech_within_150_m_blocks_the_drop(self):
        a = self.act(make_obs(t=100, drops=[self.drop()], mechs=[mech(5, (-120, 0, 0))]))
        self.assertAlmostEqual(math.hypot(*a["move"]), 0.0)
        a = self.act(make_obs(tick=2, t=100, drops=[self.drop()], mechs=[mech(5, (-170, 0, 0))]))
        self.assertGreater(a["move"][0], 0.5)

    def test_a_mech_near_the_drop_blocks_it_even_when_far_from_us(self):
        far_drop = dict(self.drop(), pos=[140, 0, 0])
        a = self.act(make_obs(t=100, drops=[far_drop], mechs=[mech(5, (190, 0, 0))]))  # 190 m from us, 50 m from it
        self.assertAlmostEqual(math.hypot(*a["move"]), 0.0)
        a = self.act(make_obs(tick=2, t=100, drops=[far_drop], mechs=[mech(5, (210, 0, 0))]))  # 70 m from it
        self.assertGreater(a["move"][0], 0.5)

    def test_recent_damage_blocks_the_drop_for_three_seconds(self):
        hit = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 3.0, "by": 5}]
        a = self.act(make_obs(t=100, drops=[self.drop()], events=hit))
        self.assertAlmostEqual(math.hypot(*a["move"]), 0.0)
        a = self.act(make_obs(tick=2, t=102.9, drops=[self.drop()]))
        self.assertAlmostEqual(math.hypot(*a["move"]), 0.0)
        a = self.act(make_obs(tick=3, t=103.5, drops=[self.drop()]))
        self.assertGreater(a["move"][0], 0.5)

    def test_holding_in_the_ring_clears_a_detour(self):
        bot = Bot(HELLO)
        for i in range(6):  # stuck on the way: a detour starts
            bot.act(make_obs(tick=i, t=100 + i / 10, drops=[self.drop()], vel=(0, 0, 0)))
        self.assertGreater(bot.detour, 0)
        obs = make_obs(tick=9, t=101, pos=(33, -17, 0), drops=[self.drop()], vel=(0, 0, 0))
        a = bot.act(obs)
        self.assertEqual(a["move"], [0.0, 0.0])
        self.assertEqual(bot.detour, 0)


class TestLaserGrace(FallbackBase):
    def setUp(self):
        super().setUp()
        self.e = mech(5, (150, 20, 0))
        self.want = self.direct_aim((0, 0, 0), self.e["pos"])

    def test_holds_charge_and_aim_for_half_a_second_after_the_target_leaves_view(self):
        a = self.act(make_obs(t=30, aim=self.want, weapons=[gun(), laser(charge=0.3)], mechs=[self.e]))
        self.assertTrue(a["fire"][1])
        a = self.act(make_obs(tick=2, t=30.3, aim=self.want, weapons=[gun(), laser(charge=0.6)], mechs=[]))
        self.assertTrue(a["fire"][1])
        self.assertAlmostEqual(a["aim"][0], self.want[0])
        self.assertAlmostEqual(a["aim"][1], self.want[1])
        a = self.act(make_obs(tick=3, t=30.6, aim=self.want, weapons=[gun(), laser(charge=0.75)], mechs=[]))
        self.assertFalse(a["fire"][1])  # more than 0.5 s since it was in view

    def test_no_grace_without_a_charge_or_when_fleeing(self):
        self.bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        self.act(make_obs(t=30, aim=self.want, weapons=[gun(), laser(charge=0.3)], mechs=[self.e]))
        a = self.act(make_obs(tick=2, t=30.1, weapons=[gun(), laser(charge=0.0)], mechs=[]))
        self.assertFalse(a["fire"][1])
        a = self.act(make_obs(tick=3, t=30.2, hp=400, weapons=[gun(), laser(charge=0.4)], mechs=[]))
        self.assertFalse(a["fire"][1])

    def test_a_full_charge_in_grace_is_let_go_once_overdue(self):
        off = (self.want[0] + 0.3, self.want[1])  # full charge but off target: holds
        for i, t_ in enumerate((30.0, 30.3, 30.45)):
            a = self.act(make_obs(tick=i, t=t_, aim=off, weapons=[gun(), laser(charge=0.75)], mechs=[self.e]))
            self.assertTrue(a["fire"][1])
        a = self.act(make_obs(tick=9, t=30.52, aim=off, weapons=[gun(), laser(charge=0.75)], mechs=[]))
        self.assertFalse(a["fire"][1])  # target just left view, but the full charge is overdue


class TestStateRestore(Base):
    def test_failed_pass_leaves_no_counters_behind(self):
        class Flaky(Bot):
            def _gate_fire(self, *args):
                raise IndexError("gate")

        bot = Flaky(HELLO)
        obs = make_obs(t=3, pickups=[pickup(1, "laser", (60, 0, 0))], mechs=[mech(5, (100, 0, 0))], vel=(0, 0, 0))
        for i in range(3):
            bot.act(dict(obs, tick=i + 1, t=3 + i / 10))
        self.assertEqual(bot.stuck, 3)  # once per decision, not twice
        self.assertEqual(bot.errors, 3)
        self.assertEqual(bot.chase, {1: 3})

    def test_events_are_counted_once_when_a_pass_fails(self):
        class Flaky(Bot):
            def _gate_fire(self, *args):
                raise IndexError("gate")

        bot = Flaky(HELLO)
        ev = [{"type": "take_failed", "id": 7, "reason": "range"}]
        bot.act(make_obs(t=3, mechs=[mech(5, (100, 0, 0))], events=ev))
        self.assertEqual(bot.fails, {7: 1})
        self.assertEqual(bot.skip, set())
        bot.act(make_obs(tick=2, t=3.1, mechs=[mech(5, (100, 0, 0))],
                         events=[{"type": "take_failed", "id": 8, "reason": "gone"}]))
        self.assertEqual(bot.skip, {8})


class TestRememberedFollowThrough(Base):
    """A mech we hit (or that hit us) in the last 5 s stays a target out of view, up to the weapon's range."""
    BUILDING = TestLineOfFire.BUILDING
    bot_with = TestLineOfFire.bot_with
    WE_HIT = [{"type": "hit", "target": 5, "amount": 315, "weapon": "laser"}]
    HIT_US = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]

    def engage(self, bot, events):
        bot.act(make_obs(t=30, mechs=[mech(5, (150, 0, 0))], weapons=[gun(), laser(cooldown=1.6)], events=events))

    def step(self, bot, t, tick, pos, weapons, aim=None):
        e = mech(5, pos)
        obs = make_obs(t=t, tick=tick, weapons=weapons, mechs=[e])
        self.assertFalse(bot.in_view(obs["self"], e))  # remembered, not seen
        if aim is not None:
            obs["self"]["aim"] = list(aim)
        a = bot.act(obs)
        self.assertEqual(problems(obs, a), [])
        self.assertEqual(bot.errors, 0)
        return a

    def test_a_charges_holds_and_releases_on_a_mech_we_hit_at_241_m(self):
        bot = self.bot_with()
        self.engage(bot, self.WE_HIT)
        a = self.step(bot, 31.6, 2, (241, 0, 0), [gun(), laser()])
        self.assertTrue(a["fire"][1])  # new charge at a remembered mech we just hit
        on = list(a["aim"])
        self.assertLess(abs(on[0]), 0.03)
        a = self.step(bot, 31.9, 3, (241, 0, 0), [gun(), laser(charge=0.3)], aim=on)
        self.assertTrue(a["fire"][1])  # keeps charging
        a = self.step(bot, 32.4, 4, (241, 0, 0), [gun(), laser(charge=0.75)], aim=on)
        self.assertFalse(a["fire"][1])  # full and on target: released

    def test_b_no_release_toward_a_remembered_mech_behind_a_building(self):
        bot = self.bot_with(self.BUILDING)
        self.engage(bot, self.WE_HIT)
        aim = bot.direct_aim(make_obs()["self"], mech(5, (241, 0, 0)))[0]
        a = self.step(bot, 31.6, 2, (241, 0, 0), [gun(), laser()], aim=aim)
        self.assertFalse(a["fire"][1])  # no new charge into the wall
        a = self.step(bot, 31.9, 3, (241, 0, 0), [gun(), laser(charge=0.75)], aim=aim)
        self.assertTrue(a["fire"][1])  # a full charge is held, not let go into the wall

    def test_c_no_new_charge_at_a_never_engaged_remembered_mech_at_230_m(self):
        bot = self.bot_with()
        bot.act(make_obs(t=30, mechs=[mech(5, (150, 0, 0))], weapons=[gun(), laser(cooldown=1.6)]))
        a = self.step(bot, 31.6, 2, (230, 0, 0), [gun(), laser()])
        self.assertFalse(a["fire"][1])

    def test_engagement_expires_after_5_s(self):
        bot = self.bot_with()
        self.engage(bot, self.WE_HIT)
        a = self.step(bot, 35.2, 2, (241, 0, 0), [gun(), laser()])
        self.assertFalse(a["fire"][1])

    def test_fallback_logic_also_charges_at_a_mech_we_hit(self):
        bot = self.bot_with()
        bot.aim_ok = False
        self.engage(bot, self.WE_HIT)
        self.assertTrue(self.step(bot, 31.6, 2, (241, 0, 0), [gun(), laser()])["fire"][1])
        bot2 = self.bot_with()
        bot2.aim_ok = False
        bot2.act(make_obs(t=30, mechs=[mech(5, (150, 0, 0))], weapons=[gun(), laser(cooldown=1.6)]))
        self.assertFalse(self.step(bot2, 31.6, 2, (230, 0, 0), [gun(), laser()])["fire"][1])

    def test_gun_answers_a_remembered_attacker_only_with_a_clear_line(self):
        # the gun keeps its 3 s return-fire window (TestGunGate); this pins the clear-path part out of view
        for blocked in (False, True):
            bot = self.bot_with(self.BUILDING) if blocked else self.bot_with()
            self.engage(bot, self.HIT_US)
            aim, _ = bot.lead_aim(make_obs()["self"], mech(5, (210, 0, 0)), 252)
            a = self.step(bot, 32.0, 2, (210, 0, 0), [gun(), None], aim=aim)
            self.assertEqual(a["fire"][0], not blocked)

    def test_visible_healthy_mech_beats_a_remembered_wounded_one_behind_a_wall(self):
        for far in ((120, 0, 0), (241, 0, 0)):
            bot = self.bot_with(self.BUILDING)
            self.engage(bot, self.WE_HIT)
            hurt, seen = mech(5, far), mech(7, (0, 150, 0))
            hurt["hp"] = 600
            obs = make_obs(t=31, tick=2, weapons=[gun(), laser(cooldown=1.6)], mechs=[hurt, seen])
            obs["self"]["aim"], _ = bot.lead_aim(obs["self"], seen, 252)
            self.assertFalse(bot.in_view(obs["self"], hurt))
            a = bot.act(obs)
            self.assertEqual(bot.errors, 0)
            self.assertLess(abs(a["aim"][0] - math.pi / 2), 0.05, far)  # stays on the mech we see
            self.assertTrue(a["fire"][0], far)

    def test_b_fallback_no_charge_into_a_wall_at_a_remembered_mech(self):
        bot = self.bot_with(self.BUILDING)
        bot.aim_ok = False
        self.engage(bot, self.WE_HIT)
        aim = bot.direct_aim(make_obs()["self"], mech(5, (241, 0, 0)))[0]
        self.assertFalse(self.step(bot, 31.6, 2, (241, 0, 0), [gun(), laser()], aim=aim)["fire"][1])
        clear = self.bot_with()
        clear.aim_ok = False
        self.engage(clear, self.WE_HIT)
        self.assertTrue(self.step(clear, 31.6, 2, (241, 0, 0), [gun(), laser()], aim=aim)["fire"][1])

    def test_lite_pass_does_not_charge_at_a_remembered_mech_through_a_wall(self):
        class Flaky(Bot):
            def _gate_fire(self, *args):
                raise IndexError("gate")  # aim path fails: the lite pass runs _v3_fire with no ray test

        bot = self.bot_with(self.BUILDING)
        bot.__class__ = Flaky
        bot.act(make_obs(t=30, mechs=[mech(5, (150, 0, 0))], weapons=[gun(), laser(cooldown=1.6)],
                         events=self.WE_HIT))
        e = mech(5, (241, 0, 0))
        obs = make_obs(t=31.6, tick=2, weapons=[gun(), laser()], mechs=[e])
        obs["self"]["aim"] = bot.direct_aim(obs["self"], e)[0]
        self.assertFalse(bot.in_view(obs["self"], e))
        a = bot.act(obs)
        self.assertEqual(bot.errors, 2)  # both decisions went through the lite pass
        self.assertFalse(a["fire"][1])

    def test_hit_on_survives_a_degraded_pass_once(self):
        class Flaky(Bot):
            def _gate_fire(self, *args):
                if not self.lite:
                    raise IndexError("gate")
                return Bot._gate_fire(self, *args)

        bot = Flaky(HELLO)
        bot.act(make_obs(t=30, mechs=[mech(5, (150, 0, 0))], events=self.WE_HIT))
        self.assertEqual(bot.errors, 1)
        self.assertEqual(bot.hit_on, {5: 30})
        bot.act(make_obs(tick=2, t=30.5, mechs=[mech(5, (150, 0, 0))]))
        self.assertEqual(bot.hit_on, {5: 30})  # no event: nothing new recorded

    def test_gun_does_not_chase_a_remembered_mech_only_we_hit(self):
        bot = self.bot_with()
        self.engage(bot, self.WE_HIT)
        aim, _ = bot.lead_aim(make_obs()["self"], mech(5, (210, 0, 0)), 252)
        self.assertFalse(self.step(bot, 34.0, 2, (210, 0, 0), [gun(), None], aim=aim)["fire"][0])


class TestLineOfFireCost(Base):
    def world_bot(self, cls=Bot):
        prisms = []
        for gx in range(-8, 9):
            for gy in range(-8, 9):
                x, y = gx * 40, gy * 40
                prisms.append({"kind": "building", "base": -10, "top": 30,
                               "ring": [x + 5, y + 5, x + 25, y + 5, x + 25, y + 25, x + 5, y + 25]})
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "v2"))
        with open(os.path.join(d, "v2", "world.json"), "w") as f:
            json.dump({"prisms": prisms}, f)
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = d
        return cls(hello)

    def crowd(self):
        out = [mech(99, (112, 12, 0))]  # the nearest one, straight ahead
        for k in range(15):
            ang = -0.9 + k * 0.12
            r = 120 + 5 * k
            out.append(mech(k + 1, (r * math.cos(ang), r * math.sin(ang), 0)))
        return out

    def test_at_most_four_lines_of_fire_per_decision(self):
        calls = []

        class Counting(Bot):
            def _los_raw(self, me, tgt):
                calls.append(tgt["id"])
                return super()._los_raw(me, tgt)

        bot = self.world_bot(Counting)
        obs = make_obs(pos=(12, 12, 0), weapons=[gun(), laser()], mechs=self.crowd())
        bot.act(obs)
        self.assertLessEqual(len(calls), 4)
        self.assertEqual(len(calls), len(set(calls)))  # each mech at most once per decision
        calls.clear()
        bot.act(dict(obs, tick=2))
        self.assertLessEqual(len(calls), 4)  # the cache does not leak across decisions

    def test_worst_case_decision_time(self):
        bot = self.world_bot()
        times = []
        for i in range(150):
            obs = make_obs(tick=i, t=30 + i / 10, pos=(12 + i % 7, 12, 0), weapons=[gun(), laser()], mechs=self.crowd(),
                           events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}])
            t0 = time.perf_counter()
            bot.act(obs)
            times.append((time.perf_counter() - t0) * 1000)
        self.assertLess(max(times), 10.0, f"slowest decision {max(times):.2f} ms")
        print(f"\nlos worst case: max {max(times):.2f} ms", file=sys.stderr)


def find_arena():
    for p in (os.environ.get("MECHFIGHT_ARENA", ""), os.path.join(HERE, "..", "kit", "mechfight-kit", "arena")):
        if p and os.path.exists(os.path.join(p, "v2", "world.json")):
            return p
    return None


class TestRealMapCost(unittest.TestCase):
    @unittest.skipUnless(find_arena(), "kit arena not found (set MECHFIGHT_ARENA)")
    def test_sixteen_mechs_around_us_on_the_real_map(self):
        import unittest.mock as um
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = find_arena()
        bot = Bot(hello)
        self.assertGreater(len(bot.prisms), 100)
        calls = [0]
        real_clear = botmod.raylib.clear

        def counting(a, b, terrain=True):
            calls[0] += 1
            return real_clear(a, b, terrain)

        rng = random.Random(5)
        times, most_calls = [], 0
        with um.patch.object(botmod.raylib, "clear", side_effect=counting):
            for k in range(400):
                cx, cy = rng.uniform(-300, 400), rng.uniform(-250, 330)
                ms = []
                for j in range(16):
                    ang, r = rng.uniform(-math.pi, math.pi), rng.uniform(40, 195)
                    m = mech(j + 1, (cx + r * math.cos(ang), cy + r * math.sin(ang), 0),
                             (rng.uniform(-10, 10), rng.uniform(-10, 10), 0))
                    m["aim"] = [rng.uniform(-3, 3), 0]
                    ms.append(m)
                ws = [[gun(), laser()], [gun(), lightning()], [gun(), None]][k % 3]
                obs = make_obs(tick=k, t=30 + k / 10, pos=(cx, cy, 0), aim=(rng.uniform(-math.pi, math.pi), 0),
                               weapons=ws, mechs=ms,
                               events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]
                               if k % 2 == 0 else [])
                calls[0] = 0
                t0 = time.perf_counter()
                a = bot.act(obs)
                times.append((time.perf_counter() - t0) * 1000)
                most_calls = max(most_calls, calls[0])
                self.assertEqual(problems(obs, a), [])
        self.assertEqual(bot.errors, 0)
        self.assertEqual(bot.aim_errors, 0)
        self.assertLessEqual(most_calls, 6)  # a few lines of fire per decision, not one per mech
        self.assertLess(max(times), 10.0, f"slowest decision {max(times):.2f} ms")
        print(f"\nreal map, 16 mechs: max {max(times):.2f} ms, most ray calls {most_calls}", file=sys.stderr)


class TestMapLog(Base):
    def log_for(self, hello):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            bot = Bot(hello)
        return buf.getvalue(), bot

    def test_hello_logs_how_many_prisms_loaded(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "v2"))
        prism = {"kind": "building", "base": 0, "top": 20, "ring": [0, 0, 5, 0, 5, 5, 0, 5]}
        with open(os.path.join(d, "v2", "world.json"), "w") as f:
            json.dump({"prisms": [prism, prism, prism]}, f)
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = d
        out, bot = self.log_for(hello)
        self.assertEqual(len(bot.prisms), 3)
        self.assertIn(f"map: prisms=3 dir={d} ray=3", out)

    def test_hello_logs_zero_when_the_map_is_missing(self):
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = "/arena"  # does not exist here
        out, bot = self.log_for(hello)
        self.assertEqual(bot.prisms, [])
        self.assertIn("map: prisms=0 dir=/arena", out)

    @unittest.skipUnless(find_arena(), "kit arena not found (set MECHFIGHT_ARENA)")
    def test_real_map_loads_259_prisms(self):
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = find_arena()
        out, bot = self.log_for(hello)
        self.assertEqual(len(bot.prisms), 259)
        self.assertIsNotNone(bot.bounds)
        self.assertIn("map: prisms=259", out)


class TestAimWiring(Base):
    HIT = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}]

    def test_aim_module_is_in_charge(self):
        self.assertTrue(self.bot.aim_ok)
        e = mech(5, (150, 20, 0), (0, 11.7, 0))
        a = self.act(make_obs(weapons=[gun(), laser()], mechs=[e]))
        self.assertTrue(a["fire"][1])  # laser starts charging
        self.assertLess(abs(a["aim"][0] - math.atan2(20, 150)), 0.03)
        on = list(a["aim"])
        a = self.act(make_obs(tick=2, aim=on, weapons=[gun(), laser(charge=0.75)], mechs=[e]))
        self.assertFalse(a["fire"][1])  # full and on target: released

    def test_engagement_gate_keeps_the_gun_quiet_until_provoked(self):
        e = mech(5, (150, 0, 0))
        obs = make_obs(mechs=[e])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertFalse(self.act(obs)["fire"][0])
        hit = make_obs(tick=2, mechs=[e], events=self.HIT)
        hit["self"]["aim"] = obs["self"]["aim"]
        self.assertTrue(self.act(hit)["fire"][0])

    def test_fleeing_lets_a_charging_laser_go(self):
        self.bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        e = mech(5, (100, 0, 0))
        a = self.act(make_obs(hp=400, weapons=[gun(), laser(charge=0.3)], mechs=[e]))
        self.assertFalse(a["fire"][1])

    def test_new_laser_charge_needs_a_mech_in_view_within_200_m(self):
        obs = make_obs(weapons=[gun(), laser()], mechs=[mech(5, (215, 0, 0))])
        obs["self"]["stats"]["vision_bonus"] = 25
        self.assertFalse(self.act(obs)["fire"][1])
        a = self.act(make_obs(tick=2, weapons=[gun(), laser()], mechs=[mech(5, (-120, 0, 0))]))
        self.assertFalse(a["fire"][1])  # memory only: behind us, beyond the proximity range

    def test_laser_is_not_started_at_a_drone_that_hit_us(self):
        e = mech(104, (40, 0, 0), kind="npc")
        obs = make_obs(weapons=[gun(), laser()], mechs=[e], events=[
            {"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 0.0, "by": 104}])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        a = self.act(obs)
        self.assertFalse(a["fire"][1])
        self.assertTrue(a["fire"][0])  # but the gun answers

    def test_idle_drone_is_not_a_target_and_a_mech_beats_a_hitting_drone(self):
        a = self.act(make_obs(mechs=[mech(104, (30, 0, 0), kind="npc")]))
        self.assertEqual(a["fire"], [False, False])
        drone = mech(104, (30, 5, 0), kind="npc")
        mech_far = mech(5, (-100, 0, 0))
        a = self.act(make_obs(tick=2, mechs=[drone, mech_far], events=[
            {"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 0.0, "by": 104}]))
        self.assertAlmostEqual(abs(a["aim"][0]), math.pi, places=1)  # faces the mech, as v3 did

    def test_grace_hold_after_the_target_leaves_view(self):
        e = mech(5, (150, 20, 0))
        a = self.act(make_obs(t=30, weapons=[gun(), laser(charge=0.3)], mechs=[e]))
        self.assertTrue(a["fire"][1])
        last = a["aim"]
        a = self.act(make_obs(tick=2, t=30.3, weapons=[gun(), laser(charge=0.6)], mechs=[]))
        self.assertTrue(a["fire"][1])
        self.assertAlmostEqual(a["aim"][0], last[0])
        a = self.act(make_obs(tick=3, t=30.6, weapons=[gun(), laser(charge=0.75)], mechs=[]))
        self.assertFalse(a["fire"][1])

    def test_a_charge_in_progress_is_kept_even_when_the_target_drifts_past_200_m(self):
        e = mech(5, (215, 0, 0))
        obs = make_obs(weapons=[gun(), laser(charge=0.3)], mechs=[e])
        obs["self"]["stats"]["vision_bonus"] = 25
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertTrue(self.act(obs)["fire"][1])

    def test_grace_hold_is_off_when_fleeing(self):
        self.bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        e = mech(5, (150, 20, 0))
        self.act(make_obs(t=30, weapons=[gun(), laser(charge=0.3)], mechs=[e]))
        a = self.act(make_obs(tick=2, t=30.1, hp=400, weapons=[gun(), laser(charge=0.4)], mechs=[]))
        self.assertFalse(a["fire"][1])

    def test_no_hold_without_a_charge_in_progress(self):
        e = mech(5, (150, 20, 0))
        self.act(make_obs(t=30, weapons=[gun(), laser()], mechs=[e]))
        a = self.act(make_obs(tick=2, t=30.1, weapons=[gun(), laser(charge=0.0)], mechs=[]))
        self.assertFalse(a["fire"][1])

    def test_a_mech_we_can_hit_beats_a_better_scored_one_behind_a_wall(self):
        hidden = mech(5, (60, 0, 0))
        hidden["hp"] = 300  # wounded and closer: the aim module's favourite
        open_ = mech(6, (100, 60, 0))
        self.bot.los_clear = lambda me, m: m.get("id") != 5
        a = self.act(make_obs(weapons=[gun(), laser()], mechs=[hidden, open_]))
        self.assertLess(abs(a["aim"][0] - math.atan2(60, 100)), 0.05)  # aims at the one it can hit

    def test_a_line_toggling_at_a_wall_edge_does_not_flip_the_target(self):
        hidden = mech(5, (60, 0, 0))
        hidden["hp"] = 300
        open_ = mech(6, (100, 60, 0))
        for k in range(8):  # mech 5's line toggles every tick (0.1 s)
            self.bot.los_clear = (lambda me, m, k=k: m.get("id") != 5 or k % 2 == 0)
            a = self.act(make_obs(t=30 + 0.1 * k, tick=k + 1, weapons=[gun(), laser()], mechs=[hidden, open_]))
            self.assertLess(abs(a["aim"][0]), 0.05, f"tick {k}")  # stays on mech 5

    def test_all_blocked_keeps_the_old_target_choice(self):
        hidden = mech(5, (60, 0, 0))
        hidden["hp"] = 300
        self.bot.los_clear = lambda me, m: False
        a = self.act(make_obs(weapons=[gun(), laser()], mechs=[hidden, mech(6, (100, 60, 0))]))
        self.assertLess(abs(a["aim"][0]), 0.05)  # still turns to the best target

    def test_reload_comes_from_the_aim_plan_and_the_v3_rule(self):
        a = self.act(make_obs(weapons=[gun(ammo=0), None], mechs=[mech(5, (100, 0, 0))]))
        self.assertTrue(a.get("reload"))
        a = self.act(make_obs(tick=2, weapons=[gun(ammo=3), None]))
        self.assertTrue(a.get("reload"))  # quiet moment, low magazine

    def test_a_wounded_remembered_mech_does_not_pull_the_aim_off_a_visible_one(self):
        near = mech(5, (100, 0, 0))
        far = mech(6, (-200, 0, 0))
        far["hp"] = 100  # wounded, but behind us and only remembered
        obs = make_obs(mechs=[near, far])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], near, 252)
        for i in range(5):
            a = self.act(dict(obs, tick=i + 1, t=30 + i / 10))
            self.assertLess(abs(a["aim"][0]), 0.05)  # still on the mech in front of us
            self.assertTrue(a["fire"][0])

    def test_a_wounded_far_mech_does_not_silence_the_gun_while_a_close_one_is_in_view(self):
        near = mech(5, (100, 0, 0))
        far = mech(6, (200, 15, 0))
        far["hp"] = 100
        obs = make_obs(mechs=[near, far])
        shots = 0
        for i in range(10):
            a = self.act(dict(obs, tick=i + 1, t=30 + i / 10))
            obs["self"]["aim"] = list(a["aim"])  # the torso follows the command
            shots += bool(a["fire"][0])
        self.assertGreaterEqual(shots, 5)

    def test_engaged_comes_from_every_seen_mech_not_only_the_chosen_target(self):
        near = mech(5, (100, 0, 0))   # inside 120 m: we are in a fight
        far = mech(6, (190, 15, 0))   # the aim module prefers this one (wounded), at 190 m and in view
        far["hp"] = 100
        obs = make_obs(mechs=[near, far])
        a = self.act(obs)
        self.assertLess(abs(a["aim"][0] - math.atan2(15, 190)), 0.03)  # it did pick the far one
        obs["self"]["aim"] = list(a["aim"])
        self.assertTrue(self.act(dict(obs, tick=2))["fire"][0])

    def test_only_drones_within_60_m_after_a_hit_are_targets(self):
        idle = mech(104, (80, 0, 0), kind="npc")
        hitter = mech(105, (0, 55, 0), kind="npc")
        ev = [{"type": "damaged", "amount": 25, "weapon": "npc", "bearing": 1.57, "by": 7}]
        a = self.act(make_obs(mechs=[idle, hitter], events=ev))
        self.assertAlmostEqual(a["aim"][0], math.pi / 2, places=1)  # the near drone, not the idle one

    def test_a_hit_by_someone_else_does_not_make_an_unseen_track_shootable(self):
        e = mech(5, (215, 0, 0))
        obs = make_obs(mechs=[e], events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 2.0, "by": 9}])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertFalse(self.act(obs)["fire"][0])
        unknown = [{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": None}]
        obs2 = dict(obs, tick=2, events=unknown)
        self.assertTrue(self.act(obs2)["fire"][0])  # source unknown: any track may be the attacker

    def test_aim_failure_falls_back_to_v3_logic(self):
        import unittest.mock as um
        e = mech(5, (100, 0, 0))
        obs = make_obs(mechs=[e])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        with um.patch.object(botmod.aimlib, "decide", side_effect=RuntimeError("boom")):
            a = self.act(obs)
            self.assertTrue(a["fire"][0])
            self.assertEqual(self.bot.aim_errors, 1)

        def failing(obs, **kw):
            botmod.aimlib.last_plan = {"error": "ValueError: x", "fire": [False, False], "aim": None}
            return None, [False, False], None

        with um.patch.object(botmod.aimlib, "decide", side_effect=failing):
            a = self.act(dict(obs, tick=2))
            self.assertTrue(a["fire"][0])
            self.assertEqual(self.bot.aim_errors, 2)

    def test_lite_pass_and_a_missing_aim_module_use_v3_logic(self):
        self.bot.lite = True
        self.assertIsNone(self.bot._aim_plan(make_obs(), []))
        self.bot.lite = False
        self.bot.aim_ok = False
        self.assertIsNone(self.bot._aim_plan(make_obs(), []))
        e = mech(5, (100, 0, 0))
        obs = make_obs(mechs=[e])
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        self.assertTrue(self.act(obs)["fire"][0])

    def test_line_of_fire_goes_through_ray_clear(self):
        import unittest.mock as um
        e = mech(5, (100, 0, 0))
        obs = make_obs(mechs=[e], events=self.HIT)
        obs["self"]["aim"], _ = self.bot.lead_aim(obs["self"], e, 252)
        calls = []

        def blocked(a, b, terrain=True):
            calls.append((a, b))
            return False

        with um.patch.object(botmod.raylib, "clear", side_effect=blocked):
            a = self.act(obs)
        self.assertFalse(a["fire"][0])
        self.assertTrue(calls)
        self.assertAlmostEqual(calls[0][0][2], 7.0)  # from the muzzle

    def test_v3_los_helper_uses_ray(self):
        import unittest.mock as um
        me = make_obs()["self"]
        e = mech(5, (100, 0, 0))
        with um.patch.object(botmod.raylib, "clear", return_value=False) as c:
            self.assertFalse(self.bot.los_clear(me, e))
            self.assertEqual(c.call_count, 1)
            self.bot.los_clear(me, e)
            self.assertEqual(c.call_count, 1)  # cached per decision
        self.bot.lite = True
        self.bot._los_cache = {}
        self.assertTrue(self.bot.los_clear(me, e))


class TestDetour(Base):
    def test_detour_keeps_one_side_for_the_whole_obstacle_then_flips_when_it_fails(self):
        bot = Bot(HELLO)
        signs, wides = [], []
        for i in range(60):
            obs = make_obs(tick=i, t=3 + i / 10, pickups=[pickup(1, "laser", (60, 0, 0))],
                           zone={"next_center": [-300, 0]})
            a = bot.act(obs)
            self.assertEqual(problems(obs, a), [])
            h = math.atan2(a["move"][1], a["move"][0])
            signs.append(0 if all(abs(abs(h) - x) > 1e-6 for x in botmod.DETOUR_ANGLES) else (1 if h > 0 else -1))
            wides.append(abs(h))
        first = signs[3:28]
        self.assertNotIn(0, first)
        self.assertEqual(len(set(first)), 1)             # one side for the whole 2.5 s detour
        self.assertAlmostEqual(wides[3], botmod.DETOUR_ANGLES[0])
        second = signs[28:53]
        self.assertEqual(set(second), {-first[0]})       # still stuck afterwards: other side...
        self.assertAlmostEqual(wides[30], botmod.DETOUR_ANGLES[1])  # ...and wider: same obstacle

    def test_a_detour_that_runs_free_then_hits_the_same_wall_widens_and_runs_full(self):
        """Seed 625440276: each detour ran free for 2 s (early exit), turned back to the goal and hit the same
        wall: 75 s stuck. Stuck again near the last detour's start: other side, wider, no early exit."""
        bot = Bot(HELLO)
        pk = [pickup(1, "laser", (60, 0, 0))]
        tick = 0

        def step(vel):
            nonlocal tick
            tick += 1
            return bot.act(make_obs(tick=tick, t=3 + tick / 10, pickups=pk, vel=vel))
        for _ in range(4):
            step((0, 0, 0))
        self.assertEqual((bot.detour > 0, bot.detour_lvl), (True, 0))
        sign0 = bot.detour_sign
        for _ in range(botmod.FREE_N):
            step((8, 0, 0))  # runs free: the first detour ends early
        self.assertEqual(bot.detour, 0)
        for _ in range(4):
            step((0, 0, 0))  # same spot (obs pos stays 0, 0): stuck again
        self.assertEqual(bot.detour_lvl, 1)
        self.assertEqual(bot.detour_sign, -sign0)
        for _ in range(botmod.FREE_N + 2):
            a = step((8, 0, 0))
        self.assertGreater(bot.detour, 0)  # no early exit this time
        self.assertAlmostEqual(abs(math.atan2(a["move"][1], a["move"][0])), botmod.DETOUR_ANGLES[1])

    def test_detour_prefers_the_side_with_room(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "v2"))
        wall = {"kind": "building", "base": -10, "top": 30, "ring": [10, 0, 30, 0, 30, 80, 10, 80]}  # north-east block
        with open(os.path.join(d, "v2", "world.json"), "w") as f:
            json.dump({"prisms": [wall]}, f)
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = d
        bot = Bot(hello)
        h = None
        for i in range(6):
            obs = make_obs(tick=i, t=3 + i / 10, pickups=[pickup(1, "laser", (60, 0, 0))],
                           zone={"next_center": [-300, 0]})
            a = bot.act(obs)
            h = math.atan2(a["move"][1], a["move"][0])
        self.assertAlmostEqual(h, -1.2)  # north-east of us is walled, so go south


class TestDegradedDecision(Base):
    def test_failing_map_part_still_fires_and_moves(self):
        class Flaky(Bot):
            def _gate_fire(self, *args):
                raise IndexError("gate")

        bot = Flaky(HELLO)
        e = mech(5, (100, 0, 0))
        obs = make_obs(mechs=[e], events=[{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 5}])
        obs["self"]["aim"], _ = bot.lead_aim(obs["self"], e, 252)
        a = bot.act(obs)
        self.assertEqual(problems(obs, a), [])
        self.assertTrue(a["fire"][0])
        self.assertEqual(bot.errors, 1)
        self.assertFalse(bot.lite)  # back to the full path on the next decision
        self.assertEqual(bot.act(dict(obs, tick=2))["fire"][0], True)
        self.assertEqual(bot.errors, 2)

    def test_everything_failing_falls_back_to_walking_to_the_zone(self):
        class Broken(Bot):
            def _decide(self, obs):
                raise RuntimeError("boom")

        bot = Broken(HELLO)
        obs = make_obs(pos=(50, 0, 0), zone={"next_center": [-200, 0]}, mechs=[mech(5, (100, 0, 0))])
        a = bot.act(obs)
        self.assertEqual(problems(obs, a), [])
        self.assertEqual(a["fire"], [False, False])
        self.assertAlmostEqual(a["move"][0], -1.0)
        bad = dict(obs, tick=3)
        del bad["zone"]
        a = bot.act(bad)  # even the fallback's own inputs may be missing
        self.assertEqual(a["tick"], 3)
        self.assertEqual(a["move"], [0.0, 0.0])

    def test_lite_pass_skips_the_extras_but_matches_the_core(self):
        bot = Bot(HELLO)
        bot.lite = True
        e = mech(5, (100, 0, 0))
        obs = make_obs(weapons=[gun(), laser()], mechs=[e])
        a = bot.act(obs)  # act() resets lite; the full path gives the same answer here
        self.assertEqual(problems(obs, a), [])
        self.assertTrue(a["fire"][1])


class TestMapEdgeAndConfig(Base):
    def bot_with_bounds(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        os.makedirs(os.path.join(d, "v2"))
        with open(os.path.join(d, "v2", "world.json"), "w") as f:
            json.dump({"prisms": [], "bounds": {"minX": -500, "maxX": 500, "minY": -300, "maxY": 300}}, f)
        hello = copy.deepcopy(HELLO)
        hello["map"]["dir"] = d
        return Bot(hello)

    def flee_obs(self, **kw):
        e = mech(5, (-150, -260, 0))  # inland of us (we are 6 m from the south edge)
        zone = {"center": [-150, -150], "radius": 300, "next_center": [-150, -150], "next_radius": 250,
                "shrink_start": 300, "shrink_end": 350}
        return make_obs(pos=(-150, -294, 0), hp=420, mechs=[e], zone=zone, vel=(5, 0, 0), **kw)

    def test_flee_runs_along_the_edge_at_full_speed(self):
        bot = self.bot_with_bounds()
        bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        a = bot.act(self.flee_obs())
        self.assertAlmostEqual(abs(a["move"][0]), 1.0)
        self.assertAlmostEqual(a["move"][1], 0.0)

    def test_blocked_along_the_edge_flips_direction_and_keeps_it(self):
        bot = self.bot_with_bounds()
        first = bot.act(self.flee_obs())["move"][0]
        xs = []
        for i in range(10):
            obs = self.flee_obs(tick=i + 2, t=30 + i / 10)
            obs["self"]["vel"] = [0, 0, 0] if -first not in xs else [5, 0, 0]  # a wall, until we turn round
            xs.append(bot.act(obs)["move"][0])
        self.assertIn(-first, xs)
        k = xs.index(-first)
        self.assertTrue(all(v == -first for v in xs[k:]))

    def test_no_slide_away_from_the_edge(self):
        bot = self.bot_with_bounds()
        bot.fight = None  # v3 flee: only without fight.py (fight.py keeps fighting at low HP)
        obs = self.flee_obs()
        obs["self"]["pos"] = [-150, 0, 0]
        obs["mechs"] = [mech(5, (-150, 40, 0))]
        a = bot.act(obs)
        self.assertLess(a["move"][1], -0.1)  # plain flee, straight away

    def test_zero_valued_config_does_not_divide_by_zero(self):
        g = gun()
        g["stats"]["speed"] = 0
        obs = make_obs(weapons=[g, None], mechs=[mech(5, (50, 0, 0))])
        obs["self"]["stats"].update(walk_speed=0, boost_speed=0, turn_rate=0, muzzle_height=0)
        self.act(obs)
        obs = make_obs(tick=2, weapons=[g, None], zone={"next_center": [-300, 0]})
        obs["self"]["stats"].update(walk_speed=0)
        self.act(obs)


class TestWrapper(unittest.TestCase):
    def test_coast_then_idle_after_repeated_failures(self):
        code = ("import mechfight as mf\n"
                "class B:\n"
                "    n = 0\n"
                "    def __init__(self, h): pass\n"
                "    def act(self, o):\n"
                "        B.n += 1\n"
                "        if B.n > 1: raise RuntimeError('boom')\n"
                "        return {'move': [1, 0], 'boost': True, 'aim': [0.5, 0]}\n"
                "mf.run(B)\n")
        hello = copy.deepcopy(HELLO)
        hello["config"]["bot"] = {"coast_s": 1.0}
        p = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=HERE, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            def send(m):
                p.stdin.write((json.dumps(m) + "\n").encode())
                p.stdin.flush()
                return json.loads(p.stdout.readline().decode())
            send(hello)
            acts = [send(make_obs(tick=i, t=1 + i / 10)) for i in range(25)]
            p.stdin.close()
            p.wait(timeout=20)
        finally:
            p.kill()
            p.stdout.close()
            p.stderr.close()
        self.assertEqual(acts[5]["move"], [1, 0])          # coasting on the last good action
        self.assertTrue(acts[5]["boost"])
        self.assertEqual(acts[20]["move"], [0, 0])         # idle after coast_s
        self.assertFalse(acts[20]["boost"])
        self.assertEqual(acts[20]["aim"], [0.5, 0])        # aim is held
        self.assertTrue(all(a["tick"] == i for i, a in enumerate(acts)))


class TestMainEntrypoint(unittest.TestCase):
    def test_dockerfile_runs_bot_py_as_main(self):
        with open(os.path.join(HERE, "Dockerfile")) as f:
            text = f.read()
        self.assertIn('ENTRYPOINT ["python3", "-u", "/bot/bot.py"]', text)
        copy_lines = [l for l in text.splitlines() if l.startswith("COPY ")]
        self.assertEqual(len(copy_lines), 1)
        for name in ("mechfight.py", "bot.py", "aim.py", "ray.py", "fight.py", "chase.py", "economy.py"):
            self.assertIn(name, copy_lines[0].split())
        with open(os.path.join(HERE, "bot.py")) as f:
            self.assertIn('if __name__ == "__main__":', f.read())

    def test_import_does_not_start_the_loop(self):
        r = subprocess.run([sys.executable, "-c", "import bot"], cwd=HERE, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=20)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout, b"")

    def test_run_as_main_speaks_the_protocol(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        p = subprocess.Popen([sys.executable, "-u", os.path.join(HERE, "bot.py")], cwd=HERE, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        try:
            def send(m):
                p.stdin.write((json.dumps(m) + "\n").encode())
                p.stdin.flush()
                return json.loads(p.stdout.readline().decode())
            ready = send(HELLO)
            self.assertEqual(ready["type"], "ready")
            for junk in (b"[1, 2]\n", b"42\n", b"null\n", b"\"text\"\n", b"{not json\n", b"\xff\xfe\xfa bad \xc3\x28\n"):
                p.stdin.write(junk)  # out of contract lines are ignored
            for tick in (7, 8, 9):
                a = send(make_obs(tick=tick, t=30 + tick / 10, mechs=[mech(5, (100, 0, 0))]))
                self.assertEqual(a["tick"], tick)
                self.assertEqual(problems(make_obs(tick=tick), a), [])
            p.stdin.write((json.dumps({"type": "end", "place": 3, "kills": 0}) + "\n").encode())
            p.stdin.close()
            self.assertEqual(p.wait(timeout=20), 0)
            self.assertNotIn("act() failed", p.stderr.read().decode())
        finally:
            p.kill()
            p.stdout.close()
            p.stderr.close()


def rand_obs(rng, i):
    ws0 = rng.choice([gun(rng.randint(0, 12), rng.choice([0, 0, 1.3])), gun(), None])
    ws1 = rng.choice([None, None, laser(rng.choice([0, 0.2, 0.75, 1.5]), rng.choice([0, 0, 1.0])), lightning(), mortar()])
    pos = (rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(-5, 30))
    mechs = []
    for k in range(rng.randint(0, 4)):
        ang, d = rng.uniform(-math.pi, math.pi), rng.choice([8, 30, 60, 120, 190, 260])
        mechs.append(mech(k + 1, (pos[0] + d * math.cos(ang), pos[1] + d * math.sin(ang), pos[2] + rng.uniform(-3, 8)),
                          (rng.uniform(-12, 12), rng.uniform(-12, 12), 0), rng.choice(["mech", "mech", "npc"])))
    pk = [pickup(8000 + k, rng.choice(["laser", "lightning", "mortar", "gun"]),
                 (pos[0] + rng.uniform(-90, 90), pos[1] + rng.uniform(-90, 90), pos[2] + rng.choice([0, 0, 15])))
          for k in range(rng.randint(0, 4))]
    snd = [{"kind": rng.choice(["gun", "step", "laser", "laser_charge", "mortar", "crate", "death", "boost", "jet"]),
            "bearing": rng.uniform(-4, 4), "dist": rng.choice(["near", "mid", "far"]), "age": rng.random() * 0.1}
           for _ in range(rng.randint(0, 3))]
    ev = []
    if rng.random() < 0.2:
        ev.append({"type": "damaged", "amount": 40, "weapon": rng.choice(["gun", "laser", "zone", "npc"]),
                   "bearing": rng.choice([None, rng.uniform(-4, 4)]), "by": 3})
    if rng.random() < 0.1:
        ev.append({"type": "take_failed", "id": 8000 + rng.randint(0, 4), "reason": rng.choice(["gone", "range", "have", "full", "busy"])})
    offer = None
    if rng.random() < 0.1:
        offer = {"id": 1, "expires": 9, "choices": [{"id": c} for c in rng.sample(
            ["armor", "plating", "nanites", "sensors", "servos", "gun_damage", "gun_rate", "jets", "reactor"], 3)]}
    r = rng.choice([40, 300, 900])
    zone = {"center": [0, 0], "radius": r, "next_center": [rng.uniform(-100, 100), rng.uniform(-100, 100)],
            "next_radius": max(r * 0.6, 20), "shrink_start": rng.choice([0, 60]), "shrink_end": rng.choice([0, 60, 110])}
    return make_obs(t=rng.uniform(0, 400), tick=i, pos=pos, aim=(rng.uniform(-3.1, 3.1), rng.uniform(-0.5, 0.5)),
                    hp=rng.choice([100, 500, 1400]), energy=rng.choice([0, 20, 60, 100]),
                    grounded=rng.random() < 0.8, weapons=[ws0, ws1], in_zone=rng.random() < 0.8, mechs=mechs,
                    pickups=pk, cubes=[{"id": 5000, "pos": [pos[0] + 10, pos[1], pos[2]]}] if rng.random() < 0.2 else [],
                    sounds=snd, events=ev, offer=offer, zone=zone, vel=(rng.uniform(-10, 10), rng.uniform(-10, 10), 0))


def find_stream():
    for p in STREAM_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return None


class TestReplay(unittest.TestCase):
    def run_stream(self, obs_list, hello, mutate=None):
        bot = Bot(hello)
        times, bad = [], []
        for n, o in enumerate(obs_list):
            o = copy.deepcopy(o)
            if mutate:
                mutate(o, n)
            t0 = time.perf_counter()
            a = bot.act(o)
            times.append((time.perf_counter() - t0) * 1000)
            bad += [(o["tick"], x) for x in problems(o, a)]
        self.assertEqual(bad, [])
        self.assertEqual(bot.errors, 0, "act() fell back to the degraded path")
        times.sort()
        self.assertLess(times[-1], 20.0, f"slowest decision {times[-1]:.2f} ms")
        return times

    def test_fuzz_5000_random_observations(self):
        rng = random.Random(1234)
        obs = [rand_obs(rng, i) for i in range(5000)]
        times = self.run_stream(obs, HELLO)
        print(f"\nfuzz: {len(obs)} obs, max {times[-1]:.3f} ms, p99 {times[int(len(times) * .99)]:.3f} ms", file=sys.stderr)

    def test_fuzz_with_one_bot_instance_is_deterministic(self):
        rng = random.Random(99)
        obs = [rand_obs(rng, i) for i in range(300)]
        runs = []
        for _ in range(2):
            bot = Bot(HELLO)
            runs.append([json.dumps(bot.act(copy.deepcopy(o)), sort_keys=True) for o in obs])
        self.assertEqual(runs[0], runs[1])

    @unittest.skipUnless(find_stream(), "kit stream.jsonl.gz not found (set MECHFIGHT_STREAM)")
    def test_replay_kit_stream(self):
        L = [json.loads(l) for l in gzip.open(find_stream(), "rt")]
        hello, obs = L[0], [m for m in L[1:] if m.get("type") == "obs"]
        self.assertEqual(len(obs), 300)
        times = self.run_stream(obs, hello)
        print(f"\nstream: {len(obs)} obs, max {times[-1]:.3f} ms, p99 {times[int(len(times) * .99)]:.3f} ms", file=sys.stderr)

    @unittest.skipUnless(find_stream(), "kit stream.jsonl.gz not found (set MECHFIGHT_STREAM)")
    def test_replay_kit_stream_with_each_slot_one_weapon(self):
        L = [json.loads(l) for l in gzip.open(find_stream(), "rt")]
        hello, obs = L[0], [m for m in L[1:] if m.get("type") == "obs"]
        for kind, w in (("laser", laser), ("lightning", lightning), ("mortar", mortar)):
            def put(o, n, w=w):
                o["self"]["weapons"][1] = w(charge=(n % 8) * 0.1) if w is laser else w()
            self.run_stream(obs, hello, put)


if __name__ == "__main__":
    unittest.main(verbosity=1)
