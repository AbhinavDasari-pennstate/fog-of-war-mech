"""Tests for bot.py's params.json loader. Run: python -B -m unittest test_params"""
import copy
import gzip
import json
import math
import os
import tempfile
import unittest

import bot as botmod
from test_bot import find_arena


def stream():
    arena = find_arena()
    p = arena and os.path.join(arena, "..", "check", "stream.jsonl.gz")
    if not p or not os.path.exists(p):
        return None
    with gzip.open(p, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def play(msgs):
    hello = copy.deepcopy(msgs[0])
    hello.setdefault("map", {})["dir"] = find_arena()
    bot = botmod.Bot(hello)
    return [bot.act(copy.deepcopy(m)) for m in msgs[1:] if m.get("type") == "obs"]


class TestParams(unittest.TestCase):
    def tearDown(self):
        botmod.apply_params({})

    def values(self):
        mods = botmod._param_modules()
        return {n: getattr(mods[m], a) for n, (m, a, _, _) in botmod.PARAMS.items() if mods[m] is not None}

    def test_defaults_are_todays_constants(self):
        botmod.apply_params({})
        self.assertEqual(botmod.DEFAULTS["retreat_hp"], 0.30)
        self.assertEqual(botmod.DEFAULTS["detour_ticks"], 25)
        self.assertEqual(botmod.DEFAULTS["chase_hp"], 0.80)
        self.assertEqual(self.values(), botmod.DEFAULTS)
        self.assertEqual(len(botmod.DEFAULTS), len(botmod.PARAMS))

    def test_valid_values_apply_and_bad_ones_keep_the_default(self):
        got = botmod.apply_params({"retreat_hp": 0.2, "chase_hp": 0.5, "detour_ticks": 30, "threat_m": "far",
                                   "hurt_hp": float("nan"), "dodge_angle": 5.0, "boost_min_energy": True,
                                   "rest_max_s": 2.5, "unknown": 1})
        self.assertEqual(sorted(got), ["chase_hp", "detour_ticks", "rest_max_s", "retreat_hp"])
        self.assertEqual((botmod.RETREAT_HP, botmod.chaselib.CHASE_HP, botmod.DETOUR_TICKS), (0.2, 0.5, 30))
        self.assertIsInstance(botmod.DETOUR_TICKS, int)
        self.assertEqual(botmod.THREAT_M, botmod.DEFAULTS["threat_m"])
        self.assertEqual(botmod.HURT_HP, botmod.DEFAULTS["hurt_hp"])
        self.assertEqual(botmod.DODGE_ANGLE, botmod.DEFAULTS["dodge_angle"])
        self.assertEqual(botmod.BOOST_MIN_ENERGY, botmod.DEFAULTS["boost_min_energy"])
        botmod.apply_params({"detour_ticks": 30.5})  # not a whole number for an int: default
        self.assertEqual(botmod.DETOUR_TICKS, 25)
        botmod.apply_params({})  # back to defaults everywhere
        self.assertEqual(self.values(), botmod.DEFAULTS)

    def test_a_bool_is_not_a_number_even_inside_the_bounds(self):
        self.assertEqual(botmod.apply_params({"chase_hp": True}), [])  # True == 1 is inside [0, 1.01]
        self.assertEqual(botmod.chaselib.CHASE_HP, 0.80)

    def test_detour_ticks_sets_the_detour_length(self):
        from test_bot import HELLO, make_obs, pickup
        botmod.apply_params({"detour_ticks": 9})
        bot = botmod.Bot(HELLO)
        for i in range(4):  # pushing at a wall: the detour starts on the 4th slow decision
            bot.act(make_obs(tick=i + 1, t=3 + i / 10, pickups=[pickup(1, "laser", (60, 0, 0))], vel=(0, 0, 0)))
        self.assertEqual(bot.detour, 8)  # started at 9, one decision used

    def test_retreat_must_stay_below_return(self):
        self.assertEqual(botmod.apply_params({"retreat_hp": 0.5, "return_hp": 0.45}), [])
        self.assertEqual((botmod.RETREAT_HP, botmod.RETURN_HP), (0.30, 0.80))

    def test_file_missing_bad_or_good(self):
        d = tempfile.mkdtemp()
        self.addCleanup(lambda: [os.remove(os.path.join(d, f)) for f in os.listdir(d)] and os.rmdir(d))
        self.assertEqual(botmod.load_params(os.path.join(d, "none.json")), [])
        p = os.path.join(d, "params.json")
        with open(p, "w") as f:
            f.write("{not json")
        self.assertEqual(botmod.load_params(p), [])
        with open(p, "w") as f:
            json.dump([1, 2], f)
        self.assertEqual(botmod.load_params(p), [])
        with open(p, "w") as f:
            json.dump({"retreat_hp": 0.25}, f)
        self.assertEqual(botmod.load_params(p), ["retreat_hp"])
        self.assertEqual(botmod.RETREAT_HP, 0.25)

    def test_committed_params_json_holds_the_defaults(self):
        botmod.apply_params({})
        with open(os.path.join(os.path.dirname(os.path.abspath(botmod.__file__)), "params.json")) as f:
            self.assertEqual(json.load(f), botmod.DEFAULTS)

    @unittest.skipUnless(stream(), "kit check stream not found (set MECHFIGHT_ARENA)")
    def test_same_actions_on_the_check_stream_with_no_file_and_with_the_defaults_file(self):
        msgs = stream()
        botmod.apply_params({})
        base = play(msgs)
        botmod.load_params(os.path.join(os.path.dirname(os.path.abspath(botmod.__file__)), "params.json"))
        self.assertEqual(play(msgs), base)
        botmod.apply_params({"retreat_hp": 0.55, "dodge_angle": 0.0, "hurt_hp": 0.0, "detour_ticks": 5})
        self.assertNotEqual(play(msgs), base)  # the numbers reach the decisions


if __name__ == "__main__":
    unittest.main()
