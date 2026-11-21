import unittest

import fight


class TestFight(unittest.TestCase):
    def test_plan_is_off(self):  # hide, sandwich and chase are gone: bot.py's own rules run
        f = fight.Fight()
        me = {"pos": [0, 0, 0], "hp": 300, "max_hp": 1400}
        mechs = [{"id": 5, "kind": "mech", "pos": [50, 0, 0], "hp": 200, "max_hp": 1400}]
        self.assertIsNone(f.plan(30.0, me, mechs, lambda a, b, *k: True))
        self.assertFalse(f.hiding)


if __name__ == "__main__":
    unittest.main()
