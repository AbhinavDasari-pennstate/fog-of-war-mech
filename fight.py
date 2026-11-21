"""Fight tactics for bot.py. Only one rule is live: while this module is loaded, bot.py never flees in the
open under 45% HP (it keeps fighting and never walks at a mech within 120 m in the zone).

ponytail: hide-behind-cover, the sandwich rule and fight.py's own chase were all switched off (they cost kills
on the board and in simulation) and are deleted; chase lives in chase.py. Git history (0693c66) has them.
`Fight.plan(...)` keeps its interface and returns None (normal behaviour).
"""


class Fight:
    def __init__(self):
        self.hiding = False
        self.threats = {}  # bot.py snapshots this for its degraded-pass restore
        self.reason = ""

    def plan(self, t, me, mechs, clear, zone=None, urgent=False, damaged_recent=False):
        return None
