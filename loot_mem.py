"""Pickup memory for bot.py: obs.pickups lists only what we see right now, so a laser that leaves view while we
turn to fight is forgotten and the walk to it stops. This keeps every seen pickup for LOOT_MEM_S, and forgets
one once we stand close enough that we would see it if it were still there.
"""
import math

LOOT_MEM_S = 30.0         # remember a seen pickup this long (0 = off: obs pickups only)
GRAB_GIVE_UP_S = 25.0     # stop going for a pickup this long after we first went for it (roof, wall)
SWAP_LIGHTNING = 1        # a laser replaces a held lightning (lightning reaches 45 m; house hunters hold 95+)


class LootMemory:
    def __init__(self):
        self.seen = {}  # pickup id -> (pickup dict, t last seen)

    def update(self, obs, me, near_r):
        """Record this observation's pickups; drop old ones and ones gone from where we would see them."""
        t = obs.get("t", 0.0)
        now = set()
        for p in obs.get("pickups") or []:
            pid = p.get("id")
            pos = p.get("pos") or ()
            if pid is None or len(pos) < 3 or not all(math.isfinite(v) for v in pos[:3]):
                continue
            self.seen[pid] = (p, t)
            now.add(pid)
        x, y = me["pos"][0], me["pos"][1]
        for pid, (p, ts) in list(self.seen.items()):
            if pid in now:
                continue
            if t - ts > LOOT_MEM_S or t < ts or math.hypot(p["pos"][0] - x, p["pos"][1] - y) < near_r - 5:
                del self.seen[pid]

    def pickups(self):
        return [p for p, _ in self.seen.values()]
