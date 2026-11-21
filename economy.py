"""Loot and rest: after a confirmed kill, walk to the victim's pile, then hold fire so regen can start."""
import math

LOOT_CAP_S = 6.0       # s: longest walk to a kill pile
LOOT_RADIUS = 70.0     # m: farthest kill spot we walk to
PILE_R = 10.0          # m: visible cubes this close to the kill spot belong to its pile
SPOT_SEEN_S = 5.0      # s: the victim must have been seen this recently, else we do not know where it fell
ARRIVE_M = 3.0         # m: at the spot (cube pickup reach is 3.5 m)
REST_MAX_S = 2.0       # s: longest fire hold after looting; the squad keeps sending hunters, so a long rest gives them free time
                       # (regen starts 5 s after the kill shot, so this hold only overlaps regen after a long loot walk)
REST_HP = 0.85         # rest ends at this HP fraction
GUARD_M = 40.0         # m: no looting while any seen mech (line of fire or not) is this close to the pile
NPC_CLEAR = 55.0       # m: no looting this close to a seen drone (drones shoot within 45 m, plus pickup reach)
ZONE_MARGIN = 30.0     # m: loot only this far inside the circle (bot.py turns urgent at cr - 25, plus pickup reach)
Z_REACH = 4.0          # m: cube pickup reach in height


class Economy:
    def __init__(self):
        self.last = {}      # mech id -> (last seen pos, time)
        self.phase = None   # None, "loot" or "rest"
        self.phase_t = 0.0
        self.spot = None    # where our last kill fell
        self.kill_t = -99.0  # time of that kill: hits taken before it are the victim's, not contact

    def update(self, obs):
        """Track where seen mechs stand; a `kill` event starts the loot phase at the victim's last position."""
        t = obs["t"]
        for ev in obs.get("events", []):
            if ev.get("type") == "kill":
                seen = self.last.get(ev.get("target"))
                self.spot = seen[0] if seen and t - seen[1] <= SPOT_SEEN_S else None
                self.phase, self.phase_t, self.kill_t = ("loot" if self.spot else "rest"), t, t
        for m in obs.get("mechs", []):
            if m.get("kind") == "mech":
                self.last[m["id"]] = (m["pos"], t)

    def plan(self, obs, contact, can_loot, npcs, ccx, ccy, cr, hp_frac, mechs=()):
        """("loot", (x, y)), ("rest", None) or (None, None). Contact (hit, or a mech with line of sight) ends both."""
        t = obs["t"]
        if self.phase is None:
            return None, None
        if contact:
            self.phase = None
            return None, None
        if self.phase == "loot":
            tgt = self._pile_target(obs, npcs, mechs, ccx, ccy, cr) if can_loot and t - self.phase_t <= LOOT_CAP_S else None
            if tgt is not None:
                return "loot", tgt
            self.phase, self.phase_t = "rest", t
        if hp_frac >= REST_HP or t - self.phase_t > REST_MAX_S:
            self.phase = None
            return None, None
        return "rest", None

    def _pile_target(self, obs, npcs, mechs, ccx, ccy, cr):
        x, y, z = obs["self"]["pos"]
        sp = self.spot
        if (math.dist(sp[:2], (x, y)) > LOOT_RADIUS or math.hypot(sp[0] - ccx, sp[1] - ccy) > cr - ZONE_MARGIN
                or any(math.dist(sp[:2], n["pos"][:2]) < NPC_CLEAR for n in npcs)
                or any(math.dist(sp[:2], m["pos"][:2]) < GUARD_M for m in mechs)):
            return None
        cubes = [c["pos"] for c in obs.get("cubes", [])
                 if math.dist(c["pos"][:2], sp[:2]) < PILE_R and abs(c["pos"][2] - z) < Z_REACH]
        if cubes:
            p = min(cubes, key=lambda p: math.dist(p[:2], (x, y)))
            return p[0], p[1]
        if math.dist(sp[:2], (x, y)) < ARRIVE_M or abs(sp[2] - z) >= Z_REACH:
            return None  # at the spot with nothing left in view, or the pile is on a roof
        return sp[0], sp[1]


if __name__ == "__main__":  # self-check
    def obs(t, mechs=(), events=(), cubes=(), pos=(0, 0, 0)):
        return {"t": t, "self": {"pos": list(pos)}, "mechs": list(mechs), "events": list(events),
                "cubes": [{"id": 5000 + i, "pos": list(p)} for i, p in enumerate(cubes)]}
    kill = [{"type": "kill", "target": 3}]
    e = Economy()
    e.update(obs(10.0, mechs=[{"id": 3, "kind": "mech", "pos": [50, 0, 0]}]))
    e.update(obs(10.1, events=kill))
    assert e.plan(obs(10.2), False, True, [], 0, 0, 500, 0.5) == ("loot", (50, 0))   # walk to the spot
    o = obs(11.0, cubes=[(48, 1, 0), (90, 0, 0)])
    assert e.plan(o, False, True, [], 0, 0, 500, 0.5) == ("loot", (48, 1))           # pile cube, not the far one
    assert e.plan(obs(16.5), False, True, [], 0, 0, 500, 0.5) == ("rest", None)      # 6 s cap: rest
    assert e.plan(obs(18.0), False, True, [], 0, 0, 500, 0.6) == ("rest", None)
    assert e.plan(obs(18.6), False, True, [], 0, 0, 500, 0.6) == (None, None)        # 2 s rest cap
    e.update(obs(30.0, mechs=[{"id": 4, "kind": "mech", "pos": [0, 40, 0]}]))
    e.update(obs(30.1, events=[{"type": "kill", "target": 4}]))
    assert e.plan(obs(30.2), True, True, [], 0, 0, 500, 0.5) == (None, None)         # contact aborts
    e.update(obs(31.0, mechs=[{"id": 5, "kind": "mech", "pos": [0, 40, 0]}]))
    e.update(obs(31.1, events=[{"type": "kill", "target": 5}]))
    assert e.plan(obs(31.2), False, True, [{"pos": [0, 80, 0]}], 0, 0, 500, 0.5) == ("rest", None)  # drone there
    assert e.plan(obs(32.0), False, True, [], 0, 0, 500, 0.9) == (None, None)        # healed: done
    e.update(obs(32.5, mechs=[{"id": 8, "kind": "mech", "pos": [0, 40, 0]}]))
    e.update(obs(32.6, events=[{"type": "kill", "target": 8}]))
    assert e.kill_t == 32.6
    assert e.plan(obs(32.7), False, True, [], 0, 0, 500, 0.5, mechs=[{"pos": [20, 40, 0]}]) == ("rest", None)  # guard
    e.update(obs(33.1, events=[{"type": "kill", "target": 9}]))                      # never seen: rest only
    assert e.plan(obs(33.2), False, True, [], 0, 0, 500, 0.5) == ("rest", None)
    e.update(obs(40.0, mechs=[{"id": 6, "kind": "mech", "pos": [0, 40, 0]}]))
    e.update(obs(40.1, events=[{"type": "kill", "target": 6}]))
    assert e.plan(obs(40.2), False, False, [], 0, 0, 500, 0.3) == ("rest", None)     # cannot loot (hurt): rest
    e.update(obs(50.0, mechs=[{"id": 7, "kind": "mech", "pos": [0, 60, 0]}]))
    e.update(obs(50.1, events=[{"type": "kill", "target": 7}]))
    assert e.plan(obs(50.2), False, True, [], 0, 0, 80, 0.5) == ("rest", None)       # spot near the zone edge
    print("economy ok")
