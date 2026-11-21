"""Survival-first mechfight/1 bot: grab a laser early, stay quiet, face threats, shoot back, flee when hurt."""
import json
import math
import os
import traceback
import mechfight as mf

try:
    import aim as aimlib
    import ray as raylib
except Exception:  # helper modules missing or broken: v3 aim, fire and line-of-fire logic only
    aimlib = raylib = None
try:
    import fight as fightlib
except Exception:  # fight tactics missing: v3 flee
    fightlib = None
try:
    import chase as chaselib
except Exception:  # chase module missing or broken: no chasing
    chaselib = None
try:
    import economy
except Exception:  # economy module missing or broken: no looting
    economy = None

TAU = 2 * math.pi
# upgrade preference order: the first id present in an offer is picked
UPGRADE_PRIO = ["armor", "plating", "nanites", "sensors", "servos", "gun_damage", "gun_rate"]
GRAB_KINDS = ("laser", "lightning")  # preference order; mortar is never picked up
FREE_N = 20  # decisions of free movement that end a detour early
ENGAGE_S = 5.0  # a mech that hit us or that we hit stays a target this long (protocol memory_s)
STATE = ("stuck", "detour", "detour_sign", "detour_end", "free_n", "edge_sign", "edge_until", "dodge_sign",
         "dodge_t", "dodge_flip_t", "full_t", "lt_t", "lt_aim", "threat_yaw", "threat_t", "dmg_t")
SOUND_KINDS = ("gun", "laser", "laser_charge", "lightning", "mortar")


def ang_diff(a, b):
    return (a - b + math.pi) % TAU - math.pi


def unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)


def inside(x, y, poly):
    res = False
    j = len(poly) - 1
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[j]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            res = not res
        j = i
    return res


def load_map(hello):
    """Solid map prisms (bbox, base, top, ring, holes) for line-of-fire checks and the playable bounds;
    ([], None) if the map is unreadable."""
    out = []
    try:
        path = os.path.join(hello["map"]["dir"], "v2", "world.json")
        with open(path) as f:
            world = json.load(f)
        for p in world["prisms"]:
            r = p["ring"]
            xs, ys = r[0::2], r[1::2]
            holes = [list(zip(h[0::2], h[1::2])) for h in (p.get("holes") or [])]
            out.append((min(xs), min(ys), max(xs), max(ys), p["base"], p["top"], list(zip(xs, ys)), holes))
    except Exception:
        return [], None
    bounds = None
    try:
        b = world["bounds"]
        bounds = (b["minX"] + 2.5, b["minY"] + 2.5, b["maxX"] - 2.5, b["maxY"] - 2.5)
    except (KeyError, TypeError):
        pass
    return out, bounds


class Bot:
    name = "fog-of-war"

    def __init__(self, hello: mf.Hello):
        c = hello["config"]
        self.reach = c.get("loot", {}).get("pickup_radius", 4)
        self.g = c.get("weapons", {}).get("mortar", {}).get("gravity", 22.0)
        v = c.get("vision", {})
        self.vis_range = v.get("range", 200)
        self.vis_near = v.get("near_range", 45)
        self.vis_fov = math.radians(v.get("fov_deg", 120))
        self.prisms, self.bounds = load_map(hello)
        self.aim_ok = False
        if aimlib is not None:
            try:
                aimlib.init(hello)  # also loads the map for the shared ray test
                self.aim_ok = True
            except Exception:
                mf.log(f"aim.init failed: {traceback.format_exc()}")
        ray_n = 0
        try:
            ray_n = len(raylib._default.prisms) if self.aim_ok else 0
        except Exception:
            pass
        mf.log(f"map: prisms={len(self.prisms)} dir={hello.get('map', {}).get('dir')} ray={ray_n}")
        self.aim_errors = 0
        self.hit_by = {}        # attacker id (None: unknown) -> time of its last hit on us
        self.hit_on = {}        # target id -> time of our last hit on it ("hit" events)
        self.stuck = 0          # decisions in a row we wanted to move but barely did
        self.detour = 0         # decisions left on a sideways detour
        self.detour_sign = 1
        self.detour_end = -99.0
        self.free_n = 0
        self.threat_yaw = None  # bearing of the last hit (or weapon sound when not hit)
        self.threat_t = -99.0
        self.dmg_t = -99.0      # last weapon damage taken
        self.skip = set()       # pickup ids not worth chasing
        self.fails = {}
        self.chase = {}         # pickup id -> time we first went for it
        self.full_t = None      # time the laser charge first reached full
        self.dodge_sign = 1
        self.edge_sign = 1
        self.lite = False       # degraded pass: skip the map-based extras
        self.dodge_flip_t = -99.0
        self.lt_t = -99.0       # last time a laser target was in view
        self.lt_aim = None      # the aim we had on it
        self._los_cache = {}    # mech id -> line of fire clear, per decision
        self.clear_t = {}       # mech id -> last time we had a clear line to it (target hysteresis)
        self.errors = 0
        self.edge_until = -99.0
        self.dodge_t = -99.0
        self.fight = fightlib.Fight() if fightlib is not None else None
        self.fight_errors = 0
        self.chaser = chaselib.Chase() if chaselib is not None else None
        self.chase_errors = 0
        self.eco = economy.Economy() if economy is not None else None

    def pick_upgrade(self, offer):
        ids = [ch["id"] for ch in offer["choices"]]
        for key in UPGRADE_PRIO:
            if key in ids:
                return ids.index(key)
        return 0

    def lead_aim(self, me, tgt, speed):
        x, y, z = me["pos"]
        mz = z + me["stats"].get("muzzle_height", 7)
        tx, ty, tz = tgt["pos"]
        vx, vy, _ = tgt.get("vel", [0, 0, 0])
        tz += 4.5 if tgt["kind"] == "mech" else 1.75
        t, px, py, d = 0.0, tx, ty, 1.0
        for _ in range(3):  # fixed-point on flight time
            px, py = tx + vx * t, ty + vy * t
            d = math.dist((px, py, tz), (x, y, mz))
            t = d / max(speed, 1.0)
        return [math.atan2(py - y, px - x), math.atan2(tz - mz, math.hypot(px - x, py - y))], d

    def direct_aim(self, me, tgt):
        """Hitscan aim: no lead, at the target's centre of mass."""
        x, y, z = me["pos"]
        st = me["stats"]
        mz = z + st.get("muzzle_height", 7)
        tx, ty, tz = tgt["pos"]
        tz += 4.5 if tgt["kind"] == "mech" else 1.75
        pitch = math.atan2(tz - mz, math.hypot(tx - x, ty - y))
        pitch = min(max(pitch, st.get("min_pitch", -0.7)), st.get("max_pitch", 1.3))
        return [math.atan2(ty - y, tx - x), pitch], math.dist((tx, ty, tz), (x, y, mz))

    def in_view(self, me, m):
        """Would we see this track right now (not only remember it)? Geometry only."""
        st = me["stats"]
        dx, dy = m["pos"][0] - me["pos"][0], m["pos"][1] - me["pos"][1]
        d = math.hypot(dx, dy)
        if d <= self.vis_near + st.get("near_range_bonus", 0):
            return True
        if d > self.vis_range + st.get("vision_bonus", 0):
            return False
        return abs(ang_diff(me["aim"][0], math.atan2(dy, dx))) <= self.vis_fov / 2 - 0.05

    def los_clear(self, me, tgt):
        """Line of fire to a target, computed once per decision per mech id."""
        key = tgt.get("id")
        if key is None:
            return self._los_raw(me, tgt)
        if key not in self._los_cache:
            self._los_cache[key] = self._los_raw(me, tgt)
        return self._los_cache[key]

    def _los_raw(self, me, tgt):
        """False when a building, stadium, bridge or the ground is in the way (the shared exact ray test)."""
        if self.lite:
            return True
        x, y, z = me["pos"]
        mz = z + me["stats"].get("muzzle_height", 7)
        tx, ty, tz = tgt["pos"]
        tz += 4.5 if tgt["kind"] == "mech" else 1.75
        if raylib is None:
            return self._los_prisms(x, y, mz, tx, ty, tz)
        try:
            return bool(raylib.clear((x, y, mz), (tx, ty, tz)))
        except Exception:
            return True

    def _los_prisms(self, x, y, mz, tx, ty, tz):
        """Fallback when ray.py is missing: sample the segment every 2 m against the loaded prisms."""
        if not self.prisms:
            return True
        lox, hix, loy, hiy = min(x, tx), max(x, tx), min(y, ty), max(y, ty)
        loz, hiz = min(mz, tz), max(mz, tz)
        cand = [p for p in self.prisms if p[0] <= hix and p[2] >= lox and p[1] <= hiy and p[3] >= loy
                and p[5] >= loz and p[4] <= hiz]
        n = max(2, min(int(math.hypot(tx - x, ty - y) / 2.0), 120))
        for i in range(1, n):
            f = i / n
            px, py, pz = x + (tx - x) * f, y + (ty - y) * f, mz + (tz - mz) * f
            for p in cand:
                if (p[4] <= pz <= p[5] and p[0] <= px <= p[2] and p[1] <= py <= p[3]
                        and inside(px, py, p[6]) and not any(inside(px, py, h) for h in p[7])):
                    return False
        return True

    def _seen(self, me, m, t):
        """A mech v3 may shoot at: in view now, or it (or an unknown attacker) hit us in the last 3 s."""
        return (self.in_view(me, m) or t - self.hit_by.get(m.get("id"), -99.0) < 3
                or t - self.hit_by.get(None, -99.0) < 3)

    def _fighting(self, m, t):
        """It hit us, or we hit it, in the last 5 s (obs.mechs keeps listing it that long out of view)."""
        tid = m.get("id")
        return min(t - self.hit_by.get(tid, -99.0), t - self.hit_on.get(tid, -99.0)) < ENGAGE_S

    def _follow_through(self, me, m, t, rng):
        """A mech we are fighting, seen or only remembered, within rng and with a clear line from the muzzle."""
        if self.lite or m is None or m.get("kind") != "mech" or not self._fighting(m, t):
            return False  # lite pass: no ray test, so no shots at mechs we only remember
        return math.dist(m["pos"][:3], me["pos"][:3]) <= rng and self.los_clear(me, m)

    def _aim_plan(self, obs, hitters, seen_mechs=None, mechs=None, hide=False):
        """aim.decide for this observation, or None when it is unavailable or failed (v3 logic then runs).

        The aim module scores targets (wounded, near, threatening) without knowing what v3 may shoot, so it
        only gets the mechs in view or hitting us; when none qualifies it gets all of them (it still turns
        to look, and the gate keeps the guns quiet). Drones are targets only when they hit us."""
        if self.lite or not self.aim_ok:
            return None
        try:
            if mechs is None:
                mechs = [m for m in obs["mechs"] if m.get("kind") == "mech"]
            cand = seen_mechs or mechs
            # shoot what we can hit: a clear line wins over a better-scored mech behind a wall
            # ponytail: nearest 4 only (each ray costs time); none clear = old choice (turn to the best)
            t = obs["t"]
            shootable = []
            for m in cand[:4]:
                if self.los_clear(obs["self"], m):
                    self.clear_t[m.get("id")] = t
                    shootable.append(m)
                elif t - self.clear_t.get(m.get("id"), -99.0) < 0.3:  # just went behind a wall edge: no flicker
                    shootable.append(m)
            keep = {id(m) for m in (shootable or cand)}
            drones = {id(m) for m in hitters}
            use = dict(obs, mechs=[m for m in obs["mechs"] if (
                id(m) in keep if m.get("kind") == "mech" else (not hitters or id(m) in drones))])
            npcs_ok = bool(hitters)
            a, f, tg = aimlib.decide(use, los=raylib.clear, hide=hide, deadline=None, include_npcs=npcs_ok)
            plan = aimlib.last_plan
            if "error" in plan:
                raise RuntimeError(plan["error"])
            return {"aim": a, "fire": list(f), "target": tg, "reload": bool(plan.get("reload"))}
        except Exception:
            self.aim_errors += 1
            if self.aim_errors <= 3:
                mf.log(f"aim.decide failed, v3 logic this tick: {traceback.format_exc()}")
            return None

    def _gate_fire(self, raw, target, me, ws, mechs, flee, t):
        """v3's engagement rules on top of the aim module's per-slot fire flags."""
        fire = [False, False]
        if target is None:  # a charging laser keeps its trigger for 0.5 s after the target left view
            for i, w in enumerate(ws):
                if (w and w.get("kind") == "laser" and (w.get("charge") or 0) > 0 and not flee
                        and t - self.lt_t <= 0.5):
                    fire[i] = bool(raw[i])
            return fire
        x, y, z = me["pos"]
        d = math.dist(target["pos"][:3], (x, y, z))
        in_view = self.in_view(me, target)
        seen = self._seen(me, target, t) if target.get("kind") == "mech" else in_view or t - self.dmg_t < 3
        laser_held = any(w and w.get("kind") == "laser" for w in ws)
        # in a fight when hit lately, holding a laser, or any seen mech is close or aiming at us
        engaged = t - self.dmg_t < 3 or laser_held or d < 120 or any(
            math.dist(m["pos"][:3], (x, y, z)) < 120
            or (m.get("aim") and abs(ang_diff(m["aim"][0], math.atan2(y - m["pos"][1], x - m["pos"][0]))) < 0.12)
            for m in mechs)
        for i, w in enumerate(ws):
            if not w or not raw[i]:
                continue
            kind = w.get("kind")
            if kind == "gun":
                fire[i] = seen and engaged
            elif kind == "lightning":
                fire[i] = seen
            elif kind == "laser":
                if flee:
                    fire[i] = False
                elif (w.get("charge") or 0) > 0:
                    fire[i] = True  # already charging: the aim module holds or releases
                else:  # a new charge at a mech in view within 200 m, or one we are fighting within range
                    rng = (w.get("stats") or {}).get("range", 280)
                    fire[i] = ((in_view and d < 200 and target.get("kind") == "mech")
                               or self._follow_through(me, target, t, rng))
        return fire

    def _v3_fire(self, me, t, target, mechs, flee, slot, gun, las, light, gun_i, las_i, light_i):
        """v3 aim and fire decision: used when the aim module is unavailable or fails. Returns (aim or None, fire)."""
        x, y, z = me["pos"]
        fire = [False, False]
        aim_cmd = None
        # laser: at a mech in view within ~200 m, or one we are fighting within range; clear line, not fleeing
        lt, aim_l, dl = None, None, 0.0
        if las and not flee:
            lrange = las.get("stats", {}).get("range", 280)
            for m in mechs[:4]:  # nearest few only: each line of fire costs time
                a_l, d_l = self.direct_aim(me, m)
                if ((d_l < min(lrange, 200) and self.in_view(me, m) and self.los_clear(me, m))
                        or self._follow_through(me, m, t, lrange)):
                    lt, aim_l, dl = m, a_l, d_l
                    break
        charge = las.get("charge", 0) if las else 0
        full = las.get("stats", {}).get("charge", 0.75) if las else 0.75
        if charge >= full - 0.01:  # the reported charge is capped at full: time the hold ourselves
            if self.full_t is None:
                self.full_t = t
        else:
            self.full_t = None
        laser_aims = grace = False
        overdue = self.full_t is not None and t - self.full_t > 0.5
        if lt is not None:
            self.lt_t, self.lt_aim = t, aim_l
            laser_aims = charge > 0 or las.get("cooldown", 0) < 0.3
            tol = min(max(math.atan2(2.0, dl), 0.01), 0.05)
            on = abs(ang_diff(me["aim"][0], aim_l[0])) < tol and abs(me["aim"][1] - aim_l[1]) < 0.08
            # hold to charge; let go when full and on target, or after 0.5 s at full
            fire[las_i] = not (charge >= full - 0.01 and (on or overdue))
        elif las and not flee and charge > 0 and self.lt_aim is not None and t - self.lt_t <= 0.5:
            grace = True  # target just left view: keep the charge and the aim for a moment
            fire[las_i] = not (charge >= full - 0.01 and overdue)

        if target is not None:
            gst = gun.get("stats", {}) if gun else {}
            aim, d = self.lead_aim(me, target, gst.get("speed", 252) or 252)
            err = abs(ang_diff(me["aim"][0], aim[0]))
            aim_cmd = aim_l if laser_aims else aim
            # shoot only what we could see now, through a clear line
            seen = self.in_view(me, target) or t - self.dmg_t < 3  # a track that is hitting us counts as seen
            if seen and self.los_clear(me, target):
                # commit to a fight: we were hit, it is close, we hold a laser, or a visible mech is aiming at us
                engaged = t - self.dmg_t < 3 or d < 120 or las is not None or any(
                    abs(ang_diff(m["aim"][0], math.atan2(y - m["pos"][1], x - m["pos"][0]))) < 0.12
                    for m in mechs if m.get("aim"))
                tol = min(0.06, max(math.atan2(2.0, d), 0.008))  # inside the target's angular size
                if gun and engaged and d < gst.get("range", 230) and err < tol and gun.get("ammo", 1) > 0:
                    fire[gun_i] = True
                if light and d < light.get("stats", {}).get("range", 45) and err < 0.4 and me["energy"] > 10:
                    fire[light_i] = True
        if grace:
            aim_cmd = list(self.lt_aim)
        return aim_cmd, fire

    def _idle_aim(self, me, t, hit_recent, empty, grab, mx, my, n):
        """Aim when there is nothing to shoot at: face the hit, scan early, face the noise or the heading."""
        facing = self.threat_yaw is not None and t - self.threat_t < 4
        if hit_recent and facing:
            return [self.threat_yaw, 0.0]   # face the attacker: rear hits do x1.5
        if t < 20 and empty and grab is None:
            return [me["aim"][0] + 1.0, 0.0]  # early: scan 360 for weapons (aim drives vision)
        if facing:
            return [self.threat_yaw, 0.0]   # face the noise
        if n > 0:
            return [math.atan2(my, mx), 0.0]
        return [me["aim"][0] + 0.5, 0.0]  # idle: scan around

    def free_run(self, x, y, z, ang, maxd=60.0):
        """Metres we can walk along a heading before a solid prism stands in the way (maxd if none)."""
        if self.lite:
            return maxd
        dx, dy = math.cos(ang), math.sin(ang)
        ex, ey = x + dx * maxd, y + dy * maxd
        cand = [p for p in self.prisms if p[0] <= max(x, ex) and p[2] >= min(x, ex)
                and p[1] <= max(y, ey) and p[3] >= min(y, ey)]
        d = 3.0
        while cand and d <= maxd:
            px, py = x + dx * d, y + dy * d
            for p in cand:
                if (p[0] <= px <= p[2] and p[1] <= py <= p[3] and inside(px, py, p[6])
                        and not any(inside(px, py, h) for h in p[7])
                        and any(p[4] <= zz <= p[5] for zz in (z + 1.3, z + 8.0))):
                    return d
            d += 3.0
        return maxd

    def pick_grab(self, obs, me, ccx, ccy, cr, t, kinds):
        """Nearest visible pickup of the first wanted kind, inside the current zone circle."""
        x, y, z = me["pos"]
        best = None
        for kind in kinds:
            cand = [p for p in obs.get("pickups", [])
                    if p["kind"] == kind and p["id"] not in self.skip and abs(p["pos"][2] - z) < 10
                    and math.hypot(p["pos"][0] - ccx, p["pos"][1] - ccy) < cr - 15]
            if cand:
                best = min(cand, key=lambda p: math.dist(p["pos"][:2], (x, y)))
                break
        if best is None:
            return None
        first = self.chase.setdefault(best["id"], t)  # first time we went for it; kept if it leaves view
        if t - first > 15:  # cannot reach it (roof, wall): stop trying
            self.skip.add(best["id"])
            return None
        return best

    def pick_drop(self, obs, me, mechs, ccx, ccy, cr):
        """A supply drop we can reach by the time it lands, inside the zone, with nobody near it or us."""
        if self.lite or obs["t"] - self.dmg_t < 3:
            return None
        x, y, _ = me["pos"]
        if any(math.dist(m["pos"][:2], (x, y)) < 150 for m in mechs):
            return None
        best, bd = None, 150.0
        for d in obs.get("drops", []):
            if d.get("holder") not in (None, me.get("id")):
                continue
            dist = math.hypot(d["pos"][0] - x, d["pos"][1] - y)
            if dist > bd or d.get("lands_in", 0) > dist / 11.7 + 8:
                continue
            if math.hypot(d["pos"][0] - ccx, d["pos"][1] - ccy) > cr - 10:
                continue
            if any(math.dist(m["pos"][:2], d["pos"][:2]) < 60 for m in mechs):
                continue
            best, bd = d, dist
        return best

    def act(self, obs: mf.Observation) -> mf.Action:
        """One failing part must not cost the whole decision: retry without the map extras, then stand safe."""
        self.lite = False
        snap = {k: getattr(self, k) for k in STATE}
        sets = (set(self.skip), dict(self.fails), dict(self.chase), dict(self.hit_by), dict(self.hit_on))
        fsnap = dict(vars(self.fight), threats=dict(self.fight.threats)) if self.fight is not None else None
        csnap = self.chaser.state() if self.chaser is not None else None
        try:
            return self._decide(obs)
        except Exception:
            self.errors += 1
            if self.errors <= 3:
                mf.log(f"decision degraded at tick {obs.get('tick')}: {traceback.format_exc()}")
        for k, v in snap.items():  # the failed pass must not leave its counters behind
            setattr(self, k, v)
        self.skip, self.fails, self.chase, self.hit_by, self.hit_on = sets
        if fsnap is not None:
            vars(self.fight).update(fsnap)
        if csnap is not None:
            self.chaser.restore(csnap)
        self.lite = True
        try:
            return self._decide(obs)
        except Exception:
            if self.errors <= 3:
                mf.log(f"fallback at tick {obs.get('tick')}: {traceback.format_exc()}")
        finally:
            self.lite = False
        return self._fallback(obs)

    def _fight_on(self):
        """fight.py runs this pass (else v3 flee): module loaded, ray test present, not degraded, never failed."""
        return self.fight is not None and not self.lite and raylib is not None and self.fight_errors == 0

    def _fight_plan(self, t, me, mechs, ccx, ccy, cr, urgent):
        """fight.py's hide decision, or None (v3 behaviour) when it is unavailable, degraded or failed."""
        if not self._fight_on():
            return None
        try:
            return self.fight.plan(t, me, mechs, raylib.clear, (ccx, ccy, cr - 20), urgent, t - self.dmg_t < 1.0)
        except Exception:
            self.fight_errors += 1  # broken once: v3 flee for the rest of the match
            mf.log(f"fight.plan failed, v3 flee from now on: {traceback.format_exc()}")
            return None

    def _chase_on(self):
        """chase.py runs this pass: module loaded, ray test present, not degraded, never failed."""
        return self.chaser is not None and not self.lite and raylib is not None and self.chase_errors == 0

    def _chase_plan(self, t, me, mechs, zone, active):
        """chase.py's plan for a wounded hunter backing off, or None (not active, no chase, off or failed)."""
        if not self._chase_on():
            return None
        try:
            if not active:
                self.chaser.observe(t, mechs)
                return None
            return self.chaser.plan(t, me, mechs, self.hit_on, self.hit_by, zone, raylib.clear)
        except Exception:
            self.chase_errors += 1  # broken once: no chasing for the rest of the match
            mf.log(f"chase.plan failed, chasing off from now on: {traceback.format_exc()}")
            return None

    def _fallback(self, obs):
        """Last resort: walk toward the next zone centre, no shooting."""
        act = {"tick": obs.get("tick", 0), "move": [0.0, 0.0], "boost": False, "jump": False, "fire": [False, False]}
        try:
            x, y = obs["self"]["pos"][:2]
            cx, cy = obs["zone"]["next_center"]
            ux, uy = unit(cx - x, cy - y)
            act["move"] = [ux, uy]
        except Exception:
            pass
        return act

    def _decide(self, obs: mf.Observation) -> mf.Action:
        self._los_cache = {}
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        act: mf.Action = {"tick": obs["tick"]}
        if obs.get("offer"):
            act["pick"] = self.pick_upgrade(obs["offer"])

        for ev in obs.get("events", []):
            kind = ev.get("type")
            if kind == "damaged" and ev.get("weapon") != "zone":
                self.dmg_t = t
                self.hit_by[ev.get("by")] = t
                if ev.get("bearing") is not None:
                    self.threat_yaw, self.threat_t = ev["bearing"], t
            elif kind == "hit" and ev.get("target") is not None:
                self.hit_on[ev["target"]] = t
            elif kind == "kill" and self.chaser is not None:
                self.chaser.forget(ev.get("target"))
            elif kind == "take_failed":
                pid = ev.get("id")
                self.fails[pid] = self.fails.get(pid, 0) + 1
                if ev.get("reason") == "gone" or self.fails[pid] >= 3:
                    self.skip.add(pid)
        if self.eco is not None and not self.lite:
            try:
                self.eco.update(obs)
            except Exception:
                self.eco = None
                mf.log(f"economy failed, looting off: {traceback.format_exc()}")
        hit_recent = t - self.dmg_t < 4  # a recent hit bearing wins over what we hear
        if not hit_recent:
            heard = [s for s in obs.get("sounds", []) if s.get("kind") in SOUND_KINDS and s.get("dist") != "far"]
            if heard:
                s = min(heard, key=lambda s: (s["dist"] != "near", s.get("age", 0)))
                self.threat_yaw, self.threat_t = s["bearing"], t - 2

        hp_frac = me["hp"] / max(me["max_hp"], 1)
        mechs = sorted((m for m in obs["mechs"] if m["kind"] == "mech"), key=lambda m: math.dist(m["pos"][:2], (x, y)))
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]
        ws = (list(me["weapons"]) + [None, None])[:2]
        slot = {}  # weapon kind -> (slot index, weapon): fire by kind, never by slot position
        for i, w in enumerate(ws):
            if w and w.get("kind") not in slot:
                slot[w.get("kind")] = (i, w)
        gun_i, gun = slot.get("gun", (0, None))
        las_i, las = slot.get("laser", (1, None))
        light_i, light = slot.get("lightning", (1, None))
        mort_i = slot["mortar"][0] if "mortar" in slot else None
        empty = [i for i in (0, 1) if ws[i] is None]
        grab_kinds = [k for k in GRAB_KINDS if k not in slot]  # never take a kind we already hold
        want_weapon = bool(grab_kinds) and (bool(empty) or mort_i is not None)

        # ---- where to stand: the quiet outer part of the circle the zone is closing to ----
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        safe_r = max(zone["next_radius"] * 0.8 - 15, 6.0)
        dz = math.hypot(cx - x, cy - y)
        s0, s1 = zone["shrink_start"], zone["shrink_end"]
        f = min(max((t - s0) / (s1 - s0), 0.0), 1.0) if s1 > s0 else 0.0
        ccx = zone["center"][0] + (cx - zone["center"][0]) * f
        ccy = zone["center"][1] + (cy - zone["center"][1]) * f
        cr = zone["radius"] + (zone["next_radius"] - zone["radius"]) * f
        in_zone = bool(me["in_zone"])
        urgent = (not in_zone) or math.hypot(ccx - x, ccy - y) > cr - 25
        mx = my = 0.0
        if dz > safe_r:
            mx, my = unit(cx - x, cy - y)

        # ---- goal: a laser (else lightning), else a supply drop; beats zone positioning unless outside the zone ----
        grab = self.pick_grab(obs, me, ccx, ccy, cr, t, grab_kinds) if want_weapon else None
        drop = None if grab is not None else self.pick_drop(obs, me, mechs, ccx, ccy, cr)
        # ---- after a kill: walk to its pile (economy.py), then rest without firing so regen starts ----
        loot, rest = None, False
        if self.eco is not None and not self.lite and self.eco.phase is not None:
            try:  # contact: hit since the kill (the victim's own hits do not count), or a mech in view with a clear line
                seen_now = [m for m in mechs if self.in_view(me, m)][:4]  # nearest few: each line of fire costs time
                contact = (self.dmg_t > self.eco.kill_t and t - self.dmg_t < 3) or any(self.los_clear(me, m) for m in seen_now)
                can_loot = grab is None and drop is None and not urgent and hp_frac >= 0.45
                mode, loot = self.eco.plan(obs, contact, can_loot, npcs, ccx, ccy, cr, hp_frac, mechs)
                rest = mode == "rest" and in_zone  # no regen outside the zone
            except Exception:
                loot, rest = None, False
                self.eco = None
                mf.log(f"economy failed, looting off: {traceback.format_exc()}")
        goal, dgoal = None, 0.0
        if loot is not None:
            goal, dgoal = (loot[0], loot[1], z), math.hypot(loot[0] - x, loot[1] - y)
        if grab is not None:
            goal, dgoal = grab["pos"], math.dist(grab["pos"][:2], (x, y))
        elif drop is not None:
            goal = drop["pos"]
            dgoal = math.dist(goal[:2], (x, y))
        if goal is not None and in_zone:
            if drop is not None and dgoal < drop.get("radius", 8) - 2:
                mx = my = 0.0  # inside the ring: hold still while it captures
            else:
                mx, my = unit(goal[0] - x, goal[1] - y)

        # ---- danger: hide behind cover to regen (fight.py), else v3: flee visible mechs when hurt; keep off drones ----
        fp = self._fight_plan(t, me, mechs, ccx, ccy, cr, urgent)
        hiding = fp is not None and fp["hide"]
        if fp is not None and (hiding or in_zone):  # hide or chase: beats loot and zone positioning
            grab = drop = goal = None
            mx, my = fp["move"]
        # ---- chase a wounded hunter backing off to regen (chase.py): beats loot and zone positioning ----
        capturing = drop is not None and dgoal < drop.get("radius", 8) - 2
        cp = self._chase_plan(t, me, mechs, zone, in_zone and not urgent and not capturing and not hiding)
        if cp is not None:
            grab = drop = goal = None
            mx, my = cp["move"]

        flee = hp_frac < 0.45 and not self._fight_on()
        if hp_frac < 0.45 and not flee and in_zone and not (fp and fp["reason"] == "chase"):
            near = [unit(m["pos"][0] - x, m["pos"][1] - y) for m in mechs[:3] if math.dist(m["pos"][:2], (x, y)) < 120]
            for _ in range(4):  # hurt and fighting: still never walk at a near mech (no flee push); repeat for 2+
                for tx_, ty_ in near:
                    comp = mx * tx_ + my * ty_
                    if comp > 0:
                        mx, my = mx - comp * tx_, my - comp * ty_
            if any(mx * tx_ + my * ty_ > 1e-6 for tx_, ty_ in near):
                mx = my = 0.0
        if flee:
            fx = fy = 0.0
            for m in mechs[:3]:
                d = math.dist(m["pos"][:2], (x, y))
                if d < 120:
                    tx_, ty_ = unit(m["pos"][0] - x, m["pos"][1] - y)  # toward the enemy
                    comp = mx * tx_ + my * ty_
                    if in_zone and comp > 0:
                        mx, my = mx - comp * tx_, my - comp * ty_
                    w = 1.5 * (120 - d) / 120
                    fx, fy = fx - tx_ * w, fy - ty_ * w
            mx, my = mx + fx, my + fy
        for npc in npcs:
            d = math.dist(npc["pos"][:2], (x, y))
            if d < 55:
                ux, uy = unit(x - npc["pos"][0], y - npc["pos"][1])
                mx, my = mx + ux * (55 - d) / 55, my + uy * (55 - d) / 55

        # ---- dodge: a laser charging at us (sidestep the line) or a mortar shell about to land near us ----
        dodging = dodge_boost = False
        for m in mechs:
            if not self.lite and not urgent and not flee and m.get("charging") and m.get("aim"):
                bx, by = x - m["pos"][0], y - m["pos"][1]
                sd = math.hypot(bx, by)
                if sd < 300 and abs(ang_diff(m["aim"][0], math.atan2(by, bx))) < 0.3:
                    ux, uy = unit(bx, by)
                    px, py = -uy, ux
                    if t - self.dodge_t > 1.5:  # a new charge: pick a side
                        self.dodge_sign = 1 if (px * mx + py * my) >= 0 else -1
                        self.dodge_flip_t = t
                    elif t - self.dodge_flip_t >= 0.5:  # then alternate so a re-aiming shooter cannot lead us
                        self.dodge_sign, self.dodge_flip_t = -self.dodge_sign, t
                    self.dodge_t = t
                    mx, my = px * self.dodge_sign, py * self.dodge_sign
                    dodging, dodge_boost = True, sd < 60
                    break
        for p in obs.get("projectiles", []):
            if self.lite or p.get("kind") != "mortar" or not in_zone:
                continue
            vz, z0 = p["vel"][2], p["pos"][2]
            disc = vz * vz + 2 * self.g * (z0 - z)
            if disc < 0:
                continue
            tl = (vz + math.sqrt(disc)) / self.g
            if not 0 <= tl <= 3:
                continue
            lx, ly = p["pos"][0] + p["vel"][0] * tl, p["pos"][1] + p["vel"][1] * tl
            dd = math.hypot(lx - x, ly - y)
            if dd < 12:
                if dd < 1:
                    ux, uy = unit(-p["vel"][1], p["vel"][0])  # sideways to the shell's path
                else:
                    ux, uy = unit(x - lx, y - ly)
                mx, my = ux, uy
                dodging = dodge_boost = True
                break

        # ---- free cubes nearby when calm ----
        calm = not mechs and t - self.dmg_t > 3 and not hiding
        if calm and not urgent and mx == 0 and my == 0:
            items = [c["pos"] for c in obs["cubes"]]
            items = [p for p in items if math.dist(p[:2], (x, y)) < 40 and math.hypot(p[0] - cx, p[1] - cy) < safe_r + 20]
            if items:
                p = min(items, key=lambda p: math.dist(p[:2], (x, y)))
                mx, my = unit(p[0] - x, p[1] - y)

        # ---- take a laser/lightning in reach (swapping out a mortar); drop a held mortar when calm ----
        if want_weapon:
            for kind in grab_kinds:
                near = [p for p in obs.get("pickups", [])
                        if p["kind"] == kind and p["id"] not in self.skip
                        and math.dist(p["pos"][:2], (x, y)) < self.reach - 0.2 and abs(p["pos"][2] - z) < 4]
                if near:
                    act["take"] = min(near, key=lambda p: math.dist(p["pos"][:2], (x, y)))["id"]
                    if not empty and mort_i is not None:
                        act["drop"] = mort_i
                    break
            else:
                if mort_i is not None and calm:
                    act["drop"] = mort_i

        # ---- map edge is a wall: slide along it instead of pushing into it ----
        if self.bounds and not self.lite and (mx or my):
            bx0, by0, bx1, by1 = self.bounds
            ox = (mx < 0 and x - bx0 < 15) or (mx > 0 and bx1 - x < 15)
            oy = (my < 0 and y - by0 < 15) or (my > 0 and by1 - y < 15)
            if ox or oy:
                rx, ry = (0.0 if ox else mx), (0.0 if oy else my)
                tx_, ty_ = (0.0, 1.0) if ox else (1.0, 0.0)  # tangent of the edge we are against
                flip = self.stuck >= 3 and t >= self.edge_until  # blocked that way (a building on the edge): try the other way
                if flip:
                    self.edge_sign, self.stuck, self.edge_until = -self.edge_sign, 0, t + 3.0
                elif t < self.edge_until:
                    pass  # keep the new way for a while
                elif abs(rx * tx_ + ry * ty_) > 0.3:
                    self.edge_sign = 1 if rx * tx_ + ry * ty_ > 0 else -1
                elif mechs:
                    away = (x - mechs[0]["pos"][0]) * tx_ + (y - mechs[0]["pos"][1]) * ty_
                    if abs(away) > 1.0:
                        self.edge_sign = 1 if away > 0 else -1
                if flip or flee or math.hypot(rx, ry) < 0.3:
                    mx, my = tx_ * self.edge_sign, ty_ * self.edge_sign  # full speed along the edge
                else:
                    mx, my = rx, ry

        # ---- stuck on a wall: sidestep, jump and jet ----
        want = math.hypot(mx, my) > 0.2
        speed = math.hypot(me["vel"][0], me["vel"][1])
        self.stuck = self.stuck + 1 if want and speed < 2.0 and me["grounded"] else 0
        if self.stuck >= 4 and self.detour == 0:
            if t - self.detour_end < 4.0:  # the last detour did not get us round: try the other side
                self.detour_sign = -self.detour_sign
            else:  # a new obstacle: go round on the side with more room
                heading = math.atan2(my, mx)
                if self.free_run(x, y, z, heading - 1.2) > self.free_run(x, y, z, heading + 1.2):
                    self.detour_sign = -1
                else:
                    self.detour_sign = 1
            self.detour = 25
        if self.detour > 0 and not (mx or my):  # holding still on purpose (drop ring): no detour
            self.detour = self.free_n = 0
        if self.detour > 0:
            self.free_n = self.free_n + 1 if speed > 5.0 else 0
            if self.free_n >= FREE_N:  # moving freely: assume we are round it, head for the goal again (same side next time)
                self.detour, self.free_n = 0, 0
            else:
                self.detour -= 1
                if self.detour == 0:
                    self.detour_end = t
                a = math.atan2(my, mx) + self.detour_sign * 1.2
                mx, my = math.cos(a), math.sin(a)
        jump = self.stuck >= 3 and me["energy"] > 30
        jet = (not me["grounded"]) and me["vel"][2] < 4 and want and self.detour > 0
        n = math.hypot(mx, my)
        if n > 1:
            mx, my, n = mx / n, my / n, 1.0
        # walk under 5 m/s when there is no hurry: no step sounds
        time_left = max(zone["shrink_end"] - t, 0.0)
        walk_t = (dz - safe_r) / 4.5
        hurry = urgent or flee or bool(mechs) or cp is not None or self.detour or goal is not None or dodging or walk_t > time_left * 0.7
        if not hurry and n > 0:
            k = min(1.0, 4.5 / max(me["stats"].get("walk_speed", 11.7), 0.1))
            mx, my = mx * k, my * k
        energy_ok = me["energy"] > 25 and me["grounded"] and n > 0
        boost = bool(urgent and energy_ok) or bool(goal is not None and loot is None and in_zone and dgoal > 12 and energy_ok) \
            or bool(dodge_boost and energy_ok) or bool(cp is not None and cp["boost"] and energy_ok and self.stuck < 2)
        act.update(move=[mx, my], boost=boost or jet, jump=jump)

        # ---- aim and fire: the aim module decides, v3's engagement rules gate it; v3 logic if it fails ----
        target = mechs[0] if mechs else None
        hitters = []
        if target is None:
            hitters = [m for m in npcs if math.dist(m["pos"][:2], (x, y)) < 60 and t - self.dmg_t < 2]
            target = hitters[0] if hitters else None
        fire = [False, False]
        aim_cmd = None
        reload_cmd = False
        seen_mechs = [m for m in mechs if self._seen(me, m, t)]
        plan = self._aim_plan(obs, hitters, seen_mechs, mechs, hide=hiding)
        if plan is not None:
            target = plan["target"]
            fire = self._gate_fire(plan["fire"], target, me, ws, seen_mechs, flee, t)  # hide=hiding already holds fire
            aim_cmd = plan["aim"]
            reload_cmd = plan["reload"]
            if las is not None:
                if target is not None and aim_cmd is not None:
                    self.lt_t, self.lt_aim = t, list(aim_cmd)  # remembered for the grace hold
                elif target is None and fire[las_i] and self.lt_aim is not None:
                    aim_cmd = list(self.lt_aim)
        else:
            aim_cmd, fire = self._v3_fire(me, t, target, mechs, flee, slot, gun, las, light, gun_i, las_i, light_i)
            if hiding:  # hold fire; a charging laser keeps its trigger (releasing would fire it)
                fire = [bool(ws[i]) and ws[i].get("kind") == "laser" and (ws[i].get("charge") or 0) > 0
                        for i in (0, 1)]
        if cp is not None and not cp["tracked"] and target is None and not (las is not None and fire[las_i]):
            ax, ay, az = cp["aim_at"]  # track dropped: look where it was going
            aim_cmd = [math.atan2(ay - y, ax - x),
                       math.atan2(az - z - me["stats"].get("muzzle_height", 7), max(math.hypot(ax - x, ay - y), 1.0))]
        if aim_cmd is None:
            aim_cmd = self._idle_aim(me, t, hit_recent, empty, grab, mx, my, n)
        act["aim"] = [float(aim_cmd[0]), float(aim_cmd[1])]
        if gun and gun.get("reload", 0) == 0:
            ammo = gun.get("ammo", 99)
            if ammo == 0 or (ammo <= 4 and target is None):
                reload_cmd = True
        if reload_cmd:
            act["reload"] = True
        if rest:  # dealing damage restarts the regen delay; a charging laser keeps its trigger (letting go fires it)
            fire = [f and bool(w and w.get("kind") == "laser" and (w.get("charge") or 0) > 0) for f, w in zip(fire, ws)]
        act["fire"] = fire

        mf.draw.label(f"hp {me['hp']:.0f} z{dz:.0f}/{safe_r:.0f}{' FLEE' if flee else ''}{' HIDE' if hiding else ''}{' GRAB' if grab else ''}{' CHASE' if cp else ''}")
        return act


if __name__ == "__main__":
    mf.run(Bot)
