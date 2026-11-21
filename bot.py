"""Survival-first mechfight/1 bot: stay deep in the next zone, stay quiet, face threats, shoot back, flee when hurt."""
import math
import mechfight as mf

TAU = 2 * math.pi
# ponytail: fixed upgrade priority; tune from practice results
UPGRADE_PRIO = ["plating", "armor", "nanites", "sensors", "servos", "gun_damage", "gun_rate"]


def ang_diff(a, b):
    return (a - b + math.pi) % TAU - math.pi


def unit(dx, dy):
    n = math.hypot(dx, dy)
    return (dx / n, dy / n) if n > 1e-6 else (0.0, 0.0)


class Bot:
    name = "fog-of-war"

    def __init__(self, hello: mf.Hello):
        c = hello["config"]
        self.reach = c["loot"]["pickup_radius"]
        self.g = c["weapons"]["mortar"]["gravity"]
        self.stuck = 0          # decisions in a row we wanted to move but barely did
        self.detour = 0         # decisions left on a sideways detour
        self.detour_sign = 1
        self.threat_yaw = None  # last bearing we were hit from
        self.threat_t = -99.0

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
            t = d / speed
        return [math.atan2(py - y, px - x), math.atan2(tz - mz, math.hypot(px - x, py - y))], d

    def mortar_pitch(self, dist, v):
        s = dist * self.g / (v * v)
        return None if s >= 1 else 0.5 * math.asin(s)

    def act(self, obs: mf.Observation) -> mf.Action:
        me = obs["self"]
        x, y, z = me["pos"]
        t = obs["t"]
        act: mf.Action = {"tick": obs["tick"]}
        if obs["offer"]:
            act["pick"] = self.pick_upgrade(obs["offer"])

        for ev in obs["events"]:
            if ev["type"] == "damaged" and ev.get("bearing") is not None:
                self.threat_yaw, self.threat_t = ev["bearing"], t

        hp_frac = me["hp"] / me["max_hp"]
        mechs = sorted((m for m in obs["mechs"] if m["kind"] == "mech"), key=lambda m: math.dist(m["pos"][:2], (x, y)))
        npcs = [m for m in obs["mechs"] if m["kind"] == "npc"]

        # ---- where to stand: deep inside the circle the zone is closing to ----
        zone = obs["zone"]
        cx, cy = zone["next_center"]
        # v2: stay on the quiet outer part of the next circle, not the crowded center
        safe_r = max(zone["next_radius"] * 0.8 - 15, 6.0)
        dz = math.hypot(cx - x, cy - y)
        s0, s1 = zone["shrink_start"], zone["shrink_end"]
        f = min(max((t - s0) / (s1 - s0), 0.0), 1.0) if s1 > s0 else 0.0
        ccx = zone["center"][0] + (cx - zone["center"][0]) * f
        ccy = zone["center"][1] + (cy - zone["center"][1]) * f
        cr = zone["radius"] + (zone["next_radius"] - zone["radius"]) * f
        urgent = (not me["in_zone"]) or math.hypot(ccx - x, ccy - y) > cr - 25
        mx = my = 0.0
        if dz > safe_r:
            mx, my = unit(cx - x, cy - y)

        # ---- danger: flee visible mechs when hurt, keep off drones ----
        flee = hp_frac < 0.45
        if flee:
            for m in mechs[:3]:
                d = math.dist(m["pos"][:2], (x, y))
                if d < 120:
                    ux, uy = unit(x - m["pos"][0], y - m["pos"][1])
                    w = 1.5 * (120 - d) / 120
                    mx, my = mx + ux * w, my + uy * w
        for npc in npcs:
            d = math.dist(npc["pos"][:2], (x, y))
            if d < 55:
                ux, uy = unit(x - npc["pos"][0], y - npc["pos"][1])
                mx, my = mx + ux * (55 - d) / 55, my + uy * (55 - d) / 55

        # ---- free loot nearby when calm: cubes, and a second weapon ----
        calm = not mechs and not urgent and t - self.threat_t > 3
        if calm and mx == 0 and my == 0:
            items = [c["pos"] for c in obs["cubes"]]
            if me["weapons"][1] is None:
                items += [p["pos"] for p in obs["pickups"] if p["kind"] != "gun"]
            items = [p for p in items if math.dist(p[:2], (x, y)) < 40 and math.hypot(p[0] - cx, p[1] - cy) < safe_r + 20]
            if items:
                p = min(items, key=lambda p: math.dist(p[:2], (x, y)))
                mx, my = unit(p[0] - x, p[1] - y)
        if me["weapons"][1] is None:
            for p in obs["pickups"]:
                if p["kind"] != "gun" and math.dist(p["pos"][:2], (x, y)) < self.reach and abs(p["pos"][2] - z) < 4:
                    act["take"] = p["id"]
                    break

        # ---- stuck on a wall: sidestep, jump and jet ----
        want = math.hypot(mx, my) > 0.2
        speed = math.hypot(me["vel"][0], me["vel"][1])
        self.stuck = self.stuck + 1 if want and speed < 2.0 and me["grounded"] else 0
        if self.stuck >= 4 and self.detour == 0:
            self.detour, self.detour_sign = 12, -self.detour_sign
        if self.detour > 0:
            self.detour -= 1
            a = math.atan2(my, mx) + self.detour_sign * 1.2
            mx, my = math.cos(a), math.sin(a)
        jump = self.stuck >= 3 and me["energy"] > 30
        jet = (not me["grounded"]) and me["vel"][2] < 4 and want and self.detour > 0
        n = math.hypot(mx, my)
        if n > 1:
            mx, my, n = mx / n, my / n, 1.0
        # walk under 5 m/s when there is no hurry: no step sounds
        # only hurry when the zone is about to reach us (time to shrink_end vs walk time) or in danger
        time_left = max(zone["shrink_end"] - t, 0.0)
        walk_t = (dz - safe_r) / 4.5
        hurry = urgent or flee or bool(mechs) or self.detour or walk_t > time_left * 0.7
        if not hurry and n > 0:
            k = 4.5 / me["stats"].get("walk_speed", 11.7)
            mx, my = mx * k, my * k
        boost = bool(urgent and me["energy"] > 25 and me["grounded"] and n > 0)
        act.update(move=[mx, my], boost=boost or jet, jump=jump)

        # ---- aim and fire ----
        target = mechs[0] if mechs else None
        if target is None:
            hitters = [m for m in npcs if math.dist(m["pos"][:2], (x, y)) < 60 and t - self.threat_t < 2]
            target = hitters[0] if hitters else None
        fire = [False, False]
        w0, w1 = me["weapons"][0], me["weapons"][1]
        gun = w0 if w0 and w0["kind"] == "gun" else None
        if target is not None:
            aim, d = self.lead_aim(me, target, gun["stats"]["speed"] if gun else 252)
            act["aim"] = aim
            err = abs(ang_diff(me["aim"][0], aim[0]))
            if gun and d < gun["stats"]["range"] and err < 0.06 and gun["ammo"] > 0:
                fire[0] = True
            if w1:
                k, st = w1["kind"], w1["stats"]
                if k == "lightning" and d < st["range"] and err < 0.4 and me["energy"] > 10:
                    fire[1] = True
                elif k == "laser" and d < st["range"]:
                    # hold to charge; let go once nearly full and on target
                    fire[1] = not (w1.get("charge", 0) >= 0.6 and err < 0.03)
                elif k == "mortar" and 25 < d < 220:
                    p = self.mortar_pitch(d, st["speed"])
                    if p is not None:
                        act["aim"] = [aim[0], p]
                        fire[1] = err < 0.05
            if gun and gun["ammo"] == 0 and gun["reload"] == 0:
                act["reload"] = True
        else:
            loud = [s for s in obs["sounds"] if s["kind"] not in ("crate", "drop", "death") and s["dist"] != "far"]
            if loud:
                self.threat_yaw, self.threat_t = loud[0]["bearing"], max(self.threat_t, t - 2)
            if self.threat_yaw is not None and t - self.threat_t < 4:
                act["aim"] = [self.threat_yaw, 0.0]   # face the attacker or the noise: rear hits do x1.5
            elif n > 0:
                act["aim"] = [math.atan2(my, mx), 0.0]
            else:
                act["aim"] = [me["aim"][0] + 0.5, 0.0]  # idle: scan around
            if gun and gun["ammo"] < gun["stats"]["mag"] and gun["reload"] == 0:
                act["reload"] = True
        act["fire"] = fire

        mf.draw.label(f"hp {me['hp']:.0f} z{dz:.0f}/{safe_r:.0f}{' FLEE' if flee else ''}")
        return act


mf.run(Bot)
