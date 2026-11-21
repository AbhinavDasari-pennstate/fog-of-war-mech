"""Target tracking, lead prediction, aim solving and fire gating for mechfight/1.

Stdlib only. Angles in radians, lengths in metres, times in seconds.
Frame (protocol section 5): x east, y north, z up; yaw from +x counter-clockwise;
pitch up-positive. A mech's `pos` is its feet.

Public API
    Tracker            per-id history, smoothed velocity, acceleration, predict(id, dt)
    TargetState        small state for choose_target (current id, who hit us and when)
    aim_solution(...)  -> (yaw, pitch, t_on_target, hit_prob)
    solve(...)         -> Solution (the same numbers plus the aim point, miss distance, ...)
    choose_target(obs, me, state, ...)
    fire_decision(me, target, weapon, solution, ...) -> (fire, reason)
    plan_shots(obs, me, target, tracker, config, ...) -> {"aim", "fire", "reload", ...}

Line of fire: `los_fn(a_xyz, b_xyz)` returns True when the segment is clear. None means
"assume clear". It is called from the muzzle (feet z + muzzle_height) to the aim point.
"""
import math
from collections import deque

TAU = 2.0 * math.pi
_SQRT2 = math.sqrt(2.0)

LATENCY_S = 0.05          # observation to first server tick that sees our action
MAX_GAP_S = 1.0           # a longer silence restarts the history of an id
JUMP_SPEED = 45.0         # implied m/s above which two samples are a teleport, not motion
DEFAULT_METHOD = "damped"  # chosen on one half of the recorded track, scored on the other
DEFAULT_DAMP = 0.85        # lead = DEFAULT_DAMP * v * dt: strafers do not hold their velocity
METHODS = ("linear", "smoothed", "accel", "circular", "damped")

# Median predictor miss (m) per range bucket 0-40 / 40-80 / 80-200 m at gun speed. Only the
# 80-200 m bucket has recorded samples (median 1.78 m, n=153); the other two are that value
# scaled by flight time (the lead error grows with the prediction horizon). Used as the
# predictor error term of the hit probability.
PRED_ERR_M = (0.45, 0.85, 1.8)
BUCKET_EDGES = (40.0, 80.0)
BUCKET_MID_M = (25.0, 60.0, 130.0)

GUN_MIN_P = 0.35
LASER_RELEASE_P = 0.6
LASER_START_P = 0.35
LIGHTNING_MIN_ENERGY = 20.0
SIGMA_FLOOR_M = 0.15


def wrap(a):
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % TAU - math.pi


def _fin(*vals):
    for v in vals:
        if v is None or v != v or v in (math.inf, -math.inf):
            return False
    return True


def bucket_of(dist):
    if dist < BUCKET_EDGES[0]:
        return 0
    if dist < BUCKET_EDGES[1]:
        return 1
    return 2


# --------------------------------------------------------------------------- config


class Cfg:
    """Flat numbers pulled from hello.config (and self.stats), with protocol defaults."""

    def __init__(self, config=None, me=None):
        config = config or {}
        mech = config.get("mech") or {}
        cubes = config.get("cubes") or {}
        vis = config.get("vision") or {}
        npc = config.get("npc") or {}
        crates = config.get("crates") or {}
        wl = config.get("weapons") or {}
        st = (me or {}).get("stats") or {}

        def g(src, key, default):
            v = src.get(key, default)
            return float(v) if _fin(v) else default

        self.ground_accel = g(mech, "ground_accel", 58.5)
        self.boost_speed = g(st, "boost_speed", g(mech, "boost_speed", 19.5))
        self.turn_rate = max(g(st, "turn_rate", g(mech, "turn_rate", 4.0)), 0.1)
        self.pitch_rate = max(g(st, "pitch_rate", g(mech, "pitch_rate", 3.0)), 0.1)
        self.min_pitch = g(mech, "min_pitch", -0.7)
        self.max_pitch = g(mech, "max_pitch", 1.3)
        self.radius = g(mech, "radius", 2.5)
        self.height = g(mech, "height", 9.0)
        self.size_per_cube = g(cubes, "size_per_cube", 0.05)
        self.memory_s = g(vis, "memory_s", 5.0)
        self.npc_radius = g(npc, "radius", 1.8)
        self.npc_height = g(npc, "height", 3.5)
        self.crate_radius = g(crates, "radius", 1.4)
        self.crate_height = g(crates, "height", 2.4)
        self.muzzle_h = g(st, "muzzle_height", g(mech, "muzzle_height", 7.0))
        self.lightning = wl.get("lightning") or {}
        self.laser = wl.get("laser") or {}
        self.gun = wl.get("gun") or {}


def my_muzzle(me, cfg):
    """Muzzle point. self.stats.muzzle_height is already cube-scaled; if absent, scale it."""
    st = me.get("stats") or {}
    h = st.get("muzzle_height")
    if not _fin(h):
        h = 7.0 * (1.0 + cfg.size_per_cube * (me.get("cubes") or 0))
    x, y, z = me["pos"][0], me["pos"][1], me["pos"][2]
    return (x, y, z + h)


def target_dims(target, cfg):
    """(radius, height) of a target, cube-scaled for mechs (1 + size_per_cube * cubes, no cap)."""
    kind = target.get("kind", "mech")
    if kind == "npc":
        return cfg.npc_radius, cfg.npc_height
    if kind == "crate":
        return cfg.crate_radius, cfg.crate_height
    s = 1.0 + cfg.size_per_cube * max(float(target.get("cubes") or 0), 0.0)
    return cfg.radius * s, cfg.height * s


# --------------------------------------------------------------------------- kinematics
# A sample is (t, x, y, z, vx, vy, vz).


def smooth_vel(h, n=4):
    """Mean of the reported velocity over the last n samples."""
    k = h[-n:]
    m = float(len(k))
    return (sum(s[4] for s in k) / m, sum(s[5] for s in k) / m, sum(s[6] for s in k) / m)


def accel_est(h, n=4, cap=58.5):
    """Least-squares slope of the horizontal velocity over the last n samples, capped.

    Returns (ax, ay, vx_end, vy_end): the acceleration and the fitted velocity at the newest time.
    """
    k = h[-n:]
    if len(k) < 3:
        s = k[-1]
        return 0.0, 0.0, s[4], s[5]
    tm = sum(s[0] for s in k) / len(k)
    vxm = sum(s[4] for s in k) / len(k)
    vym = sum(s[5] for s in k) / len(k)
    sxx = sum((s[0] - tm) ** 2 for s in k)
    if sxx < 1e-4:
        s = k[-1]
        return 0.0, 0.0, s[4], s[5]
    ax = sum((s[0] - tm) * (s[4] - vxm) for s in k) / sxx
    ay = sum((s[0] - tm) * (s[5] - vym) for s in k) / sxx
    a = math.hypot(ax, ay)
    if a > cap:
        ax, ay = ax * cap / a, ay * cap / a
    te = k[-1][0] - tm
    return ax, ay, vxm + ax * te, vym + ay * te


def _integrate(x, y, vx, vy, ax, ay, dt, vmax, steps=6):
    """Constant-acceleration motion with a speed cap; a component that decelerates to 0 stays 0."""
    h = dt / steps
    for _ in range(steps):
        nvx, nvy = vx + ax * h, vy + ay * h
        if vx * nvx < 0:
            nvx, ax = 0.0, 0.0
        if vy * nvy < 0:
            nvy, ay = 0.0, 0.0
        sp = math.hypot(nvx, nvy)
        if sp > vmax:
            nvx, nvy = nvx * vmax / sp, nvy * vmax / sp
        x += 0.5 * (vx + nvx) * h
        y += 0.5 * (vy + nvy) * h
        vx, vy = nvx, nvy
    return x, y


def extrapolate(h, dt, method="linear", ground_accel=58.5, vmax=20.5, n=4, damp=None):
    """Predicted (x, y, z) of a target `dt` seconds after its newest sample in `h`."""
    t0, x, y, z, vx, vy, vz = h[-1]
    if damp is None:
        damp = DEFAULT_DAMP
    if dt <= 0.0:
        return x, y, z
    dz = max(-5.0, min(5.0, vz * dt))
    if method == "linear":
        return x + vx * dt, y + vy * dt, z + dz
    if method == "damped":
        return x + damp * vx * dt, y + damp * vy * dt, z + dz
    if len(h) < 2:
        return x + vx * dt, y + vy * dt, z + dz
    if method == "smoothed":
        sx, sy, _ = smooth_vel(h, n)
        return x + sx * dt, y + sy * dt, z + dz
    if method == "accel":
        ax, ay, ex, ey = accel_est(h, n, ground_accel)
        px, py = _integrate(x, y, ex, ey, ax, ay, dt, vmax)
        return px, py, z + dz
    if method == "circular":
        k = h[-max(n, 4):]
        half = len(k) // 2
        if half < 1:
            return x + vx * dt, y + vy * dt, z + dz
        a, b = k[:half], k[half:]
        ov = smooth_vel(a, len(a))
        nv = smooth_vel(b, len(b))
        dtc = (sum(s[0] for s in b) / len(b)) - (sum(s[0] for s in a) / len(a))
        so, sn = math.hypot(ov[0], ov[1]), math.hypot(nv[0], nv[1])
        if dtc < 0.05 or so < 2.0 or sn < 2.0:
            sx, sy, _ = smooth_vel(h, n)
            return x + sx * dt, y + sy * dt, z + dz
        om = wrap(math.atan2(nv[1], nv[0]) - math.atan2(ov[1], ov[0])) / dtc
        om = max(-3.0, min(3.0, om))
        th = math.atan2(nv[1], nv[0])
        if abs(om) < 1e-3:
            return x + nv[0] * dt, y + nv[1] * dt, z + dz
        px = x + sn / om * (math.sin(th + om * dt) - math.sin(th))
        py = y - sn / om * (math.cos(th + om * dt) - math.cos(th))
        return px, py, z + dz
    raise ValueError("unknown predictor %r" % (method,))


# --------------------------------------------------------------------------- tracker


class Track:
    __slots__ = ("id", "kind", "hist", "last_t", "hp", "max_hp", "cubes", "aim", "charging", "stale", "n_seen")

    def __init__(self, tid, kind, max_hist):
        self.id = tid
        self.kind = kind
        self.hist = deque(maxlen=max_hist)
        self.last_t = -1e9
        self.hp = None
        self.max_hp = None
        self.cubes = 0
        self.aim = None
        self.charging = False
        self.stale = False
        self.n_seen = 0


class Tracker:
    """Per-mech history built from obs.mechs.

    - dt = 0 (same t again) replaces the newest sample; t going backwards restarts the history.
    - a gap over MAX_GAP_S, or an implied speed over JUMP_SPEED (teleport, respawn), restarts it.
    - an id missing from the latest obs is stale (memory only): predict() dead-reckons at half
      velocity for at most 2 s and the track is dropped after memory_s.
    """

    def __init__(self, config=None, method=None, max_hist=6, smooth_n=4):
        self.cfg = Cfg(config)
        self.method = method or DEFAULT_METHOD
        self.max_hist = max_hist
        self.smooth_n = smooth_n
        self.tracks = {}
        self.now = 0.0

    def update(self, obs):
        t = obs.get("t")
        if not _fin(t):
            return
        if t < self.now - 1e-9:
            self.tracks.clear()  # time went backwards: a new match
        self.now = t
        seen = set()
        for m in obs.get("mechs") or []:
            tid = m.get("id")
            pos = m.get("pos")
            if tid is None or not pos or len(pos) < 3 or not _fin(*pos[:3]):
                continue
            vel = m.get("vel") or (0.0, 0.0, 0.0)
            if len(vel) < 3 or not _fin(*vel[:3]):
                vel = (0.0, 0.0, 0.0)
            seen.add(tid)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid, m.get("kind", "mech"), self.max_hist)
            self._push(tr, t, pos, vel)
            tr.stale = False
            tr.n_seen += 1
            tr.kind = m.get("kind", tr.kind)
            if _fin(m.get("hp")):
                tr.hp = m["hp"]
            if _fin(m.get("max_hp")):
                tr.max_hp = m["max_hp"]
            tr.cubes = m.get("cubes", tr.cubes) or 0
            tr.aim = m.get("aim", tr.aim)
            tr.charging = bool(m.get("charging", False))
        for tid in list(self.tracks):
            if tid in seen:
                continue
            tr = self.tracks[tid]
            tr.stale = True
            if t - tr.last_t > self.cfg.memory_s:
                del self.tracks[tid]

    def _push(self, tr, t, pos, vel):
        h = tr.hist
        if h:
            lt = h[-1][0]
            dt = t - lt
            if dt < -1e-9 or dt > MAX_GAP_S:
                h.clear()
            elif dt <= 1e-9:
                h.pop()  # same instant: replace
            else:
                d = math.dist(pos[:3], h[-1][1:4])
                if d / dt > JUMP_SPEED:
                    h.clear()
        h.append((t, pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]))
        tr.last_t = t

    def get(self, tid):
        return self.tracks.get(tid)

    def has(self, tid):
        return tid in self.tracks and len(self.tracks[tid].hist) > 0

    def velocity(self, tid):
        tr = self.tracks.get(tid)
        if not tr or not tr.hist:
            return (0.0, 0.0, 0.0)
        return smooth_vel(list(tr.hist), self.smooth_n)

    def accel(self, tid):
        tr = self.tracks.get(tid)
        if not tr or not tr.hist:
            return (0.0, 0.0)
        ax, ay, _, _ = accel_est(list(tr.hist), self.smooth_n, self.cfg.ground_accel)
        return (ax, ay)

    def age(self, tid):
        tr = self.tracks.get(tid)
        return self.now - tr.last_t if tr else math.inf

    def predict(self, tid, dt, method=None):
        """Position (x, y, z) of `tid` at (latest obs time + dt), or None for an unknown id."""
        tr = self.tracks.get(tid)
        if tr is None or not tr.hist:
            return None
        h = list(tr.hist)
        method = method or self.method
        c = self.cfg
        if tr.stale:
            age = max(self.now - tr.last_t, 0.0)
            hor = min(age + max(dt, 0.0), 2.0)
            _, x, y, z, vx, vy, _ = h[-1]
            return x + 0.5 * vx * hor, y + 0.5 * vy * hor, z
        return extrapolate(h, dt, method, c.ground_accel, c.boost_speed * 1.05, self.smooth_n)

    def synth(self, tid):
        """An obs-style dict for a tracked id (used for memory-only targets)."""
        tr = self.tracks.get(tid)
        if tr is None or not tr.hist:
            return None
        p = self.predict(tid, 0.0)
        vx, vy, vz = smooth_vel(list(tr.hist), 2)
        d = {"id": tid, "kind": tr.kind, "pos": list(p), "vel": [vx, vy, vz], "cubes": tr.cubes,
             "stale": tr.stale}
        if tr.hp is not None:
            d["hp"] = tr.hp
        if tr.max_hp is not None:
            d["max_hp"] = tr.max_hp
        if tr.aim is not None:
            d["aim"] = tr.aim
        d["charging"] = tr.charging
        return d


# --------------------------------------------------------------------------- geometry helpers


def _p1(offset, sigma, half):
    """P(|offset + N(0, sigma^2)| < half)."""
    if half <= 0.0:
        return 0.0
    if sigma < 1e-6:
        return 1.0 if offset <= half else 0.0
    s2 = sigma * _SQRT2
    return max(0.0, min(1.0, 0.5 * (math.erf((half - offset) / s2) + math.erf((half + offset) / s2))))


def rect_hit_prob(lat, vert, sigma, half_w, half_h):
    """Gaussian miss (sigma per axis) against a half_w x half_h rectangle, offset (lat, vert)."""
    return _p1(lat, sigma, half_w) * _p1(vert, sigma, half_h)


def _miss_offsets(dist, cur_yaw, cur_pitch, yaw, pitch):
    """(lateral, vertical) metres between the actual aim ray and the aim point at `dist`."""
    dy = min(abs(wrap(cur_yaw - yaw)), 0.5 * math.pi)
    dp = min(abs(cur_pitch - pitch), 0.5 * math.pi)
    return dist * math.cos(pitch) * math.sin(dy), dist * math.sin(dp)


def pred_sigma(dist, horizon, speed):
    """Per-axis predictor error (m) at `dist` for a prediction `horizon` seconds ahead."""
    med = PRED_ERR_M[bucket_of(dist)]
    ref = max(BUCKET_MID_M[bucket_of(dist)] / 252.0 + LATENCY_S, 0.05)
    scale = max(0.1, min(2.0, horizon / ref))
    sp = 0.3 + 0.7 * min(speed / 8.0, 1.0)
    return med / 1.1774 * scale * sp


# --------------------------------------------------------------------------- aim solution


class Solution:
    __slots__ = ("yaw", "pitch", "t_on_target", "hit_prob", "p_aimed", "aim_point", "dist", "radius",
                 "flight", "miss_m", "inside", "kind", "in_range", "in_cone", "ideal_yaw", "ideal_pitch")

    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.t_on_target = 0.0
        self.hit_prob = 0.0
        self.p_aimed = 0.0
        self.aim_point = (0.0, 0.0, 0.0)
        self.dist = 0.0
        self.radius = 0.0
        self.flight = 0.0
        self.miss_m = 0.0
        self.inside = False
        self.kind = "gun"
        self.in_range = False
        self.in_cone = False
        self.ideal_yaw = 0.0
        self.ideal_pitch = 0.0

    def as_tuple(self):
        return (self.yaw, self.pitch, self.t_on_target, self.hit_prob)


def _direction(src, dst, keep_yaw):
    dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
    hd = math.hypot(dx, dy)
    d = math.sqrt(hd * hd + dz * dz)
    yaw = math.atan2(dy, dx) if hd > 1e-6 else keep_yaw
    pitch = math.atan2(dz, hd) if d > 1e-6 else 0.0
    return yaw, pitch, d


def lightning_nearest(me, target, others, cfg, los_fn=None, cone_deg=None, rng=None):
    """(is_target_nearest_in_cone, target_in_cone, target_in_range) along our ACTUAL aim yaw.

    The protocol says "within cone_deg of your aim yaw". It does not say half or full width, so
    the target must sit inside half the cone, and anything inside the full cone can steal the arc.
    `others`: iterable of obs entries (mechs, npcs, crates) with a `pos`.
    """
    ls = cfg.lightning
    cone = math.radians(cone_deg if cone_deg is not None else (ls.get("cone_deg") or 30.0))
    rng = rng if rng is not None else (ls.get("range") or 45.0)
    mx, my, mz = me["pos"][0], me["pos"][1], me["pos"][2]
    ayaw = me["aim"][0]
    tpos = target["pos"]
    td = math.dist((mx, my, mz), tpos[:3])
    tyaw = math.atan2(tpos[1] - my, tpos[0] - mx) if math.hypot(tpos[0] - mx, tpos[1] - my) > 1e-6 else ayaw
    in_cone = abs(wrap(tyaw - ayaw)) <= 0.5 * cone
    in_range = td <= rng - 1.0
    nearest = True
    tid = target.get("id")
    for o in others or ():
        if o is target or (tid is not None and o.get("id") == tid and o.get("kind") == target.get("kind")):
            continue
        op = o.get("pos")
        if not op or not _fin(*op[:3]):
            continue
        hd = math.hypot(op[0] - mx, op[1] - my)
        od = math.dist((mx, my, mz), op[:3])
        if od > rng + 3.0:
            continue
        oyaw = math.atan2(op[1] - my, op[0] - mx) if hd > 1e-6 else ayaw
        if abs(wrap(oyaw - ayaw)) > cone:
            continue
        if od < td:
            if los_fn is not None:
                try:
                    clear = los_fn((mx, my, mz + 4.5), (op[0], op[1], op[2] + 1.0))
                except Exception:
                    clear = True
                if not clear:
                    continue
            nearest = False
            break
    return nearest, in_cone, in_range


def solve(me, target, weapon_kind, weapon_stats, config=None, tracker=None, others=None, los_fn=None,
          latency=LATENCY_S, method=None):
    """Full aim solution of one weapon against one target. See aim_solution for the short form."""
    cfg = Cfg(config, me)
    sol = Solution()
    sol.kind = weapon_kind
    ws = weapon_stats or {}
    muz = my_muzzle(me, cfg)
    mvx = me["vel"][0] if me.get("vel") else 0.0
    mvy = me["vel"][1] if me.get("vel") else 0.0
    if not _fin(mvx, mvy):
        mvx = mvy = 0.0
    cur_yaw, cur_pitch = me["aim"][0], me["aim"][1]
    radius, height = target_dims(target, cfg)
    sol.radius = radius
    hc = 0.5 * height
    tid = target.get("id")

    tvel = target.get("vel") or (0.0, 0.0, 0.0)
    tpos = target["pos"]

    def pred(dt):
        if tracker is not None and tid is not None and tracker.has(tid):
            p = tracker.predict(tid, dt, method)
            if p is not None:
                return p
        h = [(0.0, tpos[0], tpos[1], tpos[2], tvel[0], tvel[1], tvel[2])]
        return extrapolate(h, dt, method or DEFAULT_METHOD)

    speed = ws.get("speed") if weapon_kind == "gun" else None
    if not _fin(speed) or (speed is not None and speed <= 1.0):
        speed = 252.0 if weapon_kind == "gun" else None

    def lead(extra, shooter):
        """Aim point and flight time when the shot leaves `extra` s from now at `shooter`."""
        tof = 0.0
        p = pred(extra)
        for _ in range(3):
            p = pred(extra + tof)
            if speed is None:
                break
            d = math.dist(shooter, (p[0], p[1], p[2] + hc))
            tof = max(d - (cfg.radius + 0.5), 0.0) / speed
        return (p[0], p[1], p[2] + hc), tof

    # fire-now solution: no slew, only latency and flight
    shoot0 = (muz[0] + mvx * latency, muz[1] + mvy * latency, muz[2])
    ap0, tof0 = lead(latency, shoot0)
    iy, ip, dist0 = _direction(shoot0, ap0, cur_yaw)

    # commanded solution: also lead by the time the torso needs to turn onto the target
    t_slew = 0.0
    ap, tof = ap0, tof0
    cy, cp = iy, max(cfg.min_pitch, min(cfg.max_pitch, ip))
    for _ in range(3):
        extra = latency + t_slew
        shooter = (muz[0] + mvx * extra, muz[1] + mvy * extra, muz[2])
        ap, tof = lead(extra, shooter)
        yy, pp, _ = _direction(shooter, ap, cur_yaw)
        cy, cp = yy, max(cfg.min_pitch, min(cfg.max_pitch, pp))
        t_slew = max(abs(wrap(cy - cur_yaw)) / cfg.turn_rate, abs(cp - cur_pitch) / cfg.pitch_rate)

    sol.yaw, sol.pitch, sol.t_on_target = cy, cp, t_slew
    sol.aim_point, sol.flight, sol.dist = ap, tof, dist0
    sol.ideal_yaw, sol.ideal_pitch = iy, ip
    # miss of the actual aim against the target silhouette: a 2r x h' rectangle seen from the muzzle
    lat, vert = _miss_offsets(dist0, cur_yaw, cur_pitch, iy, ip)
    half_v = 0.5 * height * math.cos(ip) + radius * abs(math.sin(ip))
    sol.miss_m = math.hypot(lat, vert)
    sol.inside = lat <= radius and vert <= half_v

    tspeed = math.hypot(tvel[0], tvel[1])
    if tracker is not None and tid is not None and tracker.has(tid):
        tspeed = math.hypot(*tracker.velocity(tid)[:2])

    if weapon_kind == "lightning":
        rng = ws.get("range") or cfg.lightning.get("range") or 45.0
        cone = ws.get("cone_deg") or cfg.lightning.get("cone_deg") or 30.0
        nearest, in_cone, in_range = lightning_nearest(me, target, others, cfg, los_fn, cone, rng)
        sol.in_cone, sol.in_range = in_cone, in_range
        sol.hit_prob = 1.0 if (nearest and in_cone and in_range) else 0.0
        sol.p_aimed = 1.0 if (nearest and in_range) else 0.0
        sol.yaw, sol.pitch = iy, max(cfg.min_pitch, min(cfg.max_pitch, ip))
        sol.t_on_target = max(0.0, (abs(wrap(iy - cur_yaw)) - 0.4 * math.radians(cone)) / cfg.turn_rate)
        return sol
    if weapon_kind not in ("gun", "laser"):
        sol.hit_prob = 0.0  # mortar and unknown kinds are not solved here
        return sol

    rng = ws.get("range") or (230.0 if weapon_kind == "gun" else 280.0)
    sol.in_range = dist0 <= rng
    spread = ws.get("spread_deg") or 0.0
    s_spread = dist0 * math.tan(math.radians(spread)) if weapon_kind == "gun" else 0.0
    horizon = latency + tof0
    s_pred = pred_sigma(dist0, horizon, tspeed)
    sigma = math.sqrt(s_spread ** 2 + s_pred ** 2 + SIGMA_FLOOR_M ** 2)
    sol.p_aimed = rect_hit_prob(0.0, 0.0, sigma, radius, half_v) if sol.in_range else 0.0
    sol.hit_prob = rect_hit_prob(lat, vert, sigma, radius, half_v) if sol.in_range else 0.0
    return sol


def aim_solution(me, target, weapon_kind, weapon_stats, config=None, tracker=None, others=None, los_fn=None):
    """(yaw, pitch, t_on_target, hit_prob).

    yaw, pitch: command that leads for slew time, bullet flight (gun) and latency; laser has zero
    flight. t_on_target: seconds the torso needs to reach it (max of yaw time, pitch time).
    hit_prob: probability that a shot fired NOW, along the actual self.aim, hits.
    """
    return solve(me, target, weapon_kind, weapon_stats, config, tracker, others, los_fn).as_tuple()


# --------------------------------------------------------------------------- target choice


class TargetState:
    """Memory for choose_target: the current target and who hurt us when."""

    def __init__(self):
        self.current = None
        self.last_hit = {}

    def feed(self, obs):
        t = obs.get("t", 0.0)
        for ev in obs.get("events") or []:
            if ev.get("type") == "damaged" and ev.get("by") is not None:
                self.last_hit[ev["by"]] = t


def _score(me, m, t, state, stale=False):
    mx, my = me["pos"][0], me["pos"][1]
    p = m["pos"]
    dx, dy = mx - p[0], my - p[1]
    d = math.hypot(dx, dy)
    hp, mhp = m.get("hp"), m.get("max_hp")
    wounded = 0.0
    if _fin(hp, mhp) and mhp > 0:
        wounded = min(max(1.0 - hp / mhp, 0.0), 1.0)
    aimed = 0.0
    chg = 0.0
    a = m.get("aim")
    if a and _fin(a[0]) and d > 1e-6:
        off = abs(wrap(math.atan2(dy, dx) - a[0]))
        if off <= math.radians(5.0):
            aimed = 1.0
        if m.get("charging"):
            chg = 1.0 if off <= math.radians(15.0) else 0.5
    elif m.get("charging"):
        chg = 0.5
    hit = 1.0 if (t - state.last_hit.get(m.get("id"), -99.0)) <= 3.0 else 0.0
    dist_term = 1.0 if d <= 80.0 else max(0.0, 1.0 - (d - 80.0) / 140.0)
    s = 0.1 + 0.6 * dist_term + 1.0 * wounded + 0.8 * aimed + 0.8 * chg + 1.0 * hit
    return s * (0.6 if stale else 1.0)


def choose_target(obs, me, state=None, tracker=None, include_npcs=False):
    """Best target (an obs-style dict) or None. Scores wounded, threat and range; +25% stickiness."""
    if state is None:
        state = TargetState()
    state.feed(obs)
    t = obs.get("t", 0.0)
    my_id = me.get("id")
    cands = []
    live = set()
    for m in obs.get("mechs") or []:
        if m.get("kind", "mech") == "npc" and not include_npcs:
            continue
        pos = m.get("pos")
        if not pos or len(pos) < 3 or not _fin(*pos[:3]) or m.get("id") == my_id:
            continue
        hp = m.get("hp")
        if hp is not None and _fin(hp) and hp <= 0:
            continue
        live.add(m.get("id"))
        cands.append((m, False))
    if tracker is not None:
        for tid, tr in tracker.tracks.items():
            if tid in live or tid == my_id or not tr.stale or tr.kind == "npc":
                continue
            if tracker.age(tid) > tracker.cfg.memory_s:
                continue
            if tr.hp is not None and tr.hp <= 0:
                continue
            s = tracker.synth(tid)
            if s is not None:
                cands.append((s, True))
    best, best_score = None, -1.0
    for m, stale in cands:
        s = _score(me, m, t, state, stale)
        if m.get("id") == state.current:
            s *= 1.25
        if s > best_score:
            best, best_score = m, s
    state.current = best.get("id") if best is not None else None
    return best


# --------------------------------------------------------------------------- fire gate


def needs_reload(weapon):
    return bool(weapon and weapon.get("kind") == "gun" and weapon.get("ammo", 1) == 0 and not weapon.get("reload"))


def fire_decision(me, target, weapon, sol, config=None, los_fn=None, hide=False, deadline=None, now=None,
                  held_full_s=0.0):
    """(fire, reason) for one weapon slot.

    gun: p >= 0.35, ammo, in range, line of fire clear.
    laser: `fire` True means HOLD the trigger. It starts a charge when a shot is plausible and
      releases (False) when p >= 0.6 on the actual aim with the aim inside the target radius,
      or just before the hold_s auto-fire. held_full_s: seconds the charge has sat at full (the
      caller times it, in case the reported charge stops at `charge`).
    lightning: only when the target is the nearest thing in the cone, in range, energy > 20.
    hide (T4) and a passed deadline (T3) win over any good shot.
    """
    if weapon is None or sol is None:
        return False, "empty"
    kind = weapon.get("kind")
    if hide:
        return False, "hide"
    if deadline is not None and now is not None and now >= deadline:
        return False, "deadline"
    cfg = Cfg(config, me)
    muz = my_muzzle(me, cfg)

    def clear():
        if los_fn is None:
            return True
        try:
            return bool(los_fn(muz, sol.aim_point))
        except Exception:
            return True

    if kind == "gun":
        if weapon.get("ammo", 1) <= 0 or weapon.get("reload", 0) > 0:
            return False, "ammo"
        if not sol.in_range:
            return False, "range"
        if sol.hit_prob < GUN_MIN_P:
            return False, "p"
        if not clear():
            return False, "blocked"
        return True, "fire"
    if kind == "laser":
        ls = (weapon.get("stats") or {})
        full = ls.get("charge", cfg.laser.get("charge", 0.75))
        minc = ls.get("min_charge", cfg.laser.get("min_charge", 0.15))
        hold = ls.get("hold_s", cfg.laser.get("hold_s", 2.0))
        charge = weapon.get("charge", 0.0) or 0.0
        if charge <= 0.0:
            if weapon.get("cooldown", 0.0) > 0.1:
                return False, "cooldown"
            if not sol.in_range or sol.p_aimed < LASER_START_P:
                return False, "p"
            if not clear():
                return False, "blocked"
            return True, "charge"
        if charge < minc:
            return True, "charging"
        held = max(charge - full, held_full_s if _fin(held_full_s) else 0.0)
        if held >= hold - 0.15:
            return False, "release_hold_limit"
        if charge >= 0.9 * full and sol.hit_prob >= LASER_RELEASE_P and sol.inside and sol.in_range and clear():
            return False, "release"
        return True, "charging"
    if kind == "lightning":
        if me.get("energy", 100.0) <= LIGHTNING_MIN_ENERGY:
            return False, "energy"
        if sol.hit_prob < 0.5:
            return False, "not_nearest_in_cone"
        if not clear():
            return False, "blocked"
        return True, "fire"
    return False, "unsupported"


def plan_shots(obs, me, target, tracker=None, config=None, los_fn=None, hide=False, deadline=None,
               others=None, held_full=(0.0, 0.0)):
    """One call for the bot: aim command, fire flags for both slots, reload flag.

    The weapon that drives the aim is lightning (target nearest in cone), then a ready or charging
    laser, otherwise the gun. Fire flags are decided per weapon against the actual torso aim.
    """
    out = {"aim": None, "fire": [False, False], "reload": False, "reasons": [None, None], "sols": [None, None]}
    if target is None:
        return out
    t = obs.get("t")
    if others is None:
        others = list(obs.get("mechs") or []) + list(obs.get("crates") or [])
    sols = [None, None]
    for i, w in enumerate(me.get("weapons") or []):
        if w is None:
            continue
        sols[i] = solve(me, target, w["kind"], w.get("stats"), config, tracker, others, los_fn)
        f, why = fire_decision(me, target, w, sols[i], config, los_fn, hide, deadline, t, held_full[i])
        out["fire"][i] = f
        out["reasons"][i] = why
    out["sols"] = sols
    out["reload"] = any(needs_reload(w) for w in (me.get("weapons") or []))
    drive = None
    ws = me.get("weapons") or []
    for i, w in enumerate(ws):
        if w and w["kind"] == "lightning" and sols[i] is not None and sols[i].p_aimed >= 0.5:
            drive = i
    if drive is None:
        for i, w in enumerate(ws):
            if w and w["kind"] == "laser" and sols[i] is not None and sols[i].in_range and (
                    (w.get("charge") or 0) > 0 or (w.get("cooldown", 0) <= 0.1 and sols[i].p_aimed >= LASER_START_P)):
                drive = i
    if drive is None:
        for i, w in enumerate(ws):
            if w and w["kind"] == "gun":
                drive = i
    if drive is None:
        for i, s in enumerate(sols):
            if s is not None:
                drive = i
                break
    if drive is not None:
        s = sols[drive]
        out["aim"] = [s.yaw, s.pitch]
    return out


# --------------------------------------------------------------------------- integration entry
# One shared tracker, target memory, config and line-of-fire test per bot process.
# aim.init(hello) is optional: without it protocol defaults are used.

_state = {"tracker": None, "targets": None, "config": None, "los": None, "slots": [None, None],
          "full_t": [None, None]}
last_plan = {}


def init(hello=None):
    """Reset the shared state from the hello message (or hello["config"]).

    When hello.map.dir is given, the map is loaded once with ray.init() and decide(obs, None)
    uses ray.clear; without a loadable map, los=None means "assume clear". Never raises.
    """
    cfg = hello
    map_dir = None
    if isinstance(hello, dict) and "config" in hello and "mech" not in hello:
        cfg = hello.get("config")
        mp = hello.get("map")
        if isinstance(mp, dict):
            map_dir = mp.get("dir")
    _state["config"] = cfg if isinstance(cfg, dict) else {}
    _state["tracker"] = Tracker(_state["config"])
    _state["targets"] = TargetState()
    _state["los"] = None
    _state["slots"] = [None, None]
    _state["full_t"] = [None, None]
    if map_dir:
        try:
            import ray
            _state["los"] = ray.init(map_dir).clear
        except Exception:  # bad mount or odd map: shoot without a line-of-fire check
            _state["los"] = None


def _laser_hold(t, weapons, config):
    """Seconds each slot's laser has sat at full charge. Any change of slot kind resets it."""
    held = [0.0, 0.0]
    lcfg = ((config or {}).get("weapons") or {}).get("laser") or {}
    for i in range(2):
        w = weapons[i] if i < len(weapons) else None
        kind = w.get("kind") if isinstance(w, dict) else None
        if kind != _state["slots"][i]:
            _state["slots"][i] = kind
            _state["full_t"][i] = None
        if kind != "laser" or not _fin(t):
            _state["full_t"][i] = None
            continue
        full = (w.get("stats") or {}).get("charge", lcfg.get("charge", 0.75))
        charge = w.get("charge") or 0.0
        if not _fin(full, charge) or charge < full - 1e-6:
            _state["full_t"][i] = None
            continue
        if _state["full_t"][i] is None or _state["full_t"][i] > t:
            _state["full_t"][i] = t
        held[i] = t - _state["full_t"][i]
    return held


def decide(obs, los=None, hide=False, deadline=None, include_npcs=False):
    """(aim, fire, target) for this observation.

    aim: [yaw, pitch] or None when there is nothing to shoot at (the caller keeps its own aim).
    fire: [slot0, slot1] trigger flags (laser True = hold the trigger, False = release/fire).
    target: the chosen obs-style dict or None.
    los: los(a_xyz, b_xyz) -> True when clear (ray.clear); None uses the map init(hello) loaded,
      else assumes clear.
    hide (T4 no-fire) and deadline (T3 exposure, compared with obs["t"]) suppress every shot.
    Details of the last call (per-slot reasons, reload flag, solutions) are in `last_plan`.
    Never raises: on an internal error it returns no aim and no fire and records `last_plan["error"]`.
    """
    global last_plan
    try:
        if _state["tracker"] is None:
            init(None)
        st_tr, st_tg, cfg = _state["tracker"], _state["targets"], _state["config"]
        st_tr.update(obs)
        me = obs["self"]
        held = _laser_hold(obs.get("t"), me.get("weapons") or [], cfg)
        # obs.mechs already carries memory_s of server-side tracking: no extra ghost targets here
        target = choose_target(obs, me, st_tg, None, include_npcs)
        if target is None:
            last_plan = {"aim": None, "fire": [False, False], "reload": any(
                needs_reload(w) for w in (me.get("weapons") or [])), "reasons": [None, None]}
            return None, [False, False], None
        los_fn = los if los is not None else _state["los"]
        plan = plan_shots(obs, me, target, st_tr, cfg, los_fn, hide, deadline, held_full=held)
        last_plan = plan
        return plan["aim"], plan["fire"], target
    except Exception as exc:  # a crash must not leave the mech without an action
        last_plan = {"error": "%s: %s" % (type(exc).__name__, exc), "fire": [False, False], "aim": None}
        return None, [False, False], None
