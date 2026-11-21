"""Exact segment line-of-fire test against the static map (stdlib only).

    clear(a_xyz, b_xyz, terrain=True) -> bool      True when nothing solid is between a and b

Solids: every prism in world.json (building, stadium, bridge): a footprint polygon (`ring`, minus
`holes`) extruded between `base` and `top`; and the ground from terrain.i16. Trees never stop
shots (protocol section 7), cars, barriers and water are scenery the server ignores.

The prism test is exact: the segment is clipped to the prism's height slab, split at every
crossing of a ring or hole edge, and the midpoint of each piece is tested for being inside the
ring and outside the holes. A uniform grid keeps only the prisms near the segment. The terrain
test is exact against the bilinear height field: per terrain cell the clearance is quadratic
along the segment. Stretches higher than the tallest ground in their 8 m blocks are skipped.

Load once: `init(map_dir)` (or the first `clear` call finds the map itself), or build a `Ray`.
"""
import json
import math
import os
import sys
from array import array

_CELL = 24.0
_BLK = 8


def _find_map_dir(map_dir=None):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [map_dir, os.environ.get("MECHFIGHT_MAP_DIR"), "/arena", os.path.join(here, "arena"),
             os.path.join(here, "..", "kit", "mechfight-kit", "arena")]
    for c in cands:
        if not c:
            continue
        for sub in ("v2", ""):
            if os.path.isfile(os.path.join(c, sub, "world.json")):
                return os.path.join(c, sub)
    return None


class Ray:
    def __init__(self, map_dir=None, world=None, terrain=None):
        """Load from a map dir, or from a world dict (and optional terrain: (array, tmeta))."""
        if world is None:
            d = _find_map_dir(map_dir)
            if d is None:
                raise FileNotFoundError("mechfight map not found (world.json)")
            with open(os.path.join(d, "world.json")) as f:
                world = json.load(f)
            tm = world.get("terrain") or {}
            tpath = os.path.join(d, tm.get("file", "terrain.i16"))
            if os.path.isfile(tpath):
                arr = array("h")
                with open(tpath, "rb") as f:
                    arr.frombytes(f.read())
                if sys.byteorder == "big":
                    arr.byteswap()
                terrain = (arr, tm)
        self.prisms = []
        for p in world.get("prisms") or []:
            self._add_prism(p)
        self._build_grid()
        self._qid = 0
        self.hmap = None
        if terrain is not None:
            self._init_terrain(*terrain)

    # ------------------------------------------------------------------ prisms

    def _add_prism(self, p):
        def poly(flat):
            return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat) - 1, 2)]

        def edges(pts):
            n = len(pts)
            return [(pts[i][0], pts[i][1], pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1])
                    for i in range(n)]

        ring = poly(p["ring"])
        if len(ring) < 3:
            return
        holes = [poly(h) for h in (p.get("holes") or []) if len(h) >= 6]
        xs = [q[0] for q in ring]
        ys = [q[1] for q in ring]
        ed = edges(ring)
        for h in holes:
            ed += edges(h)
        self.prisms.append({
            "kind": p.get("kind", "building"), "base": float(p["base"]), "top": float(p["top"]),
            "bbox": (min(xs), min(ys), max(xs), max(ys)), "edges": ed,
            "ring": edges(ring), "holes": [edges(h) for h in holes],
        })

    def _build_grid(self):
        self.grid = {}
        for pid, p in enumerate(self.prisms):
            x0, y0, x1, y1 = p["bbox"]
            for cx in range(int(math.floor(x0 / _CELL)), int(math.floor(x1 / _CELL)) + 1):
                for cy in range(int(math.floor(y0 / _CELL)), int(math.floor(y1 / _CELL)) + 1):
                    self.grid.setdefault((cx, cy), []).append(pid)
        self._stamp = [-1] * len(self.prisms)

    @staticmethod
    def _inside_edges(edges, x, y):
        c = False
        for x0, y0, ex, ey in edges:
            y1 = y0 + ey
            if (y0 > y) != (y1 > y):
                if x < x0 + ex * (y - y0) / ey:
                    c = not c
        return c

    def _inside(self, p, x, y):
        if not self._inside_edges(p["ring"], x, y):
            return False
        for h in p["holes"]:
            if self._inside_edges(h, x, y):
                return False
        return True

    def _seg_prism(self, p, ax, ay, az, dx, dy, dz):
        """Does the segment a + t*d, t in [0, 1], pass through solid prism p?"""
        t0, t1 = 0.0, 1.0
        base, top = p["base"], p["top"]
        # a level segment is already known to lie inside [base, top]: blocker() rejected the rest
        if abs(dz) >= 1e-12:
            ta, tb = (base - az) / dz, (top - az) / dz
            if ta > tb:
                ta, tb = tb, ta
            if ta > t0:
                t0 = ta
            if tb < t1:
                t1 = tb
            if t0 > t1:
                return False
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return self._inside(p, ax, ay)
        ts = [t0, t1]
        for x0, y0, ex, ey in p["edges"]:
            den = dx * ey - dy * ex
            if abs(den) < 1e-12:
                continue
            qx, qy = x0 - ax, y0 - ay
            t = (qx * ey - qy * ex) / den
            u = (qx * dy - qy * dx) / den
            if 0.0 <= u <= 1.0 and t0 < t < t1:
                ts.append(t)
        ts.sort()
        for i in range(len(ts) - 1):
            if ts[i + 1] - ts[i] > 1e-9:
                m = 0.5 * (ts[i] + ts[i + 1])
                if self._inside(p, ax + dx * m, ay + dy * m):
                    return True
        return False

    def _cells(self, ax, ay, bx, by):
        """Grid cells crossed by the 2D segment (Amanatides-Woo walk)."""
        cx, cy = int(math.floor(ax / _CELL)), int(math.floor(ay / _CELL))
        ex, ey = int(math.floor(bx / _CELL)), int(math.floor(by / _CELL))
        out = [(cx, cy)]
        if cx == ex and cy == ey:
            return out
        dx, dy = bx - ax, by - ay
        sx = 1 if dx > 0 else -1
        sy = 1 if dy > 0 else -1
        tdx = abs(_CELL / dx) if dx != 0 else math.inf
        tdy = abs(_CELL / dy) if dy != 0 else math.inf
        nx = ((cx + (1 if sx > 0 else 0)) * _CELL - ax) / dx if dx != 0 else math.inf
        ny = ((cy + (1 if sy > 0 else 0)) * _CELL - ay) / dy if dy != 0 else math.inf
        for _ in range(4096):
            if nx < ny:
                cx += sx
                nx += tdx
            else:
                cy += sy
                ny += tdy
            out.append((cx, cy))
            if (cx == ex and cy == ey) or (nx > 1.0 and ny > 1.0):
                break
        return out

    def blocker(self, a, b):
        """The first prism dict that blocks segment a-b, or None (debug and tests)."""
        ax, ay, az = a[0], a[1], a[2]
        dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
        bx, by = b[0], b[1]
        lx, hx = (ax, bx) if ax < bx else (bx, ax)
        ly, hy = (ay, by) if ay < by else (by, ay)
        lz, hz = (az, b[2]) if az < b[2] else (b[2], az)
        self._qid += 1
        q = self._qid
        stamp = self._stamp
        prisms = self.prisms
        grid = self.grid
        for cell in self._cells(ax, ay, bx, by):
            for pid in grid.get(cell, ()):
                if stamp[pid] == q:
                    continue
                stamp[pid] = q
                p = prisms[pid]
                x0, y0, x1, y1 = p["bbox"]
                if hx < x0 or lx > x1 or hy < y0 or ly > y1 or hz < p["base"] or lz > p["top"]:
                    continue
                if self._seg_prism(p, ax, ay, az, dx, dy, dz):
                    return p
        return None

    # ------------------------------------------------------------------ terrain

    def _init_terrain(self, arr, tm):
        self.h = arr
        self.t_minx = float(tm.get("minX", 0.0))
        self.t_miny = float(tm.get("minY", 0.0))
        self.t_step = float(tm.get("step", 1.0))
        self.t_nx = int(tm["nx"])
        self.t_ny = int(tm["ny"])
        self.t_scale = float(tm.get("scale", 0.01))
        nx, ny = self.t_nx, self.t_ny
        nbx = (nx - 1) // _BLK + 1
        nby = (ny - 1) // _BLK + 1
        bm = [[-1e9] * nbx for _ in range(nby)]
        for j in range(ny):
            row = j * nx
            by0 = max(0, (j - 1) // _BLK) if j % _BLK == 0 else j // _BLK
            for bx in range(nbx):
                i0 = bx * _BLK
                i1 = min(i0 + _BLK, nx - 1)
                m = max(arr[row + i0: row + i1 + 1]) * self.t_scale
                for by in {j // _BLK, by0}:
                    if by < nby and m > bm[by][bx]:
                        bm[by][bx] = m
        self.bm, self.nbx, self.nby = bm, nbx, nby
        self.hmap = True

    def ground(self, x, y):
        """Bilinear ground height at (x, y); clamped to the map edge. 0 when there is no terrain."""
        if not self.hmap:
            return 0.0
        fx = (x - self.t_minx) / self.t_step
        fy = (y - self.t_miny) / self.t_step
        nx, ny = self.t_nx, self.t_ny
        i = int(fx) if fx > 0.0 else 0
        j = int(fy) if fy > 0.0 else 0
        if i > nx - 2:
            i = nx - 2
        if j > ny - 2:
            j = ny - 2
        u = min(max(fx - i, 0.0), 1.0)
        v = min(max(fy - j, 0.0), 1.0)
        h = self.h
        k = j * nx + i
        h00, h10, h01, h11 = h[k], h[k + 1], h[k + nx], h[k + nx + 1]
        return self.t_scale * ((h00 * (1 - u) + h10 * u) * (1 - v) + (h01 * (1 - u) + h11 * u) * v)

    def _terrain_blocked(self, a, b):
        ax, ay, az = a[0], a[1], a[2]
        dx, dy, dz = b[0] - ax, b[1] - ay, b[2] - az
        L = math.hypot(dx, dy)
        chunk = float(_BLK)
        n = max(1, int(math.ceil(L / chunk)))
        for c in range(n):
            ta, tb = c / n, (c + 1) / n
            xa, ya, za = ax + dx * ta, ay + dy * ta, az + dz * ta
            xb, yb, zb = ax + dx * tb, ay + dy * tb, az + dz * tb
            zmin = za if za < zb else zb
            bx0 = int((min(xa, xb) - self.t_minx) // _BLK)
            bx1 = int((max(xa, xb) - self.t_minx) // _BLK)
            by0 = int((min(ya, yb) - self.t_miny) // _BLK)
            by1 = int((max(ya, yb) - self.t_miny) // _BLK)
            bx0, bx1 = max(bx0, 0), min(bx1, self.nbx - 1)
            by0, by1 = max(by0, 0), min(by1, self.nby - 1)
            peak = -1e9
            for by in range(by0, by1 + 1):
                for bx in range(bx0, bx1 + 1):
                    v = self.bm[by][bx]
                    if v > peak:
                        peak = v
            if bx0 > bx1 or by0 > by1:
                peak = 1e9  # off the map: fall through to the clamped fine test
            if zmin > peak + 1e-6:
                continue
            if self._below_exact(xa, ya, za, xb, yb, zb):
                return True
        return False

    def _below_exact(self, xa, ya, za, xb, yb, zb):
        """Exact: does the segment go under the bilinear ground? Inside one terrain cell the
        clearance z - ground is quadratic in the segment parameter, so 3 samples fix it."""
        st, mx, my = self.t_step, self.t_minx, self.t_miny
        gxa, gya = (xa - mx) / st, (ya - my) / st
        gxb, gyb = (xb - mx) / st, (yb - my) / st
        ts = [0.0, 1.0]
        for g0, g1 in ((gxa, gxb), (gya, gyb)):
            if abs(g1 - g0) > 1e-12:
                lo, hi = (g0, g1) if g0 < g1 else (g1, g0)
                for k in range(int(math.floor(lo)) + 1, int(math.ceil(hi))):
                    ts.append((k - g0) / (g1 - g0))
        ts.sort()
        dx, dy, dz = xb - xa, yb - ya, zb - za
        gr = self.ground

        def c(t):
            return za + dz * t - gr(xa + dx * t, ya + dy * t)

        prev = c(0.0)
        if prev < -1e-6:
            return True
        for i in range(len(ts) - 1):
            t0, t1 = ts[i], ts[i + 1]
            if t1 - t0 <= 1e-12:
                continue
            c1 = c(t1)
            if c1 < -1e-6:
                return True
            tm = 0.5 * (t0 + t1)
            cm = c(tm)
            if cm < -1e-6:
                return True
            # quadratic through (t0, prev), (tm, cm), (t1, c1): vertex inside -> check it
            h = 0.5 * (t1 - t0)
            a2 = (prev - 2.0 * cm + c1) / (2.0 * h * h)
            if a2 > 1e-12:
                b1 = (c1 - prev) / (2.0 * h)
                tv = tm - b1 / (2.0 * a2)
                if t0 < tv < t1 and cm - b1 * b1 / (4.0 * a2) < -1e-6:
                    return True
            prev = c1
        return False

    # ------------------------------------------------------------------ public

    def clear(self, a, b, terrain=True):
        """True when the segment a-b touches no prism and (if terrain) no ground."""
        for v in (a[0], a[1], a[2], b[0], b[1], b[2]):
            if v != v or v in (math.inf, -math.inf):
                return True  # garbage in: do not invent a block
        if self.blocker(a, b) is not None:
            return False
        if terrain and self.hmap and self._terrain_blocked(a, b):
            return False
        return True


_default = None
_failed = False


def init(map_dir=None):
    """Load the map once (hello.map.dir). Returns the shared Ray."""
    global _default, _failed
    _default = Ray(map_dir)
    _failed = False
    return _default


def clear(a, b, terrain=True):
    """Module-level line-of-fire test; loads the map on first use. No map: assumes clear."""
    global _default, _failed
    if _default is None:
        if _failed:
            return True
        try:
            _default = Ray()
        except (OSError, ValueError, KeyError):
            _failed = True
            return True
    return _default.clear(a, b, terrain)


def ground(x, y):
    """Ground height at (x, y) from the shared map (0 if the map is not available)."""
    global _default, _failed
    if _default is None and not _failed:
        try:
            _default = Ray()
        except (OSError, ValueError, KeyError):
            _failed = True
    return _default.ground(x, y) if _default is not None else 0.0
