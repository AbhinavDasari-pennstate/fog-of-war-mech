"""Tests for ray.py. Run: python test_ray.py"""
import math
import os
import random
import sys
import time
import unittest
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ray  # noqa: E402

try:
    REAL = ray.Ray()
except FileNotFoundError:
    REAL = None

need_map = unittest.skipIf(REAL is None, "map files not found (set MECHFIGHT_MAP_DIR)")


def box(x0, y0, x1, y1, base, top, kind="building", holes=None):
    p = {"kind": kind, "base": base, "top": top, "ring": [x0, y0, x1, y0, x1, y1, x0, y1]}
    if holes:
        p["holes"] = holes
    return p


def flat_world(*prisms):
    return Ray_from({"prisms": list(prisms)})


def Ray_from(world, terrain=None):
    return ray.Ray(world=world, terrain=terrain)


def inside_point(r, p):
    """Some point strictly inside prism dict p (scan its bbox)."""
    x0, y0, x1, y1 = p["bbox"]
    n = 40
    for i in range(1, n):
        for j in range(1, n):
            x, y = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * j / n
            if r._inside(p, x, y):
                return x, y
    return None


class TestSynthetic(unittest.TestCase):
    def test_through_building_blocked_over_it_clear(self):
        r = flat_world(box(0, 0, 20, 20, 0, 15))
        self.assertFalse(r.clear((-30, 10, 5), (50, 10, 5)))
        self.assertTrue(r.clear((-30, 10, 20), (50, 10, 20)))
        self.assertTrue(r.clear((-30, 40, 5), (50, 40, 5)))

    def test_slanted_ray_clips_the_slab(self):
        r = flat_world(box(0, 0, 20, 20, 0, 15))
        self.assertFalse(r.clear((-10, 10, 2), (30, 10, 40)))      # enters below the top
        self.assertTrue(r.clear((-10, 10, 14), (-1, 10, 40)))      # climbs away before the wall

    def test_low_wall_blocks_muzzle_height_ray_but_not_eye_height(self):
        wall = box(10, -10, 11, 10, 0, 7.5)
        r = flat_world(wall)
        self.assertFalse(r.clear((0, 0, 7.0), (30, 0, 7.0)))   # muzzle, feet at 0
        self.assertTrue(r.clear((0, 0, 8.0), (30, 0, 8.0)))    # eye clears it

    def test_hole_in_ring_is_clear_solid_part_blocks(self):
        hole = [8, 8, 12, 8, 12, 12, 8, 12]
        r = flat_world(box(0, 0, 20, 20, 0, 15, holes=[hole]))
        self.assertTrue(r.clear((9, 10, 5), (11, 10, 5)))      # inside the hole
        self.assertFalse(r.clear((10, 10, 5), (30, 10, 5)))    # leaves the hole through the wall
        self.assertFalse(r.clear((2, 10, 5), (4, 10, 5)))      # inside the solid part
        self.assertTrue(r.clear((10, 10, 20), (10, 10, 30)))

    def test_under_bridge_clear_through_deck_blocked(self):
        r = flat_world(box(0, 0, 30, 6, 8.0, 10.0, kind="bridge"))
        self.assertTrue(r.clear((15, -20, 3), (15, 30, 3)))     # walk under the deck
        self.assertFalse(r.clear((15, -20, 9), (15, 30, 9)))    # through the deck
        self.assertTrue(r.clear((15, -20, 12), (15, 30, 12)))   # over it

    def test_concave_ring_notch_is_clear(self):
        # an L shape: the notch (10..20, 10..20) is open
        ring = [0, 0, 20, 0, 20, 10, 10, 10, 10, 20, 0, 20]
        r = flat_world({"kind": "building", "base": 0, "top": 10, "ring": ring})
        self.assertTrue(r.clear((12, 18, 5), (19, 12, 5)))
        self.assertFalse(r.clear((12, 18, 5), (5, 12, 5)))
        self.assertFalse(r.clear((25, 15, 5), (2, 15, 5)))      # reaches the arm of the L

    def test_vertical_and_degenerate_segments(self):
        r = flat_world(box(0, 0, 20, 20, 0, 15))
        self.assertFalse(r.clear((10, 10, 20), (10, 10, 5)))    # straight down into the roof
        self.assertTrue(r.clear((50, 50, 20), (50, 50, 5)))
        self.assertTrue(r.clear((50, 50, 5), (50, 50, 5)))      # zero length

    def test_nan_input_does_not_raise_or_invent_a_block(self):
        r = flat_world(box(0, 0, 20, 20, 0, 15))
        self.assertTrue(r.clear((float("nan"), 0, 0), (1, 1, 1)))
        self.assertTrue(r.clear((0, 0, 0), (float("inf"), 1, 1)))

    def test_terrain_ridge_blocks_low_ray(self):
        nx = ny = 41
        arr = array("h", [0] * (nx * ny))
        for j in range(ny):
            for i in range(nx):
                if 18 <= i <= 22:
                    arr[j * nx + i] = 1000  # a 10 m ridge, 5 m wide, along y
        meta = {"minX": 0.0, "minY": 0.0, "step": 1.0, "nx": nx, "ny": ny, "scale": 0.01}
        r = Ray_from({"prisms": []}, (arr, meta))
        self.assertFalse(r.clear((2, 20, 4), (38, 20, 4)))
        self.assertTrue(r.clear((2, 20, 4), (38, 20, 4), terrain=False))
        self.assertTrue(r.clear((2, 20, 14), (38, 20, 14)))
        self.assertTrue(r.clear((2, 20, 4), (15, 30, 4)))       # stops before the ridge
        self.assertAlmostEqual(r.ground(20, 20), 10.0, places=6)
        self.assertAlmostEqual(r.ground(-5, 20), 0.0, places=6)  # clamped outside the grid


@need_map
class TestRealMap(unittest.TestCase):
    def test_through_a_real_building_blocked_over_it_clear(self):
        p = next(q for q in REAL.prisms if q["kind"] == "building" and q["top"] - q["base"] > 12 and not q["holes"])
        c = inside_point(REAL, p)
        self.assertIsNotNone(c)
        mid = 0.5 * (p["base"] + p["top"])
        a, b = (c[0] - 80, c[1], mid), (c[0] + 80, c[1], mid)
        self.assertFalse(REAL.clear(a, b, terrain=False))
        self.assertFalse(REAL.clear((c[0], c[1], p["top"] + 30), (c[0], c[1], mid), terrain=False))
        self.assertTrue(REAL.clear((c[0] - 80, c[1], p["top"] + 100), (c[0] + 80, c[1], p["top"] + 100)))

    def test_open_ground_is_clear(self):
        random.seed(7)
        found = 0
        for _ in range(4000):
            x, y = random.uniform(-400, 450), random.uniform(-250, 330)
            if all(not (x - 40 < p["bbox"][2] and x + 40 > p["bbox"][0] and y - 40 < p["bbox"][3]
                        and y + 40 > p["bbox"][1]) for p in REAL.prisms):
                g0, g1 = REAL.ground(x, y), REAL.ground(x + 20, y)
                if abs(g1 - g0) < 2.0:
                    self.assertTrue(REAL.clear((x, y, g0 + 7), (x + 20, y, g1 + 4.5)))
                    found += 1
                    if found == 5:
                        break
        self.assertGreaterEqual(found, 3)

    def test_under_a_real_bridge_is_clear_through_the_deck_blocked(self):
        done = 0
        for p in REAL.prisms:
            if p["kind"] != "bridge":
                continue
            c = inside_point(REAL, p)
            if c is None:
                continue
            g = REAL.ground(*c)
            if p["base"] - g < 3.0:
                continue
            x0, y0, x1, y1 = p["bbox"]
            # cross the deck along its short side so the segment leaves the footprint
            ang = math.atan2(y1 - y0, x1 - x0) + math.pi / 2
            dx, dy = 10 * math.cos(ang), 10 * math.sin(ang)
            deck = 0.5 * (p["base"] + p["top"])
            self.assertFalse(REAL.clear((c[0] - dx, c[1] - dy, deck), (c[0] + dx, c[1] + dy, deck), terrain=False))
            under = g + 0.5 * (p["base"] - g)
            a, b = (c[0] - dx, c[1] - dy, under), (c[0] + dx, c[1] + dy, under)
            if REAL.blocker(a, b) is None:
                self.assertTrue(REAL.clear(a, b, terrain=False))
                done += 1
        self.assertGreaterEqual(done, 1)

    def test_hole_in_a_real_ring_is_clear(self):
        done = 0
        for p in REAL.prisms:
            if not p["holes"]:
                continue
            hole = p["holes"][0]
            pts = [(e[0], e[1]) for e in hole]
            cx, cy = sum(q[0] for q in pts) / len(pts), sum(q[1] for q in pts) / len(pts)
            if not REAL._inside_edges(hole, cx, cy):
                continue
            # the shortest step toward each vertex stays inside a star-shaped hole
            tx, ty = cx + 0.4 * (pts[0][0] - cx), cy + 0.4 * (pts[0][1] - cy)
            if not REAL._inside_edges(hole, tx, ty):
                continue
            z = 0.5 * (p["base"] + p["top"])
            others = [q for q in REAL.prisms if q is not p and REAL._inside(q, cx, cy) and q["base"] <= z <= q["top"]]
            if others or REAL._inside(p, cx, cy) or REAL._inside(p, tx, ty):
                continue
            self.assertTrue(REAL.clear((cx, cy, z), (tx, ty, z), terrain=False))
            # the same level, from the hole centre straight out through the solid ring: blocked
            ex = max(q[0] for q in pts) + 0.0
            ok = False
            for ang in range(0, 360, 15):
                a = math.radians(ang)
                far = (cx + 400 * math.cos(a), cy + 400 * math.sin(a), z)
                if REAL.blocker((cx, cy, z), far) is p:
                    ok = True
                    break
            self.assertTrue(ok)
            done += 1
            if done >= 3:
                break
        self.assertGreaterEqual(done, 1)

    def test_sharp_ridge_dip_is_blocked(self):
        # this segment dips about 1 m under the ground near (-362.0, 107.55); a 1 m sampling step missed it
        a, b = (-247.35, 139.86, -8.53), (-367.55, 105.99, 49.59)
        self.assertFalse(REAL.clear(a, b))
        self.assertTrue(REAL.clear(a, b, terrain=False))

    def test_terrain_matches_dense_sampling(self):
        """Exact terrain test vs sampling every 0.02 m: no segment that dips > 5 cm below ground is called clear."""
        random.seed(13)
        missed = false_block = 0
        for _ in range(300):
            x, y = random.uniform(-500, 540), random.uniform(-320, 380)
            ang, L = random.uniform(0, 6.283), random.uniform(5, 150)
            a = (x, y, REAL.ground(x, y) + random.uniform(-1, 8))
            bx, by = x + L * math.cos(ang), y + L * math.sin(ang)
            b = (bx, by, REAL.ground(bx, by) + random.uniform(-1, 8))
            n = int(L * 50) + 2
            dip = min(a[2] + (b[2] - a[2]) * k / n - REAL.ground(a[0] + (bx - a[0]) * k / n, a[1] + (by - a[1]) * k / n)
                      for k in range(n + 1))
            got = REAL._terrain_blocked(a, b)
            if dip < -0.05 and not got:
                missed += 1
            if dip > 0.05 and got:
                false_block += 1
        self.assertEqual((missed, false_block), (0, 0))

    def test_matches_dense_sampling(self):
        """Cross-check against brute force: sample each segment every 0.1 m against nearby prisms."""
        random.seed(11)
        bad = 0
        total = 0
        for _ in range(400):
            x, y, z = random.uniform(-450, 500), random.uniform(-300, 350), random.uniform(0, 30)
            ang, L = random.uniform(0, 6.283), random.uniform(5, 60)
            a = (x, y, z)
            b = (x + L * math.cos(ang), y + L * math.sin(ang), z + random.uniform(-15, 15))
            lx, hx = min(a[0], b[0]), max(a[0], b[0])
            ly, hy = min(a[1], b[1]), max(a[1], b[1])
            near = [p for p in REAL.prisms if not (hx < p["bbox"][0] or lx > p["bbox"][2]
                                                   or hy < p["bbox"][1] or ly > p["bbox"][3])]
            n = int(L * 10) + 2
            hit = False
            for k in range(n + 1):
                f = k / n
                px, py, pz = a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f
                for p in near:
                    if p["base"] <= pz <= p["top"] and REAL._inside(p, px, py):
                        hit = True
                        break
                if hit:
                    break
            total += 1
            if hit != (not REAL.clear(a, b, terrain=False)):
                bad += 1
        # sampling can miss a corner clipped for under 0.1 m: allow a rare disagreement
        self.assertLessEqual(bad, 2, "%d of %d segments disagree with dense sampling" % (bad, total))

    def test_perf_under_point_three_ms(self):
        random.seed(3)
        segs = []
        for _ in range(1000):
            x, y, z = random.uniform(-450, 500), random.uniform(-300, 350), random.uniform(0, 12)
            ang, L = random.uniform(0, 6.283), random.uniform(5, 200)
            segs.append(((x, y, z), (x + L * math.cos(ang), y + L * math.sin(ang), random.uniform(0, 12))))
        t0 = time.perf_counter()
        for a, b in segs:
            REAL.clear(a, b)
        avg_ms = (time.perf_counter() - t0) / len(segs) * 1000.0
        print("ray.clear average over 1000 random segments: %.3f ms" % avg_ms)
        self.assertLess(avg_ms, 0.3)

    def test_module_level_clear_matches_instance(self):
        ray.init()
        a, b = (0.0, 0.0, 30.0), (100.0, 50.0, 30.0)
        self.assertEqual(ray.clear(a, b), REAL.clear(a, b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
