"""Synthetic LiDAR roofs with known answers, for testing without network access."""
import numpy as np

GROUND, BUILDING, VEG = 2, 6, 5


def _scene(roof_fn, inside_fn, half=30.0, density=4.0, seed=1, classify=True, tree=True):
    rng = np.random.default_rng(seed)
    n = int((2 * half) ** 2 * density)
    xy = rng.uniform(-half, half, size=(n, 2))
    inside = inside_fn(xy[:, 0], xy[:, 1])
    z = np.where(inside, roof_fn(xy[:, 0], xy[:, 1]), 0.0) + rng.normal(0, 0.03, n)
    cls = np.where(inside, BUILDING, GROUND)
    pts = [np.column_stack([xy, z])]
    classes = [cls]
    if tree:  # a bushy tree next to the house
        m = 400
        c = np.array([14.0, 9.0, 5.0])
        t = c + rng.normal(0, [1.8, 1.8, 1.5], size=(m, 3))
        pts.append(t)
        classes.append(np.full(m, VEG))
    xyz = np.vstack(pts) + np.array([500000.0, 3770000.0, 100.0])
    cls = np.concatenate(classes).astype(np.uint8)
    if not classify:
        cls = np.where(cls == GROUND, GROUND, 1).astype(np.uint8)
    return xyz, cls


def hip_roof(L=16.0, W=10.0, pitch=6.0, **kw):
    t = pitch / 12
    inside = lambda x, y: (np.abs(x) <= L / 2) & (np.abs(y) <= W / 2)
    roof = lambda x, y: 3.0 + t * np.minimum(L / 2 - np.abs(x), W / 2 - np.abs(y))
    return _scene(roof, inside, **kw)


def gable_roof(L=12.0, W=8.0, pitch=8.0, **kw):
    t = pitch / 12
    inside = lambda x, y: (np.abs(x) <= L / 2) & (np.abs(y) <= W / 2)
    roof = lambda x, y: 3.0 + t * (W / 2 - np.abs(y))
    return _scene(roof, inside, **kw)


def l_shaped_gable(pitch=6.0, **kw):
    """Two gables meeting in an L: has valleys."""
    t = pitch / 12
    a = lambda x, y: (np.abs(x) <= 8) & (np.abs(y) <= 4)                 # main wing, ridge along x
    b = lambda x, y: (x >= 0) & (x <= 8) & (y >= -12) & (y <= 4)          # side wing, ridge along y at x=4
    inside = lambda x, y: a(x, y) | b(x, y)
    za = lambda x, y: 3.0 + t * (4 - np.abs(y))
    zb = lambda x, y: 3.0 + t * (4 - np.abs(x - 4))
    def roof(x, y):
        z = np.where(a(x, y), za(x, y), -np.inf)
        z = np.maximum(z, np.where(b(x, y), zb(x, y), -np.inf))
        return z
    return _scene(roof, inside, **kw)


def rect(cx, cy, L, W):
    return lambda x, y: (np.abs(x - cx) <= L / 2) & (np.abs(y - cy) <= W / 2)


def property_scene(seed=3, density=4.0):
    """A lot with a house, garage and an old shed, next to a neighbour's house and shed.

    Street is to the north; the address point is at (0, 12) relative to the house centre.
    Returns (xyz, cls, outlines) where outlines are the "current building map" rectangles
    (x0, y0, x1, y1) in local metres, deliberately offset 0.6 m like real map data:
      - includes a 6x8 m ADU and a 8x4 m addition that the LiDAR scan doesn't have
      - leaves out the old shed, which the LiDAR has but was torn down since
    """
    t = 0.5
    hip = lambda x, y: 3.0 + t * np.minimum(8 - np.abs(x), 5 - np.abs(y))
    roofs = [
        (rect(0, 0, 16, 10), hip),                                              # main house
        (rect(0, -15, 7, 6), lambda x, y: 2.5 + 0.5 * (3 - np.abs(y + 15))),    # garage
        (rect(24, 0, 14, 10), lambda x, y: 3.0 + 0.5 * (5 - np.abs(y))),        # neighbour
        (rect(26, -14, 4, 4), lambda x, y: 2.4 + 0 * x),                        # neighbour shed
        (rect(-13, -12, 4, 4), lambda x, y: 2.4 + 0 * x),                       # old shed
    ]
    rng = np.random.default_rng(seed)
    half = 70.0
    n = int((2 * half) ** 2 * density)
    xy = rng.uniform(-half, half, size=(n, 2))
    z = np.zeros(n)
    inside = np.zeros(n, bool)
    for ins, rf in roofs:
        m = ins(xy[:, 0], xy[:, 1])
        z = np.where(m, np.maximum(np.where(inside, z, -np.inf), rf(xy[:, 0], xy[:, 1])), z)
        inside |= m
    z += rng.normal(0, 0.03, n)
    cls = np.where(inside, BUILDING, GROUND).astype(np.uint8)
    xyz = np.column_stack([xy, z]) + np.array([500000.0, 3770000.0, 100.0])
    off = (0.6, -0.4)
    outlines = [
        (-8, -5, 8, 5), (-8, -9, 0, -5),             # house + new addition at the back
        (-3.5, -18, 3.5, -12),                       # garage
        (17, -5, 31, 5), (24, -16, 28, -12),         # neighbour + shed
        (-16, -26, -10, -18),                        # new ADU (not in LiDAR)
    ]
    outlines = [(a + off[0], b + off[1], c + off[0], d + off[1]) for a, b, c, d in outlines]
    return xyz, cls, outlines
