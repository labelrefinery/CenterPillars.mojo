"""PointPillars feature net: raw points -> BEV pillar pseudo-image.

Mirrors ``PillarFeatureNet.forward`` from CenterPillars.py. Every in-range point
is lifted to the 9-D PointPillars feature

    (x, y, z, intensity, x - xm, y - ym, z - zm, x - xp, y - yp)

where ``(xm, ym, zm)`` is the mean of the points sharing the pillar and
``(xp, yp)`` the pillar centre, run through ``Linear(9 -> C)`` (with the
BatchNorm1d folded in at export) + ReLU, and max-pooled per pillar. There is no
cap on points per pillar, so this is exactly two passes: pillar sums, then
features + max. Empty pillars stay exactly zero, which is valid because the ReLU
makes every pooled feature non-negative.
"""

from std.math import floor

from .io import Config
from .tensor import Tensor

comptime POINT_FEAT_DIM = 9
"""Per-point input features of the pillar net."""


def pillar_grid(
    points: Tensor, points_mask: Tensor, w: Tensor, b: Tensor, cfg: Config
) raises -> Tensor:
    """Build the pillar pseudo-image ``(C, grid_h, grid_w)`` for one sweep.

    Args:
        points: ``(N, 4)`` lidar points ``(x, y, z, intensity)`` in the ego frame.
        points_mask: ``(N,)`` 0/1 validity mask (padding rows are 0).
        w: Folded pillar-net weight ``(C, 9)`` in torch Linear layout.
        b: Folded pillar-net bias ``(C,)``.
        cfg: Detector config (range, pillar size, grid dims).
    """
    if points.rank() != 2 or points.dim(1) != 4:
        raise Error("pillar_grid: points must be (N, 4)")
    if w.rank() != 2 or w.dim(1) != POINT_FEAT_DIM:
        raise Error("pillar_grid: weight must be (C, 9)")

    var n = points.dim(0)
    var h = cfg.grid_h
    var wd = cfg.grid_w
    var c_out = w.dim(0)
    if b.numel() != c_out:
        raise Error("pillar_grid: bias size mismatch")

    var n_pillars = h * wd
    var sums = List[Float32](length=n_pillars * 3, fill=0.0)
    var counts = List[Float32](length=n_pillars, fill=0.0)
    var cell_of = List[Int](length=n, fill=-1)

    # Pass 1: assign points to pillars and accumulate xyz sums.
    for p in range(n):
        if points_mask[p] < 0.5:
            continue
        var px = points.at2(p, 0)
        var py = points.at2(p, 1)
        var pz = points.at2(p, 2)
        if pz < cfg.z_min or pz > cfg.z_max:
            continue
        var fcol = floor((px - cfg.x_min) / cfg.pillar_size)
        var frow = floor((py - cfg.y_min) / cfg.pillar_size)
        if fcol < 0.0 or fcol >= Float32(wd) or frow < 0.0 or frow >= Float32(h):
            continue
        var cell = Int(frow) * wd + Int(fcol)
        cell_of[p] = cell
        sums[cell * 3] += px
        sums[cell * 3 + 1] += py
        sums[cell * 3 + 2] += pz
        counts[cell] += 1.0

    # Pass 2: build features, apply the point net, max-pool into the grid.
    var grid = Tensor([c_out, h, wd])
    var wp = w.data.unsafe_ptr()
    var gp = grid.data.unsafe_ptr()
    for p in range(n):
        var cell = cell_of[p]
        if cell < 0:
            continue
        var row = cell // wd
        var col = cell - row * wd
        var cnt = counts[cell]
        var px = points.at2(p, 0)
        var py = points.at2(p, 1)
        var pz = points.at2(p, 2)
        var xc = cfg.x_min + (Float32(col) + 0.5) * cfg.pillar_size
        var yc = cfg.y_min + (Float32(row) + 0.5) * cfg.pillar_size

        var feat = SIMD[DType.float32, 16](0.0)
        feat[0] = px
        feat[1] = py
        feat[2] = pz
        feat[3] = points.at2(p, 3)
        feat[4] = px - sums[cell * 3] / cnt
        feat[5] = py - sums[cell * 3 + 1] / cnt
        feat[6] = pz - sums[cell * 3 + 2] / cnt
        feat[7] = px - xc
        feat[8] = py - yc

        for c in range(c_out):
            var acc = b[c]
            var base = c * POINT_FEAT_DIM
            for k in range(POINT_FEAT_DIM):
                acc += wp[unsafe_offset=base + k] * feat[k]
            if acc < 0.0:
                acc = 0.0
            var idx = (c * h + row) * wd + col
            if acc > gp[unsafe_offset=idx]:
                gp[unsafe_offset=idx] = acc
    return grid^
