"""Unit tests for the CenterPillars tensor ops, layers, pillar net and decoder."""

from std.math import atan2, exp, log, sin, cos
from std.testing import assert_almost_equal, assert_equal, assert_true, TestSuite

from centerpillars.decode import circle_nms, decode_heatmap
from centerpillars.io import Config
from centerpillars.layers import conv2d, conv2d_relu, maxpool3x3_peaks, upsample_nearest_to
from centerpillars.pillars import pillar_grid
from centerpillars.tensor import Tensor, concat_channels, max_abs_diff, relu_, sigmoid_

comptime TOL: Float64 = 1e-5


def make(shape: List[Int], vals: List[Float32]) raises -> Tensor:
    """Build a tensor of `shape` filled row-major from `vals`."""
    var t = Tensor(shape)
    if t.numel() != len(vals):
        raise Error("make: value count mismatch")
    for i in range(len(vals)):
        t[i] = vals[i]
    return t^


def close(got: Float32, expected: Float32) raises:
    """Assert `got` is within TOL of `expected`."""
    assert_almost_equal(got, expected, atol=TOL, rtol=TOL)


def toy_config() raises -> Config:
    """A 4x4-pillar / 4x4-heatmap, 1-class, 1-stage config over [-2, 2] metres."""
    var v: List[Float32] = [
        -2.0, 2.0, -2.0, 2.0, -1.0, 1.0,  # x/y/z ranges
        1.0, 3.0,                          # pillar_size, pillar_feat_dim
        4.0, 4.0, 1.0,                     # grid_h, grid_w, out_stride
        4.0, 4.0,                          # hm_h, hm_w
        1.0, 0.1, 10.0,                    # num_classes, score_thresh, max_dets
        1.0, 2.0, 2.0, 2.0,                # n_stages, neck/head/neck_out channels
        1.5,                               # nms radius, class 0
        2.0, 1.0, 1.0, 1.0,                # stage channels / layers / stride / neck stride
    ]
    return Config(make([len(v)], v))


# ---------------------------------------------------------------------- conv2d


def test_conv2d_ones_pad1() raises:
    var x = make([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var w = make([1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    var b = make([1], [0.0])
    var y = conv2d(x, w, b, 1, 1)
    assert_equal(y.dim(0), 1)
    assert_equal(y.dim(1), 3)
    assert_equal(y.dim(2), 3)
    var expected: List[Float32] = [
        12.0, 21.0, 16.0,
        27.0, 45.0, 33.0,
        24.0, 39.0, 28.0,
    ]
    for i in range(9):
        close(y[i], expected[i])


def test_conv2d_stride2() raises:
    var x = make([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var w = make([1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    var b = make([1], [0.0])
    var y = conv2d(x, w, b, 2, 1)
    assert_equal(y.dim(1), 2)
    assert_equal(y.dim(2), 2)
    close(y.at3(0, 0, 0), 12.0)
    close(y.at3(0, 0, 1), 16.0)
    close(y.at3(0, 1, 0), 24.0)
    close(y.at3(0, 1, 1), 28.0)


def test_conv2d_1x1_channels() raises:
    var x = make([2, 1, 1], [3.0, 5.0])
    # w[oc, ic] with oc=0 -> [1, 0], oc=1 -> [0, 2]
    var w = make([2, 2, 1, 1], [1.0, 0.0, 0.0, 2.0])
    var b = make([2], [0.5, -1.0])
    var y = conv2d(x, w, b, 1, 0)
    assert_equal(y.dim(0), 2)
    close(y.at3(0, 0, 0), 3.5)
    close(y.at3(1, 0, 0), 9.0)


def test_conv2d_wide_row_matches_scalar_reference() raises:
    """Exercise the SIMD stride-1 fast path against a hand-rolled reference."""
    var wd = 37  # not a multiple of the SIMD width
    var x = Tensor([2, 5, wd])
    for i in range(x.numel()):
        x[i] = Float32((i * 7) % 13) - 6.0
    var w = Tensor([3, 2, 3, 3])
    for i in range(w.numel()):
        w[i] = Float32((i * 5) % 11) * 0.1 - 0.5
    var b = make([3], [0.25, -0.5, 1.0])
    var y = conv2d(x, w, b, 1, 1)
    assert_equal(y.dim(0), 3)
    assert_equal(y.dim(1), 5)
    assert_equal(y.dim(2), wd)

    for oc in range(3):
        for oy in range(5):
            for ox in range(wd):
                var acc = b[oc]
                for ic in range(2):
                    for ky in range(3):
                        var iy = oy + ky - 1
                        if iy < 0 or iy >= 5:
                            continue
                        for kx in range(3):
                            var ix = ox + kx - 1
                            if ix < 0 or ix >= wd:
                                continue
                            var wi = ((oc * 2 + ic) * 3 + ky) * 3 + kx
                            acc += x.at3(ic, iy, ix) * w[wi]
                assert_almost_equal(y.at3(oc, oy, ox), acc, atol=1e-4, rtol=1e-4)


def test_conv2d_relu_clamps() raises:
    var x = make([1, 1, 1], [1.0])
    var w = make([2, 1, 1, 1], [1.0, -1.0])
    var b = make([2], [0.0, 0.0])
    var y = conv2d_relu(x, w, b, 1, 0)
    close(y[0], 1.0)
    close(y[1], 0.0)


# -------------------------------------------------------------------- upsample


def test_upsample_exact_2x() raises:
    var x = make([1, 2, 2], [1.0, 2.0, 3.0, 4.0])
    var y = upsample_nearest_to(x, 4, 4)
    assert_equal(y.dim(1), 4)
    for oy in range(4):
        for ox in range(4):
            close(y.at3(0, oy, ox), x.at3(0, oy // 2, ox // 2))


def test_upsample_non_integer() raises:
    var x = make([1, 3, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    var y = upsample_nearest_to(x, 4, 4)
    # src index for i in 0..3 with scale 3/4: floor(i * 3 / 4) = 0, 0, 1, 2
    var src: List[Int] = [0, 0, 1, 2]
    for oy in range(4):
        for ox in range(4):
            close(y.at3(0, oy, ox), x.at3(0, src[oy], src[ox]))


def test_upsample_identity() raises:
    """A 63x63 stage resized onto a 125x125 neck grid is not an integer scale."""
    var x = make([1, 2, 2], [1.0, 2.0, 3.0, 4.0])
    var y = upsample_nearest_to(x, 2, 2)
    for i in range(4):
        close(y[i], x[i])
    var z = upsample_nearest_to(x, 3, 3)
    var src: List[Int] = [0, 0, 1]
    for oy in range(3):
        for ox in range(3):
            close(z.at3(0, oy, ox), x.at3(0, src[oy], src[ox]))


# ------------------------------------------------------------------ peak test


def test_maxpool3x3_peaks_keeps_only_local_maxima() raises:
    var x = make(
        [1, 3, 3],
        [0.1, 0.2, 0.1,
         0.2, 0.9, 0.2,
         0.1, 0.2, 0.1],
    )
    var p = maxpool3x3_peaks(x)
    close(p.at3(0, 1, 1), 0.9)
    for i in range(9):
        if i != 4:
            close(p[i], 0.0)


def test_maxpool3x3_peaks_border_and_plateau() raises:
    # Corner maxima survive (padding is not a competitor), and an exact plateau
    # keeps every tied cell -- matching torch's `pooled == hm` test.
    var x = make([1, 2, 2], [0.5, 0.5, 0.5, 0.5])
    var p = maxpool3x3_peaks(x)
    for i in range(4):
        close(p[i], 0.5)

    var y = make([1, 1, 3], [0.3, 0.1, 0.7])
    var q = maxpool3x3_peaks(y)
    close(q[0], 0.3)
    close(q[1], 0.0)
    close(q[2], 0.7)


def test_maxpool3x3_peaks_per_channel() raises:
    var x = make([2, 1, 2], [0.4, 0.9, 0.8, 0.2])
    var p = maxpool3x3_peaks(x)
    close(p[0], 0.0)
    close(p[1], 0.9)
    close(p[2], 0.8)
    close(p[3], 0.0)


# ------------------------------------------------------------ tensor utilities


def test_concat_channels() raises:
    var a = make([1, 2, 2], [1.0, 2.0, 3.0, 4.0])
    var b = make([2, 2, 2], [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    var c = concat_channels([a^, b^])
    assert_equal(c.dim(0), 3)
    assert_equal(c.dim(1), 2)
    assert_equal(c.dim(2), 2)
    for i in range(12):
        close(c[i], Float32(i + 1))


def test_concat_channels_rejects_mismatch() raises:
    var a = make([1, 2, 2], [1.0, 2.0, 3.0, 4.0])
    var b = make([1, 1, 2], [5.0, 6.0])
    var raised = False
    try:
        var _c = concat_channels([a^, b^])
    except:
        raised = True
    assert_true(raised, "concat_channels accepted mismatched spatial dims")


def test_sigmoid_and_relu() raises:
    var x = make([4], [0.0, -100.0, 100.0, 1.0])
    sigmoid_(x)
    close(x[0], 0.5)
    close(x[1], 0.0)
    close(x[2], 1.0)
    close(x[3], 1.0 / (1.0 + exp(Float32(-1.0))))

    var y = make([3], [-1.0, 0.0, 2.0])
    relu_(y)
    close(y[0], 0.0)
    close(y[1], 0.0)
    close(y[2], 2.0)


def test_max_abs_diff() raises:
    var a = make([3], [1.0, 2.0, 3.0])
    var b = make([3], [1.0, 2.5, 2.0])
    close(max_abs_diff(a, b), 1.0)


# ------------------------------------------------------------------ circle NMS


def det_rows(xs: List[Float32], ys: List[Float32], scores: List[Float32]) raises -> Tensor:
    """Build an ``(n, 9)`` detection tensor from centres and scores (score desc)."""
    var t = Tensor([len(xs), 9])
    for i in range(len(xs)):
        t.set2(i, 0, xs[i])
        t.set2(i, 1, ys[i])
        t.set2(i, 7, scores[i])
    return t^


def test_circle_nms_suppresses_close_lower_score() raises:
    var boxes = det_rows([0.0, 0.5, 5.0], [0.0, 0.0, 0.0], [0.9, 0.8, 0.7])
    var kept = circle_nms(boxes, [0, 1, 2], 1.0)
    assert_equal(len(kept), 2)
    assert_equal(kept[0], 0)
    assert_equal(kept[1], 2)


def test_circle_nms_keeps_all_when_far() raises:
    var boxes = det_rows([0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [0.9, 0.8, 0.7])
    var kept = circle_nms(boxes, [0, 1, 2], 1.0)
    assert_equal(len(kept), 3)


def test_circle_nms_is_greedy_not_transitive() raises:
    """B suppressed by A does not itself suppress C."""
    var boxes = det_rows([0.0, 0.9, 1.8], [0.0, 0.0, 0.0], [0.9, 0.8, 0.7])
    var kept = circle_nms(boxes, [0, 1, 2], 1.0)
    assert_equal(len(kept), 2)
    assert_equal(kept[0], 0)
    assert_equal(kept[1], 2)


def test_circle_nms_boundary_inclusive() raises:
    var boxes = det_rows([0.0, 1.0], [0.0, 0.0], [0.9, 0.8])
    assert_equal(len(circle_nms(boxes, [0, 1], 1.0)), 1)


# --------------------------------------------------------------------- config


def test_config_decode() raises:
    var cfg = toy_config()
    assert_equal(cfg.grid_h, 4)
    assert_equal(cfg.grid_w, 4)
    assert_equal(cfg.num_classes, 1)
    assert_equal(cfg.n_stages, 1)
    assert_equal(cfg.stage_channels[0], 2)
    assert_equal(cfg.neck_conv_stride[0], 1)
    close(cfg.cell(), 1.0)
    close(cfg.nms_radius[0], 1.5)


# ---------------------------------------------------------------- pillar grid


def test_pillar_grid_single_point() raises:
    var cfg = toy_config()
    # One point at (0.75, -0.75, 0.5) with intensity 2 -> pillar col 2, row 1
    # whose centre is (0.5, -0.5), so the in-pillar x offset is +0.25.
    var points = make([1, 4], [0.75, -0.75, 0.5, 2.0])
    var mask = make([1], [1.0])
    # Identity-ish point net: channel c reads feature c.
    var w = Tensor([3, 9])
    w.set2(0, 0, 1.0)  # x
    w.set2(1, 7, 1.0)  # x - pillar centre x
    w.set2(2, 4, 1.0)  # x - pillar mean x (0 for a lone point)
    var b = make([3], [0.0, 0.0, 0.0])
    var grid = pillar_grid(points, mask, w, b, cfg)
    assert_equal(grid.dim(0), 3)
    assert_equal(grid.dim(1), 4)
    assert_equal(grid.dim(2), 4)
    close(grid.at3(0, 1, 2), 0.75)          # relu(x)
    close(grid.at3(1, 1, 2), 0.25)          # x - pillar centre x
    close(grid.at3(2, 1, 2), 0.0)           # lone point: x equals the pillar mean
    # Everything else is exactly zero.
    var nonzero = 0
    for i in range(grid.numel()):
        if grid[i] != 0.0:
            nonzero += 1
    assert_equal(nonzero, 2)


def test_pillar_grid_masks_and_range() raises:
    var cfg = toy_config()
    var points = make(
        [4, 4],
        [
            0.25, 0.25, 0.0, 1.0,    # valid
            0.25, 0.25, 0.0, 1.0,    # masked out
            50.0, 0.0, 0.0, 1.0,     # out of the x range
            0.25, 0.25, 5.0, 1.0,    # out of the z range
        ],
    )
    var mask = make([4], [1.0, 0.0, 1.0, 1.0])
    var w = Tensor([1, 9])
    w.set2(0, 3, 1.0)  # read the intensity
    var b = make([1], [0.0])
    var grid = pillar_grid(points, mask, w, b, cfg)
    var nonzero = 0
    for i in range(grid.numel()):
        if grid[i] != 0.0:
            nonzero += 1
    assert_equal(nonzero, 1)
    close(grid.at3(0, 2, 2), 1.0)


def test_pillar_grid_pools_max_and_means() raises:
    var cfg = toy_config()
    # Two points in the same pillar: mean x = 0.5, so offsets are -0.25 / +0.25.
    var points = make([2, 4], [0.25, 0.25, 0.0, 1.0, 0.75, 0.25, 0.0, 3.0])
    var mask = make([2], [1.0, 1.0])
    var w = Tensor([2, 9])
    w.set2(0, 3, 1.0)  # intensity -> max pooling picks 3.0
    w.set2(1, 4, 1.0)  # x - mean x -> max is +0.25
    var b = make([2], [0.0, 0.0])
    var grid = pillar_grid(points, mask, w, b, cfg)
    close(grid.at3(0, 2, 2), 3.0)
    close(grid.at3(1, 2, 2), 0.25)


# ------------------------------------------------------------------- decoding


def test_decode_round_trip() raises:
    var cfg = toy_config()
    var hm = Tensor([1, 4, 4])
    hm.set3(0, 2, 3, 0.8)
    hm.set3(0, 0, 0, 0.05)  # below score_thresh

    var yaw: Float32 = 0.7
    var reg = Tensor([8, 4, 4])
    reg.set3(0, 2, 3, 0.25)          # dx
    reg.set3(1, 2, 3, -0.5)          # dy
    reg.set3(2, 2, 3, 1.25)          # z
    reg.set3(3, 2, 3, log(Float32(4.0)))
    reg.set3(4, 2, 3, log(Float32(2.0)))
    reg.set3(5, 2, 3, log(Float32(1.5)))
    reg.set3(6, 2, 3, sin(yaw))
    reg.set3(7, 2, 3, cos(yaw))

    var boxes = decode_heatmap(hm, reg, cfg)
    assert_equal(boxes.dim(0), 1)
    assert_equal(boxes.dim(1), 9)
    close(boxes.at2(0, 0), (3.0 + 0.25) * 1.0 - 2.0)
    close(boxes.at2(0, 1), (2.0 - 0.5) * 1.0 - 2.0)
    close(boxes.at2(0, 2), 1.25)
    close(boxes.at2(0, 3), 4.0)
    close(boxes.at2(0, 4), 2.0)
    close(boxes.at2(0, 5), 1.5)
    close(boxes.at2(0, 6), yaw)
    close(boxes.at2(0, 7), 0.8)
    close(boxes.at2(0, 8), 0.0)


def test_decode_sorts_by_score_and_applies_nms() raises:
    var cfg = toy_config()
    # Three isolated peaks; the 1.5 m NMS radius drops the one 1 m from the best.
    var hm = Tensor([1, 4, 4])
    hm.set3(0, 0, 0, 0.5)
    hm.set3(0, 0, 2, 0.9)
    hm.set3(0, 2, 2, 0.7)
    var reg = Tensor([8, 4, 4])
    for c in range(3):
        for i in range(16):
            reg[(3 + c) * 16 + i] = 0.0  # log size 0 -> size 1
    var boxes = decode_heatmap(hm, reg, cfg)
    # (0,2) at x=0.0,y=-2.0 kills nothing within 1.5 m except... (2,2) is 2 m away.
    assert_equal(boxes.dim(0), 3)
    assert_true(boxes.at2(0, 7) >= boxes.at2(1, 7), "decoded boxes are not score-sorted")
    assert_true(boxes.at2(1, 7) >= boxes.at2(2, 7), "decoded boxes are not score-sorted")
    close(boxes.at2(0, 7), 0.9)


def test_decode_empty_below_threshold() raises:
    var cfg = toy_config()
    var hm = Tensor([1, 4, 4])
    for i in range(16):
        hm[i] = 0.05
    var reg = Tensor([8, 4, 4])
    assert_equal(decode_heatmap(hm, reg, cfg).dim(0), 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
