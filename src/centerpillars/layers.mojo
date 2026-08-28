"""Elementary layers: conv2d, nearest upsample and the 3x3 max-pool peak test.

``conv2d`` is the hot loop of the detector (the smoke model spends ~5.8 GMAC per
sweep here), so it is written in the "direct" form: the weight scalar is hoisted
out and the innermost loop walks the *output width* contiguously, which lets the
stride-1 case run on SIMD lanes.
"""

from std.sys.info import simd_width_of

from .tensor import Tensor

comptime LANES = simd_width_of[DType.float32]()
"""SIMD lanes used by the stride-1 convolution fast path."""


def conv2d(x: Tensor, w: Tensor, b: Tensor, stride: Int, pad: Int) raises -> Tensor:
    """2-D convolution (cross-correlation) matching ``torch.nn.Conv2d``.

    Args:
        x: Input of shape ``(C_in, H, W)``.
        w: Weights of shape ``(C_out, C_in, KH, KW)``.
        b: Bias of shape ``(C_out,)``.
        stride: Stride applied on both H and W.
        pad: Zero padding applied on both sides of H and W.

    Returns:
        Tensor of shape ``(C_out, (H + 2*pad - KH)//stride + 1,
        (W + 2*pad - KW)//stride + 1)``.
    """
    if x.rank() != 3:
        raise Error("conv2d: x must be rank 3 (C_in, H, W)")
    if w.rank() != 4:
        raise Error("conv2d: w must be rank 4 (C_out, C_in, KH, KW)")
    if stride < 1:
        raise Error("conv2d: stride must be >= 1")

    var cin = x.dim(0)
    var h = x.dim(1)
    var wd = x.dim(2)
    var cout = w.dim(0)
    var kh = w.dim(2)
    var kw = w.dim(3)
    if w.dim(1) != cin:
        raise Error("conv2d: weight input channels mismatch")
    if b.numel() != cout:
        raise Error("conv2d: bias size mismatch")

    var oh = (h + 2 * pad - kh) // stride + 1
    var ow = (wd + 2 * pad - kw) // stride + 1
    if oh < 1 or ow < 1:
        raise Error("conv2d: empty output")

    var out = Tensor([cout, oh, ow])
    var xp = x.data.unsafe_ptr()
    var op = out.data.unsafe_ptr()
    var wp = w.data.unsafe_ptr()

    for oc in range(cout):
        var plane = oc * oh * ow
        var bias = b[oc]
        for i in range(oh * ow):
            op[unsafe_offset=plane + i] = bias
        for ic in range(cin):
            var xplane = ic * h * wd
            for ky in range(kh):
                for kx in range(kw):
                    var wv = wp[unsafe_offset=((oc * cin + ic) * kh + ky) * kw + kx]
                    if wv == 0.0:
                        continue
                    # Output columns whose source column falls inside the input.
                    var ox0 = 0
                    if pad - kx > 0:
                        ox0 = (pad - kx + stride - 1) // stride
                    var ox1 = (wd + pad - kx + stride - 1) // stride
                    if ox1 > ow:
                        ox1 = ow
                    if ox0 >= ox1:
                        continue
                    for oy in range(oh):
                        var iy = oy * stride + ky - pad
                        if iy < 0 or iy >= h:
                            continue
                        var xrow = xplane + iy * wd + kx - pad
                        var orow = plane + oy * ow
                        if stride == 1:
                            var ox = ox0
                            var wvec = SIMD[DType.float32, LANES](wv)
                            while ox + LANES <= ox1:
                                var acc = op.unsafe_load[width=LANES](orow + ox)
                                var src = xp.unsafe_load[width=LANES](xrow + ox)
                                op.unsafe_store(orow + ox, acc + wvec * src)
                                ox += LANES
                            while ox < ox1:
                                op[unsafe_offset=orow + ox] += wv * xp[unsafe_offset=xrow + ox]
                                ox += 1
                        else:
                            for ox in range(ox0, ox1):
                                op[unsafe_offset=orow + ox] += wv * xp[unsafe_offset=xrow + ox * stride]
    return out^


def conv2d_relu(x: Tensor, w: Tensor, b: Tensor, stride: Int, pad: Int) raises -> Tensor:
    """``relu(conv2d(...))`` -- the ``conv_bn_relu`` unit with the BN folded in."""
    var y = conv2d(x, w, b, stride, pad)
    for i in range(y.numel()):
        if y[i] < 0.0:
            y[i] = 0.0
    return y^


def upsample_nearest_to(x: Tensor, oh: Int, ow: Int) raises -> Tensor:
    """Nearest-neighbour resize matching ``F.interpolate(size=(oh, ow), mode="nearest")``.

    Args:
        x: Input of shape ``(C, H, W)``.
        oh: Output height.
        ow: Output width.

    Returns:
        Tensor of shape ``(C, oh, ow)``; source index is
        ``min(floor(i * H / oh), H - 1)`` with the float scale ``H / oh``.
    """
    if x.rank() != 3:
        raise Error("upsample_nearest_to: x must be rank 3 (C, H, W)")
    if oh < 1 or ow < 1:
        raise Error("upsample_nearest_to: output dims must be >= 1")

    var c = x.dim(0)
    var h = x.dim(1)
    var wd = x.dim(2)
    if h == oh and wd == ow:
        return Tensor(copy=x)
    var scale_y = Float32(h) / Float32(oh)
    var scale_x = Float32(wd) / Float32(ow)

    var src_x = List[Int]()
    for ox in range(ow):
        var sx = Int(Float32(ox) * scale_x)
        if sx > wd - 1:
            sx = wd - 1
        src_x.append(sx)

    var out = Tensor([c, oh, ow])
    for ch in range(c):
        for oy in range(oh):
            var sy = Int(Float32(oy) * scale_y)
            if sy > h - 1:
                sy = h - 1
            for ox in range(ow):
                out.set3(ch, oy, ox, x.at3(ch, sy, src_x[ox]))
    return out^


def maxpool3x3_peaks(x: Tensor) raises -> Tensor:
    """CenterNet peak test: keep cells equal to their 3x3 max, zero the rest.

    Mirrors ``hm * (max_pool2d(hm, 3, stride=1, padding=1) == hm)``. Because the
    pooled value is the max of the very same float32 cells, the equality test is
    exact in both PyTorch and Mojo.

    Args:
        x: Scores of shape ``(K, H, W)``.

    Returns:
        Tensor of shape ``(K, H, W)`` holding ``x`` at local maxima and 0 elsewhere.
    """
    if x.rank() != 3:
        raise Error("maxpool3x3_peaks: x must be rank 3 (K, H, W)")
    var k = x.dim(0)
    var h = x.dim(1)
    var w = x.dim(2)
    var out = Tensor([k, h, w])
    for c in range(k):
        for y in range(h):
            var y0 = y - 1
            if y0 < 0:
                y0 = 0
            var y1 = y + 2
            if y1 > h:
                y1 = h
            for xx in range(w):
                var x0 = xx - 1
                if x0 < 0:
                    x0 = 0
                var x1 = xx + 2
                if x1 > w:
                    x1 = w
                var v = x.at3(c, y, xx)
                var m = v
                for yy in range(y0, y1):
                    for xz in range(x0, x1):
                        var u = x.at3(c, yy, xz)
                        if u > m:
                            m = u
                if m == v:
                    out.set3(c, y, xx, v)
    return out^
