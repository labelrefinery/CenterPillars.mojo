"""Minimal owned float32 tensor (row-major) plus the elementwise ops the detector needs."""


struct Tensor(Copyable, Movable, Writable):
    """Row-major float32 tensor with up to 4 dims."""

    var shape: List[Int]
    var data: List[Float32]

    def __init__(out self, shape: List[Int]):
        var n = 1
        for d in shape:
            n *= d
        self.shape = shape.copy()
        self.data = List[Float32](length=n, fill=0.0)

    def __init__(out self, *, copy: Self):
        self.shape = copy.shape.copy()
        self.data = copy.data.copy()

    def __init__(out self, *, deinit move: Self):
        self.shape = move.shape^
        self.data = move.data^

    def numel(self) -> Int:
        return len(self.data)

    def dim(self, i: Int) -> Int:
        return self.shape[i]

    def rank(self) -> Int:
        return len(self.shape)

    def __getitem__(self, i: Int) -> Float32:
        return self.data[i]

    def __setitem__(mut self, i: Int, v: Float32):
        self.data[i] = v

    def at2(self, i: Int, j: Int) -> Float32:
        return self.data[i * self.shape[1] + j]

    def at3(self, i: Int, j: Int, k: Int) -> Float32:
        return self.data[(i * self.shape[1] + j) * self.shape[2] + k]

    def set2(mut self, i: Int, j: Int, v: Float32):
        self.data[i * self.shape[1] + j] = v

    def set3(mut self, i: Int, j: Int, k: Int, v: Float32):
        self.data[(i * self.shape[1] + j) * self.shape[2] + k] = v

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Tensor(shape=[")
        for i in range(len(self.shape)):
            if i > 0:
                writer.write(", ")
            writer.write(self.shape[i])
        writer.write("], numel=", len(self.data), ")")


def relu_(mut x: Tensor):
    """In-place ``max(x, 0)``."""
    for i in range(x.numel()):
        if x[i] < 0.0:
            x[i] = 0.0


def sigmoid_(mut x: Tensor):
    """In-place logistic sigmoid, computed the numerically stable way torch uses."""
    from std.math import exp

    for i in range(x.numel()):
        var v = x[i]
        if v >= 0.0:
            x[i] = 1.0 / (1.0 + exp(-v))
        else:
            var e = exp(v)
            x[i] = e / (1.0 + e)


def add_(mut x: Tensor, y: Tensor) raises:
    """In-place elementwise add."""
    if x.numel() != y.numel():
        raise Error("add_: shape mismatch")
    for i in range(x.numel()):
        x[i] = x[i] + y[i]


def concat_channels(parts: List[Tensor]) raises -> Tensor:
    """Concatenate rank-3 ``(C_i, H, W)`` tensors along the channel axis.

    Args:
        parts: tensors that must agree on ``H`` and ``W``.

    Returns:
        Tensor of shape ``(sum C_i, H, W)`` -- the Mojo equivalent of
        ``torch.cat(parts, dim=0)`` for a single batch item.
    """
    if len(parts) == 0:
        raise Error("concat_channels: no inputs")
    var h = parts[0].dim(1)
    var w = parts[0].dim(2)
    var total = 0
    for i in range(len(parts)):
        if parts[i].rank() != 3:
            raise Error("concat_channels: inputs must be rank 3 (C, H, W)")
        if parts[i].dim(1) != h or parts[i].dim(2) != w:
            raise Error("concat_channels: spatial size mismatch")
        total += parts[i].dim(0)

    var out = Tensor([total, h, w])
    var off = 0
    for i in range(len(parts)):
        var n = parts[i].numel()
        for j in range(n):
            out[off + j] = parts[i][j]
        off += n
    return out^


def max_abs_diff(a: Tensor, b: Tensor) raises -> Float32:
    """Largest absolute elementwise difference; raises if the sizes differ."""
    if a.numel() != b.numel():
        raise Error(
            "max_abs_diff: size mismatch " + String(a.numel()) + " vs " + String(b.numel())
        )
    var m: Float32 = 0.0
    for i in range(a.numel()):
        var d = a[i] - b[i]
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m
