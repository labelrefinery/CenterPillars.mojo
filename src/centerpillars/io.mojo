"""Reader for the LFT1 tensor container written by CenterPillars.py's export_mojo.py.

Layout (little-endian):
    b"LFT1" | u32 n_tensors | per tensor:
        u32 name_len | name utf8 | u32 ndim | u32 shape[ndim] | f32 data (C order)

The ``__config__`` tensor of a weights file makes the network self-describing:
20 fixed scalars, then ``num_classes`` circle-NMS radii, then four blocks of
``n_stages`` values (channels, layers, stride, neck conv stride).
"""

from std.memory import bitcast

from .tensor import Tensor

comptime N_FIXED = 20
"""Number of fixed scalars at the head of ``__config__``."""


struct Config(Copyable, Movable, Writable):
    """Detector geometry, backbone structure and post-processing thresholds."""

    var x_min: Float32
    var x_max: Float32
    var y_min: Float32
    var y_max: Float32
    var z_min: Float32
    var z_max: Float32
    var pillar_size: Float32
    var pillar_feat_dim: Int
    var grid_h: Int
    var grid_w: Int
    var out_stride: Int
    var hm_h: Int
    var hm_w: Int
    var num_classes: Int
    var score_thresh: Float32
    var max_dets: Int
    var n_stages: Int
    var neck_channels: Int
    var head_channels: Int
    var neck_out_channels: Int
    var nms_radius: List[Float32]
    var stage_channels: List[Int]
    var stage_layers: List[Int]
    var stage_stride: List[Int]
    var neck_conv_stride: List[Int]

    def __init__(out self, c: Tensor) raises:
        """Decode a ``__config__`` tensor."""
        if c.numel() < N_FIXED:
            raise Error("bad __config__ tensor: too short")
        self.x_min = c[0]
        self.x_max = c[1]
        self.y_min = c[2]
        self.y_max = c[3]
        self.z_min = c[4]
        self.z_max = c[5]
        self.pillar_size = c[6]
        self.pillar_feat_dim = Int(c[7])
        self.grid_h = Int(c[8])
        self.grid_w = Int(c[9])
        self.out_stride = Int(c[10])
        self.hm_h = Int(c[11])
        self.hm_w = Int(c[12])
        self.num_classes = Int(c[13])
        self.score_thresh = c[14]
        self.max_dets = Int(c[15])
        self.n_stages = Int(c[16])
        self.neck_channels = Int(c[17])
        self.head_channels = Int(c[18])
        self.neck_out_channels = Int(c[19])

        var k = self.num_classes
        var s = self.n_stages
        if c.numel() != N_FIXED + k + 4 * s:
            raise Error("bad __config__ tensor: unexpected length")
        self.nms_radius = List[Float32]()
        for i in range(k):
            self.nms_radius.append(c[N_FIXED + i])
        var base = N_FIXED + k
        self.stage_channels = List[Int]()
        self.stage_layers = List[Int]()
        self.stage_stride = List[Int]()
        self.neck_conv_stride = List[Int]()
        for i in range(s):
            self.stage_channels.append(Int(c[base + i]))
            self.stage_layers.append(Int(c[base + s + i]))
            self.stage_stride.append(Int(c[base + 2 * s + i]))
            self.neck_conv_stride.append(Int(c[base + 3 * s + i]))

    def __init__(out self, *, copy: Self):
        self.x_min = copy.x_min
        self.x_max = copy.x_max
        self.y_min = copy.y_min
        self.y_max = copy.y_max
        self.z_min = copy.z_min
        self.z_max = copy.z_max
        self.pillar_size = copy.pillar_size
        self.pillar_feat_dim = copy.pillar_feat_dim
        self.grid_h = copy.grid_h
        self.grid_w = copy.grid_w
        self.out_stride = copy.out_stride
        self.hm_h = copy.hm_h
        self.hm_w = copy.hm_w
        self.num_classes = copy.num_classes
        self.score_thresh = copy.score_thresh
        self.max_dets = copy.max_dets
        self.n_stages = copy.n_stages
        self.neck_channels = copy.neck_channels
        self.head_channels = copy.head_channels
        self.neck_out_channels = copy.neck_out_channels
        self.nms_radius = copy.nms_radius.copy()
        self.stage_channels = copy.stage_channels.copy()
        self.stage_layers = copy.stage_layers.copy()
        self.stage_stride = copy.stage_stride.copy()
        self.neck_conv_stride = copy.neck_conv_stride.copy()

    def __init__(out self, *, deinit move: Self):
        self.x_min = move.x_min
        self.x_max = move.x_max
        self.y_min = move.y_min
        self.y_max = move.y_max
        self.z_min = move.z_min
        self.z_max = move.z_max
        self.pillar_size = move.pillar_size
        self.pillar_feat_dim = move.pillar_feat_dim
        self.grid_h = move.grid_h
        self.grid_w = move.grid_w
        self.out_stride = move.out_stride
        self.hm_h = move.hm_h
        self.hm_w = move.hm_w
        self.num_classes = move.num_classes
        self.score_thresh = move.score_thresh
        self.max_dets = move.max_dets
        self.n_stages = move.n_stages
        self.neck_channels = move.neck_channels
        self.head_channels = move.head_channels
        self.neck_out_channels = move.neck_out_channels
        self.nms_radius = move.nms_radius^
        self.stage_channels = move.stage_channels^
        self.stage_layers = move.stage_layers^
        self.stage_stride = move.stage_stride^
        self.neck_conv_stride = move.neck_conv_stride^

    def cell(self) -> Float32:
        """Metres per heatmap cell."""
        return self.pillar_size * Float32(self.out_stride)

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "Config(pillars ", self.grid_h, "x", self.grid_w,
            " heatmap ", self.hm_h, "x", self.hm_w,
            " classes ", self.num_classes,
            " stages ", self.n_stages,
            " cell ", self.cell(), "m)",
        )


def _u32(bytes: List[UInt8], off: Int) -> UInt32:
    var v: UInt32 = 0
    v |= UInt32(bytes[off])
    v |= UInt32(bytes[off + 1]) << 8
    v |= UInt32(bytes[off + 2]) << 16
    v |= UInt32(bytes[off + 3]) << 24
    return v


def _f32(bytes: List[UInt8], off: Int) -> Float32:
    return bitcast[DType.float32, 1](SIMD[DType.uint32, 1](_u32(bytes, off)))[0]


def load_lft(path: String) raises -> Dict[String, Tensor]:
    """Parse an LFT1 file into a name -> Tensor dictionary."""
    var f = open(path, "r")
    var bytes = f.read_bytes()
    f.close()
    if len(bytes) < 8 or bytes[0] != 76 or bytes[1] != 70 or bytes[2] != 84 or bytes[3] != 49:
        raise Error("not an LFT1 file: " + path)

    var out = Dict[String, Tensor]()
    var n_tensors = Int(_u32(bytes, 4))
    var off = 8
    for _ in range(n_tensors):
        var name_len = Int(_u32(bytes, off))
        off += 4
        var name = String("")
        for i in range(name_len):
            name += chr(Int(bytes[off + i]))
        off += name_len

        var ndim = Int(_u32(bytes, off))
        off += 4
        var shape = List[Int]()
        var numel = 1
        for _ in range(ndim):
            var d = Int(_u32(bytes, off))
            off += 4
            shape.append(d)
            numel *= d
        if ndim == 0:
            shape.append(1)

        var t = Tensor(shape)
        for i in range(numel):
            t[i] = _f32(bytes, off)
            off += 4
        out[name] = t^
    if off != len(bytes):
        raise Error("trailing bytes in " + path)
    return out^


def decode_config(tensors: Dict[String, Tensor]) raises -> Config:
    """Build a Config from the ``__config__`` tensor of a weights file."""
    if "__config__" not in tensors:
        raise Error("weights file has no __config__ tensor")
    return Config(tensors["__config__"])
