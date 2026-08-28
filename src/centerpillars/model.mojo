"""Full CenterPillars forward pass (inference only) from exported LFT1 weights.

Faithfully mirrors CenterPillars.py's eval-mode forward: pillar feature net ->
SECOND-style BEV stages -> nearest-upsample + concat neck -> CenterPoint head.
Every BatchNorm was folded into the preceding conv/linear at export time, so
every layer here is a plain conv (or linear) with a bias.
"""

from .decode import decode_heatmap
from .io import Config, decode_config, load_lft
from .layers import conv2d, conv2d_relu, upsample_nearest_to
from .pillars import pillar_grid
from .tensor import Tensor, concat_channels, sigmoid_


struct CenterPillarsModel(Movable):
    """Weight store + forward implementation."""

    var tensors: Dict[String, Tensor]
    var cfg: Config

    def __init__(out self, weights_path: String) raises:
        self.tensors = load_lft(weights_path)
        self.cfg = decode_config(self.tensors)

    def w(self, name: String) raises -> Tensor:
        if name not in self.tensors:
            raise Error("missing weight tensor: " + name)
        return self.tensors[name].copy()

    def backbone(self, grid: Tensor) raises -> Tensor:
        """Pillar pseudo-image ``(C, gh, gw)`` -> fused BEV features ``(neck_out, hm_h, hm_w)``."""
        var feats = List[Tensor]()
        var x = grid.copy()
        for si in range(self.cfg.n_stages):
            for bi in range(self.cfg.stage_layers[si]):
                var stride = 1
                if bi == 0:
                    stride = self.cfg.stage_stride[si]
                var p = "s" + String(si) + "b" + String(bi)
                x = conv2d_relu(x, self.w(p + ".w"), self.w(p + ".b"), stride, 1)
            feats.append(x.copy())

        var outs = List[Tensor]()
        for si in range(self.cfg.n_stages):
            var p = "neck" + String(si)
            var s = self.cfg.neck_conv_stride[si]
            if s == 1:
                # Coarser (or equal) stage: nearest-resize onto the heatmap grid
                # first -- stride-2 convs round up, so sizes are not exact
                # multiples and the target size must be given explicitly.
                var up = upsample_nearest_to(feats[si], self.cfg.hm_h, self.cfg.hm_w)
                outs.append(conv2d_relu(up, self.w(p + ".w"), self.w(p + ".b"), 1, 1))
            else:
                outs.append(conv2d_relu(feats[si], self.w(p + ".w"), self.w(p + ".b"), s, 1))
        return concat_channels(outs)

    def head(self, feat: Tensor) raises -> Dict[String, Tensor]:
        """Shared 3x3 conv, then the heatmap (sigmoid) and 8-channel regression branches."""
        var shared = conv2d_relu(feat, self.w("head.shared.w"), self.w("head.shared.b"), 1, 1)

        var hm = conv2d_relu(shared, self.w("head.hm0.w"), self.w("head.hm0.b"), 1, 1)
        hm = conv2d(hm, self.w("head.hm1.w"), self.w("head.hm1.b"), 1, 0)
        sigmoid_(hm)

        var reg = conv2d_relu(shared, self.w("head.reg0.w"), self.w("head.reg0.b"), 1, 1)
        reg = conv2d(reg, self.w("head.reg1.w"), self.w("head.reg1.b"), 1, 0)

        var out = Dict[String, Tensor]()
        out["hm"] = hm^
        out["reg"] = reg^
        return out^

    def forward(self, sample: Dict[String, Tensor]) raises -> Dict[String, Tensor]:
        """Run one sweep: ``points (N, 4)`` + ``points_mask (N,)`` -> grid, hm, reg, boxes."""
        if "points" not in sample or "points_mask" not in sample:
            raise Error("sample is missing points / points_mask")
        var grid = pillar_grid(
            sample["points"], sample["points_mask"], self.w("pillars.w"), self.w("pillars.b"), self.cfg
        )
        var feat = self.backbone(grid)
        var out = self.head(feat)
        out["boxes"] = decode_heatmap(out["hm"], out["reg"], self.cfg)
        out["pillar_grid"] = grid^
        return out^
