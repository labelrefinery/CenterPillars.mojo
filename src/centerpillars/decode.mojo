"""Heatmap decoding: peaks -> boxes, plus circle NMS.

Mirrors CenterPillars.py's ``decode.py`` exactly -- same peak rule, same top-K
ordering, same greedy per-class suppression by BEV centre distance -- so keep
any change in sync with the PyTorch side.
"""

from std.math import atan2, exp

from .io import Config
from .layers import maxpool3x3_peaks
from .tensor import Tensor

comptime BOX_DIM = 9
"""Decoded detection row: ``x, y, z, l, w, h, yaw, score, class``."""


def circle_nms(boxes: Tensor, keep_in: List[Int], radius: Float32) raises -> List[Int]:
    """Greedily drop lower-scored rows within ``radius`` metres of a kept centre.

    Args:
        boxes: ``(n, 9)`` detections sorted by descending score.
        keep_in: row indices of one class, in descending-score order.
        radius: suppression radius in metres.

    Returns:
        The subset of ``keep_in`` that survives, in the same order.
    """
    var out = List[Int]()
    var m = len(keep_in)
    var suppressed = List[Bool](length=m, fill=False)
    var r2 = radius * radius
    for i in range(m):
        if suppressed[i]:
            continue
        out.append(keep_in[i])
        var xi = boxes.at2(keep_in[i], 0)
        var yi = boxes.at2(keep_in[i], 1)
        for j in range(i + 1, m):
            if suppressed[j]:
                continue
            var dx = boxes.at2(keep_in[j], 0) - xi
            var dy = boxes.at2(keep_in[j], 1) - yi
            if dx * dx + dy * dy <= r2:
                suppressed[j] = True
    return out^


def decode_heatmap(hm: Tensor, reg: Tensor, cfg: Config) raises -> Tensor:
    """Decode ``hm (K, H, W)`` (already sigmoid) + ``reg (8, H, W)`` into ``(n, 9)`` boxes.

    Peaks are cells equal to their 3x3 max; the top ``max_dets`` peaks over all
    classes above ``score_thresh`` are kept, then per-class circle NMS runs. The
    result is sorted by descending score.
    """
    if hm.rank() != 3 or reg.rank() != 3:
        raise Error("decode_heatmap: hm and reg must be rank 3")
    var k = hm.dim(0)
    var h = hm.dim(1)
    var w = hm.dim(2)
    if reg.dim(0) != 8 or reg.dim(1) != h or reg.dim(2) != w:
        raise Error("decode_heatmap: reg must be (8, H, W) matching hm")

    var peaks = maxpool3x3_peaks(hm)

    # Candidates in flat (class, row, col) order, so score ties break on the
    # lower flat index -- the same tie-break torch.topk gives here.
    var cand = List[Int]()
    var cand_score = List[Float32]()
    for i in range(peaks.numel()):
        if peaks[i] > cfg.score_thresh:
            cand.append(i)
            cand_score.append(peaks[i])

    var n_cand = len(cand)
    var take = cfg.max_dets
    if take > n_cand:
        take = n_cand
    var used = List[Bool](length=n_cand, fill=False)
    var order = List[Int]()
    for _ in range(take):
        var best = -1
        for i in range(n_cand):
            if used[i]:
                continue
            if best < 0 or cand_score[i] > cand_score[best]:
                best = i
        if best < 0:
            break
        used[best] = True
        order.append(best)

    var cell_m = cfg.cell()
    var boxes = Tensor([len(order), BOX_DIM])
    for i in range(len(order)):
        var flat = cand[order[i]]
        var cls = flat // (h * w)
        var cell = flat - cls * h * w
        var row = cell // w
        var col = cell - row * w
        var r0 = reg[0 * h * w + cell]
        var r1 = reg[1 * h * w + cell]
        boxes.set2(i, 0, (Float32(col) + r0) * cell_m + cfg.x_min)
        boxes.set2(i, 1, (Float32(row) + r1) * cell_m + cfg.y_min)
        boxes.set2(i, 2, reg[2 * h * w + cell])
        boxes.set2(i, 3, exp(reg[3 * h * w + cell]))
        boxes.set2(i, 4, exp(reg[4 * h * w + cell]))
        boxes.set2(i, 5, exp(reg[5 * h * w + cell]))
        boxes.set2(i, 6, atan2(reg[6 * h * w + cell], reg[7 * h * w + cell]))
        boxes.set2(i, 7, cand_score[order[i]])
        boxes.set2(i, 8, Float32(cls))

    # Per-class circle NMS; survivors are merged back in score order.
    var keep = List[Bool](length=boxes.dim(0), fill=False)
    for c in range(k):
        var idx = List[Int]()
        for i in range(boxes.dim(0)):
            if Int(boxes.at2(i, 8)) == c:
                idx.append(i)
        var radius: Float32 = 0.0
        if c < len(cfg.nms_radius):
            radius = cfg.nms_radius[c]
        var kept = circle_nms(boxes, idx, radius)
        for i in range(len(kept)):
            keep[kept[i]] = True

    var n_keep = 0
    for i in range(len(keep)):
        if keep[i]:
            n_keep += 1
    var out = Tensor([n_keep, BOX_DIM])
    var o = 0
    for i in range(boxes.dim(0)):
        if not keep[i]:
            continue
        for j in range(BOX_DIM):
            out.set2(o, j, boxes.at2(i, j))
        o += 1
    return out^
