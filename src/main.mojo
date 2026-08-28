"""CenterPillars Mojo inferencer CLI.

Usage:
    pixi run mojo run src/main.mojo data/weights.lft data/sample_0.lft [more samples...]
    pixi run mojo run src/main.mojo data/weights.lft data/sample_0.lft --csv dets.csv

Runs the exported detector on each exported sweep and, when the sample carries
the expected tensors from PyTorch, reports the max-abs difference per stage
(``hm``, ``reg``, ``boxes``) plus a final ``PARITY: PASS/FAIL`` line. With
``--csv`` the detections are also written in the column layout produced by
CenterPillars.py's ``centerpillars.predict``:

    t,cls,x,y,z,l,w,h,theta,conf
"""

from std.sys import argv
from std.time import perf_counter_ns

from centerpillars.io import load_lft
from centerpillars.model import CenterPillarsModel
from centerpillars.tensor import Tensor, max_abs_diff

comptime HM_TOL: Float32 = 1e-3
comptime REG_TOL: Float32 = 1e-3
comptime BOX_TOL: Float32 = 1e-2


def class_names(tensors: Dict[String, Tensor]) raises -> List[String]:
    """Decode ``__class_names__`` (K, 32 ASCII codes) into strings."""
    var out = List[String]()
    if "__class_names__" not in tensors:
        return out^
    ref t = tensors["__class_names__"]
    for i in range(t.dim(0)):
        var s = String("")
        for j in range(t.dim(1)):
            var c = Int(t.at2(i, j))
            if c == 0:
                break
            s += chr(c)
        out.append(s)
    return out^


def decode_timestamp(t: Tensor) -> Int:
    """Reassemble a nanosecond timestamp from three exact 24-bit float32 chunks."""
    var ts = 0
    for i in range(t.numel()):
        ts += Int(t[i]) << (24 * i)
    return ts


def fmt4(v: Float32) -> String:
    """Fixed-point rendering with 4 decimals, matching predict.py's CSV."""
    var a = Float64(v)
    var neg = a < 0.0
    if neg:
        a = -a
    var scaled = Int(a * 10000.0 + 0.5)
    var ip = scaled // 10000
    var fp = scaled - ip * 10000
    var s = String(ip) + "."
    if fp < 1000:
        s += "0"
    if fp < 100:
        s += "0"
    if fp < 10:
        s += "0"
    s += String(fp)
    if neg:
        return "-" + s
    return s


def report(name: String, got: Tensor, sample: Dict[String, Tensor], tol: Float32) raises -> Bool:
    """Print ``max|diff|`` for one stage; returns False only when the sample has an expectation and it fails."""
    if name not in sample:
        return True
    ref want = sample[name]
    if got.numel() != want.numel():
        print(
            "  ", name, "COUNT MISMATCH: mojo", got.numel(), "vs torch", want.numel(), "FAIL"
        )
        return False
    var diff = max_abs_diff(got, want)
    var ok = diff <= tol
    print("  ", name, "max|diff| =", diff, "PASS" if ok else "FAIL")
    return ok


def main() raises:
    var args = argv()
    var weights = String("")
    var samples = List[String]()
    var csv_path = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--csv":
            if i + 1 >= len(args):
                raise Error("--csv needs a path")
            csv_path = String(args[i + 1])
            i += 2
            continue
        if weights.byte_length() == 0:
            weights = a
        else:
            samples.append(a)
        i += 1

    if weights.byte_length() == 0 or len(samples) == 0:
        print("usage: mojo run src/main.mojo <weights.lft> <sample.lft>... [--csv out.csv]")
        return

    var model = CenterPillarsModel(weights)
    var names = class_names(model.tensors)
    print("loaded weights:", len(model.tensors), "tensors |", model.cfg)

    var all_pass = True
    var csv = String("t,cls,x,y,z,l,w,h,theta,conf\n")
    for s in range(len(samples)):
        var path = samples[s]
        var sample = load_lft(path)
        var n_pts = sample["points"].dim(0)

        var t0 = perf_counter_ns()
        var out = model.forward(sample)
        var ms = Float64(perf_counter_ns() - t0) / 1.0e6

        ref boxes = out["boxes"]
        print(path, "|", n_pts, "points |", boxes.dim(0), "detections |", ms, "ms")
        if not report("hm", out["hm"], sample, HM_TOL):
            all_pass = False
        if not report("reg", out["reg"], sample, REG_TOL):
            all_pass = False
        if not report("pillar_grid", out["pillar_grid"], sample, HM_TOL):
            all_pass = False
        if not report("boxes", boxes, sample, BOX_TOL):
            all_pass = False

        if boxes.dim(0) > 0:
            print(
                "   top det: cls", Int(boxes.at2(0, 8)),
                "at (", fmt4(boxes.at2(0, 0)), ",", fmt4(boxes.at2(0, 1)), ")",
                "score", fmt4(boxes.at2(0, 7)),
            )

        var ts = 0
        if "timestamp" in sample:
            ts = decode_timestamp(sample["timestamp"])
        for r in range(boxes.dim(0)):
            var cls = Int(boxes.at2(r, 8))
            var label = String(cls)
            if cls < len(names):
                label = names[cls]
            csv += String(ts) + "," + label
            for c in range(7):
                csv += "," + fmt4(boxes.at2(r, c))
            csv += "," + fmt4(boxes.at2(r, 7)) + "\n"

    if csv_path.byte_length() > 0:
        var f = open(csv_path, "w")
        f.write(csv)
        f.close()
        print("wrote", csv_path)

    print("PARITY:", "PASS" if all_pass else "FAIL")
