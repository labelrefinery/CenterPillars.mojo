# CenterPillars.mojo

[![CI](https://github.com/labelrefinery/CenterPillars.mojo/actions/workflows/ci.yml/badge.svg)](https://github.com/labelrefinery/CenterPillars.mojo/actions/workflows/ci.yml)

Pure-[Mojo](https://www.modular.com/mojo) inference implementation of **CenterPillars** — a single-stage anchor-free 3D LiDAR object detector combining the [PointPillars](https://arxiv.org/abs/1812.05784) encoder (Lang et al., CVPR 2019), a [SECOND](https://www.mdpi.com/1424-8220/18/10/3337)-style BEV backbone and a [CenterPoint](https://arxiv.org/abs/2006.11275) head (Yin et al., CVPR 2021).

Points are scattered into a BEV pillar grid, a 2-D convolutional backbone with a multi-scale concat neck produces a stride-2 feature map, and a center head predicts per-class Gaussian heatmaps plus one 8-channel box regression; detections are heatmap peaks followed by per-class circle NMS.

The PyTorch reference implementation and training code (ArgoVerse 2) live at [labelrefinery/CenterPillars.py](https://github.com/labelrefinery/CenterPillars.py). This repo reimplements the full forward pass from scratch in Mojo — no MAX, no Python interop at inference time — and verifies numerical parity against PyTorch.

It is the detector stage of an offboard auto-labeling pipeline: **CenterPillars** → [OfflinePoly.mojo](https://github.com/labelrefinery/OfflinePoly.mojo) (offline 3D MOT) → [LabelFormer.mojo](https://github.com/labelrefinery/LabelFormer.mojo) (trajectory refinement). Its `--csv` output is exactly the format the tracker reads.

## Layout

- `src/centerpillars/` — the package: `tensor` (minimal f32 tensor, concat, sigmoid/relu), `io` (LFT1 container reader + self-describing `Config`), `layers` (conv2d, nearest upsample, 3×3 max-pool peak test), `pillars` (9-feature point net + scatter-max pillar grid), `model` (backbone, neck, head, full forward), `decode` (peaks → boxes, circle NMS).
- `src/main.mojo` — inferencer CLI with per-stage parity reporting and CSV output.
- `tests/test_ops.mojo` — 26 hand-computed unit tests covering every op.

`conv2d` is the hot loop (~5.8 GMAC per sweep for the smoke model): the weight scalar is hoisted out and the innermost loop walks the output width contiguously, which lets the stride-1 case (the necks and the whole head) run on SIMD lanes.

## Supported Mojo versions

Both **stable Mojo 1.0** and the **Modular nightly** are supported and tested in CI (unit tests, Linux and macOS-arm64):

| pixi environment | compiler | run it |
|---|---|---|
| `default` | nightly (`modular` ≥ 26.6 nightly) | `pixi run test` / `pixi run infer` |
| `stable` | `mojo-compiler == 1.0.0` | `pixi run -e stable test` / `pixi run -e stable infer` |

The package build pins stable 1.0 (the `pixi-build-mojo` backend requires it) and declares `mojo-compiler >=1.0,<2` as its run dependency.

## Setup

Requires [pixi](https://pixi.sh). `pixi install` pulls the toolchains declared in `pixi.toml`.

Weights and parity samples are not committed. Export them from the PyTorch side (in a CenterPillars.py checkout with a trained checkpoint):

```sh
uv run python scripts/export_mojo.py --checkpoint runs/smoke/best.pt --out export
cp export/*.lft ../CenterPillars.mojo/data/
```

## Run

```sh
pixi run test    # 26 op unit tests
pixi run infer   # run the exported sweeps, report per-stage parity vs PyTorch
# or directly:
pixi run mojo run src/main.mojo data/weights.lft data/sample_0.lft
# write detections for the tracker:
pixi run mojo run src/main.mojo data/weights.lft data/sample_0.lft --csv dets.csv
```

The CSV columns match CenterPillars.py's `centerpillars.predict`:

```
t,cls,x,y,z,l,w,h,theta,conf
315969911860040000,VEHICLE,39.0756,-1.3507,0.3698,4.2770,1.9612,1.5701,-0.1300,0.5204
```

(`t` is the AV2 nanosecond sweep timestamp, carried through the LFT1 sample as three exact 24-bit float32 chunks; `cls` comes from the `__class_names__` tensor in `weights.lft`.)

## Parity vs PyTorch

Verified on 3 real ArgoVerse 2 val sweeps (77k–97k points, 86–101 detections each, smoke checkpoint) — max abs difference per stage, with identical detection counts:

| stage | max \|diff\| | tolerance |
|---|---|---|
| `pillar_grid` (point net + scatter max) | ≤ 1.7e-6 | 1e-3 |
| `hm` (post-sigmoid heatmap, 2×125×125) | ≤ 6.6e-7 | 1e-3 |
| `reg` (8×125×125) | ≤ 1.0e-5 | 1e-3 |
| `boxes` (n×9 after circle NMS) | ≤ 1.0e-5 | 1e-2 |

Inference is **~0.6 s per full sweep** on a single Apple M4 CPU core (250×250 pillars → 125×125 heatmap, ~97k points).

The residual difference is float32 summation order: PyTorch accumulates the per-pillar xyz sums with a parallel `index_add_`, this code accumulates them sequentially. Everything downstream (conv order, the `pooled == hm` peak test, the top-K tie-break, greedy NMS) is bit-for-bit the same rule.

`export_mojo.py --pillar-grid` adds the pillar pseudo-image to the sample so `main.mojo` reports it as an extra stage — the quickest way to localize a divergence to before or after the encoder.

## Implementation notes

The `__config__` tensor in `weights.lft` makes the network self-describing: detection ranges, pillar size, grid and heatmap dims, output stride, class count, score threshold, max detections, per-class NMS radii, and per-stage channels / layers / strides / neck conv strides. `model.mojo` builds the whole forward pass from it, so a differently-shaped checkpoint (e.g. the paper-scale 512×512 / 3-class config) runs without a code change.

Details that matter for parity:

- **BatchNorms are folded at export**, so every layer here is a plain conv (or linear) with a bias — including the `bias=False` convs, which gain one.
- **No points-per-pillar cap**: two passes over the points (pillar xyz sums, then features + ReLU + max), matching the PyTorch dense scatter exactly. Empty pillars stay exactly zero, which is valid because the ReLU makes every pooled feature non-negative.
- **Explicit upsample target size.** Stride-2 convs round *up*, so the backbone stages are 125 / 63 / 32 for a 250-pillar grid — not exact multiples. Each stage is nearest-resized to the heatmap grid by size, never by scale factor.
- **The peak test is exact.** `maxpool3x3_peaks` compares each cell to the max of the very same float32 cells, so `pooled == hm` behaves identically in both languages, plateaus included.
- **Top-K tie-break.** Candidates are collected in flat `(class, row, col)` order and selected with a strict `>`, so score ties resolve to the lower flat index — the same choice `torch.topk` makes here.

## License

MIT — see [LICENSE](LICENSE).
