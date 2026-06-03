# GateNet — Vision-Based Gate Segmentation for the Drone Racer

## Motivation

The Dreamer-style world model trained on this drone-racer environment originally
consumed the simulator's privileged `semantic_segmentation` channel directly:
each pixel was tagged with the class ID of the gate it belonged to, and the
env wrapper filtered that down to a binary "is the target gate?" mask before
handing it to the encoder. That works in sim but is impossible to obtain on a
real drone — the deployed system has only an RGB camera and no oracle telling
it which pixels belong to a gate.

GateNet closes the sim-to-real gap on the perception side. It is a compact
U-Net that takes the same 64×64 RGB observation that the policy already sees
and produces a single-channel binary mask of "any-gate" pixels (SkyDreamer
convention, see [arXiv:2510.14783] Appendix A). The mask is then fed to the
DreamerV3 world-model encoder in place of the privileged segmentation, so the
training pipeline becomes fully transferable to hardware.

[arXiv:2510.14783]: https://arxiv.org/abs/2510.14783

## What we built

The work landed across these files (commits prefixed `feat(gatenet):` and
`feat(eval_gatenet):` on `master`):

| Path | Purpose |
|------|---------|
| [dreamer/gatenet.py](../dreamer/gatenet.py) | The U-Net itself — five output heads at multiple resolutions for deep supervision, optional pose-regression heads, optional multi-gate prediction. |
| [scripts/data/collect_gatenet_data.py](../scripts/data/collect_gatenet_data.py) | Runs the Dreamer env under random actions, records `(RGB, any-gate mask)` pairs along with target-gate body-frame pose labels, dumps to a single compressed `.npz`. Supports `--frame_stack` for temporal parallax (channel-axis concatenation of consecutive RGB frames). |
| [scripts/train_gatenet.py](../scripts/train_gatenet.py) | Supervised training. Loss is `4·L0 + 2·L1 + L2 + L3 + L4` with `L_i = Dice + 2·BCE` at each scale (paper recipe). Adds an optional pose head trained with smooth-L1 in body frame. |
| [scripts/eval_gatenet.py](../scripts/eval_gatenet.py) | Offline metric dump (IoU, precision, recall, pixel accuracy, body-frame position / quaternion errors, visibility classifier) plus PNG grids of best-and-worst predictions. |
| [scripts/render_gatenet_video.py](../scripts/render_gatenet_video.py) | Renders an mp4 of `(RGB \| GT \| Prediction)` frames for figures / demos. |
| [dreamer/env_wrapper.py](../dreamer/env_wrapper.py) | Production integration. When `gatenet_ckpt=…` is passed, the env wrapper replaces the privileged-segmentation mask with the GateNet's vision-only prediction. The Dreamer training loop then sees the exact same data as the real drone would. |

## Architecture

GateNet follows the SkyDreamer U-Net layout but is scaled for the small 64×64
image used in this sim:

```
inc       :  in_channels   -> 64
down1     :  64   -> 128   (MaxPool 2×2)
down2     :  128  -> 256
down3     :  256  -> 512
down4     :  512  -> 512   (bottleneck, 4×4 feature map)
up4       :  concat skip + ConvTranspose -> 256
up3       :  -> 128
up2       :  -> 64
up1       :  -> 64
out_i (×5):  1×1 Conv -> 1-channel logit map at each scale
```

A scaling factor `f` divides every channel count by `f`. The paper uses `f=2`
at 196×196 and `f=4` at 384×384; for 64×64 we default to `f=1` so the encoder
keeps its full capacity at the small resolution. With `multi_gate=True` the
output head emits one binary mask per gate; with `with_pose=True` the encoder's
bottleneck branches into smooth-L1-regressed position / quaternion heads in
the drone body frame.

Loss at training time:

```
total = 4·dice_bce(L0)  +  2·dice_bce(L1)  +  dice_bce(L2)
                       +  dice_bce(L3)     +  dice_bce(L4)
dice_bce(L) = dice(L) + 2 · binary_cross_entropy(L)
```

The deep-supervision weighting biases gradients toward the full-resolution
output while still pushing structure into the bottleneck.

## Data collection

`collect_gatenet_data.py` drives the standard `Isaac-Drone-Racer-Dreamer-RGB-v0`
task with random actions and a sim pre-warm phase. The pre-warm matters: the
TiledCamera does not populate `idToLabels` for semantic_segmentation until at
least one render pass has run for every gate prim, so the collector waits for
all `gate_*` class IDs to be discoverable before it starts writing samples
(fixed in `39f0aec`).

Each sample contains:

| Key | Shape | Dtype | Meaning |
|-----|-------|-------|---------|
| `images` | `(N, H, W, 3·F)` | `uint8` | RGB (or stacked RGB for `frame_stack > 1`). |
| `masks` | `(N, H, W)` | `uint8` (0/255) | Any-gate binary mask. |
| `target_idx` | `(N,)` | `int32` | Which gate the policy was targeting at the time. |
| `target_pos_b` / `target_quat_b` | `(N, 3)` / `(N, 4)` | `float32` | Target gate pose in drone body frame. |
| `target_visible` | `(N,)` | `bool` | Whether the target gate is in-frame. |
| `all_pos_b` / `all_quat_b` / `all_visible` | `(N, G, …)` | `float32` / `bool` | Same but for every gate on the track (used by `multi_gate=True`). |

The bug-fix history in the commit log is worth knowing:

- `8bfb051` — early collector returned an "all-non-background" mask, which
  included the ground / track decals. Now filters strictly to the discovered
  gate class IDs.
- `39f0aec` — pre-warm the sim so the class-id table is non-empty.
- `c0d2821` — added pose-regression labels for comparative study.
- `84cd3a5` — multi-gate output head, one mask per gate.
- `792f1ba` — frame-stack option for temporal parallax (channel-axis stack of
  the last K RGB frames), helps depth and gate-pose estimates.

## Results on this track

The most recent eval run (`logs/gatenet/2026-05-22_22-57-26/eval/metrics.txt`)
on a 100k-sample dataset with the standard 10% deterministic seed-42 val split:

```
mean_iou                          0.9255
median_iou                        1.0000
iou_visible                       0.8125
iou_empty                         0.9947
precision                         0.9567
recall                            0.8788
pixel_accuracy                    0.9969
pos_b_err_visible_median_m        1.6221
quat_b_angle_visible_median_deg  18.4753
visible_accuracy                  0.8934
```

Median IoU is 1.0 — most frames where the gate is either fully visible or
fully absent are predicted exactly. The mean-IoU drag (0.93) comes from the
hard cases: gates entering / leaving the field of view at the image edge,
heavy yaw rotations that compress the gate to a few pixels, and motion blur
during fast turns. The `worst_iou.png` / `worst_pose.png` outputs in the same
`eval/` directory show what those failure cases look like — almost all
correspond to gates that are *just* about to exit the frame.

## Integration with DreamerV3

When `train_dreamer.py` is invoked with `--gatenet_ckpt …`, the env wrapper
swaps the privileged-segmentation path for the GateNet prediction:

```python
# dreamer/env_wrapper.py
if gatenet_ckpt is not None:
    ckpt = torch.load(gatenet_ckpt, map_location="cpu")
    self._gatenet = GateNet(in_channels=ckpt["in_channels"], f=ckpt["f"])
    self._gatenet.load_state_dict(ckpt["model"])
    self._gatenet.eval().to(seg_or_rgb.device)

# inside _extract_obs:
if self._gatenet is not None:
    mask_u8 = _gatenet_predict_mask_u8(self._gatenet, rgb_u8)
else:
    mask_u8 = _extract_gate_mask_u8(seg, ...)   # fallback: oracle segmentation
```

The downstream world model sees the same `(image, state)` dict either way,
so training is identical. With `--obs_mode mask` or `--obs_mode rgb_mask` the
GateNet output replaces the sim-only segmentation channel, removing the only
ground-truth-only signal the policy depends on for perception.

Note: GateNet outputs the *any-gate* mask (SkyDreamer convention) — not the
target gate. The flight-plan / command manager handles gate ordering
downstream, so the drone perceives "where are all the gates?" not "which one
is mine right now?".

## Generating the demo video

The repo includes `scripts/render_gatenet_video.py` to produce a short mp4
suitable for talks / demos. Each frame is `[RGB | RGB+GT overlay (green) |
RGB+Pred overlay (red)]`, upscaled with nearest-neighbour interpolation so
the pixel boundaries stay crisp on a projector:

```bash
python3 scripts/render_gatenet_video.py \
    --checkpoint logs/gatenet/<RUN>/checkpoints/gatenet_best.pt \
    --data data/gatenet/train.npz \
    --num_frames 300 \
    --fps 30 \
    --output logs/gatenet/<RUN>/eval/gatenet_demo.mp4
```

300 frames at 30 fps is a 10-second clip, which is enough to show the network
tracking a gate as the drone yaws through it and then re-acquiring the next
one.
