# Line strokes (protocol v3 candidate, 2026-11-18)

## Why

The anime decompositions reach only 24.0 dB (faces 30.2 dB). Generalised Gaussians break thin line art into
dots. The model then learns to place those dots, which it cannot do from a caption: the specks in the anime
samples. Removing the strokes under 3 px (25% of them) from base_v3's anime samples leaves the judge scores
unchanged or slightly better (tools/stroke_filter_eval.py). So those strokes are failed attempts at lines.

## The stroke

A quadratic Bezier curve: ends P0 and P2, control point P1, constant width, colour, opacity
(`lpp.render.bezier`). The hard edge is a one-pixel antialiased ramp, `clamp(w/2 - d + 0.5, 0, 1)`; for fitting
it is a sigmoid. The distance is measured to a 16-piece polyline. Each curve is rendered in a 48 px window
around itself, so the cost is N * 48^2 instead of N * 256^2. Compositing is the same as for the Gaussians.

## Layers and order (the user's design, 2026-11-18)

Lines are not a separate top layer. Every layer has its own lines, composited right after that layer's fills.
A face outline is then hidden by the hair in front of it, which a top line layer would draw over. A line
belongs to the topmost layer it touches, so an outline goes with the part that occludes.

Painting order (separate from compositing, as with draft order):

    colour sketch of every layer -> lines of every layer -> refined fills -> a final line pass per layer

A real top layer stays an option, only for lines that cross layers (the figure's silhouette).

## Step 1: equal-budget probe (tools/line_probe.py)

Per image:

1. Extract the line art: grayscale black-hat (dark structures narrower than a 2 px disk).
2. Skeletonise and trace it, then cut it into pieces of at most 14 px.
3. Turn each piece into a Bezier through its ends and its middle. Width comes from the distance transform, colour
   from the darker half of the pixels under it.
4. Remove the K smallest Gaussians and add the K curves.
5. Fit the curves with Adam (300 steps, soft edge annealed 0.7 -> 0.2 px). The Gaussians stay fixed.

Metrics: PSNR, and line F1 (line art of the reconstruction against the original's, 1 px tolerance).

Smoke test, 2 images, 60 steps, about 180 curves per image:

| | all Gaussians | minus the K smallest | plus K curves |
|---|---|---|---|
| PSNR | 24.2 dB | 23.5 dB | 25.4 dB |
| line F1 | 0.61 | 0.56 | 0.77 |

Full probe: 16 images (anime_v2_l6_t0/decomp_003), 300 steps, 56-200 curves per image (mean 173):

| | all Gaussians (692) | minus the K smallest | plus K curves (same count) |
|---|---|---|---|
| PSNR | 24.59 dB | 23.86 dB | **26.39 dB** |
| line F1 | 0.682 | 0.634 | **0.823** |

Every image improves: +0.7 to +4.1 dB (median +1.7) and line F1 +0.06 to +0.22. The lines in the samples are
continuous where the Gaussians left dotted smears (hair strands, face outline, eyes). The gate was 27 dB. These
numbers fit the curves alone, on top of fixed Gaussians that were fitted without them. Next: fit both jointly,
and tune the piece length and the budget.

## P1 ablation (16 images, 300 steps, curves quantized at evaluation, 2026-11-18)

Refit control: all 692 original Gaussians, no curves, the same joint fitting and steps. It shows that the
decomposer's Gaussians are far from converged (80 Adam steps per round).

| configuration | curves | PSNR | line F1 |
|---|---|---|---|
| current data (all Gaussians as decomposed) | 0 | 24.59 | 0.682 |
| curves only, Gaussians fixed | 173 | 26.23 | 0.822 |
| + contour first | 173 | 26.25 | 0.820 |
| + skeleton distance term (1e-4) | 173 | 26.09 | 0.822 |
| refit control (Gaussians only) | 0 | 27.46 | 0.803 |
| **joint (Gaussians + curves)** | 173 | **28.60** | **0.902** |
| joint + contour first + distance term | 173 | 28.46 | 0.901 |
| same, pieces of 10 px | 181 | 28.38 | 0.888 |

- Joint fitting with curves: +4.0 dB over the current data. Of that, +1.1 dB and +0.10 line F1 are the curves'
  own share over the same-steps refit. The gate (27 dB) is passed.
- The distance term does not help (the thresholded skeleton is not precise enough to pull towards). It is off.
- Contour-first changes nothing in reconstruction. It stays on, because it puts the curves where the model can
  read them off the canvas.
- Piece length and budget (joint + contour first + distance term):

  | setting | curves | PSNR | line F1 |
  |---|---|---|---|
  | pieces 10 px | 181 | 28.38 | 0.888 |
  | pieces 14 px (default) | 173 | 28.46 | 0.901 |
  | pieces 20 px | 161 | 28.55 | 0.903 |
  | at most 100 curves | 97 | 28.13 | 0.868 |
  | at most 300 curves | 200 | 28.63 | 0.911 |

  Longer pieces mean fewer curves for the same quality. The budget saturates near 200, which is about all the
  line art there is (a cap of 300 uses 200). About 100 curves already give most of the gain.
  Chosen: 20 px pieces and a cap of 200 curves, about a quarter of a ~700-stroke painting.
- Shadows (2 images, 100 steps; the refit control is 26.29 dB):
  - one hard Gaussian per shadow region under clipped multiply: 25.84 dB;
  - closed Bezier regions (`bezier.Regions`, 6 pieces, 3 Gaussians removed per region): 25.18 dB.

  Both are worse. A single base colour per layer is too crude for hair and clothes with several colours, so the
  "shadows" include parts that are simply darker. Reconstruction also cannot show the benefit the multiply design
  aims at, which is easier learning. Shadows stay out of protocol v3 until per-layer colour clustering gives base
  colours, and until a small model shows the multiply strokes are easier to learn.
- The refit itself, +2.9 dB, applies to every decomposition. Polishing the existing data (faces too) is a
  separate, cheap-per-image win.

## Polishing the decompositions (tools/refit_probe.py, 2026-11-18)

Refit all strokes of a decomposition jointly, snapped to the token grid every step. Sixteen images each, laptop,
one image at a time:

| domain | as decomposed | +100 steps | +300 steps | time for 300 steps |
|---|---|---|---|---|
| faces (v2, trajectory 0) | 30.22 dB | 31.36 | 31.57 | 17 s |
| anime (anime_v2_l6_t0) | 24.59 dB | | 27.46 | |

The anime decompositions are far from converged: line art is hard for the round-by-round fitting. The faces
gain about 1.1 dB, most of it in the first 100 steps. Anime gets its polish for free in the line-stroke
re-decomposition (P6). For faces, the model, not the data, holds back realism (83% of the FD gap). So polishing
the 360k face trajectories, roughly 20 DCU-hours in a batched run, waits.

## Production decomposer: where the line gain goes (10 images, 200 steps, 2026-11-18)

The line probe gained +1.12 dB over its refit control, and the first `decompose_lines.py` gained nothing. Measured
step by step on the same images:

| variant | PSNR | over control |
|---|---|---|
| control: no lines, same joint refit | 26.45 | - |
| lines snapped to tokens every step (straight-through) | 26.44 | +0.00 |
| lines snapped only when written | 26.59 | +0.14 |
| **position and angle snapped and frozen for the last 20% of the steps, the rest adapts (default now)** | **26.93** | **+0.48** |
| before the final snap (ceiling of the token parametrisation) | 27.24 | +0.79 |
| probe: free control points, positions/colours quantised only at the end | 27.59 | +1.14 |

The global and per-layer swaps give the same result, and the Triton and torch line renderers agree (26.59 against
26.67). The remaining gap has two parts:

- The symmetric parametrisation (no control-point offset along the chord) costs 0.35 dB.
- The grids cost the rest. Quantising one attribute at a time: position 0.33 dB, angle 0.12, width 0.06, bend 0.05,
  length and opacity 0.01. A 0.5 px position grid is coarse for a 1 px line. The fix is a finer x/y grid in the
  protocol (1024 bins), which touches the whole data path (later).

Line F1 rises regardless (0.78 to 0.85 on the first production shard), and so does the model trained on it: anime
judge FD 0.367 (control) to 0.332 after the attribute-loss fix.

## Formal training (2×RTX 5090, 2026-11-19)

base_v3 (126M, Gaussians only) converted to protocol v3 (`tools/init_v3_from_v2.py`) and trained with lines and
the draft paint order (`ops/v3_train/`). Data: 89k face sequences (4 Gaussian-only v2 trajectories, widened on load:
replay against forgetting) + 50,321 anime images with Bézier lines (trajectory t0), about 64:36. LP-FT:
- stage A, 1000 steps: only the parts v3 adds train (LINE heads, bend, event tables; 2.9M of 129M);
- stage B, 12,000 steps: everything, lr 1e-4 → 1e-5.

DDP on two GPUs, per-GPU batch 8 × accumulation 2 (global 32), 55k tokens/s, 1 h 35 min in total. Every 2000 steps
a snapshot went to the laptop and was evaluated there (192 face + 192 anime prompts, CFG 5 / top-p 0.6, judge
CLIP L/14; one standard error of R@5 is about 0.035). Troubleshooting log: docs/formal_training_incidents.md.

| model | faces R@1 | faces R@5 | faces FD | anime R@1 | anime R@5 | anime FD |
|---|---|---|---|---|---|---|
| base_v3 (reference) | 0.172 | 0.380 | 0.373 | 0.115 | 0.375 | 0.227 |
| A (LP), step 1000 | 0.130 | 0.307 | 0.380 | 0.099 | 0.276 | 0.271 |
| B step 2000 | 0.182 | 0.349 | 0.376 | 0.115 | 0.312 | 0.237 |
| B step 4000 | 0.198 | 0.380 | 0.366 | 0.125 | 0.312 | 0.233 |
| B step 6000 | 0.146 | 0.385 | 0.376 | 0.146 | 0.370 | 0.228 |
| B step 8000 | 0.203 | 0.396 | 0.368 | 0.130 | 0.432 | 0.215 |
| B step 10000 | 0.182 | 0.375 | 0.372 | 0.156 | 0.417 | 0.213 |
| **B step 12000 (final)** | 0.188 | **0.406** | 0.371 | 0.135 | **0.417** | **0.206** |

- No forgetting: faces dip while the frozen trunk meets the new paint order (stage A), are back by step 4000, and end
  at or slightly above base_v3.
- Anime gains from the lines: R@5 +0.042 and FD 0.227 → 0.206, the most realistic anime yet.
- Final model: `runs/v3_formal/B_ft/model_step012000.pt` (sha256 8bc318e992a192e1…).

Best-of-4 (4 samples per prompt, CLIP B/16 picks, L/14 judges; same setting as base_v3's best):

| best-of-4 | faces R@1 | faces R@5 | faces FD | anime R@1 | anime R@5 | anime FD |
|---|---|---|---|---|---|---|
| base_v3 | 0.193 | 0.432 | 0.375 | 0.151 | 0.479 | 0.221 |
| **v3 final** | 0.177 | 0.422 | **0.369** | **0.208** | **0.500** | **0.197** |

Next steps (quality problems seen in the images, image-to-painting by the model, deployment): docs/plan_v4.md.
