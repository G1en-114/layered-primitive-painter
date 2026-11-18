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
- Not run yet (paused): pieces of 20 px, budgets of 100 and 300 curves. Rerunning
  `ops/plan_v3/local_line_ablation.sh` skips the finished runs.
- Shadows, first attempt (2 images): one hard Gaussian per shadow region under clipped multiply is worse than the
  refit (25.84 vs 26.29 dB). Next: closed Bezier regions for cel shadows, and a stricter choice of which strokes
  paint shadows.
- The refit itself, +2.9 dB, applies to every decomposition. Polishing the existing data (faces too) is a
  separate, cheap-per-image win.
