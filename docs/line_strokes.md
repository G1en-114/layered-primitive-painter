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
