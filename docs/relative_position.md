# Relative position heads (`ModelConfig.rel_pos`, 2026-11-18)

## Why

The base_v3 validation loss for stroke position is x 4.99 and y 4.48 nats (soft labels, σ = 1 bin: the floor
is ~1.42). The model is unsure where each stroke goes. Fine strokes that are a few pixels off turn into the
specks and smears in the samples. Lowering the sampling temperature removes the specks, but only by flattening
the pictures (docs: sampling sweep), so the uncertainty itself has to come down.

Inside a round, the decomposer orders strokes along a Z-order curve, so the next stroke is usually close to the
previous one. An absolute 512-bin head has to learn that offset distribution separately at every location.

## What

For x and y, a second head predicts the offset from the previous primitive (any layer). The logits are added
in absolute space:

    logit_x[b] = abs_x(h)[b] + rel_x(h)[b - prev_x + 511]      (rel_x has 2*512 - 1 outputs)

The model can use either term: absolute for the first stroke of a layer (where a face goes), relative inside a
round. Everything downstream is unchanged: the protocol, the grammar, the codec, the soft-label loss and the
sampler all still see absolute bins. Only the model computes logits differently. The relative heads are
zero-initialised, so the model starts exactly as the absolute one (tested).

Not yet in the ONNX export or the C++ runtime: `export.graphs` refuses a `rel_pos` model.

## Tests (`tests/test_rel_pos.py`)

- The previous-primitive lookup.
- Zero init gives identical losses to the absolute model.
- The offset arithmetic.
- A sampler that only knows "x = previous x + 1" paints climbing strokes.
- Greedy samples re-scored by teacher forcing reproduce every argmax, so training, GRPO log-probs and sampling
  agree on the previous primitive.

## Experiment

The same run as the 12-trajectory small ablation (runs/plan11_4090/plan11_results/abl_small_12traj: layer order,
12000 steps, identical data and hyperparameters), plus `--rel-pos`. Compare:

- validation x/y loss;
- judge R@1/R@5 and FD;
- the samples, by eye.
