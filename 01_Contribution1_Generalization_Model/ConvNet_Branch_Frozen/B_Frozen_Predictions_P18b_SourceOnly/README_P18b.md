# P18b — CORAL / MMD Feature Alignment (the main DA lever)

**Phase question:** does aligning the *learned features* of source and target
(instead of P18a's blunt input rescale) lift the LODO-DA mean above the P17 lower
bound — and crucially, can it help the far folds (KuHar, WISDM) *without* collapsing
the near ones (RW-Waist, UCI, MotionSense) the way transductive norm did?

## The one change vs P17

Add a feature-distribution alignment loss between the source batch and a
target-**unlabeled** batch, in the model's learned embedding space:

```
total = L_classification(source)  +  λ_dann · L_domain(source-vs-source)   [unchanged P9 recipe]
                                  +  λ_align · L_align(h_source, h_target)  [P18b: the new lever]
```

- **GNN:** alignment on the graph embedding `h` (the same vector fed to the heads).
- **ConvNet:** alignment on the last-timestep LSTM representation (penultimate feature).
- **L_align** = **CORAL** (match feature covariances; default) or **MMD** (multi-kernel
  Gaussian) — selected with `--align`.
- Target **labels are never used**; the target pool only contributes its features to the
  alignment term. Model selection stays on **source-val**. The existing DANN-over-sources
  term is untouched, so alignment is the single new lever.

Why this should behave better than P18a: input renormalization rescales *everything*
globally and can't serve a near and a far target at once. A feature-space alignment loss
is *learned and selective* — the encoder can move target representations toward the source
manifold where it helps, without globally rescaling the inputs the near folds rely on.

## How to run (Colab)

`notebooks/Run_P18b_FeatAlign_Colab.ipynb`, top to bottom. Two arms:

```bash
# baseline arm (source-only = P17 lower bound)  -> results/probs_dg/
python -m da_baseline.gnn_predict     --data_path "$DATA_PATH" --targets all --epochs 100
python -m da_baseline.convnet_predict --data_path "$DATA_PATH" --targets all --epochs 100
# DA arm (CORAL alignment, lambda=1.0)          -> results/probs_align/
python -m da_baseline.gnn_predict     --data_path "$DATA_PATH" --targets all --epochs 100 --align coral --align_lambda 1.0
python -m da_baseline.convnet_predict --data_path "$DATA_PATH" --targets all --epochs 100 --align coral --align_lambda 1.0
# compare + gate
python da_baseline/run_compare.py
```

Swap `--align coral` for `--align mmd` to try MMD. **λ (`--align_lambda`) is the key
knob** — too small does nothing, too large hurts classification. Tune it on **source-val**
(watch that source-val accuracy doesn't drop), never on the target. If you already have
trustworthy P17 baseline dumps, copy them into `results/probs_dg/` and skip the baseline arm.

## Keep / revert gate (decided before the run)

**KEEP** alignment and carry it into P18c **iff**:

- fused mean improves by **≥ +1.0pp** over source-only, **and**
- **no single fold drops by > 2pp** (the KuHar/RW-Waist collapse guard from P18a).

Otherwise retune λ; if no λ clears the gate, **REVERT** and proceed to **P18c**
(DANN source→target adversarial). `run_compare.py` prints the verdict and writes
`results/p18b_compare.json`.

## Honesty / caveats

Measured as +Δpp over the **P17 source-only lower bound (76.88%)**, same transductive
LODO-DA protocol; **not** comparable to the `P11_P16` DG numbers. The GNN is noisy
run-to-run and KuHar's test set is only 144 windows — if the verdict is borderline,
**seed-average 2–3 runs** before deciding. Start with CORAL (deterministic, stable);
reach for MMD only if CORAL is promising but you want to squeeze more.
