# Neuron Pruning Experiment Design
**Date:** 2026-03-17
**Project:** Module-Regularized GRN Inference
**Status:** Approved

---

## Goals

1. **Model compression** — reduce parameter count and inference cost by removing redundant neurons
2. **Capacity analysis** — understand how much of the 512-dim hidden space is actually used by the trained two-tower GRN model

---

## Architecture Context

The `HybridEmbeddingModel` is a two-tower MLP:

```
TF tower:   [embed(512) + expr(~2000)] → fc1(→512) → ReLU → fc2(→512) → cosine sim
Gene tower: [embed(512) + expr(~2000)] → fc1(→512) → ReLU → fc2(→512) ↗
```

Four FC layers total: `tf_fc1`, `tf_fc2`, `gene_fc1`, `gene_fc2`. Hidden dim = 512 in each.
Baseline: ~5.2M parameters, 80.14% accuracy, AUROC computed on test set (seed=42).

---

## Approach: Activation Profiling + Structured Pruning (Option A)

Symmetric structured pruning of all four FC layers, with neurons ranked by a combined importance score. Three-phase pipeline:

1. **Profile** — single forward pass over training set, accumulate activation stats
2. **Post-hoc prune + evaluate** — sweep sparsity levels, remove neurons, measure AUROC immediately
3. **Fine-tune + evaluate** — continue training pruned models for 10 epochs, re-measure AUROC

---

## Neuron Importance Scoring

Each neuron `j` in a layer is scored:

```
importance(j) = α × activation_freq(j) + (1 - α) × weight_magnitude(j)
```

- `activation_freq(j)` = fraction of training examples where post-ReLU output > 0
  (for fc2 layers without ReLU: mean absolute activation instead)
- `weight_magnitude(j)` = L2 norm of incoming weight column, normalized to [0,1] per layer
- `α = 0.5` (equal weighting, logged as reproducible hyperparameter)

**Structural constraint:** pruning neuron j from fc1 removes the corresponding input row from fc2. These are pruned as a pair to maintain shape consistency. Applied symmetrically to both TF and Gene towers.

---

## Sparsity Sweep

Sparsity levels: `0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%, 60%, 70%, 80%, 90%`

Fine resolution at low sparsity (5% steps to 30%) to precisely locate AUROC-retention thresholds.

**At each sparsity level:**
1. Post-hoc AUROC (no fine-tuning)
2. Fine-tuned AUROC (10 epochs, LR=0.005, Adam, same data splits)
3. Parameter count (compression ratio)

**AUROC retention:**
```
retention(s) = AUROC(pruned at s%) / AUROC(baseline)
```

Primary outputs: sparsity level where retention first drops below **0.95** and **0.90**.

**Baseline:** single trained model from `train_standard_mlp`, seed=42. Each sparsity level is an independent branch from the same checkpoint.

---

## Output

Results written to `results/neuron_pruning_results.json`:

```json
{
  "baseline_auroc": 0.xxx,
  "alpha": 0.5,
  "fine_tune_epochs": 10,
  "results": [
    {
      "sparsity": 0.10,
      "neurons_removed": 102,
      "params_remaining": 4200000,
      "posthoc_auroc": 0.xxx,
      "finetuned_auroc": 0.xxx,
      "posthoc_retention": 0.xxx,
      "finetuned_retention": 0.xxx
    }
  ],
  "retention_95_threshold_sparsity": 0.XX,
  "retention_90_threshold_sparsity": 0.XX
}
```

---

## Implementation Plan

### 1. `src/models/nn.rs` — two new methods on `LinearLayer`

- `prune_outputs(keep_indices: &[usize])` — resize weights `(in, out)→(in, k)`, bias `out→k`, grad arrays
- `prune_inputs(keep_indices: &[usize])` — resize weights `(in, out)→(k, out)`, grad arrays

### 2. `src/models/hybrid_embeddings.rs` — three new methods

- `profile_activations(tf_indices, gene_indices, tf_expr, gene_expr) -> NeuronStats`
  Accumulates per-neuron activation frequency and weight L2 norms across all training examples.
- `importance_scores(stats: &NeuronStats, alpha: f32) -> LayerScores`
  Combines activation frequency + weight magnitude into a single score per neuron per layer.
- `prune_to_sparsity(scores: &LayerScores, sparsity: f32)`
  Physically removes bottom-k% neurons. Prunes fc1 outputs and fc2 inputs together as pairs.

Supporting structs: `NeuronStats` (raw activation sums/counts + weight norms), `LayerScores` (final importance scores per layer).

### 3. `src/bin/neuron_pruning.rs` — new experiment binary

- Load trained model from `results/` checkpoint
- Phase 1: profile → save activation stats
- Phase 2: sparsity sweep, post-hoc eval
- Phase 3: fine-tune each pruned variant, re-eval
- Write `results/neuron_pruning_results.json`

No changes to training pipeline, data loading, config, or any existing binaries.

---

## Success Criteria

- Sparsity-vs-AUROC curve produced for both post-hoc and fine-tuned variants
- 95% AUROC-retention threshold identified (expected: ~20–40% sparsity based on typical ReLU networks)
- 90% AUROC-retention threshold identified (expected: ~40–60% sparsity)
- Results reproducible from seed=42 baseline checkpoint
