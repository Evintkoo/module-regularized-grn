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
Baseline: ~5.2M parameters, single model AUROC evaluated on test set (seed=42, LR=0.001).

In `LinearLayer`, weight matrices have shape `(in_dim, out_dim)`, so column `j` of `weights` is the incoming weight vector for output neuron `j`.

---

## Approach: Activation Profiling + Structured Pruning

Symmetric structured pruning of all four FC layers, with neurons ranked by a combined importance score. Three-phase pipeline:

1. **Profile** — inference-only forward pass over training set, accumulate activation stats (no backward, no gradient accumulation, no optimizer step)
2. **Post-hoc prune + evaluate** — sweep sparsity levels, remove neurons, measure AUROC immediately
3. **Fine-tune + evaluate** — continue training pruned models for 10 epochs, re-measure AUROC

**Baseline model:** re-train from scratch with seed=42 inside the `neuron_pruning` binary (same config as `train_standard_mlp`: LR=0.001, Adam, batch=256, 60 epochs, patience=10). This avoids a checkpoint dependency and ensures full reproducibility.

---

## Neuron Importance Scoring

Neurons in a paired (fc1, fc2) unit are scored based on their role in the fc1 hidden layer. The fc2 layer does not have a ReLU and its output feeds directly into cosine similarity — so only fc1 neurons are scored and ranked; fc2 is pruned as a structural consequence.

For each neuron `j` (index into fc1's output dimension, 0..511):

```
importance(j) = α × activation_freq(j) + (1 - α) × weight_magnitude(j)
```

- **`activation_freq(j)`** = fraction of training examples where fc1 post-ReLU output neuron `j` > 0.0
  Computed over an inference-only forward pass of the full training set.

- **`weight_magnitude(j)`** = combined normalized L2 magnitude of neuron j's weights in both fc1 and fc2.
  Each component is normalized independently to [0, 1] before summing, to avoid the larger fc1 weight vectors (shape ~2512) dominating fc2 (shape 512):
  ```
  fc1_norm(j)  = || fc1.weights.column(j) ||₂         # normalized to [0,1] across j=0..511
  fc2_norm(j)  = || fc2.weights.row(j)    ||₂         # normalized to [0,1] across j=0..511
  weight_magnitude(j) = (fc1_norm(j) + fc2_norm(j)) / 2
  ```
  Normalization is per-tower (TF tower neurons normalized separately from Gene tower neurons).

- **`α = 0.5`** (equal weighting; logged as a reproducible hyperparameter in output JSON)

**Pruning mask:** computed **independently** for TF tower and Gene tower. The same sparsity percentage is applied to both towers (equal percentage, not mirrored neuron selection), but the specific neurons removed may differ between towers, reflecting the independently learned representations.

**Structural constraint:** removing neuron `j` from fc1 requires:
- Removing output column `j` from `fc1.weights` and entry `j` from `fc1.bias`
- Removing input row `j` from `fc2.weights`

Both changes happen together via `prune_outputs(keep)` on fc1 and `prune_inputs(keep)` on fc2.

---

## Sparsity Sweep

Sparsity levels: `0%, 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%, 60%, 70%, 80%, 90%`

Fine resolution at low sparsity (5% steps to 30%) to precisely locate AUROC-retention thresholds. Coarser at high sparsity.

**At each sparsity level, three measurements:**
1. Post-hoc AUROC — prune from baseline checkpoint, evaluate immediately on test set
2. Fine-tuned AUROC — re-initialize Adam state (fresh moments, t=0), fine-tune for 10 epochs at LR=0.001, evaluate on test set
3. Total parameters remaining across all four FC layers + both embedding tables

**AUROC retention:**
```
retention(s) = AUROC(pruned at s%) / AUROC(baseline)
```

**Threshold reporting:** `retention_95_threshold_sparsity` records the **last sparsity level at or above** the 0.95 threshold (i.e., the highest sparsity where retention ≥ 0.95). Same convention for 0.90.

Each sparsity level is an independent branch from the same trained baseline (no cumulative pruning). Accuracy is defined as fraction of test examples correctly classified with a prediction threshold of 0.5 (score ≥ 0.5 → positive). This matches the definition used in `train_standard_mlp`.

The embedding tables (TF and Gene) are not pruned and remain at full size throughout. `compression_ratio` therefore has a floor above 0 even at 90% FC sparsity, since embedding parameters dominate the total count.

---

## Output

Results written to `results/neuron_pruning_results.json`:

```json
{
  "seed": 42,
  "alpha": 0.5,
  "fine_tune_epochs": 10,
  "fine_tune_lr": 0.001,
  "baseline_auroc": 0.xxx,
  "baseline_accuracy": 0.xxx,
  "baseline_params": 5200000,
  "results": [
    {
      "sparsity": 0.10,
      "neurons_removed_per_tower": 51,
      "neurons_removed_total": 102,
      "params_remaining": 4200000,
      "compression_ratio": 0.81,
      "posthoc_auroc": 0.xxx,
      "posthoc_accuracy": 0.xxx,
      "posthoc_retention": 0.xxx,
      "finetuned_auroc": 0.xxx,
      "finetuned_accuracy": 0.xxx,
      "finetuned_retention": 0.xxx
    }
  ],
  "retention_95_threshold_sparsity": 0.XX,
  "retention_90_threshold_sparsity": 0.XX
}
```

`neurons_removed_per_tower`: count removed from one tower's fc1 layer.
`neurons_removed_total`: sum across both towers (always 2× per_tower since equal sparsity % is applied to both 512-neuron towers).
`compression_ratio`: `params_remaining / baseline_params`.

---

## Implementation Plan

### 1. `src/models/nn.rs` — two new methods on `LinearLayer`

- `prune_outputs(keep_indices: &[usize])` — rebuild weights as `(in_dim, keep_count)`, bias as `(keep_count,)`, reset grad arrays to matching shape
- `prune_inputs(keep_indices: &[usize])` — rebuild weights as `(keep_count, out_dim)`, reset grad arrays

### 2. `src/models/hybrid_embeddings.rs` — new types and methods

**Prerequisite:** Add `#[derive(Clone)]` to `HybridEmbeddingModel`. All fields are either `Array2<f32>`, `Array1<f32>`, `LinearLayer` (already derives Clone), `f32`, `usize`, or `Option<Array2<f32>>` — all are `Clone`. This is required so the binary can snapshot the trained baseline and branch it independently at each sparsity level.

New supporting structs:
- `NeuronStats` — stores four separate weight norm vectors (one per FC layer) plus per-fc1-neuron activation counts:
  ```
  tf_fc1_activation_counts: Vec<u32>,   // length 512, counts per neuron
  gene_fc1_activation_counts: Vec<u32>, // length 512
  total_examples: usize,
  tf_fc1_col_norms: Vec<f32>,           // ||fc1.weights.column(j)||₂ for j=0..511
  tf_fc2_row_norms: Vec<f32>,           // ||fc2.weights.row(j)||₂ for j=0..511
  gene_fc1_col_norms: Vec<f32>,
  gene_fc2_row_norms: Vec<f32>,
  ```
  Combination into `weight_magnitude(j)` happens inside `importance_scores()`, not here.
- `LayerScores` — final importance scores: `{ tf_fc1: Vec<f32>, gene_fc1: Vec<f32> }` (scores only for the prunable fc1 output neurons; fc2 follows structurally)

New methods:
- `profile_activations(&mut self, batches: &[Batch]) -> NeuronStats`
  Inference-only pass. For each batch: run forward() (which writes to `self.tf_h1` and `self.gene_h1` caches), read those caches, count neurons > 0.0. **Does not call backward() or any LinearLayer gradient accumulation.** Weight column/row L2 norms are computed once from the weight matrices before the loop (not per-batch).
- `importance_scores(&self, stats: &NeuronStats, alpha: f32) -> LayerScores`
  Normalizes activation_freq and weight_magnitude independently per tower, combines with α.
- `prune_to_sparsity(&mut self, scores: &LayerScores, sparsity: f32)`
  Computes keep_indices as the top-(1-sparsity) neurons by score. Calls `tf_fc1.prune_outputs()`, `tf_fc2.prune_inputs()`, `gene_fc1.prune_outputs()`, `gene_fc2.prune_inputs()`.

### 3. `src/bin/neuron_pruning.rs` — new experiment binary

- Re-train baseline model from scratch (seed=42, same config as `train_standard_mlp`)
- Record baseline AUROC and accuracy
- Phase 1: profile activations (inference-only pass over training set)
- Phase 2: for each sparsity level — clone baseline model, prune, evaluate, record post-hoc metrics
- Phase 3: for each sparsity level — clone pruned model, fresh Adam state, fine-tune 10 epochs, evaluate, record fine-tuned metrics
- Write `results/neuron_pruning_results.json`

No changes to training pipeline, data loading, config, or any existing binaries.

---

## Success Criteria

- Sparsity-vs-AUROC curve produced for both post-hoc and fine-tuned variants
- 95% AUROC-retention threshold sparsity identified
- 90% AUROC-retention threshold sparsity identified
- Results fully reproducible from seed=42 with no external checkpoint dependency
- `compression_ratio` and `neurons_removed_total` reported at each level
