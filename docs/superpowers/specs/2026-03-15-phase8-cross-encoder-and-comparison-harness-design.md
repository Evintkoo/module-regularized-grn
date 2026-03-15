# Phase 8 Sub-Project 1: Cross-Encoder Baseline + Comparison Harness

## Goal

Implement Model D (cross-encoder monolithic baseline) and a unified comparison harness, enabling the core dissertation comparison between monolithic and modular architectures across two negative-sampling regimes (1:1 and 5:1).

## Context

This is Phase 8 of the dissertation, which compares modular (two-tower) vs monolithic (cross-encoder) neural networks for GRN inference. Sub-project 1 establishes the baseline comparison before the collaboration algorithms (InfoNCE, DML, co-teaching) are added in Sub-project 2.

**Existing baseline:** Two-tower `HybridEmbeddingModel` in `src/models/hybrid_embeddings.rs` — 80.14% single-model accuracy, 83.06% ensemble (1:1 negative ratio, 5 seeds).

---

## Architecture

### Model D: CrossEncoderModel

**File:** `src/models/cross_encoder.rs`

A monolithic MLP taking joint TF+Gene features. No separate encoding towers — the cross-encoder sees all information simultaneously and learns a single joint representation.

**Input construction (1558 dims total):**

| Component | Dims | Description |
|---|---|---|
| TF embedding | 512 | Learned embedding for TF identity |
| Gene embedding | 512 | Learned embedding for gene identity |
| TF expression | 11 | Cell-type expression profile for TF |
| Gene expression | 11 | Cell-type expression profile for gene |
| TF ⊙ Gene (interaction) | 512 | Element-wise product of TF and Gene embeddings |

**Input dimension:** computed dynamically as `3 * embed_dim + 2 * expr_dim` (= 1558 for embed_dim=512, expr_dim=11 on the brain dataset). Do not hardcode 1558.

**Layers:**
- FC1: `(3*embed_dim + 2*expr_dim)` → 512, ReLU
- FC2: 512 → 512, ReLU
- FC3: 512 → 1, **no activation (raw logit)**

**Loss and sigmoid convention:** FC3 outputs a raw logit. `bce_loss` and `bce_loss_backward` from `nn.rs` apply sigmoid internally — do NOT apply sigmoid in FC3. This matches the convention in `HybridEmbeddingModel`. The forward pass returns a raw logit; the training loop calls `bce_loss(logit, label)` for the loss value.

**Parameter count:** ~1.06M FC params + embedding tables (same size as two-tower). The FC layer count is parameter-matched to the two-tower's dual-tower FC stack. Temperature is not used — the cross-encoder has no dot-product scoring step.

**Implementation:** Uses `LinearLayer`, `relu`, `relu_backward`, `bce_loss`, `bce_loss_backward` from `src/models/nn.rs`. Manual backpropagation, caching forward activations for the backward pass — same pattern as `hybrid_embeddings.rs`.

**Optimizer:** Adam implemented inline in `train_cross_encoder.rs` — **new code, no existing reference to copy**. Use the same inline-at-update-step structure as `train_ensemble.rs` (no standalone optimizer struct), but with full Adam: maintain per-parameter `m` and `v` moment vectors (same shape as each parameter array), update rule β1=0.9, β2=0.999, ε=1e-8, LR=0.005, bias-corrected. Gradient clipping at norm=1.0 applied before the Adam update (compute global gradient norm across all params, scale down if > 1.0).

---

## Training

### train_cross_encoder.rs

**File:** `src/bin/train_cross_encoder.rs`

Trains Model D under both negative-sampling regimes in sequence:

1. **1:1 ratio**, 5 seeds → `results/cross_encoder_1to1.json`
2. **5:1 ratio**, 5 seeds → `results/cross_encoder_5to1.json`

**Hyperparameters (held constant, matching two-tower):**

| Param | Value |
|---|---|
| Learning rate | 0.005 |
| Embed dim | 512 |
| Hidden dim | 512 |
| Epochs | 60 |
| Batch size | 256 |
| Seeds | [42, 123, 456, 789, 1337] |

**Result JSON schema** (same for both output files):
```json
{
  "model": "cross_encoder",
  "neg_ratio": 1,
  "seed_accuracies": [0.801, 0.798, ...],
  "seed_aurocs": [0.812, 0.815, ...],
  "seed_f1s": [0.839, 0.835, ...],
  "mean_accuracy": 0.800,
  "std_accuracy": 0.017,
  "ensemble_accuracy": 0.830,
  "bootstrap_ci_lower": 0.793,
  "bootstrap_ci_upper": 0.810
}
```

### train_standard_mlp.rs (modification)

The existing file trains a single seed from config. Two things must be added:

1. **`--neg-ratio <N>` CLI flag** (default: 1) — controls the negative-to-positive ratio for batch sampling.
2. **Seed iteration loop** — iterate over seeds `[42, 123, 456, 789, 1337]`, train one model per seed, collect per-seed accuracy, AUROC, and F1.

**New evaluation work required:** The existing file only computes accuracy. Add per-seed AUROC (sort predictions by score, compute trapezoidal AUC) and F1 (precision/recall at threshold 0.5) — use the `calculate_auroc` helper already present in `train_ensemble.rs` as a reference.

**Output:** writes unified schema to:
- `results/two_tower_1to1.json` (when `--neg-ratio 1`, the default)
- `results/two_tower_5to1.json` (when `--neg-ratio 5`)

Schema fields: `model` ("two_tower"), `neg_ratio`, `seed_accuracies`, `seed_aurocs`, `seed_f1s`, `mean_accuracy`, `std_accuracy`, `ensemble_accuracy`, `bootstrap_ci_lower`, `bootstrap_ci_upper`.

**Bootstrap CI:** Use `bootstrap_ci_accuracy` from `src/bin/statistical_analysis.rs` (n=1000 bootstrap samples, 95% CI, seed=42). The CI is computed over test-set predictions from the best (highest val-accuracy) seed model, not over per-seed accuracies.

**neg-ratio scope:** `--neg-ratio N` changes only the training batch negative sampling (`positives.len() * neg_ratio` negatives per batch). Val and test sets are always evaluated at 1:1 to keep metrics comparable across ratios.

The existing `results/standard_mlp_results.json` output should be kept alongside the new file for backward compatibility.

---

## Comparison Harness

### compare_models.rs

**File:** `src/bin/compare_models.rs`

Aggregation-only — no training. Reads available result JSONs and prints a comparison table.

**Input files (missing files are skipped with a note):**
- `results/two_tower_1to1.json`
- `results/two_tower_5to1.json`
- `results/cross_encoder_1to1.json`
- `results/cross_encoder_5to1.json`

**Output (stdout):**
```
Model               | Neg Ratio | Accuracy (±std)  | AUROC  | F1      | 95% CI
--------------------|-----------|------------------|--------|---------|------------------
Two-Tower           |      1:1  | 80.14% ± 1.70%  | 0.814  | 83.90%  | [79.3%, 81.0%]
Two-Tower           |      5:1  | ...              | ...    | ...     | ...
Cross-Encoder       |      1:1  | ...              | ...    | ...     | ...
Cross-Encoder       |      5:1  | ...              | ...    | ...     | ...
```

**Output file:** `results/model_comparison.json` — machine-readable version for `generate_tables` to pick up in the paper pipeline.

---

## File Map

### Create
- `src/models/cross_encoder.rs` — CrossEncoderModel struct
- `src/bin/train_cross_encoder.rs` — trains Model D at 1:1 and 5:1
- `src/bin/compare_models.rs` — reads JSONs, prints comparison table

### Modify
- `src/models/mod.rs` — add `pub mod cross_encoder; pub use cross_encoder::CrossEncoderModel;`
- `src/bin/train_standard_mlp.rs` — add `--neg-ratio` CLI flag
- `Cargo.toml` — add `[[bin]]` stanzas for `train_cross_encoder` and `compare_models`

---

## Testing Strategy

Each new file includes `#[cfg(test)]` unit tests:

- **`cross_encoder.rs`**: forward pass shape test (batch=4, correct output dims); gradient flow test (loss decreases after one update step)
- **`train_cross_encoder.rs`**: smoke test — 2 epochs, 2 seeds, tiny synthetic dataset, result JSON written and parseable
- **`compare_models.rs`**: table rendering test — given mock result structs, output string contains expected column headers and model names

---

## What This Enables

After this sub-project:
- Model D results exist at both negative ratios
- Model E (two-tower) results exist at both negative ratios
- `compare_models` produces the baseline comparison table for the dissertation

Sub-project 2 (InfoNCE, DML, co-teaching) slots in by adding more result JSON files — `compare_models` picks them up automatically via the skip-missing design.
