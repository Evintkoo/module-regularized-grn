# Reviewer 2 Revisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address all Reviewer 2 major concerns (M1–M7) and minor comments (m1–m10) to bring the TNNLS paper from Major Revision to acceptance.

**Architecture:** Five experimental additions (element-wise product ablation, AUPRC+ensemble-AUROC+ROC data in training scripts, cross-encoder pruning, multi-seed pruning, temperature grid search) plus comprehensive paper text revision.

**Tech Stack:** Rust + ndarray (existing), serde_json, plotters; LaTeX (IEEEtran)

---

## File Map

### New files
- `src/models/cross_encoder_no_product.rs` — CrossEncoderNoProduct model (input: 2×embed + 2×expr, no element-wise product)
- `src/bin/train_ablation_no_product.rs` — Train ablation at 1:1 and 5:1, 5 seeds → `results/ablation_no_product_1to1.json` + `results/ablation_no_product_5to1.json`
- `src/bin/cross_encoder_pruning.rs` — Same pruning protocol as neuron_pruning.rs but on CrossEncoderModel → `results/cross_encoder_pruning.json`
- `src/bin/temperature_search.rs` — Two-tower τ ∈ {0.01, 0.05, 0.1, 0.5, 1.0}, 3 seeds, 1:1 only → `results/temperature_search.json`

### Modified files
- `src/models/mod.rs` — Add `pub mod cross_encoder_no_product`
- `src/bin/train_standard_mlp.rs` — Add `calculate_auprc`, ensemble AUROC, save raw predictions + labels for best seed
- `src/bin/train_cross_encoder.rs` — Same additions as above
- `src/bin/neuron_pruning.rs` — Run seeds [42, 123, 456] instead of just 42; report mean ± std retention
- `src/bin/generate_figures.rs` — Add multi-model ROC comparison figure (fig5_roc_comparison) and PR curve comparison (fig6_pr_comparison) reading from saved prediction JSONs
- `paper/paper_tnnls.tex` — Comprehensive revision addressing all review concerns

---

## Task 1: CrossEncoderNoProduct model

**Files:**
- Create: `src/models/cross_encoder_no_product.rs`
- Modify: `src/models/mod.rs`

- [ ] **Step 1: Write `CrossEncoderNoProductModel`**

Copy `cross_encoder.rs` exactly. The only changes:
1. `input_dim = 2 * embed_dim + 2 * expr_dim` (remove the `embed_dim` product term)
2. In `forward()`: remove the `let interaction = &tf_emb * &gene_emb;` line and remove `interaction.view()` from the concatenate
3. In `backward()`: remove the `g_interaction` split/backprop; only split `g_tf_emb` (0..e) and `g_gene_emb` (e..2e); expression grads are 2e..(2e+2x) — discard

```rust
// src/models/cross_encoder_no_product.rs
// Header: CrossEncoderNoProduct — same as CrossEncoder but WITHOUT the element-wise
// product interaction term. Input = [tf_emb | gene_emb | tf_expr | gene_expr].
// Used for the ablation in Reviewer-2 response (M1).
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array2, Axis};

pub struct CrossEncoderNoProductModel {
    pub tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    pub gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,
    embed_dim: usize,
    expr_dim: usize,
    pub fc1: LinearLayer,
    pub fc2: LinearLayer,
    pub fc3: LinearLayer,
    tf_indices_cache:   Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_embed_cache:     Option<Array2<f32>>,
    gene_embed_cache:   Option<Array2<f32>>,
    h1_pre_cache:       Option<Array2<f32>>,
    h2_pre_cache:       Option<Array2<f32>>,
}
```

Key difference in `new()`: `input_dim = 2 * embed_dim + 2 * expr_dim`

Key difference in `forward()` — build input without interaction:
```rust
let input = ndarray::concatenate![
    Axis(1),
    tf_emb.view(),
    gene_emb.view(),
    tf_expr.view(),
    gene_expr.view()
];
```

Key difference in `backward()` — no interaction backprop:
```rust
let g_tf_emb   = grad_input.slice(ndarray::s![.., 0..e]).to_owned();
let g_gene_emb = grad_input.slice(ndarray::s![.., e..2*e]).to_owned();
// expression grads (2e..2e+2x) discarded
// accumulate into embedding grad tables as in CrossEncoderModel
```

- [ ] **Step 2: Export from mod.rs**

In `src/models/mod.rs` add:
```rust
pub mod cross_encoder_no_product;
```

- [ ] **Step 3: Build to check compilation**

```bash
cargo build --release 2>&1 | tail -20
```
Expected: compiles cleanly.

---

## Task 2: Ablation training script

**Files:**
- Create: `src/bin/train_ablation_no_product.rs`

- [ ] **Step 1: Write the script**

Copy `train_cross_encoder.rs` in full. Then:
1. Change import to `use module_regularized_grn::models::cross_encoder_no_product::CrossEncoderNoProductModel;`
2. Replace all `CrossEncoderModel` with `CrossEncoderNoProductModel`
3. Remove the fc3.grad_{weights,bias} from AdamState — wait, CrossEncoderNoProductModel still has fc3. The only structural difference is `input_dim`. The Adam state struct is identical.
4. Output files: `results/ablation_no_product_1to1.json` and `results/ablation_no_product_5to1.json`
5. Keep the same AUPRC function added in Task 3 (do Task 3 first, actually — see execution order note)
6. Model "name" field in JSON: `"cross_encoder_no_product"`

- [ ] **Step 2: Build**

```bash
cargo build --release --bin train_ablation_no_product 2>&1 | tail -20
```

- [ ] **Step 3: Run**

```bash
cargo run --release --bin train_ablation_no_product 2>&1 | tail -30
```
Expected: produces `results/ablation_no_product_1to1.json` and `results/ablation_no_product_5to1.json` with seed_aurocs, seed_f1s, seed_auprcs, ensemble_auroc.

---

## Task 3: Add AUPRC + ensemble AUROC to train_standard_mlp.rs

**Files:**
- Modify: `src/bin/train_standard_mlp.rs`

- [ ] **Step 1: Add `calculate_auprc` function** (after `calculate_f1`)

```rust
fn calculate_auprc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels.iter())
        .map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    if n_pos == 0.0 { return 0.0; }
    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut ap = 0.0f32;
    let mut prev_recall = 0.0f32;
    for (_, &label) in &pairs {
        if label == 1.0 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let recall    = tp / n_pos;
        let precision = tp / (tp + fp);
        ap += precision * (recall - prev_recall);
        prev_recall = recall;
    }
    ap
}
```

- [ ] **Step 2: Collect seed_auprcs in the per-seed loop**

In the seed loop, after `evaluate_detailed` returns `(test_acc, test_auroc, test_f1)`, also compute auprc with a second pass of the test predictions. The cleanest way: modify `evaluate_detailed` to also return auprc, or create a separate `evaluate_all_metrics` that returns `(acc, auroc, f1, auprc, Vec<f32> preds, Vec<f32> labels)`.

Replace the `evaluate_detailed` signature to also return `(f32, f32, f32, f32)` where the 4th is auprc, using the new `calculate_auprc`.

Also return the raw predictions vector from `evaluate_detailed` so we don't need the second pass below.

- [ ] **Step 3: Compute ensemble AUROC**

After the seed loop, after computing `ensemble_preds`, add:
```rust
let ensemble_auroc = calculate_auroc(&ensemble_preds, &test_labels_once);
let ensemble_auprc = calculate_auprc(&ensemble_preds, &test_labels_once);
```

- [ ] **Step 4: Save predictions for best seed**

Save best_test_preds + test_labels_once as `results/two_tower_1to1_predictions.json` (or `5to1` variant) for ROC/PR figure generation:
```rust
let pred_out = serde_json::json!({
    "labels": test_labels_once.iter().map(|&x| x as f64).collect::<Vec<_>>(),
    "predictions": best_test_preds.iter().map(|&x| x as f64).collect::<Vec<_>>(),
});
let pred_file = if neg_ratio == 1 {
    "results/two_tower_1to1_predictions.json"
} else {
    "results/two_tower_5to1_predictions.json"
};
std::fs::write(pred_file, serde_json::to_string_pretty(&pred_out)?)?;
```

- [ ] **Step 5: Update the result JSON to include new fields**

```rust
let result = serde_json::json!({
    "model": "two_tower",
    "neg_ratio": neg_ratio,
    "seeds": [42u64, 123, 456, 789, 1337],
    "seed_accuracies": seed_accuracies,
    "seed_aurocs": seed_aurocs,
    "seed_f1s": seed_f1s,
    "seed_auprcs": seed_auprcs,          // NEW
    "mean_accuracy": mean_acc,
    "std_accuracy": std_acc,
    "ensemble_accuracy": ensemble_acc,
    "ensemble_auroc": ensemble_auroc,    // NEW
    "ensemble_auprc": ensemble_auprc,    // NEW
    "bootstrap_ci_lower": ci_lower,
    "bootstrap_ci_upper": ci_upper,
});
```

- [ ] **Step 6: Build and run**

```bash
cargo build --release --bin train_standard_mlp && \
cargo run --release --bin train_standard_mlp 2>&1 | tail -20
```

---

## Task 4: Add AUPRC + ensemble AUROC to train_cross_encoder.rs

**Files:**
- Modify: `src/bin/train_cross_encoder.rs`

- [ ] **Same steps as Task 3 but for CrossEncoderModel**

Same `calculate_auprc` function, same changes to `evaluate_detailed`, same JSON output fields. Output files already named `results/cross_encoder_1to1.json` and `results/cross_encoder_5to1.json`. Prediction files: `results/cross_encoder_1to1_predictions.json` and `results/cross_encoder_5to1_predictions.json`.

Note: `train_cross_encoder.rs` currently uses `lr = 0.005`. The paper says 0.001 was used for comparable conditions. Verify the existing JSON results match the paper tables before re-running — if they do, do NOT change the lr (the existing results are correct). Only add the new metric fields; the re-run will reproduce the same results since seeds are fixed.

- [ ] **Build and run**

```bash
cargo build --release --bin train_cross_encoder && \
cargo run --release --bin train_cross_encoder 2>&1 | tail -20
```

---

## Task 5: Multi-seed pruning for two-tower

**Files:**
- Modify: `src/bin/neuron_pruning.rs`

- [ ] **Step 1: Run seeds [42, 123, 456] instead of just 42**

Wrap the existing single-seed training + profiling + sweep loop in a `for &seed in &[42u64, 123, 456]` outer loop.

Per-seed: record `Vec<PruningResult>` (post-hoc and fine-tuned retention at each sparsity level).

After outer loop: compute per-sparsity mean and std of post-hoc retention and fine-tuned retention across the 3 seeds.

- [ ] **Step 2: Update JSON output schema**

```json
{
  "seeds": [42, 123, 456],
  "baseline_auroc_mean": 0.xxxx,
  "baseline_auroc_std": 0.xxxx,
  "results": [
    {
      "sparsity": 0.0,
      "neurons_per_tower": 512,
      "posthoc_retention_mean": 1.0,
      "posthoc_retention_std": 0.0,
      "finetuned_retention_mean": 1.0xxx,
      "finetuned_retention_std": 0.0xxx
    },
    ...
  ]
}
```

Output to `results/neuron_pruning_multiseed.json` (keep the old `neuron_pruning_results.json` for backward compat).

- [ ] **Step 3: Build and run**

```bash
cargo build --release --bin neuron_pruning && \
cargo run --release --bin neuron_pruning 2>&1 | tail -30
```

---

## Task 6: Cross-encoder pruning

**Files:**
- Create: `src/bin/cross_encoder_pruning.rs`

- [ ] **Step 1: Write the script**

Copy `neuron_pruning.rs` as starting point. Key differences:

1. Use `CrossEncoderModel` (from `cross_encoder.rs`) instead of `HybridEmbeddingModel`
2. Profiling: the cross-encoder has fc1, fc2, fc3 (hidden_dim=512 each). Profile fc1 neurons (512 per model, not per tower — there's only one pathway).
3. Importance scoring: same formula but only one tower's fc1/fc2 instead of two; `importance(j) = 0.5 * activation_freq(j) + 0.5 * weight_magnitude(j)` where weight_magnitude uses fc1's column j and fc2's row j.
4. Pruning: delete column j from fc1.weights and row j from fc2.weights (fc3 is the output layer, don't prune it).
5. Same sparsity levels, same fine-tuning protocol (10 epochs, fresh Adam).
6. Output: `results/cross_encoder_pruning.json` with same schema as `neuron_pruning_results.json`.
7. Single seed (42) is fine since the goal is a qualitative comparison with the two-tower.

- [ ] **Step 2: Build and run**

```bash
cargo build --release --bin cross_encoder_pruning && \
cargo run --release --bin cross_encoder_pruning 2>&1 | tail -30
```

---

## Task 7: Temperature grid search for two-tower

**Files:**
- Create: `src/bin/temperature_search.rs`

- [ ] **Step 1: Write the script**

Copy `train_standard_mlp.rs`, wrap the outer loop to iterate over:
```rust
let temperatures = [0.01f32, 0.05, 0.1, 0.5, 1.0];
```
For each temperature, train 3 seeds (42, 123, 456) at 1:1 ratio only. Report mean AUROC ± std. Output: `results/temperature_search.json`.

JSON schema:
```json
{
  "temperatures": [0.01, 0.05, 0.1, 0.5, 1.0],
  "results": [
    { "tau": 0.01, "mean_auroc": 0.xxxx, "std_auroc": 0.xxxx, "seed_aurocs": [...] },
    ...
  ]
}
```

- [ ] **Step 2: Build and run**

```bash
cargo build --release --bin temperature_search && \
cargo run --release --bin temperature_search 2>&1 | tail -30
```

---

## Task 8: ROC + PR comparison figures

**Files:**
- Modify: `src/bin/generate_figures.rs`

The existing `compute_roc_curve` and `draw_roc_curve` functions already exist. Add:

- [ ] **Step 1: Add `draw_roc_comparison` function**

Draw all 4 conditions on one plot: two-tower 1:1 (blue), cross-encoder 1:1 (red), two-tower 5:1 (blue dashed), cross-encoder 5:1 (red dashed). Reads from `results/*_predictions.json`.

Output: `paper/figures/fig5_roc_comparison.svg` (and .png).

- [ ] **Step 2: Add `draw_pr_comparison` function**

Same 4 conditions on a PR curve plot. Output: `paper/figures/fig6_pr_comparison.svg`.

- [ ] **Step 3: Wire into `main()`**

Call both new functions from `main()`.

- [ ] **Step 4: Build and run**

```bash
cargo build --release --bin generate_figures && \
cargo run --release --bin generate_figures 2>&1
```

---

## Task 9: Paper text revision

**Files:**
- Modify: `paper/paper_tnnls.tex`

This is the final and most important task. Edit the LaTeX directly addressing every review concern:

### 9A: Title (m3)
Change title to:
```
Modular versus Monolithic Neural Architectures for Gene Regulatory Network
Inference: An Empirical Comparison Under Parameter-Matched Conditions
```
(Drops "Capacity, Robustness, and Representational Efficiency" which overpromises; adds "Empirical Comparison Under Parameter-Matched Conditions" which is accurate.)

### 9B: Abstract (m1, M1, M6)
- Replace accuracy-first framing with AUROC-first throughout
- Add AUPRC values for both architectures under both conditions
- Add ensemble AUROC
- Add one sentence summarizing ablation result (no-product variant) with appropriate hedging
- Remove the claim that cosine scoring is "disproportionately sensitive" — soften to "more sensitive"

### 9C: Contribution C1 (M1, M4)
Update C1 to state explicitly:
1. The comparison includes an element-wise product inductive bias in the cross-encoder
2. An ablation without the product term has now been run (report result from Task 2)
3. The learning rate was shared; temperature was not grid-searched (now searched — report from Task 7)

### 9D: Contribution C2 (M6)
Add AUPRC to the C2 evidence: "both architectures exhibit AUPRC degradation under imbalance (two-tower −X pts, cross-encoder −Y pts)".

### 9E: Methods — Evaluation (M6, m4, m5)
- Add AUPRC as a fourth metric alongside accuracy, AUROC, F1
- Clarify F1 threshold: "Macro F1 is computed at a fixed decision threshold of 0.5."
- State ensemble AUROC is reported

### 9F: Results — Tables I–III (M6, m5)
- Add `AUPRC` column to Table I (1:1 results) and Table III (imbalance results)
- Add `Ensemble AUROC` row to Table I
- Add Table V (new): Ablation — cross-encoder without element-wise product, comparing with and without product under 1:1 and 5:1

### 9G: Results — ROC/PR curves (m7)
- Add Figure 5: multi-model ROC curve comparison (all 4 conditions on one plot)
- Add Figure 6: multi-model PR curve comparison (all 4 conditions)

### 9H: Results — Pruning section (M5a, M5b, m6)
- Update Table IV to report mean ± std across 3 seeds instead of single-seed values
- Remove the footnote claiming "different AUROC estimator" (they are actually identical functions; the baseline difference is due to re-training a fresh model — correct the explanation)
- Add a paragraph reporting the cross-encoder pruning results (from Task 6) and comparing to two-tower redundancy

### 9I: Discussion — temperature grid search (M4)
- Add paragraph reporting temperature search results (Task 7): "A grid search over τ ∈ {0.01, 0.05, 0.1, 0.5, 1.0} on the two-tower model shows that τ=0.05 achieves [rank] AUROC, confirming [or: revealing that optimal τ=X achieves Y AUROC]. The main comparison was re-run at optimal τ; results in Table VI."
- If a different τ is substantially better, re-run main experiments with that τ and add Table VI.

### 9J: Discussion — limitations updates (M1, M2, M3, M4, M5, M6)
Revise each limitation paragraph to reflect what has been added:
- M1 (element-wise product): Partially addressed — ablation run and reported. State residual limitation if product term accounts for most of the gap.
- M2 (no baselines): Remain as limitation; add BEELINE citation and note it as planned future work
- M3 (single dataset): Remain as limitation; hedge all architectural recommendations
- M4 (hyperparameters): Addressed for τ (grid search done); learning rate matching remains a limitation
- M5 (single-seed pruning): Addressed — multi-seed results added
- M6 (AUPRC): Addressed — now included
- M7 (implementation): Add brief validation note (convergence curve matches expected BCE behavior; gradient norms tracked and clipped consistently)

### 9K: Welch df footnote (m2)
Add footnote to the C2 t-test result showing Welch-Satterthwaite calculation:
```
The Welch--Satterthwaite effective degrees of freedom are
$\nu = (s_1^2/n + s_2^2/n)^2 / [(s_1^2/n)^2/(n{-}1) + (s_2^2/n)^2/(n{-}1)]$
where $s_1^2 = 0.0059^2$, $s_2^2 = 0.0154^2$, $n = 5$, giving $\nu \approx 5.2$.
```

### 9L: Conclusion (M3)
Hedge the conclusion: replace "the cross-encoder is the preferred architecture" with "on this dataset, the cross-encoder is the preferred architecture when discriminative accuracy is the primary criterion".

- [ ] **For each subsection (9A–9L): Edit tex, verify LaTeX compiles with `scripts/make_pdf.sh` or `pdflatex`**

---

## Execution Order

Tasks 1–8 are code/experiment tasks. Task 9 (paper) depends on experimental results from Tasks 2–7.

**Parallel batch 1** (no dependencies between them):
- Tasks 1 + 2 (new model + training script) → can run immediately
- Tasks 3 + 4 (add AUPRC to existing training scripts) → can run immediately
- Task 7 (temperature search) → can run immediately after Task 3 (needs calculate_auprc pattern)

**Parallel batch 2** (each depends on trained models):
- Task 5 (multi-seed pruning) → run after two-tower training verified
- Task 6 (cross-encoder pruning) → independent, run anytime
- Task 8 (figures) → needs prediction JSON files from Tasks 3–4

**Final:**
- Task 9 (paper) → after all experiments complete and results JSONs are updated

---

## Verification

After all tasks complete:
```bash
# All result JSONs exist
ls results/ablation_no_product_1to1.json results/ablation_no_product_5to1.json \
   results/neuron_pruning_multiseed.json results/cross_encoder_pruning.json \
   results/temperature_search.json \
   results/two_tower_1to1_predictions.json results/cross_encoder_1to1_predictions.json

# New figures exist
ls paper/figures/fig5_roc_comparison.svg paper/figures/fig6_pr_comparison.svg

# Paper compiles
cd paper && pdflatex paper_tnnls.tex && echo "PDF OK"
```
