# Neuron Pruning Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a structured neuron pruning experiment that sweeps sparsity 0–90%, measures AUROC before/after fine-tuning at each level, and identifies the thresholds where 95% and 90% AUROC retention are first violated.

**Architecture:** Three-phase pipeline: (1) train baseline two-tower MLP from scratch (seed=42), (2) profile neuron activity across full training set using inference-only forward passes, (3) sweep 13 sparsity levels — for each: clone baseline, prune fc1 outputs + fc2 inputs, measure post-hoc AUROC, fine-tune 10 epochs with fresh Adam state, re-measure AUROC. Results written to `results/neuron_pruning_results.json`.

**Tech Stack:** Rust, ndarray, serde_json. Run tests with `cargo test`. Run experiment with `cargo run --release --bin neuron_pruning`.

**Spec:** `docs/superpowers/specs/2026-03-17-neuron-pruning-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/models/nn.rs` | Modify | Add `prune_outputs` and `prune_inputs` to `LinearLayer` |
| `src/models/hybrid_embeddings.rs` | Modify | Add `#[derive(Clone)]`; add `NeuronStats`, `LayerScores` structs; add `profile_activations`, `importance_scores`, `prune_to_sparsity` methods |
| `src/bin/neuron_pruning.rs` | Create | Experiment binary: train baseline, profile, sweep sparsity, fine-tune, write JSON |

No other files are modified.

---

## Chunk 1: `LinearLayer` Pruning Methods

### Task 1: Add `prune_outputs` to `LinearLayer` in `src/models/nn.rs`

**Files:**
- Modify: `src/models/nn.rs`

- [ ] **Step 1: Write the failing test**

  Add this test to the existing `#[cfg(test)]` block in `src/models/nn.rs`:

  ```rust
  #[test]
  fn test_prune_outputs() {
      // layer: in=4, out=3
      let mut layer = LinearLayer::new(4, 3, 42);
      // Override weights/bias with known values for testing
      layer.weights = ndarray::Array2::from_shape_vec((4, 3), vec![
          1.0, 2.0, 3.0,
          4.0, 5.0, 6.0,
          7.0, 8.0, 9.0,
          10.0, 11.0, 12.0,
      ]).unwrap();
      layer.bias = ndarray::Array1::from_vec(vec![0.1, 0.2, 0.3]);

      // Keep output neurons 0 and 2 (drop neuron 1)
      layer.prune_outputs(&[0, 2]);

      assert_eq!(layer.weights.dim(), (4, 2));
      assert_eq!(layer.bias.len(), 2);
      assert_eq!(layer.grad_weights.dim(), (4, 2));
      assert_eq!(layer.grad_bias.len(), 2);

      // Check correct columns were kept
      assert_eq!(layer.weights[[0, 0]], 1.0); // col 0, row 0
      assert_eq!(layer.weights[[0, 1]], 3.0); // col 2, row 0
      assert!((layer.bias[0] - 0.1).abs() < 1e-6);
      assert!((layer.bias[1] - 0.3).abs() < 1e-6);
  }
  ```

- [ ] **Step 2: Run test — expect compile error or FAIL**

  ```bash
  cargo test test_prune_outputs -- --nocapture
  ```
  Expected: compile error `no method named 'prune_outputs'`

- [ ] **Step 3: Implement `prune_outputs`**

  Add this method inside the `impl LinearLayer` block in `src/models/nn.rs`, after `zero_grad`:

  ```rust
  /// Remove output neurons, keeping only those at `keep_indices`.
  /// Resizes weights from (in_dim, out_dim) to (in_dim, keep_count),
  /// bias from (out_dim,) to (keep_count,). Resets grad arrays.
  pub fn prune_outputs(&mut self, keep_indices: &[usize]) {
      let in_dim = self.weights.nrows();
      let k = keep_indices.len();
      let mut new_weights = ndarray::Array2::zeros((in_dim, k));
      let mut new_bias = ndarray::Array1::zeros(k);
      for (new_j, &old_j) in keep_indices.iter().enumerate() {
          new_weights.column_mut(new_j).assign(&self.weights.column(old_j));
          new_bias[new_j] = self.bias[old_j];
      }
      self.weights = new_weights;
      self.bias = new_bias;
      self.grad_weights = ndarray::Array2::zeros((in_dim, k));
      self.grad_bias = ndarray::Array1::zeros(k);
      self.input_cache = None;
  }
  ```

- [ ] **Step 4: Run test — expect PASS**

  ```bash
  cargo test test_prune_outputs -- --nocapture
  ```
  Expected: `test models::nn::tests::test_prune_outputs ... ok`

---

### Task 2: Add `prune_inputs` to `LinearLayer` in `src/models/nn.rs`

**Files:**
- Modify: `src/models/nn.rs`

- [ ] **Step 1: Write the failing test**

  Add to the `#[cfg(test)]` block:

  ```rust
  #[test]
  fn test_prune_inputs() {
      // layer: in=3, out=2
      let mut layer = LinearLayer::new(3, 2, 42);
      layer.weights = ndarray::Array2::from_shape_vec((3, 2), vec![
          1.0, 2.0,
          3.0, 4.0,
          5.0, 6.0,
      ]).unwrap();
      layer.bias = ndarray::Array1::from_vec(vec![0.5, 0.6]);

      // Keep input neurons 0 and 2 (drop neuron 1)
      layer.prune_inputs(&[0, 2]);

      assert_eq!(layer.weights.dim(), (2, 2));
      assert_eq!(layer.bias.len(), 2); // bias unchanged
      assert_eq!(layer.grad_weights.dim(), (2, 2));

      // Check correct rows were kept
      assert_eq!(layer.weights[[0, 0]], 1.0); // row 0, col 0
      assert_eq!(layer.weights[[0, 1]], 2.0); // row 0, col 1
      assert_eq!(layer.weights[[1, 0]], 5.0); // row 2, col 0
      assert_eq!(layer.weights[[1, 1]], 6.0); // row 2, col 1
      // bias unchanged
      assert!((layer.bias[0] - 0.5).abs() < 1e-6);
  }
  ```

- [ ] **Step 2: Run test — expect compile error**

  ```bash
  cargo test test_prune_inputs -- --nocapture
  ```
  Expected: compile error `no method named 'prune_inputs'`

- [ ] **Step 3: Implement `prune_inputs`**

  Add after `prune_outputs` in `impl LinearLayer`:

  ```rust
  /// Remove input neurons, keeping only those at `keep_indices`.
  /// Resizes weights from (in_dim, out_dim) to (keep_count, out_dim).
  /// Bias shape (out_dim,) is unchanged — it depends on output, not input.
  pub fn prune_inputs(&mut self, keep_indices: &[usize]) {
      let out_dim = self.weights.ncols();
      let k = keep_indices.len();
      let mut new_weights = ndarray::Array2::zeros((k, out_dim));
      for (new_i, &old_i) in keep_indices.iter().enumerate() {
          new_weights.row_mut(new_i).assign(&self.weights.row(old_i));
      }
      self.weights = new_weights;
      self.grad_weights = ndarray::Array2::zeros((k, out_dim));
      self.input_cache = None;
      // bias is untouched
  }
  ```

- [ ] **Step 4: Run all nn tests — expect all PASS**

  ```bash
  cargo test --lib models::nn -- --nocapture
  ```
  Expected: 4 tests pass (linear_layer, relu, prune_outputs, prune_inputs)

- [ ] **Step 5: Commit**

  ```bash
  git add src/models/nn.rs
  git commit -m "feat: add prune_outputs and prune_inputs to LinearLayer"
  ```

---

## Chunk 2: `HybridEmbeddingModel` Pruning Infrastructure

### Task 3: Add `#[derive(Clone)]` to `HybridEmbeddingModel`

**Files:**
- Modify: `src/models/hybrid_embeddings.rs`

- [ ] **Step 1: Write the failing test**

  Add to the `#[cfg(test)]` block in `src/models/hybrid_embeddings.rs`:

  ```rust
  #[test]
  fn test_model_clone() {
      let model = HybridEmbeddingModel::new(
          10, 20, 8, 16, 16, 8, 0.05, 0.01, 42
      );
      let cloned = model.clone();
      // Embeddings should be equal but independent
      assert_eq!(model.tf_embed.dim(), cloned.tf_embed.dim());
      assert_eq!(model.tf_fc1.weights.dim(), cloned.tf_fc1.weights.dim());
  }
  ```

- [ ] **Step 2: Run test — expect compile error**

  ```bash
  cargo test test_model_clone -- --nocapture
  ```
  Expected: compile error about Clone not being implemented

- [ ] **Step 3: Add `#[derive(Clone)]`**

  In `src/models/hybrid_embeddings.rs`, change the struct definition:

  ```rust
  // Before:
  pub struct HybridEmbeddingModel {

  // After:
  #[derive(Clone)]
  pub struct HybridEmbeddingModel {
  ```

- [ ] **Step 4: Run test — expect PASS**

  ```bash
  cargo test test_model_clone -- --nocapture
  ```
  Expected: `test models::hybrid_embeddings::tests::test_model_clone ... ok`

---

### Task 4: Add `NeuronStats`, `LayerScores` structs and `profile_activations` to `hybrid_embeddings.rs`

**Files:**
- Modify: `src/models/hybrid_embeddings.rs`

- [ ] **Step 1: Write the failing test**

  Add to the `#[cfg(test)]` block:

  ```rust
  #[test]
  fn test_profile_activations() {
      use std::collections::HashMap;
      use ndarray::Array1;

      let mut model = HybridEmbeddingModel::new(
          5, 10, 4, 8, 8, 4, 0.05, 0.01, 42
      );
      let hidden_dim = 8;

      // Build tiny fake expr maps
      let mut tf_expr: HashMap<usize, Array1<f32>> = HashMap::new();
      let mut gene_expr: HashMap<usize, Array1<f32>> = HashMap::new();
      for i in 0..5  { tf_expr.insert(i,   Array1::zeros(8)); }
      for i in 0..10 { gene_expr.insert(i,  Array1::zeros(8)); }

      // 4 training examples
      let data: Vec<(usize, usize, f32)> = vec![
          (0, 0, 1.0), (1, 1, 0.0), (2, 2, 1.0), (3, 3, 0.0),
      ];

      let stats = model.profile_activations(&data, &tf_expr, &gene_expr, 8, 4);

      assert_eq!(stats.tf_fc1_activation_counts.len(), hidden_dim);
      assert_eq!(stats.gene_fc1_activation_counts.len(), hidden_dim);
      assert_eq!(stats.total_examples, 4);
      assert_eq!(stats.tf_fc1_col_norms.len(), hidden_dim);
      assert_eq!(stats.tf_fc2_row_norms.len(), hidden_dim);
      assert_eq!(stats.gene_fc1_col_norms.len(), hidden_dim);
      assert_eq!(stats.gene_fc2_row_norms.len(), hidden_dim);

      // activation counts must be in [0, total_examples]
      for &c in &stats.tf_fc1_activation_counts {
          assert!(c as usize <= stats.total_examples);
      }
      // weight norms must be non-negative
      for &n in &stats.tf_fc1_col_norms {
          assert!(n >= 0.0);
      }
  }
  ```

- [ ] **Step 2: Run test — expect compile error**

  ```bash
  cargo test test_profile_activations -- --nocapture
  ```
  Expected: compile errors about missing types/methods

- [ ] **Step 3: Add `NeuronStats` and `LayerScores` structs**

  Add these structs at the top of `src/models/hybrid_embeddings.rs`, after the `use` statements:

  ```rust
  use std::collections::HashMap;

  /// Raw activation and weight statistics collected during profiling pass.
  /// All counts are per neuron (index = fc1 output neuron j, 0..hidden_dim-1).
  pub struct NeuronStats {
      pub tf_fc1_activation_counts:   Vec<u32>,  // # examples where TF fc1 neuron j > 0
      pub gene_fc1_activation_counts: Vec<u32>,  // # examples where Gene fc1 neuron j > 0
      pub total_examples: usize,                 // total training examples profiled
      pub tf_fc1_col_norms:   Vec<f32>,  // ||tf_fc1.weights.column(j)||₂
      pub tf_fc2_row_norms:   Vec<f32>,  // ||tf_fc2.weights.row(j)||₂
      pub gene_fc1_col_norms: Vec<f32>,
      pub gene_fc2_row_norms: Vec<f32>,
  }

  /// Final combined importance scores per tower's fc1 neurons (0..hidden_dim-1).
  pub struct LayerScores {
      pub tf_fc1:   Vec<f32>,
      pub gene_fc1: Vec<f32>,
  }
  ```

- [ ] **Step 4: Add `profile_activations` method**

  Add a private helper function before the `impl HybridEmbeddingModel` block:

  ```rust
  fn build_expr_batch_for_profiling(
      indices:  &[usize],
      expr_map: &HashMap<usize, Array1<f32>>,
      expr_dim: usize,
  ) -> Array2<f32> {
      let mut batch = Array2::zeros((indices.len(), expr_dim));
      for (i, &idx) in indices.iter().enumerate() {
          if let Some(expr) = expr_map.get(&idx) {
              batch.row_mut(i).assign(expr);
          }
      }
      batch
  }
  ```

  Then add this method inside `impl HybridEmbeddingModel`:

  ```rust
  /// Inference-only forward pass over all training data.
  /// Accumulates per-neuron activation counts and weight norms.
  /// Does NOT call backward() or accumulate any gradients.
  pub fn profile_activations(
      &mut self,
      data:          &[(usize, usize, f32)],
      tf_expr_map:   &HashMap<usize, Array1<f32>>,
      gene_expr_map: &HashMap<usize, Array1<f32>>,
      expr_dim:      usize,
      batch_size:    usize,
  ) -> NeuronStats {
      let hidden_dim = self.tf_fc1.bias.len();

      // Compute weight norms once — O(hidden_dim * in_dim), not per-batch
      let l2 = |v: ndarray::ArrayView1<f32>| -> f32 {
          v.iter().map(|&x| x * x).sum::<f32>().sqrt()
      };
      let tf_fc1_col_norms:   Vec<f32> = (0..hidden_dim).map(|j| l2(self.tf_fc1.weights.column(j))).collect();
      let tf_fc2_row_norms:   Vec<f32> = (0..hidden_dim).map(|j| l2(self.tf_fc2.weights.row(j))).collect();
      let gene_fc1_col_norms: Vec<f32> = (0..hidden_dim).map(|j| l2(self.gene_fc1.weights.column(j))).collect();
      let gene_fc2_row_norms: Vec<f32> = (0..hidden_dim).map(|j| l2(self.gene_fc2.weights.row(j))).collect();

      let mut tf_counts   = vec![0u32; hidden_dim];
      let mut gene_counts = vec![0u32; hidden_dim];
      let mut total_examples = 0usize;

      for start in (0..data.len()).step_by(batch_size) {
          let end   = (start + batch_size).min(data.len());
          let batch = &data[start..end];
          let bsz   = batch.len();

          let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
          let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
          let tf_e   = build_expr_batch_for_profiling(&tf_idx,   tf_expr_map,   expr_dim);
          let gene_e = build_expr_batch_for_profiling(&gene_idx, gene_expr_map, expr_dim);

          // forward() writes post-ReLU activations to self.tf_h1 and self.gene_h1
          self.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
          // No backward() call — this is inference only

          let tf_h1   = self.tf_h1.as_ref().unwrap();
          let gene_h1 = self.gene_h1.as_ref().unwrap();

          for i in 0..bsz {
              for j in 0..hidden_dim {
                  if tf_h1[[i, j]]   > 0.0 { tf_counts[j]   += 1; }
                  if gene_h1[[i, j]] > 0.0 { gene_counts[j] += 1; }
              }
          }
          total_examples += bsz;
      }

      NeuronStats {
          tf_fc1_activation_counts:   tf_counts,
          gene_fc1_activation_counts: gene_counts,
          total_examples,
          tf_fc1_col_norms,
          tf_fc2_row_norms,
          gene_fc1_col_norms,
          gene_fc2_row_norms,
      }
  }
  ```

- [ ] **Step 5: Run test — expect PASS**

  ```bash
  cargo test test_profile_activations -- --nocapture
  ```
  Expected: `test models::hybrid_embeddings::tests::test_profile_activations ... ok`

---

### Task 5: Add `importance_scores` and `prune_to_sparsity`

**Files:**
- Modify: `src/models/hybrid_embeddings.rs`

- [ ] **Step 1: Write the failing tests**

  Add to `#[cfg(test)]`:

  ```rust
  #[test]
  fn test_importance_scores_range() {
      use std::collections::HashMap;
      use ndarray::Array1;

      let mut model = HybridEmbeddingModel::new(5, 10, 4, 8, 8, 4, 0.05, 0.01, 42);
      let hidden_dim = 8;

      let mut tf_expr: HashMap<usize, Array1<f32>> = HashMap::new();
      let mut gene_expr: HashMap<usize, Array1<f32>> = HashMap::new();
      for i in 0..5  { tf_expr.insert(i,  Array1::zeros(8)); }
      for i in 0..10 { gene_expr.insert(i, Array1::zeros(8)); }

      let data: Vec<(usize, usize, f32)> = (0..8).map(|i| (i % 5, i % 10, 1.0)).collect();
      let stats = model.profile_activations(&data, &tf_expr, &gene_expr, 8, 8);
      let scores = model.importance_scores(&stats, 0.5);

      assert_eq!(scores.tf_fc1.len(), hidden_dim);
      assert_eq!(scores.gene_fc1.len(), hidden_dim);
      for &s in &scores.tf_fc1   { assert!(s >= 0.0 && s <= 1.0 + 1e-5, "score out of range: {}", s); }
      for &s in &scores.gene_fc1 { assert!(s >= 0.0 && s <= 1.0 + 1e-5, "score out of range: {}", s); }
  }

  #[test]
  fn test_prune_to_sparsity_dimensions() {
      let mut model = HybridEmbeddingModel::new(5, 10, 4, 8, 8, 4, 0.05, 0.01, 42);
      let hidden_dim = 8usize;

      // Manually build LayerScores (descending 1.0..0.125 for tf, reverse for gene)
      let tf_scores:   Vec<f32> = (0..hidden_dim).map(|i| 1.0 - i as f32 / hidden_dim as f32).collect();
      let gene_scores: Vec<f32> = (0..hidden_dim).map(|i| i as f32 / hidden_dim as f32).collect();
      let scores = LayerScores { tf_fc1: tf_scores, gene_fc1: gene_scores };

      // Prune 50% → keep 4
      model.prune_to_sparsity(&scores, 0.5);

      let keep = (hidden_dim as f32 * 0.5).round() as usize;
      assert_eq!(model.tf_fc1.bias.len(), keep,   "tf_fc1 output dim wrong");
      assert_eq!(model.tf_fc2.weights.nrows(), keep, "tf_fc2 input dim wrong");
      assert_eq!(model.gene_fc1.bias.len(), keep,   "gene_fc1 output dim wrong");
      assert_eq!(model.gene_fc2.weights.nrows(), keep, "gene_fc2 input dim wrong");
  }
  ```

- [ ] **Step 2: Run tests — expect compile errors**

  ```bash
  cargo test test_importance_scores_range test_prune_to_sparsity_dimensions -- --nocapture
  ```
  Expected: compile errors about missing methods

- [ ] **Step 3: Add private helper `top_k_indices`**

  Add as a free function in `src/models/hybrid_embeddings.rs` (before or after `build_expr_batch_for_profiling`):

  ```rust
  /// Returns indices of the top-k highest-scoring neurons, sorted ascending.
  fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
      let mut indexed: Vec<(usize, f32)> = scores.iter().cloned().enumerate().collect();
      // Sort descending by score; break ties by index (lower index first for stability)
      indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
      let mut keep: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
      keep.sort(); // ascending order so weight matrix slicing is predictable
      keep
  }
  ```

- [ ] **Step 4: Add `importance_scores` method**

  Add inside `impl HybridEmbeddingModel`:

  ```rust
  /// Compute combined importance score for each fc1 output neuron.
  /// score(j) = alpha * activation_freq(j) + (1-alpha) * weight_magnitude(j)
  /// weight_magnitude uses normalized fc1 column norms + fc2 row norms (each to [0,1]).
  pub fn importance_scores(&self, stats: &NeuronStats, alpha: f32) -> LayerScores {
      let n = stats.total_examples.max(1) as f32;

      let tf_freq:   Vec<f32> = stats.tf_fc1_activation_counts.iter()
          .map(|&c| c as f32 / n).collect();
      let gene_freq: Vec<f32> = stats.gene_fc1_activation_counts.iter()
          .map(|&c| c as f32 / n).collect();

      // Normalize a slice to [0,1]; if all values equal, return 0.5 uniformly
      let normalize = |norms: &[f32]| -> Vec<f32> {
          let min = norms.iter().cloned().fold(f32::INFINITY, f32::min);
          let max = norms.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
          let range = max - min;
          if range < 1e-10 {
              vec![0.5; norms.len()]
          } else {
              norms.iter().map(|&v| (v - min) / range).collect()
          }
      };

      let tf_fc1_n  = normalize(&stats.tf_fc1_col_norms);
      let tf_fc2_n  = normalize(&stats.tf_fc2_row_norms);
      let g_fc1_n   = normalize(&stats.gene_fc1_col_norms);
      let g_fc2_n   = normalize(&stats.gene_fc2_row_norms);

      let hidden_dim = tf_freq.len();

      let tf_scores: Vec<f32> = (0..hidden_dim).map(|j| {
          let wm = (tf_fc1_n[j] + tf_fc2_n[j]) / 2.0;
          alpha * tf_freq[j] + (1.0 - alpha) * wm
      }).collect();

      let gene_scores: Vec<f32> = (0..hidden_dim).map(|j| {
          let wm = (g_fc1_n[j] + g_fc2_n[j]) / 2.0;
          alpha * gene_freq[j] + (1.0 - alpha) * wm
      }).collect();

      LayerScores { tf_fc1: tf_scores, gene_fc1: gene_scores }
  }
  ```

- [ ] **Step 5: Add `prune_to_sparsity` method**

  Add inside `impl HybridEmbeddingModel`:

  ```rust
  /// Structurally remove the lowest-scoring (sparsity * hidden_dim) fc1 neurons.
  /// For each tower: prunes fc1 output columns + matching fc2 input rows as a pair.
  /// Towers are pruned independently (same sparsity %, different neurons may be selected).
  pub fn prune_to_sparsity(&mut self, scores: &LayerScores, sparsity: f32) {
      let hidden_dim = scores.tf_fc1.len();
      let keep_count = ((1.0 - sparsity) * hidden_dim as f32).round() as usize;
      let keep_count = keep_count.max(1).min(hidden_dim);

      let tf_keep   = top_k_indices(&scores.tf_fc1,   keep_count);
      let gene_keep = top_k_indices(&scores.gene_fc1, keep_count);

      // TF tower: fc1 outputs + fc2 inputs must match
      self.tf_fc1.prune_outputs(&tf_keep);
      self.tf_fc2.prune_inputs(&tf_keep);

      // Gene tower
      self.gene_fc1.prune_outputs(&gene_keep);
      self.gene_fc2.prune_inputs(&gene_keep);
  }
  ```

- [ ] **Step 6: Run all tests — expect PASS**

  ```bash
  cargo test --lib models -- --nocapture
  ```
  Expected: all tests in `models::nn` and `models::hybrid_embeddings` pass

- [ ] **Step 7: Commit**

  ```bash
  git add src/models/hybrid_embeddings.rs
  git commit -m "feat: add neuron profiling and structured pruning to HybridEmbeddingModel"
  ```

---

## Chunk 3: `neuron_pruning.rs` Experiment Binary

### Task 6: Create the experiment binary `src/bin/neuron_pruning.rs`

**Files:**
- Create: `src/bin/neuron_pruning.rs`

This binary re-uses the same data loading, training loop, and eval helpers from `train_standard_mlp.rs`, but is self-contained.

- [ ] **Step 1: Write the binary skeleton — compile check only**

  Create `src/bin/neuron_pruning.rs`:

  ```rust
  /// Neuron Pruning Experiment
  /// Phase 0: Train baseline (seed=42)
  /// Phase 1: Profile neuron activations
  /// Phase 2: Post-hoc sparsity sweep
  /// Phase 3: Fine-tune pruned models, re-evaluate
  /// Output: results/neuron_pruning_results.json
  use module_regularized_grn::{
      Config,
      models::hybrid_embeddings::HybridEmbeddingModel,
      data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
  };
  use ndarray::{Array1, Array2};
  use rand::SeedableRng;
  use rand::rngs::StdRng;
  use rand::seq::SliceRandom;
  use anyhow::Result;
  use std::collections::HashMap;

  fn main() -> Result<()> {
      println!("=== Neuron Pruning Experiment ===");
      Ok(())
  }
  ```

- [ ] **Step 2: Verify it compiles**

  ```bash
  cargo build --bin neuron_pruning 2>&1 | head -20
  ```
  Expected: compiles cleanly (possible warnings OK, no errors)

---

### Task 7: Add Adam state, helpers, and data loading

**Files:**
- Modify: `src/bin/neuron_pruning.rs`

- [ ] **Step 1: Add `AdamState` struct and `adam_step` function**

  These are identical to `train_standard_mlp.rs`. Add them to `neuron_pruning.rs`:

  ```rust
  // ── Adam state ────────────────────────────────────────────────────────────────

  struct AdamState {
      m_tf: Array2<f32>, v_tf: Array2<f32>,
      m_gene: Array2<f32>, v_gene: Array2<f32>,
      m_tf_fc1_w: Array2<f32>, v_tf_fc1_w: Array2<f32>,
      m_tf_fc1_b: Array1<f32>, v_tf_fc1_b: Array1<f32>,
      m_tf_fc2_w: Array2<f32>, v_tf_fc2_w: Array2<f32>,
      m_tf_fc2_b: Array1<f32>, v_tf_fc2_b: Array1<f32>,
      m_g_fc1_w:  Array2<f32>, v_g_fc1_w:  Array2<f32>,
      m_g_fc1_b:  Array1<f32>, v_g_fc1_b:  Array1<f32>,
      m_g_fc2_w:  Array2<f32>, v_g_fc2_w:  Array2<f32>,
      m_g_fc2_b:  Array1<f32>, v_g_fc2_b:  Array1<f32>,
      t: i32,
  }

  impl AdamState {
      fn new(model: &HybridEmbeddingModel) -> Self {
          Self {
              m_tf:       Array2::zeros(model.tf_embed.dim()),
              v_tf:       Array2::zeros(model.tf_embed.dim()),
              m_gene:     Array2::zeros(model.gene_embed.dim()),
              v_gene:     Array2::zeros(model.gene_embed.dim()),
              m_tf_fc1_w: Array2::zeros(model.tf_fc1.weights.dim()),
              v_tf_fc1_w: Array2::zeros(model.tf_fc1.weights.dim()),
              m_tf_fc1_b: Array1::zeros(model.tf_fc1.bias.len()),
              v_tf_fc1_b: Array1::zeros(model.tf_fc1.bias.len()),
              m_tf_fc2_w: Array2::zeros(model.tf_fc2.weights.dim()),
              v_tf_fc2_w: Array2::zeros(model.tf_fc2.weights.dim()),
              m_tf_fc2_b: Array1::zeros(model.tf_fc2.bias.len()),
              v_tf_fc2_b: Array1::zeros(model.tf_fc2.bias.len()),
              m_g_fc1_w:  Array2::zeros(model.gene_fc1.weights.dim()),
              v_g_fc1_w:  Array2::zeros(model.gene_fc1.weights.dim()),
              m_g_fc1_b:  Array1::zeros(model.gene_fc1.bias.len()),
              v_g_fc1_b:  Array1::zeros(model.gene_fc1.bias.len()),
              m_g_fc2_w:  Array2::zeros(model.gene_fc2.weights.dim()),
              v_g_fc2_w:  Array2::zeros(model.gene_fc2.weights.dim()),
              m_g_fc2_b:  Array1::zeros(model.gene_fc2.bias.len()),
              v_g_fc2_b:  Array1::zeros(model.gene_fc2.bias.len()),
              t: 0,
          }
      }
  }

  fn adam_step(model: &mut HybridEmbeddingModel, state: &mut AdamState, lr: f32, clip: f32) {
      state.t += 1;
      let b1 = 0.9f32; let b2 = 0.999f32; let eps = 1e-8f32;

      let norm_sq: f32 = [
          model.tf_embed_grad.iter().map(|x| x*x).sum::<f32>(),
          model.gene_embed_grad.iter().map(|x| x*x).sum::<f32>(),
          model.tf_fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
          model.tf_fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
          model.tf_fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
          model.tf_fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
          model.gene_fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
          model.gene_fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
          model.gene_fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
          model.gene_fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
      ].iter().sum();
      let scale = if norm_sq.sqrt() > clip { clip / norm_sq.sqrt() } else { 1.0 };

      fn adam2d(p: &mut Array2<f32>, g: &Array2<f32>, m: &mut Array2<f32>, v: &mut Array2<f32>,
                lr: f32, b1: f32, b2: f32, eps: f32, t: i32, sc: f32) {
          let bc1 = 1.0 - b1.powi(t); let bc2 = 1.0 - b2.powi(t);
          ndarray::Zip::from(m.view_mut()).and(g.view()).for_each(|m,&g| *m = b1 * *m + (1.0-b1)*g*sc);
          ndarray::Zip::from(v.view_mut()).and(g.view()).for_each(|v,&g| *v = b2 * *v + (1.0-b2)*(g*sc).powi(2));
          ndarray::Zip::from(p.view_mut()).and(m.view()).and(v.view())
              .for_each(|p,&m,&v| *p -= lr*(m/bc1)/((v/bc2).sqrt()+eps));
      }
      fn adam1d(p: &mut Array1<f32>, g: &Array1<f32>, m: &mut Array1<f32>, v: &mut Array1<f32>,
                lr: f32, b1: f32, b2: f32, eps: f32, t: i32, sc: f32) {
          let bc1 = 1.0 - b1.powi(t); let bc2 = 1.0 - b2.powi(t);
          ndarray::Zip::from(m.view_mut()).and(g.view()).for_each(|m,&g| *m = b1 * *m + (1.0-b1)*g*sc);
          ndarray::Zip::from(v.view_mut()).and(g.view()).for_each(|v,&g| *v = b2 * *v + (1.0-b2)*(g*sc).powi(2));
          ndarray::Zip::from(p.view_mut()).and(m.view()).and(v.view())
              .for_each(|p,&m,&v| *p -= lr*(m/bc1)/((v/bc2).sqrt()+eps));
      }

      adam2d(&mut model.tf_embed,         &model.tf_embed_grad,         &mut state.m_tf,       &mut state.v_tf,       lr, b1, b2, eps, state.t, scale);
      adam2d(&mut model.gene_embed,       &model.gene_embed_grad,       &mut state.m_gene,     &mut state.v_gene,     lr, b1, b2, eps, state.t, scale);
      adam2d(&mut model.tf_fc1.weights,   &model.tf_fc1.grad_weights,   &mut state.m_tf_fc1_w, &mut state.v_tf_fc1_w, lr, b1, b2, eps, state.t, scale);
      adam1d(&mut model.tf_fc1.bias,      &model.tf_fc1.grad_bias,      &mut state.m_tf_fc1_b, &mut state.v_tf_fc1_b, lr, b1, b2, eps, state.t, scale);
      adam2d(&mut model.tf_fc2.weights,   &model.tf_fc2.grad_weights,   &mut state.m_tf_fc2_w, &mut state.v_tf_fc2_w, lr, b1, b2, eps, state.t, scale);
      adam1d(&mut model.tf_fc2.bias,      &model.tf_fc2.grad_bias,      &mut state.m_tf_fc2_b, &mut state.v_tf_fc2_b, lr, b1, b2, eps, state.t, scale);
      adam2d(&mut model.gene_fc1.weights, &model.gene_fc1.grad_weights, &mut state.m_g_fc1_w,  &mut state.v_g_fc1_w,  lr, b1, b2, eps, state.t, scale);
      adam1d(&mut model.gene_fc1.bias,    &model.gene_fc1.grad_bias,    &mut state.m_g_fc1_b,  &mut state.v_g_fc1_b,  lr, b1, b2, eps, state.t, scale);
      adam2d(&mut model.gene_fc2.weights, &model.gene_fc2.grad_weights, &mut state.m_g_fc2_w,  &mut state.v_g_fc2_w,  lr, b1, b2, eps, state.t, scale);
      adam1d(&mut model.gene_fc2.bias,    &model.gene_fc2.grad_bias,    &mut state.m_g_fc2_b,  &mut state.v_g_fc2_b,  lr, b1, b2, eps, state.t, scale);
  }
  ```

- [ ] **Step 2: Add eval helpers**

  Add these functions below `adam_step`:

  ```rust
  // ── Helpers ───────────────────────────────────────────────────────────────────

  fn build_expr_batch(
      indices:  &[usize],
      expr_map: &HashMap<usize, Array1<f32>>,
      expr_dim: usize,
  ) -> Array2<f32> {
      let mut batch = Array2::zeros((indices.len(), expr_dim));
      for (i, &idx) in indices.iter().enumerate() {
          if let Some(expr) = expr_map.get(&idx) { batch.row_mut(i).assign(expr); }
      }
      batch
  }

  fn calculate_auroc(predictions: &[f32], labels: &[f32]) -> f32 {
      let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels.iter())
          .map(|(&p, &l)| (p, l)).collect();
      pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
      let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
      let n_neg = labels.len() as f32 - n_pos;
      if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
      let mut tp = 0.0f32; let mut fp = 0.0f32;
      let mut auc = 0.0f32; let mut prev_fp = 0.0f32;
      for (_, label) in &pairs {
          if *label == 1.0 { tp += 1.0; } else {
              fp += 1.0;
              auc += (tp / n_pos) * ((fp - prev_fp) / n_neg);
              prev_fp = fp;
          }
      }
      auc
  }

  fn evaluate(
      model:         &mut HybridEmbeddingModel,
      data:          &[(usize, usize, f32)],
      tf_expr_map:   &HashMap<usize, Array1<f32>>,
      gene_expr_map: &HashMap<usize, Array1<f32>>,
      expr_dim:      usize,
      batch_size:    usize,
  ) -> (f32, f32) {  // (accuracy, auroc)
      let mut all_preds: Vec<f32> = Vec::new();
      let mut all_labels: Vec<f32> = Vec::new();
      for start in (0..data.len()).step_by(batch_size) {
          let end   = (start + batch_size).min(data.len());
          let batch = &data[start..end];
          let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
          let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
          let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
          let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
          let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
          all_preds.extend_from_slice(preds.as_slice().unwrap());
          all_labels.extend_from_slice(&batch.iter().map(|x| x.2).collect::<Vec<_>>());
      }
      let n_correct = all_preds.iter().zip(all_labels.iter())
          .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count();
      let accuracy = n_correct as f32 / all_preds.len() as f32;
      let auroc    = calculate_auroc(&all_preds, &all_labels);
      (accuracy, auroc)
  }

  ```

- [ ] **Step 3: Verify the binary still compiles**

  ```bash
  cargo build --bin neuron_pruning 2>&1 | head -20
  ```

---

### Task 8: Implement `main()` — data loading and baseline training

**Files:**
- Modify: `src/bin/neuron_pruning.rs`

- [ ] **Step 1: Replace the `main()` stub with data loading + baseline training**

  Replace the body of `main()`:

  ```rust
  fn main() -> Result<()> {
      println!("=== Neuron Pruning Experiment ===");

      // ── Config & data loading ─────────────────────────────────────────────────
      let config    = Config::load_default()?;
      let base_seed = config.project.seed;
      let mut rng   = StdRng::seed_from_u64(base_seed);

      let priors  = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
      let builder = PriorDatasetBuilder::new(priors.clone());
      let num_tfs   = builder.num_tfs();
      let num_genes = builder.num_genes();

      let expr_data = ExpressionData::from_processed_dir(
          "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
      )?;
      let expr_dim = expr_data.n_cell_types;

      let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
      for (i, name) in expr_data.gene_names.iter().enumerate() {
          gene_to_expr_idx.insert(name.clone(), i);
      }
      let tf_vocab   = builder.get_tf_vocab();
      let gene_vocab = builder.get_gene_vocab();

      let mut tf_expr_map:   HashMap<usize, Array1<f32>> = HashMap::new();
      let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
      for (name, &idx) in tf_vocab.iter() {
          let expr = gene_to_expr_idx.get(name)
              .map(|&ei| expr_data.expression.column(ei).to_owned())
              .unwrap_or_else(|| Array1::zeros(expr_dim));
          tf_expr_map.insert(idx, expr);
      }
      for (name, &idx) in gene_vocab.iter() {
          let expr = gene_to_expr_idx.get(name)
              .map(|&ei| expr_data.expression.column(ei).to_owned())
              .unwrap_or_else(|| Array1::zeros(expr_dim));
          gene_expr_map.insert(idx, expr);
      }

      // ── Dataset split (fixed seed, same as train_standard_mlp) ───────────────
      let positives = builder.get_positive_examples();
      let negatives = builder.sample_negative_examples(positives.len(), base_seed);
      let mut examples: Vec<(usize, usize, f32)> = Vec::new();
      examples.extend(positives.iter().map(|&(tf, g)| (tf, g, 1.0f32)));
      examples.extend(negatives.iter().map(|&(tf, g)| (tf, g, 0.0f32)));
      examples.shuffle(&mut rng);

      let n_total = examples.len();
      let n_train = (n_total as f32 * 0.7) as usize;
      let n_val   = (n_total as f32 * 0.15) as usize;
      let train_data = examples[..n_train].to_vec();
      let val_data   = examples[n_train..n_train+n_val].to_vec();
      let test_data  = examples[n_train+n_val..].to_vec();
      println!("Dataset: {} train / {} val / {} test", train_data.len(), val_data.len(), test_data.len());

      // ── Hyperparameters ───────────────────────────────────────────────────────
      let embed_dim   = 512usize;
      let hidden_dim  = 512usize;
      let output_dim  = 512usize;
      let temperature = 0.05f32;
      let lr          = 0.001f32;
      let clip        = 5.0f32;
      let batch_size  = 256usize;
      let epochs      = 60usize;
      let patience    = 10usize;
      let seed        = 42u64;

      // ── Phase 0: Train baseline (seed=42) ─────────────────────────────────────
      println!("\n[Phase 0] Training baseline (seed={})...", seed);
      let mut baseline = HybridEmbeddingModel::new(
          num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, temperature, 0.01, seed,
      );
      let mut adam = AdamState::new(&baseline);
      let mut best_val_acc = 0.0f32;
      let mut patience_ctr = 0usize;
      let mut shuffled_train = train_data.clone();

      for epoch in 0..epochs {
          let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
          shuffled_train.shuffle(&mut epoch_rng);

          for start in (0..shuffled_train.len()).step_by(batch_size) {
              let end   = (start + batch_size).min(shuffled_train.len());
              let batch = &shuffled_train[start..end];
              let bsz   = batch.len();
              let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
              let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
              let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
              let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
              let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
              let preds  = baseline.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
              let grad: Array1<f32> = preds.iter().zip(labels.iter())
                  .map(|(&p, &l)| (p - l) / bsz as f32).collect();
              baseline.backward(&grad);
              adam_step(&mut baseline, &mut adam, lr, clip);
              baseline.zero_grad();
          }

          if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
              let (_, val_acc) = evaluate(&mut baseline, &val_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
              println!("  Epoch {:2} | Val Acc: {:.2}%", epoch + 1, val_acc * 100.0);
              if val_acc > best_val_acc {
                  best_val_acc = val_acc; patience_ctr = 0;
              } else {
                  patience_ctr += 1;
                  if patience_ctr >= patience { println!("  Early stop."); break; }
              }
          }
      }

      let (baseline_acc, baseline_auroc) = evaluate(
          &mut baseline, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
      );
      let baseline_params = baseline.num_parameters();
      println!("Baseline: acc={:.2}% auroc={:.4} params={}", baseline_acc*100.0, baseline_auroc, baseline_params);

      // ── Phase 1: Profile activations ──────────────────────────────────────────
      println!("\n[Phase 1] Profiling activations...");
      let stats = baseline.profile_activations(
          &train_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
      );
      let scores = baseline.importance_scores(&stats, 0.5);
      println!("  Profiled {} examples", stats.total_examples);

      // ── Phase 2 & 3: Sparsity sweep ───────────────────────────────────────────
      let sparsity_levels: &[f32] = &[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];
      let fine_tune_epochs = 10usize;
      let alpha = 0.5f32;

      let mut results = Vec::new();

      for &sparsity in sparsity_levels {
          let keep_count = ((1.0 - sparsity) * hidden_dim as f32).round() as usize;
          let keep_count = keep_count.max(1);
          let neurons_removed_per_tower = hidden_dim - keep_count;
          println!("\n[Sparsity {:.0}%] keep={} neurons/tower", sparsity * 100.0, keep_count);

          // ── Post-hoc: clone baseline, prune, evaluate ─────────────────────────
          let mut pruned = baseline.clone();
          pruned.prune_to_sparsity(&scores, sparsity);

          let (posthoc_acc, posthoc_auroc) = evaluate(
              &mut pruned, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
          );
          let params_remaining = pruned.num_parameters();
          let compression_ratio = params_remaining as f32 / baseline_params as f32;
          let posthoc_retention = posthoc_auroc / baseline_auroc;
          println!("  Post-hoc:   acc={:.2}% auroc={:.4} retention={:.3} params={} ratio={:.3}",
              posthoc_acc*100.0, posthoc_auroc, posthoc_retention, params_remaining, compression_ratio);

          // ── Fine-tune: fresh Adam, 10 epochs, re-evaluate ─────────────────────
          let mut ft_model = pruned.clone();
          let mut ft_adam = AdamState::new(&ft_model);  // fresh moments, t=0

          let mut ft_shuffled = train_data.clone();
          for epoch in 0..fine_tune_epochs {
              let mut epoch_rng = StdRng::seed_from_u64(seed + 10000 + epoch as u64);
              ft_shuffled.shuffle(&mut epoch_rng);
              for start in (0..ft_shuffled.len()).step_by(batch_size) {
                  let end   = (start + batch_size).min(ft_shuffled.len());
                  let batch = &ft_shuffled[start..end];
                  let bsz   = batch.len();
                  let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                  let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                  let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                  let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                  let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                  let preds  = ft_model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                  let grad: Array1<f32> = preds.iter().zip(labels.iter())
                      .map(|(&p, &l)| (p - l) / bsz as f32).collect();
                  ft_model.backward(&grad);
                  adam_step(&mut ft_model, &mut ft_adam, lr, clip);
                  ft_model.zero_grad();
              }
          }

          let (ft_acc, ft_auroc) = evaluate(
              &mut ft_model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
          );
          let ft_retention = ft_auroc / baseline_auroc;
          println!("  Fine-tuned: acc={:.2}% auroc={:.4} retention={:.3}",
              ft_acc*100.0, ft_auroc, ft_retention);

          results.push(serde_json::json!({
              "sparsity": sparsity,
              "neurons_removed_per_tower": neurons_removed_per_tower,
              "neurons_removed_total": neurons_removed_per_tower * 2,
              "params_remaining": params_remaining,
              "compression_ratio": compression_ratio,
              "posthoc_auroc":     posthoc_auroc,
              "posthoc_accuracy":  posthoc_acc,
              "posthoc_retention": posthoc_retention,
              "finetuned_auroc":   ft_auroc,
              "finetuned_accuracy": ft_acc,
              "finetuned_retention": ft_retention,
          }));
      }

      // ── Compute retention thresholds ──────────────────────────────────────────
      // Last sparsity level at or above the threshold (highest sparsity where retention >= threshold)
      let threshold_95 = results.iter()
          .filter(|r| r["finetuned_retention"].as_f64().unwrap_or(0.0) >= 0.95)
          .map(|r| r["sparsity"].as_f64().unwrap_or(0.0))
          .fold(f64::NEG_INFINITY, f64::max);
      let threshold_90 = results.iter()
          .filter(|r| r["finetuned_retention"].as_f64().unwrap_or(0.0) >= 0.90)
          .map(|r| r["sparsity"].as_f64().unwrap_or(0.0))
          .fold(f64::NEG_INFINITY, f64::max);
      let threshold_95 = if threshold_95.is_finite() { threshold_95 } else { -1.0 };
      let threshold_90 = if threshold_90.is_finite() { threshold_90 } else { -1.0 };
      println!("\n95% AUROC retention up to sparsity: {:.0}%", threshold_95 * 100.0);
      println!("90% AUROC retention up to sparsity: {:.0}%", threshold_90 * 100.0);

      // ── Write output ──────────────────────────────────────────────────────────
      std::fs::create_dir_all("results")?;
      let output = serde_json::json!({
          "seed": seed,
          "alpha": alpha,
          "fine_tune_epochs": fine_tune_epochs,
          "fine_tune_lr": lr,
          "baseline_auroc": baseline_auroc,
          "baseline_accuracy": baseline_acc,
          "baseline_params": baseline_params,
          "results": results,
          "retention_95_threshold_sparsity": threshold_95,
          "retention_90_threshold_sparsity": threshold_90,
      });
      std::fs::write("results/neuron_pruning_results.json", serde_json::to_string_pretty(&output)?)?;
      println!("\n✓ Results written to results/neuron_pruning_results.json");

      Ok(())
  }
  ```

- [ ] **Step 2: Build in release mode to catch any remaining errors**

  ```bash
  cargo build --release --bin neuron_pruning 2>&1 | head -40
  ```
  Expected: compiles cleanly (warnings OK, no errors)

- [ ] **Step 3: Run all library tests to confirm nothing regressed**

  ```bash
  cargo test --lib -- --nocapture 2>&1 | tail -20
  ```
  Expected: all existing tests still pass

- [ ] **Step 4: Commit**

  ```bash
  git add src/bin/neuron_pruning.rs
  git commit -m "feat: add neuron_pruning experiment binary (3-phase pruning pipeline)"
  ```

---

### Task 9: Run the experiment and verify output

**Files:**
- Read: `results/neuron_pruning_results.json`

- [ ] **Step 1: Run the full experiment**

  ```bash
  cargo run --release --bin neuron_pruning 2>&1 | tee /tmp/pruning_run.log
  ```
  Expected runtime: ~10–20 minutes (baseline training ~5min, 13 × fine-tune × 10 epochs ~10–15min).
  Expected final output lines:
  ```
  95% AUROC retention up to sparsity: XX%
  90% AUROC retention up to sparsity: XX%
  ✓ Results written to results/neuron_pruning_results.json
  ```

- [ ] **Step 2: Validate output JSON structure**

  ```bash
  python3 -c "
  import json
  with open('results/neuron_pruning_results.json') as f:
      r = json.load(f)
  assert 'baseline_auroc' in r
  assert 'results' in r
  assert len(r['results']) == 13  # 13 sparsity levels
  for entry in r['results']:
      assert 'finetuned_retention' in entry
      assert 'compression_ratio' in entry
  print('PASS: JSON structure valid')
  print(f'Baseline AUROC: {r[\"baseline_auroc\"]:.4f}')
  print(f'95% retention threshold: {r[\"retention_95_threshold_sparsity\"]:.0%}')
  print(f'90% retention threshold: {r[\"retention_90_threshold_sparsity\"]:.0%}')
  "
  ```
  Expected: `PASS: JSON structure valid` followed by threshold values

- [ ] **Step 3: Commit results**

  ```bash
  git add results/neuron_pruning_results.json
  git commit -m "results: neuron pruning experiment — sparsity-vs-AUROC sweep complete"
  ```

---

## Summary

| Task | File | What it adds |
|---|---|---|
| 1 | `src/models/nn.rs` | `prune_outputs` — remove output neurons from a LinearLayer |
| 2 | `src/models/nn.rs` | `prune_inputs` — remove input neurons from a LinearLayer |
| 3 | `src/models/hybrid_embeddings.rs` | `#[derive(Clone)]` on the model struct |
| 4 | `src/models/hybrid_embeddings.rs` | `NeuronStats`, `LayerScores`, `profile_activations` |
| 5 | `src/models/hybrid_embeddings.rs` | `importance_scores`, `prune_to_sparsity` |
| 6–9 | `src/bin/neuron_pruning.rs` | Full experiment binary: train, profile, sweep, fine-tune, write JSON |
