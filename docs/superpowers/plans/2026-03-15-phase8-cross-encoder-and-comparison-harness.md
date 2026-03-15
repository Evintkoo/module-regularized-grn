# Phase 8 Sub-Project 1: Cross-Encoder + Comparison Harness Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Model D (monolithic cross-encoder) and a unified comparison harness to produce the core dissertation comparison between modular and monolithic architectures at both 1:1 and 5:1 negative-sampling ratios.

**Architecture:** A new `CrossEncoderModel` struct in `src/models/cross_encoder.rs` uses the same `LinearLayer`/`relu`/`bce_loss` primitives as the two-tower. Its training script (`train_cross_encoder.rs`) runs 5 seeds × 2 ratios with inline Adam + gradient clipping. `train_standard_mlp.rs` is updated with a seed loop + neg-ratio flag to produce comparable two-tower results. `compare_models.rs` reads all result JSONs and prints a side-by-side table.

**Tech Stack:** Rust, `ndarray 0.15`, `serde_json`, `rand 0.8`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-03-15-phase8-cross-encoder-and-comparison-harness-design.md`

---

## File Map

### Create
- `src/models/cross_encoder.rs` — CrossEncoderModel struct with forward/backward/zero_grad
- `src/bin/train_cross_encoder.rs` — trains Model D at 1:1 and 5:1 (5 seeds each), with inline Adam
- `src/bin/compare_models.rs` — reads result JSONs, prints comparison table, writes `results/model_comparison.json`

### Modify
- `src/models/mod.rs` — add `pub mod cross_encoder; pub use cross_encoder::CrossEncoderModel;`
- `src/bin/train_standard_mlp.rs` — add seed loop, `--neg-ratio` flag, AUROC/F1 metrics, write new schema
- `Cargo.toml` — add two new `[[bin]]` stanzas

---

## Chunk 1: CrossEncoderModel

### Task 1: Implement CrossEncoderModel

**Files:**
- Create: `src/models/cross_encoder.rs`

**Background — critical conventions:**

1. **`bce_loss(predictions, labels)` applies sigmoid internally.** FC3 must output raw logits (no activation). Do NOT call sigmoid in `forward()`.
2. **`relu_backward(x, grad_output)`** — `x` is the pre-activation input (the thing you cache before calling `relu`), `grad_output` is the upstream gradient.
3. **Input dim** is computed dynamically: `3 * embed_dim + 2 * expr_dim` (= 1558 for the brain dataset). Do not hardcode.
4. **`LinearLayer::backward` stores gradients internally** in `grad_weights`/`grad_bias`. These are the sum (not mean) over the batch. The `update(lr)` method divides by batch size internally. For Adam (used in the training script, not in the model), we access `fc.grad_weights`/`fc.grad_bias` directly.

- [ ] **Step 1: Write failing tests**

Create `src/models/cross_encoder.rs` with ONLY the tests (no implementation yet):

```rust
// src/models/cross_encoder.rs
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array1, Array2, Axis};

pub struct CrossEncoderModel; // stub — tests will fail

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::nn::{bce_loss, bce_loss_backward};

    #[test]
    fn test_forward_output_shape() {
        // 10 TFs, 20 genes, embed=4, expr=3, hidden=8
        let mut model = CrossEncoderModel::new(10, 20, 4, 3, 8, 42);
        let tf_idx    = vec![0usize, 1, 2, 3];
        let gene_idx  = vec![5usize, 6, 7, 8];
        let tf_expr   = Array2::<f32>::zeros((4, 3));
        let gene_expr = Array2::<f32>::zeros((4, 3));
        let out = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        assert_eq!(out.dim(), (4, 1), "forward must return [batch, 1] raw logits");
    }

    #[test]
    fn test_loss_decreases_after_sgd_step() {
        let mut model = CrossEncoderModel::new(10, 20, 4, 3, 8, 99);
        let tf_idx    = vec![0usize, 1, 2, 3];
        let gene_idx  = vec![5usize, 6, 7, 8];
        let tf_expr   = Array2::<f32>::zeros((4, 3));
        let gene_expr = Array2::<f32>::zeros((4, 3));
        let labels    = Array1::<f32>::from(vec![1.0, 0.0, 1.0, 0.0]);

        let logits1 = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        let loss1   = bce_loss(&logits1, &labels);
        let grad    = bce_loss_backward(&logits1, &labels);
        model.backward(&grad);

        // SGD update (just to verify gradient direction)
        let lr = 0.5f32;
        model.tf_embed.scaled_add(-lr, &model.tf_embed_grad);
        model.gene_embed.scaled_add(-lr, &model.gene_embed_grad);
        model.fc1.update(lr);
        model.fc2.update(lr);
        model.fc3.update(lr);
        model.zero_grad();

        let logits2 = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        let loss2   = bce_loss(&logits2, &labels);
        assert!(loss2 < loss1, "loss must decrease: {:.4} → {:.4}", loss1, loss2);
    }
}
```

- [ ] **Step 2: Run to confirm compile failure**

```bash
cargo test --lib 2>&1 | head -20
```
Expected: compile error — `CrossEncoderModel::new` not defined.

- [ ] **Step 3: Implement CrossEncoderModel**

Replace the **entire file contents** of `src/models/cross_encoder.rs` with:

```rust
/// Cross-encoder model for GRN edge prediction.
/// Input: [TF_emb | Gene_emb | TF_expr | Gene_expr | TF_emb⊙Gene_emb]
/// Output: raw logit [batch, 1]. Use bce_loss/bce_loss_backward from nn.rs.
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array2, Axis};

pub struct CrossEncoderModel {
    // Embedding tables
    pub tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    pub gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,

    embed_dim: usize,
    expr_dim: usize,

    // FC layers (pub so Adam in training script can access grad_weights/grad_bias)
    pub fc1: LinearLayer,  // (3*embed+2*expr) → hidden, ReLU
    pub fc2: LinearLayer,  // hidden → hidden, ReLU
    pub fc3: LinearLayer,  // hidden → 1, raw logit

    // Caches for backward pass
    tf_indices_cache:  Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_embed_cache:    Option<Array2<f32>>,
    gene_embed_cache:  Option<Array2<f32>>,
    h1_pre_cache:      Option<Array2<f32>>,  // pre-ReLU input to fc1 (for relu_backward)
    h2_pre_cache:      Option<Array2<f32>>,  // pre-ReLU input to fc2 (for relu_backward)
}

impl CrossEncoderModel {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,
        expr_dim: usize,
        hidden_dim: usize,
        seed: u64,
    ) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0f32, 0.1).unwrap();
        let tf_embed   = Array2::random_using((num_tfs,   embed_dim), dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, embed_dim), dist, &mut rng);
        let input_dim  = 3 * embed_dim + 2 * expr_dim;

        Self {
            tf_embed_grad:   Array2::zeros((num_tfs,   embed_dim)),
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            fc1: LinearLayer::new(input_dim,  hidden_dim, seed),
            fc2: LinearLayer::new(hidden_dim, hidden_dim, seed + 1),
            fc3: LinearLayer::new(hidden_dim, 1,          seed + 2),
            embed_dim,
            expr_dim,
            tf_embed,
            gene_embed,
            tf_indices_cache:  None,
            gene_indices_cache: None,
            tf_embed_cache:    None,
            gene_embed_cache:  None,
            h1_pre_cache:      None,
            h2_pre_cache:      None,
        }
    }

    /// Forward pass. Returns raw logits [batch, 1]. No sigmoid applied.
    pub fn forward(
        &mut self,
        tf_indices:  &[usize],
        gene_indices: &[usize],
        tf_expr:     &Array2<f32>,
        gene_expr:   &Array2<f32>,
    ) -> Array2<f32> {
        let batch = tf_indices.len();

        // Embedding lookup
        let mut tf_emb   = Array2::zeros((batch, self.embed_dim));
        let mut gene_emb = Array2::zeros((batch, self.embed_dim));
        for (i, &idx) in tf_indices.iter().enumerate() {
            tf_emb.row_mut(i).assign(&self.tf_embed.row(idx));
        }
        for (i, &idx) in gene_indices.iter().enumerate() {
            gene_emb.row_mut(i).assign(&self.gene_embed.row(idx));
        }

        self.tf_indices_cache   = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());
        self.tf_embed_cache     = Some(tf_emb.clone());
        self.gene_embed_cache   = Some(gene_emb.clone());

        // Interaction: element-wise product [batch, embed_dim]
        let interaction = &tf_emb * &gene_emb;

        // Build input [batch, 3*embed+2*expr]
        let input = ndarray::concatenate![
            Axis(1),
            tf_emb.view(),
            gene_emb.view(),
            tf_expr.view(),
            gene_expr.view(),
            interaction.view()
        ];

        // FC1 → ReLU (cache pre-activation for relu_backward)
        let h1_pre = self.fc1.forward(&input);
        self.h1_pre_cache = Some(h1_pre.clone());
        let h1 = relu(&h1_pre);

        // FC2 → ReLU (cache pre-activation for relu_backward)
        let h2_pre = self.fc2.forward(&h1);
        self.h2_pre_cache = Some(h2_pre.clone());
        let h2 = relu(&h2_pre);

        // FC3 → raw logit (no activation)
        self.fc3.forward(&h2)
    }

    /// Backward pass. `grad_output` is [batch, 1] from bce_loss_backward.
    pub fn backward(&mut self, grad_output: &Array2<f32>) {
        // Backward through fc3
        let grad_h2 = self.fc3.backward(grad_output);

        // Backward through ReLU + fc2 (relu_backward takes pre-activation as first arg)
        let h2_pre = self.h2_pre_cache.as_ref().unwrap();
        let grad_h2_pre = relu_backward(h2_pre, &grad_h2);
        let grad_h1 = self.fc2.backward(&grad_h2_pre);

        // Backward through ReLU + fc1 (relu_backward takes pre-activation as first arg)
        let h1_pre = self.h1_pre_cache.as_ref().unwrap();
        let grad_h1_pre = relu_backward(h1_pre, &grad_h1);
        let grad_input = self.fc1.backward(&grad_h1_pre);
        // grad_input: [batch, 3*embed_dim + 2*expr_dim]

        let e = self.embed_dim;
        let x = self.expr_dim;

        // Split gradients from concatenated input
        let g_tf_emb      = grad_input.slice(ndarray::s![.., 0..e]).to_owned();
        let g_gene_emb    = grad_input.slice(ndarray::s![.., e..2*e]).to_owned();
        // Indices 2e..(2e+2x) are expression grads — no embedding params there, discard
        let g_interaction = grad_input.slice(ndarray::s![.., (2*e+2*x)..]).to_owned();

        // Interaction term backprop: interaction = tf_emb * gene_emb
        //   d(interaction)/d(tf_emb) = gene_emb * g_interaction
        //   d(interaction)/d(gene_emb) = tf_emb * g_interaction
        let tf_emb   = self.tf_embed_cache.as_ref().unwrap();
        let gene_emb = self.gene_embed_cache.as_ref().unwrap();

        let total_g_tf   = g_tf_emb   + &(gene_emb * &g_interaction);
        let total_g_gene = g_gene_emb + &(tf_emb   * &g_interaction);

        // Scatter into embedding gradient tables
        let tf_indices   = self.tf_indices_cache.as_ref().unwrap();
        let gene_indices = self.gene_indices_cache.as_ref().unwrap();
        for (i, &idx) in tf_indices.iter().enumerate() {
            self.tf_embed_grad.row_mut(idx) += &total_g_tf.row(i);
        }
        for (i, &idx) in gene_indices.iter().enumerate() {
            self.gene_embed_grad.row_mut(idx) += &total_g_gene.row(i);
        }
    }

    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.fc3.zero_grad();
    }

    pub fn num_parameters(&self) -> usize {
        self.tf_embed.len() + self.gene_embed.len()
            + self.fc1.weights.len() + self.fc1.bias.len()
            + self.fc2.weights.len() + self.fc2.bias.len()
            + self.fc3.weights.len() + self.fc3.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::nn::{bce_loss, bce_loss_backward};
    use ndarray::Array1;

    #[test]
    fn test_forward_output_shape() {
        let mut model = CrossEncoderModel::new(10, 20, 4, 3, 8, 42);
        let tf_idx    = vec![0usize, 1, 2, 3];
        let gene_idx  = vec![5usize, 6, 7, 8];
        let tf_expr   = Array2::<f32>::zeros((4, 3));
        let gene_expr = Array2::<f32>::zeros((4, 3));
        let out = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        assert_eq!(out.dim(), (4, 1), "forward must return [batch, 1] raw logits");
    }

    #[test]
    fn test_loss_decreases_after_sgd_step() {
        let mut model = CrossEncoderModel::new(10, 20, 4, 3, 8, 99);
        let tf_idx    = vec![0usize, 1, 2, 3];
        let gene_idx  = vec![5usize, 6, 7, 8];
        let tf_expr   = Array2::<f32>::zeros((4, 3));
        let gene_expr = Array2::<f32>::zeros((4, 3));
        let labels    = Array1::<f32>::from(vec![1.0, 0.0, 1.0, 0.0]);

        let logits1 = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        let loss1   = bce_loss(&logits1, &labels);
        let grad    = bce_loss_backward(&logits1, &labels);
        model.backward(&grad);

        let lr = 0.5f32;
        model.tf_embed.scaled_add(-lr, &model.tf_embed_grad);
        model.gene_embed.scaled_add(-lr, &model.gene_embed_grad);
        model.fc1.update(lr);
        model.fc2.update(lr);
        model.fc3.update(lr);
        model.zero_grad();

        let logits2 = model.forward(&tf_idx, &gene_idx, &tf_expr, &gene_expr);
        let loss2   = bce_loss(&logits2, &labels);
        assert!(loss2 < loss1, "loss must decrease: {:.4} → {:.4}", loss1, loss2);
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cargo test --lib 2>&1 | grep -E "FAILED|ok|error"
```
Expected: both tests pass.

- [ ] **Step 5: Update `src/models/mod.rs`**

```rust
pub mod nn;
pub mod hybrid_embeddings;
pub mod cross_encoder;

pub use nn::{
    LinearLayer, Dropout,
    relu, relu_backward,
    sigmoid, sigmoid_backward,
    bce_loss, bce_loss_backward,
};
pub use hybrid_embeddings::HybridEmbeddingModel;
pub use cross_encoder::CrossEncoderModel;
```

- [ ] **Step 6: Add `[[bin]]` stanzas to `Cargo.toml`**

After the last existing `[[bin]]` stanza (after `generate_figures`), add:

```toml
[[bin]]
name = "train_cross_encoder"
path = "src/bin/train_cross_encoder.rs"

[[bin]]
name = "compare_models"
path = "src/bin/compare_models.rs"
```

- [ ] **Step 7: Build to verify**

```bash
cargo build --release --lib 2>&1 | grep -E "^error|Finished"
```
Expected: `Finished release` — only the lib; bins don't exist yet.

- [ ] **Step 8: Commit**

```bash
git add src/models/cross_encoder.rs src/models/mod.rs Cargo.toml
git commit -m "feat: add CrossEncoderModel (Model D) with tests"
```

---

## Chunk 2: train_cross_encoder.rs

### Task 2: Implement train_cross_encoder.rs

**Files:**
- Create: `src/bin/train_cross_encoder.rs`

**What this script does:**
1. Loads data (priors + expression) once
2. Trains 5 seeds at 1:1 negative ratio → `results/cross_encoder_1to1.json`
3. Trains 5 seeds at 5:1 negative ratio → `results/cross_encoder_5to1.json`
4. Uses inline Adam with gradient clipping (not the model's built-in `update()`)
5. Computes per-seed accuracy, AUROC, F1 and bootstrap CI over test predictions

**Key hyperparameters:**
- `embed_dim = 512`, `hidden_dim = 512`, `lr = 0.005`, `batch_size = 256`, `epochs = 60`
- `seeds = [42u64, 123, 456, 789, 1337]`
- `grad_clip_norm = 1.0` (no weight decay — spec does not require L2 for cross-encoder)

**AdamState struct** (inline in this file, not shared):

```rust
struct AdamState {
    m_tf: ndarray::Array2<f32>,  v_tf: ndarray::Array2<f32>,
    m_gene: ndarray::Array2<f32>, v_gene: ndarray::Array2<f32>,
    m_fc1_w: ndarray::Array2<f32>, v_fc1_w: ndarray::Array2<f32>,
    m_fc1_b: ndarray::Array1<f32>, v_fc1_b: ndarray::Array1<f32>,
    m_fc2_w: ndarray::Array2<f32>, v_fc2_w: ndarray::Array2<f32>,
    m_fc2_b: ndarray::Array1<f32>, v_fc2_b: ndarray::Array1<f32>,
    m_fc3_w: ndarray::Array2<f32>, v_fc3_w: ndarray::Array2<f32>,
    m_fc3_b: ndarray::Array1<f32>, v_fc3_b: ndarray::Array1<f32>,
    t: i32,
}

impl AdamState {
    fn new(model: &CrossEncoderModel) -> Self {
        Self {
            m_tf:    Array2::zeros(model.tf_embed.dim()),
            v_tf:    Array2::zeros(model.tf_embed.dim()),
            m_gene:  Array2::zeros(model.gene_embed.dim()),
            v_gene:  Array2::zeros(model.gene_embed.dim()),
            m_fc1_w: Array2::zeros(model.fc1.weights.dim()),
            v_fc1_w: Array2::zeros(model.fc1.weights.dim()),
            m_fc1_b: Array1::zeros(model.fc1.bias.len()),
            v_fc1_b: Array1::zeros(model.fc1.bias.len()),
            m_fc2_w: Array2::zeros(model.fc2.weights.dim()),
            v_fc2_w: Array2::zeros(model.fc2.weights.dim()),
            m_fc2_b: Array1::zeros(model.fc2.bias.len()),
            v_fc2_b: Array1::zeros(model.fc2.bias.len()),
            m_fc3_w: Array2::zeros(model.fc3.weights.dim()),
            v_fc3_w: Array2::zeros(model.fc3.weights.dim()),
            m_fc3_b: Array1::zeros(model.fc3.bias.len()),
            v_fc3_b: Array1::zeros(model.fc3.bias.len()),
            t: 0,
        }
    }
}
```

**`adam_step` function** (inline, applies update to model params from grad fields):

```rust
fn adam_step(model: &mut CrossEncoderModel, state: &mut AdamState, lr: f32, clip: f32) {
    state.t += 1;
    let b1  = 0.9f32;
    let b2  = 0.999f32;
    let eps = 1e-8f32;

    // Global gradient clipping (L2 norm across all params)
    let norm_sq: f32 = [
        model.tf_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.gene_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.fc3.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc3.grad_bias.iter().map(|x| x*x).sum::<f32>(),
    ].iter().sum();
    let scale = if norm_sq.sqrt() > clip { clip / norm_sq.sqrt() } else { 1.0 };

    // Adam update helper for 2D arrays
    fn adam2d(
        param: &mut ndarray::Array2<f32>,
        grad:  &ndarray::Array2<f32>,
        m: &mut ndarray::Array2<f32>,
        v: &mut ndarray::Array2<f32>,
        lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32,
    ) {
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(grad.view())
            .for_each(|m, &g| *m = b1 * *m + (1.0 - b1) * g * scale);
        ndarray::Zip::from(v.view_mut()).and(grad.view())
            .for_each(|v, &g| *v = b2 * *v + (1.0 - b2) * (g * scale).powi(2));
        ndarray::Zip::from(param.view_mut()).and(m.view()).and(v.view())
            .for_each(|p, &m, &v| *p -= lr * (m / bc1) / ((v / bc2).sqrt() + eps));
    }

    // Adam update helper for 1D arrays
    fn adam1d(
        param: &mut ndarray::Array1<f32>,
        grad:  &ndarray::Array1<f32>,
        m: &mut ndarray::Array1<f32>,
        v: &mut ndarray::Array1<f32>,
        lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32,
    ) {
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(grad.view())
            .for_each(|m, &g| *m = b1 * *m + (1.0 - b1) * g * scale);
        ndarray::Zip::from(v.view_mut()).and(grad.view())
            .for_each(|v, &g| *v = b2 * *v + (1.0 - b2) * (g * scale).powi(2));
        ndarray::Zip::from(param.view_mut()).and(m.view()).and(v.view())
            .for_each(|p, &m, &v| *p -= lr * (m / bc1) / ((v / bc2).sqrt() + eps));
    }

    adam2d(&mut model.tf_embed,      &model.tf_embed_grad,       &mut state.m_tf,    &mut state.v_tf,    lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,    &model.gene_embed_grad,     &mut state.m_gene,  &mut state.v_gene,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc1.weights,   &model.fc1.grad_weights,    &mut state.m_fc1_w, &mut state.v_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc1.bias,      &model.fc1.grad_bias,       &mut state.m_fc1_b, &mut state.v_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc2.weights,   &model.fc2.grad_weights,    &mut state.m_fc2_w, &mut state.v_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc2.bias,      &model.fc2.grad_bias,       &mut state.m_fc2_b, &mut state.v_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc3.weights,   &model.fc3.grad_weights,    &mut state.m_fc3_w, &mut state.v_fc3_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc3.bias,      &model.fc3.grad_bias,       &mut state.m_fc3_b, &mut state.v_fc3_b, lr, b1, b2, eps, state.t, scale);
}
```

**Helper functions** (copy from `train_ensemble.rs`; these are private functions so not importable):

```rust
fn build_expr_batch(
    indices:  &[usize],
    expr_map: &std::collections::HashMap<usize, ndarray::Array1<f32>>,
    expr_dim: usize,
) -> ndarray::Array2<f32> {
    let mut batch = ndarray::Array2::zeros((indices.len(), expr_dim));
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(expr) = expr_map.get(&idx) {
            batch.row_mut(i).assign(expr);
        }
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
    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut auc = 0.0f32;
    let mut prev_fp = 0.0f32;
    for (_, label) in &pairs {
        if *label == 1.0 { tp += 1.0; } else {
            fp += 1.0;
            auc += (tp / n_pos) * ((fp - prev_fp) / n_neg);
            prev_fp = fp;
        }
    }
    auc
}

fn calculate_f1(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut fn_ = 0.0f32;
    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        let pred = if p >= 0.5 { 1.0f32 } else { 0.0 };
        match (pred as i32, l as i32) {
            (1, 1) => tp += 1.0,
            (1, 0) => fp += 1.0,
            (0, 1) => fn_ += 1.0,
            _ => {}
        }
    }
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall    = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 }
}

/// Bootstrap CI over test-set predictions (f64 version matching statistical_analysis.rs).
/// Returns (mean, lower_95, upper_95).
fn bootstrap_ci(y_true: &[f64], y_pred: &[f64], n: usize, seed: u64) -> (f64, f64, f64) {
    use rand::{SeedableRng, rngs::StdRng};
    use rand::seq::SliceRandom;
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<usize> = (0..y_true.len()).collect();
    let mut scores: Vec<f64> = (0..n).map(|_| {
        let sample: Vec<usize> = (0..y_true.len())
            .map(|_| *indices.choose(&mut rng).unwrap())
            .collect();
        sample.iter()
            .filter(|&&i| (y_pred[i] > 0.5) == (y_true[i] > 0.5))
            .count() as f64 / sample.len() as f64
    }).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean  = scores.iter().sum::<f64>() / scores.len() as f64;
    let lower = scores[(0.025 * scores.len() as f64) as usize];
    let upper = scores[(0.975 * scores.len() as f64) as usize];
    (mean, lower, upper)
}
```

**`evaluate_detailed` helper** (collects per-example predictions for full evaluation):

```rust
fn evaluate_detailed(
    model:         &mut CrossEncoderModel,
    data:          &[(usize, usize, f32)],
    tf_expr_map:   &std::collections::HashMap<usize, ndarray::Array1<f32>>,
    gene_expr_map: &std::collections::HashMap<usize, ndarray::Array1<f32>>,
    expr_dim:      usize,
    batch_size:    usize,
) -> (f32, f32, f32) {
    // Returns (accuracy, auroc, f1)
    let mut all_preds:  Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();

    for start in (0..data.len()).step_by(batch_size) {
        let end   = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx,  gene_expr_map, expr_dim);

        let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        // Apply sigmoid to get probabilities (logits → probs)
        let probs: Vec<f32> = logits.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        all_preds.extend_from_slice(&probs);
        all_labels.extend_from_slice(&labels);
    }

    let n_correct = all_preds.iter().zip(all_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0))
        .count();
    let accuracy = n_correct as f32 / all_preds.len() as f32;
    let auroc    = calculate_auroc(&all_preds, &all_labels);
    let f1       = calculate_f1(&all_preds, &all_labels);
    (accuracy, auroc, f1)
}
```

**Seed loop structure** (design sketch — logic is inlined directly in `main()`, NOT a separate function):

**`main()` structure:**

```
1. Load priors + expression data (once)
2. Build base 1:1 dataset; shuffle with base_seed
3. Split 70/15/15 (fixed splits used for all runs)
4. For neg_ratio in [1, 5]:
   a. Print header
   b. For seed in seeds:
      - Init model(seed), AdamState
      - Train 60 epochs with Adam:
          - Sample training batch
          - For 5:1 ratio: build training examples with 5× negatives per positive
            (val/test stay 1:1 from the base split)
          - Forward → bce_loss_backward → backward → adam_step → zero_grad
          - Every 10 epochs: evaluate on val_data for early stopping
      - Test evaluation: accuracy, AUROC, F1
   c. Compute mean/std/ensemble accuracy
   d. Bootstrap CI from best-seed test predictions
   e. Write results/cross_encoder_{1to1,5to1}.json
```

**Notes on 5:1 training data:**
For the 5:1 ratio, the training examples are rebuilt to have 5× negatives: `builder.sample_negative_examples(positives.len() * 5, seed)`. Val and test data are always the base 1:1 split.

**Ensemble accuracy** — same as `train_ensemble.rs`: average predictions from all 5 models over the test set, then threshold at 0.5.

**Result JSON schema:**
```json
{
    "model": "cross_encoder",
    "neg_ratio": 1,
    "seeds": [42, 123, 456, 789, 1337],
    "seed_accuracies": [0.801, ...],
    "seed_aurocs":     [0.812, ...],
    "seed_f1s":        [0.839, ...],
    "mean_accuracy":   0.800,
    "std_accuracy":    0.017,
    "ensemble_accuracy": 0.830,
    "bootstrap_ci_lower": 0.793,
    "bootstrap_ci_upper": 0.810
}
```

- [ ] **Step 1: Write smoke test**

Add at the bottom of the new file (the test runs 2 epochs / 2 seeds on synthetic data and checks JSON is written):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_json_is_written() {
        // Minimal smoke test: confirms result JSON schema is valid
        let accs  = vec![0.80f32, 0.81];
        let aurocs = vec![0.81f32, 0.82];
        let f1s   = vec![0.82f32, 0.83];
        let mean_acc = accs.iter().sum::<f32>() / accs.len() as f32;
        let std_acc  = (accs.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>()
                        / accs.len() as f32).sqrt();

        let result = serde_json::json!({
            "model": "cross_encoder",
            "neg_ratio": 1,
            "seeds": [42u64, 123],
            "seed_accuracies": accs,
            "seed_aurocs": aurocs,
            "seed_f1s": f1s,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "ensemble_accuracy": 0.82f32,
            "bootstrap_ci_lower": 0.79f64,
            "bootstrap_ci_upper": 0.83f64,
        });

        // Validate round-trip serialization
        let s = serde_json::to_string(&result).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["model"].as_str().unwrap(), "cross_encoder");
        assert!(v["seed_accuracies"].is_array());
        assert!(v["bootstrap_ci_lower"].is_number());
    }
}
```

- [ ] **Step 2: Run smoke test to confirm it compiles and passes**

```bash
cargo test --bin train_cross_encoder 2>&1 | grep -E "FAILED|ok|error"
```
Expected: `test tests::test_result_json_is_written ... ok`

- [ ] **Step 3: Write full `train_cross_encoder.rs`**

Write the entire file. Assemble all the pieces defined above in this order:
1. Module docstring + `use` imports (shown below)
2. `AdamState` struct + `impl AdamState` (shown above in "AdamState struct")
3. `adam_step` function (shown above in "adam_step function")
4. `build_expr_batch`, `calculate_auroc`, `calculate_f1`, `bootstrap_ci` helpers (shown above in "Helper functions")
5. `evaluate_detailed` function (shown above — use `(f32, f32, f32)` return, not 4-tuple)
6. `build_expression_maps` function (shown above in `main()` section)
7. `main()` function (shown below; the seed loop is inlined in `main()` — do **not** create a separate `run_seeds` function, the signature above was a design sketch only)
8. `#[cfg(test)]` block (written in Step 1)

The `use` imports and `main()` body:

```rust
/// Phase 8 Model D: Cross-Encoder training
/// Trains CrossEncoderModel at 1:1 and 5:1 negative ratios, 5 seeds each.
/// Writes results/cross_encoder_1to1.json and results/cross_encoder_5to1.json
use module_regularized_grn::{
    Config,
    models::cross_encoder::CrossEncoderModel,
    models::nn::{bce_loss, bce_loss_backward},
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Phase 8: Cross-Encoder Training ===\n");

    let config   = Config::load_default()?;
    let priors   = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder  = PriorDatasetBuilder::new(priors.clone());
    let num_tfs  = builder.num_tfs();
    let num_genes = builder.num_genes();

    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    let expr_dim = expr_data.n_cell_types;

    // Build expression maps (same pattern as train_ensemble.rs)
    let (tf_expr_map, gene_expr_map) = build_expression_maps(&builder, &expr_data, num_tfs, num_genes);

    // Base 1:1 dataset — fixed splits used for ALL runs (both ratios)
    let base_seed = config.project.seed;
    let mut rng   = StdRng::seed_from_u64(base_seed);
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, g)| (tf, g, 1.0f32)));
    examples.extend(negatives.iter().map(|&(tf, g)| (tf, g, 0.0f32)));
    examples.shuffle(&mut rng);

    let n       = examples.len();
    let n_train = (n as f32 * 0.7) as usize;
    let n_val   = (n as f32 * 0.15) as usize;
    let train_base = examples[..n_train].to_vec();
    let val_data   = examples[n_train..n_train+n_val].to_vec();
    let test_data  = examples[n_train+n_val..].to_vec();

    println!("Train: {} | Val: {} | Test: {}", train_base.len(), val_data.len(), test_data.len());
    println!("TFs: {} | Genes: {} | expr_dim: {}\n", num_tfs, num_genes, expr_dim);

    let seeds = [42u64, 123, 456, 789, 1337];
    let embed_dim  = 512usize;
    let hidden_dim = 512usize;
    let lr         = 0.005f32;
    let batch_size = 256usize;
    let epochs     = 60usize;

    std::fs::create_dir_all("results")?;

    for &neg_ratio in &[1usize, 5] {
        println!("=== neg_ratio = {}:1 ===", neg_ratio);

        // Build training data at this ratio
        let train_data = if neg_ratio == 1 {
            train_base.clone()
        } else {
            let pos_train: Vec<_> = train_base.iter().filter(|x| x.2 == 1.0).cloned().collect();
            let neg_train_extra = builder.sample_negative_examples(
                pos_train.len() * neg_ratio, base_seed + neg_ratio as u64
            );
            let mut data: Vec<(usize, usize, f32)> = pos_train;
            data.extend(neg_train_extra.iter().map(|&(tf, g)| (tf, g, 0.0f32)));
            let mut r = StdRng::seed_from_u64(base_seed + 1000);
            data.shuffle(&mut r);
            data
        };

        let mut seed_accuracies = Vec::new();
        let mut seed_aurocs     = Vec::new();
        let mut seed_f1s        = Vec::new();
        let mut all_test_preds_for_ensemble: Vec<Vec<f32>> = Vec::new();
        let mut best_seed_final_val_acc = 0.0f32;  // best val acc seen across all seeds
        let mut best_test_preds: Vec<f32> = Vec::new();
        let test_labels_once: Vec<f32> = test_data.iter().map(|x| x.2).collect();

        for &seed in &seeds {
            println!("  seed {}:", seed);
            let mut model = CrossEncoderModel::new(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, seed);
            let mut state = AdamState::new(&model);
            let mut best_seed_val_acc = 0.0f32;
            let mut patience = 0usize;

            for epoch in 0..epochs {
                let mut epoch_data = train_data.clone();
                let mut epoch_rng  = StdRng::seed_from_u64(seed + epoch as u64);
                epoch_data.shuffle(&mut epoch_rng);

                for start in (0..epoch_data.len()).step_by(batch_size) {
                    let end   = (start + batch_size).min(epoch_data.len());
                    let batch = &epoch_data[start..end];
                    let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                    let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                    let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                    let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                    let gene_e = build_expr_batch(&gene_idx,  &gene_expr_map, expr_dim);

                    let logits      = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                    let labels_arr  = Array1::from(labels);
                    let grad        = bce_loss_backward(&logits, &labels_arr);
                    model.backward(&grad);
                    // Adam update per batch (not per epoch — avoids gradient accumulation)
                    adam_step(&mut model, &mut state, lr, 1.0);
                    model.zero_grad();
                }

                // Val check every 5 epochs
                if epoch % 5 == 0 || epoch == epochs - 1 {
                    let (val_acc, _, _) = evaluate_detailed(
                        &mut model, &val_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
                    );
                    if val_acc > best_seed_val_acc {
                        best_seed_val_acc = val_acc;
                        patience = 0;
                    } else {
                        patience += 1;
                        if patience >= 3 { break; }  // 15-epoch patience (check every 5)
                    }
                }
            }

            let (test_acc, test_auroc, test_f1) = evaluate_detailed(
                &mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
            );
            println!("    acc={:.2}% auroc={:.4} f1={:.4}", test_acc*100.0, test_auroc, test_f1);

            // Collect test predictions for ensemble averaging
            let mut preds: Vec<f32> = Vec::new();
            for start in (0..test_data.len()).step_by(batch_size) {
                let end   = (start + batch_size).min(test_data.len());
                let batch = &test_data[start..end];
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx,  &gene_expr_map, expr_dim);
                let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                preds.extend(logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())));
            }

            // Select best seed by val accuracy (not test accuracy — spec requirement)
            if best_seed_val_acc > best_seed_final_val_acc {
                best_seed_final_val_acc = best_seed_val_acc;
                best_test_preds = preds.clone();
            }
            all_test_preds_for_ensemble.push(preds);

            seed_accuracies.push(test_acc);
            seed_aurocs.push(test_auroc);
            seed_f1s.push(test_f1);
        }

        // Ensemble accuracy
        let n_test = test_data.len();
        let ensemble_preds: Vec<f32> = (0..n_test)
            .map(|i| all_test_preds_for_ensemble.iter().map(|p| p[i]).sum::<f32>()
                     / seeds.len() as f32)
            .collect();
        let ensemble_acc = ensemble_preds.iter().zip(test_labels_once.iter())
            .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0))
            .count() as f32 / n_test as f32;

        // Bootstrap CI from best-seed predictions
        let bp: Vec<f64> = best_test_preds.iter().map(|&x| x as f64).collect();
        let bl: Vec<f64> = test_labels_once.iter().map(|&x| x as f64).collect();
        let (_, ci_lower, ci_upper) = bootstrap_ci(&bl, &bp, 1000, 42);

        // Stats
        let mean_acc = seed_accuracies.iter().sum::<f32>() / seeds.len() as f32;
        let std_acc  = (seed_accuracies.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>()
                        / seeds.len() as f32).sqrt();

        println!("  mean={:.2}% ±{:.2}% | ensemble={:.2}%\n",
                 mean_acc*100.0, std_acc*100.0, ensemble_acc*100.0);

        let result = serde_json::json!({
            "model": "cross_encoder",
            "neg_ratio": neg_ratio,
            "seeds": seeds,
            "seed_accuracies": seed_accuracies,
            "seed_aurocs": seed_aurocs,
            "seed_f1s": seed_f1s,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "ensemble_accuracy": ensemble_acc,
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
        });

        let fname = if neg_ratio == 1 {
            "results/cross_encoder_1to1.json"
        } else {
            "results/cross_encoder_5to1.json"
        };
        std::fs::write(fname, serde_json::to_string_pretty(&result)?)?;
        println!("✓ Saved {}", fname);
    }

    Ok(())
}

fn build_expression_maps(
    builder:   &PriorDatasetBuilder,
    expr_data: &ExpressionData,
    _num_tfs:  usize,
    _num_genes: usize,
) -> (HashMap<usize, Array1<f32>>, HashMap<usize, Array1<f32>>) {
    let mut gene_to_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in expr_data.gene_names.iter().enumerate() {
        gene_to_idx.insert(name.clone(), i);
    }
    let expr_dim = expr_data.n_cell_types;
    let mut tf_map:   HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_map: HashMap<usize, Array1<f32>> = HashMap::new();

    for (name, &idx) in builder.get_tf_vocab().iter() {
        let v = gene_to_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned())
            .unwrap_or_else(|| Array1::zeros(expr_dim));
        tf_map.insert(idx, v);
    }
    for (name, &idx) in builder.get_gene_vocab().iter() {
        let v = gene_to_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned())
            .unwrap_or_else(|| Array1::zeros(expr_dim));
        gene_map.insert(idx, v);
    }
    (tf_map, gene_map)
}
```

- [ ] **Step 4: Build release binary**

```bash
cargo build --release --bin train_cross_encoder 2>&1 | grep -E "^error|Finished"
```
Expected: `Finished release`

- [ ] **Step 5: Run smoke test**

```bash
cargo test --bin train_cross_encoder 2>&1 | grep -E "FAILED|ok"
```
Expected: `test tests::test_result_json_is_written ... ok`

- [ ] **Step 6: Commit**

```bash
git add src/bin/train_cross_encoder.rs
git commit -m "feat: add train_cross_encoder.rs (Model D, inline Adam, 2 neg ratios)"
```

---

## Chunk 3: Modify train_standard_mlp.rs + compare_models.rs

### Task 3: Modify train_standard_mlp.rs

**Files:**
- Modify: `src/bin/train_standard_mlp.rs`

**Changes needed:**
1. Update hyperparams to match cross-encoder: `embed_dim=512`, `hidden_dim=512`, `output_dim=512`
2. Add seed iteration loop over `[42u64, 123, 456, 789, 1337]`
3. Add `--neg-ratio N` CLI flag (default 1) — changes training data only, not val/test
4. Add per-seed AUROC and F1 computation (`calculate_auroc` / `calculate_f1` — copy from train_ensemble.rs pattern)
5. Add bootstrap CI over best-seed test predictions (`bootstrap_ci` — copy from train_cross_encoder.rs)
6. Write new unified schema to `results/two_tower_1to1.json` (or `two_tower_5to1.json`)
7. Keep the existing `results/standard_mlp_results.json` write for backward compatibility

**CLI flag parsing** (standard Rust `std::env::args`):
```rust
let neg_ratio: usize = std::env::args()
    .position(|a| a == "--neg-ratio")
    .and_then(|i| std::env::args().nth(i + 1))
    .and_then(|v| v.parse().ok())
    .unwrap_or(1);
```

**Output filename logic:**
```rust
let out_file = if neg_ratio == 1 {
    "results/two_tower_1to1.json"
} else {
    "results/two_tower_5to1.json"
};
```

**Note on `HybridEmbeddingModel.forward()` convention:** The existing model applies sigmoid internally in `forward()` and then `bce_loss_backward` applies sigmoid again. This is a pre-existing behaviour — do NOT change it in this task (it would alter existing results). Just add the seed loop and neg-ratio on top.

- [ ] **Step 1: Check current hyperparams**

```bash
grep -n "embed_dim\|hidden_dim\|output_dim\|learning_rate\|epochs\b\|batch_size" src/bin/train_standard_mlp.rs
```

Expected: `embed_dim = 64`, `hidden_dim = 128`, `output_dim = 64`, `learning_rate = 0.001`, `epochs = 50`.

- [ ] **Step 2: Update hyperparams**

The existing hyperparams are in two separate code blocks — update both:
- **Model init block (~line 113):** `embed_dim`, `hidden_dim`, `output_dim`
- **Training params block (~lines 139-143):** `learning_rate`, `batch_size`, `epochs`, `weight_decay`

Change to:
```rust
let embed_dim   = 512;
let hidden_dim  = 512;
let output_dim  = 512;
let learning_rate = 0.005f32;
let batch_size  = 256;
let epochs      = 60;
let weight_decay = 0.01f32;  // kept — HybridEmbeddingModel.update_with_weight_decay uses this
```

- [ ] **Step 3: Add CLI neg-ratio parsing** (at top of `main()`, before Config::load_default):

```rust
let neg_ratio: usize = std::env::args()
    .position(|a| a == "--neg-ratio")
    .and_then(|i| std::env::args().nth(i + 1))
    .and_then(|v| v.parse().ok())
    .unwrap_or(1);
println!("Negative ratio: {}:1", neg_ratio);
```

- [ ] **Step 4: Wrap training in seed loop**

**Prerequisite:** Chunk 2 (train_cross_encoder.rs) must be implemented first. The helpers `calculate_auroc`, `calculate_f1`, and `bootstrap_ci` were originally taken from `train_ensemble.rs` and `src/bin/statistical_analysis.rs` — copy the identical implementations verbatim from `train_cross_encoder.rs` (Chunk 2).

Replace the single-seed training block with a loop over `seeds = [42u64, 123, 456, 789, 1337]`. Key design decisions:

- **Data split is FIXED (done once, outside the seed loop)** — use the existing single-split logic. Only the model is re-initialized per seed.
- **Training data for neg_ratio=5:** use `builder.sample_negative_examples(positives.len() * 5, base_seed + neg_ratio as u64)` where `base_seed = config.project.seed` (the existing `seed` variable at line 25). This ensures the negative sample set is consistent across seeds for a given ratio.
- **Val and test sets stay at 1:1** regardless of neg_ratio (from the base split).
- **Adam update:** call `adam_step` per batch, not per epoch.
- **Best-seed tracking:** track best val accuracy across seeds; store that seed's test predictions for bootstrap CI.
- **Remove** the existing inline F1 computation (the file currently computes `f1` inline using `tp`, `fp`, `fn_count` variables). Replace with `calculate_f1(&all_test_preds, &all_test_labels)`.

Inside the seed loop:
```rust
for &seed in &seeds {
    let mut model = HybridEmbeddingModel::new(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, 0.05, 0.01, seed);
    // ... existing epoch training loop (unchanged, using model.update_with_weight_decay) ...
    // After training, evaluate test set:
    let mut all_test_preds: Vec<f32> = Vec::new();
    let mut all_test_labels: Vec<f32> = Vec::new();
    // ... collect predictions over test_data batches ...
    let test_acc  = all_test_preds.iter().zip(all_test_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0))
        .count() as f32 / all_test_preds.len() as f32;
    let test_auroc = calculate_auroc(&all_test_preds, &all_test_labels);
    let test_f1    = calculate_f1(&all_test_preds, &all_test_labels);
    seed_accuracies.push(test_acc);
    seed_aurocs.push(test_auroc);
    seed_f1s.push(test_f1);
    if best_seed_val_acc > best_final_val_acc {
        best_final_val_acc = best_seed_val_acc;
        best_test_preds    = all_test_preds.clone();
    }
    all_test_preds_for_ensemble.push(all_test_preds);
}
```

Before the seed loop, declare:
```rust
let seeds = [42u64, 123, 456, 789, 1337];
let mut seed_accuracies: Vec<f32> = Vec::new();
let mut seed_aurocs:     Vec<f32> = Vec::new();
let mut seed_f1s:        Vec<f32> = Vec::new();
let mut all_test_preds_for_ensemble: Vec<Vec<f32>> = Vec::new();
let mut best_final_val_acc  = 0.0f32;
let mut best_test_preds:    Vec<f32> = Vec::new();
let test_labels_once: Vec<f32> = test_data.iter().map(|x| x.2).collect();
```

- [ ] **Step 5: Write new unified schema**

After the seed loop, compute aggregate stats and write:
```rust
// Ensemble accuracy (average probabilities from all seeds, threshold at 0.5)
let n_test = test_labels_once.len();
let ensemble_preds: Vec<f32> = (0..n_test)
    .map(|i| all_test_preds_for_ensemble.iter().map(|p| p[i]).sum::<f32>()
             / seeds.len() as f32)
    .collect();
let ensemble_acc = ensemble_preds.iter().zip(test_labels_once.iter())
    .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0))
    .count() as f32 / n_test as f32;

// Bootstrap CI from best-seed predictions
let bp: Vec<f64> = best_test_preds.iter().map(|&x| x as f64).collect();
let bl: Vec<f64> = test_labels_once.iter().map(|&x| x as f64).collect();
let (_, ci_lower, ci_upper) = bootstrap_ci(&bl, &bp, 1000, 42);

let mean_acc = seed_accuracies.iter().sum::<f32>() / seeds.len() as f32;
let std_acc  = (seed_accuracies.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>()
                / seeds.len() as f32).sqrt();

let out_file = if neg_ratio == 1 { "results/two_tower_1to1.json" } else { "results/two_tower_5to1.json" };
std::fs::create_dir_all("results")?;
let result = serde_json::json!({
    "model": "two_tower",
    "neg_ratio": neg_ratio,
    "seeds": [42u64, 123, 456, 789, 1337],
    "seed_accuracies": seed_accuracies,
    "seed_aurocs": seed_aurocs,
    "seed_f1s": seed_f1s,
    "mean_accuracy": mean_acc,
    "std_accuracy": std_acc,
    "ensemble_accuracy": ensemble_acc,
    "bootstrap_ci_lower": ci_lower,
    "bootstrap_ci_upper": ci_upper,
});
std::fs::write(out_file, serde_json::to_string_pretty(&result)?)?;
println!("✓ Saved {}", out_file);
```

Keep the existing `standard_mlp_results.json` write (just append it after the new write).

**Note on neg_ratio:** The CLI flag `--neg-ratio` (from Step 3) is already parsed into `neg_ratio`. Use that value directly — do NOT add a `for &neg_ratio in &[1, 5]` loop here, as that would shadow the CLI variable and produce silent logic errors. To run both ratios, invoke the binary twice: once without the flag (defaults to 1:1) and once with `--neg-ratio 5`.

- [ ] **Step 6: Build**

```bash
cargo build --release --bin train_standard_mlp 2>&1 | grep -E "^error|Finished"
```
Expected: `Finished release`

- [ ] **Step 7: Commit**

```bash
git add src/bin/train_standard_mlp.rs
git commit -m "feat: update train_standard_mlp with seed loop, neg-ratio flag, AUROC/F1"
```

---

### Task 4: Implement compare_models.rs

**Files:**
- Create: `src/bin/compare_models.rs`

**What it does:** Reads up to 4 result JSONs (skipping missing ones with a note), prints a formatted comparison table, writes `results/model_comparison.json`.

**Expected JSON schema** (all result files share this schema from Tasks 2+3):
```json
{
    "model": "cross_encoder",
    "neg_ratio": 1,
    "mean_accuracy": 0.80,
    "std_accuracy":  0.017,
    "seed_aurocs":   [0.81, ...],
    "seed_f1s":      [0.83, ...],
    "ensemble_accuracy": 0.83,
    "bootstrap_ci_lower": 0.793,
    "bootstrap_ci_upper": 0.810
}
```

Mean AUROC = average of `seed_aurocs`. Mean F1 = average of `seed_f1s`.

- [ ] **Step 1: Write test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_rendering() {
        let rows = vec![
            ModelRow {
                model:     "Two-Tower".to_string(),
                neg_ratio: 1,
                mean_acc:  0.8014,
                std_acc:   0.017,
                mean_auroc: 0.814,
                mean_f1:   0.839,
                ensemble_acc: 0.8306,
                ci_lower:  0.793,
                ci_upper:  0.810,
            },
        ];
        let table = render_table(&rows);
        assert!(table.contains("Two-Tower"), "table must contain model name");
        assert!(table.contains("1:1"),       "table must contain neg ratio");
        assert!(table.contains("80.14"),     "table must contain accuracy");
        assert!(table.contains("AUROC"),     "table must have AUROC header");
    }
}
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cargo test --bin compare_models 2>&1 | head -10
```
Expected: compile error.

- [ ] **Step 3: Implement compare_models.rs**

Replace the **entire file contents** from Step 1 with the complete implementation below (the test block from Step 1 is included at the bottom):

```rust
/// compare_models — reads result JSONs and prints comparison table.
/// Input files (missing files skipped):
///   results/two_tower_1to1.json
///   results/two_tower_5to1.json
///   results/cross_encoder_1to1.json
///   results/cross_encoder_5to1.json
/// Output:
///   stdout: formatted table
///   results/model_comparison.json
use anyhow::Result;
use serde_json::Value;

struct ModelRow {
    model:        String,
    neg_ratio:    usize,
    mean_acc:     f64,
    std_acc:      f64,
    mean_auroc:   f64,
    mean_f1:      f64,
    ensemble_acc: f64,
    ci_lower:     f64,
    ci_upper:     f64,
}

fn load_row(path: &str) -> Option<ModelRow> {
    let content = std::fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&content).ok()?;

    let aurocs: Vec<f64> = v["seed_aurocs"].as_array()?
        .iter().filter_map(|x| x.as_f64()).collect();
    let f1s: Vec<f64> = v["seed_f1s"].as_array()?
        .iter().filter_map(|x| x.as_f64()).collect();
    let mean_auroc = if aurocs.is_empty() { 0.0 } else { aurocs.iter().sum::<f64>() / aurocs.len() as f64 };
    let mean_f1    = if f1s.is_empty()    { 0.0 } else { f1s.iter().sum::<f64>()    / f1s.len() as f64    };

    let model_key = v["model"].as_str().unwrap_or("unknown");
    let model_name = match model_key {
        "two_tower"     => "Two-Tower",
        "cross_encoder" => "Cross-Encoder",
        other => other,
    };

    Some(ModelRow {
        model:        model_name.to_string(),
        neg_ratio:    v["neg_ratio"].as_u64().unwrap_or(1) as usize,
        mean_acc:     v["mean_accuracy"].as_f64().unwrap_or(0.0),
        std_acc:      v["std_accuracy"].as_f64().unwrap_or(0.0),
        mean_auroc,
        mean_f1,
        ensemble_acc: v["ensemble_accuracy"].as_f64().unwrap_or(0.0),
        ci_lower:     v["bootstrap_ci_lower"].as_f64().unwrap_or(0.0),
        ci_upper:     v["bootstrap_ci_upper"].as_f64().unwrap_or(0.0),
    })
}

fn render_table(rows: &[ModelRow]) -> String {
    let header = format!(
        "{:<18} | {:>9} | {:>16} | {:>6} | {:>6} | {:>18}\n{}\n",
        "Model", "Neg Ratio", "Accuracy (±std)", "AUROC", "F1", "95% CI",
        "-".repeat(90)
    );
    let body: String = rows.iter().map(|r| {
        format!(
            "{:<18} | {:>9} | {:>7.2}% ±{:.2}%  | {:.4} | {:.4} | [{:.2}%, {:.2}%]\n",
            r.model,
            format!("{}:1", r.neg_ratio),
            r.mean_acc * 100.0,
            r.std_acc  * 100.0,
            r.mean_auroc,
            r.mean_f1,
            r.ci_lower * 100.0,
            r.ci_upper * 100.0,
        )
    }).collect();
    header + &body
}

fn main() -> Result<()> {
    println!("=== Model Comparison ===\n");

    let sources = [
        ("results/two_tower_1to1.json",     "Two-Tower 1:1"),
        ("results/two_tower_5to1.json",     "Two-Tower 5:1"),
        ("results/cross_encoder_1to1.json", "Cross-Encoder 1:1"),
        ("results/cross_encoder_5to1.json", "Cross-Encoder 5:1"),
    ];

    let mut rows: Vec<ModelRow> = Vec::new();
    for (path, label) in &sources {
        match load_row(path) {
            Some(row) => rows.push(row),
            None      => println!("  (skipping {}: file not found or invalid)", label),
        }
    }

    if rows.is_empty() {
        println!("No result files found. Run train_standard_mlp and/or train_cross_encoder first.");
        return Ok(());
    }

    let table = render_table(&rows);
    println!("{}", table);

    // Write machine-readable comparison
    std::fs::create_dir_all("results")?;
    let comparison: Vec<serde_json::Value> = rows.iter().map(|r| serde_json::json!({
        "model":           r.model,
        "neg_ratio":       r.neg_ratio,
        "mean_accuracy":   r.mean_acc,
        "std_accuracy":    r.std_acc,
        "mean_auroc":      r.mean_auroc,
        "mean_f1":         r.mean_f1,
        "ensemble_accuracy": r.ensemble_acc,
        "bootstrap_ci_lower": r.ci_lower,
        "bootstrap_ci_upper": r.ci_upper,
    })).collect();
    std::fs::write(
        "results/model_comparison.json",
        serde_json::to_string_pretty(&comparison)?
    )?;
    println!("✓ Saved results/model_comparison.json");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_rendering() {
        let rows = vec![
            ModelRow {
                model:        "Two-Tower".to_string(),
                neg_ratio:    1,
                mean_acc:     0.8014,
                std_acc:      0.017,
                mean_auroc:   0.814,
                mean_f1:      0.839,
                ensemble_acc: 0.8306,
                ci_lower:     0.793,
                ci_upper:     0.810,
            },
        ];
        let table = render_table(&rows);
        assert!(table.contains("Two-Tower"), "table must contain model name");
        assert!(table.contains("1:1"),       "table must contain neg ratio");
        assert!(table.contains("80.14"),     "table must contain accuracy");
        assert!(table.contains("AUROC"),     "table must have AUROC header");
    }
}
```

- [ ] **Step 4: Run test**

```bash
cargo test --bin compare_models 2>&1 | grep -E "FAILED|ok"
```
Expected: `test tests::test_table_rendering ... ok`

- [ ] **Step 5: Build release**

```bash
cargo build --release --bin compare_models 2>&1 | grep -E "^error|Finished"
```
Expected: `Finished release`

- [ ] **Step 6: Full build — zero warnings**

```bash
cargo build --release 2>&1 | grep -E "^error|^warning|Finished"
```
Expected: `Finished release` with zero warnings.

- [ ] **Step 7: Commit**

```bash
git add src/bin/compare_models.rs
git commit -m "feat: add compare_models bin (Phase 8 comparison harness)"
```

---

### Task 5: Final verification

- [ ] **Confirm all 3 new bins build cleanly:**

```bash
cargo build --release --bin train_cross_encoder 2>&1 | grep -E "^error|Finished"
cargo build --release --bin train_standard_mlp  2>&1 | grep -E "^error|Finished"
cargo build --release --bin compare_models      2>&1 | grep -E "^error|Finished"
```
Expected: all three show `Finished release`.

- [ ] **Run all tests — confirm 0 failures:**

```bash
cargo test 2>&1 | grep -E "test result"
```
Expected: all `test result: ok`.

- [ ] **Smoke-test compare_models with no result files (should exit gracefully):**

```bash
cargo run --release --bin compare_models 2>&1
```
Expected: prints "(skipping ...)" lines and "No result files found. Run..." message without panicking.

- [ ] **Final commit if any uncommitted changes remain:**

```bash
git status
git add -A && git commit -m "chore: Phase 8 sub-project 1 complete (cross-encoder + comparison harness)"
```
(only if there are uncommitted changes)
