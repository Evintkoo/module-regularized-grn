/// Gated Cross-Attention Network (GCAN) for GRN edge prediction
///
/// Key innovations over the baseline two-tower MLP:
/// 1. Gated cross-modulation: TF and gene encodings inform each other before scoring
/// 2. Rich interaction features: [tf; gene; tf*gene; tf-gene] captures diverse relationships
/// 3. Deep scoring MLP: replaces linear cosine similarity with non-linear scoring
///
/// This approximates transformer-level cross-attention with far fewer parameters
/// by using gated bilinear interaction instead of full multi-head attention.

use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use rand::SeedableRng;

/// Adam optimizer state for a single parameter array
#[derive(Clone)]
pub struct AdamState {
    pub m: Vec<f32>,  // first moment
    pub v: Vec<f32>,  // second moment
}

impl AdamState {
    pub fn new(size: usize) -> Self {
        Self {
            m: vec![0.0; size],
            v: vec![0.0; size],
        }
    }
}

/// Apply Adam update using iterators (handles non-contiguous ndarray layouts)
pub fn adam_update_iter<'a>(
    params: impl Iterator<Item = &'a mut f32>,
    grads: impl Iterator<Item = &'a f32>,
    state: &mut AdamState,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    weight_decay: f32,
    batch_size: f32,
) {
    let bias_correction1 = 1.0 - beta1.powi(t as i32);
    let bias_correction2 = 1.0 - beta2.powi(t as i32);

    for (i, (p, g)) in params.zip(grads).enumerate() {
        let grad = *g / batch_size + weight_decay * *p;
        state.m[i] = beta1 * state.m[i] + (1.0 - beta1) * grad;
        state.v[i] = beta2 * state.v[i] + (1.0 - beta2) * grad * grad;
        let m_hat = state.m[i] / bias_correction1;
        let v_hat = state.v[i] / bias_correction2;
        *p -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// Layer normalization
#[derive(Clone)]
pub struct LayerNorm {
    pub gamma: Array1<f32>,
    pub beta: Array1<f32>,
    pub grad_gamma: Array1<f32>,
    pub grad_beta: Array1<f32>,
    dim: usize,
    eps: f32,
    // cache
    x_norm_cache: Option<Array2<f32>>,
    std_inv_cache: Option<Array1<f32>>,
}

impl LayerNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            gamma: Array1::ones(dim),
            beta: Array1::zeros(dim),
            grad_gamma: Array1::zeros(dim),
            grad_beta: Array1::zeros(dim),
            dim,
            eps: 1e-5,
            x_norm_cache: None,
            std_inv_cache: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let batch_size = x.nrows();
        let mut output = Array2::zeros(x.raw_dim());
        let mut x_norm = Array2::zeros(x.raw_dim());
        let mut std_inv = Array1::zeros(batch_size);

        for i in 0..batch_size {
            let row = x.row(i);
            let mean = row.mean().unwrap();
            let var = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let si = 1.0 / (var + self.eps).sqrt();
            std_inv[i] = si;

            for j in 0..self.dim {
                x_norm[[i, j]] = (x[[i, j]] - mean) * si;
                output[[i, j]] = x_norm[[i, j]] * self.gamma[j] + self.beta[j];
            }
        }

        self.x_norm_cache = Some(x_norm);
        self.std_inv_cache = Some(std_inv);
        output
    }

    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let x_norm = self.x_norm_cache.as_ref().unwrap();
        let std_inv = self.std_inv_cache.as_ref().unwrap();
        let batch_size = grad_output.nrows();
        let d = self.dim as f32;

        // Accumulate gamma/beta gradients
        for j in 0..self.dim {
            for i in 0..batch_size {
                self.grad_gamma[j] += grad_output[[i, j]] * x_norm[[i, j]];
                self.grad_beta[j] += grad_output[[i, j]];
            }
        }

        // Compute input gradient
        let mut grad_x = Array2::zeros(grad_output.raw_dim());
        for i in 0..batch_size {
            let si = std_inv[i];
            let mut dx_norm_sum = 0.0f32;
            let mut dx_norm_xnorm_sum = 0.0f32;

            for j in 0..self.dim {
                let dn = grad_output[[i, j]] * self.gamma[j];
                dx_norm_sum += dn;
                dx_norm_xnorm_sum += dn * x_norm[[i, j]];
            }

            for j in 0..self.dim {
                let dn = grad_output[[i, j]] * self.gamma[j];
                grad_x[[i, j]] = si * (dn - (dx_norm_sum + x_norm[[i, j]] * dx_norm_xnorm_sum) / d);
            }
        }

        grad_x
    }

    pub fn zero_grad(&mut self) {
        self.grad_gamma.fill(0.0);
        self.grad_beta.fill(0.0);
    }
}

/// Apply inverted dropout: scale by 1/(1-p) during training, identity during eval
fn apply_dropout(x: &Array2<f32>, rate: f32, training: bool, rng: &mut impl Rng) -> (Array2<f32>, Option<Array2<f32>>) {
    if !training || rate <= 0.0 {
        return (x.clone(), None);
    }
    let scale = 1.0 / (1.0 - rate);
    let mask = Array2::from_shape_fn(x.raw_dim(), |_| {
        if rng.gen::<f32>() > rate { scale } else { 0.0 }
    });
    (x * &mask, Some(mask))
}

/// Gated Cross-Attention Network
pub struct CrossAttentionModel {
    // Embeddings
    tf_embed: Array2<f32>,
    gene_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,

    // TF encoder (3 layers with layer norm)
    tf_fc1: LinearLayer,
    tf_ln1: LayerNorm,
    tf_fc2: LinearLayer,
    tf_ln2: LayerNorm,
    tf_fc3: LinearLayer,

    // Gene encoder
    gene_fc1: LinearLayer,
    gene_ln1: LayerNorm,
    gene_fc2: LinearLayer,
    gene_ln2: LayerNorm,
    gene_fc3: LinearLayer,

    // Cross-modulation gates
    tf_gate_fc: LinearLayer,
    gene_gate_fc: LinearLayer,
    tf_cross_fc: LinearLayer,
    gene_cross_fc: LinearLayer,

    // Scoring head
    score_fc1: LinearLayer,
    score_ln1: LayerNorm,
    score_fc2: LinearLayer,
    score_fc3: LinearLayer,

    // Dimensions
    embed_dim: usize,
    expr_dim: usize,
    enc_dim: usize,

    // Dropout
    dropout_rate: f32,
    training: bool,

    // Forward pass caches
    tf_indices_cache: Vec<usize>,
    gene_indices_cache: Vec<usize>,
    tf_embed_out: Option<Array2<f32>>,
    gene_embed_out: Option<Array2<f32>>,
    tf_concat: Option<Array2<f32>>,
    gene_concat: Option<Array2<f32>>,
    // Encoder caches
    tf_h1_pre: Option<Array2<f32>>,
    tf_h1: Option<Array2<f32>>,
    tf_h1_ln: Option<Array2<f32>>,
    tf_h2_pre: Option<Array2<f32>>,
    tf_h2: Option<Array2<f32>>,
    tf_h2_ln: Option<Array2<f32>>,
    tf_enc: Option<Array2<f32>>,
    gene_h1_pre: Option<Array2<f32>>,
    gene_h1: Option<Array2<f32>>,
    gene_h1_ln: Option<Array2<f32>>,
    gene_h2_pre: Option<Array2<f32>>,
    gene_h2: Option<Array2<f32>>,
    gene_h2_ln: Option<Array2<f32>>,
    gene_enc: Option<Array2<f32>>,
    // Cross-modulation caches
    joint: Option<Array2<f32>>,
    tf_gate_pre: Option<Array2<f32>>,
    tf_gate: Option<Array2<f32>>,
    gene_gate_pre: Option<Array2<f32>>,
    gene_gate: Option<Array2<f32>>,
    tf_cross_pre: Option<Array2<f32>>,
    tf_cross: Option<Array2<f32>>,
    gene_cross_pre: Option<Array2<f32>>,
    gene_cross: Option<Array2<f32>>,
    tf_mod: Option<Array2<f32>>,
    gene_mod: Option<Array2<f32>>,
    // Interaction caches
    interaction: Option<Array2<f32>>,
    tf_times_gene: Option<Array2<f32>>,
    tf_minus_gene: Option<Array2<f32>>,
    // Scoring caches
    score_h1_pre: Option<Array2<f32>>,
    score_h1: Option<Array2<f32>>,
    score_h1_ln: Option<Array2<f32>>,
    score_h2_pre: Option<Array2<f32>>,
    score_h2: Option<Array2<f32>>,
    score_logit: Option<Array2<f32>>,
    score_out: Option<Array1<f32>>,
    // Dropout mask caches (for backward pass)
    drop_tf_h1: Option<Array2<f32>>,
    drop_tf_h2: Option<Array2<f32>>,
    drop_gene_h1: Option<Array2<f32>>,
    drop_gene_h2: Option<Array2<f32>>,
    drop_score_h1: Option<Array2<f32>>,
    drop_score_h2: Option<Array2<f32>>,
    // RNG for dropout
    drop_rng: rand::rngs::StdRng,
}

impl CrossAttentionModel {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,
        expr_dim: usize,
        hidden_dim: usize,  // encoder hidden dim
        enc_dim: usize,      // encoding output dim
        dropout_rate: f32,
        seed: u64,
    ) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);
        let std_dev = 0.05;
        let dist = Normal::new(0.0, std_dev).unwrap();

        let tf_embed = Array2::random_using((num_tfs, embed_dim), dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, embed_dim), dist, &mut rng);

        let input_dim = embed_dim + expr_dim;
        let mut s = seed;
        let mut next_seed = || { s += 1; s };

        // TF encoder
        let tf_fc1 = LinearLayer::new(input_dim, hidden_dim, next_seed());
        let tf_ln1 = LayerNorm::new(hidden_dim);
        let tf_fc2 = LinearLayer::new(hidden_dim, hidden_dim, next_seed());
        let tf_ln2 = LayerNorm::new(hidden_dim);
        let tf_fc3 = LinearLayer::new(hidden_dim, enc_dim, next_seed());

        // Gene encoder
        let gene_fc1 = LinearLayer::new(input_dim, hidden_dim, next_seed());
        let gene_ln1 = LayerNorm::new(hidden_dim);
        let gene_fc2 = LinearLayer::new(hidden_dim, hidden_dim, next_seed());
        let gene_ln2 = LayerNorm::new(hidden_dim);
        let gene_fc3 = LinearLayer::new(hidden_dim, enc_dim, next_seed());

        // Cross-modulation (takes concat of both encodings)
        let tf_gate_fc = LinearLayer::new(2 * enc_dim, enc_dim, next_seed());
        let gene_gate_fc = LinearLayer::new(2 * enc_dim, enc_dim, next_seed());
        let tf_cross_fc = LinearLayer::new(enc_dim, enc_dim, next_seed());
        let gene_cross_fc = LinearLayer::new(enc_dim, enc_dim, next_seed());

        // Scoring head: 4*enc_dim (interaction) → 2*enc_dim → enc_dim → 1
        let score_fc1 = LinearLayer::new(4 * enc_dim, 2 * enc_dim, next_seed());
        let score_ln1 = LayerNorm::new(2 * enc_dim);
        let score_fc2 = LinearLayer::new(2 * enc_dim, enc_dim, next_seed());
        let score_fc3 = LinearLayer::new(enc_dim, 1, next_seed());

        Self {
            tf_embed,
            gene_embed,
            tf_embed_grad: Array2::zeros((num_tfs, embed_dim)),
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            tf_fc1, tf_ln1, tf_fc2, tf_ln2, tf_fc3,
            gene_fc1, gene_ln1, gene_fc2, gene_ln2, gene_fc3,
            tf_gate_fc, gene_gate_fc, tf_cross_fc, gene_cross_fc,
            score_fc1, score_ln1, score_fc2, score_fc3,
            embed_dim, expr_dim, enc_dim,
            dropout_rate,
            training: true,
            tf_indices_cache: Vec::new(),
            gene_indices_cache: Vec::new(),
            tf_embed_out: None, gene_embed_out: None,
            tf_concat: None, gene_concat: None,
            tf_h1_pre: None, tf_h1: None, tf_h1_ln: None,
            tf_h2_pre: None, tf_h2: None, tf_h2_ln: None, tf_enc: None,
            gene_h1_pre: None, gene_h1: None, gene_h1_ln: None,
            gene_h2_pre: None, gene_h2: None, gene_h2_ln: None, gene_enc: None,
            joint: None,
            tf_gate_pre: None, tf_gate: None,
            gene_gate_pre: None, gene_gate: None,
            tf_cross_pre: None, tf_cross: None,
            gene_cross_pre: None, gene_cross: None,
            tf_mod: None, gene_mod: None,
            interaction: None, tf_times_gene: None, tf_minus_gene: None,
            score_h1_pre: None, score_h1: None, score_h1_ln: None,
            score_h2_pre: None, score_h2: None,
            score_logit: None, score_out: None,
            drop_tf_h1: None, drop_tf_h2: None,
            drop_gene_h1: None, drop_gene_h2: None,
            drop_score_h1: None, drop_score_h2: None,
            drop_rng: StdRng::seed_from_u64(seed + 9999),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    pub fn forward(
        &mut self,
        tf_indices: &[usize],
        gene_indices: &[usize],
        tf_expr: &Array2<f32>,
        gene_expr: &Array2<f32>,
    ) -> Array1<f32> {
        let batch_size = tf_indices.len();

        self.tf_indices_cache = tf_indices.to_vec();
        self.gene_indices_cache = gene_indices.to_vec();

        // Take RNG out of self to avoid borrow conflicts, put back at end
        let mut drop_rng = std::mem::replace(
            &mut self.drop_rng,
            rand::rngs::StdRng::seed_from_u64(0), // placeholder
        );

        // === Embedding lookup ===
        let mut tf_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        let mut gene_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        for (i, &idx) in tf_indices.iter().enumerate() {
            tf_embed_batch.row_mut(i).assign(&self.tf_embed.row(idx));
        }
        for (i, &idx) in gene_indices.iter().enumerate() {
            gene_embed_batch.row_mut(i).assign(&self.gene_embed.row(idx));
        }
        self.tf_embed_out = Some(tf_embed_batch.clone());
        self.gene_embed_out = Some(gene_embed_batch.clone());

        // === Concatenate embeddings + expression ===
        let tf_input = ndarray::concatenate![Axis(1), tf_embed_batch, tf_expr.clone()];
        let gene_input = ndarray::concatenate![Axis(1), gene_embed_batch, gene_expr.clone()];
        self.tf_concat = Some(tf_input.clone());
        self.gene_concat = Some(gene_input.clone());

        // === TF Encoder: FC→LN→ReLU→Drop → FC→LN→ReLU→Drop → FC ===
        let tf_h1_pre = self.tf_fc1.forward(&tf_input);
        let tf_h1_ln = self.tf_ln1.forward(&tf_h1_pre);
        let tf_h1_act = relu(&tf_h1_ln);
        let (tf_h1, tf_h1_mask) = apply_dropout(&tf_h1_act, self.dropout_rate, self.training, &mut drop_rng);
        self.tf_h1_pre = Some(tf_h1_pre);
        self.tf_h1_ln = Some(tf_h1_ln.clone());
        self.tf_h1 = Some(tf_h1_act);
        self.drop_tf_h1 = tf_h1_mask;

        let tf_h2_pre = self.tf_fc2.forward(&tf_h1);
        let tf_h2_ln = self.tf_ln2.forward(&tf_h2_pre);
        let tf_h2_act = relu(&tf_h2_ln);
        let (tf_h2, tf_h2_mask) = apply_dropout(&tf_h2_act, self.dropout_rate, self.training, &mut drop_rng);
        self.tf_h2_pre = Some(tf_h2_pre);
        self.tf_h2_ln = Some(tf_h2_ln.clone());
        self.tf_h2 = Some(tf_h2_act);
        self.drop_tf_h2 = tf_h2_mask;

        let tf_enc = self.tf_fc3.forward(&tf_h2);
        self.tf_enc = Some(tf_enc.clone());

        // === Gene Encoder ===
        let gene_h1_pre = self.gene_fc1.forward(&gene_input);
        let gene_h1_ln = self.gene_ln1.forward(&gene_h1_pre);
        let gene_h1_act = relu(&gene_h1_ln);
        let (gene_h1, gene_h1_mask) = apply_dropout(&gene_h1_act, self.dropout_rate, self.training, &mut drop_rng);
        self.gene_h1_pre = Some(gene_h1_pre);
        self.gene_h1_ln = Some(gene_h1_ln.clone());
        self.gene_h1 = Some(gene_h1_act);
        self.drop_gene_h1 = gene_h1_mask;

        let gene_h2_pre = self.gene_fc2.forward(&gene_h1);
        let gene_h2_ln = self.gene_ln2.forward(&gene_h2_pre);
        let gene_h2_act = relu(&gene_h2_ln);
        let (gene_h2, gene_h2_mask) = apply_dropout(&gene_h2_act, self.dropout_rate, self.training, &mut drop_rng);
        self.gene_h2_pre = Some(gene_h2_pre);
        self.gene_h2_ln = Some(gene_h2_ln.clone());
        self.gene_h2 = Some(gene_h2_act);
        self.drop_gene_h2 = gene_h2_mask;

        let gene_enc = self.gene_fc3.forward(&gene_h2);
        self.gene_enc = Some(gene_enc.clone());

        // === Cross-Modulation ===
        let joint = ndarray::concatenate![Axis(1), tf_enc, gene_enc];
        self.joint = Some(joint.clone());

        // TF gate: sigmoid(FC([tf_enc; gene_enc]))
        let tf_gate_pre = self.tf_gate_fc.forward(&joint);
        let tf_gate = tf_gate_pre.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        self.tf_gate_pre = Some(tf_gate_pre);
        self.tf_gate = Some(tf_gate.clone());

        // Gene gate
        let gene_gate_pre = self.gene_gate_fc.forward(&joint);
        let gene_gate = gene_gate_pre.mapv(|v| 1.0 / (1.0 + (-v).exp()));
        self.gene_gate_pre = Some(gene_gate_pre);
        self.gene_gate = Some(gene_gate.clone());

        // Cross transforms
        let tf_cross_pre = self.tf_cross_fc.forward(&gene_enc);
        let tf_cross = relu(&tf_cross_pre);
        self.tf_cross_pre = Some(tf_cross_pre);
        self.tf_cross = Some(tf_cross.clone());

        let gene_cross_pre = self.gene_cross_fc.forward(&tf_enc);
        let gene_cross = relu(&gene_cross_pre);
        self.gene_cross_pre = Some(gene_cross_pre);
        self.gene_cross = Some(gene_cross.clone());

        // Gated modulation with residual
        let tf_mod = &tf_enc + &(&tf_gate * &tf_cross);
        let gene_mod = &gene_enc + &(&gene_gate * &gene_cross);
        self.tf_mod = Some(tf_mod.clone());
        self.gene_mod = Some(gene_mod.clone());

        // === Interaction Features ===
        let tf_times_gene = &tf_mod * &gene_mod;
        let tf_minus_gene = &tf_mod - &gene_mod;
        self.tf_times_gene = Some(tf_times_gene.clone());
        self.tf_minus_gene = Some(tf_minus_gene.clone());

        let interaction = ndarray::concatenate![
            Axis(1), tf_mod, gene_mod, tf_times_gene, tf_minus_gene
        ];
        self.interaction = Some(interaction.clone());

        // === Scoring Head ===
        let score_h1_pre = self.score_fc1.forward(&interaction);
        let score_h1_ln = self.score_ln1.forward(&score_h1_pre);
        let score_h1_act = relu(&score_h1_ln);
        let (score_h1, score_h1_mask) = apply_dropout(&score_h1_act, self.dropout_rate, self.training, &mut drop_rng);
        self.score_h1_pre = Some(score_h1_pre);
        self.score_h1_ln = Some(score_h1_ln.clone());
        self.score_h1 = Some(score_h1_act);
        self.drop_score_h1 = score_h1_mask;

        let score_h2_pre = self.score_fc2.forward(&score_h1);
        let score_h2_act = relu(&score_h2_pre);
        let (score_h2, score_h2_mask) = apply_dropout(&score_h2_act, self.dropout_rate, self.training, &mut drop_rng);
        self.score_h2_pre = Some(score_h2_pre.clone());
        self.score_h2 = Some(score_h2_act);
        self.drop_score_h2 = score_h2_mask;

        let score_logit = self.score_fc3.forward(&score_h2);
        self.score_logit = Some(score_logit.clone());

        // Sigmoid output
        let scores = score_logit.column(0).mapv(|v| 1.0 / (1.0 + (-v).exp()));
        self.score_out = Some(scores.clone());

        // Restore RNG with advanced state
        self.drop_rng = drop_rng;

        scores
    }

    /// Backward pass.
    /// NOTE: grad_output should be d(loss)/d(logit), i.e. (pred - target)/n,
    /// which already accounts for the sigmoid derivative. Do NOT multiply by sigmoid' again.
    pub fn backward(&mut self, grad_output: &Array1<f32>) {
        let batch_size = grad_output.len();

        // grad_output = (p - t) / n is already d(loss)/d(logit)
        // No sigmoid derivative needed here
        let mut grad_logit = Array2::zeros((batch_size, 1));
        for i in 0..batch_size {
            grad_logit[[i, 0]] = grad_output[i];
        }

        // === Backward through scoring head ===
        let grad_score_h2 = self.score_fc3.backward(&grad_logit);
        // Apply dropout mask before relu_backward (reverse of forward: relu→dropout)
        let grad_score_h2_drop = if let Some(ref mask) = self.drop_score_h2 {
            &grad_score_h2 * mask
        } else { grad_score_h2 };
        let grad_score_h2_pre = relu_backward(self.score_h2.as_ref().unwrap(), &grad_score_h2_drop);

        let grad_score_h1 = self.score_fc2.backward(&grad_score_h2_pre);
        let grad_score_h1_drop = if let Some(ref mask) = self.drop_score_h1 {
            &grad_score_h1 * mask
        } else { grad_score_h1 };
        let grad_score_h1_ln = relu_backward(self.score_h1.as_ref().unwrap(), &grad_score_h1_drop);
        let grad_score_h1_pre = self.score_ln1.backward(&grad_score_h1_ln);
        let grad_interaction = self.score_fc1.backward(&grad_score_h1_pre);

        // === Backward through interaction features ===
        // interaction = [tf_mod; gene_mod; tf*gene; tf-gene] each enc_dim wide
        let d = self.enc_dim;
        let grad_tf_from_concat = grad_interaction.slice(ndarray::s![.., 0..d]).to_owned();
        let grad_gene_from_concat = grad_interaction.slice(ndarray::s![.., d..2*d]).to_owned();
        let grad_product = grad_interaction.slice(ndarray::s![.., 2*d..3*d]).to_owned();
        let grad_diff = grad_interaction.slice(ndarray::s![.., 3*d..4*d]).to_owned();

        let tf_mod = self.tf_mod.as_ref().unwrap();
        let gene_mod = self.gene_mod.as_ref().unwrap();

        // tf*gene product: d_tf = grad * gene, d_gene = grad * tf
        let grad_tf_from_product = &grad_product * gene_mod;
        let grad_gene_from_product = &grad_product * tf_mod;

        // tf-gene diff: d_tf = grad, d_gene = -grad
        let grad_tf_from_diff = &grad_diff;
        let grad_gene_from_diff = -&grad_diff;

        let mut d_tf_mod = grad_tf_from_concat + grad_tf_from_product + grad_tf_from_diff;
        let mut d_gene_mod = grad_gene_from_concat + grad_gene_from_product + grad_gene_from_diff;

        // === Backward through cross-modulation ===
        // tf_mod = tf_enc + tf_gate * tf_cross
        let tf_gate = self.tf_gate.as_ref().unwrap();
        let tf_cross = self.tf_cross.as_ref().unwrap();
        let gene_gate = self.gene_gate.as_ref().unwrap();
        let gene_cross = self.gene_cross.as_ref().unwrap();

        // d_tf_enc from skip connection
        let mut d_tf_enc = d_tf_mod.clone();
        // d(tf_gate * tf_cross)
        let d_tf_gate = &d_tf_mod * tf_cross;
        let d_tf_cross_val = &d_tf_mod * tf_gate;

        // d_gene_enc from skip connection
        let mut d_gene_enc = d_gene_mod.clone();
        let d_gene_gate = &d_gene_mod * gene_cross;
        let d_gene_cross_val = &d_gene_mod * gene_gate;

        // Backward through tf_cross ReLU and FC (gene_enc → tf_cross)
        let d_tf_cross_pre = relu_backward(self.tf_cross.as_ref().unwrap(), &d_tf_cross_val);
        let d_gene_enc_from_tf_cross = self.tf_cross_fc.backward(&d_tf_cross_pre);
        d_gene_enc = d_gene_enc + d_gene_enc_from_tf_cross;

        // Backward through gene_cross ReLU and FC (tf_enc → gene_cross)
        let d_gene_cross_pre = relu_backward(self.gene_cross.as_ref().unwrap(), &d_gene_cross_val);
        let d_tf_enc_from_gene_cross = self.gene_cross_fc.backward(&d_gene_cross_pre);
        d_tf_enc = d_tf_enc + d_tf_enc_from_gene_cross;

        // Backward through tf_gate sigmoid and FC
        let d_tf_gate_pre = &d_tf_gate * tf_gate * &(1.0 - tf_gate);
        let d_joint_from_tf_gate = self.tf_gate_fc.backward(&d_tf_gate_pre);

        // Backward through gene_gate sigmoid and FC
        let d_gene_gate_pre = &d_gene_gate * gene_gate * &(1.0 - gene_gate);
        let d_joint_from_gene_gate = self.gene_gate_fc.backward(&d_gene_gate_pre);

        // Combine joint gradients
        let d_joint = d_joint_from_tf_gate + d_joint_from_gene_gate;
        d_tf_enc = d_tf_enc + d_joint.slice(ndarray::s![.., 0..self.enc_dim]).to_owned();
        d_gene_enc = d_gene_enc + d_joint.slice(ndarray::s![.., self.enc_dim..]).to_owned();

        // === Backward through TF encoder ===
        let d_tf_h2 = self.tf_fc3.backward(&d_tf_enc);
        let d_tf_h2_d = if let Some(ref mask) = self.drop_tf_h2 { &d_tf_h2 * mask } else { d_tf_h2 };
        let d_tf_h2_ln = relu_backward(self.tf_h2.as_ref().unwrap(), &d_tf_h2_d);
        let d_tf_h2_pre = self.tf_ln2.backward(&d_tf_h2_ln);
        let d_tf_h1 = self.tf_fc2.backward(&d_tf_h2_pre);
        let d_tf_h1_d = if let Some(ref mask) = self.drop_tf_h1 { &d_tf_h1 * mask } else { d_tf_h1 };
        let d_tf_h1_ln = relu_backward(self.tf_h1.as_ref().unwrap(), &d_tf_h1_d);
        let d_tf_h1_pre = self.tf_ln1.backward(&d_tf_h1_ln);
        let d_tf_input = self.tf_fc1.backward(&d_tf_h1_pre);

        // === Backward through Gene encoder ===
        let d_gene_h2 = self.gene_fc3.backward(&d_gene_enc);
        let d_gene_h2_d = if let Some(ref mask) = self.drop_gene_h2 { &d_gene_h2 * mask } else { d_gene_h2 };
        let d_gene_h2_ln = relu_backward(self.gene_h2.as_ref().unwrap(), &d_gene_h2_d);
        let d_gene_h2_pre = self.gene_ln2.backward(&d_gene_h2_ln);
        let d_gene_h1 = self.gene_fc2.backward(&d_gene_h2_pre);
        let d_gene_h1_d = if let Some(ref mask) = self.drop_gene_h1 { &d_gene_h1 * mask } else { d_gene_h1 };
        let d_gene_h1_ln = relu_backward(self.gene_h1.as_ref().unwrap(), &d_gene_h1_d);
        let d_gene_h1_pre = self.gene_ln1.backward(&d_gene_h1_ln);
        let d_gene_input = self.gene_fc1.backward(&d_gene_h1_pre);

        // === Accumulate embedding gradients ===
        let d_tf_embed = d_tf_input.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();
        let d_gene_embed = d_gene_input.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();

        for (i, &idx) in self.tf_indices_cache.iter().enumerate() {
            let mut row = self.tf_embed_grad.row_mut(idx);
            row += &d_tf_embed.row(i);
        }
        for (i, &idx) in self.gene_indices_cache.iter().enumerate() {
            let mut row = self.gene_embed_grad.row_mut(idx);
            row += &d_gene_embed.row(i);
        }
    }

    /// Update all parameters using Adam optimizer
    pub fn update_adam(
        &mut self,
        adam_states: &mut ModelAdamStates,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        batch_size: f32,
    ) {
        adam_states.step += 1;
        let t = adam_states.step;

        // Helper macro to update a linear layer
        macro_rules! update_layer {
            ($layer:expr, $w_state:expr, $b_state:expr) => {
                adam_update_iter(
                    $layer.weights.iter_mut(),
                    $layer.grad_weights.iter(),
                    $w_state, lr, beta1, beta2, eps, t, weight_decay, batch_size,
                );
                adam_update_iter(
                    $layer.bias.iter_mut(),
                    $layer.grad_bias.iter(),
                    $b_state, lr, beta1, beta2, eps, t, 0.0, batch_size,
                );
            };
        }

        macro_rules! update_ln {
            ($ln:expr, $g_state:expr, $b_state:expr) => {
                adam_update_iter(
                    $ln.gamma.iter_mut(),
                    $ln.grad_gamma.iter(),
                    $g_state, lr, beta1, beta2, eps, t, 0.0, batch_size,
                );
                adam_update_iter(
                    $ln.beta.iter_mut(),
                    $ln.grad_beta.iter(),
                    $b_state, lr, beta1, beta2, eps, t, 0.0, batch_size,
                );
            };
        }

        // Embeddings
        adam_update_iter(
            self.tf_embed.iter_mut(),
            self.tf_embed_grad.iter(),
            &mut adam_states.tf_embed, lr, beta1, beta2, eps, t, weight_decay, batch_size,
        );
        adam_update_iter(
            self.gene_embed.iter_mut(),
            self.gene_embed_grad.iter(),
            &mut adam_states.gene_embed, lr, beta1, beta2, eps, t, weight_decay, batch_size,
        );

        // TF encoder
        update_layer!(self.tf_fc1, &mut adam_states.tf_fc1_w, &mut adam_states.tf_fc1_b);
        update_ln!(self.tf_ln1, &mut adam_states.tf_ln1_g, &mut adam_states.tf_ln1_b);
        update_layer!(self.tf_fc2, &mut adam_states.tf_fc2_w, &mut adam_states.tf_fc2_b);
        update_ln!(self.tf_ln2, &mut adam_states.tf_ln2_g, &mut adam_states.tf_ln2_b);
        update_layer!(self.tf_fc3, &mut adam_states.tf_fc3_w, &mut adam_states.tf_fc3_b);

        // Gene encoder
        update_layer!(self.gene_fc1, &mut adam_states.gene_fc1_w, &mut adam_states.gene_fc1_b);
        update_ln!(self.gene_ln1, &mut adam_states.gene_ln1_g, &mut adam_states.gene_ln1_b);
        update_layer!(self.gene_fc2, &mut adam_states.gene_fc2_w, &mut adam_states.gene_fc2_b);
        update_ln!(self.gene_ln2, &mut adam_states.gene_ln2_g, &mut adam_states.gene_ln2_b);
        update_layer!(self.gene_fc3, &mut adam_states.gene_fc3_w, &mut adam_states.gene_fc3_b);

        // Cross-modulation
        update_layer!(self.tf_gate_fc, &mut adam_states.tf_gate_w, &mut adam_states.tf_gate_b);
        update_layer!(self.gene_gate_fc, &mut adam_states.gene_gate_w, &mut adam_states.gene_gate_b);
        update_layer!(self.tf_cross_fc, &mut adam_states.tf_cross_w, &mut adam_states.tf_cross_b);
        update_layer!(self.gene_cross_fc, &mut adam_states.gene_cross_w, &mut adam_states.gene_cross_b);

        // Scoring head
        update_layer!(self.score_fc1, &mut adam_states.score_fc1_w, &mut adam_states.score_fc1_b);
        update_ln!(self.score_ln1, &mut adam_states.score_ln1_g, &mut adam_states.score_ln1_b);
        update_layer!(self.score_fc2, &mut adam_states.score_fc2_w, &mut adam_states.score_fc2_b);
        update_layer!(self.score_fc3, &mut adam_states.score_fc3_w, &mut adam_states.score_fc3_b);
    }

    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.tf_fc1.zero_grad(); self.tf_ln1.zero_grad();
        self.tf_fc2.zero_grad(); self.tf_ln2.zero_grad();
        self.tf_fc3.zero_grad();
        self.gene_fc1.zero_grad(); self.gene_ln1.zero_grad();
        self.gene_fc2.zero_grad(); self.gene_ln2.zero_grad();
        self.gene_fc3.zero_grad();
        self.tf_gate_fc.zero_grad();
        self.gene_gate_fc.zero_grad();
        self.tf_cross_fc.zero_grad();
        self.gene_cross_fc.zero_grad();
        self.score_fc1.zero_grad(); self.score_ln1.zero_grad();
        self.score_fc2.zero_grad();
        self.score_fc3.zero_grad();
    }

    pub fn num_parameters(&self) -> usize {
        let embed = self.tf_embed.len() + self.gene_embed.len();
        let fc = |l: &LinearLayer| l.weights.len() + l.bias.len();
        let ln = |l: &LayerNorm| l.gamma.len() + l.beta.len();

        embed
            + fc(&self.tf_fc1) + ln(&self.tf_ln1) + fc(&self.tf_fc2) + ln(&self.tf_ln2) + fc(&self.tf_fc3)
            + fc(&self.gene_fc1) + ln(&self.gene_ln1) + fc(&self.gene_fc2) + ln(&self.gene_ln2) + fc(&self.gene_fc3)
            + fc(&self.tf_gate_fc) + fc(&self.gene_gate_fc)
            + fc(&self.tf_cross_fc) + fc(&self.gene_cross_fc)
            + fc(&self.score_fc1) + ln(&self.score_ln1) + fc(&self.score_fc2) + fc(&self.score_fc3)
    }
}

/// Adam optimizer states for all model parameters
pub struct ModelAdamStates {
    pub step: usize,
    // Embeddings
    pub tf_embed: AdamState,
    pub gene_embed: AdamState,
    // TF encoder
    pub tf_fc1_w: AdamState, pub tf_fc1_b: AdamState,
    pub tf_ln1_g: AdamState, pub tf_ln1_b: AdamState,
    pub tf_fc2_w: AdamState, pub tf_fc2_b: AdamState,
    pub tf_ln2_g: AdamState, pub tf_ln2_b: AdamState,
    pub tf_fc3_w: AdamState, pub tf_fc3_b: AdamState,
    // Gene encoder
    pub gene_fc1_w: AdamState, pub gene_fc1_b: AdamState,
    pub gene_ln1_g: AdamState, pub gene_ln1_b: AdamState,
    pub gene_fc2_w: AdamState, pub gene_fc2_b: AdamState,
    pub gene_ln2_g: AdamState, pub gene_ln2_b: AdamState,
    pub gene_fc3_w: AdamState, pub gene_fc3_b: AdamState,
    // Cross-modulation
    pub tf_gate_w: AdamState, pub tf_gate_b: AdamState,
    pub gene_gate_w: AdamState, pub gene_gate_b: AdamState,
    pub tf_cross_w: AdamState, pub tf_cross_b: AdamState,
    pub gene_cross_w: AdamState, pub gene_cross_b: AdamState,
    // Scoring
    pub score_fc1_w: AdamState, pub score_fc1_b: AdamState,
    pub score_ln1_g: AdamState, pub score_ln1_b: AdamState,
    pub score_fc2_w: AdamState, pub score_fc2_b: AdamState,
    pub score_fc3_w: AdamState, pub score_fc3_b: AdamState,
}

impl ModelAdamStates {
    pub fn new(model: &CrossAttentionModel) -> Self {
        let fc_w = |l: &LinearLayer| AdamState::new(l.weights.len());
        let fc_b = |l: &LinearLayer| AdamState::new(l.bias.len());
        let ln_g = |l: &LayerNorm| AdamState::new(l.gamma.len());
        let ln_b = |l: &LayerNorm| AdamState::new(l.beta.len());

        Self {
            step: 0,
            tf_embed: AdamState::new(model.tf_embed.len()),
            gene_embed: AdamState::new(model.gene_embed.len()),
            tf_fc1_w: fc_w(&model.tf_fc1), tf_fc1_b: fc_b(&model.tf_fc1),
            tf_ln1_g: ln_g(&model.tf_ln1), tf_ln1_b: ln_b(&model.tf_ln1),
            tf_fc2_w: fc_w(&model.tf_fc2), tf_fc2_b: fc_b(&model.tf_fc2),
            tf_ln2_g: ln_g(&model.tf_ln2), tf_ln2_b: ln_b(&model.tf_ln2),
            tf_fc3_w: fc_w(&model.tf_fc3), tf_fc3_b: fc_b(&model.tf_fc3),
            gene_fc1_w: fc_w(&model.gene_fc1), gene_fc1_b: fc_b(&model.gene_fc1),
            gene_ln1_g: ln_g(&model.gene_ln1), gene_ln1_b: ln_b(&model.gene_ln1),
            gene_fc2_w: fc_w(&model.gene_fc2), gene_fc2_b: fc_b(&model.gene_fc2),
            gene_ln2_g: ln_g(&model.gene_ln2), gene_ln2_b: ln_b(&model.gene_ln2),
            gene_fc3_w: fc_w(&model.gene_fc3), gene_fc3_b: fc_b(&model.gene_fc3),
            tf_gate_w: fc_w(&model.tf_gate_fc), tf_gate_b: fc_b(&model.tf_gate_fc),
            gene_gate_w: fc_w(&model.gene_gate_fc), gene_gate_b: fc_b(&model.gene_gate_fc),
            tf_cross_w: fc_w(&model.tf_cross_fc), tf_cross_b: fc_b(&model.tf_cross_fc),
            gene_cross_w: fc_w(&model.gene_cross_fc), gene_cross_b: fc_b(&model.gene_cross_fc),
            score_fc1_w: fc_w(&model.score_fc1), score_fc1_b: fc_b(&model.score_fc1),
            score_ln1_g: ln_g(&model.score_ln1), score_ln1_b: ln_b(&model.score_ln1),
            score_fc2_w: fc_w(&model.score_fc2), score_fc2_b: fc_b(&model.score_fc2),
            score_fc3_w: fc_w(&model.score_fc3), score_fc3_b: fc_b(&model.score_fc3),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_attention_forward() {
        let mut model = CrossAttentionModel::new(100, 500, 32, 11, 64, 32, 0.0, 42);

        let tf_indices = vec![0, 1, 2, 3];
        let gene_indices = vec![10, 20, 30, 40];
        let tf_expr = Array2::from_elem((4, 11), 0.1);
        let gene_expr = Array2::from_elem((4, 11), 0.1);

        let scores = model.forward(&tf_indices, &gene_indices, &tf_expr, &gene_expr);

        assert_eq!(scores.len(), 4);
        assert!(scores.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_cross_attention_backward() {
        let mut model = CrossAttentionModel::new(100, 500, 32, 11, 64, 32, 0.0, 42);

        let tf_indices = vec![0, 1];
        let gene_indices = vec![10, 20];
        let tf_expr = Array2::from_elem((2, 11), 0.1);
        let gene_expr = Array2::from_elem((2, 11), 0.1);

        let scores = model.forward(&tf_indices, &gene_indices, &tf_expr, &gene_expr);
        let grad = Array1::from(vec![1.0, -1.0]);
        model.backward(&grad);

        // Check gradients are non-zero
        assert!(model.tf_embed_grad.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_layer_norm() {
        let mut ln = LayerNorm::new(4);
        let x = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = ln.forward(&x);
        assert_eq!(y.shape(), &[2, 4]);

        // Check normalization: each row should have mean ≈ 0
        for i in 0..2 {
            let row_mean: f32 = y.row(i).mean().unwrap();
            assert!(row_mean.abs() < 0.01, "row {} mean = {}", i, row_mean);
        }
    }

    #[test]
    fn test_adam_update() {
        let mut param = vec![1.0, 2.0, 3.0];
        let grad = vec![0.1, 0.2, 0.3];
        let mut state = AdamState::new(3);

        adam_update_iter(
            param.iter_mut(), grad.iter(), &mut state,
            0.001, 0.9, 0.999, 1e-8, 1, 0.0, 1.0,
        );

        // Parameters should have changed
        assert!(param[0] != 1.0);
        assert!(param[1] != 2.0);
    }
}
