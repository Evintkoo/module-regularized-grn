/// B5: Two-Tower with MLP Scoring Head
/// Two-tower encoders (embed+expr → FC1(ReLU) → FC2) feeding a 2-layer MLP scorer
/// instead of cosine similarity. Inline model implementation (no separate model file).
///
/// Architecture:
///   TF  tower: [embed(512) | expr(D)] → FC(512+D→512, ReLU) → FC(512→512) = tf_enc
///   Gene tower: same structure = gene_enc
///   Head:       [tf_enc | gene_enc] → FC(1024→256, ReLU) → FC(256→1) → sigmoid
///
/// 5 seeds [42, 123, 456, 789, 1337], lr=0.001, batch=256, 60 epochs, 1:1 ratio.
/// Results → results/tt_mlp_head_results.json
use module_regularized_grn::{
    Config,
    models::nn::{LinearLayer, relu, relu_backward},
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

// ── Inline model ──────────────────────────────────────────────────────────────

struct TwoTowerMlpHead {
    // Embeddings
    pub tf_embed:    Array2<f32>,   // [num_tfs, embed_dim]
    pub tf_embed_grad:  Array2<f32>,
    pub gene_embed:  Array2<f32>,   // [num_genes, embed_dim]
    pub gene_embed_grad: Array2<f32>,

    // TF tower
    pub tf_fc1: LinearLayer,   // (embed_dim+expr_dim) → 512, ReLU
    pub tf_fc2: LinearLayer,   // 512 → 512, linear

    // Gene tower
    pub gene_fc1: LinearLayer, // (embed_dim+expr_dim) → 512, ReLU
    pub gene_fc2: LinearLayer, // 512 → 512, linear

    // Scoring head
    pub head_fc1: LinearLayer, // 1024 → 256, ReLU
    pub head_fc2: LinearLayer, // 256 → 1, linear → sigmoid

    embed_dim: usize,

    // Caches for backward pass
    tf_idx_cache:    Option<Vec<usize>>,
    gene_idx_cache:  Option<Vec<usize>>,
    tf_fc1_pre:      Option<Array2<f32>>,  // pre-ReLU output of tf_fc1
    gene_fc1_pre:    Option<Array2<f32>>,  // pre-ReLU output of gene_fc1
    head_fc1_pre:    Option<Array2<f32>>,  // pre-ReLU output of head_fc1
    pred_cache:      Option<Array2<f32>>,  // sigmoid output [batch, 1]
}

impl TwoTowerMlpHead {
    fn new(
        num_tfs:   usize,
        num_genes: usize,
        embed_dim: usize,
        expr_dim:  usize,
        seed:      u64,
    ) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;

        let mut rng  = StdRng::seed_from_u64(seed);
        let emb_dist = Normal::new(0.0f32, 0.01).unwrap();

        let tf_embed   = Array2::random_using((num_tfs,   embed_dim), emb_dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, embed_dim), emb_dist, &mut rng);

        let tower_in = embed_dim + expr_dim;
        Self {
            tf_embed_grad:   Array2::zeros((num_tfs,   embed_dim)),
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            tf_fc1:   LinearLayer::new(tower_in, 512, seed),
            tf_fc2:   LinearLayer::new(512,      512, seed + 1),
            gene_fc1: LinearLayer::new(tower_in, 512, seed + 2),
            gene_fc2: LinearLayer::new(512,      512, seed + 3),
            head_fc1: LinearLayer::new(1024,     256, seed + 4),
            head_fc2: LinearLayer::new(256,      1,   seed + 5),
            embed_dim,
            tf_embed, gene_embed,
            tf_idx_cache: None, gene_idx_cache: None,
            tf_fc1_pre: None, gene_fc1_pre: None, head_fc1_pre: None,
            pred_cache: None,
        }
    }

    /// Forward pass. Returns sigmoid probabilities [batch, 1].
    fn forward(
        &mut self,
        tf_indices:   &[usize],
        gene_indices: &[usize],
        tf_expr:      &Array2<f32>,
        gene_expr:    &Array2<f32>,
    ) -> Array2<f32> {
        let batch = tf_indices.len();

        // Embedding lookups
        let mut tf_emb_out   = Array2::zeros((batch, self.embed_dim));
        let mut gene_emb_out = Array2::zeros((batch, self.embed_dim));
        for (i, &idx) in tf_indices.iter().enumerate() {
            tf_emb_out.row_mut(i).assign(&self.tf_embed.row(idx));
        }
        for (i, &idx) in gene_indices.iter().enumerate() {
            gene_emb_out.row_mut(i).assign(&self.gene_embed.row(idx));
        }

        // TF tower: concat → FC1(ReLU) → FC2
        let tf_input    = ndarray::concatenate(Axis(1), &[tf_emb_out.view(), tf_expr.view()]).unwrap();
        let tf_fc1_pre  = self.tf_fc1.forward(&tf_input);
        let tf_h1       = relu(&tf_fc1_pre);
        let tf_enc      = self.tf_fc2.forward(&tf_h1);

        // Gene tower: same
        let gene_input   = ndarray::concatenate(Axis(1), &[gene_emb_out.view(), gene_expr.view()]).unwrap();
        let gene_fc1_pre = self.gene_fc1.forward(&gene_input);
        let gene_h1      = relu(&gene_fc1_pre);
        let gene_enc     = self.gene_fc2.forward(&gene_h1);

        // Head: concat → FC1(ReLU) → FC2 → sigmoid
        let combined  = ndarray::concatenate(Axis(1), &[tf_enc.view(), gene_enc.view()]).unwrap();
        let h1_pre    = self.head_fc1.forward(&combined);
        let h1_out    = relu(&h1_pre);
        let logit     = self.head_fc2.forward(&h1_out);
        let pred      = logit.mapv(|x| 1.0 / (1.0 + (-x).exp()));

        // Cache for backward
        self.tf_idx_cache   = Some(tf_indices.to_vec());
        self.gene_idx_cache = Some(gene_indices.to_vec());
        self.tf_fc1_pre     = Some(tf_fc1_pre);
        self.gene_fc1_pre   = Some(gene_fc1_pre);
        self.head_fc1_pre   = Some(h1_pre);
        self.pred_cache     = Some(pred.clone());

        pred
    }

    /// Backward pass given labels (Vec<f32>). Gradient = (pred - label) / bsz (BCE+sigmoid trick).
    fn backward(&mut self, labels: &[f32]) {
        let pred      = self.pred_cache.as_ref().unwrap();
        let bsz       = labels.len() as f32;
        let tf_idx    = self.tf_idx_cache.as_ref().unwrap().clone();
        let gene_idx  = self.gene_idx_cache.as_ref().unwrap().clone();

        // d_logit = (sigmoid(logit) - label) / bsz = (pred - label) / bsz  [batch, 1]
        let mut d_logit = Array2::zeros(pred.dim());
        for (i, &l) in labels.iter().enumerate() {
            d_logit[[i, 0]] = (pred[[i, 0]] - l) / bsz;
        }

        // Head FC2 backward → d_h1_out [batch, 256]
        let d_h1_out = self.head_fc2.backward(&d_logit);

        // ReLU backward for head_fc1
        let d_h1_pre = relu_backward(self.head_fc1_pre.as_ref().unwrap(), &d_h1_out);

        // Head FC1 backward → d_combined [batch, 1024]
        let d_combined = self.head_fc1.backward(&d_h1_pre);

        // Split: d_tf_enc = d_combined[:, :512], d_gene_enc = d_combined[:, 512:]
        let d_tf_enc   = d_combined.slice(ndarray::s![.., ..512]).to_owned();
        let d_gene_enc = d_combined.slice(ndarray::s![.., 512..]).to_owned();

        // TF tower backward
        let d_tf_h1       = self.tf_fc2.backward(&d_tf_enc);
        let d_tf_fc1_pre  = relu_backward(self.tf_fc1_pre.as_ref().unwrap(), &d_tf_h1);
        let d_tf_input    = self.tf_fc1.backward(&d_tf_fc1_pre);
        let d_tf_emb      = d_tf_input.slice(ndarray::s![.., ..self.embed_dim]).to_owned();
        for (i, &idx) in tf_idx.iter().enumerate() {
            let mut row = self.tf_embed_grad.row_mut(idx);
            row += &d_tf_emb.row(i);
        }

        // Gene tower backward
        let d_gene_h1      = self.gene_fc2.backward(&d_gene_enc);
        let d_gene_fc1_pre = relu_backward(self.gene_fc1_pre.as_ref().unwrap(), &d_gene_h1);
        let d_gene_input   = self.gene_fc1.backward(&d_gene_fc1_pre);
        let d_gene_emb     = d_gene_input.slice(ndarray::s![.., ..self.embed_dim]).to_owned();
        for (i, &idx) in gene_idx.iter().enumerate() {
            let mut row = self.gene_embed_grad.row_mut(idx);
            row += &d_gene_emb.row(i);
        }
    }

    fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.tf_fc1.zero_grad();   self.tf_fc2.zero_grad();
        self.gene_fc1.zero_grad(); self.gene_fc2.zero_grad();
        self.head_fc1.zero_grad(); self.head_fc2.zero_grad();
    }

    fn num_parameters(&self) -> usize {
        self.tf_embed.len() + self.gene_embed.len()
            + self.tf_fc1.weights.len()   + self.tf_fc1.bias.len()
            + self.tf_fc2.weights.len()   + self.tf_fc2.bias.len()
            + self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
            + self.gene_fc2.weights.len() + self.gene_fc2.bias.len()
            + self.head_fc1.weights.len() + self.head_fc1.bias.len()
            + self.head_fc2.weights.len() + self.head_fc2.bias.len()
    }
}

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
    m_h1_w: Array2<f32>, v_h1_w: Array2<f32>,
    m_h1_b: Array1<f32>, v_h1_b: Array1<f32>,
    m_h2_w: Array2<f32>, v_h2_w: Array2<f32>,
    m_h2_b: Array1<f32>, v_h2_b: Array1<f32>,
    t: i32,
}

impl AdamState {
    fn new(model: &TwoTowerMlpHead) -> Self {
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
            m_h1_w:     Array2::zeros(model.head_fc1.weights.dim()),
            v_h1_w:     Array2::zeros(model.head_fc1.weights.dim()),
            m_h1_b:     Array1::zeros(model.head_fc1.bias.len()),
            v_h1_b:     Array1::zeros(model.head_fc1.bias.len()),
            m_h2_w:     Array2::zeros(model.head_fc2.weights.dim()),
            v_h2_w:     Array2::zeros(model.head_fc2.weights.dim()),
            m_h2_b:     Array1::zeros(model.head_fc2.bias.len()),
            v_h2_b:     Array1::zeros(model.head_fc2.bias.len()),
            t: 0,
        }
    }
}

fn adam_step(model: &mut TwoTowerMlpHead, state: &mut AdamState, lr: f32, clip: f32) {
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
        model.head_fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.head_fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.head_fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.head_fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
    ].iter().sum();
    let scale = if norm_sq.sqrt() > clip { clip / norm_sq.sqrt() } else { 1.0 };

    fn adam2d(p: &mut Array2<f32>, g: &Array2<f32>, m: &mut Array2<f32>, v: &mut Array2<f32>,
              lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32) {
        let bc1 = 1.0 - b1.powi(t); let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(g.view()).for_each(|m, &g| *m = b1**m + (1.0-b1)*g*scale);
        ndarray::Zip::from(v.view_mut()).and(g.view()).for_each(|v, &g| *v = b2**v + (1.0-b2)*(g*scale).powi(2));
        ndarray::Zip::from(p.view_mut()).and(m.view()).and(v.view()).for_each(|p,&m,&v| *p -= lr*(m/bc1)/((v/bc2).sqrt()+eps));
    }
    fn adam1d(p: &mut Array1<f32>, g: &Array1<f32>, m: &mut Array1<f32>, v: &mut Array1<f32>,
              lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32) {
        let bc1 = 1.0 - b1.powi(t); let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(g.view()).for_each(|m, &g| *m = b1**m + (1.0-b1)*g*scale);
        ndarray::Zip::from(v.view_mut()).and(g.view()).for_each(|v, &g| *v = b2**v + (1.0-b2)*(g*scale).powi(2));
        ndarray::Zip::from(p.view_mut()).and(m.view()).and(v.view()).for_each(|p,&m,&v| *p -= lr*(m/bc1)/((v/bc2).sqrt()+eps));
    }

    adam2d(&mut model.tf_embed,           &model.tf_embed_grad,           &mut state.m_tf,       &mut state.v_tf,       lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,         &model.gene_embed_grad,         &mut state.m_gene,     &mut state.v_gene,     lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.tf_fc1.weights,     &model.tf_fc1.grad_weights,     &mut state.m_tf_fc1_w, &mut state.v_tf_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.tf_fc1.bias,        &model.tf_fc1.grad_bias,        &mut state.m_tf_fc1_b, &mut state.v_tf_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.tf_fc2.weights,     &model.tf_fc2.grad_weights,     &mut state.m_tf_fc2_w, &mut state.v_tf_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.tf_fc2.bias,        &model.tf_fc2.grad_bias,        &mut state.m_tf_fc2_b, &mut state.v_tf_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_fc1.weights,   &model.gene_fc1.grad_weights,   &mut state.m_g_fc1_w,  &mut state.v_g_fc1_w,  lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.gene_fc1.bias,      &model.gene_fc1.grad_bias,      &mut state.m_g_fc1_b,  &mut state.v_g_fc1_b,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_fc2.weights,   &model.gene_fc2.grad_weights,   &mut state.m_g_fc2_w,  &mut state.v_g_fc2_w,  lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.gene_fc2.bias,      &model.gene_fc2.grad_bias,      &mut state.m_g_fc2_b,  &mut state.v_g_fc2_b,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.head_fc1.weights,   &model.head_fc1.grad_weights,   &mut state.m_h1_w,     &mut state.v_h1_w,     lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.head_fc1.bias,      &model.head_fc1.grad_bias,      &mut state.m_h1_b,     &mut state.v_h1_b,     lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.head_fc2.weights,   &model.head_fc2.grad_weights,   &mut state.m_h2_w,     &mut state.v_h2_w,     lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.head_fc2.bias,      &model.head_fc2.grad_bias,      &mut state.m_h2_b,     &mut state.v_h2_b,     lr, b1, b2, eps, state.t, scale);
}

// ── Metric helpers ────────────────────────────────────────────────────────────

fn build_expr_batch(indices: &[usize], expr_map: &HashMap<usize, Array1<f32>>, expr_dim: usize) -> Array2<f32> {
    let mut batch = Array2::zeros((indices.len(), expr_dim));
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(expr) = expr_map.get(&idx) { batch.row_mut(i).assign(expr); }
    }
    batch
}

fn calculate_auroc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels.iter()).map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    let n_neg = labels.len() as f32 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    let mut tp = 0.0f32; let mut fp = 0.0f32; let mut auc = 0.0f32; let mut prev_fp = 0.0f32;
    for (_, label) in &pairs {
        if *label == 1.0 { tp += 1.0; } else {
            fp += 1.0; auc += (tp / n_pos) * ((fp - prev_fp) / n_neg); prev_fp = fp;
        }
    }
    auc
}

fn calculate_f1(predictions: &[f32], labels: &[f32]) -> f32 {
    let (mut tp, mut fp, mut fn_) = (0.0f32, 0.0f32, 0.0f32);
    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        match ((p >= 0.5) as i32, l as i32) {
            (1, 1) => tp += 1.0, (1, 0) => fp += 1.0, (0, 1) => fn_ += 1.0, _ => {}
        }
    }
    let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let rec  = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 }
}

fn mean(v: &[f32]) -> f32 { v.iter().sum::<f32>() / v.len() as f32 }
fn std_dev(v: &[f32]) -> f32 {
    let m = mean(v); (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== B5: Two-Tower with MLP Scoring Head ===");

    let config    = Config::load_default()?;
    let base_seed = config.project.seed;
    let mut rng   = StdRng::seed_from_u64(base_seed);

    println!("Loading data...");
    let priors  = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs   = builder.num_tfs();
    let num_genes = builder.num_genes();
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    let expr_dim = expr_data.n_cell_types;

    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in expr_data.gene_names.iter().enumerate() { gene_to_expr_idx.insert(name.clone(), i); }
    let mut tf_expr_map:   HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    for (name, &idx) in builder.get_tf_vocab().iter() {
        tf_expr_map.insert(idx, gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned()).unwrap_or_else(|| Array1::zeros(expr_dim)));
    }
    for (name, &idx) in builder.get_gene_vocab().iter() {
        gene_expr_map.insert(idx, gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned()).unwrap_or_else(|| Array1::zeros(expr_dim)));
    }

    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, g)| (tf, g, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, g)| (tf, g, 0.0)));
    examples.shuffle(&mut rng);

    let n_total = examples.len();
    let n_train = (n_total as f32 * 0.7) as usize;
    let n_val   = (n_total as f32 * 0.15) as usize;
    let train_data = examples[..n_train].to_vec();
    let val_data   = examples[n_train..n_train+n_val].to_vec();
    let test_data  = examples[n_train+n_val..].to_vec();
    println!("  TFs: {} | Genes: {} | expr_dim: {} | Train: {} | Val: {} | Test: {}",
        num_tfs, num_genes, expr_dim, train_data.len(), val_data.len(), test_data.len());

    let embed_dim  = 512usize;
    let lr         = 0.001f32;
    let clip       = 5.0f32;
    let batch_size = 256usize;
    let epochs     = 60usize;
    let seeds      = [42u64, 123, 456, 789, 1337];

    // Print parameter count from first model
    {
        let probe = TwoTowerMlpHead::new(num_tfs, num_genes, embed_dim, expr_dim, 42);
        println!("  Model parameters: {}", probe.num_parameters());
    }

    let test_labels: Vec<f32> = test_data.iter().map(|x| x.2).collect();
    let mut seed_aurocs:  Vec<f32> = Vec::new();
    let mut seed_accs:    Vec<f32> = Vec::new();
    let mut seed_f1s:     Vec<f32> = Vec::new();
    let mut all_test_preds: Vec<Vec<f32>> = Vec::new();

    for &seed in &seeds {
        println!("\n  Seed {}:", seed);
        let mut model = TwoTowerMlpHead::new(num_tfs, num_genes, embed_dim, expr_dim, seed);
        let mut state = AdamState::new(&model);
        let mut best_val_auroc = 0.0f32;
        let mut patience = 0usize;

        for epoch in 0..epochs {
            let mut epoch_data = train_data.clone();
            let mut epoch_rng  = StdRng::seed_from_u64(seed + epoch as u64);
            epoch_data.shuffle(&mut epoch_rng);

            for start in (0..epoch_data.len()).step_by(batch_size) {
                let end = (start + batch_size).min(epoch_data.len());
                let batch = &epoch_data[start..end];
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                model.backward(&labels);
                adam_step(&mut model, &mut state, lr, clip);
                model.zero_grad();
            }

            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                let mut val_preds: Vec<f32> = Vec::new();
                let mut val_labels: Vec<f32> = Vec::new();
                for start in (0..val_data.len()).step_by(batch_size) {
                    let end = (start + batch_size).min(val_data.len());
                    let batch = &val_data[start..end];
                    let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                    let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                    let lbls:     Vec<f32>   = batch.iter().map(|x| x.2).collect();
                    let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                    let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                    let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                    val_preds.extend(preds.iter().copied());
                    val_labels.extend_from_slice(&lbls);
                }
                let val_auroc = calculate_auroc(&val_preds, &val_labels);
                if val_auroc > best_val_auroc { best_val_auroc = val_auroc; patience = 0; }
                else { patience += 1; if patience >= 3 { break; } }
            }
        }

        // Test evaluation
        let mut test_preds: Vec<f32> = Vec::new();
        for start in (0..test_data.len()).step_by(batch_size) {
            let end = (start + batch_size).min(test_data.len());
            let batch = &test_data[start..end];
            let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
            let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
            let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
            test_preds.extend(preds.iter().copied());
        }
        let auroc = calculate_auroc(&test_preds, &test_labels);
        let f1    = calculate_f1(&test_preds, &test_labels);
        let acc   = test_preds.iter().zip(test_labels.iter())
            .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / test_data.len() as f32;
        println!("    acc={:.2}% auroc={:.4} f1={:.4}", acc*100.0, auroc, f1);
        seed_aurocs.push(auroc); seed_accs.push(acc); seed_f1s.push(f1);
        all_test_preds.push(test_preds);
    }

    // Ensemble
    let n_test = test_data.len();
    let ensemble_preds: Vec<f32> = (0..n_test)
        .map(|i| all_test_preds.iter().map(|p| p[i]).sum::<f32>() / seeds.len() as f32)
        .collect();
    let ensemble_auroc = calculate_auroc(&ensemble_preds, &test_labels);
    let ensemble_acc = ensemble_preds.iter().zip(test_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / n_test as f32;

    println!("\n=== Summary ===");
    println!("  Mean AUROC: {:.4} ± {:.4}", mean(&seed_aurocs), std_dev(&seed_aurocs));
    println!("  Ensemble AUROC: {:.4}", ensemble_auroc);
    println!("  Ensemble acc: {:.2}%", ensemble_acc * 100.0);

    std::fs::create_dir_all("results")?;
    let output = serde_json::json!({
        "model":        "two_tower_mlp_head",
        "seeds":        seeds,
        "seed_aurocs":  seed_aurocs,
        "mean_auroc":   mean(&seed_aurocs),
        "std_auroc":    std_dev(&seed_aurocs),
        "mean_accuracy": mean(&seed_accs),
        "mean_f1":      mean(&seed_f1s),
        "ensemble_auroc": ensemble_auroc,
        "note": "Two-tower with MLP scoring head instead of cosine similarity. Parameter-matched to cross-encoder.",
    });
    std::fs::write("results/tt_mlp_head_results.json", serde_json::to_string_pretty(&output)?)?;
    println!("✓ Saved results/tt_mlp_head_results.json");
    Ok(())
}
