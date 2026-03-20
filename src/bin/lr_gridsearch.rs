/// B2: Per-architecture LR grid search
/// Trains two-tower (HybridEmbeddingModel) and cross-encoder (CrossEncoderModel)
/// at LRs ∈ {0.0005, 0.001, 0.005} × 3 seeds [42, 123, 456], 60 epochs each.
/// Reports mean ± std AUROC across seeds per (model, lr) cell.
/// Results → results/lr_gridsearch.json
use module_regularized_grn::{
    Config,
    models::hybrid_embeddings::HybridEmbeddingModel,
    models::cross_encoder::CrossEncoderModel,
    models::nn::bce_loss_backward,
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

// ── Adam state for TwoTower ───────────────────────────────────────────────────

struct TtAdamState {
    m_tf:       Array2<f32>, v_tf:       Array2<f32>,
    m_gene:     Array2<f32>, v_gene:     Array2<f32>,
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

impl TtAdamState {
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

fn adam_step_tt(model: &mut HybridEmbeddingModel, state: &mut TtAdamState, lr: f32, clip: f32) {
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

// ── Adam state for CrossEncoder ───────────────────────────────────────────────

struct CeAdamState {
    m_tf:    Array2<f32>, v_tf:    Array2<f32>,
    m_gene:  Array2<f32>, v_gene:  Array2<f32>,
    m_fc1_w: Array2<f32>, v_fc1_w: Array2<f32>,
    m_fc1_b: Array1<f32>, v_fc1_b: Array1<f32>,
    m_fc2_w: Array2<f32>, v_fc2_w: Array2<f32>,
    m_fc2_b: Array1<f32>, v_fc2_b: Array1<f32>,
    m_fc3_w: Array2<f32>, v_fc3_w: Array2<f32>,
    m_fc3_b: Array1<f32>, v_fc3_b: Array1<f32>,
    t: i32,
}

impl CeAdamState {
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

fn adam_step_ce(model: &mut CrossEncoderModel, state: &mut CeAdamState, lr: f32, clip: f32) {
    state.t += 1;
    let b1 = 0.9f32; let b2 = 0.999f32; let eps = 1e-8f32;
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

    adam2d(&mut model.tf_embed,      &model.tf_embed_grad,    &mut state.m_tf,    &mut state.v_tf,    lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,    &model.gene_embed_grad,  &mut state.m_gene,  &mut state.v_gene,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc1.weights,   &model.fc1.grad_weights, &mut state.m_fc1_w, &mut state.v_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc1.bias,      &model.fc1.grad_bias,    &mut state.m_fc1_b, &mut state.v_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc2.weights,   &model.fc2.grad_weights, &mut state.m_fc2_w, &mut state.v_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc2.bias,      &model.fc2.grad_bias,    &mut state.m_fc2_b, &mut state.v_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc3.weights,   &model.fc3.grad_weights, &mut state.m_fc3_w, &mut state.v_fc3_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc3.bias,      &model.fc3.grad_bias,    &mut state.m_fc3_b, &mut state.v_fc3_b, lr, b1, b2, eps, state.t, scale);
}

// ── Shared helpers ────────────────────────────────────────────────────────────

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

fn mean(v: &[f32]) -> f32 { v.iter().sum::<f32>() / v.len() as f32 }
fn std_dev(v: &[f32]) -> f32 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
}

fn eval_tt(
    model: &mut HybridEmbeddingModel,
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize, batch_size: usize,
) -> f32 {
    let mut all_preds: Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        all_preds.extend_from_slice(preds.as_slice().unwrap());
        all_labels.extend_from_slice(&labels);
    }
    calculate_auroc(&all_preds, &all_labels)
}

fn eval_ce(
    model: &mut CrossEncoderModel,
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize, batch_size: usize,
) -> f32 {
    let mut all_preds: Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        let probs: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        all_preds.extend_from_slice(&probs);
        all_labels.extend_from_slice(&labels);
    }
    calculate_auroc(&all_preds, &all_labels)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== B2: Per-Architecture LR Grid Search ===");

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

    // Fixed data split (seed=42 for shuffle, same as main experiments)
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

    let lrs        = [0.0005f32, 0.001, 0.005];
    let seeds      = [42u64, 123, 456];
    let embed_dim  = 512usize;
    let hidden_dim = 512usize;
    let output_dim = 512usize;
    let temperature = 0.05f32;
    let batch_size = 256usize;
    let epochs     = 60usize;
    let tt_clip    = 5.0f32;
    let ce_clip    = 1.0f32;

    // Results: two_tower[lr_idx][seed_idx] and cross_encoder[lr_idx][seed_idx]
    let mut tt_aurocs: Vec<Vec<f32>> = vec![Vec::new(); lrs.len()];
    let mut ce_aurocs: Vec<Vec<f32>> = vec![Vec::new(); lrs.len()];

    // ── Two-tower grid search ─────────────────────────────────────────────────
    println!("\n--- Two-Tower ---");
    for (lr_idx, &lr) in lrs.iter().enumerate() {
        println!("  lr={}:", lr);
        for &seed in &seeds {
            let mut model = HybridEmbeddingModel::new(
                num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, temperature, 0.01, seed,
            );
            let mut state = TtAdamState::new(&model);
            let mut best_val_auroc = 0.0f32;
            let mut patience = 0usize;

            for epoch in 0..epochs {
                let mut epoch_data = train_data.clone();
                let mut epoch_rng  = StdRng::seed_from_u64(seed + epoch as u64);
                epoch_data.shuffle(&mut epoch_rng);

                for start in (0..epoch_data.len()).step_by(batch_size) {
                    let end = (start + batch_size).min(epoch_data.len());
                    let batch = &epoch_data[start..end];
                    let bsz = batch.len();
                    let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                    let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                    let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                    let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                    let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                    let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                    let grad: Array1<f32> = preds.iter().zip(labels.iter())
                        .map(|(&p, &l)| (p - l) / bsz as f32).collect();
                    model.backward(&grad);
                    adam_step_tt(&mut model, &mut state, lr, tt_clip);
                    model.zero_grad();
                }

                if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                    let val_auroc = eval_tt(&mut model, &val_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
                    if val_auroc > best_val_auroc { best_val_auroc = val_auroc; patience = 0; }
                    else { patience += 1; if patience >= 3 { break; } }
                }
            }

            let test_auroc = eval_tt(&mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
            println!("    seed={} auroc={:.4}", seed, test_auroc);
            tt_aurocs[lr_idx].push(test_auroc);
        }
        println!("  lr={} mean={:.4} std={:.4}", lr, mean(&tt_aurocs[lr_idx]), std_dev(&tt_aurocs[lr_idx]));
    }

    // ── Cross-encoder grid search ─────────────────────────────────────────────
    println!("\n--- Cross-Encoder ---");
    for (lr_idx, &lr) in lrs.iter().enumerate() {
        println!("  lr={}:", lr);
        for &seed in &seeds {
            let mut model = CrossEncoderModel::new(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, seed);
            let mut state = CeAdamState::new(&model);
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
                    let logits     = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                    let labels_arr = Array1::from(labels);
                    let grad       = bce_loss_backward(&logits, &labels_arr);
                    model.backward(&grad);
                    adam_step_ce(&mut model, &mut state, lr, ce_clip);
                    model.zero_grad();
                }

                if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                    let val_auroc = eval_ce(&mut model, &val_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
                    if val_auroc > best_val_auroc { best_val_auroc = val_auroc; patience = 0; }
                    else { patience += 1; if patience >= 3 { break; } }
                }
            }

            let test_auroc = eval_ce(&mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
            println!("    seed={} auroc={:.4}", seed, test_auroc);
            ce_aurocs[lr_idx].push(test_auroc);
        }
        println!("  lr={} mean={:.4} std={:.4}", lr, mean(&ce_aurocs[lr_idx]), std_dev(&ce_aurocs[lr_idx]));
    }

    // ── Write results ─────────────────────────────────────────────────────────
    std::fs::create_dir_all("results")?;
    let mut tt_obj = serde_json::Map::new();
    let mut ce_obj = serde_json::Map::new();
    for (lr_idx, &lr) in lrs.iter().enumerate() {
        let lr_key = lr.to_string();
        tt_obj.insert(lr_key.clone(), serde_json::json!({
            "mean": mean(&tt_aurocs[lr_idx]),
            "std":  std_dev(&tt_aurocs[lr_idx]),
            "per_seed": tt_aurocs[lr_idx],
        }));
        ce_obj.insert(lr_key, serde_json::json!({
            "mean": mean(&ce_aurocs[lr_idx]),
            "std":  std_dev(&ce_aurocs[lr_idx]),
            "per_seed": ce_aurocs[lr_idx],
        }));
    }
    let output = serde_json::json!({
        "lrs":           lrs,
        "seeds":         seeds,
        "two_tower":     tt_obj,
        "cross_encoder": ce_obj,
    });
    std::fs::write("results/lr_gridsearch.json", serde_json::to_string_pretty(&output)?)?;
    println!("\n✓ Saved results/lr_gridsearch.json");
    Ok(())
}
