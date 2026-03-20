/// B6: Extended seeds (10 seeds) for two-tower and cross-encoder
/// Provides more robust mean±std estimates for Table 2 (C2 comparison)
/// Seeds: [42, 123, 456, 789, 1337, 2023, 3141, 7919, 9999, 12345]
/// Results → results/extended_seeds_results.json
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

fn tt_adam_step(model: &mut HybridEmbeddingModel, state: &mut TtAdamState, lr: f32, clip: f32) {
    state.t += 1;
    let b1  = 0.9f32;
    let b2  = 0.999f32;
    let eps = 1e-8f32;

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

    fn adam2d(
        param: &mut Array2<f32>, grad: &Array2<f32>,
        m: &mut Array2<f32>, v: &mut Array2<f32>,
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

    fn adam1d(
        param: &mut Array1<f32>, grad: &Array1<f32>,
        m: &mut Array1<f32>, v: &mut Array1<f32>,
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

    adam2d(&mut model.tf_embed,     &model.tf_embed_grad,    &mut state.m_tf,       &mut state.v_tf,       lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,   &model.gene_embed_grad,  &mut state.m_gene,     &mut state.v_gene,     lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.tf_fc1.weights, &model.tf_fc1.grad_weights, &mut state.m_tf_fc1_w, &mut state.v_tf_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.tf_fc1.bias,    &model.tf_fc1.grad_bias,    &mut state.m_tf_fc1_b, &mut state.v_tf_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.tf_fc2.weights, &model.tf_fc2.grad_weights, &mut state.m_tf_fc2_w, &mut state.v_tf_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.tf_fc2.bias,    &model.tf_fc2.grad_bias,    &mut state.m_tf_fc2_b, &mut state.v_tf_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_fc1.weights, &model.gene_fc1.grad_weights, &mut state.m_g_fc1_w, &mut state.v_g_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.gene_fc1.bias,    &model.gene_fc1.grad_bias,    &mut state.m_g_fc1_b, &mut state.v_g_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_fc2.weights, &model.gene_fc2.grad_weights, &mut state.m_g_fc2_w, &mut state.v_g_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.gene_fc2.bias,    &model.gene_fc2.grad_bias,    &mut state.m_g_fc2_b, &mut state.v_g_fc2_b, lr, b1, b2, eps, state.t, scale);
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

fn ce_adam_step(model: &mut CrossEncoderModel, state: &mut CeAdamState, lr: f32, clip: f32) {
    state.t += 1;
    let b1  = 0.9f32;
    let b2  = 0.999f32;
    let eps = 1e-8f32;

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

    fn adam2d(
        param: &mut Array2<f32>, grad: &Array2<f32>,
        m: &mut Array2<f32>, v: &mut Array2<f32>,
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

    fn adam1d(
        param: &mut Array1<f32>, grad: &Array1<f32>,
        m: &mut Array1<f32>, v: &mut Array1<f32>,
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

    adam2d(&mut model.tf_embed,    &model.tf_embed_grad,    &mut state.m_tf,    &mut state.v_tf,    lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,  &model.gene_embed_grad,  &mut state.m_gene,  &mut state.v_gene,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc1.weights, &model.fc1.grad_weights, &mut state.m_fc1_w, &mut state.v_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc1.bias,    &model.fc1.grad_bias,    &mut state.m_fc1_b, &mut state.v_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc2.weights, &model.fc2.grad_weights, &mut state.m_fc2_w, &mut state.v_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc2.bias,    &model.fc2.grad_bias,    &mut state.m_fc2_b, &mut state.v_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc3.weights, &model.fc3.grad_weights, &mut state.m_fc3_w, &mut state.v_fc3_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc3.bias,    &model.fc3.grad_bias,    &mut state.m_fc3_b, &mut state.v_fc3_b, lr, b1, b2, eps, state.t, scale);
}

// ── Utilities ─────────────────────────────────────────────────────────────────

fn build_expr_batch(
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

fn calculate_f1(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut tp = 0.0f32; let mut fp = 0.0f32; let mut fn_ = 0.0f32;
    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        let pred = if p >= 0.5 { 1.0f32 } else { 0.0 };
        match (pred as i32, l as i32) {
            (1, 1) => tp += 1.0,
            (1, 0) => fp += 1.0,
            (0, 1) => fn_ += 1.0,
            _ => {}
        }
    }
    let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let rec  = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
    if prec + rec > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 }
}

fn calculate_auprc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels.iter())
        .map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    if n_pos == 0.0 { return 0.0; }
    let mut tp = 0.0f32; let mut fp = 0.0f32;
    let mut ap = 0.0f32; let mut prev_recall = 0.0f32;
    for &(_, label) in &pairs {
        if label == 1.0 { tp += 1.0; } else { fp += 1.0; }
        let recall    = tp / n_pos;
        let precision = tp / (tp + fp);
        ap += precision * (recall - prev_recall);
        prev_recall = recall;
    }
    ap
}

fn evaluate_tt(
    model:         &mut HybridEmbeddingModel,
    data:          &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim:      usize,
    batch_size:    usize,
) -> (f32, f32, f32, f32) {
    let mut all_preds:  Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end   = (start + batch_size).min(data.len());
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
    let n_correct = all_preds.iter().zip(all_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count();
    (
        n_correct as f32 / all_preds.len() as f32,
        calculate_auroc(&all_preds, &all_labels),
        calculate_f1(&all_preds, &all_labels),
        calculate_auprc(&all_preds, &all_labels),
    )
}

fn evaluate_ce(
    model:         &mut CrossEncoderModel,
    data:          &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim:      usize,
    batch_size:    usize,
) -> (f32, f32, f32, f32) {
    let mut all_preds:  Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end   = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        let probs: Vec<f32> = logits.iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        all_preds.extend_from_slice(&probs);
        all_labels.extend_from_slice(&labels);
    }
    let n_correct = all_preds.iter().zip(all_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count();
    (
        n_correct as f32 / all_preds.len() as f32,
        calculate_auroc(&all_preds, &all_labels),
        calculate_f1(&all_preds, &all_labels),
        calculate_auprc(&all_preds, &all_labels),
    )
}

fn mean_std(vals: &[f32]) -> (f32, f32) {
    let m = vals.iter().sum::<f32>() / vals.len() as f32;
    let s = (vals.iter().map(|x| (x - m).powi(2)).sum::<f32>() / vals.len() as f32).sqrt();
    (m, s)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== B6: Extended Seeds (10 seeds) ===\n");

    let config    = Config::load_default()?;
    let priors    = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder   = PriorDatasetBuilder::new(priors.clone());
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

    let mut tf_expr_map:   HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    for (name, &idx) in builder.get_tf_vocab().iter() {
        let v = gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned())
            .unwrap_or_else(|| Array1::zeros(expr_dim));
        tf_expr_map.insert(idx, v);
    }
    for (name, &idx) in builder.get_gene_vocab().iter() {
        let v = gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned())
            .unwrap_or_else(|| Array1::zeros(expr_dim));
        gene_expr_map.insert(idx, v);
    }

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
    let train_data = examples[..n_train].to_vec();
    let val_data   = examples[n_train..n_train+n_val].to_vec();
    let test_data  = examples[n_train+n_val..].to_vec();
    println!("Train: {} | Val: {} | Test: {}", train_data.len(), val_data.len(), test_data.len());
    println!("TFs: {} | Genes: {} | expr_dim: {}\n", num_tfs, num_genes, expr_dim);

    let seeds: [u64; 10] = [42, 123, 456, 789, 1337, 2023, 3141, 7919, 9999, 12345];
    let embed_dim  = 512usize;
    let hidden_dim = 512usize;
    let output_dim = 512usize;
    let temperature = 0.05f32;
    let lr         = 0.001f32;
    let clip       = 5.0f32;
    let batch_size = 256usize;
    let epochs     = 60usize;

    std::fs::create_dir_all("results")?;

    // ── Two-Tower ──────────────────────────────────────────────────────────────
    println!("--- Two-Tower (10 seeds) ---");
    let mut tt_accs:   Vec<f32> = Vec::new();
    let mut tt_aurocs: Vec<f32> = Vec::new();
    let mut tt_f1s:    Vec<f32> = Vec::new();
    let mut tt_auprcs: Vec<f32> = Vec::new();

    for &seed in &seeds {
        let mut model = HybridEmbeddingModel::new(
            num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, temperature, 0.01, seed,
        );
        let mut state = TtAdamState::new(&model);
        let mut best_val_acc = 0.0f32;
        let mut patience_ctr = 0usize;
        let mut train_shuffled = train_data.clone();

        for epoch in 0..epochs {
            let mut erng = StdRng::seed_from_u64(seed + epoch as u64);
            train_shuffled.shuffle(&mut erng);

            for start in (0..train_shuffled.len()).step_by(batch_size) {
                let end   = (start + batch_size).min(train_shuffled.len());
                let batch = &train_shuffled[start..end];
                let bsz   = batch.len();
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let grad: Array1<f32> = preds.iter().zip(labels.iter())
                    .map(|(&p, &l)| (p - l) / bsz as f32).collect();
                model.backward(&grad);
                tt_adam_step(&mut model, &mut state, lr, clip);
                model.zero_grad();
            }

            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                let (_, val_acc) = {
                    let mut total_loss = 0.0f32; let mut n_correct = 0usize; let mut n_total = 0usize;
                    for start in (0..val_data.len()).step_by(batch_size) {
                        let end   = (start + batch_size).min(val_data.len());
                        let batch = &val_data[start..end];
                        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                        let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                        let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                        let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                        for (&p, &l) in preds.iter().zip(labels.iter()) {
                            let pc = p.clamp(1e-7, 1.0 - 1e-7);
                            total_loss += -(l * pc.ln() + (1.0 - l) * (1.0 - pc).ln());
                            if (p >= 0.5) == (l == 1.0) { n_correct += 1; }
                            n_total += 1;
                        }
                    }
                    (total_loss / n_total as f32, n_correct as f32 / n_total as f32)
                };
                if val_acc > best_val_acc { best_val_acc = val_acc; patience_ctr = 0; }
                else { patience_ctr += 1; if patience_ctr >= 10 { break; } }
            }
        }

        let (acc, auroc, f1, auprc) = evaluate_tt(
            &mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
        );
        println!("  TT seed {}: acc={:.2}% auroc={:.4}", seed, acc*100.0, auroc);
        tt_accs.push(acc); tt_aurocs.push(auroc); tt_f1s.push(f1); tt_auprcs.push(auprc);
    }

    let (tt_mean_acc, tt_std_acc)     = mean_std(&tt_accs);
    let (tt_mean_auroc, tt_std_auroc) = mean_std(&tt_aurocs);
    let (tt_mean_f1, tt_std_f1)       = mean_std(&tt_f1s);
    let (tt_mean_auprc, tt_std_auprc) = mean_std(&tt_auprcs);
    println!("\nTT 10-seed: acc={:.2}%±{:.2}% auroc={:.4}±{:.4}",
             tt_mean_acc*100.0, tt_std_acc*100.0, tt_mean_auroc, tt_std_auroc);

    // ── Cross-Encoder ──────────────────────────────────────────────────────────
    println!("\n--- Cross-Encoder (10 seeds) ---");
    let mut ce_accs:   Vec<f32> = Vec::new();
    let mut ce_aurocs: Vec<f32> = Vec::new();
    let mut ce_f1s:    Vec<f32> = Vec::new();
    let mut ce_auprcs: Vec<f32> = Vec::new();

    for &seed in &seeds {
        let mut model = CrossEncoderModel::new(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, seed);
        let mut state = CeAdamState::new(&model);
        let mut best_val_acc = 0.0f32;
        let mut patience_ctr = 0usize;
        let mut train_shuffled = train_data.clone();

        for epoch in 0..epochs {
            let mut erng = StdRng::seed_from_u64(seed + epoch as u64);
            train_shuffled.shuffle(&mut erng);

            for start in (0..train_shuffled.len()).step_by(batch_size) {
                let end   = (start + batch_size).min(train_shuffled.len());
                let batch = &train_shuffled[start..end];
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                let labels_arr = Array1::from(labels);
                let logits  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let grad    = bce_loss_backward(&logits, &labels_arr);
                model.backward(&grad);
                ce_adam_step(&mut model, &mut state, 0.005, 1.0);
                model.zero_grad();
            }

            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                let val_acc = {
                    let mut n_correct = 0usize; let mut n_total = 0usize;
                    for start in (0..val_data.len()).step_by(batch_size) {
                        let end   = (start + batch_size).min(val_data.len());
                        let batch = &val_data[start..end];
                        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                        let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                        let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                        let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                        for (&logit, &l) in logits.iter().zip(labels.iter()) {
                            let p = 1.0 / (1.0 + (-logit).exp());
                            if (p >= 0.5) == (l == 1.0) { n_correct += 1; }
                            n_total += 1;
                        }
                    }
                    n_correct as f32 / n_total as f32
                };
                if val_acc > best_val_acc { best_val_acc = val_acc; patience_ctr = 0; }
                else { patience_ctr += 1; if patience_ctr >= 10 { break; } }
            }
        }

        let (acc, auroc, f1, auprc) = evaluate_ce(
            &mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
        );
        println!("  CE seed {}: acc={:.2}% auroc={:.4}", seed, acc*100.0, auroc);
        ce_accs.push(acc); ce_aurocs.push(auroc); ce_f1s.push(f1); ce_auprcs.push(auprc);
    }

    let (ce_mean_acc, ce_std_acc)     = mean_std(&ce_accs);
    let (ce_mean_auroc, ce_std_auroc) = mean_std(&ce_aurocs);
    let (ce_mean_f1, ce_std_f1)       = mean_std(&ce_f1s);
    let (ce_mean_auprc, ce_std_auprc) = mean_std(&ce_auprcs);
    println!("\nCE 10-seed: acc={:.2}%±{:.2}% auroc={:.4}±{:.4}",
             ce_mean_acc*100.0, ce_std_acc*100.0, ce_mean_auroc, ce_std_auroc);

    // ── Write results ─────────────────────────────────────────────────────────
    let result = serde_json::json!({
        "seeds": seeds,
        "n_seeds": 10,
        "two_tower": {
            "seed_accuracies": tt_accs,
            "seed_aurocs":     tt_aurocs,
            "seed_f1s":        tt_f1s,
            "seed_auprcs":     tt_auprcs,
            "mean_accuracy":   tt_mean_acc,
            "std_accuracy":    tt_std_acc,
            "mean_auroc":      tt_mean_auroc,
            "std_auroc":       tt_std_auroc,
            "mean_f1":         tt_mean_f1,
            "std_f1":          tt_std_f1,
            "mean_auprc":      tt_mean_auprc,
            "std_auprc":       tt_std_auprc,
        },
        "cross_encoder": {
            "seed_accuracies": ce_accs,
            "seed_aurocs":     ce_aurocs,
            "seed_f1s":        ce_f1s,
            "seed_auprcs":     ce_auprcs,
            "mean_accuracy":   ce_mean_acc,
            "std_accuracy":    ce_std_acc,
            "mean_auroc":      ce_mean_auroc,
            "std_auroc":       ce_std_auroc,
            "mean_f1":         ce_mean_f1,
            "std_f1":          ce_std_f1,
            "mean_auprc":      ce_mean_auprc,
            "std_auprc":       ce_std_auprc,
        }
    });
    std::fs::write("results/extended_seeds_results.json", serde_json::to_string_pretty(&result)?)?;
    println!("\n✓ Saved results/extended_seeds_results.json");
    Ok(())
}
