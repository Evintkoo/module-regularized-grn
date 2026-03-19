/// Extended imbalance experiments: 10:1 and 50:1 negative ratios.
/// Runs both two-tower (HybridEmbeddingModel) and cross-encoder (CrossEncoderModel)
/// at neg_ratio ∈ {10, 50}, 5 seeds each.
/// Evaluation always uses the same balanced (1:1) test set as the main paper experiments.
/// Writes:
///   results/two_tower_10to1.json,   results/two_tower_50to1.json
///   results/cross_encoder_10to1.json, results/cross_encoder_50to1.json
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

// ── Adam state for HybridEmbeddingModel (two-tower) ──────────────────────────

struct AdamTT {
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

impl AdamTT {
    fn new(m: &HybridEmbeddingModel) -> Self {
        Self {
            m_tf:       Array2::zeros(m.tf_embed.dim()),
            v_tf:       Array2::zeros(m.tf_embed.dim()),
            m_gene:     Array2::zeros(m.gene_embed.dim()),
            v_gene:     Array2::zeros(m.gene_embed.dim()),
            m_tf_fc1_w: Array2::zeros(m.tf_fc1.weights.dim()),
            v_tf_fc1_w: Array2::zeros(m.tf_fc1.weights.dim()),
            m_tf_fc1_b: Array1::zeros(m.tf_fc1.bias.len()),
            v_tf_fc1_b: Array1::zeros(m.tf_fc1.bias.len()),
            m_tf_fc2_w: Array2::zeros(m.tf_fc2.weights.dim()),
            v_tf_fc2_w: Array2::zeros(m.tf_fc2.weights.dim()),
            m_tf_fc2_b: Array1::zeros(m.tf_fc2.bias.len()),
            v_tf_fc2_b: Array1::zeros(m.tf_fc2.bias.len()),
            m_g_fc1_w:  Array2::zeros(m.gene_fc1.weights.dim()),
            v_g_fc1_w:  Array2::zeros(m.gene_fc1.weights.dim()),
            m_g_fc1_b:  Array1::zeros(m.gene_fc1.bias.len()),
            v_g_fc1_b:  Array1::zeros(m.gene_fc1.bias.len()),
            m_g_fc2_w:  Array2::zeros(m.gene_fc2.weights.dim()),
            v_g_fc2_w:  Array2::zeros(m.gene_fc2.weights.dim()),
            m_g_fc2_b:  Array1::zeros(m.gene_fc2.bias.len()),
            v_g_fc2_b:  Array1::zeros(m.gene_fc2.bias.len()),
            t: 0,
        }
    }
}

// ── Adam state for CrossEncoderModel ─────────────────────────────────────────

struct AdamCE {
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

impl AdamCE {
    fn new(m: &CrossEncoderModel) -> Self {
        Self {
            m_tf:    Array2::zeros(m.tf_embed.dim()),
            v_tf:    Array2::zeros(m.tf_embed.dim()),
            m_gene:  Array2::zeros(m.gene_embed.dim()),
            v_gene:  Array2::zeros(m.gene_embed.dim()),
            m_fc1_w: Array2::zeros(m.fc1.weights.dim()),
            v_fc1_w: Array2::zeros(m.fc1.weights.dim()),
            m_fc1_b: Array1::zeros(m.fc1.bias.len()),
            v_fc1_b: Array1::zeros(m.fc1.bias.len()),
            m_fc2_w: Array2::zeros(m.fc2.weights.dim()),
            v_fc2_w: Array2::zeros(m.fc2.weights.dim()),
            m_fc2_b: Array1::zeros(m.fc2.bias.len()),
            v_fc2_b: Array1::zeros(m.fc2.bias.len()),
            m_fc3_w: Array2::zeros(m.fc3.weights.dim()),
            v_fc3_w: Array2::zeros(m.fc3.weights.dim()),
            m_fc3_b: Array1::zeros(m.fc3.bias.len()),
            v_fc3_b: Array1::zeros(m.fc3.bias.len()),
            t: 0,
        }
    }
}

// ── Shared Adam update primitives ─────────────────────────────────────────────

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

fn grad_clip_scale(norms: &[f32], clip: f32) -> f32 {
    let norm: f32 = norms.iter().sum::<f32>().sqrt();
    if norm > clip { clip / norm } else { 1.0 }
}

fn adam_step_tt(model: &mut HybridEmbeddingModel, s: &mut AdamTT, lr: f32, clip: f32) {
    s.t += 1;
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let scale = grad_clip_scale(&[
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
    ], clip);
    adam2d(&mut model.tf_embed,         &model.tf_embed_grad,         &mut s.m_tf,       &mut s.v_tf,       lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_embed,       &model.gene_embed_grad,       &mut s.m_gene,     &mut s.v_gene,     lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.tf_fc1.weights,   &model.tf_fc1.grad_weights,   &mut s.m_tf_fc1_w, &mut s.v_tf_fc1_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.tf_fc1.bias,      &model.tf_fc1.grad_bias,      &mut s.m_tf_fc1_b, &mut s.v_tf_fc1_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.tf_fc2.weights,   &model.tf_fc2.grad_weights,   &mut s.m_tf_fc2_w, &mut s.v_tf_fc2_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.tf_fc2.bias,      &model.tf_fc2.grad_bias,      &mut s.m_tf_fc2_b, &mut s.v_tf_fc2_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_fc1.weights, &model.gene_fc1.grad_weights, &mut s.m_g_fc1_w,  &mut s.v_g_fc1_w,  lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.gene_fc1.bias,    &model.gene_fc1.grad_bias,    &mut s.m_g_fc1_b,  &mut s.v_g_fc1_b,  lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_fc2.weights, &model.gene_fc2.grad_weights, &mut s.m_g_fc2_w,  &mut s.v_g_fc2_w,  lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.gene_fc2.bias,    &model.gene_fc2.grad_bias,    &mut s.m_g_fc2_b,  &mut s.v_g_fc2_b,  lr, b1, b2, eps, s.t, scale);
}

fn adam_step_ce(model: &mut CrossEncoderModel, s: &mut AdamCE, lr: f32, clip: f32) {
    s.t += 1;
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let scale = grad_clip_scale(&[
        model.tf_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.gene_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.fc3.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc3.grad_bias.iter().map(|x| x*x).sum::<f32>(),
    ], clip);
    adam2d(&mut model.tf_embed,    &model.tf_embed_grad,    &mut s.m_tf,    &mut s.v_tf,    lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_embed,  &model.gene_embed_grad,  &mut s.m_gene,  &mut s.v_gene,  lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.fc1.weights, &model.fc1.grad_weights, &mut s.m_fc1_w, &mut s.v_fc1_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.fc1.bias,    &model.fc1.grad_bias,    &mut s.m_fc1_b, &mut s.v_fc1_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.fc2.weights, &model.fc2.grad_weights, &mut s.m_fc2_w, &mut s.v_fc2_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.fc2.bias,    &model.fc2.grad_bias,    &mut s.m_fc2_b, &mut s.v_fc2_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.fc3.weights, &model.fc3.grad_weights, &mut s.m_fc3_w, &mut s.v_fc3_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.fc3.bias,    &model.fc3.grad_bias,    &mut s.m_fc3_b, &mut s.v_fc3_b, lr, b1, b2, eps, s.t, scale);
}

// ── Metrics ───────────────────────────────────────────────────────────────────

fn calculate_auroc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels.iter())
        .map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    let n_neg = labels.len() as f32 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    let mut tp = 0.0f32; let mut fp = 0.0f32;
    let mut auc = 0.0f32; let mut prev_fp = 0.0f32;
    for &(_, label) in &pairs {
        if label == 1.0 { tp += 1.0; } else {
            fp += 1.0;
            auc += (tp / n_pos) * ((fp - prev_fp) / n_neg);
            prev_fp = fp;
        }
    }
    auc
}

fn calculate_f1(predictions: &[f32], labels: &[f32]) -> f32 {
    let (mut tp, mut fp, mut fn_) = (0.0f32, 0.0f32, 0.0f32);
    for (&p, &l) in predictions.iter().zip(labels.iter()) {
        match ((p >= 0.5) as i32, l as i32) {
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
        let recall = tp / n_pos;
        let prec   = tp / (tp + fp);
        ap += prec * (recall - prev_recall);
        prev_recall = recall;
    }
    ap
}

fn bootstrap_ci(y_true: &[f64], y_pred: &[f64], n: usize, seed: u64) -> (f64, f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<usize> = (0..y_true.len()).collect();
    let mut scores: Vec<f64> = (0..n).map(|_| {
        let sample: Vec<usize> = (0..y_true.len())
            .map(|_| *indices.choose(&mut rng).unwrap())
            .collect();
        sample.iter().filter(|&&i| (y_pred[i] > 0.5) == (y_true[i] > 0.5)).count() as f64
            / sample.len() as f64
    }).collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean  = scores.iter().sum::<f64>() / scores.len() as f64;
    let lower = scores[(0.025 * scores.len() as f64) as usize];
    let upper = scores[(0.975 * scores.len() as f64) as usize];
    (mean, lower, upper)
}

// ── Expression helpers ────────────────────────────────────────────────────────

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

fn build_expression_maps(
    builder:   &PriorDatasetBuilder,
    expr_data: &ExpressionData,
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

// ── Evaluation helpers ────────────────────────────────────────────────────────

fn eval_tt(
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

fn eval_ce(
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
        let probs: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
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

// ── Training loops ────────────────────────────────────────────────────────────

fn run_two_tower(
    train_data:    &[(usize, usize, f32)],
    val_data:      &[(usize, usize, f32)],
    test_data:     &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    num_tfs:       usize,
    num_genes:     usize,
    expr_dim:      usize,
    neg_ratio:     usize,
) -> Result<serde_json::Value> {
    let seeds      = [42u64, 123, 456, 789, 1337];
    let embed_dim  = 512usize;
    let hidden_dim = 512usize;
    let output_dim = 512usize;
    let temperature = 0.05f32;
    let lr         = 0.001f32;
    let clip       = 5.0f32;
    let batch_size = 256usize;
    let epochs     = 60usize;
    let patience   = 10usize;

    let mut seed_accuracies: Vec<f32> = Vec::new();
    let mut seed_aurocs:     Vec<f32> = Vec::new();
    let mut seed_f1s:        Vec<f32> = Vec::new();
    let mut seed_auprcs:     Vec<f32> = Vec::new();
    let mut all_ensemble_preds: Vec<Vec<f32>> = Vec::new();
    let mut best_final_val_acc  = 0.0f32;
    let mut best_test_preds: Vec<f32> = Vec::new();
    let test_labels_once: Vec<f32> = test_data.iter().map(|x| x.2).collect();

    for &seed in &seeds {
        println!("  [TT] seed {}:", seed);
        let mut model = HybridEmbeddingModel::new(
            num_tfs, num_genes, embed_dim, expr_dim,
            hidden_dim, output_dim, temperature, 0.01, seed,
        );
        let mut state = AdamTT::new(&model);
        let mut best_seed_val_acc = 0.0f32;
        let mut patience_counter  = 0usize;
        let mut train_shuffled    = train_data.to_vec();

        for epoch in 0..epochs {
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            train_shuffled.shuffle(&mut epoch_rng);

            for start in (0..train_shuffled.len()).step_by(batch_size) {
                let end   = (start + batch_size).min(train_shuffled.len());
                let batch = &train_shuffled[start..end];
                let bsz   = batch.len();
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
                let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let grad: Array1<f32> = preds.iter().zip(labels.iter())
                    .map(|(&p, &l)| (p - l) / bsz as f32).collect();
                model.backward(&grad);
                adam_step_tt(&mut model, &mut state, lr, clip);
                model.zero_grad();
            }

            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                let (val_acc, _, _, _) = eval_tt(
                    &mut model, val_data, tf_expr_map, gene_expr_map, expr_dim, batch_size
                );
                if val_acc > best_seed_val_acc {
                    best_seed_val_acc = val_acc;
                    patience_counter  = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience { break; }
                }
            }
        }

        let (test_acc, test_auroc, test_f1, test_auprc) = eval_tt(
            &mut model, test_data, tf_expr_map, gene_expr_map, expr_dim, batch_size
        );
        println!("    acc={:.2}% auroc={:.4} f1={:.4} auprc={:.4}",
                 test_acc * 100.0, test_auroc, test_f1, test_auprc);

        let mut preds: Vec<f32> = Vec::new();
        for start in (0..test_data.len()).step_by(batch_size) {
            let end   = (start + batch_size).min(test_data.len());
            let batch = &test_data[start..end];
            let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
            let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
            let p = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
            preds.extend_from_slice(p.as_slice().unwrap());
        }
        if best_seed_val_acc > best_final_val_acc {
            best_final_val_acc = best_seed_val_acc;
            best_test_preds    = preds.clone();
        }
        all_ensemble_preds.push(preds);
        seed_accuracies.push(test_acc);
        seed_aurocs.push(test_auroc);
        seed_f1s.push(test_f1);
        seed_auprcs.push(test_auprc);
    }

    let n_test = test_data.len();
    let ensemble_preds: Vec<f32> = (0..n_test)
        .map(|i| all_ensemble_preds.iter().map(|p| p[i]).sum::<f32>() / seeds.len() as f32)
        .collect();
    let ensemble_acc   = ensemble_preds.iter().zip(test_labels_once.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / n_test as f32;
    let ensemble_auroc = calculate_auroc(&ensemble_preds, &test_labels_once);
    let ensemble_auprc = calculate_auprc(&ensemble_preds, &test_labels_once);

    let bp: Vec<f64> = best_test_preds.iter().map(|&x| x as f64).collect();
    let bl: Vec<f64> = test_labels_once.iter().map(|&x| x as f64).collect();
    let (_, ci_lower, ci_upper) = bootstrap_ci(&bl, &bp, 1000, 42);

    let mean_acc = seed_accuracies.iter().sum::<f32>() / seeds.len() as f32;
    let std_acc  = (seed_accuracies.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>()
                    / seeds.len() as f32).sqrt();
    let mean_auroc = seed_aurocs.iter().sum::<f32>() / seeds.len() as f32;
    let std_auroc  = (seed_aurocs.iter().map(|x| (x - mean_auroc).powi(2)).sum::<f32>()
                      / seeds.len() as f32).sqrt();
    println!("  [TT] {}:1 → mean_acc={:.2}%±{:.2}%  mean_auroc={:.4}±{:.4}  ensemble_auroc={:.4}",
             neg_ratio, mean_acc*100.0, std_acc*100.0, mean_auroc, std_auroc, ensemble_auroc);

    Ok(serde_json::json!({
        "model": "two_tower",
        "neg_ratio": neg_ratio,
        "seeds": seeds,
        "seed_accuracies": seed_accuracies,
        "seed_aurocs": seed_aurocs,
        "seed_f1s": seed_f1s,
        "seed_auprcs": seed_auprcs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "ensemble_accuracy": ensemble_acc,
        "ensemble_auroc": ensemble_auroc,
        "ensemble_auprc": ensemble_auprc,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    }))
}

fn run_cross_encoder(
    train_data:    &[(usize, usize, f32)],
    val_data:      &[(usize, usize, f32)],
    test_data:     &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    num_tfs:       usize,
    num_genes:     usize,
    expr_dim:      usize,
    neg_ratio:     usize,
) -> Result<serde_json::Value> {
    let seeds      = [42u64, 123, 456, 789, 1337];
    let embed_dim  = 512usize;
    let hidden_dim = 512usize;
    let lr         = 0.001f32;
    let clip       = 1.0f32;
    let batch_size = 256usize;
    let epochs     = 60usize;

    let mut seed_accuracies: Vec<f32> = Vec::new();
    let mut seed_aurocs:     Vec<f32> = Vec::new();
    let mut seed_f1s:        Vec<f32> = Vec::new();
    let mut seed_auprcs:     Vec<f32> = Vec::new();
    let mut all_ensemble_preds: Vec<Vec<f32>> = Vec::new();
    let mut best_final_val_acc  = 0.0f32;
    let mut best_test_preds: Vec<f32> = Vec::new();
    let test_labels_once: Vec<f32> = test_data.iter().map(|x| x.2).collect();

    for &seed in &seeds {
        println!("  [CE] seed {}:", seed);
        let mut model = CrossEncoderModel::new(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, seed);
        let mut state = AdamCE::new(&model);
        let mut best_seed_val_acc = 0.0f32;
        let mut patience = 0usize;
        let mut train_shuffled = train_data.to_vec();

        for epoch in 0..epochs {
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            train_shuffled.shuffle(&mut epoch_rng);

            for start in (0..train_shuffled.len()).step_by(batch_size) {
                let end   = (start + batch_size).min(train_shuffled.len());
                let batch = &train_shuffled[start..end];
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
                let logits     = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let labels_arr = Array1::from(labels);
                let grad       = bce_loss_backward(&logits, &labels_arr);
                model.backward(&grad);
                adam_step_ce(&mut model, &mut state, lr, clip);
                model.zero_grad();
            }

            if epoch % 5 == 0 || epoch == epochs - 1 {
                let (val_acc, _, _, _) = eval_ce(
                    &mut model, val_data, tf_expr_map, gene_expr_map, expr_dim, batch_size
                );
                if val_acc > best_seed_val_acc {
                    best_seed_val_acc = val_acc;
                    patience = 0;
                } else {
                    patience += 1;
                    if patience >= 3 { break; }
                }
            }
        }

        let (test_acc, test_auroc, test_f1, test_auprc) = eval_ce(
            &mut model, test_data, tf_expr_map, gene_expr_map, expr_dim, batch_size
        );
        println!("    acc={:.2}% auroc={:.4} f1={:.4} auprc={:.4}",
                 test_acc * 100.0, test_auroc, test_f1, test_auprc);

        let mut preds: Vec<f32> = Vec::new();
        for start in (0..test_data.len()).step_by(batch_size) {
            let end   = (start + batch_size).min(test_data.len());
            let batch = &test_data[start..end];
            let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
            let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
            let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
            preds.extend(logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())));
        }
        if best_seed_val_acc > best_final_val_acc {
            best_final_val_acc = best_seed_val_acc;
            best_test_preds    = preds.clone();
        }
        all_ensemble_preds.push(preds);
        seed_accuracies.push(test_acc);
        seed_aurocs.push(test_auroc);
        seed_f1s.push(test_f1);
        seed_auprcs.push(test_auprc);
    }

    let n_test = test_data.len();
    let ensemble_preds: Vec<f32> = (0..n_test)
        .map(|i| all_ensemble_preds.iter().map(|p| p[i]).sum::<f32>() / seeds.len() as f32)
        .collect();
    let ensemble_acc   = ensemble_preds.iter().zip(test_labels_once.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / n_test as f32;
    let ensemble_auroc = calculate_auroc(&ensemble_preds, &test_labels_once);
    let ensemble_auprc = calculate_auprc(&ensemble_preds, &test_labels_once);

    let bp: Vec<f64> = best_test_preds.iter().map(|&x| x as f64).collect();
    let bl: Vec<f64> = test_labels_once.iter().map(|&x| x as f64).collect();
    let (_, ci_lower, ci_upper) = bootstrap_ci(&bl, &bp, 1000, 42);

    let mean_acc = seed_accuracies.iter().sum::<f32>() / seeds.len() as f32;
    let std_acc  = (seed_accuracies.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>()
                    / seeds.len() as f32).sqrt();
    let mean_auroc = seed_aurocs.iter().sum::<f32>() / seeds.len() as f32;
    let std_auroc  = (seed_aurocs.iter().map(|x| (x - mean_auroc).powi(2)).sum::<f32>()
                      / seeds.len() as f32).sqrt();
    println!("  [CE] {}:1 → mean_acc={:.2}%±{:.2}%  mean_auroc={:.4}±{:.4}  ensemble_auroc={:.4}",
             neg_ratio, mean_acc*100.0, std_acc*100.0, mean_auroc, std_auroc, ensemble_auroc);

    Ok(serde_json::json!({
        "model": "cross_encoder",
        "neg_ratio": neg_ratio,
        "seeds": seeds,
        "seed_accuracies": seed_accuracies,
        "seed_aurocs": seed_aurocs,
        "seed_f1s": seed_f1s,
        "seed_auprcs": seed_auprcs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
        "ensemble_accuracy": ensemble_acc,
        "ensemble_auroc": ensemble_auroc,
        "ensemble_auprc": ensemble_auprc,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    }))
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== Extended Imbalance Experiments: 10:1 and 50:1 ===\n");

    let config    = Config::load_default()?;
    let base_seed = config.project.seed;
    let priors    = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder   = PriorDatasetBuilder::new(priors);
    let num_tfs   = builder.num_tfs();
    let num_genes = builder.num_genes();

    println!("Loading expression data...");
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    let expr_dim = expr_data.n_cell_types;
    println!("  Cell types: {} | Expression genes: {}", expr_dim, expr_data.n_genes);

    let (tf_expr_map, gene_expr_map) = build_expression_maps(&builder, &expr_data);

    // Base 1:1 dataset — fixed splits (same seed as main experiments)
    let mut rng = StdRng::seed_from_u64(base_seed);
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

    println!("Dataset: total={} | train_base={} | val={} | test={}",
             n, train_base.len(), val_data.len(), test_data.len());
    println!("TFs: {} | Genes: {} | expr_dim: {}\n", num_tfs, num_genes, expr_dim);

    std::fs::create_dir_all("results")?;

    for &neg_ratio in &[10usize, 50] {
        println!("─── neg_ratio = {}:1 ───", neg_ratio);

        // Build imbalanced training set (same construction as main experiments)
        let pos_train: Vec<_> = train_base.iter().filter(|x| x.2 == 1.0).cloned().collect();
        let neg_extra = builder.sample_negative_examples(
            pos_train.len() * neg_ratio, base_seed + neg_ratio as u64
        );
        let mut train_data: Vec<(usize, usize, f32)> = pos_train;
        train_data.extend(neg_extra.iter().map(|&(tf, g)| (tf, g, 0.0f32)));
        let mut r = StdRng::seed_from_u64(base_seed + 1000);
        train_data.shuffle(&mut r);
        println!("  Training set: {} examples ({} pos + {} neg)\n",
                 train_data.len(),
                 train_data.iter().filter(|x| x.2 == 1.0).count(),
                 train_data.iter().filter(|x| x.2 == 0.0).count());

        // ── Two-tower ──
        let tt_result = run_two_tower(
            &train_data, &val_data, &test_data,
            &tf_expr_map, &gene_expr_map,
            num_tfs, num_genes, expr_dim, neg_ratio,
        )?;
        let tt_file = format!("results/two_tower_{}to1.json", neg_ratio);
        std::fs::write(&tt_file, serde_json::to_string_pretty(&tt_result)?)?;
        println!("  ✓ Saved {}", tt_file);

        // ── Cross-encoder ──
        let ce_result = run_cross_encoder(
            &train_data, &val_data, &test_data,
            &tf_expr_map, &gene_expr_map,
            num_tfs, num_genes, expr_dim, neg_ratio,
        )?;
        let ce_file = format!("results/cross_encoder_{}to1.json", neg_ratio);
        std::fs::write(&ce_file, serde_json::to_string_pretty(&ce_result)?)?;
        println!("  ✓ Saved {}\n", ce_file);
    }

    println!("=== All extended imbalance experiments complete ===");
    Ok(())
}
