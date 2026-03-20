/// B3: Embedding-only baseline
/// Trains HybridEmbeddingModel and CrossEncoderModel with ALL expression features
/// zeroed out, isolating the contribution of expression vs. embedding-only learning.
/// 5 seeds [42, 123, 456, 789, 1337], 1:1 ratio, lr=0.001, batch=256, 60 epochs.
/// Results → results/embedding_only_results.json
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
    println!("=== B3: Embedding-Only Baseline ===");
    println!("Expression features ZEROED OUT — measures embedding-only contribution\n");

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
    // expr_dim is kept to preserve model architecture (same layer sizes), but values are zeroed
    let expr_dim = expr_data.n_cell_types;
    println!("  TFs: {} | Genes: {} | expr_dim: {} (zeroed out)", num_tfs, num_genes, expr_dim);

    // We don't need expression maps since we zero everything out, but we need expr_dim for model sizing
    // Build trivial zero maps (all zeros = embedding-only)
    let tf_expr_map:   HashMap<usize, Array1<f32>> = builder.get_tf_vocab().iter()
        .map(|(_, &idx)| (idx, Array1::zeros(expr_dim))).collect();
    let gene_expr_map: HashMap<usize, Array1<f32>> = builder.get_gene_vocab().iter()
        .map(|(_, &idx)| (idx, Array1::zeros(expr_dim))).collect();

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
    println!("  Train: {} | Val: {} | Test: {}\n", train_data.len(), val_data.len(), test_data.len());

    let embed_dim   = 512usize;
    let hidden_dim  = 512usize;
    let output_dim  = 512usize;
    let temperature = 0.05f32;
    let lr          = 0.001f32;
    let batch_size  = 256usize;
    let epochs      = 60usize;
    let seeds       = [42u64, 123, 456, 789, 1337];

    // Helper: build zero expression batch (already zeros in our maps, but explicit here for clarity)
    let zero_expr = |n: usize| -> Array2<f32> { Array2::zeros((n, expr_dim)) };

    // ── Two-Tower Embedding-Only ──────────────────────────────────────────────
    println!("=== Two-Tower (embedding-only) ===");
    let mut tt_aurocs:    Vec<f32> = Vec::new();
    let mut tt_accs:      Vec<f32> = Vec::new();
    let mut tt_f1s:       Vec<f32> = Vec::new();
    let mut tt_all_preds: Vec<Vec<f32>> = Vec::new();
    let test_labels: Vec<f32> = test_data.iter().map(|x| x.2).collect();

    for &seed in &seeds {
        println!("  seed {}:", seed);
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
                let tf_e   = zero_expr(bsz);
                let gene_e = zero_expr(bsz);
                let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let grad: Array1<f32> = preds.iter().zip(labels.iter())
                    .map(|(&p, &l)| (p - l) / bsz as f32).collect();
                model.backward(&grad);
                adam_step_tt(&mut model, &mut state, lr, 5.0);
                model.zero_grad();
            }

            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                // Val evaluation with zeroed expression
                let mut val_preds: Vec<f32> = Vec::new();
                let mut val_labels: Vec<f32> = Vec::new();
                for start in (0..val_data.len()).step_by(batch_size) {
                    let end = (start + batch_size).min(val_data.len());
                    let batch = &val_data[start..end];
                    let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                    let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                    let lbls:     Vec<f32>   = batch.iter().map(|x| x.2).collect();
                    let bsz = batch.len();
                    let preds = model.forward(&tf_idx, &gene_idx, &zero_expr(bsz), &zero_expr(bsz));
                    val_preds.extend_from_slice(preds.as_slice().unwrap());
                    val_labels.extend_from_slice(&lbls);
                }
                let val_auroc = calculate_auroc(&val_preds, &val_labels);
                if val_auroc > best_val_auroc { best_val_auroc = val_auroc; patience = 0; }
                else { patience += 1; if patience >= 3 { break; } }
            }
        }

        // Test evaluation
        let mut test_preds:  Vec<f32> = Vec::new();
        for start in (0..test_data.len()).step_by(batch_size) {
            let end = (start + batch_size).min(test_data.len());
            let batch = &test_data[start..end];
            let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let bsz = batch.len();
            let preds = model.forward(&tf_idx, &gene_idx, &zero_expr(bsz), &zero_expr(bsz));
            test_preds.extend_from_slice(preds.as_slice().unwrap());
        }
        let auroc = calculate_auroc(&test_preds, &test_labels);
        let f1    = calculate_f1(&test_preds, &test_labels);
        let acc   = test_preds.iter().zip(test_labels.iter())
            .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / test_data.len() as f32;
        println!("    acc={:.2}% auroc={:.4} f1={:.4}", acc*100.0, auroc, f1);
        tt_aurocs.push(auroc); tt_accs.push(acc); tt_f1s.push(f1);
        tt_all_preds.push(test_preds);
    }

    // ── Cross-Encoder Embedding-Only ──────────────────────────────────────────
    println!("\n=== Cross-Encoder (embedding-only) ===");
    let mut ce_aurocs:    Vec<f32> = Vec::new();
    let mut ce_accs:      Vec<f32> = Vec::new();
    let mut ce_f1s:       Vec<f32> = Vec::new();

    for &seed in &seeds {
        println!("  seed {}:", seed);
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
                let bsz = batch.len();
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                let tf_e   = zero_expr(bsz);
                let gene_e = zero_expr(bsz);
                let logits     = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let labels_arr = Array1::from(labels);
                let grad       = bce_loss_backward(&logits, &labels_arr);
                model.backward(&grad);
                adam_step_ce(&mut model, &mut state, lr, 1.0);
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
                    let bsz = batch.len();
                    let logits = model.forward(&tf_idx, &gene_idx, &zero_expr(bsz), &zero_expr(bsz));
                    let probs: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
                    val_preds.extend_from_slice(&probs);
                    val_labels.extend_from_slice(&lbls);
                }
                let val_auroc = calculate_auroc(&val_preds, &val_labels);
                if val_auroc > best_val_auroc { best_val_auroc = val_auroc; patience = 0; }
                else { patience += 1; if patience >= 3 { break; } }
            }
        }

        let mut test_preds: Vec<f32> = Vec::new();
        for start in (0..test_data.len()).step_by(batch_size) {
            let end = (start + batch_size).min(test_data.len());
            let batch = &test_data[start..end];
            let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let bsz = batch.len();
            let logits = model.forward(&tf_idx, &gene_idx, &zero_expr(bsz), &zero_expr(bsz));
            test_preds.extend(logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())));
        }
        let auroc = calculate_auroc(&test_preds, &test_labels);
        let f1    = calculate_f1(&test_preds, &test_labels);
        let acc   = test_preds.iter().zip(test_labels.iter())
            .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / test_data.len() as f32;
        println!("    acc={:.2}% auroc={:.4} f1={:.4}", acc*100.0, auroc, f1);
        ce_aurocs.push(auroc); ce_accs.push(acc); ce_f1s.push(f1);
    }

    // Summary
    println!("\n=== Summary ===");
    println!("Two-Tower:    AUROC = {:.4} ± {:.4}", mean(&tt_aurocs), std_dev(&tt_aurocs));
    println!("Cross-Encoder: AUROC = {:.4} ± {:.4}", mean(&ce_aurocs), std_dev(&ce_aurocs));

    std::fs::create_dir_all("results")?;
    let output = serde_json::json!({
        "model": "embedding_only",
        "seeds": seeds,
        "two_tower": {
            "seed_aurocs":    tt_aurocs,
            "mean_auroc":     mean(&tt_aurocs),
            "std_auroc":      std_dev(&tt_aurocs),
            "mean_accuracy":  mean(&tt_accs),
            "mean_f1":        mean(&tt_f1s),
        },
        "cross_encoder": {
            "seed_aurocs":    ce_aurocs,
            "mean_auroc":     mean(&ce_aurocs),
            "std_auroc":      std_dev(&ce_aurocs),
            "mean_accuracy":  mean(&ce_accs),
            "mean_f1":        mean(&ce_f1s),
        },
        "note": "Expression features zeroed out. Delta from main results shows expression contribution.",
    });
    // Suppress unused variable warning for tf_expr_map/gene_expr_map
    let _ = tf_expr_map;
    let _ = gene_expr_map;
    std::fs::write("results/embedding_only_results.json", serde_json::to_string_pretty(&output)?)?;
    println!("✓ Saved results/embedding_only_results.json");
    Ok(())
}
