/// Standard MLP Training: Two-pathway feedforward network with embeddings + expression
/// Architecture: NOT a transformer - uses standard linear layers with ReLU
/// Similarity: Cosine similarity with temperature scaling (NOT attention)
/// Optimizer: Per-batch Adam with stable gradient (p - l) / batch_size
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

// ── Adam state for HybridEmbeddingModel ──────────────────────────────────────

struct AdamState {
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
    let b1  = 0.9f32;
    let b2  = 0.999f32;
    let eps = 1e-8f32;

    // Global gradient clipping
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

// ── Metric helpers ────────────────────────────────────────────────────────────

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
    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut ap = 0.0f32;
    let mut prev_recall = 0.0f32;
    for &(_, label) in &pairs {
        if label == 1.0 { tp += 1.0; } else { fp += 1.0; }
        let recall    = tp / n_pos;
        let precision = tp / (tp + fp);
        ap += precision * (recall - prev_recall);
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

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== Standard MLP Training (NOT Transformer) ===");
    println!("Architecture: Two-pathway feedforward network");
    println!("  - Pathway 1: TF encoder (embedding + expression → hidden → output)");
    println!("  - Pathway 2: Gene encoder (embedding + expression → hidden → output)");
    println!("  - Similarity: Cosine similarity with temperature scaling");
    println!("  - Optimizer: Epoch-level SGD with weight decay + dropout\n");

    let neg_ratio: usize = std::env::args()
        .position(|a| a == "--neg-ratio")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    println!("Negative ratio: {}:1", neg_ratio);

    let config    = Config::load_default()?;
    let base_seed = config.project.seed;
    let mut rng   = StdRng::seed_from_u64(base_seed);

    println!("Loading prior knowledge...");
    let priors  = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs   = builder.num_tfs();
    let num_genes = builder.num_genes();
    println!("  TFs: {} | Genes: {}", num_tfs, num_genes);

    println!("Loading expression data...");
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    println!("  Cell types: {} | Expression genes: {}",
             expr_data.n_cell_types, expr_data.n_genes);

    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in expr_data.gene_names.iter().enumerate() {
        gene_to_expr_idx.insert(name.clone(), i);
    }

    let tf_vocab   = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();
    let expr_dim   = expr_data.n_cell_types;

    let mut tf_expr_map:   HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut tf_found = 0usize; let mut gene_found = 0usize;

    for (name, &idx) in tf_vocab.iter() {
        if let Some(&ei) = gene_to_expr_idx.get(name) {
            tf_expr_map.insert(idx, expr_data.expression.column(ei).to_owned());
            tf_found += 1;
        } else {
            tf_expr_map.insert(idx, Array1::zeros(expr_dim));
        }
    }
    for (name, &idx) in gene_vocab.iter() {
        if let Some(&ei) = gene_to_expr_idx.get(name) {
            gene_expr_map.insert(idx, expr_data.expression.column(ei).to_owned());
            gene_found += 1;
        } else {
            gene_expr_map.insert(idx, Array1::zeros(expr_dim));
        }
    }
    println!("  TFs with expression: {}/{} ({:.1}%)",
             tf_found, num_tfs, 100.0 * tf_found as f32 / num_tfs as f32);
    println!("  Genes with expression: {}/{} ({:.1}%)\n",
             gene_found, num_genes, 100.0 * gene_found as f32 / num_genes as f32);

    println!("Creating dataset...");
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, g)| (tf, g, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, g)| (tf, g, 0.0)));
    examples.shuffle(&mut rng);

    let n_total = examples.len();
    let n_train = (n_total as f32 * 0.7) as usize;
    let n_val   = (n_total as f32 * 0.15) as usize;
    let train_base = examples[..n_train].to_vec();
    let val_data   = examples[n_train..n_train+n_val].to_vec();
    let test_data  = examples[n_train+n_val..].to_vec();
    println!("  Total: {} | Train: {} | Val: {} | Test: {}\n",
             n_total, train_base.len(), val_data.len(), test_data.len());

    // Build training data at neg_ratio
    let train_data: Vec<(usize, usize, f32)> = if neg_ratio == 1 {
        train_base.clone()
    } else {
        let pos_train: Vec<_> = train_base.iter().filter(|x| x.2 == 1.0).cloned().collect();
        let neg_extra = builder.sample_negative_examples(
            pos_train.len() * neg_ratio, base_seed + neg_ratio as u64
        );
        let mut data: Vec<(usize, usize, f32)> = pos_train;
        data.extend(neg_extra.iter().map(|&(tf, g)| (tf, g, 0.0f32)));
        let mut r = StdRng::seed_from_u64(base_seed + 1000);
        data.shuffle(&mut r);
        data
    };

    // Model configuration: per-batch Adam (matches cross-encoder approach)
    let embed_dim   = 512usize;
    let hidden_dim  = 512usize;
    let output_dim  = 512usize;
    let temperature = 0.05f32;
    let lr          = 0.001f32;
    let clip        = 5.0f32;
    let batch_size  = 256usize;
    let epochs      = 60usize;
    let patience    = 10usize;

    println!("Model Architecture (Standard MLP / Two-Tower):");
    println!("  Input: Embedding ({}d) + Expression ({}d)", embed_dim, expr_dim);
    println!("  Hidden layer: {} dimensions with ReLU", hidden_dim);
    println!("  Output encoding: {} dimensions", output_dim);
    println!("  Similarity: Cosine with temperature={}", temperature);
    println!("\nTraining Configuration:");
    println!("  Optimizer: Adam lr={} | grad_clip={}", lr, clip);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", epochs);

    let seeds = [42u64, 123, 456, 789, 1337];
    let mut seed_accuracies: Vec<f32>       = Vec::new();
    let mut seed_aurocs:     Vec<f32>       = Vec::new();
    let mut seed_f1s:        Vec<f32>       = Vec::new();
    let mut seed_auprcs:     Vec<f32>       = Vec::new();
    let mut all_test_preds_ensemble: Vec<Vec<f32>> = Vec::new();
    let mut best_final_val_acc = 0.0f32;
    let mut best_test_preds:   Vec<f32>     = Vec::new();
    let test_labels_once: Vec<f32> = test_data.iter().map(|x| x.2).collect();

    for (seed_idx, &seed) in seeds.iter().enumerate() {
        println!("\n=== Seed {} ({}/{}) ===", seed, seed_idx + 1, seeds.len());

        let mut model = HybridEmbeddingModel::new(
            num_tfs, num_genes, embed_dim, expr_dim,
            hidden_dim, output_dim, temperature, 0.01, seed,
        );
        let mut state = AdamState::new(&model);
        let mut best_seed_val_acc = 0.0f32;
        let mut patience_counter  = 0usize;
        let mut train_shuffled    = train_data.clone();

        for epoch in 0..epochs {
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            train_shuffled.shuffle(&mut epoch_rng);

            let mut epoch_train_loss = 0.0f32;
            let mut epoch_n = 0usize;

            for start in (0..train_shuffled.len()).step_by(batch_size) {
                let end   = (start + batch_size).min(train_shuffled.len());
                let batch = &train_shuffled[start..end];
                let bsz   = batch.len();
                let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
                let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);

                let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);

                // Stable gradient: (p - l) / batch_size
                // model.backward multiplies by sigmoid'(logit)/temp internally.
                // Using (p-l) avoids 1/(p*(1-p)) blow-up for confident predictions.
                let grad: Array1<f32> = preds.iter().zip(labels.iter())
                    .map(|(&p, &l)| (p - l) / bsz as f32)
                    .collect();

                // Track approximate BCE loss for logging
                for (&p, &l) in preds.iter().zip(labels.iter()) {
                    let pc = p.clamp(1e-7, 1.0 - 1e-7);
                    epoch_train_loss += -(l * pc.ln() + (1.0 - l) * (1.0 - pc).ln());
                    epoch_n += 1;
                }

                model.backward(&grad);
                adam_step(&mut model, &mut state, lr, clip);
                model.zero_grad();
            }

            // Validation every 10 epochs
            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                let (val_loss, val_acc) = eval_loss_acc(
                    &mut model, &val_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
                );
                let train_loss_avg = epoch_train_loss / epoch_n as f32;
                println!("  Epoch {:2} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.2}%",
                         epoch + 1, train_loss_avg, val_loss, val_acc * 100.0);

                if val_acc > best_seed_val_acc {
                    best_seed_val_acc = val_acc;
                    patience_counter  = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience { break; }
                }
            }
        }

        // Test evaluation
        let (test_acc, test_auroc, test_f1, test_auprc) = evaluate_detailed(
            &mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size
        );
        println!("  acc={:.2}% auroc={:.4} f1={:.4} auprc={:.4}", test_acc*100.0, test_auroc, test_f1, test_auprc);

        // Collect test predictions for ensemble
        let mut preds: Vec<f32> = Vec::new();
        for start in (0..test_data.len()).step_by(batch_size) {
            let end   = (start + batch_size).min(test_data.len());
            let batch = &test_data[start..end];
            let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let tf_e   = build_expr_batch(&tf_idx,   &tf_expr_map,   expr_dim);
            let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
            let p = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
            preds.extend_from_slice(p.as_slice().unwrap());
        }

        if best_seed_val_acc > best_final_val_acc {
            best_final_val_acc = best_seed_val_acc;
            best_test_preds    = preds.clone();
        }
        all_test_preds_ensemble.push(preds);
        seed_accuracies.push(test_acc);
        seed_aurocs.push(test_auroc);
        seed_f1s.push(test_f1);
        seed_auprcs.push(test_auprc);
    }

    // Ensemble accuracy
    let n_test = test_data.len();
    let ensemble_preds: Vec<f32> = (0..n_test)
        .map(|i| all_test_preds_ensemble.iter().map(|p| p[i]).sum::<f32>() / seeds.len() as f32)
        .collect();
    let ensemble_acc = ensemble_preds.iter().zip(test_labels_once.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0))
        .count() as f32 / n_test as f32;
    let ensemble_auroc = calculate_auroc(&ensemble_preds, &test_labels_once);
    let ensemble_auprc = calculate_auprc(&ensemble_preds, &test_labels_once);

    let bp: Vec<f64> = best_test_preds.iter().map(|&x| x as f64).collect();
    let bl: Vec<f64> = test_labels_once.iter().map(|&x| x as f64).collect();
    let (_, ci_lower, ci_upper) = bootstrap_ci(&bl, &bp, 1000, 42);

    let mean_acc = seed_accuracies.iter().sum::<f32>() / seeds.len() as f32;
    let std_acc  = (seed_accuracies.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>()
                    / seeds.len() as f32).sqrt();

    println!("\n=== Summary ===");
    println!("Mean accuracy: {:.2}% ± {:.2}%", mean_acc*100.0, std_acc*100.0);
    println!("Ensemble accuracy: {:.2}%", ensemble_acc*100.0);

    std::fs::create_dir_all("results")?;
    let out_file = if neg_ratio == 1 { "results/two_tower_1to1.json" } else { "results/two_tower_5to1.json" };
    let result = serde_json::json!({
        "model": "two_tower",
        "neg_ratio": neg_ratio,
        "seeds": [42u64, 123, 456, 789, 1337],
        "seed_accuracies": seed_accuracies,
        "seed_aurocs": seed_aurocs,
        "seed_f1s": seed_f1s,
        "seed_auprcs": seed_auprcs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "ensemble_accuracy": ensemble_acc,
        "ensemble_auroc": ensemble_auroc,
        "ensemble_auprc": ensemble_auprc,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
    });
    std::fs::write(out_file, serde_json::to_string_pretty(&result)?)?;
    println!("✓ Saved {}", out_file);

    // Save raw predictions for ROC/PR curve generation
    let pred_file = if neg_ratio == 1 {
        "results/two_tower_1to1_predictions.json"
    } else {
        "results/two_tower_5to1_predictions.json"
    };
    let pred_json = serde_json::json!({
        "labels": test_labels_once.iter().map(|&x| x as f64).collect::<Vec<_>>(),
        "predictions": best_test_preds.iter().map(|&x| x as f64).collect::<Vec<_>>(),
    });
    std::fs::write(pred_file, serde_json::to_string_pretty(&pred_json)?)?;
    println!("✓ Saved {}", pred_file);

    // Backward-compat single-seed result
    let compat = serde_json::json!({
        "architecture": "Standard MLP (Two-pathway)",
        "note": "NOT a transformer - uses feedforward layers only",
        "model": { "embed_dim": embed_dim, "expr_dim": expr_dim,
                   "hidden_dim": hidden_dim, "output_dim": output_dim, "temperature": temperature },
        "metrics": { "accuracy": seed_accuracies[0], "f1_score": seed_f1s[0] },
        "training": { "epochs_completed": epochs }
    });
    std::fs::write("results/standard_mlp_results.json", serde_json::to_string_pretty(&compat)?)?;
    println!("✓ Saved results/standard_mlp_results.json");
    println!("\n=== Training Complete ===");
    Ok(())
}

fn eval_loss_acc(
    model:         &mut HybridEmbeddingModel,
    data:          &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim:      usize,
    batch_size:    usize,
) -> (f32, f32) {
    let mut total_loss = 0.0f32;
    let mut n_correct  = 0usize;
    let mut n_total    = 0usize;
    for start in (0..data.len()).step_by(batch_size) {
        let end   = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        for (&p, &l) in preds.iter().zip(labels.iter()) {
            let p_c = p.clamp(1e-7, 1.0 - 1e-7);
            total_loss += -(l * p_c.ln() + (1.0 - l) * (1.0 - p_c).ln());
            if (p >= 0.5) == (l == 1.0) { n_correct += 1; }
            n_total += 1;
        }
    }
    (total_loss / n_total as f32, n_correct as f32 / n_total as f32)
}

fn evaluate_detailed(
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
    let accuracy = n_correct as f32 / all_preds.len() as f32;
    let auroc    = calculate_auroc(&all_preds, &all_labels);
    let f1       = calculate_f1(&all_preds, &all_labels);
    let auprc    = calculate_auprc(&all_preds, &all_labels);
    (accuracy, auroc, f1, auprc)
}
