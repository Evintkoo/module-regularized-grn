/// Phase 8 Model D: Cross-Encoder training
/// Trains CrossEncoderModel at 1:1 and 5:1 negative ratios, 5 seeds each.
/// Writes results/cross_encoder_1to1.json and results/cross_encoder_5to1.json
use module_regularized_grn::{
    Config,
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

struct AdamState {
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

    fn adam2d(
        param: &mut Array2<f32>,
        grad:  &Array2<f32>,
        m: &mut Array2<f32>,
        v: &mut Array2<f32>,
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
        param: &mut Array1<f32>,
        grad:  &Array1<f32>,
        m: &mut Array1<f32>,
        v: &mut Array1<f32>,
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

    adam2d(&mut model.tf_embed,      &model.tf_embed_grad,    &mut state.m_tf,    &mut state.v_tf,    lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,    &model.gene_embed_grad,  &mut state.m_gene,  &mut state.v_gene,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc1.weights,   &model.fc1.grad_weights, &mut state.m_fc1_w, &mut state.v_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc1.bias,      &model.fc1.grad_bias,    &mut state.m_fc1_b, &mut state.v_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc2.weights,   &model.fc2.grad_weights, &mut state.m_fc2_w, &mut state.v_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc2.bias,      &model.fc2.grad_bias,    &mut state.m_fc2_b, &mut state.v_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc3.weights,   &model.fc3.grad_weights, &mut state.m_fc3_w, &mut state.v_fc3_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc3.bias,      &model.fc3.grad_bias,    &mut state.m_fc3_b, &mut state.v_fc3_b, lr, b1, b2, eps, state.t, scale);
}

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

/// Bootstrap CI over test-set predictions.
/// Returns (mean, lower_95, upper_95).
fn bootstrap_ci(y_true: &[f64], y_pred: &[f64], n: usize, seed: u64) -> (f64, f64, f64) {
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

fn evaluate_detailed(
    model:         &mut CrossEncoderModel,
    data:          &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim:      usize,
    batch_size:    usize,
) -> (f32, f32, f32) {
    let mut all_preds:  Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();

    for start in (0..data.len()).step_by(batch_size) {
        let end   = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,  tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);

        let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
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

fn main() -> Result<()> {
    println!("=== Phase 8: Cross-Encoder Training ===\n");

    let config    = Config::load_default()?;
    let priors    = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder   = PriorDatasetBuilder::new(priors.clone());
    let num_tfs   = builder.num_tfs();
    let num_genes = builder.num_genes();

    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    let expr_dim = expr_data.n_cell_types;

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

    let seeds      = [42u64, 123, 456, 789, 1337];
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
        let mut best_seed_final_val_acc = 0.0f32;
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
                    let tf_e   = build_expr_batch(&tf_idx,  &tf_expr_map,   expr_dim);
                    let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);

                    let logits     = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                    let labels_arr = Array1::from(labels);
                    let grad       = bce_loss_backward(&logits, &labels_arr);
                    model.backward(&grad);
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
                let tf_e   = build_expr_batch(&tf_idx,  &tf_expr_map,   expr_dim);
                let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                preds.extend(logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())));
            }

            // Select best seed by val accuracy
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_json_is_written() {
        let accs   = vec![0.80f32, 0.81];
        let aurocs = vec![0.81f32, 0.82];
        let f1s    = vec![0.82f32, 0.83];
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

        let s = serde_json::to_string(&result).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["model"].as_str().unwrap(), "cross_encoder");
        assert!(v["seed_accuracies"].is_array());
        assert!(v["bootstrap_ci_lower"].is_number());
    }
}
