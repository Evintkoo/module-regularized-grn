/// Standard MLP Training: Two-pathway feedforward network with embeddings + expression
/// Architecture: NOT a transformer - uses standard linear layers with ReLU
/// Similarity: Cosine similarity with temperature scaling (NOT attention)
use module_regularized_grn::{
    Config,
    models::{hybrid_embeddings::HybridEmbeddingModel, nn::{bce_loss, bce_loss_backward}},
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

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

fn bootstrap_ci(y_true: &[f64], y_pred: &[f64], n: usize, seed: u64) -> (f64, f64, f64) {
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

fn main() -> Result<()> {
    println!("=== Standard MLP Training (NOT Transformer) ===");
    println!("Architecture: Two-pathway feedforward network");
    println!("  - Pathway 1: TF encoder (embedding + expression → hidden → output)");
    println!("  - Pathway 2: Gene encoder (embedding + expression → hidden → output)");
    println!("  - Similarity: Cosine similarity with temperature scaling");
    println!("  - NO attention mechanism, NO transformer layers\n");

    let neg_ratio: usize = std::env::args()
        .position(|a| a == "--neg-ratio")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    println!("Negative ratio: {}:1", neg_ratio);

    let config = Config::load_default()?;
    let base_seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(base_seed);

    // Load prior knowledge
    println!("Loading prior knowledge...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();

    println!("  TFs: {} | Genes: {}", num_tfs, num_genes);

    // Load expression data
    println!("Loading expression data...");
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    println!("  Cell types: {} | Expression genes: {}",
             expr_data.n_cell_types, expr_data.n_genes);

    // Build gene name to expression index mapping
    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, gene_name) in expr_data.gene_names.iter().enumerate() {
        gene_to_expr_idx.insert(gene_name.clone(), i);
    }

    let tf_vocab   = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();

    println!("Mapping genes to expression features...");
    let expr_dim = expr_data.n_cell_types;

    let mut tf_expr_map:   HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();

    let mut tf_found   = 0;
    let mut gene_found = 0;

    for (tf_name, &tf_idx) in tf_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(tf_name) {
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            tf_expr_map.insert(tf_idx, expr_profile);
            tf_found += 1;
        } else {
            tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
        }
    }

    for (gene_name, &gene_idx) in gene_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(gene_name) {
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            gene_expr_map.insert(gene_idx, expr_profile);
            gene_found += 1;
        } else {
            gene_expr_map.insert(gene_idx, Array1::zeros(expr_dim));
        }
    }

    println!("  TFs with expression: {}/{} ({:.1}%)",
             tf_found, num_tfs, 100.0 * tf_found as f32 / num_tfs as f32);
    println!("  Genes with expression: {}/{} ({:.1}%)\n",
             gene_found, num_genes, 100.0 * gene_found as f32 / num_genes as f32);

    // Base 1:1 dataset — fixed splits
    println!("Creating dataset...");
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);

    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, gene)| (tf, gene, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, gene)| (tf, gene, 0.0)));
    examples.shuffle(&mut rng);

    let n_total = examples.len();
    let n_train = (n_total as f32 * 0.7) as usize;
    let n_val   = (n_total as f32 * 0.15) as usize;

    let train_base = examples[..n_train].to_vec();
    let val_data   = examples[n_train..n_train+n_val].to_vec();
    let test_data  = examples[n_train+n_val..].to_vec();

    println!("  Total: {} | Train: {} | Val: {} | Test: {}\n",
             n_total, train_base.len(), val_data.len(), test_data.len());

    // Build training data at the specified neg ratio
    let train_data: Vec<(usize, usize, f32)> = if neg_ratio == 1 {
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

    // Model hyperparameters (matched to cross-encoder for fair comparison)
    let embed_dim   = 512;
    let hidden_dim  = 512;
    let output_dim  = 512;
    let temperature = 0.05f32;

    println!("Model Architecture (Standard MLP / Two-Tower):");
    println!("  Input: Embedding ({}d) + Expression ({}d)", embed_dim, expr_dim);
    println!("  Hidden layer: {} dimensions with ReLU", hidden_dim);
    println!("  Output encoding: {} dimensions", output_dim);
    println!("  Similarity: Cosine with temperature={}", temperature);
    println!();

    // Training parameters
    let learning_rate: f32 = 0.005;
    let batch_size    = 256;
    let epochs        = 60;
    let weight_decay  = 0.01f32;

    println!("Training Configuration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", epochs);
    println!("  Weight decay (L2): {}", weight_decay);
    println!();

    let seeds = [42u64, 123, 456, 789, 1337];
    let mut seed_accuracies: Vec<f32> = Vec::new();
    let mut seed_aurocs:     Vec<f32> = Vec::new();
    let mut seed_f1s:        Vec<f32> = Vec::new();
    let mut all_test_preds_for_ensemble: Vec<Vec<f32>> = Vec::new();
    let mut best_final_val_acc = 0.0f32;
    let mut best_test_preds:    Vec<f32> = Vec::new();
    let test_labels_once: Vec<f32> = test_data.iter().map(|x| x.2).collect();

    // Per-seed storage for backward-compat single-seed result (use first seed)
    let mut compat_accuracy  = 0.0f32;
    let mut compat_precision = 0.0f32;
    let mut compat_recall    = 0.0f32;
    let mut compat_f1        = 0.0f32;
    let mut compat_best_val  = f32::INFINITY;

    for (seed_idx, &seed) in seeds.iter().enumerate() {
        println!("=== Seed {} ({}/5) ===", seed, seed_idx + 1);

        let mut model = HybridEmbeddingModel::new(
            num_tfs,
            num_genes,
            embed_dim,
            expr_dim,
            hidden_dim,
            output_dim,
            temperature,
            0.1,  // std_dev for init
            seed,
        );

        let mut best_val_loss    = f32::INFINITY;
        let mut best_seed_val_acc = 0.0f32;
        let mut patience_counter = 0;
        let patience             = 10;

        let mut train_data_shuffled = train_data.clone();

        for epoch in 0..epochs {
            // Shuffle training data each epoch
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            train_data_shuffled.shuffle(&mut epoch_rng);

            let mut train_loss    = 0.0f32;
            let mut train_batches = 0;

            for batch_start in (0..train_data_shuffled.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(train_data_shuffled.len());
                let batch = &train_data_shuffled[batch_start..batch_end];

                let tf_indices:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:       Vec<f32>   = batch.iter().map(|x| x.2).collect();

                let tf_expr_batch   = build_expr_batch(&tf_indices,   &tf_expr_map,   expr_dim);
                let gene_expr_batch = build_expr_batch(&gene_indices, &gene_expr_map, expr_dim);

                let predictions = model.forward(
                    &tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch,
                );

                let labels_arr    = Array1::from(labels);
                let predictions_2d = predictions.clone().insert_axis(ndarray::Axis(1));
                let loss   = bce_loss(&predictions_2d, &labels_arr);
                let grad_2d = bce_loss_backward(&predictions_2d, &labels_arr);
                let grad   = grad_2d.column(0).to_owned();

                model.backward(&grad);
                train_loss    += loss;
                train_batches += 1;
            }

            // Update parameters after all batches (existing convention)
            model.update_with_weight_decay(learning_rate, weight_decay);
            model.zero_grad();

            let avg_train_loss = train_loss / train_batches as f32;

            // Validation phase
            let mut val_loss    = 0.0f32;
            let mut val_correct = 0;
            let mut val_total   = 0;

            for batch_start in (0..val_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(val_data.len());
                let batch = &val_data[batch_start..batch_end];

                let tf_indices:   Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels:       Vec<f32>   = batch.iter().map(|x| x.2).collect();

                let tf_expr_batch   = build_expr_batch(&tf_indices,   &tf_expr_map,   expr_dim);
                let gene_expr_batch = build_expr_batch(&gene_indices, &gene_expr_map, expr_dim);

                let predictions = model.forward(
                    &tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch,
                );

                let labels_arr    = Array1::from(labels.clone());
                let predictions_2d = predictions.clone().insert_axis(ndarray::Axis(1));
                val_loss += bce_loss(&predictions_2d, &labels_arr);

                for (pred, label) in predictions.iter().zip(labels.iter()) {
                    if (*pred >= 0.5) == (*label == 1.0) { val_correct += 1; }
                    val_total += 1;
                }
            }

            let avg_val_loss = val_loss / ((val_data.len() as f32 / batch_size as f32).ceil());
            let val_acc = val_correct as f32 / val_total as f32;

            if val_acc > best_seed_val_acc { best_seed_val_acc = val_acc; }

            if epoch % 10 == 9 || epoch == epochs - 1 {
                println!("  Epoch {:2} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.2}%",
                         epoch + 1, avg_train_loss, avg_val_loss, val_acc * 100.0);
            }

            if avg_val_loss < best_val_loss {
                best_val_loss    = avg_val_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience {
                    println!("  Early stopping at epoch {}", epoch + 1);
                    break;
                }
            }
        }

        // Test evaluation — collect all predictions
        let mut all_test_preds:  Vec<f32> = Vec::new();
        let mut all_test_labels: Vec<f32> = Vec::new();

        for batch_start in (0..test_data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(test_data.len());
            let batch = &test_data[batch_start..batch_end];

            let tf_indices:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let labels:       Vec<f32>   = batch.iter().map(|x| x.2).collect();

            let tf_expr_batch   = build_expr_batch(&tf_indices,   &tf_expr_map,   expr_dim);
            let gene_expr_batch = build_expr_batch(&gene_indices, &gene_expr_map, expr_dim);

            let predictions = model.forward(
                &tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch,
            );

            all_test_preds.extend_from_slice(predictions.as_slice().unwrap());
            all_test_labels.extend_from_slice(&labels);
        }

        let test_acc   = all_test_preds.iter().zip(all_test_labels.iter())
            .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0))
            .count() as f32 / all_test_preds.len() as f32;
        let test_auroc = calculate_auroc(&all_test_preds, &all_test_labels);
        let test_f1    = calculate_f1(&all_test_preds, &all_test_labels);

        println!("  acc={:.2}% auroc={:.4} f1={:.4}\n", test_acc*100.0, test_auroc, test_f1);

        // Track best seed by val accuracy for bootstrap CI
        if best_seed_val_acc > best_final_val_acc {
            best_final_val_acc = best_seed_val_acc;
            best_test_preds    = all_test_preds.clone();
        }
        all_test_preds_for_ensemble.push(all_test_preds);

        seed_accuracies.push(test_acc);
        seed_aurocs.push(test_auroc);
        seed_f1s.push(test_f1);

        // Backward-compat: save first-seed metrics
        if seed_idx == 0 {
            let tp: usize = seed_accuracies[0].ceil() as usize;  // placeholder
            let _  = tp;  // unused intentionally
            compat_accuracy  = test_acc;
            compat_f1        = test_f1;
            compat_precision = {
                let mut tpp = 0.0f32; let mut fpp = 0.0f32;
                for (&p, &l) in best_test_preds.iter().zip(test_labels_once.iter()) {
                    if p >= 0.5 && l == 1.0 { tpp += 1.0; }
                    if p >= 0.5 && l == 0.0 { fpp += 1.0; }
                }
                if tpp + fpp > 0.0 { tpp / (tpp + fpp) } else { 0.0 }
            };
            compat_recall = {
                let mut tpr = 0.0f32; let mut fnr = 0.0f32;
                for (&p, &l) in best_test_preds.iter().zip(test_labels_once.iter()) {
                    if p >= 0.5 && l == 1.0 { tpr += 1.0; }
                    if p < 0.5  && l == 1.0 { fnr += 1.0; }
                }
                if tpr + fnr > 0.0 { tpr / (tpr + fnr) } else { 0.0 }
            };
            compat_best_val = best_val_loss;
        }
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

    println!("=== Summary ===");
    println!("Mean accuracy: {:.2}% ± {:.2}%", mean_acc*100.0, std_acc*100.0);
    println!("Ensemble accuracy: {:.2}%", ensemble_acc*100.0);

    // Write new unified schema
    std::fs::create_dir_all("results")?;
    let out_file = if neg_ratio == 1 { "results/two_tower_1to1.json" } else { "results/two_tower_5to1.json" };
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

    // Backward-compat: keep existing standard_mlp_results.json (first-seed result)
    let compat_results = serde_json::json!({
        "architecture": "Standard MLP (Two-pathway)",
        "note": "NOT a transformer - uses feedforward layers only",
        "model": {
            "embed_dim": embed_dim,
            "expr_dim": expr_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "temperature": temperature,
        },
        "metrics": {
            "accuracy": compat_accuracy,
            "precision": compat_precision,
            "recall": compat_recall,
            "f1_score": compat_f1,
        },
        "training": {
            "epochs_completed": epochs,
            "best_val_loss": compat_best_val,
        }
    });
    std::fs::write(
        "results/standard_mlp_results.json",
        serde_json::to_string_pretty(&compat_results)?
    )?;
    println!("✓ Saved results/standard_mlp_results.json");

    println!("\n=== Training Complete ===");

    Ok(())
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
