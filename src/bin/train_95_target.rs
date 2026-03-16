/// Train model targeting 95% accuracy
/// Uses: Large ensemble, better hyperparameters, advanced techniques
use module_regularized_grn::{
    Config,
    models::{hybrid_embeddings::HybridEmbeddingModel, nn::bce_loss_backward},
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2, Axis};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("      PUSHING TO 95% ACCURACY - ADVANCED TRAINING PIPELINE");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let config = Config::load_default()?;
    let base_seed = config.project.seed;
    
    // STABLE configuration for 95% - based on best performing setup
    let embed_dim = 512;        // Same as best config
    let hidden_dim = 2048;      // Same as best config  
    let output_dim = 512;       // Same as best config
    let initial_lr = 0.005;     // Conservative LR (same as best)
    let min_lr = 0.0005;        // Minimum LR
    let batch_size = 256;       // Same as best config
    let epochs = 80;            // More epochs
    let dropout_rate = 0.25;    // Moderate dropout
    let weight_decay = 0.01;    // Same as best config
    let num_models = 10;        // Larger ensemble for variance reduction
    let label_smoothing = 0.05; // Light label smoothing
    let warmup_epochs = 5;      // Shorter warmup
    
    println!("Configuration for 95% target:");
    println!("  Embedding dim: {}", embed_dim);
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Output dim: {}", output_dim);
    println!("  Initial LR: {} → {}", initial_lr, min_lr);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {} (with {} warmup)", epochs, warmup_epochs);
    println!("  Dropout: {}", dropout_rate);
    println!("  Weight decay: {}", weight_decay);
    println!("  Label smoothing: {}", label_smoothing);
    println!("  Ensemble size: {} models", num_models);
    println!();
    
    // Load data
    println!("Loading data...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    
    println!("  TFs: {} | Genes: {}", num_tfs, num_genes);
    println!("  Expression: {} cell types, {} genes\n", 
             expr_data.n_cell_types, expr_data.gene_names.len());
    
    // Build expression maps with feature augmentation
    let (tf_expr_map, gene_expr_map, expr_dim) = build_expression_maps(
        &builder,
        &expr_data,
        num_tfs,
        num_genes,
    );
    
    // Create dataset with augmentation
    println!("Creating dataset with augmentation...");
    let mut rng = StdRng::seed_from_u64(base_seed);
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);
    
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, gene)| (tf, gene, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, gene)| (tf, gene, 0.0)));
    examples.shuffle(&mut rng);
    
    let n_total = examples.len();
    let n_train = (n_total as f32 * 0.7) as usize;
    let n_val = (n_total as f32 * 0.15) as usize;
    
    let train_data = examples[..n_train].to_vec();
    let val_data = examples[n_train..n_train+n_val].to_vec();
    let test_data = examples[n_train+n_val..].to_vec();
    
    println!("  Train: {} | Val: {} | Test: {}\n", 
             train_data.len(), val_data.len(), test_data.len());
    
    // Train ensemble
    println!("═══════════════════════════════════════════════════════════════════");
    println!("Training {} models with cosine annealing + warmup...\n", num_models);
    
    let mut models = Vec::new();
    let mut individual_results = Vec::new();
    let total_start = Instant::now();
    
    for i in 0..num_models {
        let seed = base_seed + i as u64 * 1000;
        println!("═══════════════════════════════════════════════════════════════════");
        println!("Model {}/{} (seed={})", i + 1, num_models, seed);
        println!("═══════════════════════════════════════════════════════════════════");
        
        let start = Instant::now();
        
        let mut model = HybridEmbeddingModel::new(
            num_tfs,
            num_genes,
            embed_dim,
            expr_dim,
            hidden_dim,
            output_dim,
            0.03,   // Lower temperature for sharper predictions
            0.05,   // Lower init std for stability
            seed,
        );
        
        let params = model.num_parameters();
        println!("  Parameters: {:.2}M", params as f64 / 1_000_000.0);
        
        // Training with all optimizations
        let mut best_val_acc = 0.0;
        let mut best_epoch = 0;
        let mut patience_counter = 0;
        let patience = 15;
        
        for epoch in 0..epochs {
            // Cosine annealing with warmup
            let lr = if epoch < warmup_epochs {
                // Linear warmup
                initial_lr * (epoch as f32 + 1.0) / warmup_epochs as f32
            } else {
                // Cosine decay
                let progress = (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs) as f32;
                min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + (PI * progress).cos())
            };
            
            // Shuffle training data each epoch
            let mut epoch_data = train_data.clone();
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            epoch_data.shuffle(&mut epoch_rng);
            
            let mut epoch_loss = 0.0;
            let mut n_batches = 0;
            
            // Training loop
            for batch_start in (0..epoch_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(epoch_data.len());
                let batch = &epoch_data[batch_start..batch_end];
                
                let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
                
                // Apply label smoothing
                let labels: Vec<f32> = batch.iter().map(|x| {
                    if x.2 == 1.0 {
                        1.0 - label_smoothing
                    } else {
                        label_smoothing
                    }
                }).collect();
                
                // Apply feature dropout for augmentation
                let tf_expr_batch = build_expr_batch_with_dropout(
                    &tf_indices, &tf_expr_map, expr_dim, 
                    dropout_rate * 0.5, seed + epoch as u64 + batch_start as u64
                );
                let gene_expr_batch = build_expr_batch_with_dropout(
                    &gene_indices, &gene_expr_map, expr_dim,
                    dropout_rate * 0.5, seed + epoch as u64 + batch_start as u64 + 1
                );
                
                // Forward pass
                let predictions = model.forward(
                    &tf_indices, &gene_indices, 
                    &tf_expr_batch, &gene_expr_batch
                );
                
                // Compute BCE loss
                let labels_arr = Array1::from(labels.clone());
                let predictions_2d = predictions.clone().insert_axis(Axis(1));
                
                // Loss computation
                let loss: f32 = predictions.iter().zip(labels.iter())
                    .map(|(p, l)| {
                        let p_clamped = p.max(1e-7).min(1.0 - 1e-7);
                        -(l * p_clamped.ln() + (1.0 - l) * (1.0 - p_clamped).ln())
                    })
                    .sum::<f32>() / predictions.len() as f32;
                
                epoch_loss += loss;
                n_batches += 1;
                
                // Backward pass
                let grad_2d = bce_loss_backward(&predictions_2d, &labels_arr);
                let grad = grad_2d.column(0).to_owned();
                
                // Apply gradient dropout for regularization
                let grad_with_dropout = apply_dropout(&grad, dropout_rate * 0.3, seed + epoch as u64);
                
                model.backward(&grad_with_dropout);
            }
            
            // Update with weight decay
            model.update_with_weight_decay(lr, weight_decay);
            model.zero_grad();
            
            // Validation
            let val_acc = evaluate(&mut model, &val_data, &tf_expr_map, 
                                  &gene_expr_map, expr_dim, batch_size);
            
            // Progress logging
            if epoch % 10 == 0 || val_acc > best_val_acc {
                let avg_loss = epoch_loss / n_batches as f32;
                println!("  Epoch {:3} | LR: {:.5} | Loss: {:.4} | Val Acc: {:.2}%{}",
                         epoch, lr, avg_loss, val_acc * 100.0,
                         if val_acc > best_val_acc { " ✓" } else { "" });
            }
            
            // Track best
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = epoch;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience && epoch > 30 {
                    println!("  Early stopping at epoch {} (best: epoch {})", epoch, best_epoch);
                    break;
                }
            }
        }
        
        // Test evaluation
        let (test_acc, test_prec, test_rec, test_f1, test_auroc, test_auprc) = evaluate_detailed(
            &mut model,
            &test_data,
            &tf_expr_map,
            &gene_expr_map,
            expr_dim,
            batch_size,
        );
        
        let elapsed = start.elapsed();
        
        println!("\n  ✓ Best Val Acc: {:.2}% (epoch {})", best_val_acc * 100.0, best_epoch);
        println!("  ✓ Test Accuracy: {:.2}%", test_acc * 100.0);
        println!("  ✓ Test Precision: {:.2}%", test_prec * 100.0);
        println!("  ✓ Test Recall: {:.2}%", test_rec * 100.0);
        println!("  ✓ Test AUROC: {:.4}", test_auroc);
        println!("  Training time: {:.1} minutes\n", elapsed.as_secs() as f64 / 60.0);
        
        individual_results.push((test_acc, test_prec, test_rec, test_f1, test_auroc, test_auprc));
        models.push(model);
    }
    
    // Ensemble evaluation
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                      ENSEMBLE RESULTS");
    println!("═══════════════════════════════════════════════════════════════════\n");
    
    // Individual model summary
    println!("Individual Models:");
    for (i, (acc, prec, rec, f1, auroc, _)) in individual_results.iter().enumerate() {
        println!("  Model {}: Acc={:.2}% | Prec={:.2}% | Rec={:.2}% | F1={:.4} | AUROC={:.4}",
                 i + 1, acc * 100.0, prec * 100.0, rec * 100.0, f1, auroc);
    }
    
    let mean_acc = individual_results.iter().map(|x| x.0).sum::<f32>() / num_models as f32;
    let std_acc = (individual_results.iter()
        .map(|x| (x.0 - mean_acc).powi(2))
        .sum::<f32>() / num_models as f32)
        .sqrt();
    
    let max_individual = individual_results.iter().map(|x| x.0).fold(0.0f32, |a, b| a.max(b));
    
    println!("\n  Mean: {:.2}% ± {:.2}%", mean_acc * 100.0, std_acc * 100.0);
    println!("  Best individual: {:.2}%", max_individual * 100.0);
    
    // Ensemble predictions (with weighted averaging based on validation performance)
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("ENSEMBLE (Average of {} models):", num_models);
    println!("═══════════════════════════════════════════════════════════════════\n");
    
    let (ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1, 
         ensemble_auroc, ensemble_auprc) = evaluate_ensemble(
        &mut models,
        &test_data,
        &tf_expr_map,
        &gene_expr_map,
        expr_dim,
        batch_size,
    );
    
    println!("🏆 ENSEMBLE PERFORMANCE:");
    println!("  Test Accuracy:  {:.2}%", ensemble_acc * 100.0);
    println!("  Test Precision: {:.2}%", ensemble_prec * 100.0);
    println!("  Test Recall:    {:.2}%", ensemble_rec * 100.0);
    println!("  Test F1-Score:  {:.4}", ensemble_f1);
    println!("  Test AUROC:     {:.4}", ensemble_auroc);
    println!("  Test AUPRC:     {:.4}", ensemble_auprc);
    
    // Final summary
    let total_elapsed = total_start.elapsed();
    
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("                        FINAL SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════\n");
    
    println!("Previous best (83.06%):  {:.2}%", 83.06);
    println!("Current ensemble:        {:.2}%", ensemble_acc * 100.0);
    println!("Improvement:             {:+.2}%", (ensemble_acc * 100.0) - 83.06);
    println!("\nGap to 95%:              {:.2}%", 95.0 - ensemble_acc * 100.0);
    println!("Total training time:     {:.1} minutes", total_elapsed.as_secs() as f64 / 60.0);
    
    if ensemble_acc >= 0.95 {
        println!("\n🎉🎉🎉 TARGET ACHIEVED! 95%+ ACCURACY! 🎉🎉🎉");
    } else if ensemble_acc >= 0.90 {
        println!("\n✅ EXCELLENT! 90%+ achieved - publication quality!");
    } else if ensemble_acc >= 0.85 {
        println!("\n✅ VERY GOOD! 85%+ achieved!");
    }
    
    // Save results
    let results_json = serde_json::json!({
        "experiment": "target_95_percent",
        "date": chrono::Utc::now().to_rfc3339(),
        "configuration": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "initial_lr": initial_lr,
            "min_lr": min_lr,
            "dropout_rate": dropout_rate,
            "weight_decay": weight_decay,
            "label_smoothing": label_smoothing,
            "num_models": num_models,
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "batch_size": batch_size,
        },
        "ensemble_results": {
            "test_accuracy": ensemble_acc,
            "test_precision": ensemble_prec,
            "test_recall": ensemble_rec,
            "test_f1": ensemble_f1,
            "test_auroc": ensemble_auroc,
            "test_auprc": ensemble_auprc,
        },
        "individual_stats": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "max_individual": max_individual,
        },
        "training_time_minutes": total_elapsed.as_secs() as f64 / 60.0,
    });
    
    std::fs::write(
        "results/target_95_results.json",
        serde_json::to_string_pretty(&results_json)?
    )?;
    
    println!("\n✓ Results saved to: results/target_95_results.json");
    
    Ok(())
}

fn apply_dropout(grad: &Array1<f32>, dropout_rate: f32, seed: u64) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let keep_prob = 1.0 - dropout_rate;
    let scale = 1.0 / keep_prob;
    
    grad.mapv(|g| {
        if rand::Rng::gen::<f32>(&mut rng) < keep_prob {
            g * scale
        } else {
            0.0
        }
    })
}

fn evaluate(
    model: &mut HybridEmbeddingModel,
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
    batch_size: usize,
) -> f32 {
    let mut correct = 0;
    let mut total = 0;
    
    for batch_start in (0..data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(data.len());
        let batch = &data[batch_start..batch_end];
        
        let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
        
        let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
        let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);
        
        let predictions = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
        
        for (pred, label) in predictions.iter().zip(labels.iter()) {
            if (*pred >= 0.5 && *label == 1.0) || (*pred < 0.5 && *label == 0.0) {
                correct += 1;
            }
            total += 1;
        }
    }
    
    correct as f32 / total as f32
}

fn evaluate_detailed(
    model: &mut HybridEmbeddingModel,
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
    batch_size: usize,
) -> (f32, f32, f32, f32, f32, f32) {
    let mut all_predictions = Vec::new();
    let mut all_labels = Vec::new();
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    
    for batch_start in (0..data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(data.len());
        let batch = &data[batch_start..batch_end];
        
        let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
        
        let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
        let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);
        
        let predictions = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
        
        for (pred, label) in predictions.iter().zip(labels.iter()) {
            all_predictions.push(*pred);
            all_labels.push(*label);
            
            let pred_class = if *pred >= 0.5 { 1.0 } else { 0.0 };
            match (pred_class as i32, *label as i32) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fp += 1,
                (0, 1) => fn_count += 1,
                _ => {}
            }
        }
    }
    
    let accuracy = (tp + tn) as f32 / (tp + tn + fp + fn_count) as f32;
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 
        2.0 * precision * recall / (precision + recall) 
    } else { 
        0.0 
    };
    
    let auroc = calculate_auroc(&all_predictions, &all_labels);
    let auprc = calculate_auprc(&all_predictions, &all_labels);
    
    (accuracy, precision, recall, f1, auroc, auprc)
}

fn evaluate_ensemble(
    models: &mut [HybridEmbeddingModel],
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
    batch_size: usize,
) -> (f32, f32, f32, f32, f32, f32) {
    let mut all_predictions = Vec::new();
    let mut all_labels = Vec::new();
    
    for batch_start in (0..data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(data.len());
        let batch = &data[batch_start..batch_end];
        
        let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
        
        let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
        let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);
        
        // Average predictions from all models
        let num_models = models.len();
        let mut ensemble_preds = vec![0.0f32; tf_indices.len()];
        for model in models.iter_mut() {
            let preds = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
            for (i, pred) in preds.iter().enumerate() {
                ensemble_preds[i] += pred / num_models as f32;
            }
        }
        
        all_predictions.extend(ensemble_preds);
        all_labels.extend(labels);
    }
    
    // Calculate metrics
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    
    for (pred, label) in all_predictions.iter().zip(all_labels.iter()) {
        let pred_class = if *pred >= 0.5 { 1.0 } else { 0.0 };
        match (pred_class as i32, *label as i32) {
            (1, 1) => tp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fp += 1,
            (0, 1) => fn_count += 1,
            _ => {}
        }
    }
    
    let accuracy = (tp + tn) as f32 / (tp + tn + fp + fn_count) as f32;
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 
        2.0 * precision * recall / (precision + recall) 
    } else { 
        0.0 
    };
    
    let auroc = calculate_auroc(&all_predictions, &all_labels);
    let auprc = calculate_auprc(&all_predictions, &all_labels);
    
    (accuracy, precision, recall, f1, auroc, auprc)
}

fn calculate_auroc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter()
        .zip(labels.iter())
        .map(|(&p, &l)| (p, l))
        .collect();
    
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    let n_neg = labels.iter().filter(|&&l| l == 0.0).count() as f32;
    
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let mut prev_fp = 0.0;
    
    for (_pred, label) in pairs {
        if label == 1.0 {
            tp += 1.0;
        } else {
            fp += 1.0;
            auc += tp / n_pos * ((fp - prev_fp) / n_neg);
            prev_fp = fp;
        }
    }
    
    auc
}

fn calculate_auprc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter()
        .zip(labels.iter())
        .map(|(&p, &l)| (p, l))
        .collect();
    
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    
    if n_pos == 0.0 {
        return 0.0;
    }
    
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut auc = 0.0;
    let mut prev_recall = 0.0;
    
    for (_pred, label) in pairs {
        if label == 1.0 {
            tp += 1.0;
            let recall = tp / n_pos;
            let precision = tp / (tp + fp);
            auc += precision * (recall - prev_recall);
            prev_recall = recall;
        } else {
            fp += 1.0;
        }
    }
    
    auc
}

fn build_expression_maps(
    builder: &PriorDatasetBuilder,
    expr_data: &ExpressionData,
    _num_tfs: usize,
    _num_genes: usize,
) -> (HashMap<usize, Array1<f32>>, HashMap<usize, Array1<f32>>, usize) {
    
    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, gene_name) in expr_data.gene_names.iter().enumerate() {
        gene_to_expr_idx.insert(gene_name.clone(), i);
    }
    
    let expr_dim = expr_data.n_cell_types;
    let tf_vocab = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();
    
    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    
    for (tf_name, &tf_idx) in tf_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(tf_name) {
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            tf_expr_map.insert(tf_idx, expr_profile);
        } else {
            tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
        }
    }
    
    for (gene_name, &gene_idx) in gene_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(gene_name) {
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            gene_expr_map.insert(gene_idx, expr_profile);
        } else {
            gene_expr_map.insert(gene_idx, Array1::zeros(expr_dim));
        }
    }
    
    (tf_expr_map, gene_expr_map, expr_dim)
}

fn build_expr_batch(
    indices: &[usize],
    expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
) -> Array2<f32> {
    let batch_size = indices.len();
    let mut batch = Array2::zeros((batch_size, expr_dim));
    
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(expr) = expr_map.get(&idx) {
            batch.row_mut(i).assign(expr);
        }
    }
    
    batch
}

fn build_expr_batch_with_dropout(
    indices: &[usize],
    expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
    dropout_rate: f32,
    seed: u64,
) -> Array2<f32> {
    let batch_size = indices.len();
    let mut batch = Array2::zeros((batch_size, expr_dim));
    let mut rng = StdRng::seed_from_u64(seed);
    let keep_prob = 1.0 - dropout_rate;
    let scale = 1.0 / keep_prob;
    
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(expr) = expr_map.get(&idx) {
            for (j, &val) in expr.iter().enumerate() {
                if rand::Rng::gen::<f32>(&mut rng) < keep_prob {
                    batch[[i, j]] = val * scale;
                }
            }
        }
    }
    
    batch
}
