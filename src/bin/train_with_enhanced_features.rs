/// Train ensemble with PROPER enhanced expression features
/// Uses preprocessed features from feature_engineering.py
use module_regularized_grn::{
    Config,
    models::{hybrid_embeddings::HybridEmbeddingModel, nn::bce_loss_backward},
    data::{PriorKnowledge, PriorDatasetBuilder},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;
use std::time::Instant;
use serde_json;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("  ENSEMBLE WITH ENHANCED EXPRESSION FEATURES");
    println!("═══════════════════════════════════════════════════════════\n");

    let config = Config::load_default()?;
    let base_seed = config.project.seed;
    
    // Configuration
    let embed_dim = 512;
    let hidden_dim = 2048;
    let output_dim = 512;
    let learning_rate = 0.005;
    let batch_size = 256;
    let epochs = 60;
    let dropout_rate = 0.2;
    let weight_decay = 0.01;
    let num_models = 5;
    
    println!("Configuration:");
    println!("  Model: Large (512d emb, 2048d hidden)");
    println!("  Ensemble: {} models", num_models);
    println!("  Features: Enhanced (17d)\n");
    
    // Load data
    println!("1. Loading prior knowledge...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    println!("   TFs: {} | Genes: {}\n", num_tfs, num_genes);
    
    // Load enhanced expression features
    println!("2. Loading enhanced expression features...");
    let tf_features = load_features("data/processed/features/tf_features.json")?;
    let gene_features = load_features("data/processed/features/gene_features.json")?;
    let feature_info: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("data/processed/features/feature_info.json")?
    )?;
    
    let expr_dim = feature_info["feature_dim"].as_u64().unwrap() as usize;
    let tf_mapped = feature_info["tf_mapped"].as_u64().unwrap();
    let gene_mapped = feature_info["gene_mapped"].as_u64().unwrap();
    
    println!("   Feature dimension: {}", expr_dim);
    println!("   TF features mapped: {}/{} ({:.1}%)", 
             tf_mapped, num_tfs, 100.0 * tf_mapped as f64 / num_tfs as f64);
    println!("   Gene features mapped: {}/{} ({:.1}%)\n", 
             gene_mapped, num_genes, 100.0 * gene_mapped as f64 / num_genes as f64);
    
    // Convert to hashmaps indexed by ID
    let (tf_expr_map, gene_expr_map) = build_expression_maps(
        &builder, &tf_features, &gene_features, expr_dim
    )?;
    
    // Create dataset
    println!("3. Creating dataset...");
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
    
    println!("   Train: {} | Val: {} | Test: {}\n", 
             train_data.len(), val_data.len(), test_data.len());
    
    // Train ensemble
    println!("═══════════════════════════════════════════════════════════");
    println!("Training {} models...\n", num_models);
    
    let mut models = Vec::new();
    let mut individual_results = Vec::new();
    
    for i in 0..num_models {
        let seed = base_seed + i as u64;
        println!("Model {}/{} (seed={}):", i + 1, num_models, seed);
        println!("{}", "-".repeat(50));
        
        let start = Instant::now();
        
        let mut model = HybridEmbeddingModel::new(
            num_tfs, num_genes, embed_dim, expr_dim,
            hidden_dim, output_dim, 0.05, 0.1, seed,
        );
        
        // Training
        let mut best_val_acc = 0.0;
        let mut patience_counter = 0;
        let patience = 10;
        
        for epoch in 0..epochs {
            let mut epoch_data = train_data.clone();
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            epoch_data.shuffle(&mut epoch_rng);
            
            for batch_start in (0..epoch_data.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(epoch_data.len());
                let batch = &epoch_data[batch_start..batch_end];
                
                let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
                
                let tf_expr_batch = build_expr_batch(&tf_indices, &tf_expr_map, expr_dim);
                let gene_expr_batch = build_expr_batch(&gene_indices, &gene_expr_map, expr_dim);
                
                let predictions = model.forward(&tf_indices, &gene_indices, 
                                               &tf_expr_batch, &gene_expr_batch);
                
                let labels_arr = Array1::from(labels);
                let predictions_2d = predictions.clone().insert_axis(ndarray::Axis(1));
                let grad_2d = bce_loss_backward(&predictions_2d, &labels_arr);
                let grad = grad_2d.column(0).to_owned();
                
                let grad_with_dropout = apply_dropout(&grad, dropout_rate, seed + epoch as u64);
                model.backward(&grad_with_dropout);
            }
            
            model.update_with_weight_decay(learning_rate, weight_decay);
            model.zero_grad();
            
            if epoch % 5 == 0 || epoch == epochs - 1 {
                let val_acc = evaluate(&mut model, &val_data, &tf_expr_map, 
                                      &gene_expr_map, expr_dim, batch_size);
                
                if epoch % 10 == 0 {
                    println!("  Epoch {:3}: Val Acc = {:.2}%", epoch, val_acc * 100.0);
                }
                
                if val_acc > best_val_acc {
                    best_val_acc = val_acc;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= patience {
                        println!("  Early stopping at epoch {}", epoch);
                        break;
                    }
                }
            }
        }
        
        let (test_acc, test_prec, test_rec, test_f1) = evaluate_detailed(
            &mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size,
        );
        
        let elapsed = start.elapsed();
        println!("  ✓ Test Accuracy: {:.2}%", test_acc * 100.0);
        println!("  Training time: {:.1} minutes\n", elapsed.as_secs() as f64 / 60.0);
        
        individual_results.push((test_acc, test_prec, test_rec, test_f1));
        models.push(model);
    }
    
    // Results
    println!("═══════════════════════════════════════════════════════════");
    println!("INDIVIDUAL MODEL RESULTS:");
    println!("═══════════════════════════════════════════════════════════\n");
    
    for (i, (acc, prec, rec, f1)) in individual_results.iter().enumerate() {
        println!("Model {}: Acc={:.2}% | Prec={:.2}% | Rec={:.2}% | F1={:.4}",
                 i + 1, acc * 100.0, prec * 100.0, rec * 100.0, f1);
    }
    
    let mean_acc = individual_results.iter().map(|x| x.0).sum::<f32>() / num_models as f32;
    let std_acc = (individual_results.iter()
        .map(|x| (x.0 - mean_acc).powi(2))
        .sum::<f32>() / num_models as f32).sqrt();
    
    println!("\nMean: {:.2}% ± {:.2}%", mean_acc * 100.0, std_acc * 100.0);
    
    // Ensemble
    println!("\n═══════════════════════════════════════════════════════════");
    println!("ENSEMBLE RESULTS:");
    println!("═══════════════════════════════════════════════════════════\n");
    
    let (ensemble_acc, ensemble_prec, ensemble_rec, ensemble_f1, 
         ensemble_auroc, ensemble_auprc) = evaluate_ensemble(
        &mut models, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size,
    );
    
    println!("🏆 ENSEMBLE PERFORMANCE:");
    println!("  Accuracy:  {:.2}%", ensemble_acc * 100.0);
    println!("  Precision: {:.2}%", ensemble_prec * 100.0);
    println!("  Recall:    {:.2}%", ensemble_rec * 100.0);
    println!("  F1-Score:  {:.4}", ensemble_f1);
    println!("  AUROC:     {:.4}", ensemble_auroc);
    println!("  AUPRC:     {:.4}", ensemble_auprc);
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("COMPARISON:");
    println!("═══════════════════════════════════════════════════════════");
    println!("Previous (no feature mapping):  83.06%");
    println!("Current (enhanced features):    {:.2}%", ensemble_acc * 100.0);
    println!("Improvement:                    {:+.2}%", (ensemble_acc - 0.8306) * 100.0);
    println!("═══════════════════════════════════════════════════════════\n");
    
    Ok(())
}

fn load_features(path: &str) -> Result<HashMap<String, Vec<f32>>> {
    let content = std::fs::read_to_string(path)?;
    let features: HashMap<String, Vec<f32>> = serde_json::from_str(&content)?;
    Ok(features)
}

fn build_expression_maps(
    builder: &PriorDatasetBuilder,
    tf_features: &HashMap<String, Vec<f32>>,
    gene_features: &HashMap<String, Vec<f32>>,
    expr_dim: usize,
) -> Result<(HashMap<usize, Array1<f32>>, HashMap<usize, Array1<f32>>)> {
    let tf_vocab = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();
    
    let mut tf_expr_map = HashMap::new();
    let mut gene_expr_map = HashMap::new();
    
    for (tf_name, &tf_idx) in tf_vocab.iter() {
        if let Some(features) = tf_features.get(tf_name) {
            tf_expr_map.insert(tf_idx, Array1::from(features.clone()));
        } else {
            tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
        }
    }
    
    for (gene_name, &gene_idx) in gene_vocab.iter() {
        if let Some(features) = gene_features.get(gene_name) {
            gene_expr_map.insert(gene_idx, Array1::from(features.clone()));
        } else {
            gene_expr_map.insert(gene_idx, Array1::zeros(expr_dim));
        }
    }
    
    Ok((tf_expr_map, gene_expr_map))
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
) -> (f32, f32, f32, f32) {
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
    
    (accuracy, precision, recall, f1)
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
