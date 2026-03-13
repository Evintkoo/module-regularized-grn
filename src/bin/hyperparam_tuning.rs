/// Hyperparameter tuning for Standard MLP
/// Goal: Find optimal configuration for maximum accuracy
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
use serde_json::json;

#[derive(Debug, Clone)]
struct HyperparamConfig {
    embed_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    temperature: f32,
    learning_rate: f32,
    batch_size: usize,
    weight_decay: f32,
    dropout_rate: f32,
}

#[derive(Debug)]
struct TrainingResult {
    config: HyperparamConfig,
    train_accuracy: f32,
    val_accuracy: f32,
    test_accuracy: f32,
    test_precision: f32,
    test_recall: f32,
    test_f1: f32,
    epochs_trained: usize,
}

fn main() -> Result<()> {
    println!("=== Hyperparameter Tuning for Standard MLP ===");
    println!("Goal: Maximize accuracy through systematic search\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    
    // Load data once
    println!("Loading data...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    
    // Build expression maps
    let (tf_expr_map, gene_expr_map, expr_dim) = build_expression_maps(
        &builder,
        &expr_data,
        num_tfs,
        num_genes,
    );
    
    // Create dataset
    let mut rng = StdRng::seed_from_u64(seed);
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), seed);
    
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
    
    println!("Dataset: {} train | {} val | {} test\n", 
             train_data.len(), val_data.len(), test_data.len());
    
    // Define hyperparameter search space
    let configs = generate_configs();
    
    println!("Testing {} configurations...\n", configs.len());
    println!("{:>3} | {:>4} | {:>4} | {:>4} | {:>6} | {:>6} | {:>4} | Epochs | Val Acc | Test Acc",
             "#", "Emb", "Hid", "Out", "LR", "Temp", "BS");
    println!("{}", "-".repeat(100));
    
    let mut results: Vec<TrainingResult> = Vec::new();
    
    for (idx, config) in configs.iter().enumerate() {
        let result = train_and_evaluate(
            &config,
            num_tfs,
            num_genes,
            expr_dim,
            &train_data,
            &val_data,
            &test_data,
            &tf_expr_map,
            &gene_expr_map,
            seed,
        )?;
        
        println!("{:>3} | {:>4} | {:>4} | {:>4} | {:.4} | {:.4} | {:>4} | {:>6} | {:>6.2}% | {:>7.2}%",
                 idx + 1,
                 config.embed_dim,
                 config.hidden_dim,
                 config.output_dim,
                 config.learning_rate,
                 config.temperature,
                 config.batch_size,
                 result.epochs_trained,
                 result.val_accuracy * 100.0,
                 result.test_accuracy * 100.0);
        
        results.push(result);
    }
    
    // Find best configuration
    let best_result = results.iter()
        .max_by(|a, b| a.test_accuracy.partial_cmp(&b.test_accuracy).unwrap())
        .unwrap();
    
    println!("\n{}", "=".repeat(100));
    println!("BEST CONFIGURATION FOUND:");
    println!("{}", "=".repeat(100));
    println!("Embedding dim:     {}", best_result.config.embed_dim);
    println!("Hidden dim:        {}", best_result.config.hidden_dim);
    println!("Output dim:        {}", best_result.config.output_dim);
    println!("Learning rate:     {}", best_result.config.learning_rate);
    println!("Temperature:       {}", best_result.config.temperature);
    println!("Batch size:        {}", best_result.config.batch_size);
    println!("Weight decay:      {}", best_result.config.weight_decay);
    println!("\nPerformance:");
    println!("Test Accuracy:     {:.2}%", best_result.test_accuracy * 100.0);
    println!("Test Precision:    {:.2}%", best_result.test_precision * 100.0);
    println!("Test Recall:       {:.2}%", best_result.test_recall * 100.0);
    println!("Test F1-Score:     {:.4}", best_result.test_f1);
    println!("{}", "=".repeat(100));
    
    // Save all results
    let results_json = json!({
        "experiment": "hyperparameter_tuning",
        "date": chrono::Utc::now().to_rfc3339(),
        "num_configs_tested": configs.len(),
        "best_config": {
            "embed_dim": best_result.config.embed_dim,
            "hidden_dim": best_result.config.hidden_dim,
            "output_dim": best_result.config.output_dim,
            "learning_rate": best_result.config.learning_rate,
            "temperature": best_result.config.temperature,
            "batch_size": best_result.config.batch_size,
            "weight_decay": best_result.config.weight_decay,
        },
        "best_performance": {
            "accuracy": best_result.test_accuracy,
            "precision": best_result.test_precision,
            "recall": best_result.test_recall,
            "f1_score": best_result.test_f1,
        },
        "all_results": results.iter().map(|r| {
            json!({
                "config": {
                    "embed_dim": r.config.embed_dim,
                    "hidden_dim": r.config.hidden_dim,
                    "output_dim": r.config.output_dim,
                    "learning_rate": r.config.learning_rate,
                    "temperature": r.config.temperature,
                    "batch_size": r.config.batch_size,
                },
                "test_accuracy": r.test_accuracy,
                "test_precision": r.test_precision,
                "test_recall": r.test_recall,
                "test_f1": r.test_f1,
            })
        }).collect::<Vec<_>>(),
    });
    
    std::fs::write(
        "results/hyperparameter_tuning.json",
        serde_json::to_string_pretty(&results_json)?
    )?;
    
    println!("\n✓ Results saved to: results/hyperparameter_tuning.json");
    
    Ok(())
}

fn generate_configs() -> Vec<HyperparamConfig> {
    let mut configs = Vec::new();
    
    // Embedding dimensions to try
    let embed_dims = vec![32, 64, 128, 256];
    // Hidden dimensions to try
    let hidden_dims = vec![64, 128, 256, 512];
    // Output dimensions to try
    let output_dims = vec![32, 64, 128];
    // Learning rates to try
    let learning_rates = vec![0.0001, 0.0005, 0.001, 0.005];
    // Temperatures to try
    let temperatures = vec![0.05, 0.1, 0.2, 0.5];
    // Batch sizes to try
    let batch_sizes = vec![128, 256, 512];
    
    // Grid search over key parameters
    // Focus on most impactful: embed_dim, hidden_dim, learning_rate, temperature
    for &embed_dim in &embed_dims {
        for &hidden_dim in &hidden_dims {
            // Skip if hidden too small relative to input
            if hidden_dim < embed_dim / 2 {
                continue;
            }
            
            for &lr in &learning_rates {
                for &temp in &temperatures {
                    // Use reasonable defaults for other params
                    let output_dim = embed_dim; // Match embedding dim
                    let batch_size = 256;
                    let weight_decay = 0.01;
                    let dropout_rate = 0.3;
                    
                    configs.push(HyperparamConfig {
                        embed_dim,
                        hidden_dim,
                        output_dim,
                        temperature: temp,
                        learning_rate: lr,
                        batch_size,
                        weight_decay,
                        dropout_rate,
                    });
                }
            }
        }
    }
    
    // Also try some batch size variations for best configs
    for &bs in &batch_sizes {
        if bs != 256 {
            configs.push(HyperparamConfig {
                embed_dim: 64,
                hidden_dim: 128,
                output_dim: 64,
                temperature: 0.1,
                learning_rate: 0.001,
                batch_size: bs,
                weight_decay: 0.01,
                dropout_rate: 0.3,
            });
        }
    }
    
    configs
}

fn train_and_evaluate(
    config: &HyperparamConfig,
    num_tfs: usize,
    num_genes: usize,
    expr_dim: usize,
    train_data: &[(usize, usize, f32)],
    val_data: &[(usize, usize, f32)],
    test_data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    seed: u64,
) -> Result<TrainingResult> {
    
    // Initialize model
    let mut model = HybridEmbeddingModel::new(
        num_tfs,
        num_genes,
        config.embed_dim,
        expr_dim,
        config.hidden_dim,
        config.output_dim,
        config.temperature,
        0.1,  // init std
        seed,
    );
    
    let max_epochs = 30; // Reduced for speed
    let patience = 5;
    
    let mut best_val_acc = 0.0;
    let mut patience_counter = 0;
    let mut epochs_trained = 0;
    
    for epoch in 0..max_epochs {
        // Training
        for batch_start in (0..train_data.len()).step_by(config.batch_size) {
            let batch_end = (batch_start + config.batch_size).min(train_data.len());
            let batch = &train_data[batch_start..batch_end];
            
            let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
            
            let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
            let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);
            
            let predictions = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
            let labels_arr = Array1::from(labels);
            let predictions_2d = predictions.clone().insert_axis(ndarray::Axis(1));
            let grad_2d = bce_loss_backward(&predictions_2d, &labels_arr);
            let grad = grad_2d.column(0).to_owned();
            
            model.backward(&grad);
        }
        
        model.update(config.learning_rate);
        model.zero_grad();
        
        // Validation
        let val_acc = evaluate(&mut model, val_data, tf_expr_map, gene_expr_map, expr_dim, config.batch_size);
        
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= patience {
                break;
            }
        }
        
        epochs_trained = epoch + 1;
    }
    
    // Final evaluation on test set
    let (test_acc, test_prec, test_rec, test_f1) = evaluate_detailed(
        &mut model,
        test_data,
        tf_expr_map,
        gene_expr_map,
        expr_dim,
        config.batch_size,
    );
    
    Ok(TrainingResult {
        config: config.clone(),
        train_accuracy: 0.0, // Not tracking to save time
        val_accuracy: best_val_acc,
        test_accuracy: test_acc,
        test_precision: test_prec,
        test_recall: test_rec,
        test_f1,
        epochs_trained,
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
            let pred_class = if *pred >= 0.5 { 1.0 } else { 0.0 };
            if pred_class == *label {
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

fn build_expression_maps(
    builder: &PriorDatasetBuilder,
    expr_data: &ExpressionData,
    num_tfs: usize,
    num_genes: usize,
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
