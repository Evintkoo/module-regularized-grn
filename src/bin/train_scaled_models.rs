/// Train progressively larger MLP models to test scaling laws
/// Goal: Push accuracy higher by increasing model capacity
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
use std::time::Instant;

#[derive(Debug, Clone)]
struct ModelScale {
    name: String,
    embed_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    epochs: usize,
}

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("        SCALING UP MODEL SIZE - TESTING LARGER MODELS");
    println!("═══════════════════════════════════════════════════════════\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    
    // Define model scales to test
    let scales = vec![
        ModelScale {
            name: "Current Best (1.2M)".to_string(),
            embed_dim: 128,
            hidden_dim: 512,
            output_dim: 128,
            epochs: 40,
        },
        ModelScale {
            name: "Medium (5M)".to_string(),
            embed_dim: 256,
            hidden_dim: 1024,
            output_dim: 256,
            epochs: 50,
        },
        ModelScale {
            name: "Large (20M)".to_string(),
            embed_dim: 512,
            hidden_dim: 2048,
            output_dim: 512,
            epochs: 50,
        },
        ModelScale {
            name: "Very Large (80M)".to_string(),
            embed_dim: 1024,
            hidden_dim: 4096,
            output_dim: 1024,
            epochs: 60,
        },
    ];
    
    // Load data once
    println!("Loading data...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    
    println!("  TFs: {} | Genes: {}", num_tfs, num_genes);
    println!("  Cell types: {}\n", expr_data.n_cell_types);
    
    // Build expression maps
    let (tf_expr_map, gene_expr_map, expr_dim) = build_expression_maps(
        &builder,
        &expr_data,
        num_tfs,
        num_genes,
    );
    
    // Create dataset
    println!("Creating dataset...");
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
    
    println!("  Train: {} | Val: {} | Test: {}\n", 
             train_data.len(), val_data.len(), test_data.len());
    
    // Train each scale
    println!("═══════════════════════════════════════════════════════════");
    println!(" Model Scale | Embed | Hidden | Params  | Time   | Test Acc");
    println!("═══════════════════════════════════════════════════════════");
    
    let mut results = Vec::new();
    
    for scale in &scales {
        let start_time = Instant::now();
        
        // Estimate parameters
        let params = estimate_params(num_tfs, num_genes, scale.embed_dim, expr_dim, 
                                     scale.hidden_dim, scale.output_dim);
        
        println!("\n{} - Starting training...", scale.name);
        println!("  Embedding: {}d | Hidden: {}d | Output: {}d", 
                 scale.embed_dim, scale.hidden_dim, scale.output_dim);
        println!("  Parameters: ~{:.1}M", params as f64 / 1_000_000.0);
        
        // Train model
        let result = train_model(
            scale,
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
        
        let elapsed = start_time.elapsed();
        
        println!(" {:15} | {:5} | {:6} | {:5.1}M | {:5.1}m | {:6.2}%",
                 scale.name,
                 scale.embed_dim,
                 scale.hidden_dim,
                 params as f64 / 1_000_000.0,
                 elapsed.as_secs() as f64 / 60.0,
                 result.test_accuracy * 100.0);
        
        results.push((scale.clone(), result, elapsed, params));
    }
    
    // Summary
    println!("\n═══════════════════════════════════════════════════════════");
    println!("                      SCALING RESULTS");
    println!("═══════════════════════════════════════════════════════════\n");
    
    println!("{:15} | {:8} | {:7} | {:8} | {:8}", 
             "Model", "Params", "Acc", "Prec", "Recall");
    println!("{}", "-".repeat(65));
    
    for (scale, result, _, params) in &results {
        println!("{:15} | {:5.1}M | {:6.2}% | {:6.2}% | {:6.2}%",
                 scale.name,
                 *params as f64 / 1_000_000.0,
                 result.test_accuracy * 100.0,
                 result.test_precision * 100.0,
                 result.test_recall * 100.0);
    }
    
    // Find best
    let best = results.iter()
        .max_by(|a, b| a.1.test_accuracy.partial_cmp(&b.1.test_accuracy).unwrap())
        .unwrap();
    
    println!("\n═══════════════════════════════════════════════════════════");
    println!("🏆 BEST MODEL: {}", best.0.name);
    println!("═══════════════════════════════════════════════════════════");
    println!("  Test Accuracy:  {:.2}%", best.1.test_accuracy * 100.0);
    println!("  Test Precision: {:.2}%", best.1.test_precision * 100.0);
    println!("  Test Recall:    {:.2}%", best.1.test_recall * 100.0);
    println!("  Test F1-Score:  {:.4}", best.1.test_f1);
    println!("  Training time:  {:.1} minutes", best.2.as_secs() as f64 / 60.0);
    println!("  Parameters:     {:.1}M", best.3 as f64 / 1_000_000.0);
    
    // Save results
    let results_json = serde_json::json!({
        "experiment": "model_scaling",
        "date": chrono::Utc::now().to_rfc3339(),
        "results": results.iter().map(|(scale, result, elapsed, params)| {
            serde_json::json!({
                "name": scale.name,
                "embed_dim": scale.embed_dim,
                "hidden_dim": scale.hidden_dim,
                "output_dim": scale.output_dim,
                "parameters": params,
                "test_accuracy": result.test_accuracy,
                "test_precision": result.test_precision,
                "test_recall": result.test_recall,
                "test_f1": result.test_f1,
                "training_time_seconds": elapsed.as_secs(),
            })
        }).collect::<Vec<_>>(),
        "best_model": {
            "name": best.0.name,
            "accuracy": best.1.test_accuracy,
            "parameters": best.3,
        }
    });
    
    std::fs::write(
        "results/model_scaling_results.json",
        serde_json::to_string_pretty(&results_json)?
    )?;
    
    println!("\n✓ Results saved to: results/model_scaling_results.json");
    println!("═══════════════════════════════════════════════════════════\n");
    
    Ok(())
}

#[derive(Debug)]
struct TrainResult {
    test_accuracy: f32,
    test_precision: f32,
    test_recall: f32,
    test_f1: f32,
}

fn train_model(
    scale: &ModelScale,
    num_tfs: usize,
    num_genes: usize,
    expr_dim: usize,
    train_data: &[(usize, usize, f32)],
    val_data: &[(usize, usize, f32)],
    test_data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    seed: u64,
) -> Result<TrainResult> {
    
    let mut model = HybridEmbeddingModel::new(
        num_tfs,
        num_genes,
        scale.embed_dim,
        expr_dim,
        scale.hidden_dim,
        scale.output_dim,
        0.05,  // temperature
        0.1,   // init std
        seed,
    );
    
    let learning_rate = 0.005;
    let batch_size = 256;
    let patience = 10;
    
    let mut best_val_acc = 0.0;
    let mut patience_counter = 0;
    
    println!("  Training for up to {} epochs...", scale.epochs);
    
    for epoch in 0..scale.epochs {
        // Training
        for batch_start in (0..train_data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_data.len());
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
        
        model.update(learning_rate);
        model.zero_grad();
        
        // Validation every 5 epochs
        if epoch % 5 == 0 || epoch == scale.epochs - 1 {
            let val_acc = evaluate(&mut model, val_data, tf_expr_map, gene_expr_map, expr_dim, batch_size);
            
            if epoch % 10 == 0 {
                println!("    Epoch {:3}: Val Acc = {:.2}%", epoch, val_acc * 100.0);
            }
            
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= patience {
                    println!("    Early stopping at epoch {}", epoch);
                    break;
                }
            }
        }
    }
    
    // Test evaluation
    let (test_acc, test_prec, test_rec, test_f1) = evaluate_detailed(
        &mut model,
        test_data,
        tf_expr_map,
        gene_expr_map,
        expr_dim,
        batch_size,
    );
    
    println!("  ✓ Test Accuracy: {:.2}%", test_acc * 100.0);
    
    Ok(TrainResult {
        test_accuracy: test_acc,
        test_precision: test_prec,
        test_recall: test_rec,
        test_f1,
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

fn estimate_params(
    num_tfs: usize,
    num_genes: usize,
    embed_dim: usize,
    expr_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> usize {
    let tf_embed = num_tfs * embed_dim;
    let gene_embed = num_genes * embed_dim;
    let input_dim = embed_dim + expr_dim;
    let fc1_params = input_dim * hidden_dim + hidden_dim;
    let fc2_params = hidden_dim * output_dim + output_dim;
    
    tf_embed + gene_embed + 2 * (fc1_params + fc2_params)
}
