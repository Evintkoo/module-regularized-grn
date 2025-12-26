/// Train Medium model with optimized hyperparameters for 95% target
use module_regularized_grn::{
    Config,
    models::{scalable_hybrid::{ScalableHybridModel, ModelConfig}, nn::{bce_loss, bce_loss_backward}},
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
    evaluation::EvaluationMetrics,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Medium Model Training with Optimized Hyperparameters ===\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load data
    println!("Loading data...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    
    // Build expression mapping (use zeros - proven to work)
    let expr_dim = expr_data.n_cell_types;
    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    
    for tf_idx in 0..num_tfs {
        tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
    }
    for gene_idx in 0..num_genes {
        gene_expr_map.insert(gene_idx, Array1::zeros(expr_dim));
    }
    
    println!("  TFs: {} | Genes: {} | Expression dims: {}", num_tfs, num_genes, expr_dim);
    
    // Create dataset
    println!("Creating dataset...");
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), seed);
    
    let mut all_samples: Vec<(usize, usize, f32)> = Vec::new();
    for (tf_idx, gene_idx) in positives {
        all_samples.push((tf_idx, gene_idx, 1.0));
    }
    for (tf_idx, gene_idx) in negatives {
        all_samples.push((tf_idx, gene_idx, 0.0));
    }
    all_samples.shuffle(&mut rng);
    
    let n_val = (all_samples.len() as f32 * 0.2) as usize;
    let n_train = all_samples.len() - n_val;
    let train_samples = &all_samples[..n_train];
    let val_samples = &all_samples[n_train..];
    
    println!("  Train: {} | Val: {}\n", train_samples.len(), val_samples.len());
    
    // Medium model configuration with optimizations
    let model_config = ModelConfig::medium();
    let learning_rate = 0.005_f32;  // Higher LR than Ultra
    let batch_size = 64;             // Larger batch than Ultra
    let num_epochs = 150;
    
    println!("Medium Model Configuration:");
    println!("  Embedding dim: {}", model_config.embed_dim);
    println!("  Hidden layers: {:?}", model_config.hidden_dims);
    println!("  Output dim: {}", model_config.output_dim);
    println!("  Dropout: {}", model_config.dropout);
    println!("  Learning rate: {} (optimized)", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}\n", num_epochs);
    
    let mut model = ScalableHybridModel::new(num_tfs, num_genes, model_config.clone(), 0.01, seed);
    println!("  Total parameters: {}\n", model.num_parameters());
    
    let mut best_val_acc = 0.0;
    let mut best_epoch = 0;
    let mut best_val_loss = f32::INFINITY;
    let mut epochs_without_improvement = 0;
    let patience = 20; // Early stopping
    
    println!("Training Medium model with early stopping...\n");
    
    for epoch in 0..num_epochs {
        model.zero_grad();
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        // Training
        for batch_start in (0..train_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_samples.len());
            let batch = &train_samples[batch_start..batch_end];
            
            let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            let mut tf_expr_batch = Array2::zeros((batch.len(), expr_dim));
            let mut gene_expr_batch = Array2::zeros((batch.len(), expr_dim));
            
            for (i, &tf_idx) in tf_indices.iter().enumerate() {
                tf_expr_batch.row_mut(i).assign(&tf_expr_map[&tf_idx]);
            }
            for (i, &gene_idx) in gene_indices.iter().enumerate() {
                gene_expr_batch.row_mut(i).assign(&gene_expr_map[&gene_idx]);
            }
            
            let scores = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
            let scores_2d = scores.clone().insert_axis(ndarray::Axis(1));
            let loss = bce_loss(&scores_2d, &labels);
            let grad_2d = bce_loss_backward(&scores_2d, &labels);
            let grad = grad_2d.column(0).to_owned() / batch.len() as f32;
            
            model.backward(&grad);
            
            epoch_loss += loss;
            num_batches += 1;
        }
        
        model.update(learning_rate);
        
        let train_loss = epoch_loss / num_batches as f32;
        
        // Validation every 5 epochs
        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            let mut val_loss = 0.0;
            let mut val_correct = 0;
            let mut val_batches = 0;
            
            for batch_start in (0..val_samples.len()).step_by(batch_size) {
                let batch_end = (batch_start + batch_size).min(val_samples.len());
                let batch = &val_samples[batch_start..batch_end];
                
                let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
                let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
                let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
                
                let mut tf_expr_batch = Array2::zeros((batch.len(), expr_dim));
                let mut gene_expr_batch = Array2::zeros((batch.len(), expr_dim));
                
                for (i, &tf_idx) in tf_indices.iter().enumerate() {
                    tf_expr_batch.row_mut(i).assign(&tf_expr_map[&tf_idx]);
                }
                for (i, &gene_idx) in gene_indices.iter().enumerate() {
                    gene_expr_batch.row_mut(i).assign(&gene_expr_map[&gene_idx]);
                }
                
                let scores = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
                let scores_2d = scores.clone().insert_axis(ndarray::Axis(1));
                let loss = bce_loss(&scores_2d, &labels);
                
                val_loss += loss;
                
                for (i, &score) in scores.iter().enumerate() {
                    let pred = if score > 0.5 { 1.0 } else { 0.0 };
                    if pred == labels[i] {
                        val_correct += 1;
                    }
                }
                
                val_batches += 1;
            }
            
            val_loss /= val_batches as f32;
            let val_acc = val_correct as f32 / val_samples.len() as f32;
            
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_val_loss = val_loss;
                best_epoch = epoch;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 5;
            }
            
            println!("Epoch {:3} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.4} ({:.2}%)",
                     epoch, train_loss, val_loss, val_acc, val_acc * 100.0);
            
            // Early stopping
            if epochs_without_improvement >= patience {
                println!("\nEarly stopping at epoch {} (no improvement for {} epochs)", epoch, patience);
                break;
            }
            
            // Target reached
            if val_acc >= 0.95 {
                println!("\n🎉 REACHED 95% TARGET AT EPOCH {}!", epoch);
                break;
            }
        }
    }
    
    println!("\n=== Training Complete ===");
    println!("Best epoch: {}", best_epoch);
    println!("Best Val Loss: {:.4}", best_val_loss);
    println!("Best Val Accuracy: {:.4} ({:.2}%)\n", best_val_acc, best_val_acc * 100.0);
    
    // Final comprehensive evaluation
    println!("Running comprehensive evaluation...");
    
    let mut all_predictions = Vec::new();
    let mut all_labels = Vec::new();
    
    for batch_start in (0..val_samples.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(val_samples.len());
        let batch = &val_samples[batch_start..batch_end];
        
        let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
        let labels: Vec<f32> = batch.iter().map(|(_, _, label)| *label).collect();
        
        let mut tf_expr_batch = Array2::zeros((batch.len(), expr_dim));
        let mut gene_expr_batch = Array2::zeros((batch.len(), expr_dim));
        
        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            tf_expr_batch.row_mut(i).assign(&tf_expr_map[&tf_idx]);
        }
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            gene_expr_batch.row_mut(i).assign(&gene_expr_map[&gene_idx]);
        }
        
        let scores = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
        
        all_predictions.extend(scores.iter());
        all_labels.extend(labels.iter());
    }
    
    let predictions = Array1::from_vec(all_predictions);
    let labels = Array1::from_vec(all_labels);
    
    let metrics = EvaluationMetrics::compute(&predictions, &labels, 0.5);
    metrics.print();
    
    // Save results
    let results_json = serde_json::json!({
        "model": "Medium",
        "parameters": model.num_parameters(),
        "epochs": num_epochs,
        "best_epoch": best_epoch,
        "accuracy": metrics.accuracy,
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1_score": metrics.f1_score,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
    });
    
    std::fs::write("results/medium_results.json", serde_json::to_string_pretty(&results_json)?)?;
    println!("\n✅ Results saved to results/medium_results.json");
    
    // Assessment
    println!("\n🎯 Final Assessment:");
    if metrics.accuracy >= 0.95 {
        println!("✅✅✅ SUCCESS: Reached 95% accuracy target!");
        println!("   Accuracy: {:.2}%", metrics.accuracy * 100.0);
        println!("   AUROC: {:.4}", metrics.auroc);
        println!("   Improvement over baseline: +{:.2}%", (metrics.accuracy - 0.5782) * 100.0);
    } else if metrics.accuracy >= 0.90 {
        println!("✅✅ EXCELLENT: {:.2}% accuracy",
                 metrics.accuracy * 100.0);
        println!("   Gap to 95%: {:.2}%", (0.95 - metrics.accuracy) * 100.0);
        println!("   AUROC: {:.4}", metrics.auroc);
        println!("   Improvement over baseline: +{:.2}%", (metrics.accuracy - 0.5782) * 100.0);
    } else if metrics.accuracy >= 0.85 {
        println!("✅ VERY GOOD: {:.2}% accuracy",
                 metrics.accuracy * 100.0);
        println!("   Gap to 95%: {:.2}%", (0.95 - metrics.accuracy) * 100.0);
        println!("   AUROC: {:.4}", metrics.auroc);
        println!("   Improvement over baseline: +{:.2}%", (metrics.accuracy - 0.5782) * 100.0);
    } else if metrics.accuracy >= 0.80 {
        println!("✅ GOOD: {:.2}% accuracy (similar to best baseline)",
                 metrics.accuracy * 100.0);
        println!("   Gap to 95%: {:.2}%", (0.95 - metrics.accuracy) * 100.0);
        println!("   AUROC: {:.4}", metrics.auroc);
    } else {
        println!("⚠️  {:.2}% accuracy (below baseline)",
                 metrics.accuracy * 100.0);
        println!("   May need different approach or hyperparameters");
    }
    
    // Comparison
    println!("\n📊 Comparison:");
    println!("   Baseline (1.27M): 80.14% accuracy");
    println!("   Medium ({:.1}M): {:.2}% accuracy", 
             model.num_parameters() as f32 / 1_000_000.0,
             metrics.accuracy * 100.0);
    if metrics.accuracy > 0.8014 {
        println!("   ✅ Medium model is better! (+{:.2}%)", 
                 (metrics.accuracy - 0.8014) * 100.0);
    } else {
        println!("   ⚠️  Baseline is still better");
    }
    
    Ok(())
}
