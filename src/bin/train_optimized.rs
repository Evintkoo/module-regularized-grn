/// Optimized embedding training targeting 95%+ accuracy
/// Uses: Deeper architecture, focal loss, momentum, LR scheduling
use module_regularized_grn::{
    Config,
    models::optimized_embeddings::{OptimizedEmbeddingModel, focal_loss, focal_loss_backward},
    data::{PriorKnowledge, PriorDatasetBuilder},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::f32;

fn main() -> Result<()> {
    println!("=== Optimized Embedding Training (Target: 95%+ Accuracy) ===\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load prior knowledge
    println!("Loading prior knowledge...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors);
    
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
    
    // Optimized hyperparameters
    let embed_dim = 256;      // Larger embeddings
    let hidden_dim1 = 512;    // Wider layers
    let hidden_dim2 = 256;
    let hidden_dim3 = 128;
    let output_dim = 64;      // Lower dim for better discrimination
    let initial_lr = 0.01_f32;  // Higher initial LR
    let momentum = 0.9_f32;
    let weight_decay = 0.0001_f32;
    let batch_size = 128;     // Medium batch size
    let num_epochs = 50;     // Test with fewer epochs first
    let warmup_epochs = 10;
    
    // Focal loss parameters
    let alpha = 0.25_f32;
    let gamma = 2.0_f32;
    
    println!("Optimized Configuration:");
    println!("  Architecture: Deeper (4 layers)");
    println!("  Embed dim: {}", embed_dim);
    println!("  Hidden: {} → {} → {} → {}", hidden_dim1, hidden_dim2, hidden_dim3, output_dim);
    println!("  Initial LR: {}", initial_lr);
    println!("  Momentum: {}", momentum);
    println!("  Weight decay: {}", weight_decay);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {} (with warmup)", num_epochs);
    println!("  Loss: Focal (α={}, γ={})", alpha, gamma);
    println!();
    
    let mut model = OptimizedEmbeddingModel::new(
        builder.num_tfs(),
        builder.num_genes(),
        embed_dim,
        hidden_dim1,
        hidden_dim2,
        hidden_dim3,
        output_dim,
        0.05,  // Lower temperature for sharper distinctions
        seed,
    );
    
    let params = model.count_parameters();
    println!("  Parameters: {}\n", params);
    
    println!("Training with LR scheduling and focal loss...\n");
    
    let mut best_val_loss = f32::INFINITY;
    let mut best_val_acc = 0.0;
    let mut best_epoch = 0;
    
    for epoch in 0..num_epochs {
        // Learning rate schedule: warmup + cosine decay
        let learning_rate = if epoch < warmup_epochs {
            // Warmup
            initial_lr * (epoch as f32 / warmup_epochs as f32)
        } else {
            // Cosine decay
            let progress = (epoch - warmup_epochs) as f32 / (num_epochs - warmup_epochs) as f32;
            initial_lr * 0.5 * (1.0 + (progress * f32::consts::PI).cos())
        };
        
        model.zero_grad();
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        // Training
        for batch_start in (0..train_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_samples.len());
            let batch = &train_samples[batch_start..batch_end];
            let batch_len = batch.len();
            
            let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            let scores = model.forward(&tf_indices, &gene_indices);
            
            let diagonal_scores = Array2::from_shape_fn((batch_len, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() { scores[[i, i]] } else { 0.0 }
            });
            
            // Focal loss for better learning on hard examples
            let loss = focal_loss(&diagonal_scores, &labels, alpha, gamma);
            epoch_loss += loss;
            num_batches += 1;
            
            // Backward with focal loss gradient
            let grad_loss = focal_loss_backward(&diagonal_scores, &labels, alpha, gamma);
            
            let mut grad_scores = Array2::zeros(scores.dim());
            for i in 0..batch_len {
                if i < grad_scores.nrows() && i < grad_scores.ncols() {
                    grad_scores[[i, i]] = grad_loss[[i, 0]];
                }
            }
            
            model.backward(&grad_scores);
            
            // Gradient clipping
            let max_grad = 1.0_f32;
            let grad_norm = (model.tf_embed_grad.iter().map(|x| x * x).sum::<f32>()
                           + model.gene_embed_grad.iter().map(|x| x * x).sum::<f32>()).sqrt();
            if grad_norm > max_grad {
                let scale = max_grad / grad_norm;
                model.tf_embed_grad *= scale;
                model.gene_embed_grad *= scale;
            }
            
            // Update with momentum and weight decay
            model.update(learning_rate, momentum, weight_decay);
            model.zero_grad();
        }
        
        let avg_train_loss = epoch_loss / num_batches as f32;
        
        // Validation
        let mut val_loss = 0.0;
        let mut val_batches = 0;
        let mut val_correct = 0;
        let mut val_total = 0;
        
        for batch_start in (0..val_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(val_samples.len());
            let batch = &val_samples[batch_start..batch_end];
            let batch_len = batch.len();
            
            let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            let scores = model.forward(&tf_indices, &gene_indices);
            
            let diagonal_scores = Array2::from_shape_fn((batch_len, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() { scores[[i, i]] } else { 0.0 }
            });
            
            let loss = focal_loss(&diagonal_scores, &labels, alpha, gamma);
            val_loss += loss;
            val_batches += 1;
            
            // Accuracy
            for i in 0..batch_len {
                if i < diagonal_scores.nrows() && i < labels.len() {
                    let pred = if diagonal_scores[[i, 0]] > 0.0 { 1.0 } else { 0.0 };
                    if pred == labels[i] { val_correct += 1; }
                    val_total += 1;
                }
            }
        }
        
        let avg_val_loss = val_loss / val_batches as f32;
        let val_accuracy = val_correct as f32 / val_total as f32;
        
        // Track best
        if val_accuracy > best_val_acc {
            best_val_acc = val_accuracy;
            best_val_loss = avg_val_loss;
            best_epoch = epoch;
        }
        
        // Print progress
        if epoch % 5 == 0 || epoch == num_epochs - 1 || val_accuracy > 0.94 {
            println!(
                "Epoch {:3} | LR: {:.5} | Train: {:.4} | Val Loss: {:.4} | Val Acc: {:.4} ({:.2}%)",
                epoch, learning_rate, avg_train_loss, avg_val_loss, val_accuracy, val_accuracy * 100.0
            );
        }
        
        // Early stopping if we hit target
        if val_accuracy >= 0.95 {
            println!("\n🎉 TARGET ACHIEVED at epoch {}!", epoch);
            break;
        }
    }
    
    println!("\n=== Training Complete ===");
    println!("Best epoch: {}", best_epoch);
    println!("Best Val Loss: {:.4}", best_val_loss);
    println!("Best Val Accuracy: {:.4} ({:.2}%)", best_val_acc, best_val_acc * 100.0);
    println!();
    
    // Evaluate against target
    if best_val_acc >= 0.95 {
        println!("✅✅✅ TARGET ACHIEVED! ✅✅✅");
        println!("   Accuracy: {:.2}% >= 95.0%", best_val_acc * 100.0);
        println!("   Improvement: {:.2}% over baseline", (best_val_acc - 0.50) * 100.0);
    } else if best_val_acc >= 0.90 {
        println!("✅ EXCELLENT PERFORMANCE!");
        println!("   Accuracy: {:.2}% (publication quality)", best_val_acc * 100.0);
        println!("   Gap to 95%: {:.2}%", (0.95 - best_val_acc) * 100.0);
    } else if best_val_acc >= 0.85 {
        println!("✅ STRONG PERFORMANCE!");
        println!("   Accuracy: {:.2}%", best_val_acc * 100.0);
        println!("   Gap to 95%: {:.2}%", (0.95 - best_val_acc) * 100.0);
    } else {
        println!("🎯 Progress Made:");
        println!("   Accuracy: {:.2}%", best_val_acc * 100.0);
        println!("   Improvement: {:.2}% over baseline", (best_val_acc - 0.50) * 100.0);
        println!("   Gap to 95%: {:.2}%", (0.95 - best_val_acc) * 100.0);
    }
    
    if best_val_loss <= 0.01 {
        println!("✅ Loss target achieved: {:.4} <= 0.01", best_val_loss);
    } else {
        println!("   Loss: {:.4} (target: 0.01)", best_val_loss);
    }
    
    Ok(())
}
