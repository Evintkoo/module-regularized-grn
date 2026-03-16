/// Training with learnable embeddings (Quick Win!)
use module_regularized_grn::{
    Config,
    models::{learnable_embeddings::EmbeddingTwoTower, nn::{bce_loss, bce_loss_backward}},
    data::{PriorKnowledge, PriorDatasetBuilder},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== GRN Training with Learnable Embeddings ===\n");

    // Load config
    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load prior knowledge
    println!("Loading prior knowledge...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let stats = priors.stats();
    
    println!("  TFs: {}", stats.num_tfs);
    println!("  Genes: {}", stats.num_genes);
    println!("  Edges: {}", stats.num_edges);
    println!();
    
    // Create dataset builder
    let builder = PriorDatasetBuilder::new(priors);
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    println!("Creating dataset...");
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), seed);
    
    // Combine and shuffle
    let mut all_samples: Vec<(usize, usize, f32)> = Vec::new();
    for (tf_idx, gene_idx) in positives {
        all_samples.push((tf_idx, gene_idx, 1.0));
    }
    for (tf_idx, gene_idx) in negatives {
        all_samples.push((tf_idx, gene_idx, 0.0));
    }
    all_samples.shuffle(&mut rng);
    
    // Split train/val
    let n_val = (all_samples.len() as f32 * 0.2) as usize;
    let n_train = all_samples.len() - n_val;
    
    let train_samples = &all_samples[..n_train];
    let val_samples = &all_samples[n_train..];
    
    println!("  Total samples: {}", all_samples.len());
    println!("  Train: {}", train_samples.len());
    println!("  Val: {}", val_samples.len());
    println!();
    
    // Hyperparameters
    let embed_dim = 128;
    let hidden_dim = 256;
    let output_dim = 128;
    let dropout = 0.0; // No dropout for now (not implemented in embedding model)
    let temperature = 0.07;
    let learning_rate = 0.001_f32; // Higher LR for embeddings
    let batch_size = 256;
    let num_epochs = 50;
    
    println!("Hyperparameters:");
    println!("  Embed dim: {}", embed_dim);
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Output dim: {}", output_dim);
    println!("  Temperature: {}", temperature);
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", num_epochs);
    println!();
    
    // Create model with learnable embeddings
    println!("Creating Embedding Two-Tower model...");
    let mut model = EmbeddingTwoTower::new(
        num_tfs,
        num_genes,
        embed_dim,
        hidden_dim,
        output_dim,
        dropout,
        temperature,
        seed,
    );
    
    let params = model.count_parameters();
    println!("  Parameters: {}", params);
    println!();
    
    // Training loop
    println!("Starting training...\n");
    let mut best_val_loss = f32::INFINITY;
    let mut best_val_acc = 0.0;
    
    for epoch in 0..num_epochs {
        // Training
        model.zero_grad();
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        // Process in batches
        for batch_start in (0..train_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_samples.len());
            let batch = &train_samples[batch_start..batch_end];
            
            let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            // Forward pass
            let scores = model.forward(&tf_indices, &gene_indices);
            
            // Extract diagonal (self-similarity scores)
            let batch_len = batch.len();
            let diagonal_scores = Array2::from_shape_fn((batch_len, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() {
                    scores[[i, i]]
                } else {
                    0.0
                }
            });
            
            let loss = bce_loss(&diagonal_scores, &labels);
            epoch_loss += loss;
            num_batches += 1;
            
            // Backward pass
            let grad_loss = bce_loss_backward(&diagonal_scores, &labels);
            
            let mut grad_scores = Array2::zeros(scores.dim());
            for i in 0..batch_len {
                if i < grad_scores.nrows() && i < grad_scores.ncols() {
                    grad_scores[[i, i]] = grad_loss[[i, 0]];
                }
            }
            
            model.backward(&grad_scores);
            model.update(learning_rate);
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
            
            let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            let scores = model.forward(&tf_indices, &gene_indices);
            
            let batch_len = batch.len();
            let diagonal_scores = Array2::from_shape_fn((batch_len, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() {
                    scores[[i, i]]
                } else {
                    0.0
                }
            });
            
            let loss = bce_loss(&diagonal_scores, &labels);
            val_loss += loss;
            val_batches += 1;
            
            // Compute accuracy
            for i in 0..batch_len {
                if i < diagonal_scores.nrows() {
                    let pred = if diagonal_scores[[i, 0]] > 0.0 { 1.0 } else { 0.0 };
                    if pred == labels[i] {
                        val_correct += 1;
                    }
                    val_total += 1;
                }
            }
        }
        
        let avg_val_loss = val_loss / val_batches as f32;
        let val_accuracy = val_correct as f32 / val_total as f32;
        
        // Track best model
        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            best_val_acc = val_accuracy;
        }
        
        // Print progress
        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            println!(
                "Epoch {:2} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.4}",
                epoch, avg_train_loss, avg_val_loss, val_accuracy
            );
        }
    }
    
    println!("\n=== Training Complete ===");
    println!("Best validation loss: {:.4}", best_val_loss);
    println!("Best validation accuracy: {:.4}", best_val_acc);
    
    if best_val_loss < 0.693 {
        println!("✅ SUCCESS: Model learned! (loss < log(2))");
    } else {
        println!("⚠️  Model at baseline (may need more epochs or tuning)");
    }
    
    if best_val_acc > 0.55 {
        println!("✅ SUCCESS: Better than random! (accuracy > 55%)");
    } else {
        println!("⚠️  Accuracy near random (may need more epochs)");
    }
    
    Ok(())
}
