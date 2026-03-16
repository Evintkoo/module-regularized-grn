/// Simple training example for Two-Tower model
use module_regularized_grn::{
    Config,
    models::{two_tower::TwoTowerModel, nn::{bce_loss, bce_loss_backward}},
    data::{create_dummy_dataset, DataLoader},
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Two-Tower GRN Training Example ===\n");

    // Load config
    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Hyperparameters
    let tf_dim = 128;
    let gene_dim = 128;
    let hidden_dim = 256;
    let embed_dim = 128;
    let dropout = 0.1;
    let temperature = 0.07;
    let learning_rate = config.training.learning_rate as f32;
    let batch_size = config.training.batch_size;
    let num_epochs = 10; // Small test
    
    println!("Hyperparameters:");
    println!("  Batch size: {}", batch_size);
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", num_epochs);
    println!();
    
    // Create dummy dataset
    println!("Creating dummy dataset...");
    let dataset = create_dummy_dataset(1000, tf_dim, gene_dim, seed);
    let (train_dataset, val_dataset) = dataset.split(0.2, seed);
    
    println!("  Train samples: {}", train_dataset.n_samples);
    println!("  Val samples: {}", val_dataset.n_samples);
    println!();
    
    // Create model
    println!("Creating Two-Tower model...");
    let mut model = TwoTowerModel::new(
        tf_dim,
        gene_dim,
        hidden_dim,
        embed_dim,
        dropout,
        temperature,
        seed,
    );
    
    let params = model.count_parameters();
    println!("  Parameters: {}", params);
    println!();
    
    // Training loop
    println!("Starting training...\n");
    
    for epoch in 0..num_epochs {
        // Training
        model.zero_grad();
        let mut train_loader = DataLoader::new(
            train_dataset.clone(),
            batch_size,
            true  // shuffle
        );
        train_loader.reset(&mut rng);
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        for batch in train_loader {
            // Forward pass
            let scores = model.forward(&batch.tf_input, &batch.gene_input, true, &mut rng);
            
            // For simplicity, use diagonal as targets (self-similarity)
            let targets = ndarray::Array1::from_vec(
                (0..batch.size).map(|i| {
                    if i < batch.labels.len() {
                        batch.labels[i]
                    } else {
                        0.0
                    }
                }).collect()
            );
            
            // Compute loss on diagonal scores
            let diagonal_scores = ndarray::Array2::from_shape_fn((batch.size, 1), |(i, _)| {
                scores[[i, i.min(scores.ncols() - 1)]]
            });
            
            let loss = bce_loss(&diagonal_scores, &targets);
            epoch_loss += loss;
            num_batches += 1;
            
            // Backward pass
            let grad_loss = bce_loss_backward(&diagonal_scores, &targets);
            
            // Expand gradient to full score matrix (simplified)
            let mut grad_scores = ndarray::Array2::zeros(scores.dim());
            for i in 0..batch.size {
                if i < grad_scores.ncols() {
                    grad_scores[[i, i]] = grad_loss[[i, 0]];
                }
            }
            
            model.backward(&grad_scores);
            
            // Update weights
            model.update(learning_rate);
            model.zero_grad();
        }
        
        let avg_train_loss = epoch_loss / num_batches as f32;
        
        // Validation
        let mut val_loader = DataLoader::new(
            val_dataset.clone(),
            batch_size,
            false  // no shuffle
        );
        
        let mut val_loss = 0.0;
        let mut val_batches = 0;
        
        for batch in val_loader {
            let scores = model.forward(&batch.tf_input, &batch.gene_input, false, &mut rng);
            
            let targets = ndarray::Array1::from_vec(
                (0..batch.size).map(|i| {
                    if i < batch.labels.len() {
                        batch.labels[i]
                    } else {
                        0.0
                    }
                }).collect()
            );
            
            let diagonal_scores = ndarray::Array2::from_shape_fn((batch.size, 1), |(i, _)| {
                scores[[i, i.min(scores.ncols() - 1)]]
            });
            
            let loss = bce_loss(&diagonal_scores, &targets);
            val_loss += loss;
            val_batches += 1;
        }
        
        let avg_val_loss = val_loss / val_batches as f32;
        
        // Print progress
        if epoch % 2 == 0 {
            println!(
                "Epoch {:2} | Train Loss: {:.4} | Val Loss: {:.4}",
                epoch, avg_train_loss, avg_val_loss
            );
        }
    }
    
    println!("\n=== Training Complete ===");
    println!("✅ Model successfully trained with backpropagation!");
    
    Ok(())
}
