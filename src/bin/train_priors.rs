/// Training with real prior knowledge
use module_regularized_grn::{
    Config,
    models::{two_tower::TwoTowerModel, nn::{bce_loss, bce_loss_backward}},
    data::{PriorKnowledge, PriorDatasetBuilder, GRNDataset, DataLoader},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== GRN Training with Real Prior Knowledge ===\n");

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
    println!("  Avg targets/TF: {:.2}", stats.avg_targets_per_tf);
    println!();
    
    // Create dataset builder
    let builder = PriorDatasetBuilder::new(priors);
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    println!("Creating dataset from priors...");
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), seed);
    
    println!("  Positive examples: {}", positives.len());
    println!("  Negative examples: {}", negatives.len());
    println!();
    
    // Create feature matrices (one-hot encoding for now)
    // In real scenario, these would be expression values
    let total_samples = positives.len() + negatives.len();
    
    let mut tf_features = Array2::<f32>::zeros((total_samples, num_tfs));
    let mut gene_features = Array2::<f32>::zeros((total_samples, num_genes));
    let mut labels = Array1::<f32>::zeros(total_samples);
    
    // Add positive examples
    for (i, (tf_idx, gene_idx)) in positives.iter().enumerate() {
        tf_features[[i, *tf_idx]] = 1.0;
        gene_features[[i, *gene_idx]] = 1.0;
        labels[i] = 1.0;
    }
    
    // Add negative examples
    let offset = positives.len();
    for (i, (tf_idx, gene_idx)) in negatives.iter().enumerate() {
        tf_features[[offset + i, *tf_idx]] = 1.0;
        gene_features[[offset + i, *gene_idx]] = 1.0;
        labels[offset + i] = 0.0;
    }
    
    println!("Creating dataset...");
    let dataset = GRNDataset::new(tf_features, gene_features, labels);
    let (train_dataset, val_dataset) = dataset.split(0.2, seed);
    
    println!("  Train samples: {}", train_dataset.n_samples);
    println!("  Val samples: {}", val_dataset.n_samples);
    println!();
    
    // Hyperparameters
    let hidden_dim = 256;
    let embed_dim = 128;
    let dropout = 0.1;
    let temperature = 0.07;
    let learning_rate = 0.0001_f32;
    let batch_size = 256;
    let num_epochs = 20;
    
    println!("Hyperparameters:");
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Embed dim: {}", embed_dim);
    println!("  Dropout: {}", dropout);
    println!("  Temperature: {}", temperature);
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", num_epochs);
    println!();
    
    // Create model
    println!("Creating Two-Tower model...");
    let mut model = TwoTowerModel::new(
        num_tfs,
        num_genes,
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
    let mut best_val_loss = f32::INFINITY;
    
    for epoch in 0..num_epochs {
        // Training
        model.zero_grad();
        let mut train_loader = DataLoader::new(
            train_dataset.clone(),
            batch_size,
            true
        );
        train_loader.reset(&mut rng);
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        for batch in train_loader {
            // Forward pass
            let scores = model.forward(&batch.tf_input, &batch.gene_input, true, &mut rng);
            
            // Use diagonal scores (self-similarity) as predictions
            let batch_size = batch.size;
            let diagonal_scores = Array2::from_shape_fn((batch_size, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() {
                    scores[[i, i]]
                } else {
                    0.0
                }
            });
            
            let targets = batch.labels.slice(ndarray::s![..batch_size]).to_owned();
            let loss = bce_loss(&diagonal_scores, &targets);
            epoch_loss += loss;
            num_batches += 1;
            
            // Backward pass
            let grad_loss = bce_loss_backward(&diagonal_scores, &targets);
            
            let mut grad_scores = Array2::zeros(scores.dim());
            for i in 0..batch_size {
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
        let mut val_loader = DataLoader::new(
            val_dataset.clone(),
            batch_size,
            false
        );
        
        let mut val_loss = 0.0;
        let mut val_batches = 0;
        let mut val_correct = 0;
        let mut val_total = 0;
        
        for batch in val_loader {
            let scores = model.forward(&batch.tf_input, &batch.gene_input, false, &mut rng);
            
            let batch_size = batch.size;
            let diagonal_scores = Array2::from_shape_fn((batch_size, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() {
                    scores[[i, i]]
                } else {
                    0.0
                }
            });
            
            let targets = batch.labels.slice(ndarray::s![..batch_size]).to_owned();
            let loss = bce_loss(&diagonal_scores, &targets);
            val_loss += loss;
            val_batches += 1;
            
            // Compute accuracy
            for i in 0..batch_size {
                if i < diagonal_scores.nrows() {
                    let pred = if diagonal_scores[[i, 0]] > 0.0 { 1.0 } else { 0.0 };
                    if pred == targets[i] {
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
        }
        
        // Print progress
        if epoch % 2 == 0 || epoch == num_epochs - 1 {
            println!(
                "Epoch {:2} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.4}",
                epoch, avg_train_loss, avg_val_loss, val_accuracy
            );
        }
    }
    
    println!("\n=== Training Complete ===");
    println!("Best validation loss: {:.4}", best_val_loss);
    println!("✅ Model trained on real prior knowledge!");
    
    Ok(())
}
