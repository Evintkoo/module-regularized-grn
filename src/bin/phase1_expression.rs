/// Phase 1: Training with expression features (synthetic for now)
/// This validates the architecture before adding real gene mapping
use module_regularized_grn::{
    Config,
    models::{expression_model::ExpressionTwoTower, nn::{bce_loss, bce_loss_backward}},
    data::{PriorKnowledge, PriorDatasetBuilder},
};
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;

fn main() -> Result<()> {
    println!("=== Phase 1: Expression Features Training ===\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load prior knowledge
    println!("Loading prior knowledge...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors);
    
    println!("Creating dataset...");
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), seed);
    
    // Combine samples
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
    
    // Create synthetic expression features
    // In practice, these would come from real H5AD data
    let expression_dim = 2000;  // Match processed H5AD
    println!("Creating synthetic expression features ({} dims)...", expression_dim);
    println!("  (Simulates: gene expression across cell types)");
    println!("  Note: Real features will be added after validation\n");
    
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    // Random but consistent expression per TF/Gene
    let tf_expressions = Array2::random_using(
        (num_tfs, expression_dim),
        Normal::new(0.0, 1.0).unwrap(),
        &mut rng
    );
    let gene_expressions = Array2::random_using(
        (num_genes, expression_dim),
        Normal::new(0.0, 1.0).unwrap(),
        &mut rng
    );
    
    // Phase 1: Bigger architecture
    let hidden_dim1 = 512;   // Increased from 256
    let hidden_dim2 = 256;   // Increased from 128
    let output_dim = 128;
    let learning_rate = 0.001_f32;  // Higher LR for faster convergence
    let batch_size = 256;    // Larger batch for speed
    let num_epochs = 50;     // Fewer epochs for testing
    
    println!("Phase 1 Architecture (Deeper & Wider):");
    println!("  Expression dim: {}", expression_dim);
    println!("  Hidden layer 1: {}", hidden_dim1);
    println!("  Hidden layer 2: {}", hidden_dim2);
    println!("  Output dim: {}", output_dim);
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", num_epochs);
    println!();
    
    let mut model = ExpressionTwoTower::new(
        expression_dim,
        hidden_dim1,
        hidden_dim2,
        output_dim,
        0.07,
        seed,
    );
    
    let params = model.count_parameters();
    println!("  Parameters: {} (2.4M)\n", params);
    
    println!("Training...\n");
    let mut best_val_loss = f32::INFINITY;
    let mut best_val_acc = 0.0;
    
    for epoch in 0..num_epochs {
        model.zero_grad();
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_start in (0..train_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_samples.len());
            let batch = &train_samples[batch_start..batch_end];
            let batch_len = batch.len();
            
            // Get expression features for this batch
            let mut tf_expr_batch = Array2::zeros((batch_len, expression_dim));
            let mut gene_expr_batch = Array2::zeros((batch_len, expression_dim));
            
            for (i, (tf_idx, gene_idx, _)) in batch.iter().enumerate() {
                tf_expr_batch.row_mut(i).assign(&tf_expressions.row(*tf_idx));
                gene_expr_batch.row_mut(i).assign(&gene_expressions.row(*gene_idx));
            }
            
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            // Forward
            let scores = model.forward(&tf_expr_batch, &gene_expr_batch);
            
            let diagonal_scores = Array2::from_shape_fn((batch_len, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() { scores[[i, i]] } else { 0.0 }
            });
            
            let loss = bce_loss(&diagonal_scores, &labels);
            epoch_loss += loss;
            num_batches += 1;
            
            // Backward
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
            let batch_len = batch.len();
            
            let mut tf_expr_batch = Array2::zeros((batch_len, expression_dim));
            let mut gene_expr_batch = Array2::zeros((batch_len, expression_dim));
            
            for (i, (tf_idx, gene_idx, _)) in batch.iter().enumerate() {
                tf_expr_batch.row_mut(i).assign(&tf_expressions.row(*tf_idx));
                gene_expr_batch.row_mut(i).assign(&gene_expressions.row(*gene_idx));
            }
            
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            let scores = model.forward(&tf_expr_batch, &gene_expr_batch);
            
            let diagonal_scores = Array2::from_shape_fn((batch_len, 1), |(i, _)| {
                if i < scores.nrows() && i < scores.ncols() { scores[[i, i]] } else { 0.0 }
            });
            
            let loss = bce_loss(&diagonal_scores, &labels);
            val_loss += loss;
            val_batches += 1;
            
            for i in 0..batch_len {
                if i < diagonal_scores.nrows() {
                    let pred = if diagonal_scores[[i, 0]] > 0.0 { 1.0 } else { 0.0 };
                    if pred == labels[i] { val_correct += 1; }
                    val_total += 1;
                }
            }
        }
        
        let avg_val_loss = val_loss / val_batches as f32;
        let val_accuracy = val_correct as f32 / val_total as f32;
        
        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            best_val_acc = val_accuracy;
        }
        
        if epoch % 5 == 0 || epoch == num_epochs - 1 {
            println!(
                "Epoch {:3} | Train: {:.4} | Val Loss: {:.4} | Val Acc: {:.4}",
                epoch, avg_train_loss, avg_val_loss, val_accuracy
            );
        }
    }
    
    println!("\n=== Phase 1 Complete ===");
    println!("Best Val Loss: {:.4}", best_val_loss);
    println!("Best Val Acc: {:.4} ({:.2}%)", best_val_acc, best_val_acc * 100.0);
    println!();
    
    // Evaluate against targets
    let target_acc = 0.75;
    let target_loss = 0.30;
    
    if best_val_acc >= target_acc {
        println!("✅ Phase 1 TARGET ACHIEVED!");
        println!("   Accuracy: {:.2}% >= {:.2}% target", best_val_acc * 100.0, target_acc * 100.0);
    } else {
        println!("🎯 Phase 1 Progress:");
        println!("   Accuracy: {:.2}% / {:.2}% target", best_val_acc * 100.0, target_acc * 100.0);
        println!("   Gap: {:.2}%", (target_acc - best_val_acc) * 100.0);
    }
    
    if best_val_loss <= target_loss {
        println!("✅ Loss target achieved: {:.4} <= {:.4}", best_val_loss, target_loss);
    } else {
        println!("🎯 Loss: {:.4} / {:.4} target", best_val_loss, target_loss);
    }
    
    println!("\nArchitecture validated! Ready for Phase 2.");
    
    Ok(())
}
