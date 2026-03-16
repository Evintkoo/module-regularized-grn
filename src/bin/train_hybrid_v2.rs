/// Hybrid training: Embeddings + Expression features
/// Target: 70%+ accuracy
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

fn main() -> Result<()> {
    println!("=== Hybrid Training: Embeddings + Expression ===\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load prior knowledge
    println!("Loading prior knowledge...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    println!("  TFs: {} | Genes: {}", num_tfs, num_genes);
    
    // Load expression data
    println!("Loading expression data...");
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    println!("  Cell types: {} | Expression genes: {}", 
             expr_data.n_cell_types, expr_data.n_genes);
    
    // Build gene name to expression index mapping
    // Load ENSEMBL to Symbol mapping for real gene matching
    println!("Loading ENSEMBL to Symbol mapping...");
    let ensembl_to_symbol_path = "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd/ensembl_to_symbol.json";
    let ensembl_to_symbol: HashMap<String, String> = {
        let file = std::fs::File::open(ensembl_to_symbol_path)?;
        serde_json::from_reader(file)?
    };
    
    // Build gene symbol to expression index mapping
    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, ensembl_id) in expr_data.gene_names.iter().enumerate() {
        // Convert ENSEMBL to symbol if possible
        if let Some(symbol) = ensembl_to_symbol.get(ensembl_id) {
            gene_to_expr_idx.insert(symbol.clone(), i);
        }
    }
    
    println!("  Converted {} ENSEMBL IDs to gene symbols", gene_to_expr_idx.len());
    
    // Average expression across cell types for global signal
    let avg_expression = expr_data.expression.mean_axis(ndarray::Axis(0)).unwrap();
    println!("  Expression stats: min={:.3}, max={:.3}, mean={:.3}\n",
             avg_expression.iter().copied().fold(f32::INFINITY, f32::min),
             avg_expression.iter().copied().fold(f32::NEG_INFINITY, f32::max),
             avg_expression.mean().unwrap());
    
    // Get vocabularies for name lookups
    let tf_vocab = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();
    
    // Create expression lookup: for each TF/gene, get expression vector
    println!("Mapping genes to expression features...");
    let expr_dim = expr_data.n_cell_types;  // Use cell types as features, not genes!
    
    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    
    let mut tf_found = 0;
    let mut gene_found = 0;
    
    // Map TFs to their expression profiles across cell types
    for (tf_name, &tf_idx) in tf_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(tf_name) {
            // Get expression across cell types for this specific TF
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            tf_expr_map.insert(tf_idx, expr_profile);
            tf_found += 1;
        } else {
            // TF not in expression data, use zeros
            tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
        }
    }
    
    // Map genes to their expression profiles across cell types
    for (gene_name, &gene_idx) in gene_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(gene_name) {
            // Get expression across cell types for this specific gene
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            gene_expr_map.insert(gene_idx, expr_profile);
            gene_found += 1;
        } else {
            // Gene not in expression data, use zeros
            gene_expr_map.insert(gene_idx, Array1::zeros(expr_dim));
        }
    }
    
    println!("  TFs with expression: {}/{} ({:.1}%)", 
             tf_found, num_tfs, 100.0 * tf_found as f32 / num_tfs as f32);
    println!("  Genes with expression: {}/{} ({:.1}%)\n",
             gene_found, num_genes, 100.0 * gene_found as f32 / num_genes as f32);
    
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
    
    // Create hybrid model
    let embed_dim = 128;
    let hidden_dim = 256;
    let output_dim = 128;
    let learning_rate = 0.005_f32;  // Higher LR for expression integration
    let batch_size = 64;  // Smaller batch for faster iterations
    let num_epochs = 50;  // Fewer epochs for debugging
    
    println!("Hybrid Model Configuration:");
    println!("  Embedding dim: {}", embed_dim);
    println!("  Expression dim: {} (cell types)", expr_dim);
    println!("  Input dim: {} (embed + expr)", embed_dim + expr_dim);
    println!("  Hidden dim: {}", hidden_dim);
    println!("  Output dim: {}", output_dim);
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}\n", num_epochs);
    
    let mut model = HybridEmbeddingModel::new(
        num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, 0.07, 0.01, seed
    );
    
    println!("  Total parameters: {}\n", model.num_parameters());
    
    let mut best_val_loss = f32::INFINITY;
    let mut best_val_acc = 0.0;
    
    println!("Training with expression features...\n");
    
    for epoch in 0..num_epochs {
        model.zero_grad();
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        // Training loop
        for batch_start in (0..train_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_samples.len());
            let batch = &train_samples[batch_start..batch_end];
            
            let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            // Gather expression features for batch
            let mut tf_expr_batch = Array2::zeros((batch.len(), expr_dim));
            let mut gene_expr_batch = Array2::zeros((batch.len(), expr_dim));
            
            for (i, &tf_idx) in tf_indices.iter().enumerate() {
                tf_expr_batch.row_mut(i).assign(&tf_expr_map[&tf_idx]);
            }
            for (i, &gene_idx) in gene_indices.iter().enumerate() {
                gene_expr_batch.row_mut(i).assign(&gene_expr_map[&gene_idx]);
            }
            
            // Forward pass
            let scores = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
            
            // BCE expects scores as Array2 with shape [batch, 1]
            let scores_2d = scores.clone().insert_axis(ndarray::Axis(1));
            let loss = bce_loss(&scores_2d, &labels);
            let grad_2d = bce_loss_backward(&scores_2d, &labels);
            // Extract back to Array1
            let grad = grad_2d.column(0).to_owned() / batch.len() as f32;
            
            // Backward pass
            model.backward(&grad);
            
            epoch_loss += loss;
            num_batches += 1;
        }
        
        // Update parameters
        model.update(learning_rate);
        
        let train_loss = epoch_loss / num_batches as f32;
        
        // Gradient monitoring (every 10 epochs)
        if epoch % 10 == 0 {
            let tf_grad_norm = model.tf_embed_grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            let gene_grad_norm = model.gene_embed_grad.iter().map(|x| x * x).sum::<f32>().sqrt();
            
            if tf_grad_norm.is_nan() || gene_grad_norm.is_nan() {
                println!("❌ NaN detected in gradients!");
                break;
            }
        }
        
        // Validation
        let mut val_loss = 0.0;
        let mut val_correct = 0;
        let mut val_batches = 0;
        
        for batch_start in (0..val_samples.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(val_samples.len());
            let batch = &val_samples[batch_start..batch_end];
            
            let tf_indices: Vec<usize> = batch.iter().map(|(tf, _, _)| *tf).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|(_, gene, _)| *gene).collect();
            let labels = Array1::from_vec(batch.iter().map(|(_, _, label)| *label).collect());
            
            // Gather expression
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
        
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            best_val_acc = val_acc;
        }
        
        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            println!("Epoch {:3} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.4}",
                     epoch, train_loss, val_loss, val_acc);
        }
    }
    
    println!("\n=== Training Complete ===");
    println!("Best validation loss: {:.4}", best_val_loss);
    println!("Best validation accuracy: {:.4}\n", best_val_acc);
    
    // Compare to baseline
    let baseline_acc = 0.5782;
    let improvement = best_val_acc - baseline_acc;
    
    println!("📊 Results:");
    println!("  Baseline (embeddings only): {:.2}%", baseline_acc * 100.0);
    println!("  Hybrid (embed + expression): {:.2}%", best_val_acc * 100.0);
    println!("  Improvement: {:+.2}% points\n", improvement * 100.0);
    
    if best_val_acc >= 0.70 {
        println!("✅ SUCCESS: Reached 70%+ accuracy target!");
    } else if best_val_acc >= 0.65 {
        println!("✅ GOOD: Significant improvement, approaching target");
    } else if improvement > 0.02 {
        println!("✅ PROGRESS: Expression features helping (+2% points)");
    } else {
        println!("⚠️  LIMITED: Expression not helping much yet");
    }
    
    Ok(())
}
