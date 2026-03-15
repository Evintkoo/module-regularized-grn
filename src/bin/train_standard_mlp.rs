/// Standard MLP Training: Two-pathway feedforward network with embeddings + expression
/// Architecture: NOT a transformer - uses standard linear layers with ReLU
/// Similarity: Cosine similarity with temperature scaling (NOT attention)
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
    println!("=== Standard MLP Training (NOT Transformer) ===");
    println!("Architecture: Two-pathway feedforward network");
    println!("  - Pathway 1: TF encoder (embedding + expression → hidden → output)");
    println!("  - Pathway 2: Gene encoder (embedding + expression → hidden → output)");
    println!("  - Similarity: Cosine similarity with temperature scaling");
    println!("  - NO attention mechanism, NO transformer layers\n");

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
    
    // Loaded once before training loop
    // Load expression data
    println!("Loading expression data...");
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    println!("  Cell types: {} | Expression genes: {}", 
             expr_data.n_cell_types, expr_data.n_genes);
    
    // Build gene name to expression index mapping
    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, gene_name) in expr_data.gene_names.iter().enumerate() {
        gene_to_expr_idx.insert(gene_name.clone(), i);
    }
    
    // Get vocabularies
    let tf_vocab = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();
    
    // Map genes to expression profiles
    println!("Mapping genes to expression features...");
    let expr_dim = expr_data.n_cell_types;
    
    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    
    let mut tf_found = 0;
    let mut gene_found = 0;
    
    for (tf_name, &tf_idx) in tf_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(tf_name) {
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            tf_expr_map.insert(tf_idx, expr_profile);
            tf_found += 1;
        } else {
            tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
        }
    }
    
    for (gene_name, &gene_idx) in gene_vocab.iter() {
        if let Some(&expr_idx) = gene_to_expr_idx.get(gene_name) {
            let expr_profile = expr_data.expression.column(expr_idx).to_owned();
            gene_expr_map.insert(gene_idx, expr_profile);
            gene_found += 1;
        } else {
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
    
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, gene)| (tf, gene, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, gene)| (tf, gene, 0.0)));
    examples.shuffle(&mut rng);
    
    let n_total = examples.len();
    let n_train = (n_total as f32 * 0.7) as usize;
    let n_val = (n_total as f32 * 0.15) as usize;
    
    let train_data = &examples[..n_train];
    let val_data = &examples[n_train..n_train+n_val];
    let test_data = &examples[n_train+n_val..];
    
    println!("  Total: {} | Train: {} | Val: {} | Test: {}\n",
             n_total, train_data.len(), val_data.len(), test_data.len());
    
    // Model hyperparameters
    let embed_dim = 64;
    let hidden_dim = 128;
    let output_dim = 64;
    let temperature = 0.1;
    
    println!("Model Architecture (Standard MLP):");
    println!("  Input: Embedding ({}d) + Expression ({}d)", embed_dim, expr_dim);
    println!("  Hidden layer: {} dimensions with ReLU", hidden_dim);
    println!("  Output encoding: {} dimensions", output_dim);
    println!("  Similarity: Cosine with temperature={}", temperature);
    println!("  Total params: ~{}", estimate_params(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim));
    println!();
    
    // Initialize model
    let mut model = HybridEmbeddingModel::new(
        num_tfs,
        num_genes,
        embed_dim,
        expr_dim,
        hidden_dim,
        output_dim,
        temperature,
        0.1,  // std_dev for init
        seed,
    );
    
    // Training parameters
    let learning_rate = 0.001;
    let batch_size = 256;
    let epochs = 50;
    let weight_decay = 0.01;
    
    println!("Training Configuration:");
    println!("  Optimizer: Adam (standard, not AdamW)");
    println!("  Learning rate: {}", learning_rate);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}", epochs);
    println!("  Weight decay (L2): {}", weight_decay);
    println!();
    
    println!("Starting training...\n");
    
    let mut best_val_loss = f32::INFINITY;
    let mut patience_counter = 0;
    let patience = 10;
    
    for epoch in 0..epochs {
        // Training phase
        let mut train_loss = 0.0;
        let mut train_batches = 0;
        
        for batch_start in (0..train_data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_data.len());
            let batch = &train_data[batch_start..batch_end];
            
            let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
            
            // Build expression batches
            let tf_expr_batch = build_expr_batch(&tf_indices, &tf_expr_map, expr_dim);
            let gene_expr_batch = build_expr_batch(&gene_indices, &gene_expr_map, expr_dim);
            
            // Forward pass
            let predictions = model.forward(
                &tf_indices,
                &gene_indices,
                &tf_expr_batch,
                &gene_expr_batch,
            );
            
            let labels_arr = Array1::from(labels);
            
            // BCE expects Array2 with shape [batch, 1]
            let predictions_2d = predictions.clone().insert_axis(ndarray::Axis(1));
            let loss = bce_loss(&predictions_2d, &labels_arr);
            let grad_2d = bce_loss_backward(&predictions_2d, &labels_arr);
            
            // Extract back to Array1
            let grad = grad_2d.column(0).to_owned();
            
            // Backward pass
            model.backward(&grad);
            
            train_loss += loss;
            train_batches += 1;
        }
        
        // Update parameters after all batches
        model.update(learning_rate);
        model.zero_grad();
        
        let avg_train_loss = train_loss / train_batches as f32;
        
        // Validation phase
        let mut val_loss = 0.0;
        let mut val_correct = 0;
        let mut val_total = 0;
        
        for batch_start in (0..val_data.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(val_data.len());
            let batch = &val_data[batch_start..batch_end];
            
            let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
            
            let tf_expr_batch = build_expr_batch(&tf_indices, &tf_expr_map, expr_dim);
            let gene_expr_batch = build_expr_batch(&gene_indices, &gene_expr_map, expr_dim);
            
            let predictions = model.forward(
                &tf_indices,
                &gene_indices,
                &tf_expr_batch,
                &gene_expr_batch,
            );
            
            let labels_arr = Array1::from(labels.clone());
            let predictions_2d = predictions.clone().insert_axis(ndarray::Axis(1));
            val_loss += bce_loss(&predictions_2d, &labels_arr);
            
            for (pred, label) in predictions.iter().zip(labels.iter()) {
                let pred_class = if *pred >= 0.5 { 1.0 } else { 0.0 };
                if pred_class == *label {
                    val_correct += 1;
                }
                val_total += 1;
            }
        }
        
        let avg_val_loss = val_loss / ((val_data.len() as f32 / batch_size as f32).ceil());
        let val_acc = val_correct as f32 / val_total as f32;
        
        println!("Epoch {:2} | Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.2}%",
                 epoch + 1, avg_train_loss, avg_val_loss, val_acc * 100.0);
        
        // Early stopping
        if avg_val_loss < best_val_loss {
            best_val_loss = avg_val_loss;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= patience {
                println!("\nEarly stopping triggered at epoch {}", epoch + 1);
                break;
            }
        }
    }
    
    // Final test evaluation
    println!("\n=== Test Set Evaluation ===");
    
    let mut test_correct = 0;
    let mut test_total = 0;
    let mut tp = 0;
    let mut tn = 0;
    let mut fp = 0;
    let mut fn_count = 0;
    
    for batch_start in (0..test_data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(test_data.len());
        let batch = &test_data[batch_start..batch_end];
        
        let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
        
        let tf_expr_batch = build_expr_batch(&tf_indices, &tf_expr_map, expr_dim);
        let gene_expr_batch = build_expr_batch(&gene_indices, &gene_expr_map, expr_dim);
        
        let predictions = model.forward(
            &tf_indices,
            &gene_indices,
            &tf_expr_batch,
            &gene_expr_batch,
        );
        
        for (pred, label) in predictions.iter().zip(labels.iter()) {
            let pred_class = if *pred >= 0.5 { 1.0 } else { 0.0 };
            let true_class = *label;
            
            if pred_class == true_class {
                test_correct += 1;
            }
            
            match (pred_class as i32, true_class as i32) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fp += 1,
                (0, 1) => fn_count += 1,
                _ => {}
            }
            
            test_total += 1;
        }
    }
    
    let accuracy = test_correct as f32 / test_total as f32;
    let precision = tp as f32 / (tp + fp) as f32;
    let recall = tp as f32 / (tp + fn_count) as f32;
    let f1 = 2.0 * precision * recall / (precision + recall);
    
    println!("Test Results:");
    println!("  Accuracy:  {:.2}%", accuracy * 100.0);
    println!("  Precision: {:.2}%", precision * 100.0);
    println!("  Recall:    {:.2}%", recall * 100.0);
    println!("  F1-Score:  {:.4}", f1);
    println!("\nConfusion Matrix:");
    println!("  TP: {} | FP: {}", tp, fp);
    println!("  FN: {} | TN: {}", fn_count, tn);
    
    // Save results
    let results = serde_json::json!({
        "architecture": "Standard MLP (Two-pathway)",
        "note": "NOT a transformer - uses feedforward layers only",
        "model": {
            "embed_dim": embed_dim,
            "expr_dim": expr_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "temperature": temperature,
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn_count,
        },
        "training": {
            "epochs_completed": epochs,
            "best_val_loss": best_val_loss,
        }
    });
    
    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/standard_mlp_results.json",
        serde_json::to_string_pretty(&results)?
    )?;
    
    println!("\n✓ Results saved to: results/standard_mlp_results.json");
    println!("\n=== Training Complete ===");
    
    Ok(())
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
    let fc1_params = input_dim * hidden_dim + hidden_dim;  // weights + biases
    let fc2_params = hidden_dim * output_dim + output_dim;
    
    // Two pathways
    let total = tf_embed + gene_embed + 2 * (fc1_params + fc2_params);
    total
}
