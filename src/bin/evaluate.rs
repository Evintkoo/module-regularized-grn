/// Comprehensive evaluation of hybrid model
use module_regularized_grn::{
    Config,
    models::{hybrid_embeddings::HybridEmbeddingModel, nn::{bce_loss, bce_loss_backward}},
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
    evaluation::EvaluationMetrics,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

fn main() -> Result<()> {
    println!("=== Comprehensive Model Evaluation ===\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load prior knowledge
    println!("Loading prior knowledge...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    // Load expression data
    println!("Loading expression data...");
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    
    let tf_vocab = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();
    
    // Build gene name mapping
    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, gene_name) in expr_data.gene_names.iter().enumerate() {
        gene_to_expr_idx.insert(gene_name.clone(), i);
    }
    
    let expr_dim = expr_data.n_cell_types;
    
    // Map genes to expression
    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    
    for (_, &tf_idx) in tf_vocab.iter() {
        tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
    }
    
    for (_, &gene_idx) in gene_vocab.iter() {
        gene_expr_map.insert(gene_idx, Array1::zeros(expr_dim));
    }
    
    println!("  TFs: {} | Genes: {} | Expression dims: {}\n", num_tfs, num_genes, expr_dim);
    
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
    
    // Model hyperparameters
    let embed_dim = 128;
    let hidden_dim = 256;
    let output_dim = 128;
    let learning_rate = 0.005_f32;
    let batch_size = 64;
    let num_epochs = 50;
    
    println!("Training model...");
    let mut model = HybridEmbeddingModel::new(
        num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, 0.07, 0.01, seed
    );
    
    // Training loop
    for epoch in 0..num_epochs {
        model.zero_grad();
        
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
        }
        
        model.update(learning_rate);
        
        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            print!("  Epoch {}/{}\r", epoch, num_epochs);
            std::io::stdout().flush().ok();
        }
    }
    
    println!("\n\nEvaluating on validation set...");
    
    // Collect all predictions
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
    
    // Compute metrics
    let metrics = EvaluationMetrics::compute(&predictions, &labels, 0.5);
    metrics.print();
    
    // Save metrics to JSON
    let metrics_json = serde_json::to_string_pretty(&metrics)?;
    let mut file = File::create("results/evaluation_metrics.json")?;
    file.write_all(metrics_json.as_bytes())?;
    println!("\n✅ Metrics saved to results/evaluation_metrics.json");
    
    // Save predictions for plotting
    let results = serde_json::json!({
        "predictions": predictions.to_vec(),
        "labels": labels.to_vec(),
        "metrics": metrics,
    });
    let mut file = File::create("results/predictions.json")?;
    file.write_all(serde_json::to_string_pretty(&results)?.as_bytes())?;
    println!("✅ Predictions saved to results/predictions.json");
    
    // Summary
    println!("\n=== Summary ===");
    println!("Model: Hybrid (Embeddings + Expression)");
    println!("Samples: {} train, {} validation", train_samples.len(), val_samples.len());
    println!("Parameters: {}", model.num_parameters());
    println!("\n🎯 Key Results:");
    println!("  Accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("  AUROC:    {:.4}", metrics.auroc);
    println!("  AUPRC:    {:.4}", metrics.auprc);
    println!("  F1 Score: {:.4}", metrics.f1_score);
    
    if metrics.auroc >= 0.80 {
        println!("\n✅ Excellent AUROC (≥0.80)!");
    } else if metrics.auroc >= 0.70 {
        println!("\n✅ Good AUROC (≥0.70)");
    }
    
    if metrics.accuracy >= 0.70 {
        println!("✅ Exceeded 70% accuracy target!");
    }
    
    Ok(())
}
