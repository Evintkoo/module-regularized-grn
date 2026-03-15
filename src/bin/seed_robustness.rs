/// Test model robustness across multiple random seeds
use module_regularized_grn::{
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

fn train_and_evaluate(seed: u64) -> Result<EvaluationMetrics> {
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load data
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    
    let expr_dim = expr_data.n_cell_types;
    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    
    for tf_idx in 0..num_tfs {
        tf_expr_map.insert(tf_idx, Array1::zeros(expr_dim));
    }
    for gene_idx in 0..num_genes {
        gene_expr_map.insert(gene_idx, Array1::zeros(expr_dim));
    }
    
    // Create dataset
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
    
    // Train model
    let embed_dim = 128;
    let hidden_dim = 256;
    let output_dim = 128;
    let learning_rate = 0.005_f32;
    let batch_size = 64;
    let num_epochs = 50;
    
    let mut model = HybridEmbeddingModel::new(
        num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, 0.07, 0.01, seed
    );
    
    for _epoch in 0..num_epochs {
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
            let _loss = bce_loss(&scores_2d, &labels);
            let grad_2d = bce_loss_backward(&scores_2d, &labels);
            let grad = grad_2d.column(0).to_owned() / batch.len() as f32;
            
            model.backward(&grad);
        }
        
        model.update(learning_rate);
    }
    
    // Evaluate
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
    
    Ok(EvaluationMetrics::compute(&predictions, &labels, 0.5))
}

fn main() -> Result<()> {
    println!("=== Multi-Seed Robustness Test ===\n");
    println!("Testing model stability across different random seeds...\n");
    
    let seeds = vec![42, 123, 456, 789, 999];
    let mut results = Vec::new();
    
    for (i, &seed) in seeds.iter().enumerate() {
        println!("Running with seed {} ({}/5)...", seed, i + 1);
        let metrics = train_and_evaluate(seed)?;
        println!("  Accuracy: {:.4} ({:.2}%)", metrics.accuracy, metrics.accuracy * 100.0);
        println!("  AUROC: {:.4}", metrics.auroc);
        println!();
        results.push(metrics);
    }
    
    // Compute statistics
    let accuracies: Vec<f32> = results.iter().map(|m| m.accuracy).collect();
    let aurocs: Vec<f32> = results.iter().map(|m| m.auroc).collect();
    let precisions: Vec<f32> = results.iter().map(|m| m.precision).collect();
    let recalls: Vec<f32> = results.iter().map(|m| m.recall).collect();
    
    let mean_acc = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
    let std_acc = (accuracies.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>() / accuracies.len() as f32).sqrt();
    
    let mean_auroc = aurocs.iter().sum::<f32>() / aurocs.len() as f32;
    let std_auroc = (aurocs.iter().map(|x| (x - mean_auroc).powi(2)).sum::<f32>() / aurocs.len() as f32).sqrt();
    
    let mean_prec = precisions.iter().sum::<f32>() / precisions.len() as f32;
    let mean_rec = recalls.iter().sum::<f32>() / recalls.len() as f32;
    
    println!("\n=== Summary Statistics ===");
    println!("\nAccuracy:");
    println!("  Mean: {:.4} ({:.2}%)", mean_acc, mean_acc * 100.0);
    println!("  Std:  {:.4} ({:.2}%)", std_acc, std_acc * 100.0);
    println!("  Min:  {:.4} ({:.2}%)", accuracies.iter().copied().fold(f32::INFINITY, f32::min), 
             accuracies.iter().copied().fold(f32::INFINITY, f32::min) * 100.0);
    println!("  Max:  {:.4} ({:.2}%)", accuracies.iter().copied().fold(f32::NEG_INFINITY, f32::max),
             accuracies.iter().copied().fold(f32::NEG_INFINITY, f32::max) * 100.0);
    
    println!("\nAUROC:");
    println!("  Mean: {:.4}", mean_auroc);
    println!("  Std:  {:.4}", std_auroc);
    println!("  Min:  {:.4}", aurocs.iter().copied().fold(f32::INFINITY, f32::min));
    println!("  Max:  {:.4}", aurocs.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    
    println!("\nPrecision: {:.4} ({:.2}%)", mean_prec, mean_prec * 100.0);
    println!("Recall:    {:.4} ({:.2}%)", mean_rec, mean_rec * 100.0);
    
    // Save results
    let results_json = serde_json::json!({
        "seeds": seeds,
        "accuracies": accuracies,
        "aurocs": aurocs,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_auroc": mean_auroc,
        "std_auroc": std_auroc,
    });
    
    std::fs::write("results/seed_robustness.json", serde_json::to_string_pretty(&results_json)?)?;
    println!("\n✅ Results saved to results/seed_robustness.json");
    
    // Assessment
    println!("\n🎯 Robustness Assessment:");
    if std_acc < 0.02 {
        println!("✅ EXCELLENT: Very stable across seeds (std < 2%)");
    } else if std_acc < 0.05 {
        println!("✅ GOOD: Stable across seeds (std < 5%)");
    } else {
        println!("⚠️  MODERATE: Some variability across seeds (std >= 5%)");
    }
    
    println!("\n📊 Confidence Interval (95%, assuming normal):");
    println!("  Accuracy: {:.2}% ± {:.2}%", 
             mean_acc * 100.0, 1.96 * std_acc * 100.0);
    println!("  AUROC:    {:.4} ± {:.4}", mean_auroc, 1.96 * std_auroc);
    
    Ok(())
}
