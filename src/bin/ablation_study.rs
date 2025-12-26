/// Ablation study to understand component contributions
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

struct AblationConfig {
    name: String,
    embed_dim: usize,
    hidden_dim: usize,
    use_two_layers: bool,
    temperature: f32,
}

fn train_and_evaluate_ablation(config: &AblationConfig, seed: u64) -> Result<EvaluationMetrics> {
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
    
    // Train model with ablation config
    let output_dim = config.embed_dim;
    let learning_rate = 0.005_f32;
    let batch_size = 64;
    let num_epochs = 50;
    
    let mut model = HybridEmbeddingModel::new(
        num_tfs, num_genes, 
        config.embed_dim, 
        expr_dim, 
        config.hidden_dim, 
        output_dim, 
        config.temperature, 
        0.01, 
        seed
    );
    
    println!("  Training {} (embed={}, hidden={}, temp={:.2})...", 
             config.name, config.embed_dim, config.hidden_dim, config.temperature);
    
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
    println!("=== Ablation Study ===\n");
    println!("Testing contribution of different model components...\n");
    
    let seed = 42;
    
    // Define ablation configurations
    let configs = vec![
        AblationConfig {
            name: "Full Model (Baseline)".to_string(),
            embed_dim: 128,
            hidden_dim: 256,
            use_two_layers: true,
            temperature: 0.07,
        },
        AblationConfig {
            name: "Small Embeddings".to_string(),
            embed_dim: 64,
            hidden_dim: 256,
            use_two_layers: true,
            temperature: 0.07,
        },
        AblationConfig {
            name: "Large Embeddings".to_string(),
            embed_dim: 256,
            hidden_dim: 256,
            use_two_layers: true,
            temperature: 0.07,
        },
        AblationConfig {
            name: "Small Hidden".to_string(),
            embed_dim: 128,
            hidden_dim: 128,
            use_two_layers: true,
            temperature: 0.07,
        },
        AblationConfig {
            name: "Large Hidden".to_string(),
            embed_dim: 128,
            hidden_dim: 512,
            use_two_layers: true,
            temperature: 0.07,
        },
        AblationConfig {
            name: "High Temperature".to_string(),
            embed_dim: 128,
            hidden_dim: 256,
            use_two_layers: true,
            temperature: 0.20,
        },
        AblationConfig {
            name: "Low Temperature".to_string(),
            embed_dim: 128,
            hidden_dim: 256,
            use_two_layers: true,
            temperature: 0.01,
        },
    ];
    
    let mut results = Vec::new();
    
    for config in &configs {
        println!("\n{}", config.name);
        let metrics = train_and_evaluate_ablation(config, seed)?;
        println!("  Accuracy: {:.4} ({:.2}%)", metrics.accuracy, metrics.accuracy * 100.0);
        println!("  AUROC: {:.4}", metrics.auroc);
        println!("  Precision: {:.4}", metrics.precision);
        println!("  Recall: {:.4}", metrics.recall);
        
        results.push((config.name.clone(), metrics));
    }
    
    // Summary table
    println!("\n\n=== Summary Table ===\n");
    println!("{:<25} {:>10} {:>10} {:>10} {:>10}", "Configuration", "Accuracy", "AUROC", "Precision", "Recall");
    println!("{}", "-".repeat(75));
    
    let baseline_acc = results[0].1.accuracy;
    
    for (name, metrics) in &results {
        let diff = (metrics.accuracy - baseline_acc) * 100.0;
        let sign = if diff > 0.0 { "+" } else { "" };
        println!("{:<25} {:>9.2}% {:>10.4} {:>10.4} {:>10.4}  ({}{}%)",
                 name, 
                 metrics.accuracy * 100.0,
                 metrics.auroc,
                 metrics.precision,
                 metrics.recall,
                 sign,
                 diff);
    }
    
    // Analysis
    println!("\n\n=== Analysis ===\n");
    
    // Find best in each category
    let best_acc = results.iter().max_by(|a, b| a.1.accuracy.partial_cmp(&b.1.accuracy).unwrap()).unwrap();
    let worst_acc = results.iter().min_by(|a, b| a.1.accuracy.partial_cmp(&b.1.accuracy).unwrap()).unwrap();
    
    println!("Best Configuration: {}", best_acc.0);
    println!("  Accuracy: {:.2}%", best_acc.1.accuracy * 100.0);
    println!("  AUROC: {:.4}", best_acc.1.auroc);
    
    println!("\nWorst Configuration: {}", worst_acc.0);
    println!("  Accuracy: {:.2}%", worst_acc.1.accuracy * 100.0);
    println!("  AUROC: {:.4}", worst_acc.1.auroc);
    
    println!("\nRange: {:.2}%", (best_acc.1.accuracy - worst_acc.1.accuracy) * 100.0);
    
    // Key insights
    println!("\n📊 Key Insights:");
    
    let embed_64 = &results[1].1;
    let embed_128 = &results[0].1;
    let embed_256 = &results[2].1;
    
    println!("\n1. Embedding Size:");
    println!("   64 dims:  {:.2}%", embed_64.accuracy * 100.0);
    println!("   128 dims: {:.2}% (baseline)", embed_128.accuracy * 100.0);
    println!("   256 dims: {:.2}%", embed_256.accuracy * 100.0);
    
    let hidden_128 = &results[3].1;
    let hidden_256 = &results[0].1;
    let hidden_512 = &results[4].1;
    
    println!("\n2. Hidden Layer Size:");
    println!("   128 dims: {:.2}%", hidden_128.accuracy * 100.0);
    println!("   256 dims: {:.2}% (baseline)", hidden_256.accuracy * 100.0);
    println!("   512 dims: {:.2}%", hidden_512.accuracy * 100.0);
    
    let temp_low = &results[6].1;
    let temp_mid = &results[0].1;
    let temp_high = &results[5].1;
    
    println!("\n3. Temperature:");
    println!("   0.01: {:.2}%", temp_low.accuracy * 100.0);
    println!("   0.07: {:.2}% (baseline)", temp_mid.accuracy * 100.0);
    println!("   0.20: {:.2}%", temp_high.accuracy * 100.0);
    
    // Save results
    let results_json = serde_json::json!({
        "results": results.iter().map(|(name, metrics)| serde_json::json!({
            "name": name,
            "accuracy": metrics.accuracy,
            "auroc": metrics.auroc,
            "precision": metrics.precision,
            "recall": metrics.recall,
        })).collect::<Vec<_>>(),
        "baseline": {
            "name": results[0].0,
            "accuracy": results[0].1.accuracy,
        },
        "best": {
            "name": best_acc.0,
            "accuracy": best_acc.1.accuracy,
        },
    });
    
    std::fs::write("results/ablation_study.json", serde_json::to_string_pretty(&results_json)?)?;
    println!("\n✅ Results saved to results/ablation_study.json");
    
    Ok(())
}
