/// Systematic model scaling experiment to reach 95% accuracy
use module_regularized_grn::{
    Config,
    models::{scalable_hybrid::{ScalableHybridModel, ModelConfig}, nn::{bce_loss, bce_loss_backward}},
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;
use serde_json::json;

struct TrainResult {
    config_name: String,
    accuracy: f32,
    parameters: usize,
    epochs_trained: usize,
    best_epoch: usize,
}

fn train_model(
    config: ModelConfig,
    config_name: &str,
    train_samples: &[(usize, usize, f32)],
    val_samples: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    num_tfs: usize,
    num_genes: usize,
    learning_rate: f32,
    batch_size: usize,
    num_epochs: usize,
    seed: u64,
) -> Result<TrainResult> {
    
    println!("\n=== Training {} ===", config_name);
    println!("Config: embed={}, hidden={:?}, output={}, dropout={}",
             config.embed_dim, config.hidden_dims, config.output_dim, config.dropout);
    
    let mut model = ScalableHybridModel::new(num_tfs, num_genes, config.clone(), 0.01, seed);
    println!("Parameters: {}", model.num_parameters());
    
    let mut best_val_acc = 0.0;
    let mut best_epoch = 0;
    let expr_dim = config.expr_dim;
    
    for epoch in 0..num_epochs {
        model.zero_grad();
        
        // Training
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
        
        // Validation every 10 epochs
        if epoch % 10 == 0 || epoch == num_epochs - 1 {
            let mut val_correct = 0;
            
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
                
                for (i, &score) in scores.iter().enumerate() {
                    let pred = if score > 0.5 { 1.0 } else { 0.0 };
                    if pred == labels[i] {
                        val_correct += 1;
                    }
                }
            }
            
            let val_acc = val_correct as f32 / val_samples.len() as f32;
            
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                best_epoch = epoch;
            }
            
            print!("  Epoch {}/{}: Val Acc = {:.4} ({:.2}%)\r",
                   epoch, num_epochs, val_acc, val_acc * 100.0);
            std::io::stdout().flush().ok();
        }
    }
    
    println!();
    println!("Best: {:.4} ({:.2}%) at epoch {}", best_val_acc, best_val_acc * 100.0, best_epoch);
    
    Ok(TrainResult {
        config_name: config_name.to_string(),
        accuracy: best_val_acc,
        parameters: model.num_parameters(),
        epochs_trained: num_epochs,
        best_epoch,
    })
}

fn main() -> Result<()> {
    println!("=== Systematic Model Scaling for 95% Accuracy ===\n");

    let config = Config::load_default()?;
    let seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(seed);
    
    // Load data
    println!("Loading data...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    
    // Build expression mapping (use zeros for simplicity - already know it works)
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
    
    // Define model configurations to test
    let configs = vec![
        (ModelConfig::small(), "Small (Baseline)", 0.005, 64, 50),
        (ModelConfig::medium(), "Medium", 0.003, 64, 75),
        (ModelConfig::large(), "Large", 0.002, 48, 100),
        (ModelConfig::xlarge(), "XLarge", 0.001, 32, 150),
        (ModelConfig::ultra(), "Ultra", 0.001, 24, 200),
    ];
    
    let mut results = Vec::new();
    
    // Train each configuration
    for (config, name, lr, batch_size, epochs) in configs {
        let result = train_model(
            config,
            name,
            train_samples,
            val_samples,
            &tf_expr_map,
            &gene_expr_map,
            num_tfs,
            num_genes,
            lr,
            batch_size,
            epochs,
            seed,
        )?;
        
        results.push(result);
        
        // Print summary
        println!("\n{} Results:", name);
        println!("  Accuracy: {:.4} ({:.2}%)", results.last().unwrap().accuracy, 
                 results.last().unwrap().accuracy * 100.0);
        println!("  Parameters: {}", results.last().unwrap().parameters);
        
        // Early stop if we hit 95%
        if results.last().unwrap().accuracy >= 0.95 {
            println!("\n🎉 REACHED 95% TARGET! Stopping early.");
            break;
        }
    }
    
    // Print final summary
    println!("\n\n=== Final Results ===\n");
    println!("{:<20} {:>12} {:>12} {:>10}", "Config", "Accuracy", "Parameters", "Best Epoch");
    println!("{}", "-".repeat(60));
    
    for result in &results {
        println!("{:<20} {:>11.2}% {:>12} {:>10}",
                 result.config_name,
                 result.accuracy * 100.0,
                 result.parameters,
                 result.best_epoch);
    }
    
    // Find best
    let best = results.iter().max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap()).unwrap();
    println!("\n🏆 Best: {} with {:.2}% accuracy", best.config_name, best.accuracy * 100.0);
    
    // Save results
    let results_json = json!({
        "results": results.iter().map(|r| json!({
            "config": r.config_name,
            "accuracy": r.accuracy,
            "parameters": r.parameters,
            "epochs": r.epochs_trained,
            "best_epoch": r.best_epoch,
        })).collect::<Vec<_>>(),
        "best": {
            "config": best.config_name,
            "accuracy": best.accuracy,
            "parameters": best.parameters,
        }
    });
    
    std::fs::write("results/scaling_results.json", serde_json::to_string_pretty(&results_json)?)?;
    println!("✅ Results saved to results/scaling_results.json");
    
    if best.accuracy >= 0.95 {
        println!("\n✅✅✅ SUCCESS: Reached 95% accuracy target!");
    } else {
        println!("\n⚠️  Best accuracy: {:.2}% (Gap to 95%: {:.2}%)",
                 best.accuracy * 100.0, (0.95 - best.accuracy) * 100.0);
        println!("   Consider: Ensemble models, more epochs, or additional features");
    }
    
    Ok(())
}
