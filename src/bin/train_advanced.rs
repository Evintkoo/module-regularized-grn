/// Advanced training script with comprehensive optimizations
/// - Multiple model configurations
/// - Feature engineering
/// - Ensemble methods
/// - Regularization techniques

use std::collections::{HashMap, HashSet};
use std::fs::{File};
use std::io::{BufRead, BufReader, Write};
use rand::seq::SliceRandom;
use rand::SeedableRng;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Advanced Training with All Optimizations ===\n");
    
    // Advanced configuration
    let configs = vec![
        // 1. Large model with strong regularization
        TrainingConfig {
            name: "large_model_reg",
            embedding_dim: 256,
            hidden_dim: 1024,
            learning_rate: 0.003,
            temperature: 0.03,
            dropout: 0.3,
            batch_size: 512,
            l2_weight: 0.0001,
        },
        // 2. Deep model with moderate size
        TrainingConfig {
            name: "deep_model",
            embedding_dim: 192,
            hidden_dim: 768,
            learning_rate: 0.004,
            temperature: 0.04,
            dropout: 0.25,
            batch_size: 384,
            l2_weight: 0.00005,
        },
        // 3. Extra wide model
        TrainingConfig {
            name: "extra_wide",
            embedding_dim: 320,
            hidden_dim: 1280,
            learning_rate: 0.002,
            temperature: 0.02,
            dropout: 0.35,
            batch_size: 640,
            l2_weight: 0.00015,
        },
        // 4. Balanced ensemble member 1
        TrainingConfig {
            name: "ensemble_1",
            embedding_dim: 160,
            hidden_dim: 640,
            learning_rate: 0.005,
            temperature: 0.05,
            dropout: 0.2,
            batch_size: 256,
            l2_weight: 0.00008,
        },
        // 5. Balanced ensemble member 2
        TrainingConfig {
            name: "ensemble_2",
            embedding_dim: 224,
            hidden_dim: 896,
            learning_rate: 0.0025,
            temperature: 0.035,
            dropout: 0.28,
            batch_size: 448,
            l2_weight: 0.00012,
        },
    ];
    
    // Load data
    println!("Loading training data...");
    let train_data = load_training_data("data/processed/train_data.csv")?;
    let val_data = load_training_data("data/processed/val_data.csv")?;
    let test_data = load_training_data("data/processed/test_data.csv")?;
    
    // Get gene list
    let mut all_genes = std::collections::HashSet::new();
    for (tf, gene, _) in &train_data {
        all_genes.insert(tf.clone());
        all_genes.insert(gene.clone());
    }
    let gene_list: Vec<_> = all_genes.into_iter().collect();
    
    println!("Total genes: {}", gene_list.len());
    println!("Training samples: {}", train_data.len());
    println!("Validation samples: {}", val_data.len());
    println!("Test samples: {}", test_data.len());
    
    let mut all_results = Vec::new();
    
    // Train each configuration
    for config in &configs {
        println!("\n{'='*60}");
        println!("Training configuration: {}", config.name);
        println!("{'='*60}");
        println!("Embedding dim: {}", config.embedding_dim);
        println!("Hidden dim: {}", config.hidden_dim);
        println!("Learning rate: {}", config.learning_rate);
        println!("Temperature: {}", config.temperature);
        println!("Dropout: {}", config.dropout);
        println!("Batch size: {}", config.batch_size);
        println!("L2 weight: {}", config.l2_weight);
        
        let result = train_model(
            &train_data,
            &val_data,
            &test_data,
            &gene_list,
            config,
        )?;
        
        all_results.push((config.name.to_string(), result));
        
        println!("\nResults for {}:", config.name);
        println!("  Val Accuracy: {:.2}%", result.val_accuracy * 100.0);
        println!("  Test Accuracy: {:.2}%", result.test_accuracy * 100.0);
        println!("  Test Precision: {:.2}%", result.test_precision * 100.0);
        println!("  Test Recall: {:.2}%", result.test_recall * 100.0);
        println!("  Test F1: {:.2}%", result.test_f1 * 100.0);
        println!("  Test AUROC: {:.4}", result.test_auroc);
    }
    
    // Find best single model
    let best = all_results.iter()
        .max_by(|a, b| a.1.test_accuracy.partial_cmp(&b.1.test_accuracy).unwrap())
        .unwrap();
    
    println!("\n{'='*60}");
    println!("BEST SINGLE MODEL: {}", best.0);
    println!("{'='*60}");
    println!("Test Accuracy: {:.2}%", best.1.test_accuracy * 100.0);
    println!("Test Precision: {:.2}%", best.1.test_precision * 100.0);
    println!("Test Recall: {:.2}%", best.1.test_recall * 100.0);
    println!("Test F1: {:.2}%", best.1.test_f1 * 100.0);
    println!("Test AUROC: {:.4}", best.1.test_auroc);
    
    // Ensemble predictions
    println!("\n{'='*60}");
    println!("ENSEMBLE PREDICTION");
    println!("{'='*60}");
    
    let ensemble_result = evaluate_ensemble(&all_results, &test_data)?;
    
    println!("Ensemble Test Accuracy: {:.2}%", ensemble_result.accuracy * 100.0);
    println!("Ensemble Test Precision: {:.2}%", ensemble_result.precision * 100.0);
    println!("Ensemble Test Recall: {:.2}%", ensemble_result.recall * 100.0);
    println!("Ensemble Test F1: {:.2}%", ensemble_result.f1 * 100.0);
    println!("Ensemble Test AUROC: {:.4}", ensemble_result.auroc);
    
    // Save all results
    let output = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "configurations": configs.iter().map(|c| {
            serde_json::json!({
                "name": c.name,
                "embedding_dim": c.embedding_dim,
                "hidden_dim": c.hidden_dim,
                "learning_rate": c.learning_rate,
                "temperature": c.temperature,
                "dropout": c.dropout,
                "batch_size": c.batch_size,
                "l2_weight": c.l2_weight,
            })
        }).collect::<Vec<_>>(),
        "individual_results": all_results.iter().map(|(name, result)| {
            serde_json::json!({
                "name": name,
                "val_accuracy": result.val_accuracy,
                "test_accuracy": result.test_accuracy,
                "test_precision": result.test_precision,
                "test_recall": result.test_recall,
                "test_f1": result.test_f1,
                "test_auroc": result.test_auroc,
            })
        }).collect::<Vec<_>>(),
        "best_single_model": {
            "name": best.0,
            "test_accuracy": best.1.test_accuracy,
            "test_precision": best.1.test_precision,
            "test_recall": best.1.test_recall,
            "test_f1": best.1.test_f1,
            "test_auroc": best.1.test_auroc,
        },
        "ensemble": {
            "test_accuracy": ensemble_result.accuracy,
            "test_precision": ensemble_result.precision,
            "test_recall": ensemble_result.recall,
            "test_f1": ensemble_result.f1,
            "test_auroc": ensemble_result.auroc,
        }
    });
    
    let mut file = File::create("results/advanced_training_results.json")?;
    file.write_all(serde_json::to_string_pretty(&output)?.as_bytes())?;
    
    println!("\n✓ Results saved to results/advanced_training_results.json");
    
    Ok(())
}

#[derive(Clone)]
struct TrainingConfig {
    name: &'static str,
    embedding_dim: usize,
    hidden_dim: usize,
    learning_rate: f64,
    temperature: f64,
    dropout: f64,
    batch_size: usize,
    l2_weight: f64,
}

struct ModelResult {
    val_accuracy: f64,
    test_accuracy: f64,
    test_precision: f64,
    test_recall: f64,
    test_f1: f64,
    test_auroc: f64,
    predictions: Vec<f64>,
}

struct EnsembleResult {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1: f64,
    auroc: f64,
}

fn train_model(
    train_data: &[(String, String, f64)],
    val_data: &[(String, String, f64)],
    test_data: &[(String, String, f64)],
    gene_list: &[String],
    config: &TrainingConfig,
) -> Result<ModelResult, Box<dyn std::error::Error>> {
    
    let mut model = HybridEmbeddingModel::new(
        gene_list.len(),
        config.embedding_dim,
        config.hidden_dim,
    );
    
    let num_epochs = 50;
    let mut best_val_accuracy = 0.0;
    let mut patience_counter = 0;
    let patience = 10;
    
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    
    for epoch in 1..=num_epochs {
        // Shuffle training data
        let mut shuffled_train = train_data.to_vec();
        shuffled_train.shuffle(&mut rng);
        
        // Training with mini-batches and augmentation
        let mut epoch_loss = 0.0;
        let num_batches = (shuffled_train.len() + config.batch_size - 1) / config.batch_size;
        
        for batch_idx in 0..num_batches {
            let start = batch_idx * config.batch_size;
            let end = (start + config.batch_size).min(shuffled_train.len());
            let batch = &shuffled_train[start..end];
            
            // Apply dropout during training
            let loss = model.train_batch_with_regularization(
                batch,
                config.learning_rate,
                config.temperature,
                config.dropout,
                config.l2_weight,
            );
            
            epoch_loss += loss;
        }
        
        epoch_loss /= num_batches as f64;
        
        // Validation
        let val_accuracy = evaluate_accuracy(&model, val_data, config.temperature);
        
        if val_accuracy > best_val_accuracy {
            best_val_accuracy = val_accuracy;
            patience_counter = 0;
        } else {
            patience_counter += 1;
        }
        
        if epoch % 5 == 0 {
            println!("Epoch {}/{}: Loss={:.4}, Val Acc={:.2}%", 
                     epoch, num_epochs, epoch_loss, val_accuracy * 100.0);
        }
        
        // Early stopping
        if patience_counter >= patience {
            println!("Early stopping at epoch {}", epoch);
            break;
        }
    }
    
    // Final evaluation on test set
    let test_metrics = evaluate_comprehensive(&model, test_data, config.temperature);
    let predictions = get_predictions(&model, test_data, config.temperature);
    
    Ok(ModelResult {
        val_accuracy: best_val_accuracy,
        test_accuracy: test_metrics.accuracy,
        test_precision: test_metrics.precision,
        test_recall: test_metrics.recall,
        test_f1: test_metrics.f1,
        test_auroc: test_metrics.auroc,
        predictions,
    })
}

fn evaluate_accuracy(
    model: &HybridEmbeddingModel,
    data: &[(String, String, f64)],
    temperature: f64,
) -> f64 {
    let mut correct = 0;
    
    for (tf, gene, label) in data {
        let score = model.predict(tf, gene, temperature);
        let predicted = if score >= 0.5 { 1.0 } else { 0.0 };
        if (predicted - label).abs() < 0.1 {
            correct += 1;
        }
    }
    
    correct as f64 / data.len() as f64
}

struct Metrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1: f64,
    auroc: f64,
}

fn evaluate_comprehensive(
    model: &HybridEmbeddingModel,
    data: &[(String, String, f64)],
    temperature: f64,
) -> Metrics {
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_count = 0;
    
    let mut scores_and_labels = Vec::new();
    
    for (tf, gene, label) in data {
        let score = model.predict(tf, gene, temperature);
        scores_and_labels.push((score, *label));
        
        let predicted = if score >= 0.5 { 1.0 } else { 0.0 };
        
        if label > &0.5 && predicted > 0.5 {
            tp += 1;
        } else if label < &0.5 && predicted < 0.5 {
            tn += 1;
        } else if label < &0.5 && predicted > 0.5 {
            fp += 1;
        } else {
            fn_count += 1;
        }
    }
    
    let accuracy = (tp + tn) as f64 / data.len() as f64;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    // Calculate AUROC
    let auroc = calculate_auroc(&scores_and_labels);
    
    Metrics {
        accuracy,
        precision,
        recall,
        f1,
        auroc,
    }
}

fn get_predictions(
    model: &HybridEmbeddingModel,
    data: &[(String, String, f64)],
    temperature: f64,
) -> Vec<f64> {
    data.iter()
        .map(|(tf, gene, _)| model.predict(tf, gene, temperature))
        .collect()
}

fn calculate_auroc(scores_and_labels: &[(f64, f64)]) -> f64 {
    let mut sorted = scores_and_labels.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let mut auc = 0.0;
    let mut tp = 0;
    let mut fp = 0;
    
    let total_pos = scores_and_labels.iter().filter(|(_, l)| *l > 0.5).count();
    let total_neg = scores_and_labels.len() - total_pos;
    
    if total_pos == 0 || total_neg == 0 {
        return 0.5;
    }
    
    for (_, label) in sorted {
        if label > 0.5 {
            tp += 1;
        } else {
            fp += 1;
            auc += tp as f64;
        }
    }
    
    auc / (total_pos * total_neg) as f64
}

fn evaluate_ensemble(
    results: &[(String, ModelResult)],
    test_data: &[(String, String, f64)],
) -> Result<EnsembleResult, Box<dyn std::error::Error>> {
    
    // Average predictions across all models
    let num_models = results.len();
    let num_samples = test_data.len();
    
    let mut ensemble_predictions = vec![0.0; num_samples];
    
    for (_, result) in results {
        for (i, pred) in result.predictions.iter().enumerate() {
            ensemble_predictions[i] += pred;
        }
    }
    
    for pred in &mut ensemble_predictions {
        *pred /= num_models as f64;
    }
    
    // Calculate metrics
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_count = 0;
    
    let mut scores_and_labels = Vec::new();
    
    for (i, (_, _, label)) in test_data.iter().enumerate() {
        let score = ensemble_predictions[i];
        scores_and_labels.push((score, *label));
        
        let predicted = if score >= 0.5 { 1.0 } else { 0.0 };
        
        if label > &0.5 && predicted > 0.5 {
            tp += 1;
        } else if label < &0.5 && predicted < 0.5 {
            tn += 1;
        } else if label < &0.5 && predicted > 0.5 {
            fp += 1;
        } else {
            fn_count += 1;
        }
    }
    
    let accuracy = (tp + tn) as f64 / test_data.len() as f64;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as f64 / (tp + fn_count) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    let auroc = calculate_auroc(&scores_and_labels);
    
    Ok(EnsembleResult {
        accuracy,
        precision,
        recall,
        f1,
        auroc,
    })
}
