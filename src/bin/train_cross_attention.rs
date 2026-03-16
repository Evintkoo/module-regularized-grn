/// Gated Cross-Attention Network (GCAN) Training
///
/// Key improvements over train_standard_mlp.rs:
/// 1. Cross-modulation: TF and gene encodings inform each other before scoring
/// 2. Rich interaction features: [tf; gene; tf*gene; tf-gene] replaces cosine similarity
/// 3. Proper mini-batch Adam optimizer (previous code updated once per epoch!)
/// 4. Label smoothing for noisy biological labels
/// 5. Cosine LR decay with warmup
/// 6. Gradient clipping
/// 7. AUROC computation

use module_regularized_grn::{
    Config,
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use module_regularized_grn::models::cross_attention_model::{
    CrossAttentionModel, ModelAdamStates,
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== Gated Cross-Attention Network (GCAN) Training ===\n");

    let config = Config::load_default()?;
    let base_seed = config.project.seed;

    // === Load data ===
    println!("Loading data...");
    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();
    println!("  TFs: {} | Genes: {}", num_tfs, num_genes);

    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    let expr_dim = expr_data.n_cell_types;
    println!("  Cell types: {} | Expression genes: {}", expr_dim, expr_data.n_genes);

    // Load ID→Symbol mappings: Ensembl + UniProt for maximum coverage
    let expr_dir = "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd";
    let ensembl_to_symbol: HashMap<String, String> = {
        let file = std::fs::File::open(format!("{}/ensembl_to_symbol.json", expr_dir))?;
        serde_json::from_reader(std::io::BufReader::new(file))?
    };
    let uniprot_to_symbol: HashMap<String, String> = {
        let path = format!("{}/uniprot_to_symbol.json", expr_dir);
        match std::fs::File::open(&path) {
            Ok(file) => serde_json::from_reader(std::io::BufReader::new(file)).unwrap_or_default(),
            Err(_) => HashMap::new(),
        }
    };
    println!("  Ensembl→Symbol mappings: {} | UniProt→Symbol: {}",
             ensembl_to_symbol.len(), uniprot_to_symbol.len());

    // Build expression maps using gene symbols (Ensembl→Symbol lookup)
    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, ensembl_id) in expr_data.gene_names.iter().enumerate() {
        if let Some(symbol) = ensembl_to_symbol.get(ensembl_id) {
            gene_to_expr_idx.insert(symbol.clone(), i);
        }
    }
    // Also build reverse lookup: symbol→expr_idx for UniProt resolution
    let symbol_to_expr_idx = gene_to_expr_idx.clone();

    let tf_vocab = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();

    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut tf_found = 0;
    let mut gene_found = 0;

    for (name, &idx) in tf_vocab.iter() {
        // Try direct symbol match first, then UniProt→Symbol lookup
        let expr_idx = gene_to_expr_idx.get(name).copied()
            .or_else(|| {
                uniprot_to_symbol.get(name)
                    .and_then(|sym| symbol_to_expr_idx.get(sym).copied())
            });
        if let Some(eidx) = expr_idx {
            tf_expr_map.insert(idx, expr_data.expression.column(eidx).to_owned());
            tf_found += 1;
        } else {
            tf_expr_map.insert(idx, Array1::zeros(expr_dim));
        }
    }
    for (name, &idx) in gene_vocab.iter() {
        let expr_idx = gene_to_expr_idx.get(name).copied()
            .or_else(|| {
                uniprot_to_symbol.get(name)
                    .and_then(|sym| symbol_to_expr_idx.get(sym).copied())
            });
        if let Some(eidx) = expr_idx {
            gene_expr_map.insert(idx, expr_data.expression.column(eidx).to_owned());
            gene_found += 1;
        } else {
            gene_expr_map.insert(idx, Array1::zeros(expr_dim));
        }
    }
    println!("  TFs with expression: {}/{} ({:.1}%)",
             tf_found, num_tfs, 100.0 * tf_found as f32 / num_tfs as f32);
    println!("  Genes with expression: {}/{} ({:.1}%)\n",
             gene_found, num_genes, 100.0 * gene_found as f32 / num_genes as f32);

    // === Build dataset ===
    let neg_ratio = 5usize;
    let mut rng = StdRng::seed_from_u64(base_seed);
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len() * neg_ratio, base_seed);

    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, gene)| (tf, gene, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, gene)| (tf, gene, 0.0)));
    examples.shuffle(&mut rng);

    let n_total = examples.len();
    let n_train = (n_total as f32 * 0.7) as usize;
    let n_val = (n_total as f32 * 0.15) as usize;

    let train_data = examples[..n_train].to_vec();
    let val_data = examples[n_train..n_train + n_val].to_vec();
    let test_data = examples[n_train + n_val..].to_vec();

    println!("Dataset: {} total | {} train | {} val | {} test\n",
             n_total, train_data.len(), val_data.len(), test_data.len());

    // === Hyperparameters ===
    let embed_dim = 128;
    let hidden_dim = 384;
    let enc_dim = 192;
    let dropout_rate = 0.0;
    let lr_max = 0.003;
    let lr_min = 1e-5;
    let warmup_epochs = 3;
    let epochs = 80;
    let batch_size = 256;
    let weight_decay = 0.005;
    let label_smooth = 0.05;
    let grad_clip = 5.0;
    let focal_gamma = 0.0f32; // 0 = standard BCE
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;

    // === Multi-seed training ===
    let seeds = [base_seed, base_seed + 100, base_seed + 200, base_seed + 300, base_seed + 400];
    let mut all_results: Vec<SeedResult> = Vec::new();

    for (seed_idx, &seed) in seeds.iter().enumerate() {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  Seed {}/{}: {}", seed_idx + 1, seeds.len(), seed);
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        let result = train_single_seed(
            num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, enc_dim,
            dropout_rate, focal_gamma,
            lr_max, lr_min, warmup_epochs, epochs, batch_size,
            weight_decay, label_smooth, grad_clip, beta1, beta2, eps,
            seed,
            &train_data, &val_data, &test_data,
            &tf_expr_map, &gene_expr_map,
        );
        all_results.push(result);
    }

    // === Summary ===
    println!("\n{}", "=".repeat(60));
    println!("  MULTI-SEED RESULTS SUMMARY");
    println!("{}\n", "=".repeat(60));

    let accs: Vec<f32> = all_results.iter().map(|r| r.accuracy).collect();
    let f1s: Vec<f32> = all_results.iter().map(|r| r.f1).collect();
    let aurocs: Vec<f32> = all_results.iter().map(|r| r.auroc).collect();

    for (i, r) in all_results.iter().enumerate() {
        println!("Seed {}: Acc={:.2}% (opt={:.2}% @{:.2}) | F1={:.4} | AUROC={:.4}",
                 i + 1, r.accuracy * 100.0, r.accuracy_opt * 100.0, r.threshold, r.f1, r.auroc);
    }

    let mean_acc = accs.iter().sum::<f32>() / accs.len() as f32;
    let std_acc = (accs.iter().map(|x| (x - mean_acc).powi(2)).sum::<f32>() / accs.len() as f32).sqrt();
    let mean_f1 = f1s.iter().sum::<f32>() / f1s.len() as f32;
    let mean_auroc = aurocs.iter().sum::<f32>() / aurocs.len() as f32;
    let accs_opt: Vec<f32> = all_results.iter().map(|r| r.accuracy_opt).collect();
    let mean_acc_opt = accs_opt.iter().sum::<f32>() / accs_opt.len() as f32;

    println!("\nMean Accuracy: {:.2}% (±{:.2}%)", mean_acc * 100.0, std_acc * 100.0);
    println!("Mean Acc (opt): {:.2}%", mean_acc_opt * 100.0);
    println!("Mean F1:       {:.4}", mean_f1);
    println!("Mean AUROC:    {:.4}", mean_auroc);

    // === Ensemble evaluation (average predictions from all seeds) ===
    let n_test = test_data.len();
    let n_seeds = all_results.len() as f32;
    let mut ensemble_preds = vec![0.0f32; n_test];
    for r in &all_results {
        for (i, &p) in r.test_predictions.iter().enumerate() {
            ensemble_preds[i] += p / n_seeds;
        }
    }
    // Compute ensemble metrics
    let mut ens_tp = 0usize;
    let mut ens_tn = 0usize;
    let mut ens_fp = 0usize;
    let mut ens_fn = 0usize;
    let mut ens_scores: Vec<(f32, f32)> = Vec::new();
    for (i, (_, _, label)) in test_data.iter().enumerate() {
        let pred_class = if ensemble_preds[i] >= 0.5 { 1.0 } else { 0.0 };
        match (pred_class as i32, *label as i32) {
            (1, 1) => ens_tp += 1,
            (0, 0) => ens_tn += 1,
            (1, 0) => ens_fp += 1,
            (0, 1) => ens_fn += 1,
            _ => {}
        }
        ens_scores.push((ensemble_preds[i], *label));
    }
    let ens_total = (ens_tp + ens_tn + ens_fp + ens_fn) as f32;
    let ens_acc = (ens_tp + ens_tn) as f32 / ens_total;
    let ens_prec = if ens_tp + ens_fp > 0 { ens_tp as f32 / (ens_tp + ens_fp) as f32 } else { 0.0 };
    let ens_rec = if ens_tp + ens_fn > 0 { ens_tp as f32 / (ens_tp + ens_fn) as f32 } else { 0.0 };
    let ens_f1 = if ens_prec + ens_rec > 0.0 { 2.0 * ens_prec * ens_rec / (ens_prec + ens_rec) } else { 0.0 };
    let ens_auroc = compute_auroc(&ens_scores);

    println!("\n{}", "=".repeat(60));
    println!("  ENSEMBLE RESULTS ({} models)", all_results.len());
    println!("{}", "=".repeat(60));
    println!("  Accuracy:  {:.2}%", ens_acc * 100.0);
    println!("  Precision: {:.2}%", ens_prec * 100.0);
    println!("  Recall:    {:.2}%", ens_rec * 100.0);
    println!("  F1-Score:  {:.4}", ens_f1);
    println!("  AUROC:     {:.4}", ens_auroc);

    // Save results
    let results_json = serde_json::json!({
        "architecture": "Gated Cross-Attention Network (GCAN)",
        "model": {
            "embed_dim": embed_dim,
            "expr_dim": expr_dim,
            "hidden_dim": hidden_dim,
            "enc_dim": enc_dim,
            "total_params": all_results[0].num_params,
        },
        "training": {
            "optimizer": "Adam",
            "lr_max": lr_max,
            "lr_schedule": "cosine_with_warmup",
            "warmup_epochs": warmup_epochs,
            "epochs": epochs,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "label_smoothing": label_smooth,
            "gradient_clipping": grad_clip,
            "update_frequency": "per_mini_batch",
        },
        "results": {
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc,
            "mean_f1": mean_f1,
            "mean_auroc": mean_auroc,
            "per_seed": all_results.iter().enumerate().map(|(i, r)| {
                serde_json::json!({
                    "seed": seeds[i],
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "auroc": r.auroc,
                })
            }).collect::<Vec<_>>(),
        },
        "ensemble": {
            "accuracy": ens_acc,
            "precision": ens_prec,
            "recall": ens_rec,
            "f1": ens_f1,
            "auroc": ens_auroc,
        }
    });

    std::fs::create_dir_all("results")?;
    std::fs::write(
        "results/gcan_results.json",
        serde_json::to_string_pretty(&results_json)?
    )?;
    println!("\nResults saved to results/gcan_results.json");

    Ok(())
}

struct SeedResult {
    accuracy: f32,
    accuracy_opt: f32,
    threshold: f32,
    precision: f32,
    recall: f32,
    f1: f32,
    auroc: f32,
    num_params: usize,
    test_predictions: Vec<f32>, // per-sample predictions for ensemble
}

fn train_single_seed(
    num_tfs: usize, num_genes: usize,
    embed_dim: usize, expr_dim: usize, hidden_dim: usize, enc_dim: usize,
    dropout_rate: f32, focal_gamma: f32,
    lr_max: f32, lr_min: f32, warmup_epochs: usize, epochs: usize, batch_size: usize,
    weight_decay: f32, label_smooth: f32, grad_clip: f32,
    beta1: f32, beta2: f32, eps: f32,
    seed: u64,
    train_data: &[(usize, usize, f32)],
    val_data: &[(usize, usize, f32)],
    test_data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
) -> SeedResult {
    let mut rng = StdRng::seed_from_u64(seed);

    // Shuffle training data
    let mut train_shuffled = train_data.to_vec();
    train_shuffled.shuffle(&mut rng);

    let mut model = CrossAttentionModel::new(
        num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, enc_dim, dropout_rate, seed,
    );
    let num_params = model.num_parameters();
    println!("Model parameters: {}", num_params);

    let mut adam_states = ModelAdamStates::new(&model);
    let mut best_val_acc = 0.0f32;
    let mut patience_counter = 0;
    let patience = 20;
    let batches_per_epoch = (train_shuffled.len() + batch_size - 1) / batch_size;

    for epoch in 0..epochs {
        // === Learning rate schedule: warmup + cosine decay ===
        let lr = if epoch < warmup_epochs {
            lr_min + (lr_max - lr_min) * (epoch as f32 / warmup_epochs as f32)
        } else {
            let progress = (epoch - warmup_epochs) as f32 / (epochs - warmup_epochs) as f32;
            lr_min + 0.5 * (lr_max - lr_min) * (1.0 + (std::f32::consts::PI * progress).cos())
        };

        // Reshuffle each epoch
        train_shuffled.shuffle(&mut rng);

        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;
        let mut epoch_total = 0;

        for batch_start in (0..train_shuffled.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_shuffled.len());
            let batch = &train_shuffled[batch_start..batch_end];
            let bs = batch.len() as f32;

            let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let mut labels: Vec<f32> = batch.iter().map(|x| x.2).collect();

            // Label smoothing
            for l in labels.iter_mut() {
                *l = *l * (1.0 - label_smooth) + 0.5 * label_smooth;
            }

            let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
            let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);

            // Forward
            model.zero_grad();
            let predictions = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);

            // BCE loss with smoothed labels
            let labels_arr = Array1::from(labels.clone());
            let loss = focal_bce_loss(&predictions, &labels_arr, focal_gamma);
            let grad = focal_bce_grad(&predictions, &labels_arr, focal_gamma);

            // Backward
            model.backward(&grad);

            // Gradient clipping (by norm on embedding grads)
            clip_grad_norm(model.tf_embed_grad.as_slice_mut().unwrap(), grad_clip);
            clip_grad_norm(model.gene_embed_grad.as_slice_mut().unwrap(), grad_clip);

            // Adam update (per mini-batch!)
            // Pass 1.0 as batch_size since bce_grad_smooth already returns mean gradient
            model.update_adam(&mut adam_states, lr, beta1, beta2, eps, weight_decay, 1.0);

            epoch_loss += loss * bs;
            for (pred, orig) in predictions.iter().zip(batch.iter()) {
                let pred_class = if *pred >= 0.5 { 1.0 } else { 0.0 };
                if pred_class == orig.2 { epoch_correct += 1; }
                epoch_total += 1;
            }

        }

        let train_loss = epoch_loss / epoch_total as f32;
        let train_acc = epoch_correct as f32 / epoch_total as f32;

        // === Validation ===
        model.set_training(false);
        let (val_loss, val_acc, _, _, _, _, _) = evaluate(
            &mut model, val_data, batch_size, tf_expr_map, gene_expr_map, expr_dim
        );
        model.set_training(true);

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            println!("Epoch {:3} | LR: {:.6} | Train: loss={:.4} acc={:.2}% | Val: loss={:.4} acc={:.2}%",
                     epoch + 1, lr, train_loss, train_acc * 100.0, val_loss, val_acc * 100.0);
        }

        // Early stopping on val accuracy
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            patience_counter = 0;
        } else {
            patience_counter += 1;
            if patience_counter >= patience {
                println!("Early stopping at epoch {} (best val acc: {:.2}%)", epoch + 1, best_val_acc * 100.0);
                break;
            }
        }
    }

    // === Find optimal threshold on validation set ===
    model.set_training(false);

    // Collect validation predictions
    let mut val_predictions: Vec<f32> = Vec::new();
    for batch_start in (0..val_data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(val_data.len());
        let batch = &val_data[batch_start..batch_end];
        let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
        let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);
        let preds = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
        val_predictions.extend(preds.iter());
    }

    // Search for optimal threshold
    let mut best_threshold = 0.5f32;
    let mut best_threshold_acc = 0.0f32;
    for t_int in 20..80 {
        let t = t_int as f32 / 100.0;
        let correct = val_predictions.iter().zip(val_data.iter())
            .filter(|(&p, (_, _, label))| {
                let pred = if p >= t { 1.0 } else { 0.0 };
                pred == *label
            }).count();
        let acc = correct as f32 / val_data.len() as f32;
        if acc > best_threshold_acc {
            best_threshold_acc = acc;
            best_threshold = t;
        }
    }
    println!("Optimal threshold: {:.2} (val acc: {:.2}%)", best_threshold, best_threshold_acc * 100.0);

    // === Test evaluation with optimal threshold ===
    let (_, test_acc, test_prec, test_rec, test_f1, test_auroc, cm) = evaluate(
        &mut model, test_data, batch_size, tf_expr_map, gene_expr_map, expr_dim
    );

    // Collect per-sample test predictions for ensemble
    let mut test_predictions: Vec<f32> = Vec::with_capacity(test_data.len());
    for batch_start in (0..test_data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(test_data.len());
        let batch = &test_data[batch_start..batch_end];
        let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
        let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);
        let preds = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
        test_predictions.extend(preds.iter());
    }

    // Compute test accuracy with optimized threshold
    let test_acc_opt = test_predictions.iter().zip(test_data.iter())
        .filter(|(&p, (_, _, label))| {
            let pred = if p >= best_threshold { 1.0 } else { 0.0 };
            pred == *label
        }).count() as f32 / test_data.len() as f32;

    println!("\nTest Results (threshold=0.5):");
    println!("  Accuracy:  {:.2}%", test_acc * 100.0);
    println!("  Precision: {:.2}%", test_prec * 100.0);
    println!("  Recall:    {:.2}%", test_rec * 100.0);
    println!("  F1-Score:  {:.4}", test_f1);
    println!("  AUROC:     {:.4}", test_auroc);
    println!("  TP: {} | FP: {} | FN: {} | TN: {}", cm.0, cm.1, cm.2, cm.3);
    println!("Test Results (threshold={:.2}): Accuracy={:.2}%\n", best_threshold, test_acc_opt * 100.0);

    SeedResult {
        accuracy: test_acc,
        accuracy_opt: test_acc_opt,
        threshold: best_threshold,
        precision: test_prec,
        recall: test_rec,
        f1: test_f1,
        auroc: test_auroc,
        num_params,
        test_predictions,
    }
}

fn evaluate(
    model: &mut CrossAttentionModel,
    data: &[(usize, usize, f32)],
    batch_size: usize,
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
) -> (f32, f32, f32, f32, f32, f32, (usize, usize, usize, usize)) {
    let mut total_loss = 0.0;
    let mut tp = 0usize;
    let mut tn = 0usize;
    let mut fp = 0usize;
    let mut fn_count = 0usize;
    let mut all_scores: Vec<(f32, f32)> = Vec::new(); // (score, label)

    for batch_start in (0..data.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(data.len());
        let batch = &data[batch_start..batch_end];

        let tf_indices: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_indices: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();

        let tf_expr_batch = build_expr_batch(&tf_indices, tf_expr_map, expr_dim);
        let gene_expr_batch = build_expr_batch(&gene_indices, gene_expr_map, expr_dim);

        let predictions = model.forward(&tf_indices, &gene_indices, &tf_expr_batch, &gene_expr_batch);
        let labels_arr = Array1::from(labels.clone());
        total_loss += focal_bce_loss(&predictions, &labels_arr, 0.0) * batch.len() as f32;

        for (pred, label) in predictions.iter().zip(labels.iter()) {
            let pred_class = if *pred >= 0.5 { 1.0 } else { 0.0 };
            match (pred_class as i32, *label as i32) {
                (1, 1) => tp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fp += 1,
                (0, 1) => fn_count += 1,
                _ => {}
            }
            all_scores.push((*pred, *label));
        }
    }

    let total = (tp + tn + fp + fn_count) as f32;
    let accuracy = (tp + tn) as f32 / total;
    let precision = if tp + fp > 0 { tp as f32 / (tp + fp) as f32 } else { 0.0 };
    let recall = if tp + fn_count > 0 { tp as f32 / (tp + fn_count) as f32 } else { 0.0 };
    let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
    let loss = total_loss / total;

    let auroc = compute_auroc(&all_scores);

    (loss, accuracy, precision, recall, f1, auroc, (tp, fp, fn_count, tn))
}

fn compute_auroc(scores_labels: &[(f32, f32)]) -> f32 {
    if scores_labels.is_empty() { return 0.0; }

    let mut sorted = scores_labels.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = sorted.iter().filter(|x| x.1 > 0.5).count() as f32;
    let total_neg = sorted.iter().filter(|x| x.1 <= 0.5).count() as f32;

    if total_pos == 0.0 || total_neg == 0.0 { return 0.5; }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tpr = 0.0;
    let mut prev_fpr = 0.0;

    for &(_, label) in &sorted {
        if label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let tpr = tp / total_pos;
        let fpr = fp / total_neg;
        // Trapezoidal rule
        auc += 0.5 * (tpr + prev_tpr) * (fpr - prev_fpr);
        prev_tpr = tpr;
        prev_fpr = fpr;
    }

    auc
}

fn focal_bce_loss(predictions: &Array1<f32>, labels: &Array1<f32>, gamma: f32) -> f32 {
    let eps = 1e-7;
    let loss: f32 = predictions.iter().zip(labels.iter()).map(|(&p, &t)| {
        let p = p.clamp(eps, 1.0 - eps);
        // focal weight: (1-pt)^gamma where pt = p if t=1, (1-p) if t=0
        let pt = t * p + (1.0 - t) * (1.0 - p);
        let focal_weight = (1.0 - pt).powf(gamma);
        focal_weight * (-t * p.ln() - (1.0 - t) * (1.0 - p).ln())
    }).sum();
    loss / predictions.len() as f32
}

fn focal_bce_grad(predictions: &Array1<f32>, labels: &Array1<f32>, gamma: f32) -> Array1<f32> {
    let eps = 1e-7;
    let n = predictions.len() as f32;
    predictions.iter().zip(labels.iter()).map(|(&p, &t)| {
        let p = p.clamp(eps, 1.0 - eps);
        if gamma == 0.0 {
            // Standard BCE gradient
            (p - t) / n
        } else {
            // Focal loss gradient: d/d(logit) of -alpha_t * (1-pt)^gamma * log(pt)
            // Where pt = sigmoid(logit) if t=1, 1-sigmoid(logit) if t=0
            let pt = t * p + (1.0 - t) * (1.0 - p);
            let focal_weight = (1.0 - pt).powf(gamma);
            // gradient = focal_weight * (p - t) + gamma * (1-pt)^(gamma-1) * pt * log(pt) * (p - t) / pt
            // Simplified: (p-t) * [(1-pt)^gamma + gamma * (1-pt)^(gamma-1) * pt * log(pt)]
            // But for numerical stability, use a simpler form:
            let log_pt = pt.clamp(eps, 1.0).ln();
            let grad = focal_weight * (p - t) - gamma * (1.0 - pt).powf(gamma - 1.0) * pt * log_pt * (p - t);
            grad / n
        }
    }).collect()
}

fn clip_grad_norm(grad: &mut [f32], max_norm: f32) {
    let norm: f32 = grad.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for g in grad.iter_mut() {
            *g *= scale;
        }
    }
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
