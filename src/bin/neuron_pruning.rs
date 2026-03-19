/// Neuron Pruning Experiment (multi-seed)
/// Trains two-tower baseline over seeds [42, 123, 456], profiles neuron activations,
/// sweeps structured sparsity 0–90%, measuring AUROC before and after 10-epoch fine-tuning.
/// Writes results/neuron_pruning_results.json (seed=42 only, backward compat)
/// and results/neuron_pruning_multiseed.json (all seeds, mean±std per sparsity).
use module_regularized_grn::{
    Config,
    models::hybrid_embeddings::HybridEmbeddingModel,
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

// ── Adam state ────────────────────────────────────────────────────────────────

struct AdamState {
    m_tf:       Array2<f32>, v_tf:       Array2<f32>,
    m_gene:     Array2<f32>, v_gene:     Array2<f32>,
    m_tf_fc1_w: Array2<f32>, v_tf_fc1_w: Array2<f32>,
    m_tf_fc1_b: Array1<f32>, v_tf_fc1_b: Array1<f32>,
    m_tf_fc2_w: Array2<f32>, v_tf_fc2_w: Array2<f32>,
    m_tf_fc2_b: Array1<f32>, v_tf_fc2_b: Array1<f32>,
    m_g_fc1_w:  Array2<f32>, v_g_fc1_w:  Array2<f32>,
    m_g_fc1_b:  Array1<f32>, v_g_fc1_b:  Array1<f32>,
    m_g_fc2_w:  Array2<f32>, v_g_fc2_w:  Array2<f32>,
    m_g_fc2_b:  Array1<f32>, v_g_fc2_b:  Array1<f32>,
    t: i32,
}

impl AdamState {
    fn new(model: &HybridEmbeddingModel) -> Self {
        Self {
            m_tf:       Array2::zeros(model.tf_embed.dim()),
            v_tf:       Array2::zeros(model.tf_embed.dim()),
            m_gene:     Array2::zeros(model.gene_embed.dim()),
            v_gene:     Array2::zeros(model.gene_embed.dim()),
            m_tf_fc1_w: Array2::zeros(model.tf_fc1.weights.dim()),
            v_tf_fc1_w: Array2::zeros(model.tf_fc1.weights.dim()),
            m_tf_fc1_b: Array1::zeros(model.tf_fc1.bias.len()),
            v_tf_fc1_b: Array1::zeros(model.tf_fc1.bias.len()),
            m_tf_fc2_w: Array2::zeros(model.tf_fc2.weights.dim()),
            v_tf_fc2_w: Array2::zeros(model.tf_fc2.weights.dim()),
            m_tf_fc2_b: Array1::zeros(model.tf_fc2.bias.len()),
            v_tf_fc2_b: Array1::zeros(model.tf_fc2.bias.len()),
            m_g_fc1_w:  Array2::zeros(model.gene_fc1.weights.dim()),
            v_g_fc1_w:  Array2::zeros(model.gene_fc1.weights.dim()),
            m_g_fc1_b:  Array1::zeros(model.gene_fc1.bias.len()),
            v_g_fc1_b:  Array1::zeros(model.gene_fc1.bias.len()),
            m_g_fc2_w:  Array2::zeros(model.gene_fc2.weights.dim()),
            v_g_fc2_w:  Array2::zeros(model.gene_fc2.weights.dim()),
            m_g_fc2_b:  Array1::zeros(model.gene_fc2.bias.len()),
            v_g_fc2_b:  Array1::zeros(model.gene_fc2.bias.len()),
            t: 0,
        }
    }
}

fn adam_step(model: &mut HybridEmbeddingModel, state: &mut AdamState, lr: f32, clip: f32) {
    state.t += 1;
    let b1  = 0.9f32;
    let b2  = 0.999f32;
    let eps = 1e-8f32;

    let norm_sq: f32 = [
        model.tf_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.gene_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.tf_fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.tf_fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.tf_fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.tf_fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.gene_fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.gene_fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.gene_fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.gene_fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
    ].iter().sum();
    let scale = if norm_sq.sqrt() > clip { clip / norm_sq.sqrt() } else { 1.0 };

    fn adam2d(
        param: &mut Array2<f32>, grad: &Array2<f32>,
        m: &mut Array2<f32>, v: &mut Array2<f32>,
        lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32,
    ) {
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(grad.view())
            .for_each(|m, &g| *m = b1 * *m + (1.0 - b1) * g * scale);
        ndarray::Zip::from(v.view_mut()).and(grad.view())
            .for_each(|v, &g| *v = b2 * *v + (1.0 - b2) * (g * scale).powi(2));
        ndarray::Zip::from(param.view_mut()).and(m.view()).and(v.view())
            .for_each(|p, &m, &v| *p -= lr * (m / bc1) / ((v / bc2).sqrt() + eps));
    }

    fn adam1d(
        param: &mut Array1<f32>, grad: &Array1<f32>,
        m: &mut Array1<f32>, v: &mut Array1<f32>,
        lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32,
    ) {
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(grad.view())
            .for_each(|m, &g| *m = b1 * *m + (1.0 - b1) * g * scale);
        ndarray::Zip::from(v.view_mut()).and(grad.view())
            .for_each(|v, &g| *v = b2 * *v + (1.0 - b2) * (g * scale).powi(2));
        ndarray::Zip::from(param.view_mut()).and(m.view()).and(v.view())
            .for_each(|p, &m, &v| *p -= lr * (m / bc1) / ((v / bc2).sqrt() + eps));
    }

    adam2d(&mut model.tf_embed,         &model.tf_embed_grad,         &mut state.m_tf,       &mut state.v_tf,       lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,       &model.gene_embed_grad,       &mut state.m_gene,     &mut state.v_gene,     lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.tf_fc1.weights,   &model.tf_fc1.grad_weights,   &mut state.m_tf_fc1_w, &mut state.v_tf_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.tf_fc1.bias,      &model.tf_fc1.grad_bias,      &mut state.m_tf_fc1_b, &mut state.v_tf_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.tf_fc2.weights,   &model.tf_fc2.grad_weights,   &mut state.m_tf_fc2_w, &mut state.v_tf_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.tf_fc2.bias,      &model.tf_fc2.grad_bias,      &mut state.m_tf_fc2_b, &mut state.v_tf_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_fc1.weights, &model.gene_fc1.grad_weights, &mut state.m_g_fc1_w,  &mut state.v_g_fc1_w,  lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.gene_fc1.bias,    &model.gene_fc1.grad_bias,    &mut state.m_g_fc1_b,  &mut state.v_g_fc1_b,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_fc2.weights, &model.gene_fc2.grad_weights, &mut state.m_g_fc2_w,  &mut state.v_g_fc2_w,  lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.gene_fc2.bias,    &model.gene_fc2.grad_bias,    &mut state.m_g_fc2_b,  &mut state.v_g_fc2_b,  lr, b1, b2, eps, state.t, scale);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_expr_batch(
    indices:  &[usize],
    expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
) -> Array2<f32> {
    let mut batch = Array2::zeros((indices.len(), expr_dim));
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(expr) = expr_map.get(&idx) {
            batch.row_mut(i).assign(expr);
        }
    }
    batch
}

fn calculate_auroc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels.iter())
        .map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    let n_neg = labels.len() as f32 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    let mut tp = 0.0f32; let mut fp = 0.0f32;
    let mut auc = 0.0f32; let mut prev_fp = 0.0f32;
    for (_, label) in &pairs {
        if *label == 1.0 { tp += 1.0; } else {
            fp += 1.0;
            auc += (tp / n_pos) * ((fp - prev_fp) / n_neg);
            prev_fp = fp;
        }
    }
    auc
}

/// Returns (accuracy, auroc) on the given data split.
fn evaluate(
    model:         &mut HybridEmbeddingModel,
    data:          &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim:      usize,
    batch_size:    usize,
) -> (f32, f32) {
    let mut all_preds:  Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end   = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let preds  = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        all_preds.extend_from_slice(preds.as_slice().unwrap());
        all_labels.extend_from_slice(&labels);
    }
    let n_correct = all_preds.iter().zip(all_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count();
    let accuracy = n_correct as f32 / all_preds.len() as f32;
    let auroc    = calculate_auroc(&all_preds, &all_labels);
    (accuracy, auroc)
}

fn count_params(model: &HybridEmbeddingModel) -> usize {
    model.tf_embed.len()          + model.gene_embed.len()
    + model.tf_fc1.weights.len()  + model.tf_fc1.bias.len()
    + model.tf_fc2.weights.len()  + model.tf_fc2.bias.len()
    + model.gene_fc1.weights.len()+ model.gene_fc1.bias.len()
    + model.gene_fc2.weights.len()+ model.gene_fc2.bias.len()
}

fn train_one_epoch(
    model:         &mut HybridEmbeddingModel,
    state:         &mut AdamState,
    data:          &[(usize, usize, f32)],
    tf_expr_map:   &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim:      usize,
    batch_size:    usize,
    lr:            f32,
    clip:          f32,
    epoch_seed:    u64,
) {
    let mut epoch_data = data.to_vec();
    let mut rng = StdRng::seed_from_u64(epoch_seed);
    epoch_data.shuffle(&mut rng);

    for start in (0..epoch_data.len()).step_by(batch_size) {
        let end   = (start + batch_size).min(epoch_data.len());
        let batch = &epoch_data[start..end];
        let bsz   = batch.len();
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);

        let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        let grad: Array1<f32> = preds.iter().zip(labels.iter())
            .map(|(&p, &l)| (p - l) / bsz as f32).collect();
        model.backward(&grad);
        adam_step(model, state, lr, clip);
        model.zero_grad();
    }
}

// ── Per-seed experiment ───────────────────────────────────────────────────────

struct SeedResult {
    baseline_auroc: f32,
    baseline_accuracy: f32,
    baseline_params: usize,
    /// One entry per sparsity level: (posthoc_auroc, posthoc_acc, posthoc_retention, finetuned_auroc, finetuned_acc, finetuned_retention, params_remaining, compression_ratio)
    sparsity_results: Vec<SparsityEntry>,
}

struct SparsityEntry {
    sparsity: f32,
    n_removed: usize,
    params_remaining: usize,
    compression_ratio: f32,
    posthoc_auroc: f32,
    posthoc_accuracy: f32,
    posthoc_retention: f32,
    finetuned_auroc: f32,
    finetuned_accuracy: f32,
    finetuned_retention: f32,
}

fn run_seed(
    seed: u64,
    num_tfs: usize,
    num_genes: usize,
    expr_dim: usize,
    train_data: &[(usize, usize, f32)],
    val_data: &[(usize, usize, f32)],
    test_data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    sparsity_levels: &[f32],
) -> Result<SeedResult> {
    let embed_dim       = 512usize;
    let hidden_dim      = 512usize;
    let output_dim      = 512usize;
    let temperature     = 0.05f32;
    let lr              = 0.001f32;
    let clip            = 5.0f32;
    let batch_size      = 256usize;
    let epochs          = 60usize;
    let patience        = 10usize;
    let alpha           = 0.5f32;
    let fine_tune_epochs = 10usize;
    let fine_tune_lr    = 0.001f32;

    println!("\n=== Training baseline (seed={}) ===", seed);
    let mut model = HybridEmbeddingModel::new(
        num_tfs, num_genes, embed_dim, expr_dim,
        hidden_dim, output_dim, temperature, 0.01, seed,
    );
    let mut state        = AdamState::new(&model);
    let mut best_val_acc = 0.0f32;
    let mut pat_counter  = 0usize;

    for epoch in 0..epochs {
        train_one_epoch(
            &mut model, &mut state, train_data,
            tf_expr_map, gene_expr_map, expr_dim, batch_size, lr, clip,
            seed + epoch as u64,
        );
        if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
            let (_, val_acc) = evaluate(&mut model, val_data, tf_expr_map, gene_expr_map, expr_dim, batch_size);
            println!("  Epoch {:2} | Val Acc: {:.2}%", epoch + 1, val_acc * 100.0);
            if val_acc > best_val_acc {
                best_val_acc = val_acc;
                pat_counter  = 0;
            } else {
                pat_counter += 1;
                if pat_counter >= patience { break; }
            }
        }
    }

    let (baseline_accuracy, baseline_auroc) = evaluate(
        &mut model, test_data, tf_expr_map, gene_expr_map, expr_dim, batch_size
    );
    let baseline_params = count_params(&model);
    println!("  Baseline: acc={:.2}% auroc={:.4} params={}", baseline_accuracy*100.0, baseline_auroc, baseline_params);

    // Profile activations
    println!("  Profiling activations...");
    let stats  = model.profile_activations(train_data, tf_expr_map, gene_expr_map, expr_dim, batch_size);
    let scores = model.importance_scores(&stats, alpha);

    // Sparsity sweep
    let mut sparsity_results = Vec::new();
    for &sparsity in sparsity_levels {
        let n_keep    = ((1.0 - sparsity) * hidden_dim as f32).round() as usize;
        let n_keep    = n_keep.max(1);
        let n_removed = hidden_dim - n_keep;
        println!("  Sparsity {:.0}%: keep={} remove={}", sparsity*100.0, n_keep, n_removed);

        // Post-hoc
        let mut pruned = model.clone();
        pruned.prune_to_sparsity(&scores, sparsity);
        let params_remaining  = count_params(&pruned);
        let compression_ratio = params_remaining as f32 / baseline_params as f32;
        let (posthoc_accuracy, posthoc_auroc) = evaluate(
            &mut pruned, test_data, tf_expr_map, gene_expr_map, expr_dim, batch_size
        );
        let posthoc_retention = posthoc_auroc / baseline_auroc;
        println!("    Post-hoc:   acc={:.2}% auroc={:.4} retention={:.4}", posthoc_accuracy*100.0, posthoc_auroc, posthoc_retention);

        // Fine-tune
        let mut ft_model = pruned;
        let mut ft_state = AdamState::new(&ft_model);
        for ft_epoch in 0..fine_tune_epochs {
            train_one_epoch(
                &mut ft_model, &mut ft_state, train_data,
                tf_expr_map, gene_expr_map, expr_dim, batch_size, fine_tune_lr, clip,
                seed + 10_000 + ft_epoch as u64,
            );
        }
        let (finetuned_accuracy, finetuned_auroc) = evaluate(
            &mut ft_model, test_data, tf_expr_map, gene_expr_map, expr_dim, batch_size
        );
        let finetuned_retention = finetuned_auroc / baseline_auroc;
        println!("    Fine-tuned: acc={:.2}% auroc={:.4} retention={:.4}", finetuned_accuracy*100.0, finetuned_auroc, finetuned_retention);

        sparsity_results.push(SparsityEntry {
            sparsity,
            n_removed,
            params_remaining,
            compression_ratio,
            posthoc_auroc,
            posthoc_accuracy,
            posthoc_retention,
            finetuned_auroc,
            finetuned_accuracy,
            finetuned_retention,
        });
    }

    Ok(SeedResult {
        baseline_auroc,
        baseline_accuracy,
        baseline_params,
        sparsity_results,
    })
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== Neuron Pruning Experiment (multi-seed) ===");

    let config    = Config::load_default()?;
    let base_seed = config.project.seed;
    let mut rng   = StdRng::seed_from_u64(base_seed);

    // ── Data loading ──────────────────────────────────────────────────────────
    println!("Loading prior knowledge...");
    let priors  = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs   = builder.num_tfs();
    let num_genes = builder.num_genes();
    println!("  TFs: {} | Genes: {}", num_tfs, num_genes);

    println!("Loading expression data...");
    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    println!("  Cell types: {} | Expression genes: {}",
             expr_data.n_cell_types, expr_data.n_genes);

    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in expr_data.gene_names.iter().enumerate() {
        gene_to_expr_idx.insert(name.clone(), i);
    }

    let tf_vocab   = builder.get_tf_vocab();
    let gene_vocab = builder.get_gene_vocab();
    let expr_dim   = expr_data.n_cell_types;

    let mut tf_expr_map:   HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    for (name, &idx) in tf_vocab.iter() {
        if let Some(&ei) = gene_to_expr_idx.get(name) {
            tf_expr_map.insert(idx, expr_data.expression.column(ei).to_owned());
        } else {
            tf_expr_map.insert(idx, Array1::zeros(expr_dim));
        }
    }
    for (name, &idx) in gene_vocab.iter() {
        if let Some(&ei) = gene_to_expr_idx.get(name) {
            gene_expr_map.insert(idx, expr_data.expression.column(ei).to_owned());
        } else {
            gene_expr_map.insert(idx, Array1::zeros(expr_dim));
        }
    }

    println!("Creating dataset...");
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, g)| (tf, g, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, g)| (tf, g, 0.0)));
    examples.shuffle(&mut rng);

    let n_total = examples.len();
    let n_train = (n_total as f32 * 0.7) as usize;
    let n_val   = (n_total as f32 * 0.15) as usize;
    let train_data = examples[..n_train].to_vec();
    let val_data   = examples[n_train..n_train+n_val].to_vec();
    let test_data  = examples[n_train+n_val..].to_vec();
    println!("  Train: {} | Val: {} | Test: {}", train_data.len(), val_data.len(), test_data.len());

    let sparsity_levels: &[f32] = &[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];
    let seeds: &[u64] = &[42, 123, 456];

    println!("\n=== Running {} seeds x {} sparsity levels ===", seeds.len(), sparsity_levels.len());

    let mut all_results: Vec<SeedResult> = Vec::new();
    for &seed in seeds {
        let result = run_seed(
            seed,
            num_tfs, num_genes, expr_dim,
            &train_data, &val_data, &test_data,
            &tf_expr_map, &gene_expr_map,
            sparsity_levels,
        )?;
        all_results.push(result);
    }

    // ── Write backward-compat single-seed file (seed=42, index 0) ─────────────
    std::fs::create_dir_all("results")?;
    {
        let r42 = &all_results[0];
        let alpha = 0.5f32;
        let fine_tune_epochs = 10usize;
        let fine_tune_lr = 0.001f32;
        let hidden_dim = 512usize;

        let results_json: Vec<serde_json::Value> = r42.sparsity_results.iter().map(|e| {
            serde_json::json!({
                "sparsity":                e.sparsity,
                "neurons_removed_per_tower": e.n_removed,
                "neurons_removed_total":   e.n_removed * 2,
                "params_remaining":        e.params_remaining,
                "compression_ratio":       e.compression_ratio,
                "posthoc_auroc":           e.posthoc_auroc,
                "posthoc_accuracy":        e.posthoc_accuracy,
                "posthoc_retention":       e.posthoc_retention,
                "finetuned_auroc":         e.finetuned_auroc,
                "finetuned_accuracy":      e.finetuned_accuracy,
                "finetuned_retention":     e.finetuned_retention,
            })
        }).collect();

        let threshold_sparsity = |thresh: f32| -> f32 {
            results_json.iter()
                .filter(|rv| rv["posthoc_retention"].as_f64().unwrap_or(0.0) as f32 >= thresh)
                .map(|rv| rv["sparsity"].as_f64().unwrap_or(0.0) as f32)
                .fold(f32::NEG_INFINITY, f32::max)
                .max(0.0)
        };
        let retention_95 = threshold_sparsity(0.95);
        let retention_90 = threshold_sparsity(0.90);

        let output = serde_json::json!({
            "seed":              42u64,
            "alpha":             alpha,
            "fine_tune_epochs":  fine_tune_epochs,
            "fine_tune_lr":      fine_tune_lr,
            "baseline_auroc":    r42.baseline_auroc,
            "baseline_accuracy": r42.baseline_accuracy,
            "baseline_params":   r42.baseline_params,
            "results":           results_json,
            "retention_95_threshold_sparsity": retention_95,
            "retention_90_threshold_sparsity": retention_90,
        });
        let _ = hidden_dim; // suppress unused warning
        std::fs::write("results/neuron_pruning_results.json", serde_json::to_string_pretty(&output)?)?;
        println!("\nWrote results/neuron_pruning_results.json (seed=42)");
        println!("  95% retention threshold: {:.0}% sparsity", retention_95*100.0);
        println!("  90% retention threshold: {:.0}% sparsity", retention_90*100.0);
    }

    // ── Write multi-seed file ─────────────────────────────────────────────────
    {
        let n_sparsity = sparsity_levels.len();
        let n_seeds    = seeds.len();

        let baseline_aurocs: Vec<f32> = all_results.iter().map(|r| r.baseline_auroc).collect();
        let baseline_mean = baseline_aurocs.iter().sum::<f32>() / n_seeds as f32;
        let baseline_std = {
            let var = baseline_aurocs.iter().map(|&x| (x - baseline_mean).powi(2)).sum::<f32>() / n_seeds as f32;
            var.sqrt()
        };

        let mut ms_results: Vec<serde_json::Value> = Vec::new();
        for si in 0..n_sparsity {
            let sparsity = sparsity_levels[si];
            let hidden_dim = 512usize;
            let n_keep = ((1.0 - sparsity) * hidden_dim as f32).round() as usize;
            let n_keep = n_keep.max(1);
            let neurons_per_tower = n_keep;

            let posthoc_retentions: Vec<f32> = all_results.iter().map(|r| r.sparsity_results[si].posthoc_retention).collect();
            let finetuned_retentions: Vec<f32> = all_results.iter().map(|r| r.sparsity_results[si].finetuned_retention).collect();
            let posthoc_aurocs: Vec<f32> = all_results.iter().map(|r| r.sparsity_results[si].posthoc_auroc).collect();
            let finetuned_aurocs: Vec<f32> = all_results.iter().map(|r| r.sparsity_results[si].finetuned_auroc).collect();

            let mean_std = |v: &[f32]| -> (f32, f32) {
                let m = v.iter().sum::<f32>() / v.len() as f32;
                let s = (v.iter().map(|&x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt();
                (m, s)
            };
            let (ph_mean, ph_std) = mean_std(&posthoc_retentions);
            let (ft_mean, ft_std) = mean_std(&finetuned_retentions);

            ms_results.push(serde_json::json!({
                "sparsity":                    sparsity,
                "neurons_per_tower":           neurons_per_tower,
                "posthoc_retention_mean":      ph_mean,
                "posthoc_retention_std":       ph_std,
                "posthoc_auroc_values":        posthoc_aurocs,
                "finetuned_retention_mean":    ft_mean,
                "finetuned_retention_std":     ft_std,
                "finetuned_auroc_values":      finetuned_aurocs,
            }));
        }

        let ms_output = serde_json::json!({
            "seeds":                   seeds,
            "baseline_auroc_per_seed": baseline_aurocs,
            "baseline_auroc_mean":     baseline_mean,
            "baseline_auroc_std":      baseline_std,
            "results":                 ms_results,
        });
        std::fs::write("results/neuron_pruning_multiseed.json", serde_json::to_string_pretty(&ms_output)?)?;
        println!("Wrote results/neuron_pruning_multiseed.json");
        println!("  Baseline AUROC: {:.4} ± {:.4}", baseline_mean, baseline_std);
    }

    println!("\n=== Experiment complete ===");
    Ok(())
}
