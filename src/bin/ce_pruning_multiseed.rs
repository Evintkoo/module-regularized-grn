/// B1: Cross-Encoder Neuron Pruning — Multi-seed (seeds = [42, 123, 456])
/// For each seed: train baseline → profile fc1 → sweep sparsity 0–90%
/// Aggregates post-hoc and fine-tuned AUROC/retention mean ± std across seeds.
/// Results → results/cross_encoder_pruning_multiseed.json
use module_regularized_grn::{
    Config,
    models::cross_encoder::CrossEncoderModel,
    models::nn::{relu, bce_loss_backward},
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
    m_tf:    Array2<f32>, v_tf:    Array2<f32>,
    m_gene:  Array2<f32>, v_gene:  Array2<f32>,
    m_fc1_w: Array2<f32>, v_fc1_w: Array2<f32>,
    m_fc1_b: Array1<f32>, v_fc1_b: Array1<f32>,
    m_fc2_w: Array2<f32>, v_fc2_w: Array2<f32>,
    m_fc2_b: Array1<f32>, v_fc2_b: Array1<f32>,
    m_fc3_w: Array2<f32>, v_fc3_w: Array2<f32>,
    m_fc3_b: Array1<f32>, v_fc3_b: Array1<f32>,
    t: i32,
}

impl AdamState {
    fn new(model: &CrossEncoderModel) -> Self {
        Self {
            m_tf:    Array2::zeros(model.tf_embed.dim()),
            v_tf:    Array2::zeros(model.tf_embed.dim()),
            m_gene:  Array2::zeros(model.gene_embed.dim()),
            v_gene:  Array2::zeros(model.gene_embed.dim()),
            m_fc1_w: Array2::zeros(model.fc1.weights.dim()),
            v_fc1_w: Array2::zeros(model.fc1.weights.dim()),
            m_fc1_b: Array1::zeros(model.fc1.bias.len()),
            v_fc1_b: Array1::zeros(model.fc1.bias.len()),
            m_fc2_w: Array2::zeros(model.fc2.weights.dim()),
            v_fc2_w: Array2::zeros(model.fc2.weights.dim()),
            m_fc2_b: Array1::zeros(model.fc2.bias.len()),
            v_fc2_b: Array1::zeros(model.fc2.bias.len()),
            m_fc3_w: Array2::zeros(model.fc3.weights.dim()),
            v_fc3_w: Array2::zeros(model.fc3.weights.dim()),
            m_fc3_b: Array1::zeros(model.fc3.bias.len()),
            v_fc3_b: Array1::zeros(model.fc3.bias.len()),
            t: 0,
        }
    }
}

fn adam_step(model: &mut CrossEncoderModel, state: &mut AdamState, lr: f32, clip: f32) {
    state.t += 1;
    let b1 = 0.9f32; let b2 = 0.999f32; let eps = 1e-8f32;

    let norm_sq: f32 = [
        model.tf_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.gene_embed_grad.iter().map(|x| x*x).sum::<f32>(),
        model.fc1.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc1.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.fc2.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc2.grad_bias.iter().map(|x| x*x).sum::<f32>(),
        model.fc3.grad_weights.iter().map(|x| x*x).sum::<f32>(),
        model.fc3.grad_bias.iter().map(|x| x*x).sum::<f32>(),
    ].iter().sum();
    let scale = if norm_sq.sqrt() > clip { clip / norm_sq.sqrt() } else { 1.0 };

    fn adam2d(p: &mut Array2<f32>, g: &Array2<f32>, m: &mut Array2<f32>, v: &mut Array2<f32>,
              lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32) {
        let bc1 = 1.0 - b1.powi(t); let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(g.view()).for_each(|m, &g| *m = b1**m + (1.0-b1)*g*scale);
        ndarray::Zip::from(v.view_mut()).and(g.view()).for_each(|v, &g| *v = b2**v + (1.0-b2)*(g*scale).powi(2));
        ndarray::Zip::from(p.view_mut()).and(m.view()).and(v.view()).for_each(|p,&m,&v| *p -= lr*(m/bc1)/((v/bc2).sqrt()+eps));
    }
    fn adam1d(p: &mut Array1<f32>, g: &Array1<f32>, m: &mut Array1<f32>, v: &mut Array1<f32>,
              lr: f32, b1: f32, b2: f32, eps: f32, t: i32, scale: f32) {
        let bc1 = 1.0 - b1.powi(t); let bc2 = 1.0 - b2.powi(t);
        ndarray::Zip::from(m.view_mut()).and(g.view()).for_each(|m, &g| *m = b1**m + (1.0-b1)*g*scale);
        ndarray::Zip::from(v.view_mut()).and(g.view()).for_each(|v, &g| *v = b2**v + (1.0-b2)*(g*scale).powi(2));
        ndarray::Zip::from(p.view_mut()).and(m.view()).and(v.view()).for_each(|p,&m,&v| *p -= lr*(m/bc1)/((v/bc2).sqrt()+eps));
    }

    adam2d(&mut model.tf_embed,      &model.tf_embed_grad,    &mut state.m_tf,    &mut state.v_tf,    lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.gene_embed,    &model.gene_embed_grad,  &mut state.m_gene,  &mut state.v_gene,  lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc1.weights,   &model.fc1.grad_weights, &mut state.m_fc1_w, &mut state.v_fc1_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc1.bias,      &model.fc1.grad_bias,    &mut state.m_fc1_b, &mut state.v_fc1_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc2.weights,   &model.fc2.grad_weights, &mut state.m_fc2_w, &mut state.v_fc2_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc2.bias,      &model.fc2.grad_bias,    &mut state.m_fc2_b, &mut state.v_fc2_b, lr, b1, b2, eps, state.t, scale);
    adam2d(&mut model.fc3.weights,   &model.fc3.grad_weights, &mut state.m_fc3_w, &mut state.v_fc3_w, lr, b1, b2, eps, state.t, scale);
    adam1d(&mut model.fc3.bias,      &model.fc3.grad_bias,    &mut state.m_fc3_b, &mut state.v_fc3_b, lr, b1, b2, eps, state.t, scale);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn build_expr_batch(indices: &[usize], expr_map: &HashMap<usize, Array1<f32>>, expr_dim: usize) -> Array2<f32> {
    let mut batch = Array2::zeros((indices.len(), expr_dim));
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(expr) = expr_map.get(&idx) { batch.row_mut(i).assign(expr); }
    }
    batch
}

fn calculate_auroc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels.iter()).map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    let n_neg = labels.len() as f32 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    let mut tp = 0.0f32; let mut fp = 0.0f32; let mut auc = 0.0f32; let mut prev_fp = 0.0f32;
    for (_, label) in &pairs {
        if *label == 1.0 { tp += 1.0; } else {
            fp += 1.0; auc += (tp / n_pos) * ((fp - prev_fp) / n_neg); prev_fp = fp;
        }
    }
    auc
}

fn mean(v: &[f32]) -> f32 { v.iter().sum::<f32>() / v.len() as f32 }

fn std_dev(v: &[f32]) -> f32 {
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
}

fn evaluate(
    model: &mut CrossEncoderModel,
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
    batch_size: usize,
) -> (f32, f32) {
    let mut all_preds: Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        let probs: Vec<f32> = logits.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        all_preds.extend_from_slice(&probs);
        all_labels.extend_from_slice(&labels);
    }
    let n_correct = all_preds.iter().zip(all_labels.iter()).filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count();
    let accuracy = n_correct as f32 / all_preds.len() as f32;
    let auroc = calculate_auroc(&all_preds, &all_labels);
    (accuracy, auroc)
}

fn train_one_epoch(
    model: &mut CrossEncoderModel, state: &mut AdamState,
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>, gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize, batch_size: usize, lr: f32, clip: f32, epoch_seed: u64,
) {
    let mut epoch_data = data.to_vec();
    let mut rng = StdRng::seed_from_u64(epoch_seed);
    epoch_data.shuffle(&mut rng);
    for start in (0..epoch_data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(epoch_data.len());
        let batch = &epoch_data[start..end];
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels:   Vec<f32>   = batch.iter().map(|x| x.2).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let logits     = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        let labels_arr = Array1::from(labels);
        let grad       = bce_loss_backward(&logits, &labels_arr);
        model.backward(&grad);
        adam_step(model, state, lr, clip);
        model.zero_grad();
    }
}

struct NeuronStats {
    fc1_activation_counts: Vec<u32>,
    total_examples: usize,
    fc1_col_norms: Vec<f32>,
    fc2_row_norms: Vec<f32>,
}

fn profile_activations(
    model: &mut CrossEncoderModel,
    data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>,
    gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize,
    batch_size: usize,
) -> NeuronStats {
    let hidden_dim = model.fc1.bias.len();
    let l2 = |v: ndarray::ArrayView1<f32>| v.iter().map(|&x| x*x).sum::<f32>().sqrt();
    let fc1_col_norms: Vec<f32> = (0..hidden_dim).map(|j| l2(model.fc1.weights.column(j))).collect();
    let fc2_row_norms: Vec<f32> = (0..hidden_dim).map(|j| l2(model.fc2.weights.row(j))).collect();
    let mut counts = vec![0u32; hidden_dim];
    let mut total_examples = 0usize;
    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let bsz = batch.len();
        let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let tf_e   = build_expr_batch(&tf_idx,   tf_expr_map,   expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        let h1 = relu(model.h1_pre_cache.as_ref().unwrap());
        for i in 0..bsz { for j in 0..hidden_dim { if h1[[i, j]] > 0.0 { counts[j] += 1; } } }
        total_examples += bsz;
    }
    NeuronStats { fc1_activation_counts: counts, total_examples, fc1_col_norms, fc2_row_norms }
}

fn importance_scores(stats: &NeuronStats) -> Vec<f32> {
    let hidden_dim = stats.fc1_activation_counts.len();
    let total = stats.total_examples as f32;
    let normalize = |v: &[f32]| -> Vec<f32> {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        if range <= 0.0 { vec![0.5; v.len()] } else { v.iter().map(|&x| (x - min) / range).collect() }
    };
    let act_freq: Vec<f32> = stats.fc1_activation_counts.iter().map(|&c| c as f32 / total).collect();
    let fc1_n = normalize(&stats.fc1_col_norms);
    let fc2_n = normalize(&stats.fc2_row_norms);
    let weight_mag: Vec<f32> = (0..hidden_dim).map(|j| (fc1_n[j] + fc2_n[j]) / 2.0).collect();
    (0..hidden_dim).map(|j| 0.5 * act_freq[j] + 0.5 * weight_mag[j]).collect()
}

/// Prune Adam m/v buffers to match pruned fc1→fc2 dimensions (warm-start).
fn prune_adam_state(state: &mut AdamState, keep_indices: &[usize]) {
    let k = keep_indices.len();
    // fc1 outputs are pruned: m/v_fc1_w columns, m/v_fc1_b entries
    let in_dim = state.m_fc1_w.nrows();
    let mut new_m_fc1_w = Array2::zeros((in_dim, k));
    let mut new_v_fc1_w = Array2::zeros((in_dim, k));
    let mut new_m_fc1_b = Array1::zeros(k);
    let mut new_v_fc1_b = Array1::zeros(k);
    for (new_j, &old_j) in keep_indices.iter().enumerate() {
        new_m_fc1_w.column_mut(new_j).assign(&state.m_fc1_w.column(old_j));
        new_v_fc1_w.column_mut(new_j).assign(&state.v_fc1_w.column(old_j));
        new_m_fc1_b[new_j] = state.m_fc1_b[old_j];
        new_v_fc1_b[new_j] = state.v_fc1_b[old_j];
    }
    state.m_fc1_w = new_m_fc1_w;
    state.v_fc1_w = new_v_fc1_w;
    state.m_fc1_b = new_m_fc1_b;
    state.v_fc1_b = new_v_fc1_b;

    // fc2 inputs are pruned: m/v_fc2_w rows
    let out_dim = state.m_fc2_w.ncols();
    let mut new_m_fc2_w = Array2::zeros((k, out_dim));
    let mut new_v_fc2_w = Array2::zeros((k, out_dim));
    for (new_i, &old_i) in keep_indices.iter().enumerate() {
        new_m_fc2_w.row_mut(new_i).assign(&state.m_fc2_w.row(old_i));
        new_v_fc2_w.row_mut(new_i).assign(&state.v_fc2_w.row(old_i));
    }
    state.m_fc2_w = new_m_fc2_w;
    state.v_fc2_w = new_v_fc2_w;
    // fc2 bias is unchanged (depends on fc2 output dim, not input dim)
}

/// Prune model and return the keep indices for warm-starting Adam.
fn prune_to_sparsity(model: &mut CrossEncoderModel, scores: &[f32], sparsity: f32) -> Vec<usize> {
    let hidden_dim = scores.len();
    let n_keep = ((1.0 - sparsity) * hidden_dim as f32).round() as usize;
    let n_keep = n_keep.max(1);
    let mut order: Vec<usize> = (0..hidden_dim).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<usize> = order[..n_keep].to_vec();
    keep.sort_unstable();
    model.fc1.prune_outputs(&keep);
    model.fc2.prune_inputs(&keep);
    keep
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== CE Pruning Multi-seed (B1) ===");

    let config    = Config::load_default()?;
    let base_seed = config.project.seed;
    let mut rng   = StdRng::seed_from_u64(base_seed);

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
    let expr_dim = expr_data.n_cell_types;
    println!("  Cell types: {} | Genes: {}", expr_data.n_cell_types, expr_data.n_genes);

    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in expr_data.gene_names.iter().enumerate() { gene_to_expr_idx.insert(name.clone(), i); }
    let mut tf_expr_map:   HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    for (name, &idx) in builder.get_tf_vocab().iter() {
        tf_expr_map.insert(idx, gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned()).unwrap_or_else(|| Array1::zeros(expr_dim)));
    }
    for (name, &idx) in builder.get_gene_vocab().iter() {
        gene_expr_map.insert(idx, gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned()).unwrap_or_else(|| Array1::zeros(expr_dim)));
    }

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

    let embed_dim        = 512usize;
    let hidden_dim       = 512usize;
    let lr               = 0.005f32;
    let clip             = 1.0f32;
    let batch_size       = 256usize;
    let epochs           = 60usize;
    let patience         = 10usize;
    let fine_tune_epochs = 20usize;
    let fine_tune_lr     = 0.0002f32;
    let seeds            = [42u64, 123, 456];

    let sparsity_levels: &[f32] = &[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90];

    // ── Phase 1: Train baseline models for all seeds ──────────────────────────
    println!("\n=== Phase 1: Training {} baseline models ===", seeds.len());
    let mut baseline_models: Vec<CrossEncoderModel> = Vec::new();
    let mut baseline_states: Vec<AdamState> = Vec::new();
    let mut baseline_scores: Vec<Vec<f32>> = Vec::new();
    let mut baseline_aurocs: Vec<f32>      = Vec::new();

    for &seed in &seeds {
        println!("  Seed {}:", seed);
        let mut model = CrossEncoderModel::new(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, seed);
        let mut state = AdamState::new(&model);
        let mut best_val_acc = 0.0f32;
        let mut pat_counter  = 0usize;

        for epoch in 0..epochs {
            train_one_epoch(&mut model, &mut state, &train_data,
                &tf_expr_map, &gene_expr_map, expr_dim, batch_size, lr, clip,
                seed + epoch as u64);
            if (epoch + 1) % 10 == 0 || epoch == epochs - 1 {
                let (_, val_acc) = evaluate(&mut model, &val_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
                if val_acc > best_val_acc { best_val_acc = val_acc; pat_counter = 0; }
                else { pat_counter += 1; if pat_counter >= patience { break; } }
            }
        }

        let (_, base_auroc) = evaluate(&mut model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        println!("    baseline_auroc={:.4}", base_auroc);

        let stats  = profile_activations(&mut model, &train_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        let scores = importance_scores(&stats);
        baseline_models.push(model);
        baseline_states.push(state);
        baseline_scores.push(scores);
        baseline_aurocs.push(base_auroc);
    }

    let baseline_auroc_mean = mean(&baseline_aurocs);
    let baseline_auroc_std  = std_dev(&baseline_aurocs);
    println!("  Baseline AUROC: {:.4} ± {:.4}", baseline_auroc_mean, baseline_auroc_std);

    // ── Phase 2: Sparsity sweep ───────────────────────────────────────────────
    println!("\n=== Phase 2: Sparsity sweep ({} levels, {} seeds) ===", sparsity_levels.len(), seeds.len());
    let mut results: Vec<serde_json::Value> = Vec::new();

    for &sparsity in sparsity_levels {
        let n_keep = ((1.0 - sparsity) * hidden_dim as f32).round() as usize;
        let n_keep = n_keep.max(1);
        let compression_ratio = n_keep as f32 / hidden_dim as f32;
        println!("  Sparsity {:.0}% (keep={}):", sparsity * 100.0, n_keep);

        let mut posthoc_aurocs:   Vec<f32> = Vec::new();
        let mut finetuned_aurocs: Vec<f32> = Vec::new();

        for (seed_idx, &seed) in seeds.iter().enumerate() {
            let base_model = &baseline_models[seed_idx];
            let scores     = &baseline_scores[seed_idx];

            // Post-hoc: clone baseline, prune, evaluate
            let mut pruned = base_model.clone();
            let keep_indices = prune_to_sparsity(&mut pruned, scores, sparsity);
            let (_, ph_auroc) = evaluate(&mut pruned, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
            posthoc_aurocs.push(ph_auroc);

            // Fine-tune: warm-start Adam (prune baseline Adam buffers to match)
            let mut ft_model = pruned;
            let base_state = &baseline_states[seed_idx];
            let mut ft_state = AdamState {
                m_tf:    base_state.m_tf.clone(),    v_tf:    base_state.v_tf.clone(),
                m_gene:  base_state.m_gene.clone(),  v_gene:  base_state.v_gene.clone(),
                m_fc1_w: base_state.m_fc1_w.clone(), v_fc1_w: base_state.v_fc1_w.clone(),
                m_fc1_b: base_state.m_fc1_b.clone(), v_fc1_b: base_state.v_fc1_b.clone(),
                m_fc2_w: base_state.m_fc2_w.clone(), v_fc2_w: base_state.v_fc2_w.clone(),
                m_fc2_b: base_state.m_fc2_b.clone(), v_fc2_b: base_state.v_fc2_b.clone(),
                m_fc3_w: base_state.m_fc3_w.clone(), v_fc3_w: base_state.v_fc3_w.clone(),
                m_fc3_b: base_state.m_fc3_b.clone(), v_fc3_b: base_state.v_fc3_b.clone(),
                t: base_state.t,
            };
            prune_adam_state(&mut ft_state, &keep_indices);
            for ft_epoch in 0..fine_tune_epochs {
                train_one_epoch(&mut ft_model, &mut ft_state, &train_data,
                    &tf_expr_map, &gene_expr_map, expr_dim, batch_size, fine_tune_lr, clip,
                    seed + 10_000 + ft_epoch as u64);
            }
            let (_, ft_auroc) = evaluate(&mut ft_model, &test_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
            finetuned_aurocs.push(ft_auroc);
        }

        // Retention per seed
        let ph_ret: Vec<f32> = posthoc_aurocs.iter().zip(baseline_aurocs.iter())
            .map(|(&pa, &ba)| pa / ba.max(1e-8)).collect();
        let ft_ret: Vec<f32> = finetuned_aurocs.iter().zip(baseline_aurocs.iter())
            .map(|(&fa, &ba)| fa / ba.max(1e-8)).collect();

        println!("    post-hoc: {:.4} ± {:.4}  fine-tuned: {:.4} ± {:.4}",
            mean(&posthoc_aurocs), std_dev(&posthoc_aurocs),
            mean(&finetuned_aurocs), std_dev(&finetuned_aurocs));

        results.push(serde_json::json!({
            "sparsity":                sparsity,
            "neurons_per_tower":       n_keep,
            "compression_ratio":       compression_ratio,
            "posthoc_auroc_per_seed":  posthoc_aurocs,
            "posthoc_auroc_mean":      mean(&posthoc_aurocs),
            "posthoc_auroc_std":       std_dev(&posthoc_aurocs),
            "posthoc_retention_mean":  mean(&ph_ret),
            "posthoc_retention_std":   std_dev(&ph_ret),
            "finetuned_auroc_per_seed": finetuned_aurocs,
            "finetuned_auroc_mean":    mean(&finetuned_aurocs),
            "finetuned_auroc_std":     std_dev(&finetuned_aurocs),
            "finetuned_retention_mean": mean(&ft_ret),
            "finetuned_retention_std":  std_dev(&ft_ret),
        }));
    }

    // ── Write results ─────────────────────────────────────────────────────────
    std::fs::create_dir_all("results")?;
    let output = serde_json::json!({
        "model":                 "cross_encoder",
        "seeds":                 seeds,
        "fine_tune_epochs":      fine_tune_epochs,
        "fine_tune_lr":          fine_tune_lr,
        "baseline_auroc_per_seed": baseline_aurocs,
        "baseline_auroc_mean":   baseline_auroc_mean,
        "baseline_auroc_std":    baseline_auroc_std,
        "results":               results,
    });
    std::fs::write("results/cross_encoder_pruning_multiseed.json", serde_json::to_string_pretty(&output)?)?;
    println!("\n✓ Saved results/cross_encoder_pruning_multiseed.json");
    Ok(())
}
