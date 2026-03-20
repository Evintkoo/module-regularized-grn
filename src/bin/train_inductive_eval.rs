/// Inductive (cold-start) evaluation: hold out 20% TFs and 20% genes from training.
/// Evaluate on pairs containing unseen TFs/genes. 3 seeds x 2 models.
/// Writes results/inductive_eval.json
use module_regularized_grn::{
    Config,
    models::hybrid_embeddings::HybridEmbeddingModel,
    models::cross_encoder::CrossEncoderModel,
    models::nn::bce_loss_backward,
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::{HashMap, HashSet};

// ── Adam helpers (same as train_standard_mlp / train_cross_encoder) ──────────

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

struct AdamTT {
    m_tf: Array2<f32>, v_tf: Array2<f32>,
    m_gene: Array2<f32>, v_gene: Array2<f32>,
    m_tf_fc1_w: Array2<f32>, v_tf_fc1_w: Array2<f32>,
    m_tf_fc1_b: Array1<f32>, v_tf_fc1_b: Array1<f32>,
    m_tf_fc2_w: Array2<f32>, v_tf_fc2_w: Array2<f32>,
    m_tf_fc2_b: Array1<f32>, v_tf_fc2_b: Array1<f32>,
    m_g_fc1_w: Array2<f32>, v_g_fc1_w: Array2<f32>,
    m_g_fc1_b: Array1<f32>, v_g_fc1_b: Array1<f32>,
    m_g_fc2_w: Array2<f32>, v_g_fc2_w: Array2<f32>,
    m_g_fc2_b: Array1<f32>, v_g_fc2_b: Array1<f32>,
    t: i32,
}
impl AdamTT {
    fn new(m: &HybridEmbeddingModel) -> Self {
        Self {
            m_tf: Array2::zeros(m.tf_embed.dim()), v_tf: Array2::zeros(m.tf_embed.dim()),
            m_gene: Array2::zeros(m.gene_embed.dim()), v_gene: Array2::zeros(m.gene_embed.dim()),
            m_tf_fc1_w: Array2::zeros(m.tf_fc1.weights.dim()), v_tf_fc1_w: Array2::zeros(m.tf_fc1.weights.dim()),
            m_tf_fc1_b: Array1::zeros(m.tf_fc1.bias.len()), v_tf_fc1_b: Array1::zeros(m.tf_fc1.bias.len()),
            m_tf_fc2_w: Array2::zeros(m.tf_fc2.weights.dim()), v_tf_fc2_w: Array2::zeros(m.tf_fc2.weights.dim()),
            m_tf_fc2_b: Array1::zeros(m.tf_fc2.bias.len()), v_tf_fc2_b: Array1::zeros(m.tf_fc2.bias.len()),
            m_g_fc1_w: Array2::zeros(m.gene_fc1.weights.dim()), v_g_fc1_w: Array2::zeros(m.gene_fc1.weights.dim()),
            m_g_fc1_b: Array1::zeros(m.gene_fc1.bias.len()), v_g_fc1_b: Array1::zeros(m.gene_fc1.bias.len()),
            m_g_fc2_w: Array2::zeros(m.gene_fc2.weights.dim()), v_g_fc2_w: Array2::zeros(m.gene_fc2.weights.dim()),
            m_g_fc2_b: Array1::zeros(m.gene_fc2.bias.len()), v_g_fc2_b: Array1::zeros(m.gene_fc2.bias.len()),
            t: 0,
        }
    }
}
fn adam_step_tt(model: &mut HybridEmbeddingModel, s: &mut AdamTT, lr: f32, clip: f32) {
    s.t += 1;
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let norms: Vec<f32> = vec![
        model.tf_embed_grad.iter().map(|x| x*x).sum(),
        model.gene_embed_grad.iter().map(|x| x*x).sum(),
        model.tf_fc1.grad_weights.iter().map(|x| x*x).sum(),
        model.tf_fc1.grad_bias.iter().map(|x| x*x).sum(),
        model.tf_fc2.grad_weights.iter().map(|x| x*x).sum(),
        model.tf_fc2.grad_bias.iter().map(|x| x*x).sum(),
        model.gene_fc1.grad_weights.iter().map(|x| x*x).sum(),
        model.gene_fc1.grad_bias.iter().map(|x| x*x).sum(),
        model.gene_fc2.grad_weights.iter().map(|x| x*x).sum(),
        model.gene_fc2.grad_bias.iter().map(|x| x*x).sum(),
    ];
    let norm: f32 = norms.iter().sum::<f32>().sqrt();
    let scale = if norm > clip { clip / norm } else { 1.0 };
    adam2d(&mut model.tf_embed, &model.tf_embed_grad, &mut s.m_tf, &mut s.v_tf, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_embed, &model.gene_embed_grad, &mut s.m_gene, &mut s.v_gene, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.tf_fc1.weights, &model.tf_fc1.grad_weights, &mut s.m_tf_fc1_w, &mut s.v_tf_fc1_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.tf_fc1.bias, &model.tf_fc1.grad_bias, &mut s.m_tf_fc1_b, &mut s.v_tf_fc1_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.tf_fc2.weights, &model.tf_fc2.grad_weights, &mut s.m_tf_fc2_w, &mut s.v_tf_fc2_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.tf_fc2.bias, &model.tf_fc2.grad_bias, &mut s.m_tf_fc2_b, &mut s.v_tf_fc2_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_fc1.weights, &model.gene_fc1.grad_weights, &mut s.m_g_fc1_w, &mut s.v_g_fc1_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.gene_fc1.bias, &model.gene_fc1.grad_bias, &mut s.m_g_fc1_b, &mut s.v_g_fc1_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_fc2.weights, &model.gene_fc2.grad_weights, &mut s.m_g_fc2_w, &mut s.v_g_fc2_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.gene_fc2.bias, &model.gene_fc2.grad_bias, &mut s.m_g_fc2_b, &mut s.v_g_fc2_b, lr, b1, b2, eps, s.t, scale);
}

struct AdamCE {
    m_tf: Array2<f32>, v_tf: Array2<f32>,
    m_gene: Array2<f32>, v_gene: Array2<f32>,
    m_fc1_w: Array2<f32>, v_fc1_w: Array2<f32>,
    m_fc1_b: Array1<f32>, v_fc1_b: Array1<f32>,
    m_fc2_w: Array2<f32>, v_fc2_w: Array2<f32>,
    m_fc2_b: Array1<f32>, v_fc2_b: Array1<f32>,
    m_fc3_w: Array2<f32>, v_fc3_w: Array2<f32>,
    m_fc3_b: Array1<f32>, v_fc3_b: Array1<f32>,
    t: i32,
}
impl AdamCE {
    fn new(m: &CrossEncoderModel) -> Self {
        Self {
            m_tf: Array2::zeros(m.tf_embed.dim()), v_tf: Array2::zeros(m.tf_embed.dim()),
            m_gene: Array2::zeros(m.gene_embed.dim()), v_gene: Array2::zeros(m.gene_embed.dim()),
            m_fc1_w: Array2::zeros(m.fc1.weights.dim()), v_fc1_w: Array2::zeros(m.fc1.weights.dim()),
            m_fc1_b: Array1::zeros(m.fc1.bias.len()), v_fc1_b: Array1::zeros(m.fc1.bias.len()),
            m_fc2_w: Array2::zeros(m.fc2.weights.dim()), v_fc2_w: Array2::zeros(m.fc2.weights.dim()),
            m_fc2_b: Array1::zeros(m.fc2.bias.len()), v_fc2_b: Array1::zeros(m.fc2.bias.len()),
            m_fc3_w: Array2::zeros(m.fc3.weights.dim()), v_fc3_w: Array2::zeros(m.fc3.weights.dim()),
            m_fc3_b: Array1::zeros(m.fc3.bias.len()), v_fc3_b: Array1::zeros(m.fc3.bias.len()),
            t: 0,
        }
    }
}
fn adam_step_ce(model: &mut CrossEncoderModel, s: &mut AdamCE, lr: f32, clip: f32) {
    s.t += 1;
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
    let norms: Vec<f32> = vec![
        model.tf_embed_grad.iter().map(|x| x*x).sum(),
        model.gene_embed_grad.iter().map(|x| x*x).sum(),
        model.fc1.grad_weights.iter().map(|x| x*x).sum(),
        model.fc1.grad_bias.iter().map(|x| x*x).sum(),
        model.fc2.grad_weights.iter().map(|x| x*x).sum(),
        model.fc2.grad_bias.iter().map(|x| x*x).sum(),
        model.fc3.grad_weights.iter().map(|x| x*x).sum(),
        model.fc3.grad_bias.iter().map(|x| x*x).sum(),
    ];
    let norm: f32 = norms.iter().sum::<f32>().sqrt();
    let scale = if norm > clip { clip / norm } else { 1.0 };
    adam2d(&mut model.tf_embed, &model.tf_embed_grad, &mut s.m_tf, &mut s.v_tf, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.gene_embed, &model.gene_embed_grad, &mut s.m_gene, &mut s.v_gene, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.fc1.weights, &model.fc1.grad_weights, &mut s.m_fc1_w, &mut s.v_fc1_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.fc1.bias, &model.fc1.grad_bias, &mut s.m_fc1_b, &mut s.v_fc1_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.fc2.weights, &model.fc2.grad_weights, &mut s.m_fc2_w, &mut s.v_fc2_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.fc2.bias, &model.fc2.grad_bias, &mut s.m_fc2_b, &mut s.v_fc2_b, lr, b1, b2, eps, s.t, scale);
    adam2d(&mut model.fc3.weights, &model.fc3.grad_weights, &mut s.m_fc3_w, &mut s.v_fc3_w, lr, b1, b2, eps, s.t, scale);
    adam1d(&mut model.fc3.bias, &model.fc3.grad_bias, &mut s.m_fc3_b, &mut s.v_fc3_b, lr, b1, b2, eps, s.t, scale);
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn build_expr_batch(indices: &[usize], expr_map: &HashMap<usize, Array1<f32>>, expr_dim: usize) -> Array2<f32> {
    let mut batch = Array2::zeros((indices.len(), expr_dim));
    for (i, &idx) in indices.iter().enumerate() {
        if let Some(expr) = expr_map.get(&idx) { batch.row_mut(i).assign(expr); }
    }
    batch
}

fn calculate_auroc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(labels).map(|(&p, &l)| (p, l)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let n_pos = labels.iter().filter(|&&l| l == 1.0).count() as f32;
    let n_neg = labels.len() as f32 - n_pos;
    if n_pos == 0.0 || n_neg == 0.0 { return 0.5; }
    let (mut tp, mut fp, mut auc, mut prev_fp) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &(_, label) in &pairs {
        if label == 1.0 { tp += 1.0; } else { fp += 1.0; auc += (tp/n_pos)*((fp-prev_fp)/n_neg); prev_fp = fp; }
    }
    auc
}

fn calculate_f1(predictions: &[f32], labels: &[f32]) -> f32 {
    let (mut tp, mut fp, mut fn_) = (0.0f32, 0.0f32, 0.0f32);
    for (&p, &l) in predictions.iter().zip(labels) {
        match ((p >= 0.5) as i32, l as i32) { (1,1) => tp += 1.0, (1,0) => fp += 1.0, (0,1) => fn_ += 1.0, _ => {} }
    }
    let prec = if tp+fp > 0.0 { tp/(tp+fp) } else { 0.0 };
    let rec = if tp+fn_ > 0.0 { tp/(tp+fn_) } else { 0.0 };
    if prec+rec > 0.0 { 2.0*prec*rec/(prec+rec) } else { 0.0 }
}

fn mean(v: &[f32]) -> f32 { v.iter().sum::<f32>() / v.len() as f32 }
fn std_dev(v: &[f32]) -> f32 { let m = mean(v); (v.iter().map(|x| (x-m).powi(2)).sum::<f32>() / v.len() as f32).sqrt() }

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== Inductive (Cold-Start) Evaluation ===\n");

    let config = Config::load_default()?;
    let base_seed = config.project.seed;
    let mut rng = StdRng::seed_from_u64(base_seed);

    let priors = PriorKnowledge::from_file("data/priors/merged_priors.json")?;
    let builder = PriorDatasetBuilder::new(priors.clone());
    let num_tfs = builder.num_tfs();
    let num_genes = builder.num_genes();

    let expr_data = ExpressionData::from_processed_dir(
        "data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    )?;
    let expr_dim = expr_data.n_cell_types;

    // Build expression maps
    let mut gene_to_expr_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in expr_data.gene_names.iter().enumerate() { gene_to_expr_idx.insert(name.clone(), i); }
    let mut tf_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    let mut gene_expr_map: HashMap<usize, Array1<f32>> = HashMap::new();
    for (name, &idx) in builder.get_tf_vocab().iter() {
        tf_expr_map.insert(idx, gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned()).unwrap_or_else(|| Array1::zeros(expr_dim)));
    }
    for (name, &idx) in builder.get_gene_vocab().iter() {
        gene_expr_map.insert(idx, gene_to_expr_idx.get(name)
            .map(|&i| expr_data.expression.column(i).to_owned()).unwrap_or_else(|| Array1::zeros(expr_dim)));
    }

    // Build full dataset
    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, g)| (tf, g, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, g)| (tf, g, 0.0)));
    examples.shuffle(&mut rng);

    let seeds = [42u64, 123, 456];
    let embed_dim = 512usize;
    let hidden_dim = 512usize;
    let output_dim = 512usize;
    let temperature = 0.05f32;
    let lr = 0.001f32;
    let ce_lr = 0.005f32;
    let clip = 5.0f32;
    let batch_size = 256usize;
    let epochs = 60usize;

    let mut results: Vec<serde_json::Value> = Vec::new();

    for &seed in &seeds {
        println!("=== Seed {} ===", seed);
        let mut split_rng = StdRng::seed_from_u64(seed + 7777);

        // Hold out 20% of TFs and 20% of genes
        let mut all_tfs: Vec<usize> = (0..num_tfs).collect();
        let mut all_genes: Vec<usize> = (0..num_genes).collect();
        all_tfs.shuffle(&mut split_rng);
        all_genes.shuffle(&mut split_rng);

        let n_heldout_tfs = (num_tfs as f32 * 0.2) as usize;
        let n_heldout_genes = (num_genes as f32 * 0.2) as usize;
        let heldout_tfs: HashSet<usize> = all_tfs[..n_heldout_tfs].iter().cloned().collect();
        let heldout_genes: HashSet<usize> = all_genes[..n_heldout_genes].iter().cloned().collect();

        println!("  Held-out TFs: {} | Held-out genes: {}", n_heldout_tfs, n_heldout_genes);

        // Split examples: train uses only seen entities, test uses cold-start pairs
        let mut train_data: Vec<(usize, usize, f32)> = Vec::new();
        let mut cold_tf: Vec<(usize, usize, f32)> = Vec::new();     // unseen TF, seen gene
        let mut cold_gene: Vec<(usize, usize, f32)> = Vec::new();   // seen TF, unseen gene
        let mut cold_both: Vec<(usize, usize, f32)> = Vec::new();   // both unseen

        for &(tf, gene, label) in &examples {
            let tf_held = heldout_tfs.contains(&tf);
            let gene_held = heldout_genes.contains(&gene);
            match (tf_held, gene_held) {
                (false, false) => train_data.push((tf, gene, label)),
                (true, false)  => cold_tf.push((tf, gene, label)),
                (false, true)  => cold_gene.push((tf, gene, label)),
                (true, true)   => cold_both.push((tf, gene, label)),
            }
        }
        println!("  Train: {} | Cold-TF: {} | Cold-Gene: {} | Cold-Both: {}",
                 train_data.len(), cold_tf.len(), cold_gene.len(), cold_both.len());

        // ── Two-Tower ──
        let mut tt_model = HybridEmbeddingModel::new(
            num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, temperature, 0.01, seed,
        );
        let mut tt_state = AdamTT::new(&tt_model);
        let mut train_shuffled = train_data.clone();

        for epoch in 0..epochs {
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            train_shuffled.shuffle(&mut epoch_rng);
            for start in (0..train_shuffled.len()).step_by(batch_size) {
                let end = (start + batch_size).min(train_shuffled.len());
                let batch = &train_shuffled[start..end];
                let bsz = batch.len();
                let tf_idx: Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
                let tf_e = build_expr_batch(&tf_idx, &tf_expr_map, expr_dim);
                let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                let preds = tt_model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let grad: Array1<f32> = preds.iter().zip(labels.iter()).map(|(&p, &l)| (p-l)/bsz as f32).collect();
                tt_model.backward(&grad);
                adam_step_tt(&mut tt_model, &mut tt_state, lr, clip);
                tt_model.zero_grad();
            }
        }

        // Evaluate two-tower on each cold-start split
        let tt_transductive = eval_tt(&mut tt_model, &train_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        let tt_cold_tf = eval_tt(&mut tt_model, &cold_tf, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        let tt_cold_gene = eval_tt(&mut tt_model, &cold_gene, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        let tt_cold_both = eval_tt(&mut tt_model, &cold_both, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);

        // ── Cross-Encoder ──
        let mut ce_model = CrossEncoderModel::new(num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, seed);
        let mut ce_state = AdamCE::new(&ce_model);
        let mut train_shuffled = train_data.clone();

        for epoch in 0..epochs {
            let mut epoch_rng = StdRng::seed_from_u64(seed + epoch as u64);
            train_shuffled.shuffle(&mut epoch_rng);
            for start in (0..train_shuffled.len()).step_by(batch_size) {
                let end = (start + batch_size).min(train_shuffled.len());
                let batch = &train_shuffled[start..end];
                let tf_idx: Vec<usize> = batch.iter().map(|x| x.0).collect();
                let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
                let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
                let tf_e = build_expr_batch(&tf_idx, &tf_expr_map, expr_dim);
                let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
                let logits = ce_model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let labels_arr = Array1::from(labels);
                let grad = bce_loss_backward(&logits, &labels_arr);
                ce_model.backward(&grad);
                adam_step_ce(&mut ce_model, &mut ce_state, ce_lr, 1.0);
                ce_model.zero_grad();
            }
        }

        let ce_transductive = eval_ce(&mut ce_model, &train_data, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        let ce_cold_tf = eval_ce(&mut ce_model, &cold_tf, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        let ce_cold_gene = eval_ce(&mut ce_model, &cold_gene, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);
        let ce_cold_both = eval_ce(&mut ce_model, &cold_both, &tf_expr_map, &gene_expr_map, expr_dim, batch_size);

        println!("  Two-Tower:     trans={:.4} cold_tf={:.4} cold_gene={:.4} cold_both={:.4}",
                 tt_transductive.0, tt_cold_tf.0, tt_cold_gene.0, tt_cold_both.0);
        println!("  Cross-Encoder: trans={:.4} cold_tf={:.4} cold_gene={:.4} cold_both={:.4}",
                 ce_transductive.0, ce_cold_tf.0, ce_cold_gene.0, ce_cold_both.0);

        results.push(serde_json::json!({
            "seed": seed,
            "n_heldout_tfs": n_heldout_tfs,
            "n_heldout_genes": n_heldout_genes,
            "n_train": train_data.len(),
            "n_cold_tf": cold_tf.len(),
            "n_cold_gene": cold_gene.len(),
            "n_cold_both": cold_both.len(),
            "two_tower": {
                "transductive": {"auroc": tt_transductive.0, "acc": tt_transductive.1, "f1": tt_transductive.2},
                "cold_tf":      {"auroc": tt_cold_tf.0,      "acc": tt_cold_tf.1,      "f1": tt_cold_tf.2},
                "cold_gene":    {"auroc": tt_cold_gene.0,    "acc": tt_cold_gene.1,    "f1": tt_cold_gene.2},
                "cold_both":    {"auroc": tt_cold_both.0,    "acc": tt_cold_both.1,    "f1": tt_cold_both.2},
            },
            "cross_encoder": {
                "transductive": {"auroc": ce_transductive.0, "acc": ce_transductive.1, "f1": ce_transductive.2},
                "cold_tf":      {"auroc": ce_cold_tf.0,      "acc": ce_cold_tf.1,      "f1": ce_cold_tf.2},
                "cold_gene":    {"auroc": ce_cold_gene.0,    "acc": ce_cold_gene.1,    "f1": ce_cold_gene.2},
                "cold_both":    {"auroc": ce_cold_both.0,    "acc": ce_cold_both.1,    "f1": ce_cold_both.2},
            },
        }));
    }

    // Aggregate across seeds
    let tt_cold_aurocs: Vec<f32> = results.iter()
        .map(|r| r["two_tower"]["cold_both"]["auroc"].as_f64().unwrap() as f32).collect();
    let ce_cold_aurocs: Vec<f32> = results.iter()
        .map(|r| r["cross_encoder"]["cold_both"]["auroc"].as_f64().unwrap() as f32).collect();

    let output = serde_json::json!({
        "experiment": "inductive_cold_start",
        "holdout_fraction": 0.2,
        "seeds": [42, 123, 456],
        "per_seed": results,
        "summary": {
            "two_tower_cold_both_auroc_mean": mean(&tt_cold_aurocs),
            "two_tower_cold_both_auroc_std": std_dev(&tt_cold_aurocs),
            "cross_encoder_cold_both_auroc_mean": mean(&ce_cold_aurocs),
            "cross_encoder_cold_both_auroc_std": std_dev(&ce_cold_aurocs),
        },
    });

    std::fs::create_dir_all("results")?;
    std::fs::write("results/inductive_eval.json", serde_json::to_string_pretty(&output)?)?;
    println!("\n✓ Saved results/inductive_eval.json");
    Ok(())
}

fn eval_tt(
    model: &mut HybridEmbeddingModel, data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>, gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize, batch_size: usize,
) -> (f32, f32, f32) {
    if data.is_empty() { return (0.5, 0.5, 0.0); }
    let mut all_preds: Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
        let tf_e = build_expr_batch(&tf_idx, tf_expr_map, expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        all_preds.extend_from_slice(preds.as_slice().unwrap());
        all_labels.extend_from_slice(&labels);
    }
    let acc = all_preds.iter().zip(all_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / all_preds.len() as f32;
    (calculate_auroc(&all_preds, &all_labels), acc, calculate_f1(&all_preds, &all_labels))
}

fn eval_ce(
    model: &mut CrossEncoderModel, data: &[(usize, usize, f32)],
    tf_expr_map: &HashMap<usize, Array1<f32>>, gene_expr_map: &HashMap<usize, Array1<f32>>,
    expr_dim: usize, batch_size: usize,
) -> (f32, f32, f32) {
    if data.is_empty() { return (0.5, 0.5, 0.0); }
    let mut all_preds: Vec<f32> = Vec::new();
    let mut all_labels: Vec<f32> = Vec::new();
    for start in (0..data.len()).step_by(batch_size) {
        let end = (start + batch_size).min(data.len());
        let batch = &data[start..end];
        let tf_idx: Vec<usize> = batch.iter().map(|x| x.0).collect();
        let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
        let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
        let tf_e = build_expr_batch(&tf_idx, tf_expr_map, expr_dim);
        let gene_e = build_expr_batch(&gene_idx, gene_expr_map, expr_dim);
        let logits = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
        let probs: Vec<f32> = logits.iter().map(|&x| 1.0/(1.0+(-x).exp())).collect();
        all_preds.extend_from_slice(&probs);
        all_labels.extend_from_slice(&labels);
    }
    let acc = all_preds.iter().zip(all_labels.iter())
        .filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / all_preds.len() as f32;
    (calculate_auroc(&all_preds, &all_labels), acc, calculate_f1(&all_preds, &all_labels))
}
