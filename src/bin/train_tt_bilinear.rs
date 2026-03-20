/// Two-tower with bilinear scoring (e_TF^T W e_gene) instead of cosine similarity.
/// 5 seeds at 1:1. Writes results/two_tower_bilinear.json
use module_regularized_grn::{
    Config,
    models::bilinear_two_tower::BilinearTwoTowerModel,
    data::{PriorKnowledge, PriorDatasetBuilder, expression::ExpressionData},
};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use anyhow::Result;
use std::collections::HashMap;

// ── Adam for BilinearTwoTowerModel ───────────────────────────────────────────

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

struct AdamBilinear {
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
    m_w: Array2<f32>, v_w: Array2<f32>,
    t: i32,
}
impl AdamBilinear {
    fn new(m: &BilinearTwoTowerModel) -> Self {
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
            m_w: Array2::zeros(m.bilinear_w.dim()), v_w: Array2::zeros(m.bilinear_w.dim()),
            t: 0,
        }
    }
}

fn adam_step(model: &mut BilinearTwoTowerModel, s: &mut AdamBilinear, lr: f32, clip: f32) {
    s.t += 1;
    let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
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
        model.bilinear_w_grad.iter().map(|x| x*x).sum::<f32>(),
    ].iter().sum();
    let scale = if norm_sq.sqrt() > clip { clip / norm_sq.sqrt() } else { 1.0 };

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
    adam2d(&mut model.bilinear_w, &model.bilinear_w_grad, &mut s.m_w, &mut s.v_w, lr, b1, b2, eps, s.t, scale);
}

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

fn main() -> Result<()> {
    println!("=== Two-Tower Bilinear Scoring ===\n");

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

    let positives = builder.get_positive_examples();
    let negatives = builder.sample_negative_examples(positives.len(), base_seed);
    let mut examples: Vec<(usize, usize, f32)> = Vec::new();
    examples.extend(positives.iter().map(|&(tf, g)| (tf, g, 1.0)));
    examples.extend(negatives.iter().map(|&(tf, g)| (tf, g, 0.0)));
    examples.shuffle(&mut rng);

    let n = examples.len();
    let n_train = (n as f32 * 0.7) as usize;
    let n_val = (n as f32 * 0.15) as usize;
    let train_data = examples[..n_train].to_vec();
    let _val_data = examples[n_train..n_train+n_val].to_vec();
    let test_data = examples[n_train+n_val..].to_vec();

    let embed_dim = 512usize;
    let hidden_dim = 512usize;
    let output_dim = 512usize;
    let lr = 0.001f32;
    let clip = 5.0f32;
    let batch_size = 256usize;
    let epochs = 60usize;
    let seeds = [42u64, 123, 456, 789, 1337];

    let mut seed_accs: Vec<f32> = Vec::new();
    let mut seed_aurocs: Vec<f32> = Vec::new();
    let mut seed_f1s: Vec<f32> = Vec::new();

    for &seed in &seeds {
        println!("Seed {}:", seed);
        let mut model = BilinearTwoTowerModel::new(
            num_tfs, num_genes, embed_dim, expr_dim, hidden_dim, output_dim, 0.01, seed,
        );
        let mut state = AdamBilinear::new(&model);
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
                let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
                let grad: Array1<f32> = preds.iter().zip(labels.iter()).map(|(&p, &l)| (p-l)/bsz as f32).collect();
                model.backward(&grad);
                adam_step(&mut model, &mut state, lr, clip);
                model.zero_grad();
            }
        }

        // Evaluate
        let mut all_preds: Vec<f32> = Vec::new();
        let mut all_labels: Vec<f32> = Vec::new();
        for start in (0..test_data.len()).step_by(batch_size) {
            let end = (start + batch_size).min(test_data.len());
            let batch = &test_data[start..end];
            let tf_idx: Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let labels: Vec<f32> = batch.iter().map(|x| x.2).collect();
            let tf_e = build_expr_batch(&tf_idx, &tf_expr_map, expr_dim);
            let gene_e = build_expr_batch(&gene_idx, &gene_expr_map, expr_dim);
            let preds = model.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
            all_preds.extend_from_slice(preds.as_slice().unwrap());
            all_labels.extend_from_slice(&labels);
        }
        let acc = all_preds.iter().zip(all_labels.iter()).filter(|(&p, &l)| (p >= 0.5) == (l == 1.0)).count() as f32 / all_preds.len() as f32;
        let auroc = calculate_auroc(&all_preds, &all_labels);
        let f1 = calculate_f1(&all_preds, &all_labels);
        println!("  acc={:.2}% auroc={:.4} f1={:.4} params={}", acc*100.0, auroc, f1, model.num_parameters());
        seed_accs.push(acc);
        seed_aurocs.push(auroc);
        seed_f1s.push(f1);
    }

    let output = serde_json::json!({
        "experiment": "two_tower_bilinear",
        "scoring": "bilinear (e_TF^T W e_gene)",
        "seeds": seeds,
        "seed_accuracies": seed_accs,
        "seed_aurocs": seed_aurocs,
        "seed_f1s": seed_f1s,
        "mean_accuracy": mean(&seed_accs),
        "std_accuracy": std_dev(&seed_accs),
        "mean_auroc": mean(&seed_aurocs),
        "std_auroc": std_dev(&seed_aurocs),
    });

    std::fs::create_dir_all("results")?;
    std::fs::write("results/two_tower_bilinear.json", serde_json::to_string_pretty(&output)?)?;
    println!("\n✓ Saved results/two_tower_bilinear.json");
    Ok(())
}
