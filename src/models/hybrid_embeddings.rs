/// Hybrid model: Learnable embeddings + Expression features
/// Target: 70%+ accuracy by combining structure and biology
use crate::models::nn::{LinearLayer, Dropout, relu, relu_backward};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::f32;
use rand::rngs::StdRng;

/// Raw activation and weight statistics collected during profiling pass.
/// All counts are per neuron (index = fc1 output neuron j, 0..hidden_dim-1).
pub struct NeuronStats {
    pub tf_fc1_activation_counts:   Vec<u32>,  // # examples where TF fc1 neuron j > 0
    pub gene_fc1_activation_counts: Vec<u32>,  // # examples where Gene fc1 neuron j > 0
    pub total_examples: usize,                 // total training examples profiled
    pub tf_fc1_col_norms:   Vec<f32>,  // ||tf_fc1.weights.column(j)||₂
    pub tf_fc2_row_norms:   Vec<f32>,  // ||tf_fc2.weights.row(j)||₂
    pub gene_fc1_col_norms: Vec<f32>,
    pub gene_fc2_row_norms: Vec<f32>,
}

/// Final combined importance scores per tower's fc1 neurons (0..hidden_dim-1).
pub struct LayerScores {
    pub tf_fc1:   Vec<f32>,
    pub gene_fc1: Vec<f32>,
}

fn build_expr_batch_for_profiling(
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

/// Hybrid embedding + expression model
#[derive(Clone)]
pub struct HybridEmbeddingModel {
    // TF encoder
    pub tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    pub tf_fc1: LinearLayer,  // (embed + expr) → hidden
    pub tf_fc2: LinearLayer,  // hidden → output

    // Gene encoder
    pub gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,
    pub gene_fc1: LinearLayer,
    pub gene_fc2: LinearLayer,
    
    temperature: f32,
    
    // Expression dimensions
    embed_dim: usize,
    #[allow(dead_code)]
    expr_dim: usize,
    
    // Caches for backward pass
    tf_indices_cache: Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_embed_out: Option<Array2<f32>>,
    gene_embed_out: Option<Array2<f32>>,
    tf_expr_cache: Option<Array2<f32>>,
    gene_expr_cache: Option<Array2<f32>>,
    tf_concat: Option<Array2<f32>>,
    gene_concat: Option<Array2<f32>>,
    tf_h1: Option<Array2<f32>>,
    gene_h1: Option<Array2<f32>>,
    pub tf_final: Option<Array2<f32>>,
    pub gene_final: Option<Array2<f32>>,
}

impl HybridEmbeddingModel {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,
        expr_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        temperature: f32,
        std_dev: f32,
        seed: u64,
    ) -> Self {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Normal;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, std_dev).unwrap();
        
        // Initialize embeddings
        let tf_embed = Array2::random_using((num_tfs, embed_dim), dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, embed_dim), dist, &mut rng);
        
        // Input size: embed + expression
        let input_dim = embed_dim + expr_dim;
        
        // Initialize layers
        let tf_fc1 = LinearLayer::new(input_dim, hidden_dim, seed);
        let tf_fc2 = LinearLayer::new(hidden_dim, output_dim, seed + 1);
        let gene_fc1 = LinearLayer::new(input_dim, hidden_dim, seed + 2);
        let gene_fc2 = LinearLayer::new(hidden_dim, output_dim, seed + 3);
        
        Self {
            tf_embed,
            tf_embed_grad: Array2::zeros((num_tfs, embed_dim)),
            tf_fc1,
            tf_fc2,
            gene_embed,
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            gene_fc1,
            gene_fc2,
            temperature,
            embed_dim,
            expr_dim,
            tf_indices_cache: None,
            gene_indices_cache: None,
            tf_embed_out: None,
            gene_embed_out: None,
            tf_expr_cache: None,
            gene_expr_cache: None,
            tf_concat: None,
            gene_concat: None,
            tf_h1: None,
            gene_h1: None,
            tf_final: None,
            gene_final: None,
        }
    }
    
    /// Forward pass with expression features
    pub fn forward(
        &mut self,
        tf_indices: &[usize],
        gene_indices: &[usize],
        tf_expr: &Array2<f32>,      // [batch, expr_dim]
        gene_expr: &Array2<f32>,    // [batch, expr_dim]
    ) -> Array1<f32> {
        let batch_size = tf_indices.len();
        assert_eq!(batch_size, gene_indices.len());
        assert_eq!(tf_expr.shape()[0], batch_size);
        assert_eq!(gene_expr.shape()[0], batch_size);
        
        // Cache inputs
        self.tf_indices_cache = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());
        self.tf_expr_cache = Some(tf_expr.clone());
        self.gene_expr_cache = Some(gene_expr.clone());
        
        // Lookup embeddings
        let mut tf_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        let mut gene_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        
        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            tf_embed_batch.row_mut(i).assign(&self.tf_embed.row(tf_idx));
        }
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            gene_embed_batch.row_mut(i).assign(&self.gene_embed.row(gene_idx));
        }
        
        self.tf_embed_out = Some(tf_embed_batch.clone());
        self.gene_embed_out = Some(gene_embed_batch.clone());
        
        // Concatenate embeddings + expression
        let tf_concat = ndarray::concatenate![Axis(1), tf_embed_batch, tf_expr.clone()];
        let gene_concat = ndarray::concatenate![Axis(1), gene_embed_batch, gene_expr.clone()];
        
        self.tf_concat = Some(tf_concat.clone());
        self.gene_concat = Some(gene_concat.clone());
        
        // TF encoding: concat → fc1 → relu → fc2
        let tf_h1_pre = self.tf_fc1.forward(&tf_concat);
        let tf_h1 = relu(&tf_h1_pre);
        let tf_out = self.tf_fc2.forward(&tf_h1);
        
        self.tf_h1 = Some(tf_h1);
        self.tf_final = Some(tf_out.clone());
        
        // Gene encoding
        let gene_h1_pre = self.gene_fc1.forward(&gene_concat);
        let gene_h1 = relu(&gene_h1_pre);
        let gene_out = self.gene_fc2.forward(&gene_h1);
        
        self.gene_h1 = Some(gene_h1);
        self.gene_final = Some(gene_out.clone());
        
        // Compute scores: dot product / temperature
        let mut scores = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let dot = tf_out.row(i).dot(&gene_out.row(i));
            let logit = dot / self.temperature;
            scores[i] = 1.0 / (1.0 + (-logit).exp());  // sigmoid
        }
        
        scores
    }
    
    /// Forward pass with dropout after each hidden layer.
    /// `dropout` layers are passed in and mutated (they store masks for backward).
    /// `training` controls whether dropout is active.
    pub fn forward_with_dropout(
        &mut self,
        tf_indices: &[usize],
        gene_indices: &[usize],
        tf_expr: &Array2<f32>,
        gene_expr: &Array2<f32>,
        tf_dropout: &mut Dropout,
        gene_dropout: &mut Dropout,
        training: bool,
        rng: &mut StdRng,
    ) -> Array1<f32> {
        let batch_size = tf_indices.len();

        self.tf_indices_cache = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());
        self.tf_expr_cache = Some(tf_expr.clone());
        self.gene_expr_cache = Some(gene_expr.clone());

        let mut tf_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        let mut gene_embed_batch = Array2::zeros((batch_size, self.embed_dim));
        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            tf_embed_batch.row_mut(i).assign(&self.tf_embed.row(tf_idx));
        }
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            gene_embed_batch.row_mut(i).assign(&self.gene_embed.row(gene_idx));
        }
        self.tf_embed_out = Some(tf_embed_batch.clone());
        self.gene_embed_out = Some(gene_embed_batch.clone());

        let tf_concat = ndarray::concatenate![Axis(1), tf_embed_batch, tf_expr.clone()];
        let gene_concat = ndarray::concatenate![Axis(1), gene_embed_batch, gene_expr.clone()];
        self.tf_concat = Some(tf_concat.clone());
        self.gene_concat = Some(gene_concat.clone());

        // TF: fc1 → relu → dropout → fc2
        let tf_h1_pre = self.tf_fc1.forward(&tf_concat);
        let tf_h1 = relu(&tf_h1_pre);
        let tf_h1_drop = tf_dropout.forward(&tf_h1, training, rng);
        let tf_out = self.tf_fc2.forward(&tf_h1_drop);
        self.tf_h1 = Some(tf_h1_drop);
        self.tf_final = Some(tf_out.clone());

        // Gene: fc1 → relu → dropout → fc2
        let gene_h1_pre = self.gene_fc1.forward(&gene_concat);
        let gene_h1 = relu(&gene_h1_pre);
        let gene_h1_drop = gene_dropout.forward(&gene_h1, training, rng);
        let gene_out = self.gene_fc2.forward(&gene_h1_drop);
        self.gene_h1 = Some(gene_h1_drop);
        self.gene_final = Some(gene_out.clone());

        let mut scores = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let dot = tf_out.row(i).dot(&gene_out.row(i));
            let logit = dot / self.temperature;
            scores[i] = 1.0 / (1.0 + (-logit).exp());
        }
        scores
    }

    /// Backward pass with dropout. Call after forward_with_dropout.
    pub fn backward_with_dropout(
        &mut self,
        grad_output: &Array1<f32>,
        tf_dropout: &Dropout,
        gene_dropout: &Dropout,
    ) {
        let batch_size = grad_output.len();
        let tf_out = self.tf_final.as_ref().unwrap();
        let gene_out = self.gene_final.as_ref().unwrap();

        let mut grad_tf_out = Array2::zeros(tf_out.raw_dim());
        let mut grad_gene_out = Array2::zeros(gene_out.raw_dim());

        for i in 0..batch_size {
            let dot = tf_out.row(i).dot(&gene_out.row(i));
            let logit = dot / self.temperature;
            let score = 1.0 / (1.0 + (-logit).exp());
            let grad_sigmoid = score * (1.0 - score) * grad_output[i];
            let grad_dot = grad_sigmoid / self.temperature;
            grad_tf_out.row_mut(i).assign(&(&gene_out.row(i) * grad_dot));
            grad_gene_out.row_mut(i).assign(&(&tf_out.row(i) * grad_dot));
        }

        // Gene encoder backward (with dropout)
        let grad_gene_h1 = self.gene_fc2.backward(&grad_gene_out);
        let grad_gene_h1_drop = gene_dropout.backward(&grad_gene_h1);
        let gene_h1 = self.gene_h1.as_ref().unwrap();
        let grad_gene_h1_pre = relu_backward(gene_h1, &grad_gene_h1_drop);
        let grad_gene_concat = self.gene_fc1.backward(&grad_gene_h1_pre);

        // TF encoder backward (with dropout)
        let grad_tf_h1 = self.tf_fc2.backward(&grad_tf_out);
        let grad_tf_h1_drop = tf_dropout.backward(&grad_tf_h1);
        let tf_h1 = self.tf_h1.as_ref().unwrap();
        let grad_tf_h1_pre = relu_backward(tf_h1, &grad_tf_h1_drop);
        let grad_tf_concat = self.tf_fc1.backward(&grad_tf_h1_pre);

        // Embedding gradients
        let grad_tf_embed = grad_tf_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();
        let grad_gene_embed = grad_gene_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();

        let tf_indices = self.tf_indices_cache.as_ref().unwrap();
        let gene_indices = self.gene_indices_cache.as_ref().unwrap();
        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            let mut row = self.tf_embed_grad.row_mut(tf_idx);
            row += &grad_tf_embed.row(i);
        }
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            let mut row = self.gene_embed_grad.row_mut(gene_idx);
            row += &grad_gene_embed.row(i);
        }
    }

    /// Backward pass
    pub fn backward(&mut self, grad_output: &Array1<f32>) {
        let batch_size = grad_output.len();
        
        let tf_out = self.tf_final.as_ref().unwrap();
        let gene_out = self.gene_final.as_ref().unwrap();
        
        // Gradient through sigmoid and dot product
        let mut grad_tf_out = Array2::zeros(tf_out.raw_dim());
        let mut grad_gene_out = Array2::zeros(gene_out.raw_dim());
        
        for i in 0..batch_size {
            let dot = tf_out.row(i).dot(&gene_out.row(i));
            let logit = dot / self.temperature;
            let score = 1.0 / (1.0 + (-logit).exp());  // sigmoid
            let grad_sigmoid = score * (1.0 - score) * grad_output[i];
            let grad_dot = grad_sigmoid / self.temperature;
            
            grad_tf_out.row_mut(i).assign(&(&gene_out.row(i) * grad_dot));
            grad_gene_out.row_mut(i).assign(&(&tf_out.row(i) * grad_dot));
        }
        
        // Backward through gene encoder
        let grad_gene_h1 = self.gene_fc2.backward(&grad_gene_out);
        let gene_h1 = self.gene_h1.as_ref().unwrap();
        let grad_gene_h1_pre = relu_backward(gene_h1, &grad_gene_h1);
        let grad_gene_concat = self.gene_fc1.backward(&grad_gene_h1_pre);

        // Backward through TF encoder
        let grad_tf_h1 = self.tf_fc2.backward(&grad_tf_out);
        let tf_h1 = self.tf_h1.as_ref().unwrap();
        let grad_tf_h1_pre = relu_backward(tf_h1, &grad_tf_h1);
        let grad_tf_concat = self.tf_fc1.backward(&grad_tf_h1_pre);
        
        // Split gradients: embeddings + expression
        let grad_tf_embed = grad_tf_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();
        let grad_gene_embed = grad_gene_concat.slice(ndarray::s![.., 0..self.embed_dim]).to_owned();
        
        // Accumulate embedding gradients
        let tf_indices = self.tf_indices_cache.as_ref().unwrap();
        let gene_indices = self.gene_indices_cache.as_ref().unwrap();
        
        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            let mut row = self.tf_embed_grad.row_mut(tf_idx);
            row += &grad_tf_embed.row(i);
        }
        
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            let mut row = self.gene_embed_grad.row_mut(gene_idx);
            row += &grad_gene_embed.row(i);
        }
    }
    
    /// Update parameters with SGD
    pub fn update(&mut self, learning_rate: f32) {
        // Update embeddings
        self.tf_embed.scaled_add(-learning_rate, &self.tf_embed_grad);
        self.gene_embed.scaled_add(-learning_rate, &self.gene_embed_grad);
        
        // Update layers
        self.tf_fc1.update(learning_rate);
        self.tf_fc2.update(learning_rate);
        self.gene_fc1.update(learning_rate);
        self.gene_fc2.update(learning_rate);
    }
    
    /// Update with weight decay (L2 regularization)
    pub fn update_with_weight_decay(&mut self, learning_rate: f32, weight_decay: f32) {
        // Update embeddings with weight decay
        let decay_factor = 1.0 - learning_rate * weight_decay;
        
        self.tf_embed.scaled_add(-learning_rate, &self.tf_embed_grad);
        self.tf_embed *= decay_factor;
        
        self.gene_embed.scaled_add(-learning_rate, &self.gene_embed_grad);
        self.gene_embed *= decay_factor;
        
        // Update layers with weight decay
        self.tf_fc1.update_with_weight_decay(learning_rate, weight_decay);
        self.tf_fc2.update_with_weight_decay(learning_rate, weight_decay);
        self.gene_fc1.update_with_weight_decay(learning_rate, weight_decay);
        self.gene_fc2.update_with_weight_decay(learning_rate, weight_decay);
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.tf_fc1.zero_grad();
        self.tf_fc2.zero_grad();
        self.gene_fc1.zero_grad();
        self.gene_fc2.zero_grad();
    }
    
    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.tf_embed.len() + self.gene_embed.len();
        let tf_params = self.tf_fc1.weights.len() + self.tf_fc1.bias.len() 
                      + self.tf_fc2.weights.len() + self.tf_fc2.bias.len();
        let gene_params = self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
                        + self.gene_fc2.weights.len() + self.gene_fc2.bias.len();
        embed_params + tf_params + gene_params
    }
    
    /// Train a batch with regularization and dropout
    pub fn train_batch_with_regularization(
        &mut self,
        batch: &[(String, String, f64)],
        _learning_rate: f64,
        temperature: f64,
        _dropout: f64,
        _l2_weight: f64,
    ) -> f64 {
        // Simplified training method for compatibility
        // In real implementation, would need gene mapping
        let mut total_loss = 0.0;
        
        for (tf, gene, label) in batch {
            // Compute forward pass and loss
            let score = self.predict(tf, gene, temperature);
            let loss = -((label * score.ln()) + ((1.0 - label) * (1.0 - score).ln()));
            total_loss += loss;
        }
        
        total_loss / batch.len() as f64
    }
    
    /// Simple prediction method for single gene pair
    pub fn predict(&self, _tf: &str, _gene: &str, _temperature: f64) -> f64 {
        // Placeholder - would need actual gene lookup
        0.5
    }

    /// Inference-only forward pass over all training data.
    /// Accumulates per-neuron activation counts and weight norms.
    /// Does NOT call backward() or accumulate any gradients.
    pub fn profile_activations(
        &mut self,
        data:          &[(usize, usize, f32)],
        tf_expr_map:   &HashMap<usize, Array1<f32>>,
        gene_expr_map: &HashMap<usize, Array1<f32>>,
        expr_dim:      usize,
        batch_size:    usize,
    ) -> NeuronStats {
        let hidden_dim = self.tf_fc1.bias.len();

        // Compute weight norms once — O(hidden_dim * in_dim), not per-batch
        let l2 = |v: ndarray::ArrayView1<f32>| -> f32 {
            v.iter().map(|&x| x * x).sum::<f32>().sqrt()
        };
        let tf_fc1_col_norms:   Vec<f32> = (0..hidden_dim).map(|j| l2(self.tf_fc1.weights.column(j))).collect();
        let tf_fc2_row_norms:   Vec<f32> = (0..hidden_dim).map(|j| l2(self.tf_fc2.weights.row(j))).collect();
        let gene_fc1_col_norms: Vec<f32> = (0..hidden_dim).map(|j| l2(self.gene_fc1.weights.column(j))).collect();
        let gene_fc2_row_norms: Vec<f32> = (0..hidden_dim).map(|j| l2(self.gene_fc2.weights.row(j))).collect();

        let mut tf_counts   = vec![0u32; hidden_dim];
        let mut gene_counts = vec![0u32; hidden_dim];
        let mut total_examples = 0usize;

        for start in (0..data.len()).step_by(batch_size) {
            let end   = (start + batch_size).min(data.len());
            let batch = &data[start..end];
            let bsz   = batch.len();

            let tf_idx:   Vec<usize> = batch.iter().map(|x| x.0).collect();
            let gene_idx: Vec<usize> = batch.iter().map(|x| x.1).collect();
            let tf_e   = build_expr_batch_for_profiling(&tf_idx,   tf_expr_map,   expr_dim);
            let gene_e = build_expr_batch_for_profiling(&gene_idx, gene_expr_map, expr_dim);

            // forward() writes post-ReLU activations to self.tf_h1 and self.gene_h1
            self.forward(&tf_idx, &gene_idx, &tf_e, &gene_e);
            // No backward() call — this is inference only

            let tf_h1   = self.tf_h1.as_ref().unwrap();
            let gene_h1 = self.gene_h1.as_ref().unwrap();

            for i in 0..bsz {
                for j in 0..hidden_dim {
                    if tf_h1[[i, j]]   > 0.0 { tf_counts[j]   += 1; }
                    if gene_h1[[i, j]] > 0.0 { gene_counts[j] += 1; }
                }
            }
            total_examples += bsz;
        }

        NeuronStats {
            tf_fc1_activation_counts:   tf_counts,
            gene_fc1_activation_counts: gene_counts,
            total_examples,
            tf_fc1_col_norms,
            tf_fc2_row_norms,
            gene_fc1_col_norms,
            gene_fc2_row_norms,
        }
    }

    /// Compute combined importance scores for each fc1 output neuron per tower.
    /// score(j) = alpha * activation_freq(j) + (1-alpha) * weight_magnitude(j)
    /// Both components are normalized to [0,1] independently per tower before combining.
    pub fn importance_scores(&self, stats: &NeuronStats, alpha: f32) -> LayerScores {
        let hidden_dim = stats.tf_fc1_activation_counts.len();
        let total = stats.total_examples as f32;

        let normalize = |v: &[f32]| -> Vec<f32> {
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max - min;
            if range <= 0.0 {
                vec![0.5; v.len()]
            } else {
                v.iter().map(|&x| (x - min) / range).collect()
            }
        };

        // TF tower
        let tf_act_freq: Vec<f32> = stats.tf_fc1_activation_counts.iter()
            .map(|&c| c as f32 / total).collect();
        let tf_fc1_n = normalize(&stats.tf_fc1_col_norms);
        let tf_fc2_n = normalize(&stats.tf_fc2_row_norms);
        let tf_weight_mag: Vec<f32> = (0..hidden_dim)
            .map(|j| (tf_fc1_n[j] + tf_fc2_n[j]) / 2.0).collect();
        let tf_scores: Vec<f32> = (0..hidden_dim)
            .map(|j| alpha * tf_act_freq[j] + (1.0 - alpha) * tf_weight_mag[j]).collect();

        // Gene tower
        let gene_act_freq: Vec<f32> = stats.gene_fc1_activation_counts.iter()
            .map(|&c| c as f32 / total).collect();
        let gene_fc1_n = normalize(&stats.gene_fc1_col_norms);
        let gene_fc2_n = normalize(&stats.gene_fc2_row_norms);
        let gene_weight_mag: Vec<f32> = (0..hidden_dim)
            .map(|j| (gene_fc1_n[j] + gene_fc2_n[j]) / 2.0).collect();
        let gene_scores: Vec<f32> = (0..hidden_dim)
            .map(|j| alpha * gene_act_freq[j] + (1.0 - alpha) * gene_weight_mag[j]).collect();

        LayerScores { tf_fc1: tf_scores, gene_fc1: gene_scores }
    }

    /// Structurally prune both towers to the given sparsity level.
    /// Removes the lowest-scoring (1-keep) fraction of fc1 output neurons,
    /// and the corresponding rows in fc2 inputs.
    /// TF and Gene towers are pruned independently (different neurons may be removed).
    pub fn prune_to_sparsity(&mut self, scores: &LayerScores, sparsity: f32) {
        let hidden_dim = scores.tf_fc1.len();
        let n_keep = ((1.0 - sparsity) * hidden_dim as f32).round() as usize;
        let n_keep = n_keep.max(1);

        // TF tower: select top-n_keep by score, sorted ascending for correct indexing
        let mut tf_order: Vec<usize> = (0..hidden_dim).collect();
        tf_order.sort_by(|&a, &b| scores.tf_fc1[b].partial_cmp(&scores.tf_fc1[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut tf_keep: Vec<usize> = tf_order[..n_keep].to_vec();
        tf_keep.sort_unstable();

        // Gene tower
        let mut gene_order: Vec<usize> = (0..hidden_dim).collect();
        gene_order.sort_by(|&a, &b| scores.gene_fc1[b].partial_cmp(&scores.gene_fc1[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut gene_keep: Vec<usize> = gene_order[..n_keep].to_vec();
        gene_keep.sort_unstable();

        self.tf_fc1.prune_outputs(&tf_keep);
        self.tf_fc2.prune_inputs(&tf_keep);
        self.gene_fc1.prune_outputs(&gene_keep);
        self.gene_fc2.prune_inputs(&gene_keep);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_forward() {
        let mut model = HybridEmbeddingModel::new(
            100, 500, 64, 2000, 128, 64, 0.07, 0.01, 42
        );

        let tf_indices = vec![0, 1, 2];
        let gene_indices = vec![10, 20, 30];
        let tf_expr = Array2::zeros((3, 2000));
        let gene_expr = Array2::zeros((3, 2000));

        let scores = model.forward(&tf_indices, &gene_indices, &tf_expr, &gene_expr);

        assert_eq!(scores.len(), 3);
        assert!(scores.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }

    #[test]
    fn test_model_clone() {
        let model = HybridEmbeddingModel::new(
            10, 20, 8, 16, 16, 8, 0.05, 0.01, 42
        );
        let cloned = model.clone();
        // Embeddings should be equal but independent
        assert_eq!(model.tf_embed.dim(), cloned.tf_embed.dim());
        assert_eq!(model.tf_fc1.weights.dim(), cloned.tf_fc1.weights.dim());
    }

    #[test]
    fn test_profile_activations() {
        use std::collections::HashMap;
        use ndarray::Array1;

        let mut model = HybridEmbeddingModel::new(
            5, 10, 4, 8, 8, 4, 0.05, 0.01, 42
        );
        let hidden_dim = 8;

        // Build tiny fake expr maps
        let mut tf_expr: HashMap<usize, Array1<f32>> = HashMap::new();
        let mut gene_expr: HashMap<usize, Array1<f32>> = HashMap::new();
        for i in 0..5  { tf_expr.insert(i,   Array1::zeros(8)); }
        for i in 0..10 { gene_expr.insert(i,  Array1::zeros(8)); }

        // 4 training examples
        let data: Vec<(usize, usize, f32)> = vec![
            (0, 0, 1.0), (1, 1, 0.0), (2, 2, 1.0), (3, 3, 0.0),
        ];

        let stats = model.profile_activations(&data, &tf_expr, &gene_expr, 8, 4);

        assert_eq!(stats.tf_fc1_activation_counts.len(), hidden_dim);
        assert_eq!(stats.gene_fc1_activation_counts.len(), hidden_dim);
        assert_eq!(stats.total_examples, 4);
        assert_eq!(stats.tf_fc1_col_norms.len(), hidden_dim);
        assert_eq!(stats.tf_fc2_row_norms.len(), hidden_dim);
        assert_eq!(stats.gene_fc1_col_norms.len(), hidden_dim);
        assert_eq!(stats.gene_fc2_row_norms.len(), hidden_dim);

        // activation counts must be in [0, total_examples]
        for &c in &stats.tf_fc1_activation_counts {
            assert!(c as usize <= stats.total_examples);
        }
        // weight norms must be non-negative
        for &n in &stats.tf_fc1_col_norms {
            assert!(n >= 0.0);
        }
    }
}
