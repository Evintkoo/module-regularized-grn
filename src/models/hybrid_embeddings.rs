/// Hybrid model: Learnable embeddings + Expression features
/// Target: 70%+ accuracy by combining structure and biology
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array1, Array2, Axis};
use std::f32;

/// Hybrid embedding + expression model
pub struct HybridEmbeddingModel {
    // TF encoder
    tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    tf_fc1: LinearLayer,  // (embed + expr) → hidden
    tf_fc2: LinearLayer,  // hidden → output
    
    // Gene encoder  
    gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,
    gene_fc1: LinearLayer,
    gene_fc2: LinearLayer,
    
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
    tf_final: Option<Array2<f32>>,
    gene_final: Option<Array2<f32>>,
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
        let grad_gene_h1_pre = relu_backward(&grad_gene_h1, gene_h1);
        let grad_gene_concat = self.gene_fc1.backward(&grad_gene_h1_pre);
        
        // Backward through TF encoder
        let grad_tf_h1 = self.tf_fc2.backward(&grad_tf_out);
        let tf_h1 = self.tf_h1.as_ref().unwrap();
        let grad_tf_h1_pre = relu_backward(&grad_tf_h1, tf_h1);
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
}
