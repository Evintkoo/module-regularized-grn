/// Optimized embedding model targeting 95%+ accuracy
/// Incorporates: Deeper architecture, better optimization, focal loss
use crate::models::nn::{LinearLayer, relu, relu_backward, sigmoid};
use ndarray::{Array1, Array2};
use std::f32;

/// Advanced embedding model with all optimizations
pub struct OptimizedEmbeddingModel {
    // Deeper TF encoder (4 layers)
    tf_embed: Array2<f32>,
    tf_embed_grad: Array2<f32>,
    tf_fc1: LinearLayer,
    tf_fc2: LinearLayer,
    tf_fc3: LinearLayer,
    tf_fc4: LinearLayer,
    
    // Deeper Gene encoder (4 layers)
    gene_embed: Array2<f32>,
    gene_embed_grad: Array2<f32>,
    gene_fc1: LinearLayer,
    gene_fc2: LinearLayer,
    gene_fc3: LinearLayer,
    gene_fc4: LinearLayer,
    
    temperature: f32,
    
    // Momentum buffers for better optimization
    tf_embed_momentum: Array2<f32>,
    gene_embed_momentum: Array2<f32>,
    
    // Caches
    tf_indices_cache: Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_embed_out: Option<Array2<f32>>,
    gene_embed_out: Option<Array2<f32>>,
    tf_h1: Option<Array2<f32>>,
    tf_h2: Option<Array2<f32>>,
    tf_h3: Option<Array2<f32>>,
    gene_h1: Option<Array2<f32>>,
    gene_h2: Option<Array2<f32>>,
    gene_h3: Option<Array2<f32>>,
    tf_final: Option<Array2<f32>>,
    gene_final: Option<Array2<f32>>,
}

impl OptimizedEmbeddingModel {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,      // 256
        hidden_dim1: usize,    // 512
        hidden_dim2: usize,    // 256
        hidden_dim3: usize,    // 128
        output_dim: usize,     // 64
        temperature: f32,
        seed: u64,
    ) -> Self {
        use rand::SeedableRng;
        use rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(seed);
        
        // Initialize embeddings with Xavier initialization
        let std = (2.0 / (num_tfs + embed_dim) as f32).sqrt();
        let tf_embed = Array2::random_using((num_tfs, embed_dim), Uniform::new(-std, std), &mut rng);
        
        let std = (2.0 / (num_genes + embed_dim) as f32).sqrt();
        let gene_embed = Array2::random_using((num_genes, embed_dim), Uniform::new(-std, std), &mut rng);
        
        Self {
            tf_embed,
            tf_embed_grad: Array2::zeros((num_tfs, embed_dim)),
            tf_fc1: LinearLayer::new(embed_dim, hidden_dim1, seed),
            tf_fc2: LinearLayer::new(hidden_dim1, hidden_dim2, seed + 1),
            tf_fc3: LinearLayer::new(hidden_dim2, hidden_dim3, seed + 2),
            tf_fc4: LinearLayer::new(hidden_dim3, output_dim, seed + 3),
            
            gene_embed,
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            gene_fc1: LinearLayer::new(embed_dim, hidden_dim1, seed + 4),
            gene_fc2: LinearLayer::new(hidden_dim1, hidden_dim2, seed + 5),
            gene_fc3: LinearLayer::new(hidden_dim2, hidden_dim3, seed + 6),
            gene_fc4: LinearLayer::new(hidden_dim3, output_dim, seed + 7),
            
            temperature,
            
            tf_embed_momentum: Array2::zeros((num_tfs, embed_dim)),
            gene_embed_momentum: Array2::zeros((num_genes, embed_dim)),
            
            tf_indices_cache: None,
            gene_indices_cache: None,
            tf_embed_out: None,
            gene_embed_out: None,
            tf_h1: None,
            tf_h2: None,
            tf_h3: None,
            gene_h1: None,
            gene_h2: None,
            gene_h3: None,
            tf_final: None,
            gene_final: None,
        }
    }
    
    /// Forward pass
    pub fn forward(&mut self, tf_indices: &[usize], gene_indices: &[usize]) -> Array2<f32> {
        let batch_size = tf_indices.len();
        
        // Lookup embeddings
        let mut tf_embedded = Array2::zeros((batch_size, self.tf_embed.ncols()));
        let mut gene_embedded = Array2::zeros((batch_size, self.gene_embed.ncols()));
        
        for (i, &idx) in tf_indices.iter().enumerate() {
            if idx < self.tf_embed.nrows() {
                tf_embedded.row_mut(i).assign(&self.tf_embed.row(idx));
            }
        }
        
        for (i, &idx) in gene_indices.iter().enumerate() {
            if idx < self.gene_embed.nrows() {
                gene_embedded.row_mut(i).assign(&self.gene_embed.row(idx));
            }
        }
        
        // TF encoder (4 layers)
        let tf_h1 = self.tf_fc1.forward(&tf_embedded);
        let tf_h1_relu = relu(&tf_h1);
        
        let tf_h2 = self.tf_fc2.forward(&tf_h1_relu);
        let tf_h2_relu = relu(&tf_h2);
        
        let tf_h3 = self.tf_fc3.forward(&tf_h2_relu);
        let tf_h3_relu = relu(&tf_h3);
        
        let tf_final = self.tf_fc4.forward(&tf_h3_relu);
        
        // Gene encoder (4 layers)
        let gene_h1 = self.gene_fc1.forward(&gene_embedded);
        let gene_h1_relu = relu(&gene_h1);
        
        let gene_h2 = self.gene_fc2.forward(&gene_h1_relu);
        let gene_h2_relu = relu(&gene_h2);
        
        let gene_h3 = self.gene_fc3.forward(&gene_h2_relu);
        let gene_h3_relu = relu(&gene_h3);
        
        let gene_final = self.gene_fc4.forward(&gene_h3_relu);
        
        // Compute scores
        let scores = tf_final.dot(&gene_final.t()) / self.temperature;
        
        // Cache
        self.tf_indices_cache = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());
        self.tf_embed_out = Some(tf_embedded);
        self.gene_embed_out = Some(gene_embedded);
        self.tf_h1 = Some(tf_h1);
        self.tf_h2 = Some(tf_h2);
        self.tf_h3 = Some(tf_h3);
        self.gene_h1 = Some(gene_h1);
        self.gene_h2 = Some(gene_h2);
        self.gene_h3 = Some(gene_h3);
        self.tf_final = Some(tf_final);
        self.gene_final = Some(gene_final);
        
        scores
    }
    
    /// Backward pass
    pub fn backward(&mut self, grad_scores: &Array2<f32>) {
        let grad_scores = grad_scores / self.temperature;
        
        let tf_final = self.tf_final.as_ref().unwrap();
        let gene_final = self.gene_final.as_ref().unwrap();
        
        // Gradient through dot product
        let grad_tf_final = grad_scores.dot(gene_final);
        let grad_gene_final = grad_scores.t().dot(tf_final);
        
        // TF backward (4 layers)
        let grad_tf_h3_relu = self.tf_fc4.backward(&grad_tf_final);
        let tf_h3 = self.tf_h3.as_ref().unwrap();
        let grad_tf_h3 = relu_backward(tf_h3, &grad_tf_h3_relu);
        
        let grad_tf_h2_relu = self.tf_fc3.backward(&grad_tf_h3);
        let tf_h2 = self.tf_h2.as_ref().unwrap();
        let grad_tf_h2 = relu_backward(tf_h2, &grad_tf_h2_relu);
        
        let grad_tf_h1_relu = self.tf_fc2.backward(&grad_tf_h2);
        let tf_h1 = self.tf_h1.as_ref().unwrap();
        let grad_tf_h1 = relu_backward(tf_h1, &grad_tf_h1_relu);
        
        let grad_tf_embedded = self.tf_fc1.backward(&grad_tf_h1);
        
        // Gene backward (4 layers)
        let grad_gene_h3_relu = self.gene_fc4.backward(&grad_gene_final);
        let gene_h3 = self.gene_h3.as_ref().unwrap();
        let grad_gene_h3 = relu_backward(gene_h3, &grad_gene_h3_relu);
        
        let grad_gene_h2_relu = self.gene_fc3.backward(&grad_gene_h3);
        let gene_h2 = self.gene_h2.as_ref().unwrap();
        let grad_gene_h2 = relu_backward(gene_h2, &grad_gene_h2_relu);
        
        let grad_gene_h1_relu = self.gene_fc2.backward(&grad_gene_h2);
        let gene_h1 = self.gene_h1.as_ref().unwrap();
        let grad_gene_h1 = relu_backward(gene_h1, &grad_gene_h1_relu);
        
        let grad_gene_embedded = self.gene_fc1.backward(&grad_gene_h1);
        
        // Accumulate embedding gradients
        let tf_indices = self.tf_indices_cache.as_ref().unwrap();
        let gene_indices = self.gene_indices_cache.as_ref().unwrap();
        
        for (i, &idx) in tf_indices.iter().enumerate() {
            if idx < self.tf_embed_grad.nrows() && i < grad_tf_embedded.nrows() {
                let grad_row = grad_tf_embedded.row(i);
                let mut current_grad = self.tf_embed_grad.row_mut(idx);
                current_grad += &grad_row;
            }
        }
        
        for (i, &idx) in gene_indices.iter().enumerate() {
            if idx < self.gene_embed_grad.nrows() && i < grad_gene_embedded.nrows() {
                let grad_row = grad_gene_embedded.row(i);
                let mut current_grad = self.gene_embed_grad.row_mut(idx);
                current_grad += &grad_row;
            }
        }
    }
    
    /// Update with momentum and weight decay
    pub fn update(&mut self, learning_rate: f32, momentum: f32, weight_decay: f32) {
        // Update embeddings with momentum
        self.tf_embed_momentum = &self.tf_embed_momentum * momentum + &self.tf_embed_grad * (1.0 - momentum);
        self.tf_embed = &self.tf_embed - &self.tf_embed_momentum * learning_rate;
        // Weight decay
        self.tf_embed *= 1.0 - learning_rate * weight_decay;
        
        self.gene_embed_momentum = &self.gene_embed_momentum * momentum + &self.gene_embed_grad * (1.0 - momentum);
        self.gene_embed = &self.gene_embed - &self.gene_embed_momentum * learning_rate;
        self.gene_embed *= 1.0 - learning_rate * weight_decay;
        
        // Update layers
        self.tf_fc1.update(learning_rate);
        self.tf_fc2.update(learning_rate);
        self.tf_fc3.update(learning_rate);
        self.tf_fc4.update(learning_rate);
        self.gene_fc1.update(learning_rate);
        self.gene_fc2.update(learning_rate);
        self.gene_fc3.update(learning_rate);
        self.gene_fc4.update(learning_rate);
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        self.tf_fc1.zero_grad();
        self.tf_fc2.zero_grad();
        self.tf_fc3.zero_grad();
        self.tf_fc4.zero_grad();
        self.gene_fc1.zero_grad();
        self.gene_fc2.zero_grad();
        self.gene_fc3.zero_grad();
        self.gene_fc4.zero_grad();
    }
    
    /// Count parameters
    pub fn count_parameters(&self) -> usize {
        let embed_params = self.tf_embed.len() + self.gene_embed.len();
        let tf_params = self.tf_fc1.weights.len() + self.tf_fc1.bias.len()
                      + self.tf_fc2.weights.len() + self.tf_fc2.bias.len()
                      + self.tf_fc3.weights.len() + self.tf_fc3.bias.len()
                      + self.tf_fc4.weights.len() + self.tf_fc4.bias.len();
        let gene_params = self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
                        + self.gene_fc2.weights.len() + self.gene_fc2.bias.len()
                        + self.gene_fc3.weights.len() + self.gene_fc3.bias.len()
                        + self.gene_fc4.weights.len() + self.gene_fc4.bias.len();
        embed_params + tf_params + gene_params
    }
}

/// Focal loss for handling hard examples
pub fn focal_loss(logits: &Array2<f32>, targets: &Array1<f32>, alpha: f32, gamma: f32) -> f32 {
    let mut total_loss = 0.0;
    let batch_size = logits.nrows();
    
    for i in 0..batch_size {
        if i >= targets.len() {
            continue;
        }
        
        let logit = logits[[i, 0]];
        let prob = sigmoid(&Array2::from_elem((1, 1), logit))[[0, 0]];
        let target = targets[i];
        
        // Focal loss: -α(1-p)^γ log(p) for positive, -α p^γ log(1-p) for negative
        let loss = if target > 0.5 {
            -alpha * (1.0 - prob).powf(gamma) * prob.max(1e-7).ln()
        } else {
            -alpha * prob.powf(gamma) * (1.0 - prob).max(1e-7).ln()
        };
        
        total_loss += loss;
    }
    
    total_loss / batch_size as f32
}

/// Focal loss gradient
pub fn focal_loss_backward(logits: &Array2<f32>, targets: &Array1<f32>, alpha: f32, gamma: f32) -> Array2<f32> {
    let batch_size = logits.nrows();
    let mut grad = Array2::zeros(logits.dim());
    
    for i in 0..batch_size {
        if i >= targets.len() {
            continue;
        }
        
        let logit = logits[[i, 0]];
        let prob = sigmoid(&Array2::from_elem((1, 1), logit))[[0, 0]];
        let target = targets[i];
        
        // Gradient of focal loss
        let grad_val = if target > 0.5 {
            let pt = prob;
            alpha * ((1.0 - pt).powf(gamma - 1.0) * (gamma * pt * pt.ln() + pt - 1.0))
        } else {
            let pt = 1.0 - prob;
            -alpha * ((1.0 - pt).powf(gamma - 1.0) * (gamma * (1.0 - pt) * (1.0 - pt).ln() + (1.0 - pt) - 1.0))
        };
        
        grad[[i, 0]] = grad_val / batch_size as f32;
    }
    
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_embedding_model() {
        let mut model = OptimizedEmbeddingModel::new(
            100, 200, 256, 512, 256, 128, 64, 0.07, 42
        );
        
        let tf_indices = vec![0, 1, 2, 3];
        let gene_indices = vec![10, 11, 12, 13];
        
        let scores = model.forward(&tf_indices, &gene_indices);
        assert_eq!(scores.shape(), &[4, 4]);
        
        let grad = Array2::ones((4, 4));
        model.backward(&grad);
        
        assert!(model.tf_embed_grad.iter().any(|&x| x != 0.0));
    }
}
