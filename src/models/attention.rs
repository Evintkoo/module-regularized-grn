/// Self-attention mechanism for gene regulatory networks
use crate::models::nn::LinearLayer;
use ndarray::{Array2, Axis};

pub struct SelfAttention {
    query: LinearLayer,
    key: LinearLayer,
    value: LinearLayer,
    output: LinearLayer,
    num_heads: usize,
    head_dim: usize,
    
    // Caches for backward
    attention_weights_cache: Option<Array2<f32>>,
    v_cache: Option<Array2<f32>>,
    q_cache: Option<Array2<f32>>,
    k_cache: Option<Array2<f32>>,
    input_cache: Option<Array2<f32>>,
}

impl SelfAttention {
    pub fn new(input_dim: usize, num_heads: usize, seed: u64) -> Self {
        assert!(input_dim % num_heads == 0, "input_dim must be divisible by num_heads");
        let head_dim = input_dim / num_heads;
        
        Self {
            query: LinearLayer::new(input_dim, input_dim, seed),
            key: LinearLayer::new(input_dim, input_dim, seed + 1),
            value: LinearLayer::new(input_dim, input_dim, seed + 2),
            output: LinearLayer::new(input_dim, input_dim, seed + 3),
            num_heads,
            head_dim,
            attention_weights_cache: None,
            v_cache: None,
            q_cache: None,
            k_cache: None,
            input_cache: None,
        }
    }
    
    /// Forward pass
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[0]; // Self-attention within batch
        
        // Linear projections
        let q = self.query.forward(x); // (batch, input_dim)
        let k = self.key.forward(x);
        let v = self.value.forward(x);
        
        // Reshape for multi-head attention
        // (batch, input_dim) -> (batch, num_heads, head_dim)
        let q_heads = self.reshape_for_heads(&q, batch_size);
        let k_heads = self.reshape_for_heads(&k, batch_size);
        let v_heads = self.reshape_for_heads(&v, batch_size);
        
        // Scaled dot-product attention for each head
        let scale = (self.head_dim as f32).sqrt();
        
        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let scores = q_heads.dot(&k_heads.t()) / scale; // (batch, batch)
        
        // Softmax over last dimension
        let attention_weights = softmax(&scores);
        
        // Apply attention to values
        let attended = attention_weights.dot(&v_heads); // (batch, input_dim)
        
        // Output projection
        let output = self.output.forward(&attended);
        
        // Cache for backward
        self.attention_weights_cache = Some(attention_weights);
        self.v_cache = Some(v_heads);
        self.q_cache = Some(q);
        self.k_cache = Some(k);
        self.input_cache = Some(x.clone());
        
        output
    }
    
    /// Backward pass
    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        // Gradient through output projection
        let grad_attended = self.output.backward(grad_output);
        
        let attention_weights = self.attention_weights_cache.as_ref().unwrap();
        let v = self.v_cache.as_ref().unwrap();
        
        // Gradient through attention application: attended = attention_weights @ v
        // grad_attended is (batch, input_dim), attention_weights is (batch, batch), v is (batch, input_dim)
        let grad_attention_weights = grad_attended.dot(&v.t()); // (batch, batch)
        let grad_v = attention_weights.t().dot(&grad_attended); // (batch, input_dim)
        
        // Gradient through softmax and scaling
        let grad_scores = softmax_backward(attention_weights, &grad_attention_weights);
        let scale = (self.head_dim as f32).sqrt();
        let grad_scores = grad_scores / scale;
        
        // Gradient through Q @ K^T: scores = q @ k^T
        // grad_scores is (batch, batch), q is (batch, input_dim), k is (batch, input_dim)
        let k = self.k_cache.as_ref().unwrap();
        let q = self.q_cache.as_ref().unwrap();
        let grad_q = grad_scores.dot(k); // (batch, input_dim)
        let grad_k = grad_scores.t().dot(q); // (batch, input_dim)
        
        // Gradient through linear projections
        let grad_x_q = self.query.backward(&grad_q);
        let grad_x_k = self.key.backward(&grad_k);
        let grad_x_v = self.value.backward(&grad_v);
        
        // Sum gradients
        grad_x_q + grad_x_k + grad_x_v
    }
    
    /// Update parameters
    pub fn update(&mut self, learning_rate: f32) {
        self.query.update(learning_rate);
        self.key.update(learning_rate);
        self.value.update(learning_rate);
        self.output.update(learning_rate);
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.query.zero_grad();
        self.key.zero_grad();
        self.value.zero_grad();
        self.output.zero_grad();
    }
    
    /// Count parameters
    pub fn count_parameters(&self) -> usize {
        self.query.weights.len() + self.query.bias.len()
            + self.key.weights.len() + self.key.bias.len()
            + self.value.weights.len() + self.value.bias.len()
            + self.output.weights.len() + self.output.bias.len()
    }
    
    fn reshape_for_heads(&self, x: &Array2<f32>, batch_size: usize) -> Array2<f32> {
        // For simplicity, we'll treat as single-head attention in this implementation
        x.clone()
    }
}

/// Softmax function
fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();
    for mut row in result.axis_iter_mut(Axis(0)) {
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for val in row.iter_mut() {
            *val = (*val - max).exp();
        }
        let sum: f32 = row.iter().sum();
        for val in row.iter_mut() {
            *val /= sum + 1e-8;
        }
    }
    result
}

/// Softmax backward
fn softmax_backward(softmax_output: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    let mut grad_input = Array2::zeros(grad_output.dim());
    
    for i in 0..softmax_output.shape()[0] {
        let s = softmax_output.row(i);
        let g = grad_output.row(i);
        
        for j in 0..s.len() {
            let mut grad_sum = 0.0;
            for k in 0..s.len() {
                if j == k {
                    grad_sum += g[k] * s[j] * (1.0 - s[j]);
                } else {
                    grad_sum += g[k] * (-s[j] * s[k]);
                }
            }
            grad_input[[i, j]] = grad_sum;
        }
    }
    
    grad_input
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_attention_forward() {
        let mut attention = SelfAttention::new(128, 4, 42);
        let batch_size = 8;
        let input = Array2::ones((batch_size, 128));
        
        let output = attention.forward(&input);
        assert_eq!(output.shape(), &[batch_size, 128]);
    }
    
    #[test]
    fn test_attention_backward() {
        let mut attention = SelfAttention::new(128, 4, 42);
        let batch_size = 8;
        let input = Array2::ones((batch_size, 128));
        
        let output = attention.forward(&input);
        let grad = Array2::ones((batch_size, 128));
        let grad_input = attention.backward(&grad);
        
        assert_eq!(grad_input.shape(), input.shape());
    }
    
    #[test]
    fn test_softmax() {
        let x = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = softmax(&x);
        
        // Check each row sums to ~1
        for row in result.axis_iter(Axis(0)) {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }
}
