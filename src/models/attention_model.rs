/// Enhanced expression model with self-attention
use crate::models::nn::{LinearLayer, relu, relu_backward};
use crate::models::attention::SelfAttention;
use ndarray::Array2;

/// Model with attention mechanism for better feature interaction
pub struct AttentionExpressionModel {
    // TF encoder with attention
    tf_fc1: LinearLayer,
    tf_attention: SelfAttention,
    tf_fc2: LinearLayer,
    tf_fc3: LinearLayer,
    
    // Gene encoder with attention
    gene_fc1: LinearLayer,
    gene_attention: SelfAttention,
    gene_fc2: LinearLayer,
    gene_fc3: LinearLayer,
    
    temperature: f32,
    
    // Caches
    tf_h1_cache: Option<Array2<f32>>,
    tf_attn_cache: Option<Array2<f32>>,
    tf_h2_cache: Option<Array2<f32>>,
    gene_h1_cache: Option<Array2<f32>>,
    gene_attn_cache: Option<Array2<f32>>,
    gene_h2_cache: Option<Array2<f32>>,
    tf_final_cache: Option<Array2<f32>>,
    gene_final_cache: Option<Array2<f32>>,
}

impl AttentionExpressionModel {
    pub fn new(
        expression_dim: usize,
        hidden_dim1: usize,
        hidden_dim2: usize,
        output_dim: usize,
        num_heads: usize,
        temperature: f32,
        seed: u64,
    ) -> Self {
        Self {
            // TF encoder
            tf_fc1: LinearLayer::new(expression_dim, hidden_dim1, seed),
            tf_attention: SelfAttention::new(hidden_dim1, num_heads, seed + 1),
            tf_fc2: LinearLayer::new(hidden_dim1, hidden_dim2, seed + 2),
            tf_fc3: LinearLayer::new(hidden_dim2, output_dim, seed + 3),
            
            // Gene encoder
            gene_fc1: LinearLayer::new(expression_dim, hidden_dim1, seed + 4),
            gene_attention: SelfAttention::new(hidden_dim1, num_heads, seed + 5),
            gene_fc2: LinearLayer::new(hidden_dim1, hidden_dim2, seed + 6),
            gene_fc3: LinearLayer::new(hidden_dim2, output_dim, seed + 7),
            
            temperature,
            
            tf_h1_cache: None,
            tf_attn_cache: None,
            tf_h2_cache: None,
            gene_h1_cache: None,
            gene_attn_cache: None,
            gene_h2_cache: None,
            tf_final_cache: None,
            gene_final_cache: None,
        }
    }
    
    pub fn forward(&mut self, tf_expr: &Array2<f32>, gene_expr: &Array2<f32>) -> Array2<f32> {
        // TF encoder with attention
        let tf_h1 = self.tf_fc1.forward(tf_expr);
        let tf_h1_relu = relu(&tf_h1);
        
        // Apply self-attention
        let tf_attn = self.tf_attention.forward(&tf_h1_relu);
        let tf_attn_relu = relu(&tf_attn);
        
        let tf_h2 = self.tf_fc2.forward(&tf_attn_relu);
        let tf_h2_relu = relu(&tf_h2);
        
        let tf_final = self.tf_fc3.forward(&tf_h2_relu);
        
        // Gene encoder with attention
        let gene_h1 = self.gene_fc1.forward(gene_expr);
        let gene_h1_relu = relu(&gene_h1);
        
        // Apply self-attention
        let gene_attn = self.gene_attention.forward(&gene_h1_relu);
        let gene_attn_relu = relu(&gene_attn);
        
        let gene_h2 = self.gene_fc2.forward(&gene_attn_relu);
        let gene_h2_relu = relu(&gene_h2);
        
        let gene_final = self.gene_fc3.forward(&gene_h2_relu);
        
        // Compute similarity scores
        let scores = tf_final.dot(&gene_final.t()) / self.temperature;
        
        // Cache for backward
        self.tf_h1_cache = Some(tf_h1);
        self.tf_attn_cache = Some(tf_attn);
        self.tf_h2_cache = Some(tf_h2);
        self.gene_h1_cache = Some(gene_h1);
        self.gene_attn_cache = Some(gene_attn);
        self.gene_h2_cache = Some(gene_h2);
        self.tf_final_cache = Some(tf_final);
        self.gene_final_cache = Some(gene_final);
        
        scores
    }
    
    pub fn backward(&mut self, grad_scores: &Array2<f32>) {
        let grad_scores = grad_scores / self.temperature;
        
        let tf_final = self.tf_final_cache.as_ref().unwrap();
        let gene_final = self.gene_final_cache.as_ref().unwrap();
        
        // Gradient through dot product
        let grad_tf_final = grad_scores.dot(gene_final);
        let grad_gene_final = grad_scores.t().dot(tf_final);
        
        // TF backward
        let grad_tf_h2_relu = self.tf_fc3.backward(&grad_tf_final);
        let tf_h2 = self.tf_h2_cache.as_ref().unwrap();
        let grad_tf_h2 = relu_backward(tf_h2, &grad_tf_h2_relu);
        
        let grad_tf_attn_relu = self.tf_fc2.backward(&grad_tf_h2);
        let tf_attn = self.tf_attn_cache.as_ref().unwrap();
        let grad_tf_attn = relu_backward(tf_attn, &grad_tf_attn_relu);
        
        let grad_tf_h1_relu = self.tf_attention.backward(&grad_tf_attn);
        let tf_h1 = self.tf_h1_cache.as_ref().unwrap();
        let grad_tf_h1 = relu_backward(tf_h1, &grad_tf_h1_relu);
        
        self.tf_fc1.backward(&grad_tf_h1);
        
        // Gene backward
        let grad_gene_h2_relu = self.gene_fc3.backward(&grad_gene_final);
        let gene_h2 = self.gene_h2_cache.as_ref().unwrap();
        let grad_gene_h2 = relu_backward(gene_h2, &grad_gene_h2_relu);
        
        let grad_gene_attn_relu = self.gene_fc2.backward(&grad_gene_h2);
        let gene_attn = self.gene_attn_cache.as_ref().unwrap();
        let grad_gene_attn = relu_backward(gene_attn, &grad_gene_attn_relu);
        
        let grad_gene_h1_relu = self.gene_attention.backward(&grad_gene_attn);
        let gene_h1 = self.gene_h1_cache.as_ref().unwrap();
        let grad_gene_h1 = relu_backward(gene_h1, &grad_gene_h1_relu);
        
        self.gene_fc1.backward(&grad_gene_h1);
    }
    
    pub fn update(&mut self, learning_rate: f32) {
        self.tf_fc1.update(learning_rate);
        self.tf_attention.update(learning_rate);
        self.tf_fc2.update(learning_rate);
        self.tf_fc3.update(learning_rate);
        
        self.gene_fc1.update(learning_rate);
        self.gene_attention.update(learning_rate);
        self.gene_fc2.update(learning_rate);
        self.gene_fc3.update(learning_rate);
    }
    
    pub fn zero_grad(&mut self) {
        self.tf_fc1.zero_grad();
        self.tf_attention.zero_grad();
        self.tf_fc2.zero_grad();
        self.tf_fc3.zero_grad();
        
        self.gene_fc1.zero_grad();
        self.gene_attention.zero_grad();
        self.gene_fc2.zero_grad();
        self.gene_fc3.zero_grad();
    }
    
    pub fn count_parameters(&self) -> usize {
        let tf_params = self.tf_fc1.weights.len() + self.tf_fc1.bias.len()
                      + self.tf_attention.count_parameters()
                      + self.tf_fc2.weights.len() + self.tf_fc2.bias.len()
                      + self.tf_fc3.weights.len() + self.tf_fc3.bias.len();
        
        let gene_params = self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
                        + self.gene_attention.count_parameters()
                        + self.gene_fc2.weights.len() + self.gene_fc2.bias.len()
                        + self.gene_fc3.weights.len() + self.gene_fc3.bias.len();
        
        tf_params + gene_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_model_forward() {
        let mut model = AttentionExpressionModel::new(
            2000,  // expression_dim
            512,   // hidden_dim1
            256,   // hidden_dim2
            128,   // output_dim
            4,     // num_heads
            0.07,  // temperature
            42
        );
        
        let batch_size = 8;
        let tf_expr = Array2::ones((batch_size, 2000));
        let gene_expr = Array2::ones((batch_size, 2000));
        
        let scores = model.forward(&tf_expr, &gene_expr);
        assert_eq!(scores.shape(), &[batch_size, batch_size]);
    }
    
    #[test]
    fn test_attention_model_backward() {
        let mut model = AttentionExpressionModel::new(
            2000, 512, 256, 128, 4, 0.07, 42
        );
        
        let batch_size = 8;
        let tf_expr = Array2::ones((batch_size, 2000));
        let gene_expr = Array2::ones((batch_size, 2000));
        
        let scores = model.forward(&tf_expr, &gene_expr);
        let grad = Array2::ones((batch_size, batch_size));
        model.backward(&grad);
        
        // Verify gradients exist
        assert!(model.tf_fc1.grad_weights.iter().any(|&x| x != 0.0));
    }
}
