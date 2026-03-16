/// Enhanced model with expression features
/// Phase 1: Integrate real biological data
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::Array2;

/// Model that uses expression features
pub struct ExpressionTwoTower {
    // TF encoder (expression → representation)
    tf_fc1: LinearLayer,
    tf_fc2: LinearLayer,
    tf_fc3: LinearLayer,
    
    // Gene encoder (expression → representation)
    gene_fc1: LinearLayer,
    gene_fc2: LinearLayer,
    gene_fc3: LinearLayer,
    
    temperature: f32,
    
    // Caches for backward
    tf_h1_cache: Option<Array2<f32>>,
    tf_h2_cache: Option<Array2<f32>>,
    gene_h1_cache: Option<Array2<f32>>,
    gene_h2_cache: Option<Array2<f32>>,
    tf_final_cache: Option<Array2<f32>>,
    gene_final_cache: Option<Array2<f32>>,
}

impl ExpressionTwoTower {
    pub fn new(
        expression_dim: usize,  // Number of genes in expression
        hidden_dim1: usize,     // 512
        hidden_dim2: usize,     // 256
        output_dim: usize,      // 128
        temperature: f32,
        seed: u64,
    ) -> Self {
        Self {
            // TF encoder: 3 layers (deeper for expression)
            tf_fc1: LinearLayer::new(expression_dim, hidden_dim1, seed),
            tf_fc2: LinearLayer::new(hidden_dim1, hidden_dim2, seed + 1),
            tf_fc3: LinearLayer::new(hidden_dim2, output_dim, seed + 2),
            
            // Gene encoder: 3 layers (symmetric)
            gene_fc1: LinearLayer::new(expression_dim, hidden_dim1, seed + 3),
            gene_fc2: LinearLayer::new(hidden_dim1, hidden_dim2, seed + 4),
            gene_fc3: LinearLayer::new(hidden_dim2, output_dim, seed + 5),
            
            temperature,
            
            tf_h1_cache: None,
            tf_h2_cache: None,
            gene_h1_cache: None,
            gene_h2_cache: None,
            tf_final_cache: None,
            gene_final_cache: None,
        }
    }
    
    /// Forward pass with expression features
    pub fn forward(&mut self, tf_expr: &Array2<f32>, gene_expr: &Array2<f32>) -> Array2<f32> {
        // TF encoder
        let tf_h1 = self.tf_fc1.forward(tf_expr);
        let tf_h1_relu = relu(&tf_h1);
        
        let tf_h2 = self.tf_fc2.forward(&tf_h1_relu);
        let tf_h2_relu = relu(&tf_h2);
        
        let tf_final = self.tf_fc3.forward(&tf_h2_relu);
        
        // Gene encoder
        let gene_h1 = self.gene_fc1.forward(gene_expr);
        let gene_h1_relu = relu(&gene_h1);
        
        let gene_h2 = self.gene_fc2.forward(&gene_h1_relu);
        let gene_h2_relu = relu(&gene_h2);
        
        let gene_final = self.gene_fc3.forward(&gene_h2_relu);
        
        // Compute scores
        let scores = tf_final.dot(&gene_final.t()) / self.temperature;
        
        // Cache
        self.tf_h1_cache = Some(tf_h1);
        self.tf_h2_cache = Some(tf_h2);
        self.gene_h1_cache = Some(gene_h1);
        self.gene_h2_cache = Some(gene_h2);
        self.tf_final_cache = Some(tf_final);
        self.gene_final_cache = Some(gene_final);
        
        scores
    }
    
    /// Backward pass
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
        
        let grad_tf_h1_relu = self.tf_fc2.backward(&grad_tf_h2);
        let tf_h1 = self.tf_h1_cache.as_ref().unwrap();
        let grad_tf_h1 = relu_backward(tf_h1, &grad_tf_h1_relu);
        
        self.tf_fc1.backward(&grad_tf_h1);
        
        // Gene backward
        let grad_gene_h2_relu = self.gene_fc3.backward(&grad_gene_final);
        let gene_h2 = self.gene_h2_cache.as_ref().unwrap();
        let grad_gene_h2 = relu_backward(gene_h2, &grad_gene_h2_relu);
        
        let grad_gene_h1_relu = self.gene_fc2.backward(&grad_gene_h2);
        let gene_h1 = self.gene_h1_cache.as_ref().unwrap();
        let grad_gene_h1 = relu_backward(gene_h1, &grad_gene_h1_relu);
        
        self.gene_fc1.backward(&grad_gene_h1);
    }
    
    /// Update parameters
    pub fn update(&mut self, learning_rate: f32) {
        self.tf_fc1.update(learning_rate);
        self.tf_fc2.update(learning_rate);
        self.tf_fc3.update(learning_rate);
        self.gene_fc1.update(learning_rate);
        self.gene_fc2.update(learning_rate);
        self.gene_fc3.update(learning_rate);
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.tf_fc1.zero_grad();
        self.tf_fc2.zero_grad();
        self.tf_fc3.zero_grad();
        self.gene_fc1.zero_grad();
        self.gene_fc2.zero_grad();
        self.gene_fc3.zero_grad();
    }
    
    /// Count parameters
    pub fn count_parameters(&self) -> usize {
        let tf_params = self.tf_fc1.weights.len() + self.tf_fc1.bias.len()
                      + self.tf_fc2.weights.len() + self.tf_fc2.bias.len()
                      + self.tf_fc3.weights.len() + self.tf_fc3.bias.len();
        let gene_params = self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
                        + self.gene_fc2.weights.len() + self.gene_fc2.bias.len()
                        + self.gene_fc3.weights.len() + self.gene_fc3.bias.len();
        tf_params + gene_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_two_tower() {
        let mut model = ExpressionTwoTower::new(
            2000,  // expression_dim (2000 genes)
            512,   // hidden_dim1
            256,   // hidden_dim2
            128,   // output_dim
            0.07,  // temperature
            42
        );
        
        let batch_size = 4;
        let tf_expr = Array2::ones((batch_size, 2000));
        let gene_expr = Array2::ones((batch_size, 2000));
        
        let scores = model.forward(&tf_expr, &gene_expr);
        assert_eq!(scores.shape(), &[batch_size, batch_size]);
        
        // Test backward
        let grad = Array2::ones((batch_size, batch_size));
        model.backward(&grad);
        
        // Check gradients exist
        assert!(model.tf_fc1.grad_weights.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_parameter_count() {
        let model = ExpressionTwoTower::new(2000, 512, 256, 128, 0.07, 42);
        let params = model.count_parameters();
        
        // TF: (2000*512 + 512) + (512*256 + 256) + (256*128 + 128) = 1,024,512 + 131,328 + 32,896 = 1,188,736
        // Gene: same = 1,188,736
        // Total: 2,377,472
        println!("Parameters: {}", params);
        assert_eq!(params, 2_377_472);
    }
}
