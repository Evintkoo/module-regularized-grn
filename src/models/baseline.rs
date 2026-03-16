use ndarray::{Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::models::nn::{LinearLayer, Dropout, relu, relu_backward};

/// Baseline model (Cross-encoder) with backpropagation
pub struct BaselineModel {
    fc1: LinearLayer,
    fc2: LinearLayer,
    fc3: LinearLayer,
    dropout1: Dropout,
    dropout2: Dropout,
    
    // Cache for backward pass
    h1_cache: Option<Array2<f32>>,
    h1_relu_cache: Option<Array2<f32>>,
    h1_drop_cache: Option<Array2<f32>>,
    h2_cache: Option<Array2<f32>>,
    h2_relu_cache: Option<Array2<f32>>,
    h2_drop_cache: Option<Array2<f32>>,
}

impl BaselineModel {
    pub fn new(
        input_dim: usize,
        hidden_dim1: usize,
        hidden_dim2: usize,
        output_dim: usize,
        dropout_prob: f32,
        seed: u64,
    ) -> Self {
        Self {
            fc1: LinearLayer::new(input_dim, hidden_dim1, seed),
            fc2: LinearLayer::new(hidden_dim1, hidden_dim2, seed + 1),
            fc3: LinearLayer::new(hidden_dim2, output_dim, seed + 2),
            dropout1: Dropout::new(dropout_prob),
            dropout2: Dropout::new(dropout_prob),
            h1_cache: None,
            h1_relu_cache: None,
            h1_drop_cache: None,
            h2_cache: None,
            h2_relu_cache: None,
            h2_drop_cache: None,
        }
    }

    /// Create parameter-matched baseline
    pub fn from_two_tower(
        tf_input_dim: usize,
        gene_input_dim: usize,
        hidden_dim: usize,
        embed_dim: usize,
        dropout_prob: f32,
        seed: u64,
    ) -> Self {
        let input_dim = tf_input_dim + gene_input_dim;
        let hidden_dim1 = hidden_dim * 2;
        let hidden_dim2 = embed_dim;
        let output_dim = 1;

        Self::new(input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_prob, seed)
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        // Layer 1
        let h1 = self.fc1.forward(x);
        let h1_relu = relu(&h1);
        let h1_drop = self.dropout1.forward(&h1_relu, training, rng);

        // Layer 2
        let h2 = self.fc2.forward(&h1_drop);
        let h2_relu = relu(&h2);
        let h2_drop = self.dropout2.forward(&h2_relu, training, rng);

        // Layer 3
        let h3 = self.fc3.forward(&h2_drop);
        
        // Cache
        self.h1_cache = Some(h1);
        self.h1_relu_cache = Some(h1_relu);
        self.h1_drop_cache = Some(h1_drop);
        self.h2_cache = Some(h2);
        self.h2_relu_cache = Some(h2_relu);
        self.h2_drop_cache = Some(h2_drop);
        
        h3
    }
    
    /// Backward pass
    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        // Backward through fc3
        let grad_h2_drop = self.fc3.backward(grad_output);
        
        // Backward through dropout2
        let grad_h2_relu = self.dropout2.backward(&grad_h2_drop);
        
        // Backward through relu2
        let h2 = self.h2_cache.as_ref().expect("Forward must be called first");
        let grad_h2 = relu_backward(h2, &grad_h2_relu);
        
        // Backward through fc2
        let grad_h1_drop = self.fc2.backward(&grad_h2);
        
        // Backward through dropout1
        let grad_h1_relu = self.dropout1.backward(&grad_h1_drop);
        
        // Backward through relu1
        let h1 = self.h1_cache.as_ref().expect("Forward must be called first");
        let grad_h1 = relu_backward(h1, &grad_h1_relu);
        
        // Backward through fc1
        let grad_x = self.fc1.backward(&grad_h1);
        
        grad_x
    }
    
    /// Update all parameters
    pub fn update(&mut self, learning_rate: f32) {
        self.fc1.update(learning_rate);
        self.fc2.update(learning_rate);
        self.fc3.update(learning_rate);
    }
    
    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
        self.fc3.zero_grad();
    }

    /// Count parameters
    pub fn count_params(&self) -> usize {
        let fc1_params = self.fc1.weights.len() + self.fc1.bias.len();
        let fc2_params = self.fc2.weights.len() + self.fc2.bias.len();
        let fc3_params = self.fc3.weights.len() + self.fc3.bias.len();
        fc1_params + fc2_params + fc3_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_forward() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = BaselineModel::new(256, 512, 128, 1, 0.1, 42);

        let input = Array2::zeros((32, 256));
        let scores = model.forward(&input, false, &mut rng);

        assert_eq!(scores.shape(), &[32, 1]);
    }

    #[test]
    fn test_parameter_matching() {
        let model = BaselineModel::from_two_tower(128, 128, 256, 128, 0.1, 42);
        let params = model.count_params();

        println!("Baseline params: {}", params);
        // Should be ~132K parameters
        assert!(params > 100_000 && params < 200_000);
    }
    
    #[test]
    fn test_baseline_backward() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = BaselineModel::new(256, 512, 128, 1, 0.1, 42);

        let input = Array2::ones((4, 256));
        let scores = model.forward(&input, true, &mut rng);
        
        // Backward with dummy gradient
        let grad_scores = Array2::ones((4, 1));
        model.backward(&grad_scores);
        
        // Check gradients exist
        assert!(model.fc1.grad_weights.iter().any(|&x| x != 0.0));
    }
}


