use ndarray::{Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use crate::models::nn::{LinearLayer, Dropout, relu, relu_backward};

/// TF Encoder with backpropagation
pub struct TFEncoder {
    fc1: LinearLayer,
    fc2: LinearLayer,
    dropout: Dropout,
    
    // Cache for backward pass
    x1_cache: Option<Array2<f32>>,
    h1_cache: Option<Array2<f32>>,
    h1_relu_cache: Option<Array2<f32>>,
    h1_drop_cache: Option<Array2<f32>>,
}

impl TFEncoder {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, dropout_prob: f32, seed: u64) -> Self {
        Self {
            fc1: LinearLayer::new(input_dim, hidden_dim, seed),
            fc2: LinearLayer::new(hidden_dim, output_dim, seed + 1),
            dropout: Dropout::new(dropout_prob),
            x1_cache: None,
            h1_cache: None,
            h1_relu_cache: None,
            h1_drop_cache: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        // Layer 1
        let h1 = self.fc1.forward(x);
        let h1_relu = relu(&h1);
        let h1_drop = self.dropout.forward(&h1_relu, training, rng);
        
        // Layer 2
        let h2 = self.fc2.forward(&h1_drop);
        
        // Cache for backward
        self.x1_cache = Some(x.clone());
        self.h1_cache = Some(h1);
        self.h1_relu_cache = Some(h1_relu);
        self.h1_drop_cache = Some(h1_drop);
        
        h2
    }
    
    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        // Backward through fc2
        let grad_h1_drop = self.fc2.backward(grad_output);
        
        // Backward through dropout
        let grad_h1_relu = self.dropout.backward(&grad_h1_drop);
        
        // Backward through ReLU
        let h1 = self.h1_cache.as_ref().expect("Forward must be called first");
        let grad_h1 = relu_backward(h1, &grad_h1_relu);
        
        // Backward through fc1
        let grad_x = self.fc1.backward(&grad_h1);
        
        grad_x
    }
    
    pub fn update(&mut self, learning_rate: f32) {
        self.fc1.update(learning_rate);
        self.fc2.update(learning_rate);
    }
    
    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }
}

/// Gene Encoder with backpropagation
pub struct GeneEncoder {
    fc1: LinearLayer,
    fc2: LinearLayer,
    dropout: Dropout,
    
    // Cache for backward pass
    x1_cache: Option<Array2<f32>>,
    h1_cache: Option<Array2<f32>>,
    h1_relu_cache: Option<Array2<f32>>,
    h1_drop_cache: Option<Array2<f32>>,
}

impl GeneEncoder {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, dropout_prob: f32, seed: u64) -> Self {
        Self {
            fc1: LinearLayer::new(input_dim, hidden_dim, seed),
            fc2: LinearLayer::new(hidden_dim, output_dim, seed + 1),
            dropout: Dropout::new(dropout_prob),
            x1_cache: None,
            h1_cache: None,
            h1_relu_cache: None,
            h1_drop_cache: None,
        }
    }

    pub fn forward(&mut self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        // Layer 1
        let h1 = self.fc1.forward(x);
        let h1_relu = relu(&h1);
        let h1_drop = self.dropout.forward(&h1_relu, training, rng);
        
        // Layer 2
        let h2 = self.fc2.forward(&h1_drop);
        
        // Cache for backward
        self.x1_cache = Some(x.clone());
        self.h1_cache = Some(h1);
        self.h1_relu_cache = Some(h1_relu);
        self.h1_drop_cache = Some(h1_drop);
        
        h2
    }
    
    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        // Backward through fc2
        let grad_h1_drop = self.fc2.backward(grad_output);
        
        // Backward through dropout
        let grad_h1_relu = self.dropout.backward(&grad_h1_drop);
        
        // Backward through ReLU
        let h1 = self.h1_cache.as_ref().expect("Forward must be called first");
        let grad_h1 = relu_backward(h1, &grad_h1_relu);
        
        // Backward through fc1
        let grad_x = self.fc1.backward(&grad_h1);
        
        grad_x
    }
    
    pub fn update(&mut self, learning_rate: f32) {
        self.fc1.update(learning_rate);
        self.fc2.update(learning_rate);
    }
    
    pub fn zero_grad(&mut self) {
        self.fc1.zero_grad();
        self.fc2.zero_grad();
    }
}

/// Two-Tower Model with backpropagation
pub struct TwoTowerModel {
    tf_encoder: TFEncoder,
    gene_encoder: GeneEncoder,
    temperature: f32,
    
    // Cache for backward pass
    tf_embed_cache: Option<Array2<f32>>,
    gene_embed_cache: Option<Array2<f32>>,
    scores_cache: Option<Array2<f32>>,
}

impl TwoTowerModel {
    pub fn new(
        tf_input_dim: usize,
        gene_input_dim: usize,
        hidden_dim: usize,
        embed_dim: usize,
        dropout_prob: f32,
        temperature: f32,
        seed: u64,
    ) -> Self {
        Self {
            tf_encoder: TFEncoder::new(tf_input_dim, hidden_dim, embed_dim, dropout_prob, seed),
            gene_encoder: GeneEncoder::new(gene_input_dim, hidden_dim, embed_dim, dropout_prob, seed + 100),
            temperature,
            tf_embed_cache: None,
            gene_embed_cache: None,
            scores_cache: None,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, tf_input: &Array2<f32>, gene_input: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        // Encode
        let tf_embed = self.tf_encoder.forward(tf_input, training, rng);
        let gene_embed = self.gene_encoder.forward(gene_input, training, rng);
        
        // Compute similarity scores via dot product
        // [batch, embed] @ [embed, batch] = [batch, batch]
        let scores = tf_embed.dot(&gene_embed.t()) / self.temperature;
        
        // Cache for backward
        self.tf_embed_cache = Some(tf_embed);
        self.gene_embed_cache = Some(gene_embed);
        self.scores_cache = Some(scores.clone());
        
        scores
    }
    
    /// Backward pass
    pub fn backward(&mut self, grad_scores: &Array2<f32>) {
        // Scale gradient by temperature
        let grad_scores = grad_scores / self.temperature;
        
        let tf_embed = self.tf_embed_cache.as_ref().expect("Forward must be called first");
        let gene_embed = self.gene_embed_cache.as_ref().expect("Forward must be called first");
        
        // Gradient of dot product:
        // scores = tf_embed @ gene_embed^T
        // grad_tf_embed = grad_scores @ gene_embed
        // grad_gene_embed = grad_scores^T @ tf_embed
        
        let grad_tf_embed = grad_scores.dot(gene_embed);
        let grad_gene_embed = grad_scores.t().dot(tf_embed);
        
        // Backward through encoders
        self.tf_encoder.backward(&grad_tf_embed);
        self.gene_encoder.backward(&grad_gene_embed);
    }
    
    /// Update all parameters
    pub fn update(&mut self, learning_rate: f32) {
        self.tf_encoder.update(learning_rate);
        self.gene_encoder.update(learning_rate);
    }
    
    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.tf_encoder.zero_grad();
        self.gene_encoder.zero_grad();
    }
    
    /// Count total parameters
    pub fn count_parameters(&self) -> usize {
        let tf_params = self.tf_encoder.fc1.weights.len() + self.tf_encoder.fc1.bias.len()
                      + self.tf_encoder.fc2.weights.len() + self.tf_encoder.fc2.bias.len();
        let gene_params = self.gene_encoder.fc1.weights.len() + self.gene_encoder.fc1.bias.len()
                        + self.gene_encoder.fc2.weights.len() + self.gene_encoder.fc2.bias.len();
        tf_params + gene_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tf_encoder_forward() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut encoder = TFEncoder::new(128, 256, 128, 0.1, 42);
        let x = Array2::zeros((32, 128));
        let output = encoder.forward(&x, false, &mut rng);
        assert_eq!(output.shape(), &[32, 128]);
    }

    #[test]
    fn test_two_tower_forward() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = TwoTowerModel::new(128, 128, 256, 128, 0.1, 0.07, 42);

        let tf_input = Array2::zeros((32, 128));
        let gene_input = Array2::zeros((32, 128));

        let scores = model.forward(&tf_input, &gene_input, false, &mut rng);
        assert_eq!(scores.shape(), &[32, 32]);
    }
    
    #[test]
    fn test_two_tower_backward() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut model = TwoTowerModel::new(128, 128, 256, 128, 0.1, 0.07, 42);

        let tf_input = Array2::ones((4, 128));
        let gene_input = Array2::ones((4, 128));

        // Forward
        let scores = model.forward(&tf_input, &gene_input, true, &mut rng);
        
        // Backward with dummy gradient
        let grad_scores = Array2::ones((4, 4));
        model.backward(&grad_scores);
        
        // Check gradients exist (non-zero)
        assert!(model.tf_encoder.fc1.grad_weights.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_parameter_count() {
        let model = TwoTowerModel::new(128, 128, 256, 128, 0.1, 0.07, 42);
        let param_count = model.count_parameters();
        
        // TF: (128*256 + 256) + (256*128 + 128) = 33024 + 32896 = 65920
        // Gene: same = 65920
        // Total: 131840
        assert_eq!(param_count, 131840);
    }
}


