/// Learnable embedding layer with backpropagation
use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Embedding layer with gradient support
#[derive(Debug, Clone)]
pub struct LearnableEmbedding {
    pub weights: Array2<f32>, // [vocab_size, embed_dim]
    pub grad_weights: Array2<f32>,
    
    // Cache for backward pass
    indices_cache: Option<Vec<usize>>,
}

impl LearnableEmbedding {
    /// Create new embedding with random initialization
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (1.0 / embed_dim as f32).sqrt();
        let weights = Array2::random_using(
            (vocab_size, embed_dim),
            Uniform::new(-std, std),
            &mut rng,
        );
        let grad_weights = Array2::zeros((vocab_size, embed_dim));
        
        Self {
            weights,
            grad_weights,
            indices_cache: None,
        }
    }

    /// Forward pass: lookup embeddings by indices
    pub fn forward(&mut self, indices: &[usize]) -> Array2<f32> {
        let batch_size = indices.len();
        let embed_dim = self.weights.ncols();
        let mut output = Array2::zeros((batch_size, embed_dim));

        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.weights.nrows() {
                output.row_mut(i).assign(&self.weights.row(idx));
            }
        }
        
        // Cache indices for backward pass
        self.indices_cache = Some(indices.to_vec());
        
        output
    }
    
    /// Backward pass: accumulate gradients for the indices used
    pub fn backward(&mut self, grad_output: &Array2<f32>) {
        let indices = self.indices_cache.as_ref().expect("Forward must be called first");
        
        // Accumulate gradients for each index
        for (i, &idx) in indices.iter().enumerate() {
            if idx < self.grad_weights.nrows() && i < grad_output.nrows() {
                let grad_row = grad_output.row(i);
                let mut current_grad = self.grad_weights.row_mut(idx);
                current_grad += &grad_row;
            }
        }
    }
    
    /// Update weights
    pub fn update(&mut self, learning_rate: f32) {
        self.weights = &self.weights - &self.grad_weights * learning_rate;
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
    }
    
    /// Count parameters
    pub fn num_parameters(&self) -> usize {
        self.weights.len()
    }
}

/// Two-Tower model with learnable embeddings
pub struct EmbeddingTwoTower {
    tf_embed: LearnableEmbedding,
    gene_embed: LearnableEmbedding,
    
    // MLP layers (using existing LinearLayer)
    tf_fc1: crate::models::nn::LinearLayer,
    tf_fc2: crate::models::nn::LinearLayer,
    gene_fc1: crate::models::nn::LinearLayer,
    gene_fc2: crate::models::nn::LinearLayer,
    
    temperature: f32,
    
    // Caches
    tf_embed_cache: Option<Array2<f32>>,
    gene_embed_cache: Option<Array2<f32>>,
    tf_h1_cache: Option<Array2<f32>>,
    gene_h1_cache: Option<Array2<f32>>,
    tf_final_cache: Option<Array2<f32>>,
    gene_final_cache: Option<Array2<f32>>,
}

impl EmbeddingTwoTower {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        embed_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        dropout_prob: f32,
        temperature: f32,
        seed: u64,
    ) -> Self {
        use crate::models::nn::LinearLayer;
        
        Self {
            tf_embed: LearnableEmbedding::new(num_tfs, embed_dim, seed),
            gene_embed: LearnableEmbedding::new(num_genes, embed_dim, seed + 1),
            
            tf_fc1: LinearLayer::new(embed_dim, hidden_dim, seed + 2),
            tf_fc2: LinearLayer::new(hidden_dim, output_dim, seed + 3),
            gene_fc1: LinearLayer::new(embed_dim, hidden_dim, seed + 4),
            gene_fc2: LinearLayer::new(hidden_dim, output_dim, seed + 5),
            
            temperature,
            
            tf_embed_cache: None,
            gene_embed_cache: None,
            tf_h1_cache: None,
            gene_h1_cache: None,
            tf_final_cache: None,
            gene_final_cache: None,
        }
    }
    
    /// Forward pass with TF and Gene indices
    pub fn forward(&mut self, tf_indices: &[usize], gene_indices: &[usize]) -> Array2<f32> {
        use crate::models::nn::relu;
        
        // Embed TFs and Genes
        let tf_embedded = self.tf_embed.forward(tf_indices);
        let gene_embedded = self.gene_embed.forward(gene_indices);
        
        // TF encoder
        let tf_h1 = self.tf_fc1.forward(&tf_embedded);
        let tf_h1_relu = relu(&tf_h1);
        let tf_final = self.tf_fc2.forward(&tf_h1_relu);
        
        // Gene encoder
        let gene_h1 = self.gene_fc1.forward(&gene_embedded);
        let gene_h1_relu = relu(&gene_h1);
        let gene_final = self.gene_fc2.forward(&gene_h1_relu);
        
        // Compute scores (dot product)
        let scores = tf_final.dot(&gene_final.t()) / self.temperature;
        
        // Cache for backward
        self.tf_embed_cache = Some(tf_embedded);
        self.gene_embed_cache = Some(gene_embedded);
        self.tf_h1_cache = Some(tf_h1);
        self.gene_h1_cache = Some(gene_h1);
        self.tf_final_cache = Some(tf_final);
        self.gene_final_cache = Some(gene_final);
        
        scores
    }
    
    /// Backward pass
    pub fn backward(&mut self, grad_scores: &Array2<f32>) {
        use crate::models::nn::relu_backward;
        
        let grad_scores = grad_scores / self.temperature;
        
        let tf_final = self.tf_final_cache.as_ref().unwrap();
        let gene_final = self.gene_final_cache.as_ref().unwrap();
        
        // Gradient through dot product
        let grad_tf_final = grad_scores.dot(gene_final);
        let grad_gene_final = grad_scores.t().dot(tf_final);
        
        // TF backward
        let grad_tf_h1_relu = self.tf_fc2.backward(&grad_tf_final);
        let tf_h1 = self.tf_h1_cache.as_ref().unwrap();
        let grad_tf_h1 = relu_backward(tf_h1, &grad_tf_h1_relu);
        let grad_tf_embedded = self.tf_fc1.backward(&grad_tf_h1);
        
        // Gene backward
        let grad_gene_h1_relu = self.gene_fc2.backward(&grad_gene_final);
        let gene_h1 = self.gene_h1_cache.as_ref().unwrap();
        let grad_gene_h1 = relu_backward(gene_h1, &grad_gene_h1_relu);
        let grad_gene_embedded = self.gene_fc1.backward(&grad_gene_h1);
        
        // Backward through embeddings
        self.tf_embed.backward(&grad_tf_embedded);
        self.gene_embed.backward(&grad_gene_embedded);
    }
    
    /// Update all parameters
    pub fn update(&mut self, learning_rate: f32) {
        self.tf_embed.update(learning_rate);
        self.gene_embed.update(learning_rate);
        self.tf_fc1.update(learning_rate);
        self.tf_fc2.update(learning_rate);
        self.gene_fc1.update(learning_rate);
        self.gene_fc2.update(learning_rate);
    }
    
    /// Zero all gradients
    pub fn zero_grad(&mut self) {
        self.tf_embed.zero_grad();
        self.gene_embed.zero_grad();
        self.tf_fc1.zero_grad();
        self.tf_fc2.zero_grad();
        self.gene_fc1.zero_grad();
        self.gene_fc2.zero_grad();
    }
    
    /// Count total parameters
    pub fn count_parameters(&self) -> usize {
        self.tf_embed.num_parameters()
            + self.gene_embed.num_parameters()
            + self.tf_fc1.weights.len() + self.tf_fc1.bias.len()
            + self.tf_fc2.weights.len() + self.tf_fc2.bias.len()
            + self.gene_fc1.weights.len() + self.gene_fc1.bias.len()
            + self.gene_fc2.weights.len() + self.gene_fc2.bias.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learnable_embedding() {
        let mut embed = LearnableEmbedding::new(100, 64, 42);
        let indices = vec![0, 5, 10];
        let output = embed.forward(&indices);

        assert_eq!(output.shape(), &[3, 64]);
        
        // Test backward
        let grad = Array2::ones((3, 64));
        embed.backward(&grad);
        
        // Check gradients accumulated
        assert!(embed.grad_weights.row(0).iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_embedding_two_tower() {
        let mut model = EmbeddingTwoTower::new(
            100,  // num_tfs
            200,  // num_genes
            64,   // embed_dim
            128,  // hidden_dim
            64,   // output_dim
            0.0,  // dropout
            0.07, // temperature
            42
        );
        
        let tf_indices = vec![0, 1, 2, 3];
        let gene_indices = vec![10, 11, 12, 13];
        
        let scores = model.forward(&tf_indices, &gene_indices);
        assert_eq!(scores.shape(), &[4, 4]);
        
        // Test backward
        let grad = Array2::ones((4, 4));
        model.backward(&grad);
        
        // Check gradients exist
        assert!(model.tf_embed.grad_weights.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_parameter_count() {
        let model = EmbeddingTwoTower::new(100, 200, 64, 128, 64, 0.0, 0.07, 42);
        let params = model.count_parameters();
        
        // TF embed: 100*64 = 6,400
        // Gene embed: 200*64 = 12,800
        // TF fc1: 64*128 + 128 = 8,320
        // TF fc2: 128*64 + 64 = 8,256
        // Gene fc1: 64*128 + 128 = 8,320
        // Gene fc2: 128*64 + 64 = 8,256
        // Total: 52,352
        assert_eq!(params, 52352);
    }
}
