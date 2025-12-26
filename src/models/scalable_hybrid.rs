/// Scalable hybrid model with configurable architecture for 95% target
use crate::models::nn::{LinearLayer, relu, relu_backward};
use ndarray::{Array1, Array2, Axis};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub embed_dim: usize,
    pub hidden_dims: Vec<usize>,  // Variable depth
    pub output_dim: usize,
    pub expr_dim: usize,
    pub temperature: f32,
    pub dropout: f32,
}

impl ModelConfig {
    pub fn small() -> Self {
        Self {
            embed_dim: 128,
            hidden_dims: vec![256],
            output_dim: 128,
            expr_dim: 11,
            temperature: 0.07,
            dropout: 0.0,
        }
    }
    
    pub fn medium() -> Self {
        Self {
            embed_dim: 256,
            hidden_dims: vec![512, 256],
            output_dim: 256,
            expr_dim: 11,
            temperature: 0.07,
            dropout: 0.1,
        }
    }
    
    pub fn large() -> Self {
        Self {
            embed_dim: 256,
            hidden_dims: vec![512, 512, 256],
            output_dim: 256,
            expr_dim: 11,
            temperature: 0.07,
            dropout: 0.15,
        }
    }
    
    pub fn xlarge() -> Self {
        Self {
            embed_dim: 512,
            hidden_dims: vec![1024, 1024, 512, 256],
            output_dim: 256,
            expr_dim: 11,
            temperature: 0.07,
            dropout: 0.2,
        }
    }
    
    pub fn ultra() -> Self {
        Self {
            embed_dim: 512,
            hidden_dims: vec![2048, 1024, 512, 256],
            output_dim: 256,
            expr_dim: 11,
            temperature: 0.07,
            dropout: 0.2,
        }
    }
}

pub struct ScalableHybridModel {
    pub config: ModelConfig,
    
    // Embeddings
    tf_embed: Array2<f32>,
    pub tf_embed_grad: Array2<f32>,
    gene_embed: Array2<f32>,
    pub gene_embed_grad: Array2<f32>,
    
    // Multi-layer encoders
    tf_layers: Vec<LinearLayer>,
    gene_layers: Vec<LinearLayer>,
    
    // Caches
    tf_indices_cache: Option<Vec<usize>>,
    gene_indices_cache: Option<Vec<usize>>,
    tf_expr_cache: Option<Array2<f32>>,
    gene_expr_cache: Option<Array2<f32>>,
    tf_activations: Vec<Option<Array2<f32>>>,
    gene_activations: Vec<Option<Array2<f32>>>,
}

impl ScalableHybridModel {
    pub fn new(
        num_tfs: usize,
        num_genes: usize,
        config: ModelConfig,
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
        let tf_embed = Array2::random_using((num_tfs, config.embed_dim), dist, &mut rng);
        let gene_embed = Array2::random_using((num_genes, config.embed_dim), dist, &mut rng);
        
        // Build multi-layer networks
        let input_dim = config.embed_dim + config.expr_dim;
        let mut tf_layers = Vec::new();
        let mut gene_layers = Vec::new();
        
        let mut current_dim = input_dim;
        for (i, &hidden_dim) in config.hidden_dims.iter().enumerate() {
            tf_layers.push(LinearLayer::new(current_dim, hidden_dim, seed + (i * 2) as u64));
            gene_layers.push(LinearLayer::new(current_dim, hidden_dim, seed + (i * 2 + 1) as u64));
            current_dim = hidden_dim;
        }
        
        // Final output layer
        let output_dim = config.output_dim;
        tf_layers.push(LinearLayer::new(current_dim, output_dim, seed + 1000));
        gene_layers.push(LinearLayer::new(current_dim, output_dim, seed + 1001));
        
        let num_layers = tf_layers.len();
        let embed_dim = config.embed_dim;
        
        Self {
            config,
            tf_embed,
            tf_embed_grad: Array2::zeros((num_tfs, embed_dim)),
            gene_embed,
            gene_embed_grad: Array2::zeros((num_genes, embed_dim)),
            tf_layers,
            gene_layers,
            tf_indices_cache: None,
            gene_indices_cache: None,
            tf_expr_cache: None,
            gene_expr_cache: None,
            tf_activations: vec![None; num_layers],
            gene_activations: vec![None; num_layers],
        }
    }
    
    pub fn forward(
        &mut self,
        tf_indices: &[usize],
        gene_indices: &[usize],
        tf_expr: &Array2<f32>,
        gene_expr: &Array2<f32>,
    ) -> Array1<f32> {
        let batch_size = tf_indices.len();
        
        // Cache inputs
        self.tf_indices_cache = Some(tf_indices.to_vec());
        self.gene_indices_cache = Some(gene_indices.to_vec());
        self.tf_expr_cache = Some(tf_expr.clone());
        self.gene_expr_cache = Some(gene_expr.clone());
        
        // Lookup embeddings
        let mut tf_embed_batch = Array2::zeros((batch_size, self.config.embed_dim));
        let mut gene_embed_batch = Array2::zeros((batch_size, self.config.embed_dim));
        
        for (i, &tf_idx) in tf_indices.iter().enumerate() {
            tf_embed_batch.row_mut(i).assign(&self.tf_embed.row(tf_idx));
        }
        for (i, &gene_idx) in gene_indices.iter().enumerate() {
            gene_embed_batch.row_mut(i).assign(&self.gene_embed.row(gene_idx));
        }
        
        // Concatenate embeddings + expression
        let tf_input = ndarray::concatenate![Axis(1), tf_embed_batch, tf_expr.clone()];
        let gene_input = ndarray::concatenate![Axis(1), gene_embed_batch, gene_expr.clone()];
        
        // Forward through TF encoder
        let mut tf_x = tf_input;
        let num_tf_layers = self.tf_layers.len();
        for i in 0..num_tf_layers {
            tf_x = self.tf_layers[i].forward(&tf_x);
            if i < num_tf_layers - 1 {
                // Apply ReLU to all except last layer
                self.tf_activations[i] = Some(tf_x.clone());
                tf_x = relu(&tf_x);
            } else {
                // Last layer, no activation
                self.tf_activations[i] = Some(tf_x.clone());
            }
        }
        let tf_out = tf_x;
        
        // Forward through gene encoder
        let mut gene_x = gene_input;
        let num_gene_layers = self.gene_layers.len();
        for i in 0..num_gene_layers {
            gene_x = self.gene_layers[i].forward(&gene_x);
            if i < num_gene_layers - 1 {
                self.gene_activations[i] = Some(gene_x.clone());
                gene_x = relu(&gene_x);
            } else {
                self.gene_activations[i] = Some(gene_x.clone());
            }
        }
        let gene_out = gene_x;
        
        // Compute scores: dot product / temperature
        let mut scores = Array1::zeros(batch_size);
        for i in 0..batch_size {
            let dot = tf_out.row(i).dot(&gene_out.row(i));
            let logit = dot / self.config.temperature;
            scores[i] = 1.0 / (1.0 + (-logit).exp());  // sigmoid
        }
        
        scores
    }
    
    pub fn backward(&mut self, grad_output: &Array1<f32>) {
        let batch_size = grad_output.len();
        
        let tf_out = self.tf_activations.last().unwrap().as_ref().unwrap();
        let gene_out = self.gene_activations.last().unwrap().as_ref().unwrap();
        
        // Gradient through sigmoid and dot product
        let mut grad_tf_out = Array2::zeros(tf_out.raw_dim());
        let mut grad_gene_out = Array2::zeros(gene_out.raw_dim());
        
        for i in 0..batch_size {
            let dot = tf_out.row(i).dot(&gene_out.row(i));
            let logit = dot / self.config.temperature;
            let score = 1.0 / (1.0 + (-logit).exp());
            let grad_sigmoid = score * (1.0 - score) * grad_output[i];
            let grad_dot = grad_sigmoid / self.config.temperature;
            
            grad_tf_out.row_mut(i).assign(&(&gene_out.row(i) * grad_dot));
            grad_gene_out.row_mut(i).assign(&(&tf_out.row(i) * grad_dot));
        }
        
        // Backward through gene encoder
        let mut grad_gene = grad_gene_out;
        for i in (0..self.gene_layers.len()).rev() {
            if i < self.gene_layers.len() - 1 {
                // Backward through ReLU
                let activation = self.gene_activations[i].as_ref().unwrap();
                grad_gene = relu_backward(&grad_gene, activation);
            }
            grad_gene = self.gene_layers[i].backward(&grad_gene);
        }
        
        // Backward through TF encoder
        let mut grad_tf = grad_tf_out;
        for i in (0..self.tf_layers.len()).rev() {
            if i < self.tf_layers.len() - 1 {
                let activation = self.tf_activations[i].as_ref().unwrap();
                grad_tf = relu_backward(&grad_tf, activation);
            }
            grad_tf = self.tf_layers[i].backward(&grad_tf);
        }
        
        // Split gradients: embeddings + expression
        let grad_tf_embed = grad_tf.slice(ndarray::s![.., 0..self.config.embed_dim]).to_owned();
        let grad_gene_embed = grad_gene.slice(ndarray::s![.., 0..self.config.embed_dim]).to_owned();
        
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
    
    pub fn update(&mut self, learning_rate: f32) {
        // Update embeddings
        self.tf_embed.scaled_add(-learning_rate, &self.tf_embed_grad);
        self.gene_embed.scaled_add(-learning_rate, &self.gene_embed_grad);
        
        // Update all layers
        for layer in &mut self.tf_layers {
            layer.update(learning_rate);
        }
        for layer in &mut self.gene_layers {
            layer.update(learning_rate);
        }
    }
    
    pub fn zero_grad(&mut self) {
        self.tf_embed_grad.fill(0.0);
        self.gene_embed_grad.fill(0.0);
        for layer in &mut self.tf_layers {
            layer.zero_grad();
        }
        for layer in &mut self.gene_layers {
            layer.zero_grad();
        }
    }
    
    pub fn num_parameters(&self) -> usize {
        let embed_params = self.tf_embed.len() + self.gene_embed.len();
        let tf_params: usize = self.tf_layers.iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum();
        let gene_params: usize = self.gene_layers.iter()
            .map(|l| l.weights.len() + l.bias.len())
            .sum();
        embed_params + tf_params + gene_params
    }
}
