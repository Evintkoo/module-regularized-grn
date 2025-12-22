use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Linear layer with weights and biases
pub struct Linear {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
}

impl Linear {
    /// He initialization for ReLU networks
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0 / input_dim as f32).sqrt();
        
        let weights = Array2::random_using(
            (input_dim, output_dim),
            Uniform::new(-std, std),
            &mut rng,
        );
        let bias = Array1::zeros(output_dim);

        Self { weights, bias }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        x.dot(&self.weights) + &self.bias
    }
}

/// ReLU activation
pub fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// Dropout (training only)
pub fn dropout(x: &Array2<f32>, prob: f32, training: bool, rng: &mut StdRng) -> Array2<f32> {
    if !training || prob == 0.0 {
        return x.clone();
    }

    let mask = Array2::random_using(x.dim(), Uniform::new(0.0, 1.0), rng);
    let scale = 1.0 / (1.0 - prob);
    x * mask.mapv(|v| if v > prob { scale } else { 0.0 })
}

/// TF Encoder
pub struct TFEncoder {
    fc1: Linear,
    fc2: Linear,
    dropout_prob: f32,
}

impl TFEncoder {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, dropout_prob: f32, seed: u64) -> Self {
        Self {
            fc1: Linear::new(input_dim, hidden_dim, seed),
            fc2: Linear::new(hidden_dim, output_dim, seed + 1),
            dropout_prob,
        }
    }

    pub fn forward(&self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        let x = self.fc1.forward(x);
        let x = relu(&x);
        let x = dropout(&x, self.dropout_prob, training, rng);
        self.fc2.forward(&x)
    }
}

/// Gene Encoder
pub struct GeneEncoder {
    fc1: Linear,
    fc2: Linear,
    dropout_prob: f32,
}

impl GeneEncoder {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, dropout_prob: f32, seed: u64) -> Self {
        Self {
            fc1: Linear::new(input_dim, hidden_dim, seed),
            fc2: Linear::new(hidden_dim, output_dim, seed + 1),
            dropout_prob,
        }
    }

    pub fn forward(&self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        let x = self.fc1.forward(x);
        let x = relu(&x);
        let x = dropout(&x, self.dropout_prob, training, rng);
        self.fc2.forward(&x)
    }
}

/// Two-Tower Model
pub struct TwoTowerModel {
    tf_encoder: TFEncoder,
    gene_encoder: GeneEncoder,
    temperature: f32,
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
        }
    }

    /// Encode TF
    pub fn encode_tf(&self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        self.tf_encoder.forward(x, training, rng)
    }

    /// Encode Gene
    pub fn encode_gene(&self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        self.gene_encoder.forward(x, training, rng)
    }

    /// Compute similarity scores via dot product
    pub fn score(&self, tf_embed: &Array2<f32>, gene_embed: &Array2<f32>) -> Array2<f32> {
        // [batch, embed] @ [embed, batch] = [batch, batch]
        let scores = tf_embed.dot(&gene_embed.t());
        scores / self.temperature
    }

    /// Forward pass
    pub fn forward(&self, tf_input: &Array2<f32>, gene_input: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        let tf_embed = self.encode_tf(tf_input, training, rng);
        let gene_embed = self.encode_gene(gene_input, training, rng);
        self.score(&tf_embed, &gene_embed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(128, 256, 42);
        let x = Array2::zeros((32, 128));
        let output = linear.forward(&x);
        assert_eq!(output.shape(), &[32, 256]);
    }

    #[test]
    fn test_relu() {
        let x = Array2::from_shape_vec((2, 3), vec![-1.0, 0.0, 1.0, -2.0, 0.5, 3.0]).unwrap();
        let output = relu(&x);
        assert_eq!(output[[0, 0]], 0.0); // -1 -> 0
        assert_eq!(output[[0, 2]], 1.0); // 1 -> 1
    }

    #[test]
    fn test_two_tower_forward() {
        let mut rng = StdRng::seed_from_u64(42);
        let model = TwoTowerModel::new(128, 128, 256, 128, 0.1, 0.07, 42);

        let tf_input = Array2::zeros((32, 128));
        let gene_input = Array2::zeros((32, 128));

        let scores = model.forward(&tf_input, &gene_input, false, &mut rng);
        assert_eq!(scores.shape(), &[32, 32]);
    }
}


