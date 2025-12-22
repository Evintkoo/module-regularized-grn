use ndarray::{Array1, Array2, Axis, concatenate};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;
use super::two_tower::{Linear, relu, dropout};

/// Baseline model (Cross-encoder)
pub struct BaselineModel {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    dropout_prob: f32,
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
            fc1: Linear::new(input_dim, hidden_dim1, seed),
            fc2: Linear::new(hidden_dim1, hidden_dim2, seed + 1),
            fc3: Linear::new(hidden_dim2, output_dim, seed + 2),
            dropout_prob,
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
    pub fn forward(&self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        let x = self.fc1.forward(x);
        let x = relu(&x);
        let x = dropout(&x, self.dropout_prob, training, rng);

        let x = self.fc2.forward(&x);
        let x = relu(&x);
        let x = dropout(&x, self.dropout_prob, training, rng);

        self.fc3.forward(&x)
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
        let model = BaselineModel::new(256, 512, 128, 1, 0.1, 42);

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
}


