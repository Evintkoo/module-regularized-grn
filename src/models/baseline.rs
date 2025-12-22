use tch::{nn, nn::Module, Device, Tensor};

/// Baseline model configuration
pub struct BaselineConfig {
    pub input_dim: i64,
    pub hidden_dim1: i64,
    pub hidden_dim2: i64,
    pub output_dim: i64,
    pub dropout_prob: f64,
}

impl BaselineConfig {
    /// Create baseline config parameter-matched to Two-Tower
    pub fn from_two_tower(
        tf_input_dim: i64,
        gene_input_dim: i64,
        hidden_dim: i64,
        embed_dim: i64,
        dropout_prob: f64,
    ) -> Self {
        let input_dim = tf_input_dim + gene_input_dim;
        let hidden_dim1 = hidden_dim * 2;
        let hidden_dim2 = embed_dim;

        Self {
            input_dim,
            hidden_dim1,
            hidden_dim2,
            output_dim: 1,
            dropout_prob,
        }
    }
}

/// Monolithic Baseline Model
#[derive(Debug)]
pub struct BaselineModel {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    dropout: nn::Dropout,
}

impl BaselineModel {
    pub fn new(vs: &nn::Path, config: &BaselineConfig) -> Self {
        let fc1 = nn::linear(
            vs / "fc1",
            config.input_dim,
            config.hidden_dim1,
            Default::default(),
        );
        let fc2 = nn::linear(
            vs / "fc2",
            config.hidden_dim1,
            config.hidden_dim2,
            Default::default(),
        );
        let fc3 = nn::linear(
            vs / "fc3",
            config.hidden_dim2,
            config.output_dim,
            Default::default(),
        );
        let dropout = nn::Dropout::new(config.dropout_prob);

        Self {
            fc1,
            fc2,
            fc3,
            dropout,
        }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply(&self.fc1)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.fc2)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.fc3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baseline_forward() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = BaselineConfig {
            input_dim: 256,
            hidden_dim1: 512,
            hidden_dim2: 128,
            output_dim: 1,
            dropout_prob: 0.1,
        };

        let model = BaselineModel::new(&vs.root(), &config);

        // Create dummy input
        let batch_size = 32;
        let input = Tensor::zeros(&[batch_size, 256], (tch::Kind::Float, Device::Cpu));

        // Forward pass
        let scores = model.forward(&input, false);

        assert_eq!(scores.size(), vec![batch_size, 1]);
    }

    #[test]
    fn test_baseline_parameter_matching() {
        let vs = nn::VarStore::new(Device::Cpu);

        let config = BaselineConfig::from_two_tower(128, 128, 256, 128, 0.1);

        let _model = BaselineModel::new(&vs.root(), &config);

        // Count parameters
        let total_params: i64 = vs.trainable_variables().iter().map(|t| t.size().iter().product::<i64>()).sum();

        println!("Baseline params: {}", total_params);
        // Should be ~132K parameters
        assert!(total_params > 100_000 && total_params < 200_000);
    }
}

