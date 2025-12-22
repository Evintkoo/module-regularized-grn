use tch::{nn, nn::Module, Device, Tensor};

/// TF Encoder configuration
pub struct EncoderConfig {
    pub input_dim: i64,
    pub hidden_dim: i64,
    pub output_dim: i64,
    pub dropout_prob: f64,
}

/// TF Encoder for Two-Tower
#[derive(Debug)]
pub struct TFEncoder {
    fc1: nn::Linear,
    fc2: nn::Linear,
    dropout: nn::Dropout,
}

impl TFEncoder {
    pub fn new(vs: &nn::Path, config: &EncoderConfig) -> Self {
        let fc1 = nn::linear(
            vs / "fc1",
            config.input_dim,
            config.hidden_dim,
            Default::default(),
        );
        let fc2 = nn::linear(
            vs / "fc2",
            config.hidden_dim,
            config.output_dim,
            Default::default(),
        );
        let dropout = nn::Dropout::new(config.dropout_prob);

        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply(&self.fc1)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.fc2)
    }
}

/// Gene Encoder for Two-Tower
#[derive(Debug)]
pub struct GeneEncoder {
    fc1: nn::Linear,
    fc2: nn::Linear,
    dropout: nn::Dropout,
}

impl GeneEncoder {
    pub fn new(vs: &nn::Path, config: &EncoderConfig) -> Self {
        let fc1 = nn::linear(
            vs / "fc1",
            config.input_dim,
            config.hidden_dim,
            Default::default(),
        );
        let fc2 = nn::linear(
            vs / "fc2",
            config.hidden_dim,
            config.output_dim,
            Default::default(),
        );
        let dropout = nn::Dropout::new(config.dropout_prob);

        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply(&self.fc1)
            .relu()
            .apply_t(&self.dropout, train)
            .apply(&self.fc2)
    }
}

/// Two-Tower model configuration
pub struct TwoTowerConfig {
    pub tf_input_dim: i64,
    pub gene_input_dim: i64,
    pub hidden_dim: i64,
    pub embed_dim: i64,
    pub dropout_prob: f64,
    pub temperature: f64,
}

/// Two-Tower Model
#[derive(Debug)]
pub struct TwoTowerModel {
    tf_encoder: TFEncoder,
    gene_encoder: GeneEncoder,
    temperature: f64,
}

impl TwoTowerModel {
    pub fn new(vs: &nn::Path, config: &TwoTowerConfig) -> Self {
        let tf_config = EncoderConfig {
            input_dim: config.tf_input_dim,
            hidden_dim: config.hidden_dim,
            output_dim: config.embed_dim,
            dropout_prob: config.dropout_prob,
        };

        let gene_config = EncoderConfig {
            input_dim: config.gene_input_dim,
            hidden_dim: config.hidden_dim,
            output_dim: config.embed_dim,
            dropout_prob: config.dropout_prob,
        };

        Self {
            tf_encoder: TFEncoder::new(&(vs / "tf_encoder"), &tf_config),
            gene_encoder: GeneEncoder::new(&(vs / "gene_encoder"), &gene_config),
            temperature: config.temperature,
        }
    }

    /// Encode TF
    pub fn encode_tf(&self, x: &Tensor, train: bool) -> Tensor {
        self.tf_encoder.forward(x, train)
    }

    /// Encode Gene
    pub fn encode_gene(&self, x: &Tensor, train: bool) -> Tensor {
        self.gene_encoder.forward(x, train)
    }

    /// Compute similarity scores
    pub fn score(&self, tf_embed: &Tensor, gene_embed: &Tensor) -> Tensor {
        // Dot product: [batch, embed_dim] @ [embed_dim, batch]
        let scores = tf_embed.matmul(&gene_embed.transpose(0, 1));
        scores / self.temperature
    }

    /// Forward pass
    pub fn forward(&self, tf_input: &Tensor, gene_input: &Tensor, train: bool) -> Tensor {
        let tf_embed = self.encode_tf(tf_input, train);
        let gene_embed = self.encode_gene(gene_input, train);
        self.score(&tf_embed, &gene_embed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_tower_forward() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = TwoTowerConfig {
            tf_input_dim: 128,
            gene_input_dim: 128,
            hidden_dim: 256,
            embed_dim: 128,
            dropout_prob: 0.1,
            temperature: 0.07,
        };

        let model = TwoTowerModel::new(&vs.root(), &config);

        // Create dummy input
        let batch_size = 32;
        let tf_input = Tensor::zeros(&[batch_size, 128], (tch::Kind::Float, Device::Cpu));
        let gene_input = Tensor::zeros(&[batch_size, 128], (tch::Kind::Float, Device::Cpu));

        // Forward pass
        let scores = model.forward(&tf_input, &gene_input, false);

        assert_eq!(scores.size(), vec![batch_size, batch_size]);
    }
}

