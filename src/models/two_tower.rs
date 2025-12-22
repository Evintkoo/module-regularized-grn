use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Dropout, DropoutConfig};
use burn::tensor::activation;

/// TF Encoder for Two-Tower architecture
#[derive(Module, Debug)]
pub struct TFEncoder<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

/// Configuration for TF Encoder
#[derive(Config, Debug)]
pub struct TFEncoderConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub dropout_prob: f64,
}

impl TFEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TFEncoder<B> {
        TFEncoder {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_prob).init(),
        }
    }
}

impl<B: Backend> TFEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::relu(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        x
    }
}

/// Gene Encoder for Two-Tower architecture
#[derive(Module, Debug)]
pub struct GeneEncoder<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

/// Configuration for Gene Encoder
#[derive(Config, Debug)]
pub struct GeneEncoderConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub dropout_prob: f64,
}

impl GeneEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> GeneEncoder<B> {
        GeneEncoder {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim).init(device),
            fc2: LinearConfig::new(self.hidden_dim, self.output_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_prob).init(),
        }
    }
}

impl<B: Backend> GeneEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::relu(x);
        let x = self.dropout.forward(x);
        let x = self.fc2.forward(x);
        x
    }
}

/// Two-Tower model for GRN inference
#[derive(Module, Debug)]
pub struct TwoTowerModel<B: Backend> {
    tf_encoder: TFEncoder<B>,
    gene_encoder: GeneEncoder<B>,
    temperature: f64,
}

/// Configuration for Two-Tower model
#[derive(Config, Debug)]
pub struct TwoTowerConfig {
    pub tf_input_dim: usize,
    pub gene_input_dim: usize,
    pub hidden_dim: usize,
    pub embed_dim: usize,
    pub dropout_prob: f64,
    pub temperature: f64,
}

impl TwoTowerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TwoTowerModel<B> {
        let tf_encoder_config = TFEncoderConfig {
            input_dim: self.tf_input_dim,
            hidden_dim: self.hidden_dim,
            output_dim: self.embed_dim,
            dropout_prob: self.dropout_prob,
        };
        
        let gene_encoder_config = GeneEncoderConfig {
            input_dim: self.gene_input_dim,
            hidden_dim: self.hidden_dim,
            output_dim: self.embed_dim,
            dropout_prob: self.dropout_prob,
        };
        
        TwoTowerModel {
            tf_encoder: tf_encoder_config.init(device),
            gene_encoder: gene_encoder_config.init(device),
            temperature: self.temperature,
        }
    }
}

impl<B: Backend> TwoTowerModel<B> {
    /// Encode TF
    pub fn encode_tf(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.tf_encoder.forward(x)
    }
    
    /// Encode Gene
    pub fn encode_gene(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.gene_encoder.forward(x)
    }
    
    /// Compute similarity scores (dot product)
    pub fn score(&self, tf_embed: Tensor<B, 2>, gene_embed: Tensor<B, 2>) -> Tensor<B, 2> {
        // Dot product: [batch, embed_dim] @ [embed_dim, batch]^T = [batch, batch]
        let scores = tf_embed.matmul(gene_embed.transpose());
        
        // Apply temperature scaling
        scores / self.temperature
    }
    
    /// Forward pass
    pub fn forward(&self, tf_input: Tensor<B, 2>, gene_input: Tensor<B, 2>) -> Tensor<B, 2> {
        let tf_embed = self.encode_tf(tf_input);
        let gene_embed = self.encode_gene(gene_input);
        self.score(tf_embed, gene_embed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    
    type TestBackend = NdArray;

    #[test]
    fn test_two_tower_forward() {
        let device = Default::default();
        let config = TwoTowerConfig {
            tf_input_dim: 128,
            gene_input_dim: 128,
            hidden_dim: 256,
            embed_dim: 128,
            dropout_prob: 0.1,
            temperature: 0.07,
        };
        
        let model = config.init::<TestBackend>(&device);
        
        // Create dummy input
        let batch_size = 32;
        let tf_input = Tensor::<TestBackend, 2>::zeros([batch_size, 128], &device);
        let gene_input = Tensor::<TestBackend, 2>::zeros([batch_size, 128], &device);
        
        // Forward pass
        let scores = model.forward(tf_input, gene_input);
        
        // Check output shape
        assert_eq!(scores.shape().dims[0], batch_size);
        assert_eq!(scores.shape().dims[1], batch_size);
    }
}
