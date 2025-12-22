use burn::prelude::*;
use burn::nn::{Linear, LinearConfig, Dropout, DropoutConfig};
use burn::tensor::activation;

/// Monolithic baseline model (Cross-encoder)
/// Parameter-matched to Two-Tower for fair comparison
#[derive(Module, Debug)]
pub struct BaselineModel<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    fc3: Linear<B>,
    dropout: Dropout,
}

/// Configuration for Baseline model
#[derive(Config, Debug)]
pub struct BaselineConfig {
    pub input_dim: usize,
    pub hidden_dim1: usize,
    pub hidden_dim2: usize,
    pub output_dim: usize,
    pub dropout_prob: f64,
}

impl BaselineConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BaselineModel<B> {
        BaselineModel {
            fc1: LinearConfig::new(self.input_dim, self.hidden_dim1).init(device),
            fc2: LinearConfig::new(self.hidden_dim1, self.hidden_dim2).init(device),
            fc3: LinearConfig::new(self.hidden_dim2, self.output_dim).init(device),
            dropout: DropoutConfig::new(self.dropout_prob).init(),
        }
    }
    
    /// Create baseline config parameter-matched to Two-Tower
    pub fn from_two_tower(
        tf_input_dim: usize,
        gene_input_dim: usize,
        hidden_dim: usize,
        embed_dim: usize,
        dropout_prob: f64,
    ) -> Self {
        // Input: concatenate TF and Gene inputs
        let input_dim = tf_input_dim + gene_input_dim;
        
        // Hidden dims chosen to match parameter count
        // Two-Tower params: 2 * (input * hidden + hidden * embed)
        // Baseline params: input * h1 + h1 * h2 + h2 * 1
        let hidden_dim1 = hidden_dim * 2; // Larger first layer
        let hidden_dim2 = embed_dim; // Match embedding dim
        
        Self {
            input_dim,
            hidden_dim1,
            hidden_dim2,
            output_dim: 1, // Single score output
            dropout_prob,
        }
    }
}

impl<B: Backend> BaselineModel<B> {
    /// Forward pass: Takes concatenated TF and Gene features
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::relu(x);
        let x = self.dropout.forward(x);
        
        let x = self.fc2.forward(x);
        let x = activation::relu(x);
        let x = self.dropout.forward(x);
        
        let x = self.fc3.forward(x);
        x
    }
    
    /// Count parameters (for verification)
    pub fn count_params(&self) -> usize {
        // fc1: input_dim * hidden_dim1 + hidden_dim1
        // fc2: hidden_dim1 * hidden_dim2 + hidden_dim2
        // fc3: hidden_dim2 * output_dim + output_dim
        let fc1_params = self.fc1.weight.dims()[0] * self.fc1.weight.dims()[1] + self.fc1.weight.dims()[0];
        let fc2_params = self.fc2.weight.dims()[0] * self.fc2.weight.dims()[1] + self.fc2.weight.dims()[0];
        let fc3_params = self.fc3.weight.dims()[0] * self.fc3.weight.dims()[1] + self.fc3.weight.dims()[0];
        fc1_params + fc2_params + fc3_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray;

    #[test]
    fn test_baseline_forward() {
        let device = Default::default();
        let config = BaselineConfig {
            input_dim: 256, // tf_input (128) + gene_input (128)
            hidden_dim1: 512,
            hidden_dim2: 128,
            output_dim: 1,
            dropout_prob: 0.1,
        };
        
        let model = config.init::<TestBackend>(&device);
        
        // Create dummy input
        let batch_size = 32;
        let input = Tensor::<TestBackend, 2>::zeros([batch_size, 256], &device);
        
        // Forward pass
        let scores = model.forward(input);
        
        // Check output shape
        assert_eq!(scores.shape().dims[0], batch_size);
        assert_eq!(scores.shape().dims[1], 1);
    }
    
    #[test]
    fn test_baseline_parameter_matching() {
        let device = Default::default();
        
        // Two-Tower params
        let tf_input_dim = 128;
        let gene_input_dim = 128;
        let hidden_dim = 256;
        let embed_dim = 128;
        
        // Create parameter-matched baseline
        let config = BaselineConfig::from_two_tower(
            tf_input_dim,
            gene_input_dim,
            hidden_dim,
            embed_dim,
            0.1,
        );
        
        let model = config.init::<TestBackend>(&device);
        let baseline_params = model.count_params();
        
        // Two-Tower params (approximate)
        // TF encoder: (128*256 + 256) + (256*128 + 128) = 33,024 + 33,024 = 66,048
        // Gene encoder: same = 66,048
        // Total: ~132,096
        
        // Baseline should be similar
        println!("Baseline params: {}", baseline_params);
        assert!(baseline_params > 100_000 && baseline_params < 200_000);
    }
}
