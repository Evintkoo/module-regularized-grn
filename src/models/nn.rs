/// Neural Network implementation with backpropagation
/// Pure Rust implementation using ndarray

use ndarray::{Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Linear layer with gradients
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub weights: Array2<f32>,
    pub bias: Array1<f32>,
    
    // Gradients
    pub grad_weights: Array2<f32>,
    pub grad_bias: Array1<f32>,
    
    // Cache for backward pass
    input_cache: Option<Array2<f32>>,
}

impl LinearLayer {
    /// Create new linear layer with He initialization
    pub fn new(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let std = (2.0 / input_dim as f32).sqrt();
        
        let weights = Array2::random_using(
            (input_dim, output_dim),
            Uniform::new(-std, std),
            &mut rng,
        );
        let bias = Array1::zeros(output_dim);
        let grad_weights = Array2::zeros((input_dim, output_dim));
        let grad_bias = Array1::zeros(output_dim);

        Self { 
            weights, 
            bias, 
            grad_weights, 
            grad_bias,
            input_cache: None,
        }
    }

    /// Forward pass
    pub fn forward(&mut self, x: &Array2<f32>) -> Array2<f32> {
        self.input_cache = Some(x.clone());
        x.dot(&self.weights) + &self.bias
    }

    /// Backward pass
    pub fn backward(&mut self, grad_output: &Array2<f32>) -> Array2<f32> {
        let input = self.input_cache.as_ref().expect("Forward pass must be called first");
        
        // grad_weights = input^T @ grad_output
        self.grad_weights = input.t().dot(grad_output);
        
        // grad_bias = sum(grad_output, axis=0)
        self.grad_bias = grad_output.sum_axis(Axis(0));
        
        // grad_input = grad_output @ weights^T
        grad_output.dot(&self.weights.t())
    }

    /// Update weights with gradients
    pub fn update(&mut self, learning_rate: f32) {
        let batch_size = self.input_cache.as_ref().map(|x| x.nrows() as f32).unwrap_or(1.0);
        self.weights = &self.weights - &self.grad_weights * (learning_rate / batch_size);
        self.bias = &self.bias - &self.grad_bias * (learning_rate / batch_size);
    }

    /// Zero gradients
    pub fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_bias.fill(0.0);
    }
}

/// ReLU activation
pub fn relu(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v.max(0.0))
}

/// ReLU backward
pub fn relu_backward(x: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    grad_output * x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
}

/// Sigmoid activation
pub fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
}

/// Sigmoid backward
pub fn sigmoid_backward(x: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32> {
    let sig = sigmoid(x);
    grad_output * sig.mapv(|v| v * (1.0 - v))
}

/// Dropout layer
pub struct Dropout {
    pub prob: f32,
    mask: Option<Array2<f32>>,
}

impl Dropout {
    pub fn new(prob: f32) -> Self {
        Self { prob, mask: None }
    }

    pub fn forward(&mut self, x: &Array2<f32>, training: bool, rng: &mut StdRng) -> Array2<f32> {
        if !training || self.prob == 0.0 {
            return x.clone();
        }

        let mask = Array2::random_using(x.dim(), Uniform::new(0.0, 1.0), rng);
        let scale = 1.0 / (1.0 - self.prob);
        let mask = mask.mapv(|v| if v > self.prob { scale } else { 0.0 });
        self.mask = Some(mask.clone());
        x * mask
    }

    pub fn backward(&self, grad_output: &Array2<f32>) -> Array2<f32> {
        if let Some(mask) = &self.mask {
            grad_output * mask
        } else {
            grad_output.clone()
        }
    }
}

/// Binary cross-entropy loss
pub fn bce_loss(predictions: &Array2<f32>, targets: &Array1<f32>) -> f32 {
    let preds = sigmoid(predictions);
    let eps = 1e-7;
    
    let loss = targets.iter().zip(preds.iter()).map(|(&t, &p)| {
        let p = p.clamp(eps, 1.0 - eps);
        -t * p.ln() - (1.0 - t) * (1.0 - p).ln()
    }).sum::<f32>();
    
    loss / targets.len() as f32
}

/// Binary cross-entropy gradient
pub fn bce_loss_backward(predictions: &Array2<f32>, targets: &Array1<f32>) -> Array2<f32> {
    let preds = sigmoid(predictions);
    let eps = 1e-7;
    let batch_size = targets.len() as f32;
    
    let mut grad = Array2::zeros(predictions.dim());
    for (i, (&t, &p)) in targets.iter().zip(preds.iter()).enumerate() {
        let p = p.clamp(eps, 1.0 - eps);
        grad[[i, 0]] = (p - t) / batch_size;
    }
    
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_layer() {
        let mut layer = LinearLayer::new(10, 5, 42);
        let x = Array2::ones((3, 10));
        let y = layer.forward(&x);
        assert_eq!(y.dim(), (3, 5));
    }

    #[test]
    fn test_relu() {
        let x = Array2::from_shape_vec((2, 2), vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
        let y = relu(&x);
        assert_eq!(y[[0, 0]], 0.0);
        assert_eq!(y[[0, 1]], 2.0);
    }
}
