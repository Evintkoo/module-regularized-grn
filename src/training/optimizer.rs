use ndarray::Array2;
use std::collections::HashMap;

/// Optimizer trait
pub trait Optimizer {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>);
    fn get_lr(&self) -> f32;
    fn set_lr(&mut self, lr: f32);
}

/// SGD with momentum
pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    velocity: HashMap<String, Array2<f32>>,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            velocity: HashMap::new(),
        }
    }

    pub fn update(&mut self, param_name: &str, params: &mut Array2<f32>, grads: &Array2<f32>) {
        // Add weight decay to gradients
        let mut grad_with_decay = grads.clone();
        if self.weight_decay > 0.0 {
            grad_with_decay = grad_with_decay + &(params.mapv(|x| x * self.weight_decay));
        }

        // Get or initialize velocity
        let velocity = self.velocity
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(params.dim()));

        // Update velocity: v = momentum * v - lr * grad
        *velocity = &*velocity * self.momentum - &grad_with_decay * self.lr;

        // Update parameters: params = params + v
        *params = &*params + &*velocity;
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        self.update("default", params, grads);
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adam optimizer
pub struct Adam {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub t: usize,
    m: HashMap<String, Array2<f32>>,  // First moment
    v: HashMap<String, Array2<f32>>,  // Second moment
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn default() -> Self {
        Self::new(0.001, 0.9, 0.999, 1e-8, 0.0)
    }

    pub fn update(&mut self, param_name: &str, params: &mut Array2<f32>, grads: &Array2<f32>) {
        self.t += 1;

        // Add weight decay to gradients
        let mut grad_with_decay = grads.clone();
        if self.weight_decay > 0.0 {
            grad_with_decay = grad_with_decay + &(params.mapv(|x| x * self.weight_decay));
        }

        // Get or initialize moments
        let m = self.m
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(params.dim()));
        let v = self.v
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(params.dim()));

        // Update biased first moment: m = beta1 * m + (1 - beta1) * grad
        *m = &*m * self.beta1 + &grad_with_decay * (1.0 - self.beta1);

        // Update biased second moment: v = beta2 * v + (1 - beta2) * grad^2
        *v = &*v * self.beta2 + &grad_with_decay.mapv(|x| x * x) * (1.0 - self.beta2);

        // Bias correction
        let m_hat = &*m / (1.0 - self.beta1.powi(self.t as i32));
        let v_hat = &*v / (1.0 - self.beta2.powi(self.t as i32));

        // Update parameters: params = params - lr * m_hat / (sqrt(v_hat) + eps)
        let update = &m_hat / &(v_hat.mapv(|x| x.sqrt()) + self.eps) * self.lr;
        *params = &*params - &update;
    }
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) {
        self.update("default", params, grads);
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Learning rate scheduler
pub enum LRScheduler {
    Constant,
    StepLR { step_size: usize, gamma: f32 },
    ExponentialLR { gamma: f32 },
    CosineAnnealing { t_max: usize, eta_min: f32 },
}

impl LRScheduler {
    pub fn step(&self, optimizer: &mut dyn Optimizer, epoch: usize, initial_lr: f32) {
        let new_lr = match self {
            LRScheduler::Constant => initial_lr,
            LRScheduler::StepLR { step_size, gamma } => {
                initial_lr * gamma.powi((epoch / step_size) as i32)
            }
            LRScheduler::ExponentialLR { gamma } => {
                initial_lr * gamma.powi(epoch as i32)
            }
            LRScheduler::CosineAnnealing { t_max, eta_min } => {
                let cos_val = (std::f32::consts::PI * (epoch as f32) / (*t_max as f32)).cos();
                eta_min + (initial_lr - eta_min) * (1.0 + cos_val) / 2.0
            }
        };
        optimizer.set_lr(new_lr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_sgd() {
        let mut sgd = SGD::new(0.1, 0.9, 0.0);
        let mut params = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let grads = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        sgd.update("test", &mut params, &grads);

        // Parameters should have moved in direction of negative gradient
        assert!(params[[0, 0]] < 1.0);
        assert!(params[[0, 1]] < 2.0);
    }

    #[test]
    fn test_adam() {
        let mut adam = Adam::default();
        let mut params = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let grads = arr2(&[[0.1, 0.2], [0.3, 0.4]]);

        adam.update("test", &mut params, &grads);

        // Parameters should have moved
        assert!(params[[0, 0]] < 1.0);
        assert!(params[[0, 1]] < 2.0);
    }

    #[test]
    fn test_lr_scheduler() {
        let mut sgd = SGD::new(0.1, 0.0, 0.0);
        let scheduler = LRScheduler::StepLR { step_size: 10, gamma: 0.1 };

        scheduler.step(&mut sgd, 0, 0.1);
        assert!((sgd.get_lr() - 0.1).abs() < 1e-6);

        scheduler.step(&mut sgd, 10, 0.1);
        assert!((sgd.get_lr() - 0.01).abs() < 1e-6);
    }
}
