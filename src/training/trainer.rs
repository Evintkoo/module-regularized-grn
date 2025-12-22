use crate::loss::{infonce_loss, binary_cross_entropy, compute_metrics, Metrics};
use crate::training::{Optimizer, Checkpoint};
use ndarray::{Array1, Array2};
use std::path::Path;
use anyhow::Result;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub log_interval: usize,
    pub checkpoint_dir: String,
    pub early_stopping_patience: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            log_interval: 10,
            checkpoint_dir: "checkpoints".to_string(),
            early_stopping_patience: 10,
        }
    }
}

/// Training history
#[derive(Debug, Clone)]
pub struct TrainHistory {
    pub train_losses: Vec<f32>,
    pub val_losses: Vec<f32>,
    pub val_metrics: Vec<Metrics>,
}

impl TrainHistory {
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            val_metrics: Vec::new(),
        }
    }

    pub fn add_epoch(&mut self, train_loss: f32, val_loss: f32, metrics: Metrics) {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.val_metrics.push(metrics);
    }

    pub fn best_val_loss(&self) -> f32 {
        self.val_losses.iter()
            .cloned()
            .fold(f32::INFINITY, f32::min)
    }

    pub fn best_epoch(&self) -> usize {
        self.val_losses.iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Simple trainer for models (without full backprop)
/// This is a simplified trainer that can run the training loop
/// Gradient computation would be added separately
pub struct Trainer {
    pub config: TrainConfig,
    pub history: TrainHistory,
}

impl Trainer {
    pub fn new(config: TrainConfig) -> Self {
        Self {
            config,
            history: TrainHistory::new(),
        }
    }

    /// Training epoch placeholder
    /// In practice, this would:
    /// 1. Loop over batches
    /// 2. Forward pass
    /// 3. Compute loss
    /// 4. Backward pass (compute gradients)
    /// 5. Optimizer step
    pub fn train_epoch_placeholder(&self) -> f32 {
        // Placeholder: return dummy loss
        // Real implementation would process batches
        0.5
    }

    /// Validation epoch placeholder
    pub fn validate_epoch_placeholder(&self) -> (f32, Metrics) {
        // Placeholder: return dummy metrics
        let mut metrics = Metrics::new();
        metrics.auroc = 0.7;
        (0.6, metrics)
    }

    /// Save checkpoint
    pub fn save_checkpoint(&self, epoch: usize, train_loss: f32, val_loss: f32, model_type: &str) -> Result<()> {
        let checkpoint = Checkpoint::new(
            epoch,
            train_loss,
            val_loss,
            self.history.best_val_loss(),
            model_type,
        );

        let path = format!("{}/checkpoint_epoch_{}.json", self.config.checkpoint_dir, epoch);
        std::fs::create_dir_all(&self.config.checkpoint_dir)?;
        checkpoint.save(path)?;

        Ok(())
    }

    /// Check early stopping
    pub fn should_stop(&self, current_epoch: usize) -> bool {
        if self.history.val_losses.is_empty() {
            return false;
        }

        let best_epoch = self.history.best_epoch();
        current_epoch - best_epoch >= self.config.early_stopping_patience
    }

    /// Print epoch summary
    pub fn print_epoch(&self, epoch: usize, train_loss: f32, val_loss: f32, metrics: &Metrics) {
        println!(
            "Epoch {:3} | Train Loss: {:.4} | Val Loss: {:.4} | Val AUROC: {:.4} | Val AUPRC: {:.4}",
            epoch, train_loss, val_loss, metrics.auroc, metrics.auprc
        );
    }
}

/// Batch iterator for training
pub struct BatchIterator<'a> {
    data: &'a [usize],
    batch_size: usize,
    current: usize,
}

impl<'a> BatchIterator<'a> {
    pub fn new(data: &'a [usize], batch_size: usize) -> Self {
        Self {
            data,
            batch_size,
            current: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = &'a [usize];

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.data.len() {
            return None;
        }

        let end = (self.current + self.batch_size).min(self.data.len());
        let batch = &self.data[self.current..end];
        self.current = end;

        Some(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_config() {
        let config = TrainConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_train_history() {
        let mut history = TrainHistory::new();
        let metrics = Metrics::new();

        history.add_epoch(0.5, 0.6, metrics.clone());
        history.add_epoch(0.4, 0.55, metrics.clone());
        history.add_epoch(0.3, 0.5, metrics.clone());

        assert_eq!(history.best_val_loss(), 0.5);
        assert_eq!(history.best_epoch(), 2);
    }

    #[test]
    fn test_early_stopping() {
        let mut config = TrainConfig::default();
        config.early_stopping_patience = 3;
        let mut trainer = Trainer::new(config);

        let metrics = Metrics::new();
        trainer.history.add_epoch(0.5, 0.5, metrics.clone());
        trainer.history.add_epoch(0.4, 0.6, metrics.clone());
        trainer.history.add_epoch(0.3, 0.7, metrics.clone());
        trainer.history.add_epoch(0.2, 0.8, metrics.clone());

        // Best was epoch 0, current is 3, patience is 3
        assert!(trainer.should_stop(3));
    }

    #[test]
    fn test_batch_iterator() {
        let data: Vec<usize> = (0..10).collect();
        let batches: Vec<_> = BatchIterator::new(&data, 3).collect();

        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0], &[0, 1, 2]);
        assert_eq!(batches[1], &[3, 4, 5]);
        assert_eq!(batches[2], &[6, 7, 8]);
        assert_eq!(batches[3], &[9]);
    }
}
