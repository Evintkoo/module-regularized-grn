/// Data loader for training GRN models
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;

/// Batch of training data
#[derive(Clone)]
pub struct Batch {
    pub tf_input: Array2<f32>,
    pub gene_input: Array2<f32>,
    pub labels: Array1<f32>,
    pub size: usize,
}

/// Simple dataset for GRN training
#[derive(Clone)]
pub struct GRNDataset {
    pub tf_features: Array2<f32>,
    pub gene_features: Array2<f32>,
    pub labels: Array1<f32>,
    pub n_samples: usize,
}

impl GRNDataset {
    pub fn new(tf_features: Array2<f32>, gene_features: Array2<f32>, labels: Array1<f32>) -> Self {
        let n_samples = tf_features.nrows();
        assert_eq!(gene_features.nrows(), n_samples);
        assert_eq!(labels.len(), n_samples);
        
        Self {
            tf_features,
            gene_features,
            labels,
            n_samples,
        }
    }
    
    /// Split dataset into train and validation
    pub fn split(self, val_ratio: f32, seed: u64) -> (Self, Self) {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..self.n_samples).collect();
        indices.shuffle(&mut rng);
        
        let n_val = (self.n_samples as f32 * val_ratio) as usize;
        let n_train = self.n_samples - n_val;
        
        let train_indices = &indices[..n_train];
        let val_indices = &indices[n_train..];
        
        // Extract train data
        let train_tf = self.tf_features.select(ndarray::Axis(0), train_indices);
        let train_gene = self.gene_features.select(ndarray::Axis(0), train_indices);
        let train_labels = Array1::from_vec(
            train_indices.iter().map(|&i| self.labels[i]).collect()
        );
        
        // Extract val data
        let val_tf = self.tf_features.select(ndarray::Axis(0), val_indices);
        let val_gene = self.gene_features.select(ndarray::Axis(0), val_indices);
        let val_labels = Array1::from_vec(
            val_indices.iter().map(|&i| self.labels[i]).collect()
        );
        
        let train = Self::new(train_tf, train_gene, train_labels);
        let val = Self::new(val_tf, val_gene, val_labels);
        
        (train, val)
    }
}

/// Data loader that creates batches
pub struct DataLoader {
    dataset: GRNDataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_idx: usize,
}

impl DataLoader {
    pub fn new(dataset: GRNDataset, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset.n_samples).collect();
        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }
    
    pub fn reset(&mut self, rng: &mut StdRng) {
        self.current_idx = 0;
        if self.shuffle {
            self.indices.shuffle(rng);
        }
    }
    
    pub fn num_batches(&self) -> usize {
        (self.dataset.n_samples + self.batch_size - 1) / self.batch_size
    }
}

impl Iterator for DataLoader {
    type Item = Batch;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.n_samples {
            return None;
        }
        
        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.n_samples);
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let batch_size = batch_indices.len();
        
        // Extract batch data
        let tf_input = self.dataset.tf_features.select(ndarray::Axis(0), batch_indices);
        let gene_input = self.dataset.gene_features.select(ndarray::Axis(0), batch_indices);
        let labels = Array1::from_vec(
            batch_indices.iter().map(|&i| self.dataset.labels[i]).collect()
        );
        
        self.current_idx = end_idx;
        
        Some(Batch {
            tf_input,
            gene_input,
            labels,
            size: batch_size,
        })
    }
}

/// Create dummy dataset for testing
pub fn create_dummy_dataset(n_samples: usize, tf_dim: usize, gene_dim: usize, seed: u64) -> GRNDataset {
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    
    let mut rng = StdRng::seed_from_u64(seed);
    
    let tf_features = Array2::random_using(
        (n_samples, tf_dim),
        Uniform::new(-1.0, 1.0),
        &mut rng
    );
    
    let gene_features = Array2::random_using(
        (n_samples, gene_dim),
        Uniform::new(-1.0, 1.0),
        &mut rng
    );
    
    // Random binary labels
    let labels = Array1::from_vec(
        (0..n_samples).map(|_| if rng.gen::<f32>() > 0.5 { 1.0 } else { 0.0 }).collect()
    );
    
    GRNDataset::new(tf_features, gene_features, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_creation() {
        let dataset = create_dummy_dataset(100, 128, 128, 42);
        assert_eq!(dataset.n_samples, 100);
        assert_eq!(dataset.tf_features.shape(), &[100, 128]);
    }

    #[test]
    fn test_dataset_split() {
        let dataset = create_dummy_dataset(100, 128, 128, 42);
        let (train, val) = dataset.split(0.2, 42);
        
        assert_eq!(train.n_samples, 80);
        assert_eq!(val.n_samples, 20);
    }

    #[test]
    fn test_dataloader() {
        let dataset = create_dummy_dataset(100, 128, 128, 42);
        let mut loader = DataLoader::new(dataset, 32, false);
        
        let num_batches = loader.num_batches();
        assert_eq!(num_batches, 4); // 100 / 32 = 4 batches
        
        let first_batch = loader.next().unwrap();
        assert_eq!(first_batch.size, 32);
        assert_eq!(first_batch.tf_input.shape(), &[32, 128]);
    }
}
