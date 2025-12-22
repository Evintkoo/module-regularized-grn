use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Embedding layer using lookup table
pub struct Embedding {
    pub weights: Array2<f32>, // [vocab_size, embed_dim]
}

impl Embedding {
    /// Create new embedding with random initialization
    pub fn new(vocab_size: usize, embed_dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let weights = Array2::random_using(
            (vocab_size, embed_dim),
            Uniform::new(-0.1, 0.1),
            &mut rng,
        );
        Self { weights }
    }

    /// Forward pass: lookup embeddings
    pub fn forward(&self, indices: &[usize]) -> Array2<f32> {
        let batch_size = indices.len();
        let embed_dim = self.weights.ncols();
        let mut output = Array2::zeros((batch_size, embed_dim));

        for (i, &idx) in indices.iter().enumerate() {
            output.row_mut(i).assign(&self.weights.row(idx));
        }

        output
    }
}

/// GRN Embeddings
pub struct GRNEmbeddings {
    pub tf_embed: Embedding,
    pub gene_embed: Embedding,
    pub state_embed: Embedding,
}

impl GRNEmbeddings {
    pub fn new(n_tfs: usize, n_genes: usize, n_states: usize, embed_dim: usize, seed: u64) -> Self {
        Self {
            tf_embed: Embedding::new(n_tfs, embed_dim, seed),
            gene_embed: Embedding::new(n_genes, embed_dim, seed + 1),
            state_embed: Embedding::new(n_states, embed_dim, seed + 2),
        }
    }

    pub fn tf_forward(&self, indices: &[usize]) -> Array2<f32> {
        self.tf_embed.forward(indices)
    }

    pub fn gene_forward(&self, indices: &[usize]) -> Array2<f32> {
        self.gene_embed.forward(indices)
    }

    pub fn state_forward(&self, indices: &[usize]) -> Array2<f32> {
        self.state_embed.forward(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_forward() {
        let embed = Embedding::new(100, 64, 42);
        let indices = vec![0, 5, 10];
        let output = embed.forward(&indices);

        assert_eq!(output.shape(), &[3, 64]);
    }

    #[test]
    fn test_grn_embeddings() {
        let embeddings = GRNEmbeddings::new(1000, 5000, 100, 128, 42);

        let tf_output = embeddings.tf_forward(&[0, 1, 2]);
        assert_eq!(tf_output.shape(), &[3, 128]);

        let gene_output = embeddings.gene_forward(&[0, 10, 20]);
        assert_eq!(gene_output.shape(), &[3, 128]);
    }
}



