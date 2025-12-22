use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};

/// Embedding configuration
pub struct EmbeddingConfig {
    pub n_tfs: i64,
    pub n_genes: i64,
    pub n_states: i64,
    pub embed_dim: i64,
}

/// GRN Embeddings
#[derive(Debug)]
pub struct GRNEmbeddings {
    tf_embed: nn::Embedding,
    gene_embed: nn::Embedding,
    state_embed: nn::Embedding,
}

impl GRNEmbeddings {
    pub fn new(vs: &nn::Path, config: &EmbeddingConfig) -> Self {
        let tf_embed = nn::embedding(
            vs / "tf_embed",
            config.n_tfs,
            config.embed_dim,
            Default::default(),
        );
        let gene_embed = nn::embedding(
            vs / "gene_embed",
            config.n_genes,
            config.embed_dim,
            Default::default(),
        );
        let state_embed = nn::embedding(
            vs / "state_embed",
            config.n_states,
            config.embed_dim,
            Default::default(),
        );

        Self {
            tf_embed,
            gene_embed,
            state_embed,
        }
    }

    pub fn tf_forward(&self, tf_ids: &Tensor) -> Tensor {
        self.tf_embed.forward(tf_ids)
    }

    pub fn gene_forward(&self, gene_ids: &Tensor) -> Tensor {
        self.gene_embed.forward(gene_ids)
    }

    pub fn state_forward(&self, state_ids: &Tensor) -> Tensor {
        self.state_embed.forward(state_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embeddings_creation() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = EmbeddingConfig {
            n_tfs: 1000,
            n_genes: 5000,
            n_states: 100,
            embed_dim: 128,
        };

        let embeddings = GRNEmbeddings::new(&vs.root(), &config);

        // Test forward pass
        let tf_ids = Tensor::of_slice(&[0i64, 1, 2]).to_kind(tch::Kind::Int64);
        let tf_emb = embeddings.tf_forward(&tf_ids);

        assert_eq!(tf_emb.size(), vec![3, 128]);
    }
}


