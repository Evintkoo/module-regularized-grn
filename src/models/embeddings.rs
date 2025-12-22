use candle_core::{Tensor, Device, Result, DType};
use candle_nn::{Linear, Embedding, VarBuilder, Module, Dropout};

/// Embedding layer for TFs, genes, and states
pub struct GRNEmbeddings {
    pub tf_embed: Embedding,
    pub gene_embed: Embedding,
    pub state_embed: Embedding,
}

impl GRNEmbeddings {
    pub fn new(
        n_tfs: usize,
        n_genes: usize,
        n_states: usize,
        embed_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let tf_embed = candle_nn::embedding(n_tfs, embed_dim, vb.pp("tf_embed"))?;
        let gene_embed = candle_nn::embedding(n_genes, embed_dim, vb.pp("gene_embed"))?;
        let state_embed = candle_nn::embedding(n_states, embed_dim, vb.pp("state_embed"))?;
        
        Ok(Self {
            tf_embed,
            gene_embed,
            state_embed,
        })
    }
    
    pub fn tf_forward(&self, tf_ids: &Tensor) -> Result<Tensor> {
        self.tf_embed.forward(tf_ids)
    }
    
    pub fn gene_forward(&self, gene_ids: &Tensor) -> Result<Tensor> {
        self.gene_embed.forward(gene_ids)
    }
    
    pub fn state_forward(&self, state_ids: &Tensor) -> Result<Tensor> {
        self.state_embed.forward(state_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;
    
    #[test]
    fn test_embeddings_creation() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        
        let embeddings = GRNEmbeddings::new(1000, 5000, 100, 128, vb)?;
        
        // Test forward pass
        let tf_ids = Tensor::new(&[0u32, 1, 2], &device)?;
        let tf_emb = embeddings.tf_forward(&tf_ids)?;
        
        assert_eq!(tf_emb.dims(), &[3, 128]);
        Ok(())
    }
}

