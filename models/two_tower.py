"""
Two-Tower (Bi-Encoder) Model for GRN Inference

Architecture:
- TF Encoder: Embedding → MLP → 128-dim representation
- Gene Encoder: Embedding → MLP → 128-dim representation  
- Scoring: Dot product similarity / temperature

Parameters: ~132,000 (matched with baseline)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTowerModel(nn.Module):
    """Two-Tower model for state-specific GRN inference."""
    
    def __init__(
        self,
        num_tfs: int,
        num_genes: int,
        num_states: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        """
        Args:
            num_tfs: Number of unique transcription factors
            num_genes: Number of target genes
            num_states: Number of cellular states
            embedding_dim: Dimension of initial embeddings
            hidden_dim: Hidden layer dimension
            output_dim: Final representation dimension
            dropout: Dropout probability
            temperature: Temperature for contrastive learning
        """
        super().__init__()
        
        self.num_tfs = num_tfs
        self.num_genes = num_genes
        self.num_states = num_states
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.temperature = temperature
        
        # Embeddings
        self.tf_embedding = nn.Embedding(num_tfs, embedding_dim)
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        self.state_embedding = nn.Embedding(num_states, embedding_dim)
        
        # TF Encoder (processes TF + state)
        self.tf_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # Gene Encoder (processes Gene + state)
        self.gene_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def encode_tf(self, tf_ids: torch.Tensor, state_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode TFs with their states.
        
        Args:
            tf_ids: (batch_size,) TF indices
            state_ids: (batch_size,) State indices
            
        Returns:
            (batch_size, output_dim) TF representations
        """
        tf_emb = self.tf_embedding(tf_ids)
        state_emb = self.state_embedding(state_ids)
        x = torch.cat([tf_emb, state_emb], dim=-1)
        z_tf = self.tf_encoder(x)
        return z_tf
    
    def encode_gene(self, gene_ids: torch.Tensor, state_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode genes with their states.
        
        Args:
            gene_ids: (batch_size,) Gene indices
            state_ids: (batch_size,) State indices
            
        Returns:
            (batch_size, output_dim) Gene representations
        """
        gene_emb = self.gene_embedding(gene_ids)
        state_emb = self.state_embedding(state_ids)
        x = torch.cat([gene_emb, state_emb], dim=-1)
        z_gene = self.gene_encoder(x)
        return z_gene
    
    def forward(
        self,
        tf_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass computing edge scores.
        
        Args:
            tf_ids: (batch_size,) TF indices
            gene_ids: (batch_size,) Gene indices
            state_ids: (batch_size,) State indices
            
        Returns:
            (batch_size,) Edge scores (logits)
        """
        z_tf = self.encode_tf(tf_ids, state_ids)
        z_gene = self.encode_gene(gene_ids, state_ids)
        scores = (z_tf * z_gene).sum(dim=-1) / self.temperature
        return scores
    
    def compute_similarity_matrix(
        self,
        tf_ids: torch.Tensor,
        gene_ids: torch.Tensor,
        state_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute full similarity matrix for contrastive learning.
        
        Returns:
            (batch_size, batch_size) Similarity matrix
        """
        z_tf = self.encode_tf(tf_ids, state_ids)
        z_gene = self.encode_gene(gene_ids, state_ids)
        similarity = torch.matmul(z_tf, z_gene.t()) / self.temperature
        return similarity
    
    def count_parameters(self) -> dict:
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        breakdown = {
            'tf_embedding': sum(p.numel() for p in self.tf_embedding.parameters()),
            'gene_embedding': sum(p.numel() for p in self.gene_embedding.parameters()),
            'state_embedding': sum(p.numel() for p in self.state_embedding.parameters()),
            'tf_encoder': sum(p.numel() for p in self.tf_encoder.parameters()),
            'gene_encoder': sum(p.numel() for p in self.gene_encoder.parameters()),
            'total': total,
        }
        
        return breakdown


if __name__ == "__main__":
    # Test the model
    model = TwoTowerModel(num_tfs=1000, num_genes=5000, num_states=100)
    print("Two-Tower Model Parameters:", model.count_parameters())
    
    # Test forward pass
    batch_size = 64
    tf_ids = torch.randint(0, 1000, (batch_size,))
    gene_ids = torch.randint(0, 5000, (batch_size,))
    state_ids = torch.randint(0, 100, (batch_size,))
    
    scores = model(tf_ids, gene_ids, state_ids)
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print("✅ Two-Tower model test passed!")
