"""
Baseline (Monolithic Cross-Encoder) Model for GRN Inference

Architecture:
- Input: Concatenated [TF_emb || Gene_emb || State_emb]
- MLP: 384 → 512 → 128 → 1
- Output: Edge score (scalar)

Parameters: ~132,000 (matched with Two-Tower)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    """Monolithic baseline model for GRN inference."""
    
    def __init__(
        self,
        num_tfs: int,
        num_genes: int,
        num_states: int,
        embedding_dim: int = 128,
        hidden_dims: list = [512, 128],
        dropout: float = 0.1,
    ):
        """
        Args:
            num_tfs: Number of unique transcription factors
            num_genes: Number of target genes
            num_states: Number of cellular states
            embedding_dim: Dimension of embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_tfs = num_tfs
        self.num_genes = num_genes
        self.num_states = num_states
        self.embedding_dim = embedding_dim
        
        # Embeddings
        self.tf_embedding = nn.Embedding(num_tfs, embedding_dim)
        self.gene_embedding = nn.Embedding(num_genes, embedding_dim)
        self.state_embedding = nn.Embedding(num_states, embedding_dim)
        
        # Cross-encoder MLP
        input_dim = 3 * embedding_dim  # Concatenated embeddings
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
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
        # Get embeddings
        tf_emb = self.tf_embedding(tf_ids)  # (batch, emb_dim)
        gene_emb = self.gene_embedding(gene_ids)  # (batch, emb_dim)
        state_emb = self.state_embedding(state_ids)  # (batch, emb_dim)
        
        # Concatenate all embeddings
        x = torch.cat([tf_emb, gene_emb, state_emb], dim=-1)  # (batch, 3*emb_dim)
        
        # Pass through MLP
        scores = self.mlp(x).squeeze(-1)  # (batch,)
        
        return scores
    
    def count_parameters(self) -> dict:
        """Count trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        breakdown = {
            'tf_embedding': sum(p.numel() for p in self.tf_embedding.parameters()),
            'gene_embedding': sum(p.numel() for p in self.gene_embedding.parameters()),
            'state_embedding': sum(p.numel() for p in self.state_embedding.parameters()),
            'mlp': sum(p.numel() for p in self.mlp.parameters()),
            'total': total,
        }
        
        return breakdown


if __name__ == "__main__":
    # Test the model
    model = BaselineModel(num_tfs=1000, num_genes=5000, num_states=100)
    print("Baseline Model Parameters:", model.count_parameters())
    
    # Test forward pass
    batch_size = 64
    tf_ids = torch.randint(0, 1000, (batch_size,))
    gene_ids = torch.randint(0, 5000, (batch_size,))
    state_ids = torch.randint(0, 100, (batch_size,))
    
    scores = model(tf_ids, gene_ids, state_ids)
    print(f"Output shape: {scores.shape}")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print("✅ Baseline model test passed!")
