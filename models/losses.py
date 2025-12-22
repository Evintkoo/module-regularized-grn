"""
Loss functions for GRN inference training.

Includes:
- InfoNCE (Contrastive Loss)
- Prior Knowledge Loss
- Reconstruction Loss
- Combined Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Info NCE (Contrastive) Loss for GRN inference."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        similarity_matrix: torch.Tensor,
        positive_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss from similarity matrix.
        
        Args:
            similarity_matrix: (batch, batch) similarity scores
            positive_mask: (batch, batch) binary mask for positive pairs
                          If None, assumes diagonal are positives
        
        Returns:
            Scalar loss value
        """
        batch_size = similarity_matrix.size(0)
        device = similarity_matrix.device
        
        # Default: positives are on diagonal
        if positive_mask is None:
            positive_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        
        # Get positive scores
        positive_scores = similarity_matrix[positive_mask]  # (batch,)
        
        # Compute log-softmax over all negatives per row
        log_probs = F.log_softmax(similarity_matrix, dim=1)
        
        # Extract log probabilities for positives
        positive_log_probs = log_probs[positive_mask]  # (batch,)
        
        # InfoNCE loss: -log P(positive | positives + negatives)
        loss = -positive_log_probs.mean()
        
        return loss


class PriorKnowledgeLoss(nn.Module):
    """Prior knowledge regularization loss."""
    
    def __init__(self, weight: float = 0.1):
        """
        Args:
            weight: Weight for prior loss term
        """
        super().__init__()
        self.weight = weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        scores: torch.Tensor,
        prior_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prior knowledge loss.
        
        Args:
            scores: (batch,) Predicted edge scores (logits)
            prior_labels: (batch,) Binary labels from prior knowledge
                         1 = known edge, 0 = unknown/negative
        
        Returns:
            Weighted prior loss
        """
        loss = self.bce_loss(scores, prior_labels.float())
        return self.weight * loss


class ReconstructionLoss(nn.Module):
    """Expression reconstruction loss."""
    
    def __init__(self, weight: float = 0.3):
        """
        Args:
            weight: Weight for reconstruction term
        """
        super().__init__()
        self.weight = weight
    
    def forward(
        self,
        predicted_expr: torch.Tensor,
        target_expr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute expression reconstruction loss.
        
        Args:
            predicted_expr: (batch, n_genes) Predicted expression
            target_expr: (batch, n_genes) True expression
        
        Returns:
            Weighted reconstruction loss (MSE)
        """
        mse_loss = F.mse_loss(predicted_expr, target_expr)
        return self.weight * mse_loss


class CombinedLoss(nn.Module):
    """Combined loss for GRN training."""
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        prior_weight: float = 0.1,
        reconstruction_weight: float = 0.3,
        temperature: float = 0.07,
    ):
        """
        Args:
            contrastive_weight: Weight for contrastive loss
            prior_weight: Weight for prior knowledge loss
            reconstruction_weight: Weight for reconstruction loss
            temperature: Temperature for contrastive learning
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.prior_weight = prior_weight
        self.reconstruction_weight = reconstruction_weight
        
        self.contrastive_loss = InfoNCELoss(temperature=temperature)
        self.prior_loss = PriorKnowledgeLoss(weight=1.0)  # Weight applied externally
        self.reconstruction_loss = ReconstructionLoss(weight=1.0)
    
    def forward(
        self,
        similarity_matrix: torch.Tensor = None,
        scores: torch.Tensor = None,
        prior_labels: torch.Tensor = None,
        predicted_expr: torch.Tensor = None,
        target_expr: torch.Tensor = None,
        positive_mask: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            similarity_matrix: For contrastive loss
            scores: For prior loss
            prior_labels: Binary labels for priors
            predicted_expr: For reconstruction loss
            target_expr: True expression values
            positive_mask: Mask for positive pairs
        
        Returns:
            Dictionary with 'total' loss and individual components
        """
        losses = {}
        total_loss = 0.0
        
        # Contrastive loss
        if similarity_matrix is not None:
            l_contrast = self.contrastive_loss(similarity_matrix, positive_mask)
            losses['contrastive'] = l_contrast
            total_loss += self.contrastive_weight * l_contrast
        
        # Prior knowledge loss
        if scores is not None and prior_labels is not None:
            l_prior = self.prior_loss(scores, prior_labels)
            losses['prior'] = l_prior
            total_loss += self.prior_weight * l_prior
        
        # Reconstruction loss
        if predicted_expr is not None and target_expr is not None:
            l_recon = self.reconstruction_loss(predicted_expr, target_expr)
            losses['reconstruction'] = l_recon
            total_loss += self.reconstruction_weight * l_recon
        
        losses['total'] = total_loss
        
        return losses


def test_losses():
    """Test loss functions with dummy data."""
    
    batch_size = 64
    n_genes = 1000
    
    print("Testing Loss Functions")
    print("=" * 50)
    
    # Test InfoNCE Loss
    print("\n1. InfoNCE Loss")
    similarity_matrix = torch.randn(batch_size, batch_size)
    infoNCE = InfoNCELoss(temperature=0.07)
    loss_contrast = infoNCE(similarity_matrix)
    print(f"   Contrastive loss: {loss_contrast.item():.4f}")
    
    # Test Prior Loss
    print("\n2. Prior Knowledge Loss")
    scores = torch.randn(batch_size)
    prior_labels = torch.randint(0, 2, (batch_size,))
    prior_loss_fn = PriorKnowledgeLoss(weight=0.1)
    loss_prior = prior_loss_fn(scores, prior_labels)
    print(f"   Prior loss: {loss_prior.item():.4f}")
    
    # Test Reconstruction Loss
    print("\n3. Reconstruction Loss")
    predicted_expr = torch.randn(batch_size, n_genes)
    target_expr = torch.randn(batch_size, n_genes)
    recon_loss_fn = ReconstructionLoss(weight=0.3)
    loss_recon = recon_loss_fn(predicted_expr, target_expr)
    print(f"   Reconstruction loss: {loss_recon.item():.4f}")
    
    # Test Combined Loss
    print("\n4. Combined Loss")
    combined_loss_fn = CombinedLoss(
        contrastive_weight=1.0,
        prior_weight=0.1,
        reconstruction_weight=0.3,
    )
    
    losses = combined_loss_fn(
        similarity_matrix=similarity_matrix,
        scores=scores,
        prior_labels=prior_labels,
        predicted_expr=predicted_expr,
        target_expr=target_expr,
    )
    
    print(f"   Contrastive: {losses['contrastive'].item():.4f}")
    print(f"   Prior: {losses['prior'].item():.4f}")
    print(f"   Reconstruction: {losses['reconstruction'].item():.4f}")
    print(f"   Total: {losses['total'].item():.4f}")
    
    print("\n✅ All loss tests passed!")


if __name__ == "__main__":
    test_losses()
