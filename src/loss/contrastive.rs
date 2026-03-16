use ndarray::{Array1, Array2, Axis};

/// InfoNCE contrastive loss for Two-Tower model
/// 
/// Loss = -log( exp(sim(i,i)/τ) / Σ_j exp(sim(i,j)/τ) )
/// where sim(i,i) is the positive pair similarity
pub fn infonce_loss(scores: &Array2<f32>) -> f32 {
    let batch_size = scores.nrows();
    
    // Scores are already temperature-scaled from model
    // scores[i,j] = similarity between TF_i and Gene_j
    
    // For contrastive learning, diagonal elements are positive pairs
    let mut total_loss = 0.0;
    
    for i in 0..batch_size {
        // Positive: scores[i, i]
        let pos_score = scores[[i, i]];
        
        // All scores for this TF (including positive)
        let row = scores.row(i);
        
        // Log-sum-exp for numerical stability
        let max_score = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = row.iter().map(|&s| (s - max_score).exp()).sum();
        let log_sum_exp = max_score + sum_exp.ln();
        
        // Loss for this example: -log(exp(pos)/sum_exp) = -(pos - log_sum_exp)
        total_loss += log_sum_exp - pos_score;
    }
    
    total_loss / batch_size as f32
}

/// Binary cross-entropy loss for baseline model
/// 
/// Loss = -1/N Σ [y*log(σ(x)) + (1-y)*log(1-σ(x))]
pub fn binary_cross_entropy(logits: &Array2<f32>, labels: &Array1<f32>) -> f32 {
    assert_eq!(logits.len(), labels.len(), "Logits and labels must have same length");
    
    let mut total_loss = 0.0;
    let n = labels.len();
    
    for (i, &label) in labels.iter().enumerate() {
        let logit = logits[[i, 0]];
        
        // Numerical stability: use log-sigmoid
        // log(sigmoid(x)) = -log(1 + exp(-x)) = -softplus(-x)
        // log(1-sigmoid(x)) = -log(1 + exp(x)) = -softplus(x)
        let loss = if label > 0.5 {
            // y=1: -log(sigmoid(x))
            softplus(-logit)
        } else {
            // y=0: -log(1-sigmoid(x))
            softplus(logit)
        };
        
        total_loss += loss;
    }
    
    total_loss / n as f32
}

/// Softplus function: log(1 + exp(x))
/// With numerical stability for large x
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // For large x, softplus(x) ≈ x
    } else if x < -20.0 {
        0.0 // For small x, softplus(x) ≈ 0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Sigmoid function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Combined loss with weighting
pub fn combined_loss(
    contrastive_loss: f32,
    reconstruction_loss: f32,
    alpha: f32,
) -> f32 {
    alpha * contrastive_loss + (1.0 - alpha) * reconstruction_loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_infonce_loss() {
        // Perfect predictions: diagonal all high, off-diagonal all low
        let scores = arr2(&[
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ]);
        
        let loss = infonce_loss(&scores);
        assert!(loss < 1.0, "Loss should be small for perfect predictions");
    }

    #[test]
    fn test_infonce_loss_random() {
        // Random predictions: loss should be high
        let scores = arr2(&[
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [3.0, 2.0, 1.0],
        ]);
        
        let loss = infonce_loss(&scores);
        assert!(loss > 0.5, "Loss should be high for random predictions");
    }

    #[test]
    fn test_binary_cross_entropy() {
        let logits = arr2(&[[10.0], [-10.0], [0.0]]);
        let labels = Array1::from_vec(vec![1.0, 0.0, 1.0]);
        
        let loss = binary_cross_entropy(&logits, &labels);
        
        // First two are correct (low loss), third is wrong (high loss)
        assert!(loss > 0.0, "Loss should be positive");
        assert!(loss < 10.0, "Loss should be reasonable");
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_softplus() {
        assert!((softplus(0.0) - 0.693).abs() < 0.01); // ln(2) ≈ 0.693
        assert!(softplus(100.0) > 99.0);
        assert!(softplus(-100.0) < 1e-6);
    }

    #[test]
    fn test_combined_loss() {
        let loss = combined_loss(2.0, 3.0, 0.5);
        assert!((loss - 2.5).abs() < 1e-6);
        
        let loss = combined_loss(2.0, 3.0, 0.7);
        assert!((loss - 2.3).abs() < 1e-6);
    }
}
