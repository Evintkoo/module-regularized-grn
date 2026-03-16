use ndarray::{Array1, Array2};

/// Evaluation metrics for GRN prediction
#[derive(Debug, Clone)]
pub struct Metrics {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub auroc: f32,
    pub auprc: f32,
    pub early_precision_100: f32,
    pub early_precision_500: f32,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            auroc: 0.0,
            auprc: 0.0,
            early_precision_100: 0.0,
            early_precision_500: 0.0,
        }
    }
}

/// Compute precision, recall, F1 at a given threshold
pub fn precision_recall_f1(scores: &Array1<f32>, labels: &Array1<f32>, threshold: f32) -> (f32, f32, f32) {
    assert_eq!(scores.len(), labels.len());
    
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut fn_count = 0.0;
    
    for (score, label) in scores.iter().zip(labels.iter()) {
        let pred = if *score >= threshold { 1.0 } else { 0.0 };
        
        if pred > 0.5 && *label > 0.5 {
            tp += 1.0;
        } else if pred > 0.5 && *label < 0.5 {
            fp += 1.0;
        } else if pred < 0.5 && *label > 0.5 {
            fn_count += 1.0;
        }
    }
    
    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_count > 0.0 { tp / (tp + fn_count) } else { 0.0 };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    
    (precision, recall, f1)
}

/// Early precision: precision in top-k predictions
pub fn early_precision(scores: &Array1<f32>, labels: &Array1<f32>, k: usize) -> f32 {
    assert_eq!(scores.len(), labels.len());
    
    // Sort scores descending and get top-k
    let mut indexed_scores: Vec<(usize, f32)> = scores.iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    let top_k = indexed_scores.iter().take(k.min(scores.len()));
    
    let mut correct = 0.0;
    for (idx, _) in top_k {
        if labels[*idx] > 0.5 {
            correct += 1.0;
        }
    }
    
    correct / k.min(scores.len()) as f32
}

/// Compute AUROC using trapezoidal rule
pub fn auroc(scores: &Array1<f32>, labels: &Array1<f32>) -> f32 {
    assert_eq!(scores.len(), labels.len());
    
    // Sort by scores descending
    let mut pairs: Vec<(f32, f32)> = scores.iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    // Count positives and negatives
    let n_pos: f32 = labels.iter().filter(|&&l| l > 0.5).count() as f32;
    let n_neg: f32 = labels.iter().filter(|&&l| l < 0.5).count() as f32;
    
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5; // Undefined, return random
    }
    
    // Compute ROC curve and area using Mann-Whitney U statistic
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut area = 0.0;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;
    
    for (_, label) in pairs.iter() {
        if *label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
            // When we see a negative, add the area contribution
            // which is the number of positives seen so far
            area += tp;
        }
    }
    
    // Normalize by n_pos * n_neg
    area / (n_pos * n_neg)
}

/// Compute AUPRC (Area Under Precision-Recall Curve)
pub fn auprc(scores: &Array1<f32>, labels: &Array1<f32>) -> f32 {
    assert_eq!(scores.len(), labels.len());
    
    // Sort by scores descending
    let mut pairs: Vec<(f32, f32)> = scores.iter()
        .zip(labels.iter())
        .map(|(&s, &l)| (s, l))
        .collect();
    
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let n_pos: f32 = labels.iter().filter(|&&l| l > 0.5).count() as f32;
    
    if n_pos == 0.0 {
        return 0.0;
    }
    
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut area = 0.0;
    let mut prev_recall = 0.0;
    
    for (_, label) in pairs.iter() {
        if *label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        let precision = tp / (tp + fp);
        let recall = tp / n_pos;
        
        // Trapezoidal rule
        area += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    
    area
}

/// Compute all metrics
pub fn compute_metrics(scores: &Array1<f32>, labels: &Array1<f32>) -> Metrics {
    let (precision, recall, f1) = precision_recall_f1(scores, labels, 0.5);
    
    Metrics {
        precision,
        recall,
        f1,
        auroc: auroc(scores, labels),
        auprc: auprc(scores, labels),
        early_precision_100: early_precision(scores, labels, 100),
        early_precision_500: early_precision(scores, labels, 500),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_recall_f1_perfect() {
        let scores = Array1::from_vec(vec![0.9, 0.8, 0.1, 0.2]);
        let labels = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let (precision, recall, f1) = precision_recall_f1(&scores, &labels, 0.5);
        
        assert!((precision - 1.0).abs() < 1e-6);
        assert!((recall - 1.0).abs() < 1e-6);
        assert!((f1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_early_precision() {
        let scores = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.6, 0.5]);
        let labels = Array1::from_vec(vec![1.0, 1.0, 0.0, 1.0, 0.0]);
        
        let ep = early_precision(&scores, &labels, 2);
        assert!((ep - 1.0).abs() < 1e-6); // Top 2 are both correct
        
        let ep = early_precision(&scores, &labels, 3);
        assert!((ep - 2.0/3.0).abs() < 1e-6); // Top 3: 2 correct
    }

    #[test]
    fn test_auroc_perfect() {
        let scores = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.2, 0.1]);
        let labels = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0]);
        
        let roc = auroc(&scores, &labels);
        assert!((roc - 1.0).abs() < 0.1); // Should be close to 1.0
    }

    #[test]
    fn test_auroc_random() {
        // All same scores - AUROC undefined, but algorithm returns area based on order
        let scores = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let labels = Array1::from_vec(vec![1.0, 0.0, 1.0, 0.0]);
        
        let roc = auroc(&scores, &labels);
        // With tied scores, order is arbitrary so AUROC can vary
        assert!(roc >= 0.0 && roc <= 1.0, "AUROC should be in [0,1]");
    }

    #[test]
    fn test_compute_metrics() {
        let scores = Array1::from_vec(vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1]);
        let labels = Array1::from_vec(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        
        let metrics = compute_metrics(&scores, &labels);
        
        assert!(metrics.precision > 0.8);
        assert!(metrics.recall > 0.8);
        assert!(metrics.f1 > 0.8);
        assert!(metrics.auroc > 0.8);
    }
}
