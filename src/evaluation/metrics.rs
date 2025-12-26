/// Comprehensive evaluation metrics for GRN prediction
use ndarray::Array1;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationMetrics {
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub auroc: f32,
    pub auprc: f32,
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurvePoint {
    pub threshold: f32,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROCCurve {
    pub points: Vec<CurvePoint>,
    pub auroc: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PRCurve {
    pub points: Vec<CurvePoint>,
    pub auprc: f32,
}

impl EvaluationMetrics {
    /// Compute all metrics from predictions and labels
    pub fn compute(predictions: &Array1<f32>, labels: &Array1<f32>, threshold: f32) -> Self {
        assert_eq!(predictions.len(), labels.len());
        
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_count = 0;
        
        for i in 0..predictions.len() {
            let pred = if predictions[i] >= threshold { 1.0 } else { 0.0 };
            let label = labels[i];
            
            match (pred, label) {
                (1.0, 1.0) => tp += 1,
                (0.0, 0.0) => tn += 1,
                (1.0, 0.0) => fp += 1,
                (0.0, 1.0) => fn_count += 1,
                _ => {}
            }
        }
        
        let accuracy = (tp + tn) as f32 / predictions.len() as f32;
        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f32 / (tp + fn_count) as f32
        } else {
            0.0
        };
        let f1_score = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        
        // Compute AUROC and AUPRC
        let roc = compute_roc_curve(predictions, labels);
        let pr = compute_pr_curve(predictions, labels);
        
        Self {
            accuracy,
            precision,
            recall,
            f1_score,
            auroc: roc.auroc,
            auprc: pr.auprc,
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives: fn_count,
        }
    }
    
    /// Print formatted metrics
    pub fn print(&self) {
        println!("\n=== Evaluation Metrics ===");
        println!("Accuracy:  {:.4} ({:.2}%)", self.accuracy, self.accuracy * 100.0);
        println!("Precision: {:.4} ({:.2}%)", self.precision, self.precision * 100.0);
        println!("Recall:    {:.4} ({:.2}%)", self.recall, self.recall * 100.0);
        println!("F1 Score:  {:.4} ({:.2}%)", self.f1_score, self.f1_score * 100.0);
        println!("AUROC:     {:.4} ({:.2}%)", self.auroc, self.auroc * 100.0);
        println!("AUPRC:     {:.4} ({:.2}%)", self.auprc, self.auprc * 100.0);
        println!("\nConfusion Matrix:");
        println!("  TP: {}  FP: {}", self.true_positives, self.false_positives);
        println!("  FN: {}  TN: {}", self.false_negatives, self.true_negatives);
    }
}

/// Compute ROC curve (Receiver Operating Characteristic)
pub fn compute_roc_curve(predictions: &Array1<f32>, labels: &Array1<f32>) -> ROCCurve {
    let mut scored_samples: Vec<(f32, f32)> = predictions.iter()
        .zip(labels.iter())
        .map(|(&pred, &label)| (pred, label))
        .collect();
    
    // Sort by prediction score (descending)
    scored_samples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let total_positives = labels.iter().filter(|&&x| x == 1.0).count() as f32;
    let total_negatives = labels.iter().filter(|&&x| x == 0.0).count() as f32;
    
    let mut points = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;
    
    // Add (0, 0) point
    points.push(CurvePoint {
        threshold: 1.0,
        x: 0.0,
        y: 0.0,
    });
    
    let mut prev_score = f32::INFINITY;
    
    for (score, label) in scored_samples.iter() {
        if *score != prev_score {
            // Add point
            let fpr = fp / total_negatives;
            let tpr = tp / total_positives;
            points.push(CurvePoint {
                threshold: *score,
                x: fpr,
                y: tpr,
            });
            prev_score = *score;
        }
        
        if *label == 1.0 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }
    
    // Add (1, 1) point
    points.push(CurvePoint {
        threshold: 0.0,
        x: 1.0,
        y: 1.0,
    });
    
    // Compute AUROC using trapezoidal rule
    let mut auroc = 0.0;
    for i in 1..points.len() {
        let dx = points[i].x - points[i - 1].x;
        let avg_y = (points[i].y + points[i - 1].y) / 2.0;
        auroc += dx * avg_y;
    }
    
    ROCCurve { points, auroc }
}

/// Compute Precision-Recall curve
pub fn compute_pr_curve(predictions: &Array1<f32>, labels: &Array1<f32>) -> PRCurve {
    let mut scored_samples: Vec<(f32, f32)> = predictions.iter()
        .zip(labels.iter())
        .map(|(&pred, &label)| (pred, label))
        .collect();
    
    // Sort by prediction score (descending)
    scored_samples.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    
    let total_positives = labels.iter().filter(|&&x| x == 1.0).count() as f32;
    
    let mut points = Vec::new();
    let mut tp = 0.0;
    let mut fp = 0.0;
    
    let mut prev_score = f32::INFINITY;
    
    for (score, label) in scored_samples.iter() {
        if *label == 1.0 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        
        if *score != prev_score {
            let precision = tp / (tp + fp);
            let recall = tp / total_positives;
            
            points.push(CurvePoint {
                threshold: *score,
                x: recall,
                y: precision,
            });
            prev_score = *score;
        }
    }
    
    // Compute AUPRC using trapezoidal rule
    let mut auprc = 0.0;
    for i in 1..points.len() {
        let dx = points[i].x - points[i - 1].x;
        let avg_y = (points[i].y + points[i - 1].y) / 2.0;
        auprc += dx.abs() * avg_y;
    }
    
    PRCurve { points, auprc }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_perfect_predictions() {
        let predictions = Array1::from_vec(vec![0.9, 0.8, 0.1, 0.2]);
        let labels = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let metrics = EvaluationMetrics::compute(&predictions, &labels, 0.5);
        
        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.auroc, 1.0);
    }
    
    #[test]
    fn test_random_predictions() {
        let predictions = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5]);
        let labels = Array1::from_vec(vec![1.0, 1.0, 0.0, 0.0]);
        
        let metrics = EvaluationMetrics::compute(&predictions, &labels, 0.5);
        
        // Random predictions should give ~0.5 accuracy
        assert!(metrics.auroc >= 0.4 && metrics.auroc <= 0.6);
    }
}
