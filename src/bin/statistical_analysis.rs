use anyhow::Result;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::HashMap;
use rand::{SeedableRng, rngs::StdRng};
use rand::seq::SliceRandom;

#[derive(Deserialize)]
struct Predictions {
    labels: Vec<f64>,
    predictions: Vec<f64>,
}

#[derive(Serialize)]
struct StatResults {
    bootstrap_ci: BootstrapCI,
    mcnemar: McnemarResult,
    error_by_confidence: HashMap<String, ConfidenceBin>,
}

#[derive(Serialize)]
struct BootstrapCI {
    accuracy: CIBand,
    // auroc CI not computed here; AUROC CI is read from evaluation_metrics.json by generate_figures
}

#[derive(Serialize)]
struct CIBand { mean: f64, lower: f64, upper: f64 }

#[derive(Serialize)]
struct McnemarResult { statistic: f64, p_value: f64 }

#[derive(Serialize)]
struct ConfidenceBin { n_samples: usize, accuracy: f64, error_rate: f64 }

pub fn bootstrap_ci_accuracy(y_true: &[f64], y_pred: &[f64], n: usize, seed: u64) -> (f64, f64, f64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let indices: Vec<usize> = (0..y_true.len()).collect();
    let mut scores = Vec::with_capacity(n);
    for _ in 0..n {
        let sample: Vec<usize> = (0..y_true.len())
            .map(|_| *indices.choose(&mut rng).unwrap())
            .collect();
        let acc = sample.iter()
            .filter(|&&i| (y_pred[i] > 0.5) == (y_true[i] > 0.5))
            .count() as f64 / sample.len() as f64;
        scores.push(acc);
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = scores.iter().sum::<f64>() / scores.len() as f64;
    let lower = scores[(0.025 * scores.len() as f64) as usize];
    let upper = scores[(0.975 * scores.len() as f64) as usize];
    (mean, lower, upper)
}

pub fn mcnemar_test(baseline_preds: &[f64], model_preds: &[f64], y_true: &[f64]) -> (f64, f64) {
    let baseline_correct: Vec<bool> = baseline_preds.iter().zip(y_true.iter())
        .map(|(p, t)| (p > &0.5) == (t > &0.5)).collect();
    let model_correct: Vec<bool> = model_preds.iter().zip(y_true.iter())
        .map(|(p, t)| (p > &0.5) == (t > &0.5)).collect();
    let b = baseline_correct.iter().zip(model_correct.iter())
        .filter(|(&bc, &mc)| bc && !mc).count() as f64;
    let c = baseline_correct.iter().zip(model_correct.iter())
        .filter(|(&bc, &mc)| !bc && mc).count() as f64;
    if b + c == 0.0 { return (0.0, 1.0); }
    let statistic = ((b - c).abs() - 1.0).powi(2) / (b + c);
    let chi2 = ChiSquared::new(1.0).unwrap();
    let p_value = 1.0 - chi2.cdf(statistic);
    (statistic, p_value)
}

fn main() -> Result<()> {
    println!("=== Statistical Analysis ===\n");
    let preds_path = "results/predictions.json";
    let data: Predictions = serde_json::from_str(&std::fs::read_to_string(preds_path)?)?;
    let y_true = &data.labels;
    let y_pred = &data.predictions;
    println!("Loaded {} predictions", y_true.len());

    let (acc_mean, acc_lower, acc_upper) = bootstrap_ci_accuracy(y_true, y_pred, 1000, 42);
    println!("Accuracy CI: {acc_mean:.4} [{acc_lower:.4}, {acc_upper:.4}]");

    let baseline_preds: Vec<f64> = vec![0.5; y_pred.len()];
    let (stat, p) = mcnemar_test(&baseline_preds, y_pred, y_true);
    println!("McNemar test vs random: statistic={stat:.4}, p={p:.6}");

    let bins = [
        ("Very Low",  0.0f64, 0.3f64),
        ("Low",       0.3,    0.5),
        ("Medium",    0.5,    0.7),
        ("High",      0.7,    0.9),
        ("Very High", 0.9,    1.001),
    ];
    let mut error_by_confidence = HashMap::new();
    for (label, lo, hi) in &bins {
        let mask: Vec<usize> = (0..y_pred.len()).filter(|&i| y_pred[i] >= *lo && y_pred[i] < *hi).collect();
        if mask.is_empty() { continue; }
        let acc = mask.iter().filter(|&&i| (y_pred[i] > 0.5) == (y_true[i] > 0.5)).count() as f64 / mask.len() as f64;
        error_by_confidence.insert(label.to_string(), ConfidenceBin {
            n_samples: mask.len(), accuracy: acc, error_rate: 1.0 - acc,
        });
    }

    let results = StatResults {
        bootstrap_ci: BootstrapCI {
            accuracy: CIBand { mean: acc_mean, lower: acc_lower, upper: acc_upper },
        },
        mcnemar: McnemarResult { statistic: stat, p_value: p },
        error_by_confidence,
    };
    std::fs::create_dir_all("results")?;
    std::fs::write("results/statistical_analysis.json", serde_json::to_string_pretty(&results)?)?;
    println!("\n✓ Written to results/statistical_analysis.json");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_ci_known_data() {
        let y_true = vec![1.0f64, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.9f64, 0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1];
        let (mean, lower, upper) = bootstrap_ci_accuracy(&y_true, &y_pred, 500, 42);
        assert!((mean - 1.0).abs() < 0.01, "mean should be ~1.0, got {mean}");
        assert!(lower > 0.9, "lower CI should be > 0.9, got {lower}");
        assert!(upper <= 1.0, "upper CI should be <= 1.0, got {upper}");
    }

    #[test]
    fn test_mcnemar_test_significant() {
        let y_true: Vec<f64> = vec![1.0; 50].into_iter().chain(vec![0.0; 50]).collect();
        let baseline: Vec<f64> = vec![0.6; 50].into_iter().chain(vec![0.6; 50]).collect();
        let model: Vec<f64> = vec![0.9; 50].into_iter().chain(vec![0.1; 50]).collect();
        let (stat, p) = mcnemar_test(&baseline, &model, &y_true);
        assert!(p < 0.05, "should be significant, got p={p}, stat={stat}");
    }
}
