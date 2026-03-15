use anyhow::Result;
use plotters::prelude::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Predictions {
    labels: Vec<f64>,
    predictions: Vec<f64>,
}

/// Compute ROC curve points and AUROC. Returns (fpr, tpr, auc).
pub fn compute_roc_curve(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let mut pairs: Vec<(f64, f64)> = y_pred
        .iter()
        .zip(y_true.iter())
        .map(|(&p, &t)| (p, t))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = y_true.iter().filter(|&&v| v > 0.5).count() as f64;
    let n_neg = y_true.len() as f64 - n_pos;

    let mut fpr = vec![0.0f64];
    let mut tpr = vec![0.0f64];
    let mut tp = 0.0f64;
    let mut fp = 0.0f64;

    for (_, label) in &pairs {
        if *label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        fpr.push(fp / n_neg);
        tpr.push(tp / n_pos);
    }

    let auc: f64 = fpr
        .windows(2)
        .zip(tpr.windows(2))
        .map(|(f, t)| (f[1] - f[0]) * (t[0] + t[1]) / 2.0)
        .sum();

    (fpr, tpr, auc)
}

/// Compute precision-recall curve. Returns (recall, precision, average_precision).
pub fn compute_pr_curve(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
    let mut pairs: Vec<(f64, f64)> = y_pred
        .iter()
        .zip(y_true.iter())
        .map(|(&p, &t)| (p, t))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut recall = vec![];
    let mut precision = vec![];
    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    let n_pos = y_true.iter().filter(|&&v| v > 0.5).count() as f64;

    for (_, label) in &pairs {
        if *label > 0.5 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        recall.push(tp / n_pos);
        precision.push(tp / (tp + fp));
    }

    // Average precision: prepend (0, 1) sentinel then sum area under steps.
    // This matches sklearn's average_precision_score behaviour.
    let ap: f64 = if recall.len() > 1 {
        let mut r_ext = vec![0.0f64];
        r_ext.extend_from_slice(&recall);
        let mut p_ext = vec![1.0f64];
        p_ext.extend_from_slice(&precision);
        r_ext
            .windows(2)
            .zip(p_ext.windows(2))
            .map(|(r, p)| (r[1] - r[0]) * p[1])
            .sum()
    } else {
        0.0
    };

    (recall, precision, ap)
}

fn draw_roc_curve(fpr: &[f64], tpr: &[f64], auc: f64, out_path: &str) -> Result<()> {
    let root = SVGBackend::new(out_path, (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("ROC Curve", ("sans-serif", 20).into_font())
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;
    chart
        .configure_mesh()
        .x_desc("False Positive Rate")
        .y_desc("True Positive Rate")
        .draw()?;
    chart
        .draw_series(LineSeries::new(
            fpr.iter().zip(tpr.iter()).map(|(&x, &y)| (x, y)),
            &RGBColor(46, 134, 171),
        ))?
        .label(format!("Model (AUC = {auc:.4})"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(46, 134, 171)));
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, 0.0), (1.0, 1.0)],
            &BLACK.mix(0.4),
        ))?
        .label("Random (AUC = 0.5000)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.4)));
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}

fn draw_pr_curve(
    recall: &[f64],
    precision: &[f64],
    ap: f64,
    baseline: f64,
    out_path: &str,
) -> Result<()> {
    let root = SVGBackend::new(out_path, (600, 600)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Precision-Recall Curve", ("sans-serif", 20).into_font())
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;
    chart
        .configure_mesh()
        .x_desc("Recall")
        .y_desc("Precision")
        .draw()?;
    chart
        .draw_series(LineSeries::new(
            recall
                .iter()
                .zip(precision.iter())
                .map(|(&r, &p)| (r, p)),
            &RGBColor(162, 59, 114),
        ))?
        .label(format!("Model (AP = {ap:.4})"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(162, 59, 114)));
    chart
        .draw_series(LineSeries::new(
            vec![(0.0, baseline), (1.0, baseline)],
            &BLACK.mix(0.4),
        ))?
        .label(format!("Random (AP = {baseline:.4})"))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.mix(0.4)));
    chart.configure_series_labels().border_style(BLACK).draw()?;
    root.present()?;
    Ok(())
}

fn draw_ablation_bars(variants: &[(String, f64)], out_path: &str) -> Result<()> {
    let root = SVGBackend::new(out_path, (700, 450)).into_drawing_area();
    root.fill(&WHITE)?;
    let max_acc =
        (variants.iter().map(|(_, a)| *a).fold(0.0f64, f64::max) * 1.1).max(0.5);
    let mut chart = ChartBuilder::on(&root)
        .caption("Ablation Study", ("sans-serif", 20).into_font())
        .margin(30)
        .x_label_area_size(60)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..(variants.len() as f64), 0f64..max_acc)?;
    chart.configure_mesh().y_desc("Accuracy").draw()?;
    for (i, (_, acc)) in variants.iter().enumerate() {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(i as f64 + 0.1, 0.0), (i as f64 + 0.9, *acc)],
            RGBColor(70, 130, 180).filled(),
        )))?;
    }
    root.present()?;
    Ok(())
}

fn draw_seed_robustness(accuracies: &[f64], out_path: &str) -> Result<()> {
    let root = SVGBackend::new(out_path, (600, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    let min_acc = (accuracies
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min)
        - 0.05)
        .max(0.0);
    let max_acc = (accuracies
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        + 0.05)
        .min(1.0);
    let mut chart = ChartBuilder::on(&root)
        .caption("Seed Robustness", ("sans-serif", 20).into_font())
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..(accuracies.len() as f64), min_acc..max_acc)?;
    chart
        .configure_mesh()
        .x_desc("Seed")
        .y_desc("Accuracy")
        .draw()?;
    chart.draw_series(LineSeries::new(
        accuracies
            .iter()
            .enumerate()
            .map(|(i, &a)| (i as f64 + 0.5, a)),
        &RGBColor(70, 130, 180),
    ))?;
    root.present()?;
    Ok(())
}

fn main() -> Result<()> {
    println!("=== Generating Figures ===\n");
    std::fs::create_dir_all("figures")?;

    let preds: Predictions =
        serde_json::from_str(&std::fs::read_to_string("results/predictions.json")?)?;
    let y_true = &preds.labels;
    let y_pred = &preds.predictions;
    let baseline_ratio =
        y_true.iter().filter(|&&v| v > 0.5).count() as f64 / y_true.len() as f64;

    // Figure 1: ROC
    let (fpr, tpr, auc) = compute_roc_curve(y_true, y_pred);
    draw_roc_curve(&fpr, &tpr, auc, "figures/figure1_roc_curve.svg")?;
    println!("figure1_roc_curve.svg written (AUC={auc:.4})");

    // Figure 2: PR Curve
    let (recall, precision, ap) = compute_pr_curve(y_true, y_pred);
    draw_pr_curve(
        &recall,
        &precision,
        ap,
        baseline_ratio,
        "figures/figure2_pr_curve.svg",
    )?;
    println!("figure2_pr_curve.svg written (AP={ap:.4})");

    // Figure 3: Ablation — schema: {"results": [...]}
    if let Ok(abl_json) = std::fs::read_to_string("results/ablation_study.json") {
        let ablation: Value = serde_json::from_str(&abl_json)?;
        if let Some(results) = ablation["results"].as_array() {
            let data: Vec<(String, f64)> = results
                .iter()
                .map(|v| {
                    (
                        v["name"].as_str().unwrap_or("").to_string(),
                        v["accuracy"].as_f64().unwrap_or(0.0),
                    )
                })
                .collect();
            draw_ablation_bars(&data, "figures/figure3_ablation.svg")?;
            println!("figure3_ablation.svg written");
        }
    }

    // Figure 4: Seed robustness — schema: {"accuracies": [...]}
    if let Ok(seed_json) = std::fs::read_to_string("results/seed_robustness.json") {
        let seeds: Value = serde_json::from_str(&seed_json)?;
        if let Some(arr) = seeds["accuracies"].as_array() {
            let accs: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
            draw_seed_robustness(&accs, "figures/figure4_seed_robustness.svg")?;
            println!("figure4_seed_robustness.svg written");
        }
    }

    println!("\nAll figures written to figures/ (SVG format)");
    println!(
        "Note: For PDF output, use: rsvg-convert figures/figN.svg -o figures/figN.pdf"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roc_curve_perfect_classifier() {
        let y_true = vec![0.0f64, 0.0, 1.0, 1.0];
        let y_pred = vec![0.1f64, 0.2, 0.8, 0.9];
        let (fpr, tpr, auc) = compute_roc_curve(&y_true, &y_pred);
        assert!((auc - 1.0).abs() < 1e-6, "AUC should be 1.0, got {auc}");
        assert_eq!(fpr.first(), Some(&0.0));
        assert_eq!(tpr.last(), Some(&1.0));
    }

    #[test]
    fn test_roc_curve_below_chance() {
        let y_true = vec![0.0f64, 0.0, 1.0, 1.0];
        let y_pred = vec![0.9f64, 0.8, 0.2, 0.1];
        let (_, _, auc) = compute_roc_curve(&y_true, &y_pred);
        assert!(auc < 0.5, "AUC should be < 0.5 for inverse classifier, got {auc}");
    }

    #[test]
    fn test_pr_curve_perfect_classifier() {
        let y_true = vec![0.0f64, 0.0, 1.0, 1.0];
        let y_pred = vec![0.1f64, 0.2, 0.8, 0.9];
        let (_, _, ap) = compute_pr_curve(&y_true, &y_pred);
        assert!(ap > 0.9, "AP should be > 0.9 for perfect classifier, got {ap}");
    }
}
