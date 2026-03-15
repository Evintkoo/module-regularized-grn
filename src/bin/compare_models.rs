/// compare_models — reads result JSONs and prints comparison table.
/// Input files (missing files skipped):
///   results/two_tower_1to1.json
///   results/two_tower_5to1.json
///   results/cross_encoder_1to1.json
///   results/cross_encoder_5to1.json
/// Output:
///   stdout: formatted table
///   results/model_comparison.json
use anyhow::Result;
use serde_json::Value;

struct ModelRow {
    model:        String,
    neg_ratio:    usize,
    mean_acc:     f64,
    std_acc:      f64,
    mean_auroc:   f64,
    mean_f1:      f64,
    ensemble_acc: f64,
    ci_lower:     f64,
    ci_upper:     f64,
}

fn load_row(path: &str) -> Option<ModelRow> {
    let content = std::fs::read_to_string(path).ok()?;
    let v: Value = serde_json::from_str(&content).ok()?;

    let aurocs: Vec<f64> = v["seed_aurocs"].as_array()?
        .iter().filter_map(|x| x.as_f64()).collect();
    let f1s: Vec<f64> = v["seed_f1s"].as_array()?
        .iter().filter_map(|x| x.as_f64()).collect();
    let mean_auroc = if aurocs.is_empty() { 0.0 } else { aurocs.iter().sum::<f64>() / aurocs.len() as f64 };
    let mean_f1    = if f1s.is_empty()    { 0.0 } else { f1s.iter().sum::<f64>()    / f1s.len() as f64    };

    let model_key = v["model"].as_str().unwrap_or("unknown");
    let model_name = match model_key {
        "two_tower"     => "Two-Tower",
        "cross_encoder" => "Cross-Encoder",
        other => other,
    };

    Some(ModelRow {
        model:        model_name.to_string(),
        neg_ratio:    v["neg_ratio"].as_u64().unwrap_or(1) as usize,
        mean_acc:     v["mean_accuracy"].as_f64().unwrap_or(0.0),
        std_acc:      v["std_accuracy"].as_f64().unwrap_or(0.0),
        mean_auroc,
        mean_f1,
        ensemble_acc: v["ensemble_accuracy"].as_f64().unwrap_or(0.0),
        ci_lower:     v["bootstrap_ci_lower"].as_f64().unwrap_or(0.0),
        ci_upper:     v["bootstrap_ci_upper"].as_f64().unwrap_or(0.0),
    })
}

fn render_table(rows: &[ModelRow]) -> String {
    let header = format!(
        "{:<18} | {:>9} | {:>16} | {:>6} | {:>6} | {:>18}\n{}\n",
        "Model", "Neg Ratio", "Accuracy (±std)", "AUROC", "F1", "95% CI",
        "-".repeat(90)
    );
    let body: String = rows.iter().map(|r| {
        format!(
            "{:<18} | {:>9} | {:>7.2}% ±{:.2}%  | {:.4} | {:.4} | [{:.2}%, {:.2}%]\n",
            r.model,
            format!("{}:1", r.neg_ratio),
            r.mean_acc * 100.0,
            r.std_acc  * 100.0,
            r.mean_auroc,
            r.mean_f1,
            r.ci_lower * 100.0,
            r.ci_upper * 100.0,
        )
    }).collect();
    header + &body
}

fn main() -> Result<()> {
    println!("=== Model Comparison ===\n");

    let sources = [
        ("results/two_tower_1to1.json",     "Two-Tower 1:1"),
        ("results/two_tower_5to1.json",     "Two-Tower 5:1"),
        ("results/cross_encoder_1to1.json", "Cross-Encoder 1:1"),
        ("results/cross_encoder_5to1.json", "Cross-Encoder 5:1"),
    ];

    let mut rows: Vec<ModelRow> = Vec::new();
    for (path, label) in &sources {
        match load_row(path) {
            Some(row) => rows.push(row),
            None      => println!("  (skipping {}: file not found or invalid)", label),
        }
    }

    if rows.is_empty() {
        println!("No result files found. Run train_standard_mlp and/or train_cross_encoder first.");
        return Ok(());
    }

    let table = render_table(&rows);
    println!("{}", table);

    // Write machine-readable comparison
    std::fs::create_dir_all("results")?;
    let comparison: Vec<serde_json::Value> = rows.iter().map(|r| serde_json::json!({
        "model":              r.model,
        "neg_ratio":          r.neg_ratio,
        "mean_accuracy":      r.mean_acc,
        "std_accuracy":       r.std_acc,
        "mean_auroc":         r.mean_auroc,
        "mean_f1":            r.mean_f1,
        "ensemble_accuracy":  r.ensemble_acc,
        "bootstrap_ci_lower": r.ci_lower,
        "bootstrap_ci_upper": r.ci_upper,
    })).collect();
    std::fs::write(
        "results/model_comparison.json",
        serde_json::to_string_pretty(&comparison)?
    )?;
    println!("✓ Saved results/model_comparison.json");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_rendering() {
        let rows = vec![
            ModelRow {
                model:        "Two-Tower".to_string(),
                neg_ratio:    1,
                mean_acc:     0.8014,
                std_acc:      0.017,
                mean_auroc:   0.814,
                mean_f1:      0.839,
                ensemble_acc: 0.8306,
                ci_lower:     0.793,
                ci_upper:     0.810,
            },
        ];
        let table = render_table(&rows);
        assert!(table.contains("Two-Tower"), "table must contain model name");
        assert!(table.contains("1:1"),       "table must contain neg ratio");
        assert!(table.contains("80.14"),     "table must contain accuracy");
        assert!(table.contains("AUROC"),     "table must have AUROC header");
    }
}
