use anyhow::Result;
use serde_json::Value;

pub fn format_latex_table(rows: &[(String, String)], caption: &str, label: &str) -> String {
    let mut out = String::new();
    out.push_str("\\begin{table}[h]\n\\centering\n");
    out.push_str(&format!("\\caption{{{caption}}}\n\\label{{{label}}}\n"));
    out.push_str("\\begin{tabular}{lr}\n\\hline\n");
    out.push_str("\\textbf{Metric} & \\textbf{Value} \\\\\n\\hline\n");
    for (metric, value) in rows {
        out.push_str(&format!("{metric} & {value} \\\\\n"));
    }
    out.push_str("\\hline\n\\end{tabular}\n\\end{table}\n");
    out
}

fn load_json(path: &str) -> Result<Value> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}

fn main() -> Result<()> {
    println!("=== Generating LaTeX Tables ===\n");
    std::fs::create_dir_all("paper/tables")?;

    // Table 1: Main performance metrics
    let metrics = load_json("results/metrics.json")?;
    let rows = vec![
        ("Accuracy".to_string(),   format!("{:.4}", metrics["accuracy"].as_f64().unwrap_or(0.0))),
        ("Precision".to_string(),  format!("{:.4}", metrics["precision"].as_f64().unwrap_or(0.0))),
        ("Recall".to_string(),     format!("{:.4}", metrics["recall"].as_f64().unwrap_or(0.0))),
        ("F1 Score".to_string(),   format!("{:.4}", metrics.get("f1_score").or_else(|| metrics.get("f1")).and_then(|v| v.as_f64()).unwrap_or(0.0))),
        ("AUROC".to_string(),      format!("{:.4}", metrics["auroc"].as_f64().unwrap_or(0.0))),
        ("AUPRC".to_string(),      format!("{:.4}", metrics["auprc"].as_f64().unwrap_or(0.0))),
    ];
    let tex = format_latex_table(&rows, "Main Performance Metrics", "tab:main_results");
    std::fs::write("paper/tables/table1_main_results.tex", &tex)?;
    println!("✓ paper/tables/table1_main_results.tex");

    // Table 2: Seed robustness
    // Schema: {"accuracies": [f64, ...], "mean_accuracy": f64, ...}
    let seeds = load_json("results/seed_robustness.json")?;
    if let Some(arr) = seeds["accuracies"].as_array() {
        let mut seed_rows = Vec::new();
        for (i, acc_val) in arr.iter().enumerate() {
            let acc = acc_val.as_f64().unwrap_or(0.0);
            seed_rows.push((format!("Seed {}", i + 1), format!("{acc:.4}")));
        }
        let tex = format_latex_table(&seed_rows, "Seed Robustness", "tab:seeds");
        std::fs::write("paper/tables/table2_seed_robustness.tex", &tex)?;
        println!("✓ paper/tables/table2_seed_robustness.tex");
    }

    // Table 3: Ablation study
    // Schema: {"results": [{"name": str, "accuracy": f64, ...}, ...]}
    let ablation = load_json("results/ablation_study.json")?;
    if let Some(results) = ablation["results"].as_array() {
        let mut abl_rows = Vec::new();
        for v in results {
            let name = v["name"].as_str().unwrap_or("").to_string();
            let acc  = v["accuracy"].as_f64().unwrap_or(0.0);
            abl_rows.push((name, format!("{acc:.4}")));
        }
        let tex = format_latex_table(&abl_rows, "Ablation Study", "tab:ablation");
        std::fs::write("paper/tables/table3_ablation.tex", &tex)?;
        println!("✓ paper/tables/table3_ablation.tex");
    }

    println!("\nAll tables written to paper/tables/");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latex_table_format() {
        let rows = vec![
            ("Accuracy".to_string(), "0.8014".to_string()),
            ("Precision".to_string(), "0.7968".to_string()),
        ];
        let tex = format_latex_table(&rows, "Performance Metrics", "tab:perf");
        assert!(tex.contains(r"\begin{tabular}"), "missing tabular env");
        assert!(tex.contains("Accuracy"), "missing row data");
        assert!(tex.contains(r"\caption{Performance Metrics}"), "missing caption");
        assert!(tex.contains(r"\label{tab:perf}"), "missing label");
    }
}
