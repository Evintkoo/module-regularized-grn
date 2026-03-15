# Python Removal and Codebase Cleanup Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all Python from the project — delete inactive scripts, port active scripts to Rust, delete dead Rust bin targets, review core Rust for hygiene/performance, and update all markdown docs to reflect the cleaned state.

**Architecture:** Code-first execution: cleanup and porting happen before docs are updated, so docs reflect ground truth. Python manuscript scripts (figures, tables, stats) are ported to new Rust bins using `plotters` and `statrs`. Dead experimental Rust bins and model files are deleted.

**Tech Stack:** Rust, `plotters = "0.3"` (PNG/SVG figures), `statrs = "0.16"` (chi-squared CDF), `serde_json` (already present, reads `results/*.json`), `pandoc` CLI (replaces `make_pdf.py`)

**Spec:** `docs/superpowers/specs/2026-03-15-python-removal-and-cleanup-design.md`

---

## File Map

### Create
- `src/bin/generate_figures.rs` — ports `generate_all_figures.py` + `plot_results.py`; reads `results/*.json`, writes PNG/SVG to `figures/`
- `src/bin/generate_tables.rs` — ports `generate_tables.py`; reads `results/*.json`, writes `.tex` files to `paper/tables/`
- `src/bin/statistical_analysis.rs` — ports `statistical_analysis.py`; reads `results/predictions.json` + `results/evaluation_metrics.json`, writes `results/statistical_analysis.json`
- `scripts/make_pdf.sh` — replaces `make_pdf.py`; one-line pandoc call

### Modify
- `Cargo.toml` — remove 10 dead `[[bin]]` stanzas; add 3 missing stanzas (`ablation_study`, `seed_robustness`, `evaluate`); add `plotters` and `statrs` dependencies
- `src/models/mod.rs` — remove 11 dead `pub mod` + `pub use` entries; keep only `nn` and `hybrid_embeddings`
- `src/lib.rs` — remove `pub mod loss` and `pub mod training` if their modules become empty after cleanup
- `README.md` — remove Python reference, update project structure section
- `CLAUDE.md` — update conventions and code paths
- `plan.md` — update Phase 7 note, Key Design Decisions

### Delete — Python scripts
- Inactive (delete immediately): `scripts/comprehensive_improvements.py`, `scripts/convert_gene_ids.py`, `scripts/download_brain_data.py`, `scripts/feature_engineering.py`, `scripts/process_brain_data.py`, `scripts/process_h5ad.py`, `scripts/train_advanced.py`, `scripts/train_attention_model.py`
- Active (delete after porting): `scripts/generate_all_figures.py`, `scripts/plot_results.py`, `scripts/generate_tables.py`, `scripts/statistical_analysis.py`, `scripts/make_pdf.py`

### Delete — Rust bins (source files)
- Has stanza: `src/bin/train.rs`, `src/bin/train_example.rs`, `src/bin/train_priors.rs`, `src/bin/train_embeddings.rs`, `src/bin/train_embeddings_extended.rs`, `src/bin/phase1_expression.rs`, `src/bin/train_optimized.rs`, `src/bin/train_scaled_models.rs`, `src/bin/train_with_enhanced_features.rs`, `src/bin/train_cross_attention.rs`
- No stanza (file only): `src/bin/train_hybrid.rs`, `src/bin/train_hybrid_v2.rs`, `src/bin/train_advanced.rs`, `src/bin/train_scaled.rs`, `src/bin/train_medium.rs`, `src/bin/train_ultra.rs`, `src/bin/train_95_target.rs`, `src/bin/train_classifier_head.rs`

### Delete — Model source files (used only by deleted bins)
`src/models/embeddings.rs`, `src/models/two_tower.rs`, `src/models/baseline.rs`, `src/models/learnable_embeddings.rs`, `src/models/expression_model.rs`, `src/models/optimized_embeddings.rs`, `src/models/scalable_hybrid.rs`, `src/models/attention.rs`, `src/models/attention_model.rs`, `src/models/classifier_head.rs`, `src/models/cross_attention_model.rs`

### Delete — loss/training modules (only used by deleted bins)
`src/loss/contrastive.rs` (used only by deleted bins), `src/training/trainer.rs` (used only by `train.rs`, which is deleted)

---

## Chunk 1: Cleanup — Delete Dead Code and Update Cargo.toml

### Task 1: Commit current clean state

- [ ] Verify working tree is clean
  ```bash
  git status
  ```
  Expected: `nothing to commit, working tree clean`

- [ ] If dirty, stash or commit pending work before proceeding

---

### Task 2: Delete inactive Python scripts

- [ ] Delete the 8 inactive scripts
  ```bash
  rm scripts/comprehensive_improvements.py \
     scripts/convert_gene_ids.py \
     scripts/download_brain_data.py \
     scripts/feature_engineering.py \
     scripts/process_brain_data.py \
     scripts/process_h5ad.py \
     scripts/train_advanced.py \
     scripts/train_attention_model.py
  ```

- [ ] Verify 5 active scripts remain
  ```bash
  ls scripts/
  ```
  Expected: `generate_all_figures.py  generate_tables.py  make_pdf.py  plot_results.py  statistical_analysis.py`

- [ ] Commit
  ```bash
  git add scripts/
  git commit -m "chore: delete inactive Python scripts"
  ```

---

### Task 3: Delete dead Rust bin source files and update Cargo.toml stanzas

- [ ] Delete all dead bin source files (both stanzaed and stanza-free)
  ```bash
  rm src/bin/train.rs \
     src/bin/train_example.rs \
     src/bin/train_priors.rs \
     src/bin/train_embeddings.rs \
     src/bin/train_embeddings_extended.rs \
     src/bin/phase1_expression.rs \
     src/bin/train_optimized.rs \
     src/bin/train_scaled_models.rs \
     src/bin/train_with_enhanced_features.rs \
     src/bin/train_cross_attention.rs \
     src/bin/train_hybrid.rs \
     src/bin/train_hybrid_v2.rs \
     src/bin/train_advanced.rs \
     src/bin/train_scaled.rs \
     src/bin/train_medium.rs \
     src/bin/train_ultra.rs \
     src/bin/train_95_target.rs \
     src/bin/train_classifier_head.rs
  ```

- [ ] Update `Cargo.toml`: remove the 10 dead `[[bin]]` stanzas and add 3 missing ones.

  The `[[bin]]` section of `Cargo.toml` should become exactly:
  ```toml
  [[bin]]
  name = "download_priors"
  path = "src/bin/download_priors.rs"

  [[bin]]
  name = "process_data"
  path = "src/bin/process_data.rs"

  [[bin]]
  name = "train_standard_mlp"
  path = "src/bin/train_standard_mlp.rs"

  [[bin]]
  name = "hyperparam_tuning"
  path = "src/bin/hyperparam_tuning.rs"

  [[bin]]
  name = "train_ensemble"
  path = "src/bin/train_ensemble.rs"

  [[bin]]
  name = "ablation_study"
  path = "src/bin/ablation_study.rs"

  [[bin]]
  name = "seed_robustness"
  path = "src/bin/seed_robustness.rs"

  [[bin]]
  name = "evaluate"
  path = "src/bin/evaluate.rs"
  ```

- [ ] Verify build succeeds with the cleaned bin list
  ```bash
  cargo build --release 2>&1 | head -40
  ```
  Expected: `Finished release` with no errors. Warnings are acceptable at this stage.

- [ ] Commit
  ```bash
  git add Cargo.toml src/bin/
  git commit -m "chore: delete dead Rust bin targets, sync Cargo.toml stanzas"
  ```

---

### Task 4: Delete dead model source files and clean models/mod.rs

- [ ] Verify no kept bin imports the models about to be deleted
  ```bash
  grep -rn "embeddings\|two_tower\|baseline\|learnable\|expression_model\|optimized\|scalable\|attention\|classifier_head\|cross_attention" \
    src/bin/train_standard_mlp.rs src/bin/train_ensemble.rs src/bin/ablation_study.rs \
    src/bin/seed_robustness.rs src/bin/hyperparam_tuning.rs src/bin/evaluate.rs \
    src/bin/process_data.rs src/bin/download_priors.rs
  ```
  Expected: no output (none of the kept bins use these models)

- [ ] Delete the 11 dead model files
  ```bash
  rm src/models/embeddings.rs \
     src/models/two_tower.rs \
     src/models/baseline.rs \
     src/models/learnable_embeddings.rs \
     src/models/expression_model.rs \
     src/models/optimized_embeddings.rs \
     src/models/scalable_hybrid.rs \
     src/models/attention.rs \
     src/models/attention_model.rs \
     src/models/classifier_head.rs \
     src/models/cross_attention_model.rs
  ```

- [ ] Replace `src/models/mod.rs` to keep only `nn` and `hybrid_embeddings`:
  ```rust
  pub mod nn;
  pub mod hybrid_embeddings;

  pub use nn::{
      LinearLayer, Dropout,
      relu, relu_backward,
      sigmoid, sigmoid_backward,
      bce_loss, bce_loss_backward,
  };
  pub use hybrid_embeddings::HybridEmbeddingModel;
  ```

- [ ] Verify build still passes
  ```bash
  cargo build --release 2>&1 | head -40
  ```
  Expected: `Finished release` with no errors.

- [ ] Commit
  ```bash
  git add src/models/
  git commit -m "chore: delete dead model files, clean models/mod.rs"
  ```

---

### Task 5: Clean loss/ and training/ modules

- [ ] Confirm `loss::Metrics` / `compute_metrics` are not imported by kept bins through the `pub use loss::*` wildcard in `src/lib.rs`:
  ```bash
  grep -rn "Metrics\b\|compute_metrics" \
    src/bin/train_standard_mlp.rs src/bin/train_ensemble.rs src/bin/ablation_study.rs \
    src/bin/seed_robustness.rs src/bin/hyperparam_tuning.rs src/bin/evaluate.rs \
    src/bin/process_data.rs src/bin/download_priors.rs
  ```
  Expected: any `Metrics` hits are `EvaluationMetrics` from `evaluation::`, not `loss::Metrics`. No `compute_metrics` hits.

- [ ] Confirm `contrastive`, `Trainer`, `infonce`, `optimizer`, `checkpoint` are not used by kept bins:
  ```bash
  grep -rn "contrastive\|Trainer\b\|infonce\|Optimizer\b\|Checkpoint\b" \
    src/bin/train_standard_mlp.rs src/bin/train_ensemble.rs src/bin/ablation_study.rs \
    src/bin/seed_robustness.rs src/bin/hyperparam_tuning.rs src/bin/evaluate.rs \
    src/bin/process_data.rs src/bin/download_priors.rs
  ```
  Expected: no output

- [ ] Delete all files in `src/loss/` and `src/training/` except their `mod.rs` files:
  ```bash
  rm src/loss/contrastive.rs src/loss/metrics.rs
  rm src/training/trainer.rs src/training/optimizer.rs src/training/checkpoint.rs
  ```
  (If any of these files don't exist, skip that rm; do not fail)

- [ ] Verify `src/evaluation/` exists before rewriting `lib.rs`:
  ```bash
  ls src/evaluation/ 2>/dev/null && echo "evaluation module exists" || echo "evaluation module does NOT exist"
  ```

- [ ] Remove `pub mod loss` and `pub use loss::*` from `src/lib.rs`, and `pub mod training` and `pub use training::*`. If `src/evaluation/` exists, the final `src/lib.rs` should be:
  ```rust
  pub mod config;
  pub mod data;
  pub mod models;
  pub mod evaluation;

  pub use config::*;
  pub use data::*;
  pub use models::*;
  pub use evaluation::*;
  ```
  If `src/evaluation/` does NOT exist, omit `pub mod evaluation` and `pub use evaluation::*`.

- [ ] Delete the now-empty `mod.rs` files and their parent directories if truly empty, OR leave them as empty `mod.rs` files — either is fine; but do NOT leave them with dead `pub mod` entries pointing to deleted files.

- [ ] Verify build:
  ```bash
  cargo build --release 2>&1 | grep -E "^error|Finished"
  ```
  Expected: `Finished release` with no errors

- [ ] Commit
  ```bash
  git add src/loss/ src/training/ src/lib.rs
  git commit -m "chore: remove dead loss and training modules"
  ```

---

## Chunk 2: Port Active Python Scripts to Rust

### Task 6: Add new crate dependencies

- [ ] Add `plotters` and `statrs` to `Cargo.toml` dependencies section:
  ```toml
  plotters = { version = "0.3", default-features = false, features = ["svg_backend", "line_series", "area_series"] }
  statrs = "0.17"
  ```
  Note: SVG-only output (no `bitmap_backend`/`bitmap_encoder` needed — we produce SVG not PNG). Use `statrs = "0.17"` (0.16 is not published; 0.17 is latest stable with identical API).

- [ ] Verify the crates resolve
  ```bash
  cargo fetch 2>&1 | tail -5
  ```
  Expected: no errors

- [ ] Commit
  ```bash
  git add Cargo.toml
  git commit -m "chore: add plotters and statrs dependencies"
  ```

---

### Task 7: Implement statistical_analysis bin

**Files:** Create `src/bin/statistical_analysis.rs`, Test: `cargo run --release --bin statistical_analysis`

The bin reads `results/evaluation_metrics.json` and `results/predictions.json`, performs bootstrap CI and McNemar's test, writes `results/statistical_analysis.json`.

- [ ] Write a failing test (as a `#[test]` in the file itself) for the bootstrap CI function:

  Add to top of `src/bin/statistical_analysis.rs`:
  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_bootstrap_ci_known_data() {
          // All predictions correct → accuracy = 1.0, CI should be tight around 1.0
          let y_true = vec![1.0f64, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
          let y_pred = vec![0.9f64, 0.1, 0.8, 0.2, 0.7, 0.3, 0.9, 0.1];
          let (mean, lower, upper) = bootstrap_ci_accuracy(&y_true, &y_pred, 500, 42);
          assert!((mean - 1.0).abs() < 0.01, "mean should be ~1.0, got {mean}");
          assert!(lower > 0.9, "lower CI should be > 0.9, got {lower}");
          assert!(upper <= 1.0, "upper CI should be <= 1.0, got {upper}");
      }

      #[test]
      fn test_mcnemar_test_significant() {
          // Model clearly better than baseline → p < 0.05
          let y_true:  Vec<f64> = vec![1.0; 50].into_iter().chain(vec![0.0; 50]).collect();
          let baseline: Vec<f64> = vec![0.6; 50].into_iter().chain(vec![0.6; 50]).collect(); // ~50% acc
          let model:    Vec<f64> = vec![0.9; 50].into_iter().chain(vec![0.1; 50]).collect(); // ~100% acc
          let (stat, p) = mcnemar_test(&baseline, &model, &y_true);
          assert!(p < 0.05, "should be significant, got p={p}, stat={stat}");
      }
  }
  ```

- [ ] Run the test to verify it fails (function not defined yet):
  ```bash
  cargo test --bin statistical_analysis 2>&1 | head -20
  ```
  Expected: compile error — `bootstrap_ci_accuracy` and `mcnemar_test` not defined

- [ ] Implement `src/bin/statistical_analysis.rs`:

  ```rust
  use anyhow::Result;
  use serde::{Deserialize, Serialize};
  use serde_json::Value;
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

      // Bootstrap CI on accuracy
      let (acc_mean, acc_lower, acc_upper) = bootstrap_ci_accuracy(y_true, y_pred, 1000, 42);
      println!("Accuracy CI: {acc_mean:.4} [{acc_lower:.4}, {acc_upper:.4}]");

      // Baseline: random (0.5 for all)
      let baseline_preds: Vec<f64> = vec![0.5; y_pred.len()];
      let (stat, p) = mcnemar_test(&baseline_preds, y_pred, y_true);
      println!("McNemar test vs random: statistic={stat:.4}, p={p:.6}");

      // Error by confidence
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

      // Write results
      // Note: AUROC bootstrap CI is not computed here (requires AUROC metric function).
      // The auroc field is intentionally omitted from BootstrapCI — downstream generate_figures
      // reads AUROC CI from evaluation_metrics.json directly.
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
      // (test code from above)
  }
  ```

- [ ] Run tests
  ```bash
  cargo test --bin statistical_analysis 2>&1
  ```
  Expected: `test tests::test_bootstrap_ci_known_data ... ok` and `test tests::test_mcnemar_test_significant ... ok`

- [ ] Add `[[bin]]` stanza for `statistical_analysis` to `Cargo.toml` (before building):
  ```toml
  [[bin]]
  name = "statistical_analysis"
  path = "src/bin/statistical_analysis.rs"
  ```

- [ ] Build the bin
  ```bash
  cargo build --release --bin statistical_analysis 2>&1 | grep -E "^error|Finished"
  ```
  Expected: `Finished release`

- [ ] Commit
  ```bash
  git add src/bin/statistical_analysis.rs Cargo.toml
  git commit -m "feat: add statistical_analysis Rust bin (ports statistical_analysis.py)"
  ```

---

### Task 8: Implement generate_tables bin

**Files:** Create `src/bin/generate_tables.rs`

- [ ] Write the failing test first:
  ```rust
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
  ```

- [ ] Run to confirm it fails:
  ```bash
  cargo test --bin generate_tables 2>&1 | head -10
  ```
  Expected: compile error

- [ ] Implement `src/bin/generate_tables.rs`:

  ```rust
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
      // Schema: {"accuracies": [f64, ...], "seeds": [...], "mean_accuracy": f64, ...}
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
      // Schema: {"results": [{"name": str, "accuracy": f64, ...}, ...], "baseline": ..., "best": ...}
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
  ```

- [ ] Run tests
  ```bash
  cargo test --bin generate_tables 2>&1
  ```
  Expected: `test tests::test_latex_table_format ... ok`

- [ ] Add `[[bin]]` stanza to `Cargo.toml` for `generate_tables` (statistical_analysis stanza was added in Task 7):
  ```toml
  [[bin]]
  name = "generate_tables"
  path = "src/bin/generate_tables.rs"
  ```

- [ ] Build
  ```bash
  cargo build --release --bin generate_tables 2>&1 | grep -E "^error|Finished"
  ```
  Expected: `Finished release`

- [ ] Commit
  ```bash
  git add src/bin/generate_tables.rs Cargo.toml
  git commit -m "feat: add generate_tables Rust bin (ports generate_tables.py)"
  ```

---

### Task 9: Implement generate_figures bin

**Files:** Create `src/bin/generate_figures.rs`

This ports `generate_all_figures.py` and `plot_results.py`. Outputs SVG to `figures/`. Implements 4 core figures: ROC curve, PR curve, ablation study bars, seed robustness line. (The Python original has 8 figures; confusion matrix, performance comparison, bootstrap distributions, and architecture diagram are out of scope for this port — the remaining 4 require more complex plotters APIs and are lower priority.)

> **Note:** `plotters` produces PNG and SVG. PDF output from Python is not reproduced. For journal submission requiring PDF, use `rsvg-convert figures/figN.svg -o figures/figN.pdf` as a post-process step.

- [ ] Write a failing test for the ROC curve data computation:

  ```rust
  #[cfg(test)]
  mod tests {
      use super::*;

      #[test]
      fn test_roc_curve_perfect_classifier() {
          let y_true = vec![0.0f64, 0.0, 1.0, 1.0];
          let y_pred = vec![0.1f64, 0.2, 0.8, 0.9];
          let (fpr, tpr, auc) = compute_roc_curve(&y_true, &y_pred);
          assert!((auc - 1.0).abs() < 1e-6, "AUC should be 1.0 for perfect classifier, got {auc}");
          assert_eq!(fpr.first(), Some(&0.0));
          assert_eq!(tpr.last(), Some(&1.0));
      }

      #[test]
      fn test_roc_curve_below_chance() {
          // Inverse classifier (always wrong): AUC < 0.5
          let y_true = vec![0.0f64, 0.0, 1.0, 1.0];
          let y_pred = vec![0.9f64, 0.8, 0.2, 0.1]; // high scores for negatives
          let (_, _, auc) = compute_roc_curve(&y_true, &y_pred);
          assert!(auc < 0.5, "AUC should be < 0.5 for inverse classifier, got {auc}");
      }
  }
  ```
  Note: avoid all-equal-score tests for AUC — tie-breaking is sort-order-dependent.

- [ ] Run to confirm compile failure:
  ```bash
  cargo test --bin generate_figures 2>&1 | head -10
  ```

- [ ] Implement `src/bin/generate_figures.rs`. Key helper functions that must be tested:

  ```rust
  use anyhow::Result;
  use plotters::prelude::*;
  use serde::{Deserialize, Serialize};
  use serde_json::Value;
  use std::path::Path;

  #[derive(Deserialize)]
  struct Predictions {
      labels: Vec<f64>,
      predictions: Vec<f64>,
  }

  /// Compute ROC curve points and AUROC. Returns (fpr, tpr, auc).
  pub fn compute_roc_curve(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
      // Collect (score, label) and sort descending by score
      let mut pairs: Vec<(f64, f64)> = y_pred.iter().zip(y_true.iter()).map(|(&p, &t)| (p, t)).collect();
      pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

      let n_pos = y_true.iter().filter(|&&v| v > 0.5).count() as f64;
      let n_neg = y_true.len() as f64 - n_pos;

      let mut fpr = vec![0.0f64];
      let mut tpr = vec![0.0f64];
      let mut tp = 0.0f64;
      let mut fp = 0.0f64;

      for (_, label) in &pairs {
          if *label > 0.5 { tp += 1.0; } else { fp += 1.0; }
          fpr.push(fp / n_neg);
          tpr.push(tp / n_pos);
      }

      // Trapezoidal AUC
      let auc: f64 = fpr.windows(2).zip(tpr.windows(2))
          .map(|(f, t)| (f[1] - f[0]) * (t[0] + t[1]) / 2.0)
          .sum();

      (fpr, tpr, auc)
  }

  /// Compute precision-recall curve. Returns (recall, precision, average_precision).
  pub fn compute_pr_curve(y_true: &[f64], y_pred: &[f64]) -> (Vec<f64>, Vec<f64>, f64) {
      let mut pairs: Vec<(f64, f64)> = y_pred.iter().zip(y_true.iter()).map(|(&p, &t)| (p, t)).collect();
      pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

      let mut recall = vec![];
      let mut precision = vec![];
      let mut tp = 0.0f64;
      let mut fp = 0.0f64;
      let n_pos = y_true.iter().filter(|&&v| v > 0.5).count() as f64;

      for (_, label) in &pairs {
          if *label > 0.5 { tp += 1.0; } else { fp += 1.0; }
          recall.push(tp / n_pos);
          precision.push(tp / (tp + fp));
      }

      // Average precision: area under PR curve
      let ap: f64 = recall.windows(2).zip(precision.windows(2))
          .map(|(r, p)| (r[1] - r[0]) * (p[0] + p[1]) / 2.0)
          .sum();

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
      chart.configure_mesh()
          .x_desc("False Positive Rate")
          .y_desc("True Positive Rate")
          .draw()?;
      // Model curve
      chart.draw_series(LineSeries::new(
          fpr.iter().zip(tpr.iter()).map(|(&x, &y)| (x, y)),
          &RGBColor(46, 134, 171),
      ))?.label(format!("Model (AUC = {auc:.4})"))
          .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(46, 134, 171)));
      // Diagonal
      chart.draw_series(LineSeries::new(
          vec![(0.0, 0.0), (1.0, 1.0)],
          &BLACK.mix(0.4),
      ))?.label("Random (AUC = 0.5000)")
          .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK.mix(0.4)));
      chart.configure_series_labels().border_style(&BLACK).draw()?;
      root.present()?;
      Ok(())
  }

  fn draw_pr_curve(recall: &[f64], precision: &[f64], ap: f64, baseline: f64, out_path: &str) -> Result<()> {
      let root = SVGBackend::new(out_path, (600, 600)).into_drawing_area();
      root.fill(&WHITE)?;
      let mut chart = ChartBuilder::on(&root)
          .caption("Precision-Recall Curve", ("sans-serif", 20).into_font())
          .margin(30)
          .x_label_area_size(40)
          .y_label_area_size(40)
          .build_cartesian_2d(0f64..1f64, 0f64..1f64)?;
      chart.configure_mesh().x_desc("Recall").y_desc("Precision").draw()?;
      chart.draw_series(LineSeries::new(
          recall.iter().zip(precision.iter()).map(|(&r, &p)| (r, p)),
          &RGBColor(162, 59, 114),
      ))?.label(format!("Model (AP = {ap:.4})"))
          .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(162, 59, 114)));
      chart.draw_series(LineSeries::new(
          vec![(0.0, baseline), (1.0, baseline)],
          &BLACK.mix(0.4),
      ))?.label(format!("Random (AP = {baseline:.4})"))
          .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLACK.mix(0.4)));
      chart.configure_series_labels().border_style(&BLACK).draw()?;
      root.present()?;
      Ok(())
  }

  fn draw_ablation_bars(variants: &[(String, f64)], out_path: &str) -> Result<()> {
      let root = SVGBackend::new(out_path, (700, 450)).into_drawing_area();
      root.fill(&WHITE)?;
      let max_acc = variants.iter().map(|(_, a)| *a).fold(0.0f64, f64::max).max(1.0);
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
      let min_acc = accuracies.iter().cloned().fold(f64::INFINITY, f64::min) - 0.05;
      let max_acc = accuracies.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.05;
      let mut chart = ChartBuilder::on(&root)
          .caption("Seed Robustness", ("sans-serif", 20).into_font())
          .margin(30)
          .x_label_area_size(40)
          .y_label_area_size(50)
          .build_cartesian_2d(0f64..(accuracies.len() as f64), min_acc..max_acc)?;
      chart.configure_mesh().x_desc("Seed").y_desc("Accuracy").draw()?;
      chart.draw_series(LineSeries::new(
          accuracies.iter().enumerate().map(|(i, &a)| (i as f64 + 0.5, a)),
          &RGBColor(70, 130, 180),
      ))?;
      root.present()?;
      Ok(())
  }

  fn main() -> Result<()> {
      println!("=== Generating Figures ===\n");
      std::fs::create_dir_all("figures")?;

      // Load data
      let preds: Predictions = serde_json::from_str(&std::fs::read_to_string("results/predictions.json")?)?;
      let y_true = &preds.labels;
      let y_pred = &preds.predictions;
      let baseline_ratio = y_true.iter().filter(|&&v| v > 0.5).count() as f64 / y_true.len() as f64;

      // Figure 1: ROC
      let (fpr, tpr, auc) = compute_roc_curve(y_true, y_pred);
      draw_roc_curve(&fpr, &tpr, auc, "figures/figure1_roc_curve.svg")?;
      println!("✓ figures/figure1_roc_curve.svg (AUC={auc:.4})");

      // Figure 2: PR Curve
      let (recall, precision, ap) = compute_pr_curve(y_true, y_pred);
      draw_pr_curve(&recall, &precision, ap, baseline_ratio, "figures/figure2_pr_curve.svg")?;
      println!("✓ figures/figure2_pr_curve.svg (AP={ap:.4})");

      // Figure 3: Ablation
      // Schema: {"results": [{"name": str, "accuracy": f64}, ...]}
      if let Ok(abl_json) = std::fs::read_to_string("results/ablation_study.json") {
          let ablation: Value = serde_json::from_str(&abl_json)?;
          if let Some(results) = ablation["results"].as_array() {
              let data: Vec<(String, f64)> = results.iter()
                  .map(|v| (v["name"].as_str().unwrap_or("").to_string(), v["accuracy"].as_f64().unwrap_or(0.0)))
                  .collect();
              draw_ablation_bars(&data, "figures/figure3_ablation.svg")?;
              println!("✓ figures/figure3_ablation.svg");
          }
      }

      // Figure 4: Seed robustness
      // Schema: {"accuracies": [f64, ...], "mean_accuracy": f64, ...}
      if let Ok(seed_json) = std::fs::read_to_string("results/seed_robustness.json") {
          let seeds: Value = serde_json::from_str(&seed_json)?;
          if let Some(arr) = seeds["accuracies"].as_array() {
              let accs: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
              draw_seed_robustness(&accs, "figures/figure4_seed_robustness.svg")?;
              println!("✓ figures/figure4_seed_robustness.svg");
          }
      }

      println!("\nAll figures written to figures/ (SVG format)");
      println!("Note: For PDF output, use: rsvg-convert figures/figN.svg -o figures/figN.pdf");
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
  ```

- [ ] Add `[[bin]]` stanza for `generate_figures` to `Cargo.toml`

- [ ] Run tests
  ```bash
  cargo test --bin generate_figures 2>&1
  ```
  Expected: all 3 tests pass

- [ ] Build
  ```bash
  cargo build --release --bin generate_figures 2>&1 | grep -E "^error|Finished"
  ```
  Expected: `Finished release`

- [ ] Commit
  ```bash
  git add src/bin/generate_figures.rs Cargo.toml
  git commit -m "feat: add generate_figures Rust bin (ports generate_all_figures.py)"
  ```

---

### Task 10: Replace make_pdf.py with a shell script

- [ ] Create `scripts/make_pdf.sh`:
  ```bash
  #!/bin/bash
  # Convert paper/manuscript.md to PDF using pandoc.
  # Requires:
  #   - pandoc:   brew install pandoc
  #   - TeX Live: brew install --cask mactex  (or brew install basictex)
  # Uses pdflatex (default, widely available). For Unicode/custom font support use --pdf-engine=xelatex.
  set -e
  pandoc paper/manuscript.md \
      --output paper/manuscript.pdf \
      --pdf-engine=pdflatex \
      --variable geometry:margin=1in \
      --variable fontsize=11pt
  echo "PDF created: paper/manuscript.pdf"
  ```

- [ ] Make it executable:
  ```bash
  chmod +x scripts/make_pdf.sh
  ```

- [ ] Delete the active Python scripts (now all ported):
  ```bash
  rm scripts/generate_all_figures.py \
     scripts/plot_results.py \
     scripts/generate_tables.py \
     scripts/statistical_analysis.py \
     scripts/make_pdf.py
  ```

- [ ] Verify scripts/ directory contains only the shell script:
  ```bash
  ls scripts/
  ```
  Expected: `make_pdf.sh`

- [ ] Run a final full build to confirm nothing broke:
  ```bash
  cargo build --release 2>&1 | grep -E "^error|Finished"
  ```
  Expected: `Finished release`

- [ ] Commit
  ```bash
  git add scripts/
  git commit -m "feat: replace make_pdf.py with pandoc shell script, delete all Python"
  ```

---

## Chunk 3: Core Rust Hygiene and Performance Review

### Task 11: Eliminate compiler warnings

- [ ] Run build and capture all warnings:
  ```bash
  cargo build --release 2>&1 | grep "^warning" | sort | uniq -c | sort -rn
  ```

- [ ] For each `unused import` warning: remove the import from the referenced file
- [ ] For each `dead_code` warning in a file that is NOT a public library item: add `#[allow(dead_code)]` only if the code is intentionally kept, otherwise delete it
- [ ] Re-run build until zero warnings:
  ```bash
  cargo build --release 2>&1 | grep -c "^warning"
  ```
  Expected: `0`

- [ ] Commit
  ```bash
  git add src/
  git commit -m "chore: eliminate all compiler warnings"
  ```

---

### Task 12: Review dataloader for per-epoch re-reads

- [ ] Find how expression data and priors are loaded in the actual codebase:
  ```bash
  grep -n "GRNDataset\|new.*dataset\|create_dataset\|PriorDataset\|ExpressionData\|\.load(" \
    src/data/dataloader.rs src/data/expression.rs src/data/priors.rs | head -30
  ```

- [ ] Check if the dataset construction occurs inside or outside the training loop in `train_standard_mlp.rs`:
  ```bash
  grep -n "GRNDataset\|PriorDataset\|ExpressionData\|\.load\|new(" src/bin/train_standard_mlp.rs | head -30
  ```
  Then read the surrounding context for any hit that appears inside a `for epoch` block.

- [ ] If data loading is inside the epoch loop: move it before the loop starts
  - Pattern to fix: any dataset construction call inside `for epoch in 0..N`
  - Fix: hoist the call to before the loop

- [ ] If already loaded once (no issue), add a comment `// Loaded once before training loop` at the call site and proceed

- [ ] Run build to confirm no breakage:
  ```bash
  cargo build --release --bin train_standard_mlp 2>&1 | grep -E "^error|Finished"
  ```

- [ ] Commit if changes were made:
  ```bash
  git add src/
  git commit -m "perf: ensure expression data is loaded once, not per epoch"
  ```

---

### Task 13: Check forward pass for unnecessary allocations and fix relu_backward bug

- [ ] **CRITICAL: Fix the `relu_backward` argument-order bug in `hybrid_embeddings.rs`**

  The function signature in `nn.rs` is `relu_backward(x: &Array2<f32>, grad_output: &Array2<f32>)` where `x` is the *pre-activation input* (used to compute the mask) and `grad_output` is the upstream gradient. Verify the bug and fix it:

  ```bash
  grep -n "relu_backward" src/models/hybrid_embeddings.rs
  ```

  Expected buggy pattern (arguments swapped — gradient passed as `x`, pre-activation as `grad_output`):
  ```rust
  let grad_gene_h1_pre = relu_backward(&grad_gene_h1, gene_h1);  // WRONG: swapped
  let grad_tf_h1_pre   = relu_backward(&grad_tf_h1,   tf_h1);    // WRONG: swapped
  ```

  The authoritative signature (from `src/models/nn.rs`):
  ```rust
  pub fn relu_backward(x: &Array2<f32>, grad_output: &Array2<f32>) -> Array2<f32>
  // x           = pre-activation input (used to compute mask: where x > 0)
  // grad_output = upstream gradient (the thing being multiplied by the mask)
  ```

  Correct calls:
  ```rust
  let grad_gene_h1_pre = relu_backward(gene_h1, &grad_gene_h1);  // pre-activation first, gradient second
  let grad_tf_h1_pre   = relu_backward(tf_h1,   &grad_tf_h1);
  ```

  Fix by swapping the arguments. Do NOT use `optimized_embeddings.rs` or `baseline.rs` as reference — those files are deleted in Task 4.

- [ ] Check `nn.rs` for `.clone()` calls inside `forward()`:
  ```bash
  grep -n "\.clone()\|Vec::new()\|to_vec()\|to_owned()" src/models/nn.rs | head -20
  ```

  **Note:** `LinearLayer::forward()` calls `self.input_cache = Some(x.clone())`. This clone is **intentional and load-bearing** — it caches the pre-activation input needed by the backward pass. Do NOT remove it.

- [ ] Check `src/models/hybrid_embeddings.rs` for avoidable allocations:
  ```bash
  grep -n "\.clone()\|Vec::new()\|to_vec()" src/models/hybrid_embeddings.rs | head -20
  ```
  For each `.clone()` not related to the input cache: check if a `view()` or borrow would suffice.

- [ ] Check rayon parallelism:
  ```bash
  grep -n "rayon\|par_iter\|into_par_iter" src/models/hybrid_embeddings.rs src/bin/train_standard_mlp.rs
  ```
  If `par_iter()` is absent but the batch loop iterates >1 item independently, add `use rayon::prelude::*` and change the iterator.

- [ ] Make targeted fixes only — do not refactor entire files. One fix per call site.

- [ ] Run build and all tests:
  ```bash
  cargo build --release 2>&1 | grep -E "^error|Finished"
  cargo test 2>&1 | tail -10
  ```

- [ ] Commit
  ```bash
  git add src/
  git commit -m "perf: reduce unnecessary allocations in forward pass"
  ```

---

## Chunk 4: Markdown Updates

### Task 14: Update README.md

- [ ] In `README.md`, replace the project structure section. The `scripts/` entry changes from:
  ```
  scripts/                       # Python analysis & figure generation
  ```
  to:
  ```
  scripts/                       # Shell scripts (make_pdf.sh)
  ```
  And add the new bins to the `src/bin/` listing:
  ```
  │   ├── generate_figures.rs    # Figure generation (SVG/PNG)
  │   ├── generate_tables.rs     # LaTeX table generation
  │   ├── statistical_analysis.rs # Bootstrap CIs, significance tests
  ```

- [ ] Update the `figures/` entry in the project structure from `# Publication figures (PDF/PNG)` to `# Publication figures (SVG/PNG)`

- [ ] Remove the Python reference in the Dependencies section (there is none; verify and move on)

- [ ] Verify no remaining Python references:
  ```bash
  grep -n -i "python\|\.py" README.md
  ```
  Expected: no output

- [ ] Commit
  ```bash
  git add README.md
  git commit -m "docs: update README to reflect pure-Rust project, add new bins"
  ```

---

### Task 15: Update CLAUDE.md

- [ ] In `CLAUDE.md`, update the Essential Code Paths bin list to match the 11 kept/new bins:
  - Keep: `train_standard_mlp.rs`, `train_ensemble.rs`, `ablation_study.rs`, `seed_robustness.rs`, `hyperparam_tuning.rs`, `evaluate.rs`, `process_data.rs`, `download_priors.rs`
  - Add: `generate_figures.rs`, `generate_tables.rs`, `statistical_analysis.rs`

- [ ] In the Conventions section, replace:
  ```
  - Figures generated via Python scripts in `scripts/`
  ```
  with:
  ```
  - Figures generated via `cargo run --release --bin generate_figures` (SVG/PNG output)
  - Tables generated via `cargo run --release --bin generate_tables` (LaTeX output)
  - Statistical analysis via `cargo run --release --bin statistical_analysis`
  - PDF manuscript via `scripts/make_pdf.sh` (requires pandoc)
  ```

- [ ] Verify no remaining Python references:
  ```bash
  grep -n -i "python\|\.py" CLAUDE.md
  ```
  Expected: no output

- [ ] Commit
  ```bash
  git add CLAUDE.md
  git commit -m "docs: update CLAUDE.md for pure-Rust project"
  ```

---

### Task 16: Update plan.md

- [ ] In `plan.md`, under Phase 7, add a note:
  ```
  - **Figures now generated in Rust**: `generate_figures`, `generate_tables`, `statistical_analysis` bins replace Python scripts
  ```

- [ ] In the Key Design Decisions section, update decision #1 to:
  ```
  1. **Pure Rust** — No Python or PyTorch dependency; all NN operations, figure generation, table output, and statistical analysis from scratch using ndarray and plotters
  ```

- [ ] Verify no remaining Python references in plan.md:
  ```bash
  grep -n -i "python\|\.py" plan.md
  ```
  Expected: no output (or only historical references in past phase descriptions — those are acceptable context)

- [ ] Commit
  ```bash
  git add plan.md
  git commit -m "docs: update plan.md to reflect pure-Rust stack"
  ```

---

### Task 17: Final verification

- [ ] Full build clean:
  ```bash
  cargo build --release 2>&1 | grep -E "^error|^warning|Finished"
  ```
  Expected: `Finished release` with zero errors and zero warnings

- [ ] Confirm scripts/ contains only the shell script:
  ```bash
  ls scripts/
  ```
  Expected: `make_pdf.sh`

- [ ] Confirm no Python references remain in key docs:
  ```bash
  grep -rn "python\|\.py" README.md CLAUDE.md plan.md
  ```
  Expected: no output

- [ ] Run all tests:
  ```bash
  cargo test 2>&1 | tail -5
  ```
  Expected: `test result: ok`

- [ ] Smoke-test the three new bins compile cleanly:
  ```bash
  cargo build --release --bin generate_figures 2>&1 | grep -E "^error|Finished"
  cargo build --release --bin generate_tables 2>&1 | grep -E "^error|Finished"
  cargo build --release --bin statistical_analysis 2>&1 | grep -E "^error|Finished"
  ```
  Expected: `Finished release` for each

- [ ] Final commit (if any remaining unstaged changes — should be clean at this point):
  ```bash
  git status
  ```
  If clean: no commit needed. If dirty: stage only the specific changed files (do not use `git add -A`).
