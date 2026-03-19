# Results Directory — Canonical File Reference

This file documents which JSON result files correspond to which tables/figures
in `paper/paper_tnnls.tex`. Multiple training runs with varying hyperparameters
exist; only files listed here are used in the paper.

## Canonical Result Files (Paper Tables)

| Paper Table/Section | File | Notes |
|---|---|---|
| Table 1 (balanced 1:1) | `cross_encoder_1to1.json`, `two_tower_1to1.json` | 5 seeds each, LR=0.001 |
| Table 2 (imbalance) | `cross_encoder_5to1.json`, `two_tower_5to1.json` | 5 seeds each, LR=0.001 |
| Table 3 (per-seed AUROC) | `cross_encoder_1to1.json`, `two_tower_1to1.json`, `ablation_no_product_1to1.json` | |
| Table 4 (ablation) | `ablation_no_product_1to1.json`, `ablation_no_product_5to1.json` | |
| Table 5 (TT pruning) | `neuron_pruning_multiseed.json` | 3 seeds |
| Table 6 (CE pruning) | `cross_encoder_pruning.json` | seed=42 only |
| Table 7 (temperature) | `temperature_search.json` | 3 seeds |
| Table 8 (baselines) | `classical_baselines.json` | expression-only, new |

## Non-Canonical Files (Earlier Experiments)

| File | Description |
|---|---|
| `standard_mlp_results.json` | Earlier single-model run (LR=0.005); AUROC ~0.801. **Not in paper.** |
| `ensemble_results.json` | Earlier ensemble (LR=0.005); accuracy 83.06%. **Not in paper.** |
| `model_comparison.json` | Earlier run with different hyperparameters. **Not in paper.** |
| `metrics.json`, `evaluation_metrics.json` | Preliminary evaluation files. |
| `advanced_training_log.txt`, `enhanced_features_log.txt` | Exploratory runs. |
| `gcan_results.json`, `target_95_results.json` | Experimental variants. |

## Key Hyperparameter Difference

The README/FINAL_SUMMARY cite 80.14%/83.06% results from an earlier
hyperparameter setting (LR=0.005 for two-tower). The paper uses LR=0.001
for both architectures (selected for cross-encoder stability), producing
different two-tower numbers (79.42%). Both sets are correct for their
respective hyperparameter choices; the paper uses the LR=0.001 setting
to enable a fair architectural comparison.
