# Module-Regularized GRN Inference

## Project Overview

A Rust-based machine learning system for predicting gene regulatory network (GRN) edges — i.e., which transcription factors (TFs) regulate which target genes — using learnable embeddings and single-cell RNA-seq expression features. Built from scratch without external ML frameworks.

## Key Architecture

- **Two-Tower MLP**: Separate TF and Gene encoder pathways (embedding + expression → FC layers → 512-dim encoding)
- **Scoring**: Cosine similarity with temperature scaling (τ=0.05) + sigmoid
- **Parameters**: ~5.2M total
- **Training**: Adam optimizer, LR=0.005, BCE loss, batch size 256, 50-60 epochs

## Current Results

- Single model: **80.14%** accuracy (±1.7%, 5 seeds)
- Ensemble (5 models): **83.06%** accuracy
- Improvement from baseline: **+20.83%** (62.23% → 83.06%)

## Essential Code Paths

### Core Model
- `src/models/hybrid_embeddings.rs` — Production two-tower model
- `src/models/nn.rs` — Neural network primitives (linear layers, activations, losses)
- `src/models/mod.rs` — Module exports

### Training
- `src/bin/train_standard_mlp.rs` — Main single-model training (80.14%)
- `src/bin/train_ensemble.rs` — Ensemble training (83.06%)

### Data Pipeline
- `src/data/priors.rs` — Prior knowledge loading (29K TF-gene pairs)
- `src/data/expression.rs` — Expression data from H5AD
- `src/data/dataloader.rs` — Batch loading
- `src/config.rs` — Configuration management

### Evaluation
- `src/bin/ablation_study.rs` — Ablation experiments
- `src/bin/seed_robustness.rs` — Multi-seed stability tests
- `src/bin/hyperparam_tuning.rs` — Hyperparameter search

### Output Generation
- `src/bin/generate_figures.rs` — Figure generation (SVG/PNG output)
- `src/bin/generate_tables.rs` — LaTeX table generation
- `src/bin/statistical_analysis.rs` — Bootstrap CIs, significance tests

## Build & Run

```bash
cargo build --release
cargo run --release --bin train_standard_mlp   # Single model
cargo run --release --bin train_ensemble        # Ensemble
```

## Data

- Expression data: Human brain single-cell RNA-seq (H5AD format) in `data/`
- Priors: Merged regulatory databases (DoRothEA/TRRUST), ~29K TF-gene pairs
- Dataset: ~39K examples, 50/50 positive/negative, 70/15/15 train/val/test split

## Conventions

- Pure Rust implementation (no PyTorch/TensorFlow dependency)
- ndarray for tensor operations
- Results stored as JSON in `results/`
- Figures generated via `cargo run --release --bin generate_figures` (SVG/PNG output)
- Tables generated via `cargo run --release --bin generate_tables` (LaTeX output)
- Statistical analysis via `cargo run --release --bin statistical_analysis`
- PDF manuscript via `scripts/make_pdf.sh` (requires pandoc)
- Historical status docs archived in `archive/`
