# Module-Regularized Gene Regulatory Network Inference

A pure-Rust system for inferring gene regulatory relationships from single-cell RNA-seq data using a two-tower MLP architecture with learnable embeddings.

## Motivation

Predicting which transcription factors (TFs) regulate which target genes is fundamental to understanding gene regulation. Existing methods either require expensive GPU compute (transformers, GNNs) or sacrifice accuracy (classical correlation-based approaches). This project demonstrates that a carefully optimized standard MLP can achieve competitive results (83%) while remaining CPU-trainable and accessible.

## Results

| Configuration | Accuracy | Notes |
|--------------|----------|-------|
| Baseline (embeddings-only) | 62.23% | Default hyperparameters |
| Tuned single model | 80.14% | ±1.7% across 5 seeds |
| Ensemble (5 models) | **83.06%** | Simple averaging |

**Ensemble metrics**: Precision 79.68%, Recall 88.59%, F1 83.90%, AUROC 0.814

## Architecture

```
TF Input ──→ [Embedding(512) + Expression(11)] ──→ FC(523→2048→1024→512) ──┐
                                                                            ├→ Cosine Sim → σ → Edge Score
Gene Input ─→ [Embedding(512) + Expression(11)] ──→ FC(523→2048→1024→512) ──┘
```

- ~5.2M parameters
- Cosine similarity scoring with temperature τ=0.05
- Adam optimizer, LR=0.005, weight decay 0.01
- Trains in ~4 hours per model on CPU

## Quick Start

```bash
# Build
cargo build --release

# Train single model
cargo run --release --bin train_standard_mlp

# Train ensemble (5 models)
cargo run --release --bin train_ensemble

# Run hyperparameter search
cargo run --release --bin hyperparam_tuning

# Run ablation study
cargo run --release --bin ablation_study
```

## Data Requirements

Place in `data/`:
- **Expression data**: Human brain single-cell RNA-seq in H5AD format
- **Prior knowledge**: Merged TF-gene regulatory databases (DoRothEA, TRRUST)

The dataset comprises ~39K examples (50/50 pos/neg split) with 70/15/15 train/val/test partitioning.

## Project Structure

```
src/
├── models/
│   ├── hybrid_embeddings.rs   # Production two-tower model
│   ├── nn.rs                  # Neural network primitives
│   └── ...                    # Experimental model variants
├── data/                      # Data loading pipeline
├── loss/                      # Loss function implementations
├── training/                  # Optimization & scheduling
├── bin/
│   ├── train_standard_mlp.rs  # Main training script
│   ├── train_ensemble.rs      # Ensemble training
│   ├── ablation_study.rs      # Ablation experiments
│   ├── seed_robustness.rs     # Stability testing
│   └── ...                    # Other experimental scripts
├── config.rs                  # Configuration management
└── lib.rs                     # Library root
scripts/                       # Python analysis & figure generation
results/                       # JSON result files
figures/                       # Publication figures (PDF/PNG)
```

## Key Findings

1. **Hyperparameter tuning is critical**: LR and embedding dimension alone account for +16% accuracy
2. **Expression features help modestly**: +1.8% from 11-dim cell-type expression profiles
3. **Ensembling is cheap and effective**: +3% from 5-model averaging
4. **Realistic ceiling**: Standard MLPs top out around 83-85%; reaching 88-92% requires transformers/GNNs and GPU compute

## Dependencies

- `ndarray` — Tensor operations
- `rayon` — Parallel computation
- `csv`, `serde`, `serde_json` — Data I/O
- `rand`, `rand_distr` — Random number generation
- `anyhow`, `thiserror` — Error handling

## License

Research use. Contact: evint.koo@gmail.com
