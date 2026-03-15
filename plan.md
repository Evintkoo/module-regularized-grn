# Research Plan: Module-Regularized GRN Inference

## Research Question

Can a parameter-matched modular (two-tower) neural network architecture outperform a monolithic cross-encoder for gene regulatory network inference, and does inter-module collaboration (contrastive learning, mutual learning, co-teaching) improve edge prediction accuracy and stability?

## Thesis Context

This project is part of a dissertation comparing **two small collaborating networks** vs **one large monolithic network** for state-conditioned GRN inference from single-nucleus brain expression data. The key comparison axis is modular/collaborative capacity vs monolithic capacity under matched parameter budgets.

---

## Completed Work (Phases 1-7)

### Phase 1: Data Preparation & Baseline
- Loaded and processed human brain scRNA-seq expression data (H5AD format)
- Integrated prior knowledge databases (DoRothEA, TRRUST) — 29K TF-gene pairs
- Constructed balanced dataset (~39K examples, 50/50 pos/neg)
- Established baseline: **62.23%** accuracy (embeddings-only, default hyperparams)

### Phase 2: Architecture & Hyperparameter Optimization
- Implemented two-tower MLP with learnable embeddings
- Systematic hyperparameter search (LR, embedding dim, hidden dim, temperature)
- Key discovery: LR 0.001→0.005 and embed 64→512 = **+16%** accuracy
- Result: **78.31%** (optimized embeddings-only)

### Phase 3: Expression Feature Integration
- Added 11-dimensional cell-type expression profiles as input features
- Implemented hybrid architecture (embeddings + expression → two-tower)
- Result: **80.14%** single model (±1.7% across 5 seeds)

### Phase 4: Ensemble Methods
- 5-model ensemble with different random seeds, simple averaging
- Result: **83.06%** ensemble accuracy
- Precision 79.68%, Recall 88.59%, F1 83.90%, AUROC 0.814

### Phase 5: Comprehensive Evaluation
- Multi-seed validation (5 seeds) with confidence intervals
- Bootstrap analysis (n=1000): 95% CI [79.33%, 81.01%]
- McNemar significance test: p < 0.000001

### Phase 6: Ablation Studies
- Component ablation: embeddings-only, expression-only, hybrid
- Architecture ablation: depth, width, temperature sensitivity
- Confirmed each component contributes meaningfully

### Phase 7: Statistical Analysis & Figures
- Generated 9 publication figures (ROC, PR, confusion matrix, ablation, bootstrap, etc.)
- Full statistical analysis with bootstrap CIs and significance tests
- Manuscript draft (HTML) completed
- **Figures now generated in Rust**: `generate_figures`, `generate_tables`, `statistical_analysis` bins replace Python scripts

---

## Next Phases

### Phase 8: Dissertation Experiments — Collaboration Algorithms

The core dissertation contribution: comparing collaboration strategies between modular networks.

#### 8.1 Experimental Models

| Model | Architecture | Collaboration | Description |
|-------|-------------|---------------|-------------|
| **A** | Two-Tower + InfoNCE | Contrastive | TF encoder + Gene encoder, retrieval-style scoring |
| **B** | Two Decoders + DML | Mutual Learning | Bilinear + MLP decoders teaching each other (KL divergence) |
| **C** | Two Decoders + Co-teaching | Robust Learning | Small-loss exchange for noisy pseudo-label edges |
| **D** | Cross-Encoder (monolithic) | None | Single MLP taking joint [TF; Gene; state] features |
| **E** | Two-Tower (no collaboration) | None | Same as A but without contrastive coupling (control) |

#### 8.2 Experimental Controls
- **Parameter-matched**: params(Net1) + params(Net2) ≈ params(BigNet)
- **Data-matched**: identical splits, candidate edge sets, negative sampling
- **Multiple seeds**: 5-10 seeds per configuration
- **Bootstrap stability**: n=1000 for confidence intervals

#### 8.3 Evaluation Metrics
- Edge classification: accuracy, precision, recall, F1, AUROC
- Enrichment vs curated references (DoRothEA/TRRUST)
- Reproducibility: Jaccard overlap across seeds/bootstraps
- Confidence calibration: high-confidence edges → higher enrichment
- Predictive utility: held-out variance explained
- Runtime/memory comparisons

#### 8.4 Hypotheses
- **H1**: Two-tower models achieve similar predictive utility with improved scalability and reproducibility
- **H2**: Mutual learning (DML) improves stability at fixed sparsity
- **H3**: Monolithic cross-encoder overfits more / produces less stable edges without heavy regularization

### Phase 9: Advanced Architectures (stretch goals)
- Attention mechanisms (multi-head, cross-attention): expected +2-3%
- Graph neural networks (message passing on prior network): expected +2-3%
- Foundation model embeddings (scGPT, Geneformer): expected +2-4%
- Realistic maximum: **88-90%** with 2-3 months work

### Phase 10: Dissertation Writing
- Methods chapter: architecture, training, evaluation methodology
- Results chapter: Phase 8 experiments with figures and statistical tests
- Discussion: modular vs monolithic tradeoffs, biological implications
- Target venues: BMC Bioinformatics, PLOS Computational Biology, Bioinformatics, MLCB/NeurIPS workshops

---

## Current Status

**Phase 7 complete. Ready for Phase 8 (dissertation collaboration experiments).**

The 83% single-architecture result provides a strong baseline. The next step is implementing the collaboration algorithms (InfoNCE, DML, Co-teaching) and the monolithic cross-encoder control to answer the core dissertation question.

---

## Key Design Decisions

1. **Pure Rust** — No Python or PyTorch dependency; all NN operations, figure generation, table output, and statistical analysis from scratch using ndarray and plotters
2. **CPU-trainable** — Accessible without GPU infrastructure (~4h per model on MacBook)
3. **Two-tower as primary** — Factorized TF/Gene encoding enables scalable scoring via matrix multiplication
4. **Cosine similarity** — Interpretable geometric scoring (vs learned bilinear)
5. **Parameter matching** — Fair comparison requires matched capacity budgets
