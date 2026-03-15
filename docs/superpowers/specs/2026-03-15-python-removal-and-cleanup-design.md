# Design: Python Removal, Codebase Cleanup & Documentation Update

**Date**: 2026-03-15
**Project**: module-regularized-grn
**Status**: Approved

---

## Goal

Remove all Python from the project, replace active Python scripts with Rust equivalents, delete inactive experimental code, review core Rust for performance and hygiene, then update all markdown documentation to reflect the cleaned state.

---

## Execution Order

Code-first: clean and port code → then update docs to reflect ground truth.

Commit checkpoints after each destructive step to ensure recoverability.

1. **Commit checkpoint** — ensure working tree is clean before starting
2. Audit & delete inactive Python scripts → **commit**
3. Delete superseded Rust bin targets + update `Cargo.toml` stanzas → verify `cargo build --release` → **commit**
4. Add new Rust crates to `Cargo.toml` (`plotters`, `statrs`, check `hdf5`)
5. Port active Python scripts to Rust new bins → **commit**
6. Review core Rust for performance and hygiene → **commit**
7. Update README.md, CLAUDE.md, plan.md → **commit**

---

## Section 1: Python Script Audit

### Delete (inactive / superseded by Rust)

| Script | Reason |
|--------|--------|
| `train_advanced.py` | Training replaced by Rust |
| `train_attention_model.py` | Training replaced by Rust |
| `comprehensive_improvements.py` | Experimental, not used |
| `download_brain_data.py` | One-time data prep, already run |
| `process_brain_data.py` | One-time data prep, already run |
| `convert_gene_ids.py` | One-time data prep, already run |
| `feature_engineering.py` | One-time data prep, already run |

### Port to Rust (active, used in manuscript workflow)

| Script | Rust target | Notes |
|--------|-------------|-------|
| `generate_all_figures.py` / `plot_results.py` | `src/bin/generate_figures.rs` | SVG/PNG via `plotters` crate |
| `generate_tables.py` | `src/bin/generate_tables.rs` | CSV/LaTeX table output |
| `statistical_analysis.py` | `src/bin/statistical_analysis.rs` | Bootstrap CIs, significance tests |
| ~~`process_h5ad.py`~~ | **Delete** | One-time data prep; outputs (`.npy` files in `data/processed/expression/`) already exist. `expression.rs` reads these outputs, not raw H5AD. Reclassified as inactive. |
| `make_pdf.py` | Shell script calling `pandoc` | 86-line reportlab wrapper; replace with `pandoc` CLI call, not a Rust binary |

**Crate additions needed** — add explicitly to `Cargo.toml` before porting:
- `plotters = "0.3"` — figure generation (PNG and SVG output)
- `statrs = "0.16"` — chi-squared CDF for McNemar's test and other statistical distributions
- `hdf5` — NOT needed; `process_h5ad.py` is reclassified as delete (see below); `expression.rs` already reads processed outputs

**Output format decisions:**
- **Figures**: `plotters` produces PNG and SVG, not PDF. Port will output PNG (300 dpi) and SVG. PDF figures are not natively supported; if PDF is required for journal submission, use a post-process step (`rsvg-convert` or `inkscape` CLI) outside Rust. Document this in README.
- **Tables**: `generate_tables.py` outputs LaTeX (`.tex`). Rust port will use manual format-string serialization for LaTeX table output; no external crate needed. Output `.tex` files with `\caption{}` and `\label{}` matching current Python output.
- **Statistical analysis**: `statistical_analysis.py` uses `scipy.stats.chi2.cdf` for McNemar's test. Rust port will use `statrs::distribution::ChiSquared` for the same computation.

---

## Section 2: Rust Bin Cleanup

### Keep

| Binary | Purpose |
|--------|---------|
| `train_standard_mlp.rs` | Primary single-model training |
| `train_ensemble.rs` | Ensemble training (5 models) |
| `ablation_study.rs` | Component ablation experiments |
| `seed_robustness.rs` | Multi-seed stability testing |
| `hyperparam_tuning.rs` | Hyperparameter search |
| `evaluate.rs` | Model evaluation |
| `process_data.rs` | Data pipeline |
| `download_priors.rs` | Prior knowledge fetching |

### Delete (~17 experimental/superseded targets)

`train.rs`, `train_hybrid.rs`, `train_hybrid_v2.rs`, `train_embeddings.rs`, `train_embeddings_extended.rs`, `train_advanced.rs`, `train_optimized.rs`, `train_scaled.rs`, `train_scaled_models.rs`, `train_medium.rs`, `train_ultra.rs`, `train_95_target.rs`, `train_priors.rs`, `train_classifier_head.rs`, `train_cross_attention.rs`, `train_with_enhanced_features.rs`, `phase1_expression.rs`, `train_example.rs`

### Cargo.toml synchronization (critical)

**Source file deletion is required for all bins in the delete list, regardless of whether a `[[bin]]` stanza is present.** Some bins on the delete list are already unregistered in Cargo.toml but still exist as source files (`train_hybrid.rs`, `train_hybrid_v2.rs`, `train_advanced.rs`, `train_scaled.rs`, `train_medium.rs`, `train_ultra.rs`, `train_95_target.rs`, `train_classifier_head.rs`). Delete these files even though no stanza removal is needed.

For bins with stanzas: remove the `[[bin]]` stanza from `Cargo.toml` AND delete the source file.

**Kept bins currently missing `[[bin]]` stanzas** — add these to `Cargo.toml`:
- `ablation_study` → `src/bin/ablation_study.rs`
- `seed_robustness` → `src/bin/seed_robustness.rs`
- `evaluate` → `src/bin/evaluate.rs`

Verify `cargo build --release` succeeds after every batch of deletions, not just at the end.

---

## Section 3: Core Rust Code Review

### Performance targets

- **`src/models/nn.rs`**: Check for unnecessary allocations, clone-heavy loops, missed `rayon` parallelism in forward passes
- **Embedding lookups**: Verify no large array copies per batch item
- **`src/data/dataloader.rs`**: Confirm H5AD is read once at startup, not re-read per epoch

### Hygiene targets

- **`src/models/mod.rs`**: Remove dead `mod` and `pub use` exports for deleted experimental models. Also delete the corresponding model source files (e.g. `attention.rs`, `attention_model.rs`, `baseline.rs`, `classifier_head.rs`, `cross_attention_model.rs`, `embeddings.rs`, `expression_model.rs`, `optimized_embeddings.rs`, `scalable_hybrid.rs`, `two_tower.rs`, `learnable_embeddings.rs`) if they are only used by deleted bins. Verify no kept bin imports them before deletion.
- **`src/loss/` and `src/training/`**: Remove modules that only existed to support deleted bins
- **Unused imports**: Clean up `use` statements across codebase after bin deletion

---

## Section 4: Markdown Updates

All three docs updated after code cleanup is complete, so they reflect the actual final state.

### README.md
- Remove `scripts/ # Python analysis & figure generation` from project structure
- Add new Rust figure/analysis bins to project structure
- Keep results table and architecture diagram unchanged
- Update dependencies section if new crates added

### CLAUDE.md
- **Conventions**: Replace "Figures generated via Python scripts in scripts/" with Rust bin reference
- **Essential Code Paths**: Update bin list to match kept targets only
- Remove any Python-related notes

### plan.md
- Mark Phase 7 figures as now generated by Rust (not Python)
- Update "Key Design Decisions" to note pure-Rust figure generation
- No changes to Phase 8/9/10 content

---

## Success Criteria

- `scripts/` directory is empty or removed
- `cargo build --release` succeeds with no errors for all kept and new bins
- All kept bins **compile** (full run requires dataset; smoke-test compile is sufficient)
- Figure generation: `cargo run --release --bin generate_figures` produces PNG/SVG output
- Table generation: `cargo run --release --bin generate_tables` produces `.tex` files
- Statistical analysis: `cargo run --release --bin statistical_analysis` runs against `results/` JSON
- README, CLAUDE.md, plan.md contain no Python references
- No dead `mod` exports or unused imports in core modules (`cargo build` emits zero warnings)
