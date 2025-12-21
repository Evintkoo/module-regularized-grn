# Human Brain Cell Atlas v1.0 Data

## Overview
This directory contains data from the Human Brain Cell Atlas v1.0 from the BRAIN Initiative.

- **Collection ID**: `283d65eb-dd53-496d-adb7-7570c7caa443`
- **Source**: [CELLxGENE](https://cellxgene.cziscience.com/collections/283d65eb-dd53-496d-adb7-7570c7caa443)
- **Publication**: Siletti et al. (2023) Science
- **Total Cells**: ~2.48 million cells
- **Format**: H5AD (AnnData)

## Directory Structure

```
data/
├── brain_v1_0/              # Raw downloaded data
│   ├── collection_metadata.json
│   └── {dataset_id}/
│       └── {dataset_id}.h5ad
└── processed/               # Processed data for GRN inference
    ├── states_info.json
    ├── pseudobulk_data.pkl
    └── pseudobulk_summary.json
```

## Scripts

### 1. Download Data
```bash
python3 scripts/download_brain_data.py
```

Downloads datasets from CELLxGENE. Currently configured to download first 5 datasets for testing.
To download all datasets, modify the script line:
```python
for i, dataset in enumerate(datasets[:5]):  # Change to enumerate(datasets)
```

### 2. Process Data
```bash
python3 scripts/process_brain_data.py
```

Processes H5AD files to extract:
- State definitions (dissection × supercluster)
- Cell counts per state
- Pseudobulk expression matrices
- Gene annotations

## Data Format

### States
Each state is defined as `{dissection}_{supercluster}` where:
- **Dissection**: Brain region (e.g., "hippocampus", "cortex")
- **Supercluster**: Cell type cluster (e.g., "excitatory neurons", "astrocytes")

### Pseudobulk Expression
For each state:
- Mean gene expression across all cells in that state
- Cell count
- Gene names

## Requirements

```bash
pip install anndata h5py scanpy requests
```

## Citation

```bibtex
@article{siletti2023transcriptomic,
  title={Transcriptomic diversity of cell types across the adult human brain},
  author={Siletti, Kimberly and others},
  journal={Science},
  year={2023},
  doi={10.1126/science.add7046}
}
```

## Notes

- Downloaded datasets are ~200MB+ each
- Full collection is >100GB
- Use `min_cell_count=50` filter when creating states (see Phase 1 plan)
- Consider downloading subsets by brain region or cell type for initial development
