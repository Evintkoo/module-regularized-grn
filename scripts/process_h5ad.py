#!/usr/bin/env python3
"""
Process H5AD brain expression data for GRN training.

Extracts:
1. Expression matrices (genes x cells)
2. Gene names
3. Cell metadata
4. Creates pseudobulk profiles by cell type

Exports to CSV/NPY format for Rust consumption.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scanpy as sc
    import anndata
except ImportError:
    print("Error: scanpy not installed")
    print("Install with: pip install scanpy anndata")
    sys.exit(1)


def process_h5ad_file(h5ad_path: Path, output_dir: Path, top_genes: int = 2000):
    """
    Process a single H5AD file.
    
    Args:
        h5ad_path: Path to H5AD file
        output_dir: Output directory
        top_genes: Number of top variable genes to keep
    
    Returns:
        Dict with processing stats
    """
    print(f"\nProcessing: {h5ad_path.name}")
    
    # Load H5AD
    print("  Loading...")
    adata = sc.read_h5ad(h5ad_path)
    
    print(f"  Shape: {adata.shape} (cells x genes)")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Cells: {adata.n_obs}")
    
    # Select top variable genes
    if adata.n_vars > top_genes:
        print(f"  Selecting top {top_genes} variable genes...")
        # Simple variance-based selection
        if hasattr(adata.X, 'todense'):
            expr = np.array(adata.X.todense())
        else:
            expr = np.array(adata.X)
        
        variances = np.var(expr, axis=0)
        top_indices = np.argsort(variances)[-top_genes:]
        adata = adata[:, top_indices]
        print(f"  Filtered to {adata.n_vars} genes")
    
    # Get gene names
    gene_names = adata.var_names.tolist()
    
    # Check if data is sparse
    is_sparse = hasattr(adata.X, 'todense')
    print(f"  Data type: {'sparse' if is_sparse else 'dense'}")
    
    # Get expression matrix (convert to dense if sparse)
    if is_sparse:
        print("  Converting sparse to dense...")
        expr_matrix = adata.X.todense()
    else:
        expr_matrix = adata.X
    
    # Convert to numpy array
    if hasattr(expr_matrix, 'A'):
        expr_matrix = expr_matrix.A
    expr_matrix = np.array(expr_matrix, dtype=np.float32)
    
    print(f"  Expression matrix: {expr_matrix.shape}")
    print(f"  Memory: {expr_matrix.nbytes / 1e6:.1f} MB")
    
    # Get cell metadata
    cell_metadata = adata.obs.copy()
    
    # Create pseudobulk by cell type if available
    pseudobulk_data = None
    if 'cell_type' in cell_metadata.columns:
        print("  Creating pseudobulk by cell type...")
        cell_types = cell_metadata['cell_type'].unique()
        print(f"  Cell types: {len(cell_types)}")
        
        pseudobulk = []
        pseudobulk_labels = []
        
        for cell_type in cell_types:
            mask = cell_metadata['cell_type'] == cell_type
            if mask.sum() > 0:
                # Mean expression across cells of this type
                mean_expr = expr_matrix[mask].mean(axis=0)
                pseudobulk.append(mean_expr)
                pseudobulk_labels.append(cell_type)
        
        pseudobulk_data = {
            'expression': np.array(pseudobulk, dtype=np.float32),
            'labels': pseudobulk_labels,
            'genes': gene_names,
        }
        print(f"  Pseudobulk shape: {pseudobulk_data['expression'].shape}")
    
    # Create output subdirectory
    file_id = h5ad_path.stem
    file_output_dir = output_dir / file_id
    file_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save gene names
    gene_file = file_output_dir / "genes.txt"
    with open(gene_file, 'w') as f:
        for gene in gene_names:
            f.write(f"{gene}\n")
    print(f"  Saved genes: {gene_file}")
    
    # Save pseudobulk if available
    if pseudobulk_data:
        # Save expression matrix
        expr_file = file_output_dir / "pseudobulk_expression.npy"
        np.save(expr_file, pseudobulk_data['expression'])
        print(f"  Saved expression: {expr_file}")
        
        # Save labels
        label_file = file_output_dir / "pseudobulk_labels.json"
        with open(label_file, 'w') as f:
            json.dump(pseudobulk_data['labels'], f)
        print(f"  Saved labels: {label_file}")
    
    # Save summary stats
    stats = {
        'file': h5ad_path.name,
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'genes': gene_names,
        'has_pseudobulk': pseudobulk_data is not None,
        'n_pseudobulk': len(pseudobulk_data['labels']) if pseudobulk_data else 0,
    }
    
    stats_file = file_output_dir / "stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved stats: {stats_file}")
    
    return stats


def load_prior_genes(priors_file: Path) -> set:
    """Load gene names from priors file."""
    print(f"\nLoading prior genes from: {priors_file}")
    with open(priors_file, 'r') as f:
        priors = json.load(f)
    
    # Extract all unique gene names
    genes = set()
    for targets in priors.values():
        genes.update(targets)
    
    print(f"  Found {len(genes)} unique genes in priors")
    return genes


def main():
    parser = argparse.ArgumentParser(description='Process H5AD files for GRN training')
    parser.add_argument('--input-dir', type=str, default='data/brain_v1_0',
                        help='Directory containing H5AD files')
    parser.add_argument('--output-dir', type=str, default='data/processed/expression',
                        help='Output directory for processed data')
    parser.add_argument('--priors', type=str, default='data/priors/merged_priors.json',
                        help='Prior knowledge file to filter genes')
    parser.add_argument('--top-genes', type=int, default=2000,
                        help='Number of top variable genes to keep')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    print("=== H5AD Expression Data Processing ===")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Top genes: {args.top_genes}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all H5AD files
    h5ad_files = list(input_dir.rglob("*.h5ad"))
    print(f"\nFound {len(h5ad_files)} H5AD files")
    
    if args.max_files:
        h5ad_files = h5ad_files[:args.max_files]
        print(f"Processing first {len(h5ad_files)} files...")
    
    # Process each file
    all_stats = []
    for i, h5ad_file in enumerate(h5ad_files, 1):
        print(f"\n[{i}/{len(h5ad_files)}]")
        try:
            stats = process_h5ad_file(h5ad_file, output_dir, args.top_genes)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save summary
    summary = {
        'n_files_processed': len(all_stats),
        'total_genes_union': len(set(g for s in all_stats for g in s.get('genes', []))),
        'files': all_stats,
    }
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== Processing Complete ===")
    print(f"Files processed: {len(all_stats)}")
    print(f"Summary saved: {summary_file}")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
