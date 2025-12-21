#!/usr/bin/env python3
"""
Process Human Brain Cell Atlas v1.0 data
Extract state information (dissection × supercluster) and prepare for GRN inference
"""

import json
import h5py
import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple
import pickle

DATA_DIR = Path("data/brain_v1_0")
OUTPUT_DIR = Path("data/processed")

def load_h5ad_file(file_path: Path) -> ad.AnnData:
    """Load h5ad file"""
    print(f"Loading {file_path.name}...")
    return ad.read_h5ad(file_path)

def extract_state_info(adata: ad.AnnData) -> Dict:
    """Extract state (dissection × supercluster) information"""
    obs = adata.obs
    
    # Check available columns
    print(f"Available columns: {obs.columns.tolist()[:20]}")
    
    # Common column names for brain regions and cell types
    dissection_cols = ['region', 'brain_region', 'dissection', 'tissue', 'anatomical_information']
    cluster_cols = ['cluster', 'supercluster', 'cell_type', 'subclass']
    
    dissection_col = None
    cluster_col = None
    
    for col in dissection_cols:
        if col in obs.columns:
            dissection_col = col
            break
    
    for col in cluster_cols:
        if col in obs.columns:
            cluster_col = col
            break
    
    print(f"Using dissection column: {dissection_col}")
    print(f"Using cluster column: {cluster_col}")
    
    if dissection_col and cluster_col:
        state_counts = obs.groupby([dissection_col, cluster_col]).size()
        return {
            'dissection_col': dissection_col,
            'cluster_col': cluster_col,
            'state_counts': state_counts.to_dict(),
            'total_cells': len(obs)
        }
    
    return {'error': 'Could not find dissection and cluster columns'}

def compute_pseudobulk(adata: ad.AnnData, dissection_col: str, cluster_col: str) -> Dict:
    """Compute pseudobulk expression per state"""
    pseudobulk_data = {}
    
    states = adata.obs.groupby([dissection_col, cluster_col]).groups
    
    for state_key, cell_indices in states.items():
        dissection, cluster = state_key
        state_id = f"{dissection}_{cluster}"
        
        # Get expression matrix for this state
        X = adata.X[cell_indices]
        
        # Compute mean expression
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        mean_expr = np.mean(X, axis=0)
        
        pseudobulk_data[state_id] = {
            'dissection': dissection,
            'cluster': cluster,
            'cell_count': len(cell_indices),
            'mean_expression': mean_expr,
            'gene_names': adata.var_names.tolist()
        }
    
    return pseudobulk_data

def main():
    """Main processing function"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Human Brain Cell Atlas v1.0 Data Processing")
    print("=" * 60)
    
    # Find all h5ad files
    h5ad_files = list(DATA_DIR.rglob("*.h5ad"))
    print(f"\nFound {len(h5ad_files)} H5AD files")
    
    all_states = []
    all_pseudobulk = {}
    
    for i, file_path in enumerate(h5ad_files[:1]):  # Process first file for testing
        print(f"\n[{i+1}/{len(h5ad_files)}] Processing {file_path.name}")
        
        try:
            # Load data
            adata = load_h5ad_file(file_path)
            
            # Extract state info
            state_info = extract_state_info(adata)
            
            if 'error' not in state_info:
                all_states.append({
                    'file': file_path.name,
                    'dataset_id': file_path.parent.name,
                    **state_info
                })
                
                # Compute pseudobulk
                print("Computing pseudobulk expression...")
                pseudobulk = compute_pseudobulk(
                    adata,
                    state_info['dissection_col'],
                    state_info['cluster_col']
                )
                all_pseudobulk.update(pseudobulk)
                
                print(f"  Found {len(pseudobulk)} states")
                print(f"  Total cells: {state_info['total_cells']:,}")
            else:
                print(f"  Error: {state_info['error']}")
                
        except Exception as e:
            print(f"  Error processing file: {e}")
            import traceback
            traceback.print_exc()
    
    # Save processed data
    if all_states:
        states_file = OUTPUT_DIR / "states_info.json"
        with open(states_file, 'w') as f:
            # Convert numpy types to native Python types for JSON
            serializable_states = []
            for state in all_states:
                state_copy = state.copy()
                if 'state_counts' in state_copy:
                    state_copy['state_counts'] = {
                        f"{k[0]}_{k[1]}": int(v) 
                        for k, v in state_copy['state_counts'].items()
                    }
                serializable_states.append(state_copy)
            
            json.dump(serializable_states, f, indent=2)
        print(f"\n✓ Saved states info to {states_file}")
    
    if all_pseudobulk:
        pseudobulk_file = OUTPUT_DIR / "pseudobulk_data.pkl"
        with open(pseudobulk_file, 'wb') as f:
            pickle.dump(all_pseudobulk, f)
        print(f"✓ Saved pseudobulk data to {pseudobulk_file}")
        
        # Save summary
        summary = {
            'total_states': len(all_pseudobulk),
            'state_ids': list(all_pseudobulk.keys()),
            'state_cell_counts': {
                state_id: data['cell_count'] 
                for state_id, data in all_pseudobulk.items()
            }
        }
        summary_file = OUTPUT_DIR / "pseudobulk_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved pseudobulk summary to {summary_file}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
