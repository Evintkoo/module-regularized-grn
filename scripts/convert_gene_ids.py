#!/usr/bin/env python3
"""
Convert ENSEMBL IDs to Gene Symbols for expression mapping
"""
import json
from pathlib import Path

def load_ensembl_mapping():
    """
    Load ENSEMBL to Gene Symbol mapping from biomart or other source.
    For now, we'll try to match what we have in the priors.
    """
    # Load gene names from expression data
    with open('data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd/genes.txt', 'r') as f:
        ensembl_ids = [line.strip() for line in f.readlines()]
    
    # Load priors to get gene symbols
    with open('data/priors/merged_priors.json', 'r') as f:
        priors = json.load(f)
    
    # Get all genes from priors
    all_genes = set()
    for tf, targets in priors.items():
        all_genes.update(targets)
        all_genes.add(tf)  # TFs are also genes
    
    print(f"Expression genes (ENSEMBL): {len(ensembl_ids)}")
    print(f"Prior genes (symbols): {len(all_genes)}")
    
    # Try using MyGene.info API
    try:
        import mygene
        print("\nUsing MyGene.info API for conversion...")
        mg = mygene.MyGeneInfo()
        
        # Query in batches
        batch_size = 1000
        all_results = []
        for i in range(0, len(ensembl_ids), batch_size):
            batch = ensembl_ids[i:i+batch_size]
            results = mg.querymany(batch, scopes='ensembl.gene', 
                                  fields='symbol', species='human',
                                  returnall=True)
            all_results.extend(results['out'])
            print(f"  Processed {i+len(batch)}/{len(ensembl_ids)} genes")
        
        # Build mapping
        ensembl_to_symbol = {}
        for result in all_results:
            if 'symbol' in result and 'query' in result:
                ensembl_to_symbol[result['query']] = result['symbol']
        
        print(f"\nMapped {len(ensembl_to_symbol)} genes")
        
        # Count overlaps
        overlap = set(ensembl_to_symbol.values()) & all_genes
        print(f"Overlap with priors: {len(overlap)} genes ({100*len(overlap)/len(all_genes):.1f}%)")
        
        # Save mapping
        output_path = 'data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd/ensembl_to_symbol.json'
        with open(output_path, 'w') as f:
            json.dump(ensembl_to_symbol, f, indent=2)
        print(f"\n✅ Saved mapping to {output_path}")
        
        # Also save overlap genes for quick reference
        overlap_path = 'data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd/overlapping_genes.json'
        with open(overlap_path, 'w') as f:
            json.dump(list(overlap), f, indent=2)
        print(f"✅ Saved {len(overlap)} overlapping genes to {overlap_path}")
        
        return ensembl_to_symbol
        
    except ImportError:
        print("\n⚠️  MyGene not installed. Installing...")
        import subprocess
        subprocess.run(['pip3', 'install', 'mygene'], check=True)
        print("✅ Installed mygene. Please run script again.")
        return None

if __name__ == '__main__':
    print("=== ENSEMBL ID to Gene Symbol Converter ===\n")
    mapping = load_ensembl_mapping()
    
    if mapping:
        print(f"\n✅ Conversion complete!")
        print(f"   Total mappings: {len(mapping)}")
        print(f"   Sample: {list(mapping.items())[:5]}")
