#!/usr/bin/env python3
"""
Comprehensive Gene ID Mapper and Feature Engineering
Converts between Ensembl, HGNC, and UniProt IDs
"""

import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, Set, List, Tuple
from pathlib import Path
import time

class GeneIDMapper:
    """Map between different gene ID systems"""
    
    def __init__(self):
        self.ensembl_to_symbol = {}
        self.symbol_to_ensembl = {}
        self.uniprot_to_symbol = {}
        self.symbol_to_uniprot = {}
        self.gene_id_mappings = {}  # Comprehensive mappings from MyGene.info
        
    def load_from_biomart(self, species="hsapiens"):
        """Download mappings from Ensembl BioMart (slow but comprehensive)"""
        print("Downloading gene ID mappings from Ensembl BioMart...")
        print("(This may take a few minutes...)")
        
        # BioMart query for human genes
        url = "http://www.ensembl.org/biomart/martservice"
        query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >
    <Dataset name = "{species}_gene_ensembl" interface = "default" >
        <Attribute name = "ensembl_gene_id" />
        <Attribute name = "external_gene_name" />
        <Attribute name = "uniprot_gn_id" />
    </Dataset>
</Query>"""
        
        try:
            response = requests.post(url, data={'query': query}, timeout=300)
            response.raise_for_status()
            
            lines = response.text.strip().split('\n')
            header = lines[0].split('\t')
            
            for line in lines[1:]:
                parts = line.split('\t')
                if len(parts) >= 2:
                    ensembl_id = parts[0].strip()
                    symbol = parts[1].strip()
                    uniprot = parts[2].strip() if len(parts) > 2 else ""
                    
                    if ensembl_id and symbol:
                        self.ensembl_to_symbol[ensembl_id] = symbol
                        self.symbol_to_ensembl[symbol] = ensembl_id
                    
                    if uniprot and symbol:
                        self.uniprot_to_symbol[uniprot] = symbol
                        self.symbol_to_uniprot[symbol] = uniprot
            
            print(f"✓ Loaded {len(self.ensembl_to_symbol)} Ensembl mappings")
            print(f"✓ Loaded {len(self.uniprot_to_symbol)} UniProt mappings")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download from BioMart: {e}")
            return False
    
    def load_from_files(self, expr_dir: Path):
        """Load existing mappings from files"""
        # Load Ensembl mappings
        ensembl_file = expr_dir / "ensembl_to_symbol.json"
        if ensembl_file.exists():
            with open(ensembl_file) as f:
                self.ensembl_to_symbol = json.load(f)
            self.symbol_to_ensembl = {v: k for k, v in self.ensembl_to_symbol.items()}
            print(f"✓ Loaded {len(self.ensembl_to_symbol)} Ensembl mappings from file")
        
        # Load UniProt mappings
        uniprot_file = expr_dir / "uniprot_to_symbol.json"
        if uniprot_file.exists():
            with open(uniprot_file) as f:
                self.uniprot_to_symbol = json.load(f)
            self.symbol_to_uniprot = {v: k for k, v in self.uniprot_to_symbol.items()}
            print(f"✓ Loaded {len(self.uniprot_to_symbol)} UniProt mappings from file")
        
        # Load comprehensive gene ID mappings
        gene_id_file = expr_dir / "gene_id_mappings.json"
        if gene_id_file.exists():
            with open(gene_id_file) as f:
                self.gene_id_mappings = json.load(f)
            print(f"✓ Loaded {len(self.gene_id_mappings)} comprehensive gene ID mappings")
        else:
            self.gene_id_mappings = {}
    
    def save_to_files(self, expr_dir: Path):
        """Save mappings for future use"""
        with open(expr_dir / "ensembl_to_symbol.json", "w") as f:
            json.dump(self.ensembl_to_symbol, f, indent=2)
        with open(expr_dir / "uniprot_to_symbol.json", "w") as f:
            json.dump(self.uniprot_to_symbol, f, indent=2)
        print("✓ Saved mappings to files")
    
    def map_gene(self, gene_id: str) -> str:
        """Map any gene ID to HGNC symbol"""
        # Try direct mapping from MyGene.info first
        if gene_id in self.gene_id_mappings:
            return self.gene_id_mappings[gene_id]
        
        # Try direct return if already a symbol
        if gene_id.isupper() and gene_id.isalpha():
            return gene_id
        
        # Try Ensembl
        if gene_id.startswith("ENSG"):
            symbol = self.ensembl_to_symbol.get(gene_id, None)
            if symbol:
                return symbol
        
        # Try UniProt
        if gene_id in self.uniprot_to_symbol:
            return self.uniprot_to_symbol[gene_id]
        
        # Try case-insensitive
        gene_id_upper = gene_id.upper()
        if gene_id_upper in self.symbol_to_ensembl:
            return gene_id_upper
        
        return None
    
    def get_ensembl_id(self, gene_symbol: str) -> str:
        """Get Ensembl ID for a gene symbol"""
        return self.symbol_to_ensembl.get(gene_symbol, None)


def create_enhanced_expression_features(
    expr_matrix: np.ndarray,
    gene_names: List[str],
    mapper: GeneIDMapper
) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Create enhanced expression features with proper gene mapping
    
    Returns:
        - gene_to_features: mapping from gene symbol to feature vector
        - gene_to_ensembl: mapping from gene symbol to Ensembl ID
    """
    print("\nCreating enhanced expression features...")
    
    gene_to_features = {}
    gene_to_ensembl = {}
    
    # Map Ensembl IDs to symbols
    for i, ensembl_id in enumerate(gene_names):
        symbol = mapper.map_gene(ensembl_id)
        if symbol:
            expr_profile = expr_matrix[:, i]
            
            # Basic expression features (11 cell types)
            basic_features = expr_profile
            
            # Statistical features
            mean_expr = np.mean(expr_profile)
            std_expr = np.std(expr_profile)
            max_expr = np.max(expr_profile)
            min_expr = np.min(expr_profile)
            cv = std_expr / (mean_expr + 1e-8)  # Coefficient of variation
            
            # Cell-type specificity (entropy-based)
            expr_norm = expr_profile / (np.sum(expr_profile) + 1e-8)
            entropy = -np.sum(expr_norm * np.log(expr_norm + 1e-8))
            specificity = 1.0 - (entropy / np.log(len(expr_profile)))
            
            # Combine features
            enhanced_features = np.concatenate([
                basic_features,  # 11 cell-type expressions
                [mean_expr, std_expr, max_expr, min_expr, cv, specificity]  # 6 statistical
            ])
            
            gene_to_features[symbol] = enhanced_features
            gene_to_ensembl[symbol] = ensembl_id
    
    print(f"✓ Mapped {len(gene_to_features)} genes with enhanced features")
    print(f"  Feature dimension: {len(next(iter(gene_to_features.values())))}")
    
    return gene_to_features, gene_to_ensembl


def load_and_process_data(data_dir: Path = Path("data")):
    """
    Complete data loading and processing pipeline
    """
    print("═══════════════════════════════════════════════════════════")
    print("    COMPREHENSIVE DATA PROCESSING & FEATURE ENGINEERING")
    print("═══════════════════════════════════════════════════════════\n")
    
    # 1. Load prior knowledge
    print("1. Loading prior knowledge...")
    with open(data_dir / "priors" / "merged_priors.json") as f:
        priors = json.load(f)
    
    prior_tfs = set(priors.keys())
    prior_genes = set()
    for tf, targets in priors.items():
        prior_genes.update(targets)
    
    print(f"   TFs: {len(prior_tfs)}")
    print(f"   Genes: {len(prior_genes)}")
    print(f"   Total unique: {len(prior_tfs | prior_genes)}\n")
    
    # 2. Load expression data
    print("2. Loading expression data...")
    expr_dir = data_dir / "processed" / "expression" / "fe1a73ab-a203-45fd-84e9-0f7fd19efcbd"
    
    with open(expr_dir / "genes.txt") as f:
        expr_genes = [line.strip() for line in f]
    
    expr_matrix = np.load(expr_dir / "pseudobulk_expression.npy")
    print(f"   Expression shape: {expr_matrix.shape}")
    print(f"   Genes: {len(expr_genes)}\n")
    
    # 3. Create/load gene ID mapper
    print("3. Setting up gene ID mapper...")
    mapper = GeneIDMapper()
    
    # Try loading existing mapping
    mapper.load_from_files(expr_dir)
    
    # If no mapping or incomplete, try downloading
    if len(mapper.ensembl_to_symbol) < 1000:
        print("   Existing mapping incomplete, downloading from BioMart...")
        success = mapper.load_from_biomart()
        if success:
            mapper.save_to_files(expr_dir)
    
    # 4. Create enhanced features
    gene_to_features, gene_to_ensembl = create_enhanced_expression_features(
        expr_matrix, expr_genes, mapper
    )
    
    # 5. Map prior genes to features
    print("\n4. Mapping prior genes to expression features...")
    
    tf_mapped = 0
    gene_mapped = 0
    
    tf_features = {}
    gene_features = {}
    
    for tf in prior_tfs:
        symbol = mapper.map_gene(tf)
        if symbol and symbol in gene_to_features:
            tf_features[tf] = gene_to_features[symbol]
            tf_mapped += 1
    
    for gene in prior_genes:
        symbol = mapper.map_gene(gene)
        if symbol and symbol in gene_to_features:
            gene_features[gene] = gene_to_features[symbol]
            gene_mapped += 1
    
    print(f"   TFs mapped: {tf_mapped}/{len(prior_tfs)} ({100*tf_mapped/len(prior_tfs):.1f}%)")
    print(f"   Genes mapped: {gene_mapped}/{len(prior_genes)} ({100*gene_mapped/len(prior_genes):.1f}%)")
    
    # 6. Create fallback for unmapped genes (zeros or mean)
    feature_dim = len(next(iter(gene_to_features.values())))
    mean_features = np.mean(list(gene_to_features.values()), axis=0)
    
    for tf in prior_tfs:
        if tf not in tf_features:
            tf_features[tf] = np.zeros(feature_dim)  # or mean_features
    
    for gene in prior_genes:
        if gene not in gene_features:
            gene_features[gene] = np.zeros(feature_dim)
    
    print(f"\n   ✓ All {len(prior_tfs)} TFs have features (unmapped=zeros)")
    print(f"   ✓ All {len(prior_genes)} genes have features (unmapped=zeros)\n")
    
    # 7. Save processed features
    print("5. Saving processed features...")
    processed_dir = data_dir / "processed" / "features"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to serializable format
    tf_features_serializable = {k: v.tolist() for k, v in tf_features.items()}
    gene_features_serializable = {k: v.tolist() for k, v in gene_features.items()}
    
    with open(processed_dir / "tf_features.json", "w") as f:
        json.dump(tf_features_serializable, f)
    
    with open(processed_dir / "gene_features.json", "w") as f:
        json.dump(gene_features_serializable, f)
    
    with open(processed_dir / "feature_info.json", "w") as f:
        json.dump({
            "feature_dim": feature_dim,
            "cell_type_dim": 11,
            "statistical_dim": 6,
            "tf_mapped": tf_mapped,
            "gene_mapped": gene_mapped,
            "total_tfs": len(prior_tfs),
            "total_genes": len(prior_genes),
            "mapping_rate_tfs": tf_mapped / len(prior_tfs),
            "mapping_rate_genes": gene_mapped / len(prior_genes)
        }, f, indent=2)
    
    print(f"   ✓ Saved to {processed_dir}/\n")
    
    # 8. Summary
    print("═══════════════════════════════════════════════════════════")
    print("SUMMARY:")
    print("═══════════════════════════════════════════════════════════")
    print(f"✓ Feature dimension: {feature_dim}")
    print(f"  - Cell-type expressions: 11")
    print(f"  - Statistical features: 6 (mean, std, max, min, CV, specificity)")
    print(f"\n✓ Mapping success:")
    print(f"  - TFs: {tf_mapped}/{len(prior_tfs)} ({100*tf_mapped/len(prior_tfs):.1f}%)")
    print(f"  - Genes: {gene_mapped}/{len(prior_genes)} ({100*gene_mapped/len(prior_genes):.1f}%)")
    print(f"\n✓ Unmapped genes: Using zero features")
    print(f"\n✓ Ready for training with enhanced features!")
    print("═══════════════════════════════════════════════════════════\n")
    
    return tf_features, gene_features, feature_dim


if __name__ == "__main__":
    load_and_process_data()
