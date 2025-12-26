#!/usr/bin/env python3
import json

# Load gene names from expression data
with open('data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd/genes.txt', 'r') as f:
    expr_genes = [line.strip() for line in f.readlines()]

print(f"Expression genes: {len(expr_genes)}")
print(f"Sample genes (first 10): {expr_genes[:10]}")

# Load prior knowledge
with open('data/priors/merged_priors.json', 'r') as f:
    priors = json.load(f)

# Get TFs and genes from priors
all_genes_in_priors = set()
for tf, targets in priors.items():
    all_genes_in_priors.update(targets)

print(f"\nPriors TFs: {len(priors)}")
print(f"Priors genes: {len(all_genes_in_priors)}")
print(f"Sample TFs: {list(priors.keys())[:10]}")
print(f"Sample genes: {list(all_genes_in_priors)[:10]}")

# Check overlap
expr_genes_set = set(expr_genes)
overlap_genes = expr_genes_set & all_genes_in_priors
print(f"\nOverlap genes: {len(overlap_genes)} ({100*len(overlap_genes)/len(all_genes_in_priors):.1f}%)")

if len(overlap_genes) > 0:
    print(f"Example overlapping genes: {list(overlap_genes)[:5]}")
