#!/usr/bin/env python3
"""
Classical ML baselines using expression features only.
Provides baseline comparison for the paper (addresses missing baseline context).
"""
import json, numpy as np, sys
from pathlib import Path

expr_dir    = Path("data/processed/expression/fe1a73ab-a203-45fd-84e9-0f7fd19efcbd")
priors_path = Path("data/priors/merged_priors.json")

# ── Load expression ───────────────────────────────────────────────────────────
with open(expr_dir/"ensembl_to_symbol.json") as f: e2s = json.load(f)
with open(expr_dir/"uniprot_to_symbol.json") as f: u2s = json.load(f)
with open(expr_dir/"genes.txt") as f: expr_ensembl = [l.strip() for l in f]
expr_matrix = np.load(expr_dir/"pseudobulk_expression.npy")   # (11, 2000)

# Build symbol -> column-index map
sym_to_col = {}
for col, ensg in enumerate(expr_ensembl):
    sym = e2s.get(ensg, ensg)
    sym_to_col[sym] = col
    sym_to_col[ensg] = col  # keep ensembl as fallback

n_cell = expr_matrix.shape[0]  # 11

def get_expr(name):
    # Try direct, then uniprot, then zeros
    for key in [name, u2s.get(name, None)]:
        if key and key in sym_to_col:
            return expr_matrix[:, sym_to_col[key]].astype(np.float32)
    return np.zeros(n_cell, dtype=np.float32)

# ── Load priors ───────────────────────────────────────────────────────────────
with open(priors_path) as f: priors = json.load(f)  # {TF: [gene, ...], ...}

tf_list   = sorted(priors.keys())
gene_set  = sorted({g for targets in priors.values() for g in targets})
positive_pairs = {(tf, gene) for tf, targets in priors.items() for gene in targets}
print(f"TFs: {len(tf_list)} | Genes: {len(gene_set)} | Positives: {len(positive_pairs)}")

# check expression coverage
tf_cov   = sum(1 for t in tf_list if get_expr(t).sum() > 0)
gene_cov = sum(1 for g in gene_set if get_expr(g).sum() > 0)
print(f"Expression coverage: TFs {tf_cov}/{len(tf_list)} ({tf_cov/len(tf_list)*100:.1f}%), "
      f"Genes {gene_cov}/{len(gene_set)} ({gene_cov/len(gene_set)*100:.1f}%)")

# ── Build dataset ─────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
n_pos = len(positive_pairs)
positives = list(positive_pairs)

# Sample negatives
all_neg = [(tf, g) for tf in tf_list for g in gene_set if (tf, g) not in positive_pairs]
neg_idx  = rng.choice(len(all_neg), size=n_pos, replace=False)
negatives = [all_neg[i] for i in neg_idx]

all_pairs  = positives + negatives
all_labels = [1]*n_pos + [0]*n_pos

perm = rng.permutation(len(all_pairs))
all_pairs  = [all_pairs[i]  for i in perm]
all_labels = [all_labels[i] for i in perm]

# Features: [tf_expr(11d) | gene_expr(11d)] = 22-dim
X = np.stack([np.concatenate([get_expr(tf), get_expr(g)]) for tf, g in all_pairs]).astype(np.float32)
y = np.array(all_labels, dtype=np.int32)
print(f"Dataset: {X.shape[0]} examples, {X.shape[1]} features | Positive rate: {y.mean():.3f}")

# ── Split (same ratios as neural models: 70/15/15 with seed 42) ───────────────
n_total = len(y)
n_train = int(n_total * 0.70)
n_val   = int(n_total * 0.15)
X_tr, y_tr = X[:n_train], y[:n_train]
X_te, y_te = X[n_train+n_val:], y[n_train+n_val:]
print(f"Train: {len(X_tr)} | Test: {len(X_te)}")

# ── Evaluation ────────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

def evaluate(name, probs, labels):
    preds = (probs >= 0.5).astype(int)
    acc   = accuracy_score(labels, preds)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)
    f1    = f1_score(labels, preds, average='macro')
    print(f"  [{name}]  Acc={acc*100:.2f}%  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  macroF1={f1:.4f}")
    return {"model": name, "accuracy": float(acc), "auroc": float(auroc),
            "auprc": float(auprc), "macro_f1": float(f1)}

results = []
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("\n=== Classical ML Baselines (expression-only, 22-dim features) ===")

# Majority class (sanity check)
majority = np.full(len(y_te), y_tr.mean())
results.append(evaluate("Majority-class",        majority, y_te))

# Logistic Regression
lr = Pipeline([("sc", StandardScaler()),
               ("lr", LogisticRegression(max_iter=500, C=1.0, random_state=42))])
lr.fit(X_tr, y_tr)
results.append(evaluate("Logistic Regression",   lr.predict_proba(X_te)[:,1], y_te))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
results.append(evaluate("Random Forest (n=200)", rf.predict_proba(X_te)[:,1], y_te))

# Gradient Boosted Trees
gbt = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42)
gbt.fit(X_tr, y_tr)
results.append(evaluate("Gradient Boosted Trees (n=200)", gbt.predict_proba(X_te)[:,1], y_te))

print("\nContext (from neural models with 512-dim learned embeddings + expression):")
print("  Two-Tower MLP  (5.58M params): AUROC=0.7941  Acc=79.42%")
print("  Cross-Encoder  (5.58M params): AUROC=0.9025  Acc=83.46%")
print("\nNote: classical baselines use ONLY the 22-dim expression features (no learned embeddings).")

out = {
    "description": "Classical ML baselines on expression-only features (11-dim cell-type profiles per entity, 22-dim pair features). No learned embeddings. Same 70/15/15 split.",
    "feature_dim": int(X.shape[1]),
    "test_size":   int(len(y_te)),
    "positive_rate_test": float(y_te.mean()),
    "results": results,
    "neural_reference": {
        "two_tower_auroc": 0.7941, "two_tower_acc": 0.7942,
        "cross_encoder_auroc": 0.9025, "cross_encoder_acc": 0.8346,
        "note": "Neural models use 512-dim learned embeddings + 11-dim expression features (523-dim per entity)"
    }
}
Path("results").mkdir(exist_ok=True)
with open("results/classical_baselines.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/classical_baselines.json")
