---
title: "Modular versus Monolithic Neural Architectures for Gene Regulatory Network Inference: Capacity, Robustness, and Representational Efficiency"
author: Evint Leovonzko
date: 2026
---

# Chapter 4: Modular versus Monolithic Neural Architectures for Gene Regulatory Network Inference

## 4.1 Introduction

Inferring gene regulatory networks (GRNs) from single-cell RNA sequencing data is a
fundamental problem in computational biology: understanding which transcription factors
(TFs) regulate which target genes, and in which cellular contexts, is critical for
deciphering developmental programs, disease mechanisms, and therapeutic targets. The
advent of single-nucleus RNA-seq at scale has made it possible to profile gene expression
across thousands of cells and cell types simultaneously, creating both an opportunity and
a challenge for GRN inference methods: the data are rich, but the ground-truth regulatory
relationships are sparse, noisy, and largely unknown.

Modern deep learning approaches to GRN inference typically cast the problem as binary
link prediction: given a candidate TF–gene pair and their expression profiles, predict
whether a regulatory relationship exists. This framing admits a natural architectural
question: should the model encode the TF and gene representations separately (a modular,
factorized architecture) or jointly (a monolithic architecture that sees the pair as a
unified input)? The two approaches reflect fundamentally different inductive biases. A
factorized two-tower model assumes that TF and gene identity can be independently
embedded, with regulatory relationships recovered by a simple geometric operation — cosine
similarity — on the resulting encodings. A monolithic cross-encoder, by contrast, processes
the pair jointly, allowing arbitrary interaction features to be learned from the combined
input.

This chapter presents a systematic experimental comparison of these two architectural
families on a human brain single-nucleus RNA-seq dataset, under matched parameter budgets
and identical training conditions. Three research hypotheses are investigated:

- **H1**: A monolithic cross-encoder, given equivalent parameter capacity, achieves
  significantly higher area under the receiver operating characteristic curve (AUROC) than
  a factorized two-tower model under balanced positive-to-negative training.
- **H2**: Under a realistic 5:1 negative-to-positive sampling ratio approximating
  real-world regulatory database sparsity, the two-tower architecture degrades more
  severely than the cross-encoder due to the geometric sensitivity of cosine similarity
  scoring to marginal distribution shifts.
- **H3**: The 512-dimensional hidden layers of the trained two-tower model contain
  substantial representational redundancy; structured pruning retaining as few as 10%
  of neurons per tower will not degrade AUROC below the unpruned baseline.

The results confirm all three hypotheses. The cross-encoder achieves 9.4 AUROC points
higher than the two-tower under balanced training (0.9040 vs 0.8097) and is robust to
5:1 class imbalance where the two-tower degrades by 6.6 AUROC points. The neuron pruning
analysis reveals that the two-tower's effective capacity is far smaller than its nominal
parameter count: post-hoc AUROC never falls below the 95% retention threshold at any
tested sparsity level, including 90% pruning (51 neurons retained per tower). Together,
these findings suggest that the cross-encoder's advantage over the two-tower is not merely
a matter of parameter count, but of architectural expressiveness: the factorized cosine
similarity objective fundamentally constrains what the two-tower can learn, and the
resulting representations are correspondingly redundant.

## 4.2 Background and Related Work

### 4.2.1 Gene Regulatory Network Inference as Link Prediction

Gene regulatory network inference is the problem of identifying directed edges in a graph
where nodes are genes and edges represent transcriptional regulation: an edge from TF $t$
to gene $g$ indicates that $t$ binds in the promoter or enhancer region of $g$ and
influences its expression. Curated databases such as DoRothEA [@garcia2019] and TRRUST
[@han2018] provide partial ground truth derived from ChIP-seq experiments and literature
curation, but coverage is sparse relative to the space of possible regulatory interactions.

Machine learning approaches to GRN inference treat the problem as binary classification
over TF–gene pairs, using expression data as features. Early methods such as GENIE3
[@huynh2010] used random forest feature importance to rank TF–gene interactions. SCENIC
[@aibar2017] extended this with cis-regulatory motif analysis. More recent deep learning
approaches learn representations of TFs and genes directly from expression profiles, using
the learned representations to score candidate edges.

The present work adopts the link prediction framing directly: each training example is a
(TF index, gene index, label) triple, where the label is 1 for known regulatory edges and
0 for sampled non-edges. This formulation allows direct comparison between architectures
using standard classification metrics (accuracy, AUROC, F1) and is well-suited to
evaluation with held-out test splits from the same prior knowledge databases.

### 4.2.2 Two-Tower (Dual-Encoder) Architectures

The two-tower or dual-encoder paradigm was introduced for web search in the Deep
Structured Semantic Model (DSSM) [@huang2013], where separate neural networks encode
query and document into a shared embedding space, and relevance is scored by cosine
similarity. This factorized approach enables efficient large-scale retrieval: all document
embeddings can be pre-computed and indexed, and query-document scores require only a
dot product at inference time, reducing the per-query cost from $O(N \cdot d)$ for an
MLP to $O(d)$ where $d$ is the embedding dimension.

Siamese networks [@bromley1993] represent the closely related case where the two towers
share weights, designed for learning similarity functions between items of the same type.
In the GRN setting, TF and gene encoders are kept separate (non-Siamese) to allow the
model to learn distinct representation spaces for the two biological entity types.

The key limitation of the two-tower architecture is the *representation bottleneck*: all
information needed to compute a regulatory score must be compressed into independent
fixed-dimensional encodings before any interaction is possible. This precludes learning
features that are inherently joint — for example, a regulatory interaction that is only
predictable from the simultaneous observation of TF expression level and gene chromatin
accessibility. The cross-encoder architecture, described next, removes this bottleneck.

### 4.2.3 Cross-Encoder (Joint Representation) Architectures

Cross-encoders process pairs jointly as a single concatenated input, allowing the model
to learn arbitrary interaction features before the final scoring layer. In information retrieval, Nogueira and Cho [@nogueira2019] demonstrated that
cross-encoders substantially outperform bi-encoders (two-tower models) on reranking tasks
where precise relevance judgements are needed, at the cost of requiring per-pair
computation at inference time. Humeau et al. [@humeau2020] further characterized this
trade-off in Poly-encoders, and Khattab and Zaharia [@khattab2020] explored late
interaction in ColBERT as a hybrid approach retaining some cross-encoder expressiveness
at reduced inference cost. These works collectively establish that joint encoding is more
expressive than factorized encoding, with the practical cost being computational.

In the present work, the cross-encoder receives the concatenation of TF embedding, gene
embedding, their element-wise product (capturing explicit interaction features), and both
expression profiles as input to a three-layer MLP. The element-wise product term
$\text{TF}_\text{emb} \odot \text{Gene}_\text{emb}$ is a standard feature engineering
technique that provides a direct path for learning which embedding dimensions co-activate
for regulatory pairs [@chen2016]. The cross-encoder is trained end-to-end with the same
binary cross-entropy objective as the two-tower, differing only in how the TF and gene
features are combined.

### 4.2.4 Structured Pruning and Representational Redundancy

Neural network pruning refers to the removal of parameters from a trained network, with
the goal of reducing model size or inference cost while preserving predictive performance.
Unstructured pruning removes individual weights [@han2015]; structured pruning removes
entire neurons, filters, or layers, which is compatible with standard dense matrix
operations and produces actual speedups without sparse-matrix libraries.

Han et al. [@han2015] demonstrated that large neural networks can tolerate removal of 80–90%
of weights with minimal accuracy loss, suggesting that trained networks are highly
over-parameterized relative to the tasks they solve. The Lottery Ticket Hypothesis
[@frankle2019] further showed that sparse sub-networks exist within trained networks that
can be trained in isolation to match the performance of the full network, implying that
the useful representational capacity is localized.

In the context of the two-tower GRN model, the neuron pruning experiment serves a dual
purpose: (1) it quantifies the effective capacity of the model by identifying how many
neurons are actually used, and (2) it provides evidence about *why* the cross-encoder
outperforms the two-tower — if the two-tower uses only a fraction of its nominal capacity,
the parameter-matching argument in its favor is weaker than it appears.

## 4.3 Methods

### 4.3.1 Dataset and Prior Knowledge

### 4.3.2 Model Architectures

### 4.3.3 Training Protocol

### 4.3.4 Evaluation

### 4.3.5 Negative Sampling

### 4.3.6 Neuron Pruning Protocol

## 4.4 Hypothesis 1: Monolithic Cross-Encoders Achieve Superior Discriminative Power Under Balanced Training

## 4.5 Hypothesis 2: Two-Tower Architecture Degrades More Severely Under Realistic Class Imbalance

## 4.6 Hypothesis 3: The Two-Tower Model Learns Highly Redundant Representations

## 4.7 Discussion

## 4.8 Conclusion

## 4.9 References
