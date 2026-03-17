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

Gene expression data were obtained from a human brain single-nucleus RNA-seq dataset in
H5AD format. Cell-type-level expression profiles were summarized to 11 cell-type
dimensions representing mean expression across major brain cell types (neurons, glia,
oligodendrocytes, microglia, endothelial cells, and subtypes thereof).

Prior regulatory knowledge was sourced from merged DoRothEA [@garcia2019] and TRRUST
[@han2018] databases, yielding approximately 29,000 TF–gene pairs as positive (regulatory)
examples. An equal number of negative (non-regulatory) examples were sampled uniformly
from the space of TF–gene pairs not present in the prior databases, producing a balanced
dataset of approximately 39,000 total examples. The dataset was split into training
(70%), validation (15%), and test (15%) sets using a fixed random seed (42) applied once
before any model training, ensuring that all model comparisons use identical data splits.

### 4.3.2 Model Architectures

**Two-Tower MLP.** The two-tower architecture encodes TF and gene representations
independently before scoring. Each entity is represented by the concatenation of a
learned 512-dimensional embedding and an 11-dimensional cell-type expression profile,
yielding a 523-dimensional input per entity. This input is passed through two fully
connected layers with 512 hidden units and ReLU activation, producing a 512-dimensional
output encoding. Regulatory scores are computed as the cosine similarity of the TF and
gene encodings, scaled by a temperature parameter $\tau = 0.05$ and passed through a
sigmoid function to produce probabilities. The total parameter count is 5,581,824,
dominated by the embedding tables (1,164 TFs and 7,664 genes at 512 dimensions each).

**Cross-Encoder MLP.** The cross-encoder receives TF and gene features jointly. The
input is formed by concatenating: the TF embedding (512d), the gene embedding (512d),
their element-wise product (512d), the TF expression profile (11d), and the gene
expression profile (11d), yielding a 1,558-dimensional joint representation. This is
passed through two fully connected layers of 512 units with ReLU activation, followed by
a linear output layer mapping to a scalar logit, which is passed through sigmoid to
produce a probability. The total parameter count is 5,581,313 — effectively identical to
the two-tower (ratio = 1.000), ensuring a parameter-matched comparison.

Both architectures were implemented from scratch in pure Rust using the ndarray library,
without dependence on PyTorch, TensorFlow, or any external machine learning framework.
Figure 4.1 illustrates the two architectures side by side.

![Architecture comparison: Two-Tower MLP (left) versus Cross-Encoder MLP (right). Both architectures use identical embedding tables and share comparable parameter counts (5.58M). The key difference is that the cross-encoder processes TF and gene features jointly, including an explicit element-wise interaction term, while the two-tower scores interactions via cosine similarity of independently computed encodings.](figures/fig4_1_architecture.svg)

### 4.3.3 Training Protocol

All models were trained with the Adam optimizer [@kingma2014] with learning rate
$\eta = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and gradient clipping at a global
norm of 5.0. The batch size was 256 examples. **Note on learning rate:** Phase 8
experiments use $\eta = 0.001$, which differs from the $\eta = 0.005$ used in Phases
2–7 of this dissertation. The lower learning rate was adopted for Phase 8 to match the
cross-encoder's training setup and ensure a fair architectural comparison; results from
Phases 2–7 (single-model accuracy 80.14%, ensemble 83.06%) are therefore not directly
comparable to the Phase 8 two-tower results reported in this chapter. Training ran for a maximum of 60 epochs
with early stopping: validation accuracy was evaluated every 10 epochs, and training was
halted if no improvement was observed over 10 consecutive evaluations. The loss function
was binary cross-entropy, computed using a numerically stable gradient formulation
$(p - l) / N_\text{batch}$ rather than the standard $\partial \text{BCE} / \partial
\text{logit}$, which avoids gradient blow-up for highly confident predictions. All
weights were initialized with normal distribution $\mathcal{N}(0, 0.01)$.

### 4.3.4 Evaluation

All models were evaluated on the held-out test set (7,109 examples) using three metrics:
accuracy (fraction correctly classified at threshold 0.5), area under the receiver
operating characteristic curve (AUROC), and macro F1 score. Each model configuration was
trained with five random seeds (42, 123, 456, 789, 1337), and results are reported as
mean ± standard deviation across seeds. Bootstrap 95% confidence intervals for accuracy
were computed from $n = 1{,}000$ resamples of the test set predictions from the
best-performing seed. All metrics are reported to four decimal places for consistency.
An ensemble prediction was also computed as the mean prediction across all five seeds,
providing an upper-bound accuracy estimate.

### 4.3.5 Negative Sampling

Two negative sampling regimes were evaluated:

- **Balanced (1:1):** equal numbers of positive and negative examples, matching the
  typical assumption in benchmark evaluations of GRN inference methods.
- **Realistic (5:1):** five negative examples per positive example, reflecting the
  approximate ratio of non-edges to edges in curated regulatory databases relative to
  the space of possible TF–gene pairs.

The 5:1 regime provides a more ecologically valid test of model robustness, since deployed
GRN inference methods must operate on data where regulatory interactions are rare.

### 4.3.6 Neuron Pruning Protocol

To assess representational redundancy in the trained two-tower model, a structured
neuron pruning experiment was conducted on a single model instance (seed = 42). Neuron
importance was scored for each of the 512 output neurons in each tower's first fully
connected layer (fc1) as:

$$\text{importance}(j) = \alpha \cdot \text{activation\_freq}(j) + (1 - \alpha) \cdot \text{weight\_magnitude}(j)$$

where $\alpha = 0.5$ (equal weighting), $\text{activation\_freq}(j)$ is the fraction of
training examples for which neuron $j$'s post-ReLU activation exceeds zero, and
$\text{weight\_magnitude}(j)$ is the mean of the normalized L2 norms of column $j$ of
the fc1 weight matrix and row $j$ of the fc2 weight matrix. Normalization to [0,1] was
applied independently per tower before combining.

Pruning was applied at 13 sparsity levels: $\{0\%, 5\%, 10\%, 15\%, 20\%, 25\%, 30\%,
40\%, 50\%, 60\%, 70\%, 80\%, 90\%\}$. At each level, the lowest-importance neurons were
removed by deleting the corresponding columns from fc1's weight matrix and rows from
fc2's weight matrix. Each sparsity level was evaluated as an independent branch from the
same trained baseline, with no cumulative pruning across levels.

Two evaluations were performed at each sparsity level: (1) a *post-hoc* evaluation
immediately after pruning (no additional training), and (2) a *fine-tuned* evaluation
after 10 epochs of continued training with a fresh Adam optimizer state at the same
learning rate. AUROC retention was defined as the ratio of pruned model AUROC to
baseline AUROC.

## 4.4 Hypothesis 1: Monolithic Cross-Encoders Achieve Superior Discriminative Power Under Balanced Training

## 4.5 Hypothesis 2: Two-Tower Architecture Degrades More Severely Under Realistic Class Imbalance

## 4.6 Hypothesis 3: The Two-Tower Model Learns Highly Redundant Representations

## 4.7 Discussion

## 4.8 Conclusion

## 4.9 References
