# TNNLS Journal Paper Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `paper/paper_tnnls.tex` — a fully restructured IEEE TNNLS journal paper from the dissertation chapter `paper/chapter_phase8.md` — and compile it to `paper/paper_tnnls.pdf`.

**Architecture:** Single LaTeX file using `\documentclass[journal]{IEEEtran}`. All content is sourced from `paper/chapter_phase8.md` (read it before starting). New sections (Introduction, Conclusion, Abstract) are written from scratch per TNNLS conventions; existing sections are condensed with specific cuts listed per task.

**Tech Stack:** LaTeX (IEEEtran), tectonic (compiler), existing `paper/chapter_phase8.bib` (BibTeX), existing PNG figures in `paper/figures/`.

**Spec:** `docs/superpowers/specs/2026-03-18-tnnls-paper-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `paper/paper_tnnls.tex` | Create | Full TNNLS journal paper LaTeX source |
| `paper/paper_tnnls.pdf` | Generated | Compiled PDF via tectonic |
| `paper/chapter_phase8.bib` | Read-only | Existing bibliography — do not modify |
| `paper/figures/fig4_*.png` | Read-only | Existing figures — do not modify |
| `paper/chapter_phase8.md` | Read-only | Source content — do not modify |

---

## Chunk 1: Document Scaffold

### Task 1: Create preamble and front matter

**Files:**
- Create: `paper/paper_tnnls.tex`

- [ ] **Step 1: Create the file with the complete preamble, title, abstract, and index terms**

```latex
\documentclass[journal]{IEEEtran}

% Packages
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{cite}

\begin{document}

\title{Modular versus Monolithic Neural Architectures for Gene Regulatory
Network Inference: Capacity, Robustness, and Representational Efficiency}

\author{Evint Leovonzko%
\thanks{E. Leovonzko is with [Department], [University], [City, Country].
E-mail: [email]. Manuscript received 2026.}}

\maketitle

\begin{abstract}
Gene regulatory network (GRN) inference — determining which transcription
factors regulate which target genes from single-cell RNA-sequencing data — is
a core problem in computational biology. A fundamental architectural choice is
whether to encode transcription factor (TF) and gene representations
independently (two-tower/dual-encoder) or jointly (cross-encoder). We present
a parameter-matched comparison of both architectures on a human brain
single-nucleus RNA-seq dataset (39K TF--gene pairs, 5.58M parameters each)
under balanced (1:1) and realistic (5:1) negative-to-positive sampling ratios.
The cross-encoder achieves an AUROC of 0.9040 versus 0.8097 for the two-tower
under balanced training — a gap of 9.4 points consistent across five random
seeds. Under 5:1 negative sampling, the two-tower AUROC degrades by 6.6 points
(0.8097~$\to$~0.7434) while the cross-encoder remains stable (0.9040~$\to$~0.9150),
demonstrating that cosine similarity scoring is geometrically sensitive to
class imbalance. A structured neuron pruning analysis on the trained two-tower
model shows that post-hoc AUROC retention never falls below 99.96\% of
baseline at any sparsity level tested up to 90\% (51 of 512 neurons retained
per tower), revealing that the two-tower fails to utilize its nominal
parameter budget. This redundancy provides a mechanistic account of the
cross-encoder's advantage and suggests that a 90\%-sparse two-tower matches
baseline performance without any fine-tuning.
\end{abstract}

\begin{IEEEkeywords}
gene regulatory network inference, two-tower architecture, cross-encoder,
dual-encoder, neural network pruning, single-cell RNA-seq, representation
learning, link prediction, class imbalance
\end{IEEEkeywords}
```

- [ ] **Step 2: Verify compile**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | tail -5
```

Expected: warnings about missing sections are fine; no fatal errors.

---

### Task 2: Write Section I — Introduction

**Files:**
- Modify: `paper/paper_tnnls.tex`

This section is written from scratch (not copied from the dissertation). Append after the `\begin{IEEEkeywords}...\end{IEEEkeywords}` block.

- [ ] **Step 1: Append the Introduction section**

```latex
% ── I. INTRODUCTION ──────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}

Gene regulatory network (GRN) inference — the problem of identifying which
transcription factors (TFs) regulate which target genes — is a fundamental
challenge in computational biology. Understanding transcriptional regulatory
relationships is essential for deciphering developmental programs, disease
mechanisms, and therapeutic targets. With the availability of large-scale
single-nucleus RNA-sequencing (snRNA-seq) data, machine learning methods have
emerged that cast GRN inference as binary link prediction: given a candidate
TF--gene pair and their expression profiles, predict whether a regulatory
relationship exists~\cite{aibar2017,huynh2010}.

A central architectural question in link prediction is whether entity
representations should be computed independently (a \emph{two-tower} or
dual-encoder model~\cite{huang2013,bromley1993}) or jointly (a
\emph{cross-encoder}~\cite{nogueira2019,humeau2020}). Two-tower models score
interactions via cosine similarity of independently computed encodings,
enabling efficient inference at scale. Cross-encoders process pairs jointly,
allowing arbitrary interaction features to be learned before scoring. In
information retrieval, cross-encoders consistently outperform two-tower models
when precise ranking quality is needed~\cite{nogueira2019,humeau2020}, but
whether this advantage holds in computational biology — where entities have
distinct biological identities and the regulatory signal is low-dimensional —
has not been established under controlled, parameter-matched conditions.

Three open questions motivate this work: (1)~Does a cross-encoder achieve
higher discriminative AUROC than a parameter-matched two-tower for TF--gene
link prediction? (2)~Is the two-tower's cosine similarity scoring disproportionately
sensitive to class imbalance relative to a cross-encoder? (3)~Does the trained
two-tower actually utilize its nominal hidden-layer capacity, or is much of it
redundant?

This paper makes three contributions:
\begin{itemize}
\item \textbf{C1:} A parameter-matched comparison (5.58M parameters each)
  of two-tower and cross-encoder MLPs on human brain snRNA-seq data under
  both balanced (1:1) and realistic (5:1) negative-to-positive sampling ratios.
\item \textbf{C2:} Demonstration that the two-tower degrades by 6.6~AUROC
  points under 5:1 negative sampling — approximating real-world regulatory
  database sparsity — while the cross-encoder remains stable.
\item \textbf{C3:} A structured neuron pruning analysis showing that
  post-hoc AUROC retention is $\geq$0.9996 at all sparsity levels up to
  90\%, revealing that the two-tower underutilizes its parameter budget
  due to the geometric constraints of cosine similarity scoring.
\end{itemize}

The remainder of this paper is organized as follows.
Section~\ref{sec:related} reviews related work.
Section~\ref{sec:methods} describes the dataset, architectures, training
protocol, and pruning procedure.
Section~\ref{sec:results} presents experimental results.
Section~\ref{sec:discussion} discusses implications and limitations.
Section~\ref{sec:conclusion} concludes.
```

- [ ] **Step 2: Compile to verify**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "feat: add IEEEtran scaffold, abstract, and introduction"
```

---

## Chunk 2: Related Work and Methods

### Task 3: Write Section II — Related Work

**Files:**
- Modify: `paper/paper_tnnls.tex`

Source: `paper/chapter_phase8.md` sections 4.2.1–4.2.4. Read those sections first, then apply the condensing rules below.

- [ ] **Step 1: Append the Related Work section**

Condense each subsection from the dissertation as follows:

**II-A** (from §4.2.1): Keep paragraphs 1 and 2. **Remove** the third paragraph beginning "The present work adopts the link prediction framing directly..." (this is method description, belongs in Methods).

**II-B** (from §4.2.2): Keep paragraphs 1 and 2. In paragraph 3 (the key limitation paragraph), **remove the last sentence** "The cross-encoder architecture, described next, removes this bottleneck." (forward reference, unnecessary).

**II-C** (from §4.2.3): Keep paragraph 1 and paragraph 2. **Remove** paragraph 3 beginning "In the present work, the cross-encoder receives the concatenation..." (this is method description, belongs in Methods III-B).

**II-D** (from §4.2.4): Keep paragraphs 1 and 2. **Remove** paragraph 3 beginning "In the context of the two-tower GRN model, the neuron pruning experiment serves a dual purpose..." (this belongs in the Introduction motivation or Section III-F).

```latex
% ── II. RELATED WORK ─────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}

\subsection{Gene Regulatory Network Inference as Link Prediction}

Gene regulatory network inference is the problem of identifying directed edges
in a graph where nodes are genes and edges represent transcriptional
regulation. Curated databases such as DoRothEA~\cite{garcia2019} and
TRRUST~\cite{han2018} provide partial ground truth derived from ChIP-seq
experiments and literature curation, but coverage is sparse relative to the
space of possible regulatory interactions.

Machine learning approaches treat the problem as binary classification over
TF--gene pairs. Early methods such as GENIE3~\cite{huynh2010} used random
forest feature importance to rank TF--gene interactions. SCENIC~\cite{aibar2017}
extended this with cis-regulatory motif analysis. More recent deep learning
approaches learn representations of TFs and genes directly from expression
profiles to score candidate edges.

\subsection{Two-Tower (Dual-Encoder) Architectures}

The two-tower paradigm was introduced for web search in the Deep Structured
Semantic Model (DSSM)~\cite{huang2013}, where separate networks encode query
and document into a shared embedding space and relevance is scored by cosine
similarity. This enables efficient large-scale retrieval: all entity
embeddings can be pre-computed and indexed, reducing per-query cost from
$O(Nd)$ for an MLP to $O(d)$ where $d$ is the embedding dimension.

Siamese networks~\cite{bromley1993} represent the closely related case where
the two towers share weights. In the GRN setting, TF and gene encoders are
kept separate (non-Siamese) to allow distinct representation spaces for the
two biological entity types.

The key limitation of the two-tower architecture is the
\emph{representation bottleneck}: all information needed to compute a
regulatory score must be compressed into independent fixed-dimensional
encodings before any interaction is possible. This precludes learning
features that are inherently joint --- for example, regulatory interactions
predictable only from the simultaneous observation of TF expression level and
gene chromatin accessibility.

\subsection{Cross-Encoder Architectures}

Cross-encoders process pairs jointly as a single concatenated input, allowing
the model to learn arbitrary interaction features before the final scoring
layer. Nogueira and Cho~\cite{nogueira2019} demonstrated that cross-encoders
substantially outperform bi-encoders on reranking tasks where precise
relevance judgements are needed, at the cost of requiring per-pair computation
at inference time. Humeau et al.~\cite{humeau2020} further characterized this
trade-off in Poly-encoders, and Khattab and Zaharia~\cite{khattab2020}
explored late interaction in ColBERT as a hybrid approach. These works
collectively establish that joint encoding is more expressive than factorized
encoding.

The element-wise product $\mathbf{a} \odot \mathbf{b}$ of two embeddings is
a standard feature engineering technique that provides a direct path for
learning which embedding dimensions co-activate for relevant pairs~\cite{chen2016}.

\subsection{Structured Pruning and Representational Redundancy}

Neural network pruning removes parameters from a trained network to reduce
model size or inference cost while preserving performance. Unstructured pruning
removes individual weights~\cite{han2015}; structured pruning removes entire
neurons or filters, producing actual speedups without sparse-matrix libraries.

Han et al.~\cite{han2015} demonstrated that large networks tolerate removal
of 80--90\% of weights with minimal accuracy loss, suggesting that trained
networks are highly over-parameterized. The Lottery Ticket Hypothesis~\cite{frankle2019}
further showed that sparse sub-networks exist within trained networks that can
be trained in isolation to match the full network, implying that useful
representational capacity is localized.
```

- [ ] **Step 2: Compile to verify**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "docs: add related work section (II)"
```

---

### Task 4: Write Section III — Methods

**Files:**
- Modify: `paper/paper_tnnls.tex`

Source: `paper/chapter_phase8.md` sections 4.3.1–4.3.6. Read those sections before writing. Content is essentially unchanged except for:
- Section labels: `\subsection{...}` with `\label{ssec:...}`
- Fig. 1 for the architecture diagram
- "This chapter" → "This paper" (not present in Methods, but check)
- The LaTeX math equation in III-F stays as-is

- [ ] **Step 1: Append the Methods section with Fig. 1**

```latex
% ── III. METHODS ─────────────────────────────────────────────────────────────
\section{Methods}
\label{sec:methods}

\subsection{Dataset and Prior Knowledge}
\label{ssec:data}

% SOURCE: paper/chapter_phase8.md §4.3.1
% Copy verbatim — no changes needed.
% [PASTE §4.3.1 text here]

\subsection{Model Architectures}
\label{ssec:arch}

% SOURCE: paper/chapter_phase8.md §4.3.2
% Copy verbatim. Replace the Markdown image reference with the LaTeX figure below.
% Remove: "Figure 4.1 illustrates..." → replace with "Fig.~\ref{fig:arch} illustrates..."
% [PASTE §4.3.2 text here, with figure reference updated]

\begin{figure}[!t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig4_1_architecture.png}
  \caption{Two-Tower MLP (left, blue) versus Cross-Encoder MLP (right, red).
    Both use identical embedding tables and 5.58M parameters; the cross-encoder
    processes TF and gene features jointly including an element-wise interaction term.}
  \label{fig:arch}
\end{figure}

\subsection{Training Protocol}
\label{ssec:training}

% SOURCE: paper/chapter_phase8.md §4.3.3
% Copy verbatim — no changes needed.
% [PASTE §4.3.3 text here]

\subsection{Evaluation}
\label{ssec:eval}

% SOURCE: paper/chapter_phase8.md §4.3.4
% Copy verbatim — no changes needed.
% [PASTE §4.3.4 text here]

\subsection{Negative Sampling}
\label{ssec:negsampling}

% SOURCE: paper/chapter_phase8.md §4.3.5
% Copy verbatim. Remove the Markdown bullet list formatting — convert to prose:
% "Two negative sampling regimes were evaluated: balanced (1:1)... and realistic (5:1)..."
% [PASTE §4.3.5 text here, bullets → prose]

\subsection{Neuron Pruning Protocol}
\label{ssec:pruning}

% SOURCE: paper/chapter_phase8.md §4.3.6
% Copy verbatim — no changes needed. The LaTeX equation stays as-is.
% [PASTE §4.3.6 text here]
```

**Important:** Replace every `[PASTE §4.3.X text here]` comment with the actual prose from `paper/chapter_phase8.md`. Do not leave comments in the final file.

- [ ] **Step 2: Compile and verify**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "docs: add methods section (III) with architecture figure"
```

---

## Chunk 3: Results

### Task 5: Write Section IV-A — Results: Balanced Training

**Files:**
- Modify: `paper/paper_tnnls.tex`

Source: `paper/chapter_phase8.md` §4.4 (Hypothesis 1). Key changes:
- Remove the `**Hypothesis.**` paragraph entirely
- Replace "Table 4.1 presents" → "Table~\ref{tab:balanced} presents"
- Replace "Table 4.2" → "Table~\ref{tab:perseed}"
- Replace "Figure 4.2" → "Fig.~\ref{fig:auroc}"
- Remove "Hypothesis 1 is supported." final sentence → replace with "These results support C1."

- [ ] **Step 1: Append the Results section header and IV-A with Table I, Table II, and Fig. 2**

```latex
% ── IV. RESULTS ──────────────────────────────────────────────────────────────
\section{Results}
\label{sec:results}

\subsection{Cross-Encoder vs.\ Two-Tower Under Balanced Training}
\label{ssec:h1}

% SOURCE: paper/chapter_phase8.md §4.4
% OMIT the "**Hypothesis.**" paragraph.
% BEGIN with: "Table~\ref{tab:balanced} presents the performance of both
%   architectures at the 1:1 negative sampling ratio across five random seeds."
% Replace all figure/table cross-references as noted above.
% Replace "Hypothesis 1 is supported." with "These results support C1."
% [PASTE adapted §4.4 prose here]
```

Now append the two tables and the AUROC figure immediately after the prose:

```latex
\begin{table}[!t]
  \caption{Performance at 1:1 Negative Sampling Ratio (Mean $\pm$ Std, 5 Seeds)}
  \label{tab:balanced}
  \centering
  \begin{tabular}{lcccc}
    \hline
    Model & Accuracy & AUROC & F1 & Ensemble \\
    \hline
    Two-Tower    & 80.90\% $\pm$0.59\% & 0.8097          & 0.8073 & 83.47\% \\
    Cross-Encoder & 83.03\% $\pm$0.48\% & \textbf{0.9040} & 0.8310 & 84.02\% \\
    \hline
  \end{tabular}
\end{table}

\begin{table}[!t]
  \caption{Per-Seed AUROC at 1:1 Negative Sampling Ratio}
  \label{tab:perseed}
  \centering
  \begin{tabular}{lcc}
    \hline
    Seed & Two-Tower & Cross-Encoder \\
    \hline
    42   & 0.8093 & 0.9061 \\
    123  & 0.7987 & 0.9054 \\
    456  & 0.8159 & 0.9032 \\
    789  & 0.8113 & 0.9075 \\
    1337 & 0.8133 & 0.8980 \\
    \hline
    Mean & 0.8097 & 0.9040 \\
    \hline
  \end{tabular}
\end{table}

\begin{figure}[!t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig4_2_auroc_comparison.png}
  \caption{AUROC by model and negative sampling ratio across 5 seeds.
    The cross-encoder maintains high AUROC under both ratios; the two-tower
    degrades substantially at 5:1.}
  \label{fig:auroc}
\end{figure}
```

- [ ] **Step 2: Compile**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "docs: add results IV-A (balanced training) with tables I, II and fig 2"
```

---

### Task 6: Write Section IV-B — Results: Class Imbalance

**Files:**
- Modify: `paper/paper_tnnls.tex`

Source: `paper/chapter_phase8.md` §4.5 (Hypothesis 2). Key changes:
- Remove `**Hypothesis.**` paragraph
- Replace "Table 4.3 presents" → "Table~\ref{tab:imbalance} presents"
- Replace "Hypothesis 2 is supported." → "These results support C2."

- [ ] **Step 1: Append IV-B with Table III**

```latex
\subsection{Robustness to Class Imbalance}
\label{ssec:h2}

% SOURCE: paper/chapter_phase8.md §4.5
% OMIT the "**Hypothesis.**" paragraph.
% BEGIN with: "Table~\ref{tab:imbalance} presents the performance of both
%   architectures at both negative sampling ratios..."
% Replace "Hypothesis 2 is supported." with "These results support C2."
% [PASTE adapted §4.5 prose here]
```

```latex
\begin{table}[!t]
  \caption{Performance Across Negative Sampling Ratios (Mean $\pm$ Std, 5 Seeds)}
  \label{tab:imbalance}
  \centering
  \begin{tabular}{llcccc}
    \hline
    Model & Ratio & AUROC & $\Delta$AUROC & F1 & Std(Acc) \\
    \hline
    Two-Tower     & 1:1 & 0.8097 & ---                     & 0.8073 & 0.0059 \\
    Two-Tower     & 5:1 & 0.7434 & $\mathbf{-6.6}$~pts     & 0.6598 & 0.0154 \\
    Cross-Encoder & 1:1 & 0.9040 & ---                     & 0.8310 & 0.0048 \\
    Cross-Encoder & 5:1 & 0.9150 & $\mathbf{+1.1}$~pts     & 0.7825 & 0.0074 \\
    \hline
  \end{tabular}
\end{table}
```

- [ ] **Step 2: Compile**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "docs: add results IV-B (class imbalance) with table III"
```

---

### Task 7: Write Section IV-C — Results: Neuron Pruning

**Files:**
- Modify: `paper/paper_tnnls.tex`

Source: `paper/chapter_phase8.md` §4.6 (Hypothesis 3). Key changes:
- Remove `**Hypothesis.**` paragraph
- Replace "Table 4.4 presents" → "Table~\ref{tab:pruning} presents"
- Replace "Figures 4.3 and 4.4" → "Figs.~\ref{fig:pruning} and~\ref{fig:scatter}"
- Replace "Hypothesis 3 is supported." → "These results support C3."

- [ ] **Step 1: Append IV-C with Table IV and Figs. 3–4**

```latex
\subsection{Representational Redundancy via Neuron Pruning}
\label{ssec:h3}

% SOURCE: paper/chapter_phase8.md §4.6
% OMIT the "**Hypothesis.**" paragraph.
% BEGIN with: "Table~\ref{tab:pruning} presents the post-hoc and fine-tuned
%   AUROC retention across all 13 tested sparsity levels."
% Replace figure/table refs as noted above.
% Replace "Hypothesis 3 is supported." with "These results support C3."
% [PASTE adapted §4.6 prose here]
```

```latex
\begin{table*}[!t]
  \caption{AUROC Retention at Each Sparsity Level. Post-hoc: pruned model evaluated immediately.
    Fine-tuned: 10 further epochs with fresh Adam state. Baseline AUROC = 0.8015.}
  \label{tab:pruning}
  \centering
  \begin{tabular}{lccccccc}
    \hline
    Sparsity & Neurons Kept & Comp.\ Ratio & Post-hoc AUROC & Post-hoc Ret. & Fine-tuned AUROC & Fine-tuned Ret. \\
    \hline
    0\%  & 512 & 1.0000 & 0.8015 & 1.0000 & 0.8199 & 1.0229 \\
    5\%  & 486 & 0.9903 & 0.8015 & 1.0000 & 0.8026 & 1.0013 \\
    10\% & 461 & 0.9811 & 0.8012 & 0.9996 & 0.8045 & 1.0038 \\
    15\% & 435 & 0.9714 & 0.8014 & 0.9998 & 0.7908 & 0.9866 \\
    20\% & 410 & 0.9621 & 0.8014 & 0.9998 & 0.8051 & 1.0044 \\
    25\% & 384 & 0.9525 & 0.8019 & 1.0005 & 0.7936 & 0.9901 \\
    30\% & 358 & 0.9428 & 0.8028 & 1.0016 & 0.8131 & 1.0144 \\
    40\% & 307 & 0.9239 & 0.8027 & 1.0014 & 0.8067 & 1.0064 \\
    50\% & 256 & 0.9050 & 0.8042 & 1.0033 & 0.7714 & 0.9625 \\
    60\% & 205 & 0.8860 & 0.8050 & 1.0043 & 0.7867 & 0.9815 \\
    70\% & 154 & 0.8671 & 0.8078 & 1.0079 & 0.8062 & 1.0059 \\
    80\% & 102 & 0.8478 & 0.8099 & 1.0104 & 0.7742 & 0.9659 \\
    \textbf{90\%} & \textbf{51} & \textbf{0.8289} & \textbf{0.8037} & \textbf{1.0027} & \textbf{0.8214} & \textbf{1.0248} \\
    \hline
  \end{tabular}
\end{table*}

\begin{figure}[!t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig4_3_pruning_curve.png}
  \caption{AUROC retention vs.\ sparsity for post-hoc (blue) and fine-tuned
    (red) evaluations; post-hoc retention never falls below 0.9996.}
  \label{fig:pruning}
\end{figure}

\begin{figure}[!t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig4_4_compression_scatter.png}
  \caption{Compression ratio vs.\ post-hoc AUROC; points above the reference
    line (baseline AUROC 0.8015) represent post-hoc improvements over the
    unpruned model.}
  \label{fig:scatter}
\end{figure}
```

- [ ] **Step 2: Compile**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "docs: add results IV-C (neuron pruning) with table IV and figs 3-4"
```

---

## Chunk 4: Discussion, Conclusion, References

### Task 8: Write Section V — Discussion

**Files:**
- Modify: `paper/paper_tnnls.tex`

Source: `paper/chapter_phase8.md` §4.7. Key changes:
- Convert the "Practical recommendations" Markdown bullet list to flowing prose (2–3 sentences)
- Trim Limitations to exactly 3 sentences: (1) single dataset, (2) CPU-only scale, (3) single pruning seed
- Keep "Unified interpretation" and "Future directions" paragraphs verbatim

- [ ] **Step 1: Append the Discussion section**

```latex
% ── V. DISCUSSION ────────────────────────────────────────────────────────────
\section{Discussion}
\label{sec:discussion}

% SOURCE: paper/chapter_phase8.md §4.7
%
% KEEP: "Unified interpretation." paragraph — verbatim.
% KEEP: "The neuron pruning results (H3)..." paragraph — verbatim
%        (change "H3" → "C3").
%
% REPLACE the bullet list under "Practical recommendations." with prose:
%   "For practitioners choosing between architectures: if AUROC is the
%    primary criterion, the cross-encoder is the clear choice. If scalability
%    to genome-scale pair evaluation ($10^6$--$10^8$ pairs) is paramount,
%    the two-tower may be preferred, with practitioners noting a 6--10~AUROC
%    point cost relative to a cross-encoder of comparable size; the 90\%-sparse
%    two-tower achieves baseline performance at reduced representation size."
%
% TRIM Limitations to 3 sentences:
%   Sentence 1: single biological dataset (human brain snRNA-seq)
%   Sentence 2: CPU-only training constrains model scale
%   Sentence 3: pruning conducted at a single seed; 10-epoch fine-tune budget
%               insufficient to fully characterize sparsity-performance relationship
%
% KEEP: "Future directions." paragraph — verbatim (change "H2"/"H3" → "C2"/"C3").
%
% [PASTE adapted §4.7 here]
```

- [ ] **Step 2: Compile**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "docs: add discussion section (V)"
```

---

### Task 9: Write Section VI — Conclusion and References

**Files:**
- Modify: `paper/paper_tnnls.tex`

The Conclusion is written from scratch (do NOT copy from dissertation — it is too long). The References use the existing `.bib` file.

- [ ] **Step 1: Append Conclusion and References**

```latex
% ── VI. CONCLUSION ───────────────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}

This paper has compared two-tower and cross-encoder neural architectures for
TF--gene regulatory edge prediction from single-nucleus RNA-seq data under
parameter-matched, data-matched conditions. The cross-encoder achieves 9.4
higher AUROC points than the two-tower under balanced training (0.9040 vs.\
0.8097) and remains stable under 5:1 negative sampling where the two-tower
degrades by 6.6~points. A structured neuron pruning analysis reveals that
post-hoc AUROC retention never falls below 99.96\% of baseline at any sparsity
level up to 90\%, with 51 neurons per tower (10\% of 512) sufficient to match
baseline performance without additional training. These findings show that the
two-tower's AUROC deficit reflects wasted representational capacity caused by
the geometric constraints of cosine similarity scoring, not a parameter
shortfall. For GRN inference tasks prioritizing discriminative accuracy, the
cross-encoder is recommended; for genome-scale applications requiring efficient
inference, the compressed two-tower at 90\% neuron sparsity offers equivalent
performance at substantially reduced representation size.

% ── REFERENCES ───────────────────────────────────────────────────────────────
\bibliographystyle{IEEEtran}
\bibliography{chapter_phase8}

\end{document}
```

- [ ] **Step 2: Compile**

```bash
tectonic paper/paper_tnnls.tex 2>&1 | grep "^error" | head -5
```

- [ ] **Step 3: Commit**

```bash
git add paper/paper_tnnls.tex
git commit -m "docs: add conclusion and references (VI)"
```

---

## Chunk 5: Final Compilation and Verification

### Task 10: Full compile, verify, and push

**Files:**
- No new content — verification and final commit only

- [ ] **Step 1: Full compile**

```bash
tectonic paper/paper_tnnls.tex 2>&1
```

Expected:
```
note: Writing `paper/paper_tnnls.pdf` (XXX KiB)
```

No `error:` lines. Warnings about `Overfull \hbox` are acceptable.

- [ ] **Step 2: Verify PDF exists and has reasonable size**

```bash
ls -lh paper/paper_tnnls.pdf
```

Expected: file exists, size > 100KB (two-column paper with 4 figures).

- [ ] **Step 3: Verify no leftover placeholder comments**

```bash
grep -n "\[PASTE\]\|TODO\|\[Department\]\|\[University\]" paper/paper_tnnls.tex | head -20
```

Expected: only the affiliation placeholders (`[Department]`, `[University]`, `[City, Country]`, `[email]`) remain — these are intentional and documented in the spec for the author to fill in before submission.

- [ ] **Step 4: Final commit**

```bash
git add paper/paper_tnnls.tex paper/paper_tnnls.pdf
git commit -m "docs: complete TNNLS journal paper and render PDF"
git push origin main
```

---

## Summary

| Chunk | Tasks | Deliverable |
|-------|-------|-------------|
| 1: Scaffold | 1–2 | Preamble, abstract, index terms, Section I |
| 2: Background | 3–4 | Sections II–III |
| 3: Results | 5–7 | Section IV (Tables I–IV, Figs. 1–4) |
| 4: Discussion | 8–9 | Sections V–VI + References |
| 5: Verification | 10 | Compiled `paper/paper_tnnls.pdf` |
