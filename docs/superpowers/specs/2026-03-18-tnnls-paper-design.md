# TNNLS Journal Paper Design Spec

## Goal

Reformat and restructure `paper/chapter_phase8.md` into a publication-ready IEEE Transactions on Neural Networks and Learning Systems (TNNLS) journal paper, producing `paper/paper_tnnls.tex` and `paper/paper_tnnls.pdf`.

## Source Material

- `paper/chapter_phase8.md` — 5,452-word dissertation chapter (source content)
- `paper/chapter_phase8.bib` — 13 BibTeX entries (reused)
- `paper/figures/fig4_1_architecture.png` through `fig4_4_compression_scatter.png` — 4 figures (reused)
- All numeric values are pre-computed and verified correct

## Target Venue

**IEEE Transactions on Neural Networks and Learning Systems (TNNLS)**

- Document class: `\documentclass[journal]{IEEEtran}`
- Layout: two-column, 10pt font
- Citation style: IEEE numbered `[1]`, `[2]`, ...
- Reference style: `\usepackage{cite}` + BibTeX
- Required packages: `cite`, `graphicx`, `amsmath`, `amssymb`, `inputenc` (utf8)

## Output Files

| File | Description |
|------|-------------|
| `paper/paper_tnnls.tex` | Full IEEE-formatted LaTeX source |
| `paper/paper_tnnls.pdf` | Compiled PDF via tectonic |

The existing `paper/chapter_phase8.bib` is reused unchanged. No new `.bib` file is needed.

---

## Paper Structure

### Front Matter

**Title:** Modular versus Monolithic Neural Architectures for Gene Regulatory Network Inference: Capacity, Robustness, and Representational Efficiency

**Author block** (`\documentclass[journal]{IEEEtran}` standard, not compsoc):
```latex
\author{Evint Leovonzko%
\thanks{E. Leovonzko is with [Department], [University], [City, Country].
E-mail: [email]. Manuscript received 2026.}}
```
Replace bracketed placeholders with actual affiliation details before submission.

**Abstract (~200 words):** Covers problem (GRN inference as link prediction), approach (parameter-matched two-tower vs cross-encoder, 1:1 and 5:1 negative sampling, structured neuron pruning), key results (cross-encoder +9.4 AUROC points at 1:1; two-tower degrades 6.6 AUROC pts at 5:1 vs cross-encoder stable; post-hoc AUROC retention never falls below 99.96% of baseline at any neuron sparsity level up to 90%), and implication (cross-encoder preferred for discriminative tasks; two-tower at 90% neuron sparsity per tower matches baseline with no additional training).

**Index Terms:** gene regulatory network inference, two-tower architecture, cross-encoder, dual-encoder, neural network pruning, single-cell RNA-seq, representation learning, link prediction, class imbalance

---

### Section I — Introduction (~450 words)

Content flow:
1. GRN inference problem and its importance (2–3 sentences)
2. Link prediction framing and the architectural question: factorized vs joint encoding (2–3 sentences)
3. Gap statement: no parameter-matched comparison exists; class imbalance robustness not studied; effective capacity of two-tower unknown
4. Three explicit contributions:
   - **(C1)** Parameter-matched comparison of two-tower and cross-encoder under balanced and 5:1 imbalanced training
   - **(C2)** Demonstration that two-tower degrades 6.6 AUROC points under 5:1 negative sampling while cross-encoder is stable
   - **(C3)** Structured neuron pruning analysis showing post-hoc AUROC retention ≥ 0.9996 at all sparsity levels up to 90%
5. "The remainder of this paper is organized as follows..." sentence

Key rewrites from dissertation:
- Remove "Three research hypotheses are investigated" framing — replace with contribution bullets
- Remove "The results confirm all three hypotheses" — move findings to Abstract
- Replace "This chapter" → "This paper" throughout

---

### Section II — Related Work (~550 words)

Four subsections:

**II-A. Gene Regulatory Network Inference as Link Prediction**
Content from §4.2.1, condensed ~25%. Remove "The present work adopts..." bridging paragraph (moved to Methods).

**II-B. Two-Tower (Dual-Encoder) Architectures**
Content from §4.2.2, condensed ~20%. Remove "The cross-encoder architecture, described next..." forward reference.

**II-C. Cross-Encoder Architectures**
Content from §4.2.3, condensed ~20%. Remove "In the present work, the cross-encoder receives..." (moved to Methods).

**II-D. Structured Pruning and Representational Redundancy**
Content from §4.2.4, condensed ~15%. Remove "In the context of the two-tower GRN model..." bridge paragraph (moved to intro or Methods motivation).

---

### Section III — Methods (~750 words)

Six subsections matching dissertation §4.3, with minor edits:

**III-A. Dataset and Prior Knowledge** — unchanged
**III-B. Model Architectures** — unchanged; Fig. 1 reference becomes `\figurename~\ref{fig:arch}`
**III-C. Training Protocol** — unchanged; lr discrepancy note retained
**III-D. Evaluation** — unchanged
**III-E. Negative Sampling** — unchanged
**III-F. Neuron Pruning Protocol** — unchanged; math equation retained as-is

---

### Section IV — Results (~1,100 words)

Three subsections replacing H1/H2/H3 structure:

**IV-A. Cross-Encoder vs. Two-Tower Under Balanced Training**
- Source: §4.4 (H1)
- Retitle: "Cross-Encoder vs. Two-Tower Under Balanced Training"
- Remove "**Hypothesis.**" label; begin directly with "Table~\ref{tab:h1} presents..."
- Keep Tables 4.1 and 4.2 (renumbered to Table I and Table II)
- Keep Fig. 4.2 reference (renumbered to Fig. 2)
- Keep full Interpretation paragraph

**IV-B. Robustness to Class Imbalance**
- Source: §4.5 (H2)
- Retitle: "Robustness to Class Imbalance"
- Remove "**Hypothesis.**" label
- Keep Table 4.3 (renumbered to Table III)

**IV-C. Representational Redundancy via Neuron Pruning**
- Source: §4.6 (H3)
- Retitle: "Representational Redundancy via Neuron Pruning"
- Remove "**Hypothesis.**" label
- Keep Table 4.4 (renumbered to Table IV) and Figs. 4.3–4.4 (renumbered to Figs. 3–4)
- Keep full Interpretation paragraph

---

### Section V — Discussion (~450 words)

Source: §4.7, condensed ~20%.

Cuts:
- Shorten "Practical recommendations" bullet list to prose (TNNLS style avoids bullet lists in discussion)
- Trim limitations to 3 sentences (keep: single dataset, CPU-only scale constraint, single pruning seed)
- Keep Future directions paragraph unchanged

---

### Section VI — Conclusion (~180 words)

Rewritten from scratch. TNNLS conclusion formula:
1. One sentence restating the problem
2. Two sentences on key quantitative findings (AUROC gap, imbalance robustness, pruning result)
3. One sentence on the mechanistic link (H3 explains H1)
4. One sentence on practical guidance

Do NOT reuse the dissertation conclusion verbatim — it is too long and recaps too much.

---

### References

12–13 references, IEEE numbered format [1]–[N]. Citation order follows first appearance in text. BibTeX keys unchanged from `chapter_phase8.bib`.

The `chen2016` entry is cited in §4.2.3 for the element-wise product feature engineering technique. **Keep this citation** in Section II-C and retain `chen2016` in the bibliography (13 references total). BibTeX will automatically exclude any entries not cited in the text.

Bibliography declaration: `\bibliography{chapter_phase8}`

---

## Figure Formatting

All four figures reused from `paper/figures/`. In IEEEtran:

```latex
\begin{figure}[!t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig4_X_name.png}
  \caption{...}
  \label{fig:X}
\end{figure}
```

Figure captions condensed to one sentence each (IEEE style).

---

## Table Formatting

All four tables converted to `IEEEtran` style:

```latex
\begin{table}[!t]
  \caption{Title Here}
  \label{tab:X}
  \centering
  \begin{tabular}{lccc}
    \hline
    ...
    \hline
  \end{tabular}
\end{table}
```

Caption above the table. `\hline` top and bottom; `\hline` after header row.

---

## Build Command

```bash
tectonic paper/paper_tnnls.tex
```

Expected output: `paper/paper_tnnls.pdf`

---

## Scope Boundaries

**In scope:**
- New `paper/paper_tnnls.tex` with full IEEE formatting and restructured content
- Compiled `paper/paper_tnnls.pdf`

**Out of scope:**
- Modifying `paper/chapter_phase8.tex` or `paper/chapter_phase8.md` (preserved as-is)
- Any new experiments or figures
- Submitting to TNNLS
