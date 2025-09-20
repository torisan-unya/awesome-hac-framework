# Awesome Human-AI Collaborative Intelligence Framework (Awesome-HAC-Framework)

**Author:** Torisan Unya [@torisan_unya]

**Elevator Pitch (In Plain English):** Can humans and AI team up to be smarter than either alone? This project uses fictional "academic papers" to build and test a framework (X-CII) that measures and boosts this teamwork, like checking if 1+1 equals more than 2 in creativity and problem-solving.

**Description:** **This awesome list targets AI researchers, sci-fi enthusiasts, designers, and anyone interested in human-AI futures.** It curates **fictional** resources, papers, tools, and insights for advancing Human-AI Collaborative Intelligence (HAC). Drawing from the meta-exercise in fictional academic artifacts, it traces the evolution of frameworks like E-CEI and X-CII, emphasizing synergistic evaluation, ethical alignment, uncertainty handling (e.g., hallucination detection with semantic entropy baselines, AUROC ~0.75-0.85), axiomatic rigor, and fairness diagnostics. **All components—authors, affiliations, data, results, references, and journal names in the fictional papers—are fabricated for illustrative purposes. No real-world empirical claims are made.** For the original archived repository and full context, see [AI-Novel-Prompt-Hybrid](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid).

**Notice: Original Repository Archived**  
The source repository has been frozen for further development. Historical files remain available there for reference. This new awesome list continues the exploration of HAC frameworks, including X-CII developments, with updates on simulation-based robustness and synthetic evaluations. **Note: Considering AI models like Gemini may have knowledge cutoffs (e.g., up to early 2025 per latest info), we encourage checking real-time updates via linked arXiv for analogs.**

**Updated as of September 21, 2025 (Version 2.6)**: Refined structure for readability; consolidated metrics (Core X-CII: Human-only ~0.78, AI-only ~0.76, Collab ~0.84); enhanced fairness diagnostics (EOD L_inf median 0.02; calibration gap proxy median 0.40); integrated group-adaptive thresholds (AUROC~0.72: median Relative X-CII 105.2%, win rate 95%; Core ≥0.75 in 100% of runs). Monte Carlo sensitivity (10,000 replicates): median Relative X-CII 108.7% [95% CI: 107.2-110.1%]; 5-95th percentile Relative X-CII 104.3-112.8%. Added axiomatic λ variations and fairness optimizations. All links are absolute for stability. **New: Added cross-references to real-world analogs for each stage; clarified fictional vs. real distinctions.**

<details>
<summary>Key Terms Glossary</summary>
<ul>
<li><b>X-CII</b>: Extended Collaborative Intelligence Index. Measures quality (Q), efficiency (E), and safety (S) in human-AI collaboration via Box-Cox average (λ=0.25).</li>
<li><b>Relative X-CII</b>: Percentage improvement of collaborative score over the best single-agent baseline (e.g., 108.7% means +8.7% uplift).</li>
<li><b>EOD L_inf</b>: Equalized Odds Difference (L_infinity norm). Fairness metric; median 0.02 indicates low bias across groups.</li>
<li><b>Box-Cox</b>: Transformation for aggregating Q/E/S; ensures monotonicity and invariance.</li>
<li><b>AUROC</b>: Area Under ROC Curve. Measures hallucination detection; ~0.75-0.85 baseline, drops to 0.72 under shifts.</li>
</ul>
</details>

---

## Framework Evolution Overview

This awesome list traces the HAC framework's progression in four stages (aligned with E-CEI's model), bridging conceptual gaps through complementarity, safety thresholds, and domain adaptation:

```mermaid
flowchart LR
    A[Stage 1: Theoretical\nE-CEI Foundations] --> B[Stage 2: Extension\nX-CII with Dynamics]
    B --> C[Stage 3: Simulation\nMonte Carlo Robustness]
    C --> D[Stage 4: Formalization\nAxiomatic X-CII]
    style A fill:#f9f,stroke:#333
    style D fill:#bbf,stroke:#333
