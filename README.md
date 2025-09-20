# Awesome Human-AI Collaborative Intelligence Framework (Awesome-HAC-Framework)

**Author:** Torisan Unya [@torisan_unya]

**Description:** This awesome list curates resources, papers, tools, and insights for advancing Human-AI Collaborative Intelligence (HAC). Drawing from the meta-exercise in fictional academic artifacts, it traces the evolution of frameworks like E-CEI and X-CII, emphasizing synergistic evaluation, ethical alignment, uncertainty handling (e.g., hallucination detection with semantic entropy baselines, AUROC ~0.75-0.85), axiomatic rigor, and fairness diagnostics. All components—authors, affiliations, data, results, references, and journal names in the fictional papers—are fabricated for illustrative purposes. No real-world empirical claims are made. For the original archived repository and full context, see [AI-Novel-Prompt-Hybrid](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid).

**Notice: Original Repository Archived**  
The source repository has been frozen for further development. Historical files remain available there for reference. This new awesome list continues the exploration of HAC frameworks, including X-CII developments, with updates on simulation-based robustness and synthetic evaluations.

**Updated as of September 21, 2025 (Version 2.6)**: Refined structure for readability; consolidated metrics (Core X-CII: Human-only ~0.78, AI-only ~0.76, Collab ~0.84); enhanced fairness diagnostics (EOD L_inf median 0.02; calibration gap proxy median 0.40); integrated group-adaptive thresholds (AUROC~0.72: median Relative X-CII 105.2%, win rate 95%; Core ≥0.75 in 100% of runs). Monte Carlo sensitivity (10,000 replicates): median Relative X-CII 108.7% [95% CI: 107.2-110.1%]; 5-95th percentile Relative X-CII 104.3-112.8%. Added axiomatic λ variations and fairness optimizations. All links are absolute for stability.

---

## Framework Evolution Overview

This awesome list traces the HAC framework's progression in four stages (aligned with E-CEI's model), bridging conceptual gaps through complementarity, safety thresholds, and domain adaptation:

- **Stage 1 (Theoretical)**: Introduces E-CEI for synergistic evaluation with trust-weighted metrics (T coefficient), reliability factor (R), and ethical principles. (Cross-ref: Basis for all subsequent stages.)
- **Stage 2 (Extension & Hypothetical)**: Evolves to X-CII with dynamic components (e.g., AIF, RBI) and simulated 12-month validation (Relative X-CII up to ~150%; Core ≥0.75 in 92% of runs). (Cross-ref: Builds on Stage 1; informs Stages 3-4.)
- **Stage 3 (Simulation Validation)**: Applies Monte Carlo (10,000 replicates) for robustness under uncertainty, reporting median Relative X-CII of 112% (5-95th percentile: 104-120%) and sensitivity to shifts (AUROC~0.72-0.85). Includes group-adaptive thresholds and win rates. (Cross-ref: Extends Stage 2; provides data for Stage 4.)
- **Stage 4 (Formalization & Synthetic)**: Defines X-CII axiomatically (Box-Cox average of Q, E, S; λ=0.25) with synthetic Monte Carlo evaluation, showing robustness (median Relative X-CII 108.7% [95% CI: 107.2-110.1%]; Core ≥0.75 in all runs). Integrates fairness diagnostics (EOD L_inf median 0.02; calibration gap proxy median 0.40) and human-anchored S variants. (Cross-ref: Integrates Stages 1-3 with axiomatic rigor.)

**X-CII Core values across stages: Human-only mean ~0.78; AI-only ~0.76; Collab ~0.84 (synthetic estimates).** Sensitivity to domain shifts: Under AUROC=0.72, Relative X-CII drops to 105.2% with 95% win rate vs. baselines. Fairness diagnostics ensure balanced representation (e.g., EOD L_inf <0.05 in 95% of runs).

---

## Core Framework Papers (Fictional Artifacts)

| # | Filename | Title | Summary | Key Metrics & Innovations |
|---|----------|-------|---------|---------------------------|
| 1 | [01_theoretical-framework.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/01_theoretical-framework.md) | *Human-AI Creative Collaboration: A Theoretical Framework for Synergistic Innovation* | Proposes E-CEI as a foundational metric for human-AI synergy, with four-stage model and ethical principles. Trust coefficient (T) and reliability factor (R) introduced. | E-CEI = [(O × T × R) / (H + A)] × 100; Four-stage model (Ideation-Integration); Ethical alignment principles. |
| 2 | [02_extended-framework-validation.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/02_extended-framework-validation.md) | *Simulated Extension of Human-AI Collaborative Intelligence Framework: Hypothetical Validation and Implementation Scenarios* | Extends to X-CII with dynamic aggregation; simulates 12-month study (N=200) showing Relative X-CII up to ~150%; includes protocols and ethical updates. Incorporates AIF and RBI for role adaptation. | Core X-CII = (Q' × E' × S')^{1/3}; Relative X-CII up to 150%; Dynamic components (AIF, RBI, TCO). |
| 3 | [03_xc-ii_validation_simulation.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/03_xc-ii_validation_simulation.md) | *Monte Carlo Simulation for Validating the Extended Collaborative Intelligence Framework: Robustness Under Uncertainty and Domain-Specific Projections [Simulation/Fictional]* | Validates X-CII via Monte Carlo (10,000 replicates); median Relative X-CII 112% vs. baselines; addresses shifts (AUROC~0.72-0.85). Builds on Paper 2's extensions. Includes group-adaptive thresholds and win rates. | Median Relative X-CII 112% (5-95th: 104-120%); Core ≥0.75 in 92%; AUROC sensitivity; Group-adaptive τ*. |
| 4 | [04_x-cii_formalization_and_synthetic_evaluation.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/04_x-cii_formalization_and_synthetic_evaluation.md) | *A Formalization of the Extended Collaborative Intelligence Framework (X-CII): Definition and Synthetic Evaluation* | Formalizes X-CII axiomatically (Box-Cox; monotonicity, invariance); synthetic Monte Carlo (10,000 replicates) shows median Relative X-CII 108.7% [95% CI: 107.2-110.1%]. Integrates simulations from Paper 3. Adds fairness and calibration diagnostics. | Box-Cox avg (λ=0.25); Median Relative 108.7%; EOD L_inf 0.02; Calibration gap 0.40; Raw S >1 proportion. |

*Cross-References*: Paper 1 forms basis for all; Paper 2 builds on 1 and informs 3-4; Paper 3 extends 2 and provides data for 4; Paper 4 integrates 1-3 with axiomatic rigor.

---

## Recommended Reading Order

To grasp the framework's evolution ( theory → extension → validation → formalization ), read in this order:

1. **[01_theoretical-framework.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/01_theoretical-framework.md)**: Establishes E-CEI foundations.
2. **[02_extended-framework-validation.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/02_extended-framework-validation.md)**: Extends to X-CII with hypothetical scenarios.
3. **[03_xc-ii_validation_simulation.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/03_xc-ii_validation_simulation.md)**: Tests robustness via simulations.
4. **[04_x-cii_formalization_and_synthetic_evaluation.md](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid/blob/main/academic-paper/04_x-cii_formalization_and_synthetic_evaluation.md)**: Formalizes X-CII with axioms and synthetic evaluation.

---

## Additional Resources (Awesome List Extensions)

- **Real-World Analogs and Related Papers**:
  - HAIC Framework (arXiv:2407.19098 v3 update, 2025): Methodological review for human-AI evaluation.
  - Semantic Entropy for Hallucinations (Nature, 2024; DOI: 10.1038/s41586-024-07421-0): AUROC ~0.75-0.85 baselines.
  - Human-Centered Human-AI Collaboration (HCHAC) (arXiv:2505.22477, 2025): Focus on human-centered relationships.
  - Group-Adaptive Threshold Optimization (arXiv:2502.04528, 2025): For robust detection under shifts.
  - Uncertainty-Aware Task Delegation (arXiv:2505.18066, 2025): Epistemic uncertainty in delegation.
  - Monte Carlo for Human-AI Synergy (arXiv:2507.01062, 2025): Simulations for student success with GenAI.
  - Additional Surveys: Human-AI Collaboration with Large Foundation Models (arXiv:2403.04931 v3, 2025).

- **Tools & Repositories**:
  - [xAI Grok API](https://x.ai/api): For accessing Grok models in collaborative setups.
  - [IRIS: Interactive Research Ideation System](https://arxiv.org/abs/2504.16728) (2025 placeholder): MCTS for ideation.
  - Simulation Code: Python snippets in Papers 3 & 4 (NumPy/SciPy-based Monte Carlo; MIT License).
  - Related Repos: [AI-Novel-Prompt-Hybrid](https://github.com/torisan-unya/AI-Novel-Prompt-Hybrid) (original archived source).

- **Communities & Discussions**:
  - Follow [@torisan_unya on X](https://x.com/torisan_unya) for updates on prompting frameworks.
  - Contribute via GitHub: Issues/PRs welcome for Monte Carlo refinements, new fairness metrics, or domain extensions (e.g., education, finance). Potential integration with tools like IRIS or feedback-aware MCTS.

---

## Future Extensions

- **Axiomatic Enhancements**: Explore λ variations (e.g., λ=0.1 for stronger imbalance penalties; λ=0.5 for milder) and weighted Box-Cox for domain-specific adaptations (e.g., healthcare: higher S weight). Integrate advanced uncertainty quantification (e.g., epistemic uncertainty via Penalty Ensemble Method).
- **Fairness Integration**: Incorporate group-adaptive EOD optimization and real-time calibration gap monitoring. Add diagnostics like TPR-FPR differences across stratified groups.
- **Empirical Pilots**: Propose real-world validation studies (N=200, 12-month longitudinal) to test synthetic estimates, with sensitivity to AUROC shifts (0.72-0.85). Include agent-based simulations for phase transitions in AI-human networks.
- **Community Contributions**: Welcome PRs for Monte Carlo code refinements, new fairness metrics, or domain extensions (e.g., education, finance). Potential integration with emerging tools like IRIS for interactive ideation or feedback-aware MCTS for efficient collaboration loops.

---

## Keywords

**Core Concepts:** Human-AI Collaboration, Collaborative Intelligence, Synergistic Innovation, E-CEI, X-CII.  
**Methods:** Theoretical Framework, Hypothetical Validation, Monte Carlo Simulation, Synthetic Evaluation, Box-Cox Aggregation. (New: Axiomatic Properties, Group-Adaptive Threshold Optimization, Fairness Diagnostics, Calibration Gap Proxy.)  
**Applications:** Creative AI, AI Ethics, Multi-Agent Systems.  
**Meta-Aspects:** Fictional Research, Meta-Project, AI Prompting.

---

## License

This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
