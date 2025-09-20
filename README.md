# Awesome Human-AI Collaborative Intelligence Framework (Awesome-HAC-Framework)

**Author:** Torisan Unya [@torisan_unya]

**Description:** This is an awesome list curating resources, papers, and tools for Human-AI Collaborative Intelligence, based on the fictional academic artifacts in the meta-exercise. All links are absolute to this repository for stability. Papers are located in the 'papers' folder.

**Updated as of September 20, 2025 (Version 2.6)**: Integrated absolute paths within this repo; enhanced with new resources from recent arXiv preprints (placeholders replaced with 2025 analogs); consolidated X-CII metrics (Collab ~0.84); fairness diagnostics updated (EOD L_inf 0.02); Monte Carlo (10,000 reps): Relative X-CII 108.7% [95% CI: 107.2-110.1%].

---

## Core Framework Papers (Fictional Artifacts)

| # | Filename | Title | Summary | Key Metrics & Innovations |
|---|----------|-------|---------|---------------------------|
| 1 | [01_theoretical-framework.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/01_theoretical-framework.md) | *Human-AI Creative Collaboration: A Theoretical Framework for Synergistic Innovation* | Proposes E-CEI as a foundational metric for human-AI synergy, with four-stage model and ethical principles. Trust coefficient (T) and reliability factor (R) introduced. | E-CEI = [(O × T × R) / (H + A)] × 100; Four-stage model (Ideation-Integration); Ethical alignment principles. |
| 2 | [02_extended-framework-validation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/02_extended-framework-validation.md) | *Simulated Extension of Human-AI Collaborative Intelligence Framework: Hypothetical Validation and Implementation Scenarios* | Extends to X-CII with dynamic aggregation; simulates 12-month study (N=200) showing Relative X-CII up to ~150%; includes protocols and ethical updates. Incorporates AIF and RBI for role adaptation. | Core X-CII = (Q' × E' × S')^{1/3}; Relative X-CII up to 150%; Dynamic components (AIF, RBI, TCO). |
| 3 | [03_xc-ii_validation_simulation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/03_xc-ii_validation_simulation.md) | *Monte Carlo Simulation for Validating the Extended Collaborative Intelligence Framework: Robustness Under Uncertainty and Domain-Specific Projections [Simulation/Fictional]* | Validates X-CII via Monte Carlo (10,000 replicates); median Relative X-CII 112% vs. baselines; addresses shifts (AUROC~0.72-0.85). Builds on Paper 2's extensions. Includes group-adaptive thresholds and win rates. | Median Relative X-CII 112% (5-95th: 104-120%); Core ≥0.75 in 92%; AUROC sensitivity; Group-adaptive τ*. |
| 4 | [04_x-cii_formalization_and_synthetic_evaluation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/04_x-cii_formalization_and_synthetic_evaluation.md) | *A Formalization of the Extended Collaborative Intelligence Framework (X-CII): Definition and Synthetic Evaluation* | Formalizes X-CII axiomatically (Box-Cox; monotonicity, invariance); synthetic Monte Carlo (10,000 replicates) shows median Relative X-CII 108.7% [95% CI: 107.2-110.1%]. Integrates simulations from Paper 3. Adds fairness and calibration diagnostics. | Box-Cox avg (λ=0.25); Median Relative 108.7%; EOD L_inf 0.02; Calibration gap 0.40; Raw S >1 proportion. |

*Cross-References*: Paper 1 forms basis for all; Paper 2 builds on 1 and informs 3-4; Paper 3 extends 2 and provides data for 4; Paper 4 integrates 1-3 with axiomatic rigor.

---

## Recommended Reading Order

To grasp the framework's evolution ( theory → extension → validation → formalization ), read in this order:

1. **[01_theoretical-framework.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/01_theoretical-framework.md)**: Establishes E-CEI foundations.
2. **[02_extended-framework-validation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/02_extended-framework-validation.md)**: Extends to X-CII with hypothetical scenarios.
3. **[03_xc-ii_validation_simulation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/03_xc-ii_validation_simulation.md)**: Tests robustness via simulations.
4. **[04_x-cii_formalization_and_synthetic_evaluation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/04_x-cii_formalization_and_synthetic_evaluation.md)**: Formalizes X-CII with axioms and synthetic evaluation.

For deeper exploration, visit this repository.

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

### License
This work is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).

### Additional Resources
- Follow [@torisan_unya on X](https://x.com/torisan_unya) for updates on prompting frameworks and discussions.
- Contribute via GitHub: Issues/PRs welcome for refinements (e.g., axiomatic extensions, simulation code, fairness metrics).
- Related Real-World Resources: For non-fictional analogs, see HAIC Framework (arXiv:2407.19098 v3 update, 2025) and Semantic Entropy for Hallucinations (Nature, 2024). (New: Human-Centered Human-AI Collaboration (HCHAC) arXiv:2505.22477; Group-Adaptive Threshold Optimization arXiv:2502.04528; Uncertainty-Aware Task Delegation arXiv:2505.18066; Monte Carlo for Human-AI Synergy arXiv:2507.01062 [placeholders for 2025 preprints].)

---
