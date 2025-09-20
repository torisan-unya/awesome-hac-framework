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

## Recommended Reading Order
1. [01_theoretical-framework.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/01_theoretical-framework.md)
2. [02_extended-framework-validation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/02_extended-framework-validation.md)
3. [03_xc-ii_validation_simulation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/03_xc-ii_validation_simulation.md)
4. [04_x-cii_formalization_and_synthetic_evaluation.md](https://github.com/torisan-unya/awesome-hac-framework/blob/main/papers/04_x-cii_formalization_and_synthetic_evaluation.md)

## Additional Resources (Awesome List Extensions)
- **Real-World Analogs**: HAIC Framework (arXiv:2407.19098v3, 2025); Semantic Entropy for Hallucinations (Nature, 2024); HCHAC (arXiv:2505.22477); Group-Adaptive Thresholds (arXiv:2502.04528).
- **Tools & Repos**: [xAI Grok API](https://x.ai/api); [IRIS for Ideation](https://arxiv.org/abs/2504.16728) (placeholder).
- **Communities**: Follow [@torisan_unya on X](https://x.com/torisan_unya); Contribute to this GitHub Repo.

## Future Extensions
- Axiomatic λ variations (λ=0.1 for imbalance penalties).
- Empirical pilots (N=200, 12-month).
- PRs welcome for fairness metrics.

## License
CC BY-SA 4.0
