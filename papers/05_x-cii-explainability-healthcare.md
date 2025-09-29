# Enhancing Explainability in Healthcare AI through the Extended Collaborative Intelligence Index (X-CII): A Synthetic Evaluation Framework

**Author:** Torisan Unya (ORCID: https://orcid.org/0009-0004-7067-9765)  
**Affiliation:** Independent Researcher  
**Keywords:** Human-AI Collaboration, Collaborative Intelligence Metrics, Explainable AI, Healthcare Applications, X-CII Framework, Synthetic Evaluation, Regulatory Compliance  
**Categories:** cs.HC; cs.AI; cs.LG; stat.ML  
**Submission Note:** v1: Added dedicated Methods and Results sections for enhanced structure and clarity; recomputed statistics using verified code execution (e.g., baseline median updated to 102.963% to reflect precise floating-point results); incorporated recent XAI advancements from semantic search; confirmed EU AI Act details with official sources. Verified references as of September 29, 2025. No empirical claims; synthetic evaluation only. Reproducible code provided in Appendix A (requires Python 3.10+, NumPy 1.23+, SciPy 1.10+; execution time approximately 0.5 seconds on standard hardware). Licensed under CC BY-SA 4.0 (paper) and MIT (code).

## Plain-Language Summary for Clinicians

In healthcare, AI tools can help with diagnosis, but their "black box" nature creates risks. If an AI suggests a treatment without a clear reason, how can you confidently explain it to a patient or justify it legally?  
X-CII is a "report card" for a human-AI team. It grades not just accuracy (Q) or speed (E), but also how well the team works together to make safe, explainable decisions (S). A good AI explanation acts like better teamwork, boosting the team's diagnostic "detection skill" by a simulated 5%. Our simulations showed this led to a typical performance improvement of around 3% for the human-AI team compared to the best individual (human or AI) working alone. In fact, the collaborative team was the better option in around 90% of cases.  
This approach aligns with new regulations like the EU AI Act¹ and IMDRF guidelines, which require transparency for high-risk medical AI. By measuring the value of an explanation, X-CII helps build trustworthy AI that supports clinicians and meets legal standards. This is a conceptual framework, not for direct clinical use, and should be calibrated to site-specific data in practice.  
¹Obligations under these regulations are being introduced in stages over the next few years. Check the latest version for updates.

## Abstract

Human-AI collaboration in healthcare demands explainable AI (XAI) to foster trust, safety, and regulatory compliance, as mandated by the EU AI Act [1] and IMDRF Good Machine Learning Practice (GMLP) [2]. This paper formalizes the Extended Collaborative Intelligence Index (X-CII) to quantify XAI's role in enhancing collaborative performance. X-CII aggregates quality (Q), efficiency (E), and safety (S) via a power mean (λ=0.25), isolating explainability's impact as a +5% uplift to the team's detectability index (d' in Signal Detection Theory, assuming equal-variance Gaussian). In this synthetic study, this uplift is applied only to collaborative d' and reflects explainability’s effect on Safety; Q/E are fixed in baseline to isolate mechanisms. This conservative estimate is derived from literature reporting 5-10% performance gains from XAI integration (ranges vary by task and study quality)¹; for example, an AUC of 0.800 (d'≈1.189) with +5% uplift yields d'≈1.248 and AUC≈0.813 (+1.3 percentage points). Addressing critiques of post-hoc XAI (e.g., Rudin, 2019 [3]), we emphasize X-CII's support for fidelity in explanations, mitigating risks like inappropriate reliance and bias amplification via explicit factors in S.  
Synthetic Monte Carlo simulations (10,000 replicates) illustrate relative X-CII around 102.963% median (IQR: 101.236–104.560%) vs. the better baseline, contingent on baseline human/AI skill, prevalence, and cost ratios. Revised Safety normalization (1 - L / L_worst) preserves boundedness and comparability; while it ensures consistency, it compresses differences in high-performance regimes. Sensitivity analyses include uplift on team-only (median 102.963%), single-only (102.130%), both (102.959%); λ variations (geometric mean: 103.051%, arithmetic: 102.694%); η (0.6: 95.055%, 0.8: 99.156%, 1.0: 102.963%); ρ (-0.5: 108.659%, 0.5: 99.637%); and domain shifts with mild single-agent F/R adjustments (AUC=0.72: median 102.818%, win rate 78.5%). Under domain shifts (AUC=0.72, applied uniformly), medians ~102.818% (win rate ~78.5%). Baseline win rate (relative X-CII >100%) is approximately 89.7% under independent assumptions (ρ=0). Results are illustrative, setting-dependent, and should not be generalized without empirical validation. We integrate multimodal foundation models (MFMs) and generative AI challenges conceptually, using uncertainty quantification (e.g., semantic entropy, with reported AUCs of ~0.75-0.85 across datasets as representative examples [6]) to address confabulation risks (conceptually referenced, not implemented in code). This framework supports implementation toward compliance in human-AI teams, but does not itself establish conformity.  
¹For example, systematic reviews report 5–10% improvements in diagnostic tasks [16], and experimental studies of human–AI interaction similarly show performance gains in the 5–10% range [17].

## Introduction

AI integration in healthcare enhances diagnostics and treatment but faces opacity challenges, risking trust erosion and non-compliance with regulations like the EU AI Act (Reg. (EU) 2024/1689; published in Official Journal L 206 on July 12, 2024; entered into force on August 1, 2024; key obligations phased: prohibitions February 2, 2025; General Purpose AI (GPAI) August 2, 2025; high-risk August 2, 2027) [1] and IMDRF GMLP [2].¹ XAI mitigates this by providing interpretable insights, enabling clinicians to understand AI decisions and justify them to patients or regulators. This paper extends the Collaborative Intelligence Index (CII) to X-CII, incorporating XAI's impact on team performance via Signal Detection Theory (SDT) metrics [13–15, 18]. We focus on synthetic evaluation to isolate mechanisms, assuming equal-variance Gaussian noise and conservative +5% uplift to collaborative d' from XAI, derived from literature reviews and studies reporting 5–10% gains in task performance [16,17]. This uplift models improved detectability through better fidelity and calibrated reliance, reducing errors like over-reliance on flawed explanations [3,17].  
The framework addresses key XAI challenges: post-hoc vs. intrinsic interpretability [3], multimodal integration, and uncertainty quantification [6,7]. X-CII quantifies these through S (incorporating fidelity F and reliance R), while maintaining fixed Q/E in baselines. We demonstrate via synthetic Monte Carlo that XAI-induced uplifts yield consistent collaborative advantages, though domain shifts compress benefits. This work complements recent XAI reviews [8–11] and SDT applications [13], providing a reproducible metric for high-risk AI compliance. Limitations include synthetic assumptions; future work should adapt and calibrate to real datasets.  
¹Specific obligations are phased in over 36 months; prohibitions after 6 months, GPAI after 12, and high-risk systems after 24-36 months. See CELEX: 32024R1689 for details.

## Methods

### Synthetic Simulation Setup

We employ Monte Carlo simulations (10,000 replicates) to evaluate X-CII robustness under parameterized uncertainty. Parameters are drawn from literature-informed priors: AUC ~ Uniform(0.75,0.85) (converted to d' via inverse normal: d' = √2 · Φ⁻¹(AUC)); π (prevalence) ~ Beta(6,14) for moderate imbalance; c_FN ~ Uniform(2,5), c_FP ~ Uniform(0.5,2) for asymmetric costs [13,18]. Q and E are fixed at 0.75 in baselines to isolate S effects. Based on prior studies [16,17], uplift is assumed at +5%, though this varies by task; ρ=0 baseline (sensitivity ±0.5), η=1.0 baseline (sensitivity 0.6-0.8). These assumptions aim to illustrate model behavior under plausible parameters.  
Team d' is calculated using a Mahalanobis distance generalization; in our implementation this reduces to the correlated Gaussian formulation (see Appendix A for code; ρ=0 baseline, sensitivity ±0.5; η=1.0 baseline, sensitivity 0.6-0.8). XAI uplift (+5%) is applied multiplicatively to d_team (team-only baseline) or variants (single-only, both).  
Expected loss L is minimized under SDT: L = c_FN · π · (1 - True Positive Rate (TPR)) + c_FP · (1 - π) · False Positive Rate (FPR), with optimal threshold τ* = 0.5 d' + log((c_FP(1-π))/(c_FN π)) / d' (equal-variance Gaussian). Safety S = 1 - L / L_worst, scaled by (α + (1-α)F)(1-R) (α=0.5; F=1.0 baseline human/AI, 0.95 collab; R=0.0 baseline human/AI, 0.05 collab; mild adjustments under shifts).  
X-CII = [(Q^λ + E^λ + S^λ)/3]^(1/λ) (λ=0.25; geometric λ→0, arithmetic λ=1 sensitivities). Relative X-CII = 100 · X-CII_collab / max(X-CII_human, X-CII_ai). Domain shifts: AUC=0.72 fixed, F/R adjusted (human/AI: F=0.98, R=0.02; collab: F=0.92, R=0.08). Seed=42 for reproducibility.

### Statistical Analysis

Medians/IQRs via NumPy percentiles; win rates as proportion relative >100%. Sensitivities vary one parameter at a time. No hypothesis testing; illustrative only.

## Results

Synthetic simulations demonstrate consistent collaborative advantages. Baseline relative X-CII: median 102.963% (IQR: 101.236–104.560%), win rate 89.7%. Table 1 summarizes sensitivities.

| Scenario                  | Median Relative X-CII (%) | IQR (%)          | Win Rate (%) |
|---------------------------|---------------------------|------------------|--------------|
| Baseline (team uplift)    | 102.963                   | 101.236–104.560  | 89.7         |
| Uplift single-only        | 102.130                   | 100.682–103.492  | 85.4         |
| Uplift both               | 102.959                   | 101.302–104.510  | 90.5         |
| λ=0 (geometric)           | 103.051                   | 101.324–104.648  | 89.7         |
| λ=1 (arithmetic)          | 102.694                   | 100.967–104.291  | 89.7         |
| η=0.6                     | 95.055                    | 94.093–95.951    | 0.0          |
| η=0.8                     | 99.156                    | 98.439–99.980    | 24.6         |
| ρ=-0.5                    | 108.659                   | 106.933–110.481  | 99.2         |
| ρ=0.5                     | 99.637                    | 98.892–100.466   | 37.9         |
| Shift AUC=0.72            | 102.818                   | 101.091–104.415  | 78.5         |

*Table 1: Sensitivity analyses of relative X-CII. Win Rate = % of simulations where collaborative > max(human, AI). All values from 10,000 replicates; illustrative only.*

Under shifts, benefits compress but remain positive. Results align with literature [16,17], though synthetic.

## Discussion

X-CII formalizes XAI's value in healthcare human-AI teams, showing modest but consistent uplifts via synthetic evaluation. Integration with MFMs [7] and uncertainty metrics [6] enhances conceptual robustness. The high sensitivity of X-CII to η (team efficiency) and ρ (skill correlation) quantitatively highlights the importance of not only technical performance but also team composition and training in AI implementation. This study is a simulation to verify the theoretical validity and sensitivity of the X-CII framework. The presented values (e.g., collaborative superiority in 90% of cases) indicate potential under the assumed parameters and do not directly predict real clinical outcomes. Future research requires calibration and validation using actual data. The introduction of X-CII should be accompanied by careful ethical and organizational considerations, including redefining clinician roles, responsibilities, and addressing risks like deskilling and automation bias. Limitations include synthetic assumptions (e.g., equal-variance, fixed Q/E); empirical validation is needed. Calibration procedure is outlined in Appendix B. Future work: real-data adaptation, interface factors, and regulatory pilots.

## Appendix A: Reproducible Code

```python
import numpy as np
from scipy.stats import norm

def auc_to_dprime(auc):
    return np.sqrt(2) * norm.ppf(np.clip(auc, 1e-6, 1-1e-6))

def team_dprime(d_h, d_ai, rho=0.0, eta=1.0):
    rho = np.clip(rho, -0.999, 0.999)
    num = d_h**2 + d_ai**2 - 2 * rho * d_h * d_ai
    den = np.maximum(1 - rho**2, 1e-12)
    return eta * np.sqrt(np.maximum(num / den, 0.0))

def expected_loss(d_prime, pi, c_fp, c_fn):
    mu0 = 0.0
    mu1 = d_prime
    delta = np.maximum(mu1 - mu0, 1e-6)
    log_k = np.log(np.maximum(c_fp, 1e-12) * (1 - pi)) - np.log(np.maximum(c_fn, 1e-12) * np.maximum(pi, 1e-12))
    tau_star = 0.5 * (mu0 + mu1) + log_k / delta
    tpr = 1 - norm.cdf(tau_star - mu1)
    fpr = 1 - norm.cdf(tau_star - mu0)
    return c_fn * pi * (1 - tpr) + c_fp * (1 - pi) * fpr

def safety(L, pi, c_fp, c_fn, F, R, alpha=0.5):
    L_allow = (1 - pi) * c_fp
    L_block = pi * c_fn
    L_worst = np.maximum(L_allow, L_block)
    base = 1 - L / np.maximum(L_worst, 1e-6)
    return np.clip(base * (alpha + (1 - alpha) * F) * (1 - R), 0, 1)

def core_xcii(q, e, s, lam=0.25):
    q = np.clip(q, 1e-12, 1)
    e = np.clip(e, 1e-12, 1)
    s = np.clip(s, 1e-12, 1)
    if abs(lam) < 1e-12:
        return np.exp((np.log(q) + np.log(e) + np.log(s)) / 3)
    else:
        return ((q**lam + e**lam + s**lam) / 3)**(1 / lam)

def run_scenario(rho=0.0, eta=1.0, lam=0.25, uplift=1.05,
                 uplift_team=True, uplift_single=False,
                 auc_fixed=None,
                 F_h=1.0, R_h=0.0, F_ai=1.0, R_ai=0.0,
                 F_collab=0.95, R_collab=0.05,
                 seed=42, n=10000, include_stats=True):
    rng = np.random.default_rng(seed)
    auc_h = rng.uniform(0.75, 0.85, n) if auc_fixed is None else np.full(n, auc_fixed)
    auc_ai = rng.uniform(0.75, 0.85, n) if auc_fixed is None else np.full(n, auc_fixed)
    d_h = auc_to_dprime(auc_h)
    d_ai = auc_to_dprime(auc_ai)
    pi = rng.beta(6, 14, n)
    c_fn = rng.uniform(2, 5, n)
    c_fp = rng.uniform(0.5, 2, n)
    if uplift_single:
        d_h_eff = d_h * uplift
        d_ai_eff = d_ai * uplift
    else:
        d_h_eff, d_ai_eff = d_h, d_ai
    d_team = team_dprime(d_h_eff, d_ai_eff, rho=rho, eta=eta)
    if uplift_team:
        d_team = d_team * uplift
    L_h = expected_loss(d_h_eff, pi, c_fp, c_fn)
    L_ai = expected_loss(d_ai_eff, pi, c_fp, c_fn)
    L_team = expected_loss(d_team, pi, c_fp, c_fn)
    alpha = 0.5
    q = e = np.full(n, 0.75)
    s_h = safety(L_h, pi, c_fp, c_fn, F_h, R_h, alpha)
    s_ai = safety(L_ai, pi, c_fp, c_fn, F_ai, R_ai, alpha)
    s_c = safety(L_team, pi, c_fp, c_fn, F_collab, R_collab, alpha)
    core_h = core_xcii(q, e, s_h, lam)
    core_ai = core_xcii(q, e, s_ai, lam)
    core_c = core_xcii(q, e, s_c, lam)
    rel = 100 * core_c / np.maximum(core_h, core_ai)
    median = np.median(rel)
    iqr = np.percentile(rel, [25, 75])
    win_rate = (rel > 100).mean() * 100
    if include_stats:
        mean = np.mean(rel)
        std = np.std(rel)
        return median, iqr, win_rate, mean, std
    return median, iqr, win_rate

# Baseline and sensitivities
print("Baseline:", run_scenario())
print("λ=0 (geom):", run_scenario(lam=0))
print("λ=0.5:", run_scenario(lam=0.5))
print("λ=1.0 (arithmetic):", run_scenario(lam=1.0))
print("η=0.6:", run_scenario(eta=0.6))
print("η=0.8:", run_scenario(eta=0.8))
print("ρ=-0.5:", run_scenario(rho=-0.5))
print("ρ=0.5:", run_scenario(rho=0.5))
print("uplift single-only:", run_scenario(uplift_team=False, uplift_single=True))
print("uplift both:", run_scenario(uplift_team=True, uplift_single=True))
# Domain shift (AUC=0.72) + F/R adjustments
print("Shift AUC=0.72:", run_scenario(auc_fixed=0.72,
      F_h=0.98, R_h=0.02, F_ai=0.98, R_ai=0.02, F_collab=0.92, R_collab=0.08))
```

## Appendix B: Calibration Checklist and Alternatives

This appendix outlines proposed methodology for calibrating X-CII in real environments.  
- ρ estimation: Compute Pearson/Spearman on z-transformed paired scores; stratify by case-mix; robustify with bootstrapping and binormal ROC fitting for covariance; pool within-class covariances (estimate ρ0, ρ1 and variances per class 0/1, weight by sample size under equal-variance); check equal-variance via slope s ≈1. Use split cross-validation to avoid leakage in experiments collecting both human confidence scores and AI outputs on the same cases.  
- F/R calibration: Counterfactual surveys, calibration curves. For α: SOP compliance rate (e.g., documentation adherence threshold 0.8, double-check rates >0.9); F: explanation fidelity from blind evaluations (threshold 0.9); R: over-/under-reliance from behavioral metrics (threshold 0.1), normalized.  
- Alternative Safety normalizations (not used; for illustration): Log-compressed: 1 - log(1 + L/L_ref)/log(1 + L_worst/L_ref); Power-transformed: 1 - (L/L_worst)^0.5 (expands high-performance differences).

## References

[1] European Parliament and Council. (2024). Regulation (EU) 2024/1689... Official Journal of the European Union, L 206, 12.7.2024, p. 1–252. CELEX: 32024R1689.  
[2] IMDRF. (2025). Good Machine Learning Practice... IMDRF/AIML WG/N88 FINAL:2025. Published January 29, 2025.  
[3] Rudin, C. (2019). Stop explaining black box... Nature Machine Intelligence, 1(5), 206-215. DOI: 10.1038/s42256-019-0048-x.  
[4] Fragiadakis, G., et al. (2024). Evaluating Human-AI Collaboration... arXiv:2407.19098 [cs.HC].  
[5] Hildt, E. (2025). What Is the Role of Explainability... Frontiers in Digital Health, 7. DOI: 10.3389/fdgth.2025.12025101.  
[6] Farquhar, S., et al. (2024). Detecting hallucinations... Nature, 630(8017), 625-630. DOI: 10.1038/s41586-024-07421-0.  
[7] Chen, R. J., et al. (2024). Towards multimodal foundation models... arXiv:2402.09849 [cs.LG].  
[8] Giorgetti, C., et al. (2025). Healthcare AI, explainability... Frontiers in Medicine, 12:1545409. DOI: 10.3389/fmed.2025.1545409.  
[9] Mohapatra, R. K. (2025). Advancing explainable AI... Computers in Biology and Medicine, 119:108599. DOI: 10.1016/j.compbiomed.2025.108599.  
[10] El-Geneedy, M., et al. (2025). A comprehensive explainable AI... Scientific Reports, 15(1):11263. DOI: 10.1038/s41598-025-11263-9.  
[11] Vani, M. S., et al. (2025). Personalized health monitoring... Scientific Reports, 15(1):15867. DOI: 10.1038/s41598-025-15867-z.  
[12] CADTH. (2025). 2025 Watch List... NCBI Bookshelf ID: NBK613808.  
[13] Kovesdi, C., et al. (2025). Application of Signal Detection Theory... Proceedings of the Human Factors and Ergonomics Society Annual Meeting. DOI: 10.1177/10711813251368829.  
[14] Green, D. M., & Swets, J. A. (1966). Signal Detection Theory and Psychophysics. Wiley.  
[15] Macmillan, N. A., & Creelman, C. D. (2005). Detection Theory: A User's Guide (2nd ed.). Lawrence Erlbaum Associates. DOI: 10.4324/9781410611147.  
[16] O'Connor, M., et al. (2024). A systematic review... Computational and Structural Biotechnology Journal, 23:101-120. DOI: 10.1016/j.csbj.2024.07.015.  
[17] Köhler, S., et al. (2025). Interacting with fallible AI... Frontiers in Psychology, 16:1574809. DOI: 10.3389/fpsyg.2025.1574809.  
[18] Sorkin, R. D., & Dai, H. (1994). Signal detection analysis of the ideal group. Organizational Behavior and Human Decision Processes, 60(1), 1-13. DOI: 10.1006/obhd.1994.1072.
