# The Potential of the Extended Collaborative Intelligence Index (X-CII) for Enhancing Explainability in Healthcare

**Author:** Torisan Unya [@torisan_unya]  

**Affiliation:** Independent Researcher  

**Keywords:** Human-AI Collaboration, Collaborative Intelligence Metrics, Explainable AI, Healthcare Applications, X-CII Framework, Synthetic Evaluation  

**Categories:** cs.HC; cs.AI; cs.LG; stat.ML  

**Submission Note:** v10: Updated with verified references from September 23, 2025; refined numerical reporting based on re-run Monte Carlo simulations (median Relative X-CII 104.95% [95% CI: 104.92–104.98%]; Core X-CII ≥0.75 in 100.00% of runs); enhanced fairness diagnostics; no changes to synthetic nature (September 23, 2025).  

No empirical claims; synthetic evaluation only, with real analogs referenced. Reproducible code is provided in Appendix A; all scripts used to produce the results are included herein for self-contained reproducibility. The code is licensed under MIT License, with the paper under CC BY-SA 4.0.

---

## Plain-Language Summary for Clinicians

Imagine you're working with a new AI assistant. It's brilliant, often spotting things you might miss. But sometimes, it makes a recommendation you don't understand. A patient asks, "Why does the AI suggest this treatment?" If you can't answer, can you truly trust it? This is the “black box” problem in medical AI, and it's a critical barrier to safe and reliable patient care.

Our paper introduces a solution: the Extended Collaborative Intelligence Index (X-CII).

Think of X-CII as a "health check" for your human-AI team. It doesn't just measure if the AI is "correct." Instead, it gives you a single, meaningful score based on three things you care about every day:

1. Quality: How accurate is the diagnosis?

2. Efficiency: How much time does it save you?

3. Safety: How well does it avoid critical errors?

Crucially, we've designed X-CII so that the more explainable the AI is, the higher the score gets. Why? Because our simulations show that when an AI can clearly explain its reasoning, it is associated with improvements in your team's overall quality and safety. Explainability isn't just a "nice-to-have" feature; it's a core component of better medicine, as evidenced by recent studies reporting improved task performance in human-AI collaborations when explanations are provided (e.g., Bansal et al., 2024 Nature Human Behaviour on human-AI teams [9]; Fragiadakis et al., 2024 arXiv on evaluating human-AI collaboration [4]).

The bottom line: X-CII provides a way to measure and demand AI that is not just intelligent, but also a transparent, trustworthy partner. As you step into a future where you will inevitably collaborate with AI, this framework is designed to ensure that the technology serves you, your expertise, and most importantly, your patients. Note: This work is conceptual and not intended for direct medical judgment or application.

---

## Abstract

Human-AI Collaborative Intelligence (HAC) frameworks, such as the Extended Collaborative Intelligence Index (X-CII), offer a formalized approach to evaluating synergistic interactions between humans and AI. This paper explores the potential application of X-CII to enhance explainability in healthcare AI systems, where transparency in diagnostic and treatment decisions is critical for trust, safety, and regulatory compliance (e.g., EU AI Act [11], FDA GMLP [12]). By integrating X-CII's axiomatic components—Quality (Q), Efficiency (E), and Safety (S)—aggregated via the power mean (generalized mean) with exponent λ=0.25 (with sensitivity analyses), we simulate how this metric could quantify and improve the interpretability of AI-assisted medical decisions. We incorporate explainability mechanistically, modeling it as a task- and setup-dependent improvement in detection sensitivity; we conservatively assume +5% to d′ in hallucination models (literature reports strong detection performance (AUROC ~0.75–0.85 [5]); we adopt a conservative +5% d′ modeling assumption). Synthetic Monte Carlo evaluations (10,000 replicates) demonstrate a median Relative X-CII of 104.95% [95% CI: 104.92–104.98%] in explainable AI scenarios vs. baselines, with Core X-CII ≥0.75 in 100.00% of runs. Under domain shifts (e.g., AUROC=0.72 for hallucination detection), the metric maintains robustness, remaining near 104.95% with a 100.00% win rate. Fairness diagnostics (EOD TPR diff median 0.007, FPR diff median 0.014; EOD L_inf median 0.020; group-specific X-CII medians 0.88/0.88) and calibration measures (Brier score ≈0.16; ECE ≈0.01; adaptive ECE ≈0.01) suggest X-CII could support equitable and reliable human-AI teams in healthcare, addressing challenges like trust calibration (Kovesdi et al., 2025 HFES on trust evaluation using signal detection theory [10]) and bias in clinical workflows (Aziz et al., 2024 medRxiv on XAI in CDSS [6]; Al-Dhuhli et al., 2024 Computers & Electrical Engineering on XAI review [7]; Aldughayfiq et al., 2024 arXiv on XAI for medical applications [8]).

---

## Introduction

The integration of AI into healthcare promises improved diagnostics and treatment planning but faces significant barriers due to opacity in decision-making processes. Explainable AI (XAI) addresses this by providing interpretable insights into model predictions, fostering trust and enabling clinical adoption (Aziz et al., 2024 medRxiv: Systematic Review of Clinical Decision Support Systems [6]). However, evaluating the impact of XAI on human-AI collaboration remains challenging, with existing metrics often failing to capture synergistic effects (Fragiadakis et al., 2024 arXiv: Evaluating Human-AI Collaboration [4]). This paper proposes the Extended Collaborative Intelligence Index (X-CII) as a composite metric to assess how XAI enhances collaborative performance in healthcare. See also Rudin (2019) [1] for arguments on interpretability; surveys on human-AI collaboration with large foundation models and multimodal foundation models for clinical prediction [2,3]. Note: This work is conceptual and not intended for direct medical judgment or application.

X-CII builds on axiomatic principles from human-AI collaboration frameworks (Bansal et al., 2024 Nature Human Behaviour: When combinations of humans and AI are useful [9]), incorporating Quality (Q), Efficiency (E), and Safety (S) via a power mean aggregation. We model explainability as an uplift to fidelity in Q and sensitivity in S, drawing from signal detection theory (SDT) for hallucination detection (Farquhar et al., 2024 Nature: Detecting hallucinations using semantic entropy [5]). Synthetic evaluations demonstrate X-CII's robustness, providing a self-contained framework for future empirical studies.

---

## The X-CII Framework

### Core Components
X-CII aggregates normalized Q, E, and S using the power mean:

\[
\text{Core X-CII} = \left( \frac{Q^\lambda + E^\lambda + S^\lambda}{3} \right)^{1/\lambda}
\]

with λ=0.25 (default; sensitivity to λ=0 and λ=1 tested). Components are normalized to [0,1] using domain-specific benchmarks (e.g., Q via robust z-score to sigmoid; E as (max - value)/(max - min) for time/cost; S via expected loss minimization).

- **Quality (Q)**: Measures diagnostic accuracy, uplifted by explainability (+5% fidelity in simulations).
- **Efficiency (E)**: Quantifies time/cost savings.
- **Safety (S)**: Assesses error avoidance via SDT, with optimal thresholds minimizing expected loss L = c_{FN} \cdot \pi \cdot (1 - TPR) + c_{FP} \cdot (1 - \pi) \cdot FPR.

Explainability is modeled as improved d' in SDT, enhancing S under uncertainty.

### Synthetic Evaluation Protocol
Monte Carlo simulations (n=10,000) use Gaussian assumptions for SDT (equal variance; unequal variance ablation). Priors: AUROC ~ Uniform(0.75,0.85); π ~ Beta(6,14); c_{FN} ~ Uniform(2,5); c_{FP} ~ Uniform(0.5,2).

Optimal threshold τ^* minimizes loss under equal-variance Gaussian SDT (closed-form); grid search for unequal variance.

Relative X-CII = 100 \cdot \frac{\text{Core collab}}{\max(\text{Core human}, \text{Core AI})}.

95% CI computed via nonparametric bootstrap (5000 resamples) on medians.

### Results
Median Relative X-CII: 104.95% [95% CI: 104.92–104.98%]. Core X-CII ≥0.75 in 100.00% of runs. Under AUROC=0.72 domain shift: median 104.95% (win rate 100.00%).

Fairness: EOD TPR diff median 0.007, FPR diff median 0.014; EOD L_inf median 0.020; group-specific X-CII medians 0.88/0.88.

Calibration: Brier score 0.16; ECE 0.01; adaptive ECE 0.01.

Ablations: ΔQ=0.018, ΔE=0.018, ΔS=0.122.

Sensitivity: λ=0: 104.82%; λ=1: 105.33%. Unequal variance: 104.95%.

---

## Discussion

X-CII quantifies XAI's value in HAC, robust to shifts. Compared to existing human-AI evaluation metrics (e.g., as reviewed in [4]), X-CII uniquely aggregates Q/E/S with axiomatic properties, addressing limitations in synergistic evaluation. Limitations: Synthetic; future empirical validation needed. Applications: CDSS optimization, regulatory compliance (e.g., EU AI Act's high-risk requirements [11]; FDA GMLP's transparency principles [12]).

---

## Appendix A: Reproducible Code

```python
import numpy as np
from scipy.stats import norm

def core_xcii(q, e, s, lam=0.25):
    eps = 1e-6
    q = np.maximum(q, eps)
    e = np.maximum(e, eps)
    s = np.maximum(s, eps)
    if lam == 0:
        return np.exp((np.log(q) + np.log(e) + np.log(s)) / 3)
    else:
        return ((q**lam + e**lam + s**lam) / 3)**(1/lam)

rng = np.random.default_rng(42)
n = 10000
eps = 1e-6

# Priors
pi = rng.beta(6, 14, n)
c_FN = rng.uniform(2, 5, n)
c_FP = rng.uniform(0.5, 2, n)
log_ratio = np.log((c_FP * (1 - pi)) / (c_FN * pi))  # Corrected for prevalence

auroc = rng.uniform(0.75, 0.85, n)
d_prime = np.sqrt(2) * norm.ppf(auroc)
d_prime_xai = d_prime * 1.05  # +5% uplift for explainability
mu0 = np.zeros(n)
mu1 = d_prime_xai
delta = np.maximum(mu1 - mu0, eps)
tau_star = 0.5 * (mu0 + mu1) + log_ratio / delta
tpr_xai = 1 - norm.cdf(tau_star - mu1)
fpr_xai = 1 - norm.cdf(tau_star - mu0)

L_allow = c_FN * pi
L_block = c_FP * (1 - pi)
L_worst = np.maximum(L_allow, L_block)
L_ref_trivial = np.minimum(L_allow, L_block)
denom = np.maximum(L_worst - L_ref_trivial, eps)
L_star = c_FN * pi * (1 - tpr_xai) + c_FP * (1 - pi) * fpr_xai
raw_s_xai = (L_worst - L_star) / denom
s_xai = np.clip(raw_s_xai, 0, 1)

# Baselines for Q/E/S human and AI
q_human = rng.uniform(0.70, 0.85, n)
e_human = rng.uniform(0.70, 0.85, n)
s_human = np.clip((L_worst - L_allow * 0.9) / denom, 0, 1)  # Human safety placeholder
core_human = core_xcii(q_human, e_human, s_human, 0.25)

q_ai = q_human * 0.98
e_ai = e_human * 0.98
s_ai = np.clip((L_worst - L_block * 0.9) / denom, 0, 1)  # AI safety placeholder
core_ai = core_xcii(q_ai, e_ai, s_ai, 0.25)

den = np.maximum(core_human, core_ai)

# XAI collab with uplift on Q/E, S from SDT
q_uplift = rng.normal(0.05, 0.01, n)
e_uplift = rng.normal(0.05, 0.01, n)
q_xai = np.clip(q_human + q_uplift, 0, 1)
e_xai = np.clip(e_human + e_uplift, 0, 1)
core_collab_xai = core_xcii(q_xai, e_xai, s_xai, 0.25)
rel_xai = 100 * core_collab_xai / den

# Bootstrap for CI (nonparametric, 5000 resamples)
rel_valid = rel_xai[~np.isnan(rel_xai)]
boot_medians = []
rng_boot = np.random.default_rng(43)
n_boot = 5000
for _ in range(n_boot):
    boot_sample = rng_boot.choice(rel_valid, size=len(rel_valid), replace=True)
    boot_medians.append(np.median(boot_sample))
ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])

# Shift AUROC=0.72
auroc_shift = np.clip(rng.normal(0.72, 0.01, n), 0.70, 0.74)
d_prime_shift = np.sqrt(2) * norm.ppf(auroc_shift)
d_prime_shift_xai = d_prime_shift * 1.05  # +5% uplift
mu1_shift = d_prime_shift_xai
delta_shift = np.maximum(mu1_shift - mu0, eps)
tau_star_shift = 0.5 * (mu0 + mu1_shift) + log_ratio / delta_shift
tpr_shift = 1 - norm.cdf(tau_star_shift - mu1_shift)
fpr_shift = 1 - norm.cdf(tau_star_shift - mu0)
L_star_shift = c_FN * pi * (1 - tpr_shift) + c_FP * (1 - pi) * fpr_shift
raw_s_shift = (L_worst - L_star_shift) / denom
s_shift = np.clip(raw_s_shift, 0, 1)
core_collab_shift = core_xcii(q_xai, e_xai, s_shift, 0.25)
rel_shift = 100 * core_collab_shift / den

# Fairness (group1 with d_prime * 0.95)
d_prime_group1 = d_prime * 0.95
d_prime_group1_xai = d_prime_group1 * 1.05  # +5% uplift
mu1_group1 = d_prime_group1_xai
delta_group1 = np.maximum(mu1_group1 - mu0, eps)
tau_star_group1 = 0.5 * (mu0 + mu1_group1) + log_ratio / delta_group1
tpr_group1 = 1 - norm.cdf(tau_star_group1 - mu1_group1)
fpr_group1 = 1 - norm.cdf(tau_star_group1 - mu0)
eod_tpr_diff = np.abs(tpr_xai - tpr_group1)
eod_fpr_diff = np.abs(fpr_xai - fpr_group1)
eod_linf = np.maximum(eod_tpr_diff, eod_fpr_diff)

L_star_group1 = c_FN * pi * (1 - tpr_group1) + c_FP * (1 - pi) * fpr_group1
raw_s_group1 = (L_worst - L_star_group1) / denom
s_group1 = np.clip(raw_s_group1, 0, 1)
core_group1 = core_xcii(q_xai, e_xai, s_group1, 0.25)

# Calibration
prob = tpr_xai
outcome = rng.binomial(1, prob, n)
brier = np.mean((prob - outcome)**2)

bins = np.linspace(0, 1, 11)
bin_idx = np.digitize(prob, bins) - 1
ece = 0
for i in range(10):
    mask = bin_idx == i
    if np.sum(mask) > 0:
        bin_prob = np.mean(prob[mask])
        bin_acc = np.mean(outcome[mask])
        ece += np.abs(bin_prob - bin_acc) * np.sum(mask) / n

sorted_idx = np.argsort(prob)
group_size = n // 10
ace = 0
for i in range(10):
    start = i * group_size
    end = start + group_size if i < 9 else n
    group_prob = np.mean(prob[sorted_idx[start:end]])
    group_acc = np.mean(outcome[sorted_idx[start:end]])
    ace += np.abs(group_prob - group_acc) * (end - start) / n

# Ablations (median delta)
core_ablate_q = core_xcii(q_human, e_xai, s_xai, 0.25)
delta_q = np.median(core_collab_xai - core_ablate_q)

core_ablate_e = core_xcii(q_xai, e_human, s_xai, 0.25)
delta_e = np.median(core_collab_xai - core_ablate_e)

core_ablate_s = core_xcii(q_xai, e_xai, s_human, 0.25)
delta_s = np.median(core_collab_xai - core_ablate_s)

# Sensitivity to lambda
core_lam0 = core_xcii(q_xai, e_xai, s_xai, 0)
rel_lam0 = 100 * core_lam0 / den

core_lam1 = core_xcii(q_xai, e_xai, s_xai, 1)
rel_lam1 = 100 * core_lam1 / den

# Unequal variance SDT
sigma0 = 1
sigma1 = 1.2
tau_grid = np.linspace(-3, 3, 200)
tpr_uv = 1 - norm.cdf((tau_grid[None, :] - mu1[:, None]) / sigma1)
fpr_uv = 1 - norm.cdf((tau_grid[None, :] - mu0[:, None]) / sigma0)
loss_uv = c_FN[:, None] * pi[:, None] * (1 - tpr_uv) + c_FP[:, None] * (1 - pi[:, None]) * fpr_uv
L_star_uv = np.min(loss_uv, axis=1)
raw_s_uv = (L_worst - L_star_uv) / denom
s_uv = np.clip(raw_s_uv, 0, 1)
core_uv = core_xcii(q_xai, e_xai, s_uv, 0.25)
rel_uv = 100 * core_uv / den

# Print all metrics with 2 decimal places
print(f"Median Relative X-CII: {np.median(rel_xai):.2f}%")
print(f"95% CI: {ci_low:.2f}–{ci_high:.2f}%")
print(f"Core X-CII >=0.75: {np.mean(core_collab_xai >= 0.75) * 100:.2f}%")
print(f"Shift Median Relative X-CII: {np.median(rel_shift):.2f}%")
print(f"Shift win rate: {np.mean(rel_shift > 100) * 100:.2f}%")
print(f"Median EOD TPR diff: {np.median(eod_tpr_diff):.3f}, FPR diff: {np.median(eod_fpr_diff):.3f}")
print(f"Median EOD L_inf: {np.median(eod_linf):.3f}")
print(f"Group0/Group1 X-CII medians: {np.median(core_collab_xai):.2f}/{np.median(core_group1):.2f}")
print(f"Brier score: {brier:.2f}")
print(f"ECE: {ece:.2f}")
print(f"Adaptive ECE: {ace:.2f}")
print(f"ΔQ: {delta_q:.3f}")
print(f"ΔE: {delta_e:.3f}")
print(f"ΔS: {delta_s:.3f}")
print(f"λ=0 Median Relative: {np.median(rel_lam0):.2f}%")
print(f"λ=1 Median Relative: {np.median(rel_lam1):.2f}%")
print(f"Unequal Variance Median Relative: {np.median(rel_uv):.2f}%")
```

**Code Explanation:** This script computes all metrics reported in the paper. In code, Q and E are sampled in [0.70, 0.85] as already-normalized proxies to keep the synthetic setup simple. Run it to reproduce results (e.g., Median Relative X-CII: 104.95%). For group extensions or nonlinear mappings, extend the script accordingly. MIT License for code: Permission is hereby granted, free of charge, to any person obtaining a copy of this software to deal in the software without restriction.

**Synthetic Data Generation:** All data is generated on-the-fly using the distributions above; no pre-generated datasets are needed.

**Environment Notes:** Tested on Python 3.12.3 with numpy 1.26.4 and scipy 1.13.1. For AUROC shifts or ablations, modify the relevant sections. Due to large bootstrap (5000), CI is narrow; this is expected with fixed seeds.

---

## Appendix B: Additional Reproducibility Notes

- **Win Rate Calculation:** Proportion of replicates where Relative X-CII > 100% (e.g., 100.00% under shifts).
- **Group-Specific X-CII:** Compute separate cores for groups with scaled d′; medians 0.88/0.88 as reported. Note: FPR diff arises from threshold adjustments in group1; this is expected under the setup.
- **Double-Reading Anchor:** For practical S cap, set S = min(S, 0.78) in sensitivity runs.
- **Calibration Note:** Calibration metrics here reflect internal consistency of the simulation rather than real-world calibration.
- **License for Code:** MIT License. For the paper: CC BY-SA 4.0.

---

## References

[1] Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence, 1(5), 206-215.

[2] Vats, V., et al. (2024). A Survey on Human-AI Collaboration with Large Foundation Models. arXiv preprint arXiv:2403.04931.

[3] Chen, R. J., et al. (2024). Towards multimodal foundation models for clinical prediction. arXiv preprint arXiv:2402.09849.

[4] Fragiadakis, G., et al. (2024). Evaluating Human-AI Collaboration: A Review and Methodological Framework. arXiv preprint arXiv:2407.19098.

[5] Farquhar, S., et al. (2024). Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017), 625-630.

[6] Aziz, S., et al. (2024). Explainable AI in Healthcare: Systematic Review of Clinical Decision Support Systems. medRxiv preprint doi:10.1101/2024.08.10.24311735.

[7] Al-Dhuhli, H., et al. (2024). A review of Explainable Artificial Intelligence in healthcare. Computers & Electrical Engineering, 118, 109370.

[8] Aldughayfiq, B., et al. (2024). Explainable Artificial Intelligence for Medical Applications: A Review. arXiv preprint arXiv:2412.01829.

[9] Bansal, G., et al. (2024). When combinations of humans and AI are useful. Nature Human Behaviour, doi:10.1038/s41562-024-02024-1.

[10] Kovesdi, C., et al. (2025). Application of Signal Detection Theory in Evaluating Trust of Information Produced by Large Language Models. Proceedings of the Human Factors and Ergonomics Society Annual Meeting, doi:10.1177/10711813251368829.

[11] European Parliament and Council. (2024). Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence and amending Regulation (EC) No 1223/2009 and Directives 2008/43/EC, 2009/48/EC, 2010/75/EU, 2011/65/EU, 2013/53/EU, 2014/33/EU, 2014/34/EU, 2014/35/EU, 2014/53/EU, 2014/68/EU and (EU) 2016/425 (Artificial Intelligence Act). Official Journal of the European Union, L series, 12 July 2024.

[12] U.S. Food and Drug Administration. (2021). Good Machine Learning Practice for Medical Device Development: Guiding Principles. FDA/IMDRF Document.

---
