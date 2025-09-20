# A Formalization of the Extended Collaborative Intelligence Index (X-CII): Definition and Synthetic Evaluation

**Author:** Torisan Unya  
**Affiliation:** Independent Researcher  
**Keywords:** Human-AI Collaboration, Collaborative Intelligence Metrics, Synthetic Evaluation, Threshold Optimization, Monte Carlo Simulation  
**arXiv Categories:** cs.HC; cs.AI; stat.ML  

**arXiv Submission Note:** v1: Finalized for baseline clarity, fairness details, and minor phrasing (September 21, 2025).  
No empirical claims; synthetic evaluation only.

---  

## Abstract  

We present a theoretical formalization and synthetic evaluation (no real-world claims). Human-AI collaboration is increasingly central to domains such as scientific research, creative industries, business strategy, and education, yet standardized metrics for assessing its effectiveness remain underdeveloped. This paper formalizes the Extended Collaborative Intelligence Index (X-CII), a composite metric capturing quality (\( Q \)), efficiency (\( E \)), and safety (\( S \)) in collaborative processes. We define X-CII via axiomatic properties (e.g., scale invariance under shared normalization bounds and pre-specified reference distributions, monotonicity) and propose aggregation rules, including Box-Cox transformations for bounded normalization and expected loss minimization for safety thresholds. To evaluate its conceptual properties in synthetic simulations, we conduct a Monte Carlo evaluation on generated data (10,000 replicates), demonstrating robustness under uncertainty (e.g., domain shifts, parameter variations). Results indicate simulated median relative scores of 108.7\% [95\% CI: 107.2-110.1\%] vs. the best single-agent baseline across synthetic configurations (5-95th percentile: 104.3-112.8\%), Core X-CII (collab) distribution: 5-95th percentile 0.80-0.88; none observed <0.75 in our runs (synthetic setup-dependent). Under an AUROC=0.72 shift scenario, the median relative score drops to 105.2\% (win rate: 95\%). This work provides a reproducible framework for future empirical validation, with all necessary equations, pseudo-code, and hyperparameters detailed in the appendix.  

---  

## 1. Introduction  

The integration of AI into human workflows promises synergistic gains but introduces challenges in measurement, including dynamic interactions, uncertainty, and safety risks (e.g., hallucinations, biases). Existing frameworks, such as those in HAIC evaluations (Fragiadakis et al., 2024 [11]; https://arxiv.org/abs/2407.19098v2), emphasize qualitative paradigms but lack formalized, composite indices for quantitative appraisal. Recent surveys highlight the need for metrics balancing complementarity and risks (Vats et al., 2024 [45]; https://arxiv.org/abs/2403.04931v3).  

This paper addresses this gap by formalizing the Extended Collaborative Intelligence Index (X-CII). Contributions are:  
1. Axiomatic definition of X-CII as a Box-Cox average (power mean) of normalized \( Q \), \( E \), \( S \) components (reducing to geometric mean in the \( \lambda \to 0 \) limit).  
2. Methodological protocols for threshold optimization (closed-form under Gaussian assumptions) and aggregation (for stability).  
3. Synthetic Monte Carlo evaluation protocol, assessing robustness on generated data.  
4. Implications for domain adaptation and future validation.  

We position X-CII as a conceptual tool for appraising collaborative effectiveness, not an empirical claim. Section 2 reviews related work; Section 3 defines the framework; Section 4 details methods; Section 5 describes simulations; Section 6 presents results; Section 7 discusses limitations; Section 8 concludes.  

---  

## 2. Related Work  

Human-AI collaboration metrics draw from HCI, ML, and decision theory. Amershi et al. (2019) [4] outline interaction guidelines, while Bansal et al. (2024) [44] quantify synergies (e.g., double-digit improvements with notable underperformance risks). Safety integrates hallucination detection (Farquhar et al., 2024 [6]; DOI: 10.1038/s41586-024-07421-0; reports of AUROC degradation to around 0.72 in low-resource settings; cf. recent benchmarks in Manakul et al., 2023 [60]; https://arxiv.org/abs/2303.08896) and bias metrics (e.g., Equalized Odds Difference; Hardt et al., 2016 [12]; DOI: 10.48550/arXiv.1610.02413).  

Frameworks like HAIC (Fragiadakis et al., 2024 [11]; https://arxiv.org/abs/2407.19098v2) use decision trees for mode-specific evaluation (AI-centric, human-centric, symbiotic), integrating quantitative (e.g., accuracy, time) and qualitative metrics. Recent surveys on human-AI collaboration with large foundation models (Vats et al., 2024 [45]; https://arxiv.org/abs/2403.04931v3) highlight the need for composite indices that account for complementarity in decision-making. Recent works extend this, e.g., Human-Centered Human-AI Collaboration (HCHAC) (Gao et al., 2025 [61]; https://arxiv.org/abs/2505.22477) on human-centered HAC relationships. Our work extends these by formalizing a bounded power mean generalization of the geometric mean (multiplicative in the \( \lambda \to 0 \) limit; see Macmillan & Creelman, 2005 [17]; DOI: 10.4324/9781410611147 for foundational signal detection theory), with synthetic robustness testing, akin to uncertainty-aware delegation approaches (e.g., Provost & Fawcett, 2001 [13]; DOI: 10.1023/A:1007601013934 for ROC analysis). For hallucination, we draw from probabilistic frameworks achieving AUROC \sim 0.75-0.85 (Hou et al., 2024 [39]; https://arxiv.org/abs/2406.06950; updated with semantic entropy baselines in Farquhar et al., 2024 [6]).  

Gaps include limited stochastic appraisal of composite metrics; we address via Monte Carlo on synthetic data, avoiding empirical overreach.  

---  

## 3. Theoretical Framework  

### 3.1 X-CII Definition  

X-CII quantifies collaborative effectiveness on [0,1] (Core) and relative (\%) scales, using the power mean \( M_\lambda \) in Box-Cox form (as \( \lambda \to 0 \) it becomes the geometric mean, \( \lim_{\lambda \to 0} M_\lambda = \exp\left(\frac{1}{3} \sum \log x_i\right) \), which penalizes imbalances strongly by making it hard for high values in one component to offset low values in others; smaller \( \lambda \) (approaching 0) yields stronger imbalance penalties, while larger \( \lambda \) relaxes toward arithmetic mean and milder penalties; note the equivalence \( M_\lambda = \left( \frac{1}{3} \sum x_i^\lambda \right)^{1/\lambda} \), which aligns with the Box-Cox inverse):  

\[  
\text{Core X-CII} = g^{-1}\left( \frac{g(Q) + g(E) + g(S)}{3} \right)  
\]  

where \( g(x) = \frac{x^\lambda - 1}{\lambda} \) (\( \lambda = 0.25 \) by default; \( \varepsilon = 10^{-6} \) for clipping), and \( g^{-1}(y) = (\lambda y + 1)^{1/\lambda} \). For numerical stability, first normalize \( S \) to [0,1] (raw \( S \) may exceed 1 for reporting), then apply an \( \varepsilon \)-floor to \( Q/E/S \) right before the Box-Cox transform. This ensures Core \( \in [\varepsilon,1] \). \( S=1 \) reflects parity with the best trivial policy (always-allow or always-block; raw \( S>1 \) indicates exceedance and is reported separately). Primary: trivial-anchored normalization with \( L_{\text{ref_trivial}} = \min(L_{\text{allow}}, L_{\text{block}}) \) and \( L_{\text{worst}} = \max(L_{\text{allow}}, L_{\text{block}}) \) (\( S \in [0,1] \)). Reference: human-anchored variant \( s_{\text{human_anchored}} \) in Appendix A. Cross-domain comparisons are limited to relative evaluations within each domain to avoid unit mismatches. Scale and unit invariance hold under shared normalization bounds (\( t_{\min}/t_{\max} \), \( C_{\min}/C_{\max} \)) and pre-specified benchmark distribution for \( Q \)'s \( z_q \)-score (to avoid leakage; fixed within comparisons); for \( E \), min-max bounds are pre-registered per domain.  

\[  
\text{Relative X-CII (\%)} = 100 \times \frac{\text{Core X-CII}_{\text{collab}}}{\max(\text{Core X-CII}_{\text{human}}, \text{Core X-CII}_{\text{AI}})}  
\]  

If \( \max(\text{Core}_{\text{human}}, \text{Core}_{\text{AI}}) < 0.1 \), Relative X-CII is reported as N/A (protective threshold to avoid numerical instability from small denominators in low-performance regimes). Baselines use identical tasks/constraints. AI-only baseline is a conservative placeholder (allow-all policy); in practice, compare to a fully optimized model for the ...(truncated 3342 characters)...allow, L_block)  
L_worst = np.maximum(L_allow, L_block)  # Primary: trivial-anchored  
eps = 1e-6  
# Closed-form tau* and S (equal-variance Gaussian)  
d_prime = np.sqrt(2) * norm.ppf(auroc_sample)  
mu0 = np.zeros(n)  
mu1 = d_prime  
delta = np.maximum(mu1 - mu0, eps)  
log_ratio = np.log((c_FP * (1 - pi) + eps) / (c_FN * pi + eps))  
tau_star = 0.5 * (mu0 + mu1) + log_ratio / delta  
tpr = 1 - norm.cdf(tau_star - mu1)  
fpr = 1 - norm.cdf(tau_star - mu0)  
L_star = c_FN * pi * (1 - tpr) + c_FP * (1 - pi) * fpr  
# Raw S before clipping for collab  
raw_s_collab = (L_worst - L_star) / np.maximum(L_worst - L_ref_trivial, eps)  
s_collab = np.clip(raw_s_collab, 0, 1)  
# Trivial-anchored for human and AI baselines  
l_human = np.minimum(l_allow, l_block) * 0.9  # Appendix B baseline
raw_s_human = (l_worst - l_human) / np.maximum(l_worst - l_ref_trivial, eps)  
s_human = np.clip(raw_s_human, 0, 1)  
raw_s_ai = (l_worst - l_allow) / np.maximum(l_worst - l_ref_trivial, eps)  
s_ai = np.clip(raw_s_ai, 0, 1)  
# Human-anchored S (reference diagnostic only; not used in Core calculations)  
l_worst_human = np.maximum(l_worst, l_human)  
s_human_anchored = (l_worst_human - L_star) / np.maximum(l_worst_human - l_human, eps)  # Reference only  
# Components (synthetic)  
q_human = rng.uniform(0.7, 0.85, n)  
e_human = rng.uniform(0.7, 0.85, n)  
core_human = core_xcii(q_human, e_human, s_human, lam=0.25)  
q_ai = rng.uniform(0.7, 0.85, n) * 0.95  
e_ai = rng.uniform(0.7, 0.85, n) * 0.98  
core_ai = core_xcii(q_ai, e_ai, s_ai, lam=0.25)  
mult_q = np.exp(rng.normal(0.13, 0.05, n))  
q_collab = np.clip(q_human * mult_q, eps, 1.0)  
e_collab = np.clip(e_human * np.exp(rng.normal(0.13, 0.05, n)), eps, 1.0)  
core_collab = core_xcii(q_collab, e_collab, s_collab, lam=0.25)  
den = np.maximum(core_human, core_ai)  
rel = np.where(den >= 0.1, 100 * core_collab / den, np.nan)  
# Raw S >1 proportion  
raw_s_exceed = np.mean(raw_s_collab > 1.0)  
print(f"Raw S >1 proportion: {raw_s_exceed:.2%}")  

# Simplified bootstrap for 95% CI on median relative (nonparametric, 10,000 resamples; excluding N/A)  
rel_valid = rel[~np.isnan(rel)]  
n_boot = 10000  
boot_medians = []  
rng_boot = np.random.default_rng(43)  # Separate seed for bootstrap  
for _ in range(n_boot):  
    boot_sample = rng_boot.choice(rel_valid, size=len(rel_valid), replace=True)  
    boot_medians.append(np.median(boot_sample))  
ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])  
print(f"Median relative: {np.median(rel_valid):.1f}% [95% CI: {ci_low:.1f}-{ci_high:.1f}%]")  
p5, p95 = np.percentile(rel_valid, [5, 95])  
print(f"5-95th percentile: {p5:.1f}-{p95:.1f}%")  

# AUROC=0.72 shift scenario (for reproducibility; constant AUROC)  
auroc_shift = np.full(n, 0.72) + rng.normal(0, 0.01, n)  # ±1% jitter for var
auroc_shift = np.clip(auroc_shift, 0.70, 0.74)  # Pessimistic bound
d_prime_shift = np.sqrt(2) * norm.ppf(auroc_shift)  
mu1_shift = d_prime_shift  
delta_shift = np.maximum(mu1_shift - mu0, eps)  # Corrected for shift  
tau_star_shift = 0.5 * (mu0 + mu1_shift) + log_ratio / delta_shift  
tpr_shift = 1 - norm.cdf(tau_star_shift - mu1_shift)  
fpr_shift = 1 - norm.cdf(tau_star_shift - mu0)  
L_star_shift = c_FN * pi * (1 - tpr_shift) + c_FP * (1 - pi) * fpr_shift  
raw_s_collab_shift = (l_worst - L_star_shift) / np.maximum(l_worst - l_ref_trivial, eps)  
s_collab_shift = np.clip(raw_s_collab_shift, 0, 1)  
core_collab_shift = core_xcii(q_collab, e_collab, s_collab_shift, lam=0.25)  
rel_shift = np.where(den >= 0.1, 100 * core_collab_shift / den, np.nan)  
rel_shift_valid = rel_shift[~np.isnan(rel_shift)]
shift_median = np.median(rel_shift_valid)
win_rate_shift = float(np.mean(rel_shift_valid > 100) * 100)
print(f"Shift median: {shift_median:.1f}% (win rate: {win_rate_shift:.0f}%)")  

# Fairness diagnostics (simplified snippet; assumes 2 groups with per-group shifts in d'; thresholds applied globally; TPR-FPR diff as proxy)  
tpr_group1 = 1 - norm.cdf(tau_star - mu1 * 0.95)  # Simplified: d' shift for group 1, global threshold  
fpr_group1 = 1 - norm.cdf(tau_star - mu0)  
eod_linf = np.maximum(np.abs(tpr - tpr_group1), np.abs(fpr - fpr_group1))  
calib_gap = np.abs(tpr - fpr)  # Simplified calibration gap proxy (TPR-FPR difference)  
print(f"EOD L_inf median: {np.median(eod_linf):.2f}, Calibration gap proxy median: {np.median(calib_gap):.2f}")  
```  

---  

## Appendix B: Robust Threshold Optimization and Fairness Diagnostics  

Implementation details use the equal-variance Gaussian closed-form \( \tau^* \) in our simulations. ROC convex-hull optimization is an optional extension and not used here. Fairness: EOD \( L_{\infty} = \max(|\Delta \mathrm{TPR}|, |\Delta \mathrm{FPR}|) \) median 0.02 [0.00-0.05]; calibration gap proxy median 0.40 (computed as \( \mathrm{TPR} - \mathrm{FPR} \) absolute difference proxy over stratified groups; note: this is a simplified proxy differing from strict calibration definitions). For Q normalization in practice: Use a fixed reference distribution (\( n \ge 1000 \), updated quarterly); robust z-score via median/MAD; winsorize to [-3,3] before sigmoid. X-CII uses equal weights for Q/E/S; extensions with domain-specific weights are possible (e.g., \( g^{-1}\left( \frac{w_Q g(Q) + w_E g(E) + w_S g(S)}{\sum w} \right) \)). Baselines: human = min(L_allow, L_block)*0.9; AI = allow-all (L_allow). For E normalization in practice: E_time = (t_max - t)/(t_max - t_min), E_cost = (c_max - c)/(c_max - c_min); overall E as weighted harmonic mean (default w_time = w_cost = 0.5), with pre-registered min/max from domain benchmarks (if min=max, set to 0.5). The denominator threshold of 0.1 avoids numerical instability from small values in low-performance regimes.  

**Simulation Distribution Parameters** (as used in Monte Carlo replicates):  
- AUROC ~ Uniform(0.75, 0.85)  
- pi ~ Beta(6, 14)  
- c_FN ~ Uniform(2, 5)  
- c_FP ~ Uniform(0.5, 2)  
- Q/E human ~ Uniform(0.7, 0.85)  
- Q/E AI multipliers: 0.95 and 0.98  
- Collab multipliers: exp(Normal(0.13, 0.05))  

**Dependencies**: numpy >=1.23, scipy >=1.10 (UTF-8 encoding assumed).  

**Symbol Table**: \( Q \): Quality; \( E \): Efficiency; \( S \): Safety; \( \tau^* \): Optimal threshold; \( \lambda \): Box-Cox parameter; \( \pi \): Harmful prevalence; \( \varepsilon \): Clipping floor; \( \Phi \): Standard normal CDF (denoted as Phi in code); \( E_t \): Time efficiency; \( E_c \): Cost efficiency; \( d' \): Detectability index; \( z_q \): Robust z-score for \( Q \); EOD: Equalized Odds Difference (primary \( L_{\infty} \): max(|\Delta \mathrm{TPR}|, |\Delta \mathrm{FPR}|)); Calibration gap proxy: TPR-FPR absolute difference; \( L_{\text{allow}} \): Loss under always-allow policy; \( L_{\text{block}} \): Loss under always-block policy; \( L_{\text{ref_trivial}} \): Min of trivial losses; \( L_{\text{worst}} \): Max of trivial losses.  

---  

## Appendix C: References (Full List)  

[4] Amershi, S., et al. (2019). Guidelines for human-AI interaction. *CHI*. DOI: 10.1145/3290605.3300233.  
[6] Farquhar, S., et al. (2024). Detecting hallucinations in large language models using semantic entropy. *Nature*, 630(8017), 625-630. DOI: 10.1038/s41586-024-07421-0.  
[11] Fragiadakis, G., et al. (2024). Evaluating Human-AI Collaboration: A Review and Methodological Framework. arXiv:2407.19098v2 [cs.HC]. https://arxiv.org/abs/2407.19098v2.  
[12] Hardt, M., et al. (2016). Equality of Opportunity in Supervised Learning. *NeurIPS*. arXiv:1610.02413. https://arxiv.org/abs/1610.02413.  
[13] Provost, F., & Fawcett, T. (2001). Robust Classification for Imprecise Environments. *Machine Learning*, 42(3), 203-231. DOI: 10.1023/A:1007601013934.  
[17] Macmillan, N. A., & Creelman, C. D. (2005). *Detection Theory: A User's Guide* (2nd ed.). Lawrence Erlbaum Associates. DOI: 10.4324/9781410611147.  
[39] Hou, B., et al. (2024). A Probabilistic Framework for Large Language Model Hallucination Detection via Belief Tree Propagation. arXiv:2406.06950. https://arxiv.org/abs/2406.06950.  
[44] Bansal, G., et al. (2024). When combinations of humans and AI are useful. *Nature Human Behaviour*. DOI: 10.1038/s41562-024-02024-1.  
[45] Vats, V., et al. (2024). A Survey on Human-AI Collaboration with Large Foundation Models. arXiv:2403.04931v3 [cs.AI]. https://arxiv.org/abs/2403.04931v3.  
[60] Manakul, P., et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. Findings of EMNLP. https://arxiv.org/abs/2303.08896.  
[61] Gao, Q., et al. (2025). Human-Centered Human-AI Collaboration (HCHAC). arXiv:2505.22477 [cs.HC]. https://arxiv.org/abs/2505.22477.  

---  

**License:** CC BY-SA 4.0  

---
