# QTEG Distribution — Analysis Code

[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00362--026--01908--w-blue)](https://doi.org/10.1007/s00362-026-01908-w)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Companion code for the published paper:

> **Quadratic functional variable transformation of the exponential-gamma distribution for flexible lifetime modelling**
> Taiwo Michael Ayeni, Elijah Ayooluwa Odukoya, Yichuan Zhao
> *Statistical Papers* (2026) **67**:140
> https://doi.org/10.1007/s00362-026-01908-w

This repository is the code referenced in the paper's *Code availability* statement. It reproduces every numerical table and figure in the article.

---

## What this repository contains

| File | Description |
|---|---|
| `QTEG_Full_Analysis_v3.py` | Complete Python script reproducing all tables and figures in the paper |

---

## The QTEG distribution

Let X ~ Gamma(α, β), the normalised exponential-gamma (EGD) baseline. The Quadratic-Transformed Exponential-Gamma (QTEG) distribution is the law of Y = X². Its probability density function is

```
f(y; α, β) = β^α / (2 Γ(α)) · y^((α−2)/2) · exp(−β√y),   y > 0,  α > 0, β > 0
```

where α is the shape parameter and β is the rate parameter. The CDF is F(y) = I(α, β√y), where I is the regularised lower incomplete gamma function.

Key properties established in the paper:

- **Normalised baseline.** The raw EGD of Ogunwale et al. (2019) integrates to 2^(−k), not 1. The paper derives the normalising constant and shows that, after reparametrising α = k and β = 2λ, the corrected baseline is Gamma(α, β). Leaving out the constant shifts the log-likelihood by n log 2.
- **Heavier tail.** The tail decay exp(−β√y) is strictly heavier than the Gamma tail exp(−βy). QTEG is the q = ½ case of the generalised Gamma family and is statistically distinct from the Gamma family (Proposition 10). All raw moments exist, E[Yⁿ] = Γ(2n + α) / (β^(2n) Γ(α)), but the moment generating function does not exist for t > 0.
- **Three hazard regimes, all controlled by α** (proved with Glaser's (1980) auxiliary-function method):
  - α < 2: decreasing hazard, unbounded at the origin
  - α = 2: decreasing hazard with finite initial value β²/2
  - α > 2: unimodal (upside-down-bathtub) hazard starting at zero
- **Closed forms** for the hazard, reversed hazard, cumulative hazard, Shannon entropy, quantile function and order statistics.
- **Inference.** Semi-closed MLE with β̂(α) = α / mean(√yᵢ), proof that the MLE exists and is unique, and an analytic Fisher information matrix for Hessian-based standard errors and 95% Wald intervals.

---

## Requirements

- Python 3.12 or later
- numpy
- scipy
- pandas
- matplotlib

Install all dependencies with:

```bash
pip install numpy scipy pandas matplotlib
```

---

## How to run

```bash
python QTEG_Full_Analysis_v3.py
```

The script creates the following outputs in the same folder as the script:

| Output | Contents | Paper |
|---|---|---|
| `QTEG_results.txt` | Theoretical moments and entropy, simulation results, parameter estimates, and model-comparison tables (AIC, BIC, AICc, CAIC, HQIC, KS, CvM, AD) | Tables 2, 4–11 |
| `fig1_theoretical.png` | Probability density, hazard, reversed hazard, and cumulative hazard functions across the three simulation scenarios (2×2) | Fig. 1 |
| `fig2_simulation.png` | Simulation diagnostics: bias, RMSE, and coverage probability across scenarios (3×3) | Fig. 2 |
| `fig3_all_datasets.png` | Fitted density, empirical vs fitted CDF, and P-P plot for all three datasets (3×3) | Fig. 3 |
| `fig4_hazard.png` | Fitted hazard functions for all three datasets | Fig. 4 |

Runtime is approximately 15–20 minutes depending on hardware (the Monte Carlo study uses N = 2000 replications across 3 scenarios × 4 sample sizes).

---

## Estimation details

- Optimiser: L-BFGS-B via `scipy.optimize.minimize`.
- Starting values: a grid α₀ ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0}, with the semi-closed β̂₀ = α₀ / mean(√y) at each point. The solution with the smallest negative log-likelihood is kept. The grid is set **a priori** in `qteg_mle()`, before any simulation results are seen; the 100% convergence rate across all scenarios is post-hoc evidence that it was sufficient for the settings examined.
- Convergence tolerances: 10⁻¹³ (function value) and 10⁻⁹ (gradient norm).
- Standard errors come from the analytic observed Fisher information matrix (Section 5.5 of the paper).

---

## Simulation study

| Scenario | True α | True β | Hazard behaviour |
|---|---|---|---|
| Sc. 1 | 1.5 | 0.5 | Decreasing, unbounded initial hazard |
| Sc. 2 | 2.0 | 1.0 | Decreasing, finite initial hazard |
| Sc. 3 | 3.0 | 2.0 | Unimodal, zero initial hazard |

Sample sizes: n = 30, 50, 100, 200. Replications: N = 2000. Data are generated as Yᵢ = Xᵢ² with Xᵢ ~ Gamma(α, β).

For each scenario and sample size the script reports, separately for α̂ and β̂: convergence rate, Bias, RMSE, AvgSE (mean Hessian-based standard error), CIW (average 95% Wald interval width = 2 × 1.96 × AvgSE), and empirical coverage probability (CP).

---

## Model comparison

For each dataset, QTEG is compared with eight competing distributions: Weibull, Gamma, Log-Normal, Exponential, normalised EGD (equal to Gamma after normalisation), Exponentiated-Gamma, Kumaraswamy-Gamma, and Lindley. Models are ranked with five information criteria (AIC, BIC, AICc, Bozdogan's CAIC, HQIC) and three goodness-of-fit measures (Kolmogorov-Smirnov statistic with p-value, Cramér-von Mises, Anderson-Darling).

Fitted parameter estimates and the full comparison results for each dataset are reported in Tables 6–11 of the paper.

---

## Datasets

Three publicly available, uncensored benchmark lifetime datasets are used. No new data were collected or generated, and all three are embedded directly in the script.

- **DS1** — Bladder cancer remission times in months (n = 128). Lee, E.T. and Wang, J.W. (2003). *Statistical Methods for Survival Data Analysis*, 3rd edn. Wiley. https://doi.org/10.1002/0471458546
- **DS2** — Boeing 720 air-conditioning system failure times (n = 213). Proschan, F. (1963). Theoretical explanation of observed decreasing failure rate. *Technometrics* 5(3), 375–383. https://doi.org/10.1080/00401706.1963.10490105
- **DS3** — Malignant melanoma survival times following surgical excision (n = 205). Alizadeh, M., Cordeiro, G.M., Pinho, L.G.B. and Ghosh, I. (2017). The Gompertz-G family of distributions. *Journal of Statistical Theory and Practice* 11(1), 179–207. https://doi.org/10.1080/15598608.2016.1267668

---

## Citation

If you use this code, please cite the paper:

```
Ayeni, T.M., Odukoya, E.A. and Zhao, Y. (2026).
Quadratic functional variable transformation of the exponential-gamma
distribution for flexible lifetime modelling.
Statistical Papers, 67, 140.
https://doi.org/10.1007/s00362-026-01908-w
```

BibTeX:

```bibtex
@article{Ayeni2026QTEG,
  author  = {Ayeni, Taiwo Michael and Odukoya, Elijah Ayooluwa and Zhao, Yichuan},
  title   = {Quadratic functional variable transformation of the exponential-gamma
             distribution for flexible lifetime modelling},
  journal = {Statistical Papers},
  year    = {2026},
  volume  = {67},
  pages   = {140},
  doi     = {10.1007/s00362-026-01908-w}
}
```

---

## Acknowledgements

Yichuan Zhao is partially supported by the Simons Foundation (MPS-TSM-00638679) and the National Science Foundation (DMS-2409876).

---

## Licence

MIT License. See `LICENSE` for details.

---

## Contact

Taiwo Michael Ayeni
Department of Mathematics and Statistics, Georgia State University, Atlanta, GA 30303, USA
tayeni2@gsu.edu
