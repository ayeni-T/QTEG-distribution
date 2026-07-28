# QTEG Distribution — Analysis Code

Companion code for the paper:

> **Quadratic Functional Variable Transformation of the Exponential-Gamma Distribution for Flexible Lifetime Modelling**
> Taiwo Michael Ayeni, Elijah Ayooluwa Odukoya, Yichuan Zhao
> *Statistical Papers* (Under review)

---

## What this repository contains

| File | Description |
|---|---|
| `QTEG_Full_Analysis_v3.py` | Complete Python script reproducing all tables and figures in the paper |

---

## The QTEG distribution

The QTEG distribution arises from the quadratic transformation Y = X² applied to a normalised exponential-gamma baseline. Its probability density function is:

```
f(y; α, β) = β^α / (2 Γ(α)) · y^((α−2)/2) · exp(−β√y),   y > 0
```

where α > 0 is the shape parameter and β > 0 is the rate parameter. The transformation produces a tail decay exp(−β√y) that is strictly heavier than the Gamma tail exp(−βy). The hazard shape is governed by α: a decreasing hazard that is unbounded at the origin for α < 2, a decreasing hazard with a finite initial value for α = 2, and a unimodal (upside-down-bathtub) hazard for α > 2 — a result established analytically in the paper via Glaser's (1980) auxiliary-function method.

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
python QTEG_Full_Analysis_v4.py
```

The script will create the following outputs in the same folder as the script:

| Output | Contents |
|---|---|
| `QTEG_results.txt` | All parameter estimates, model-comparison tables (AIC, BIC, AICc, CAIC, HQIC, KS, CvM, AD), and simulation results |
| `fig1_theoretical.png` | Probability density, hazard, reversed hazard, and cumulative hazard functions across the three simulation scenarios (2×2) |
| `fig2_simulation.png` | Simulation diagnostics: bias, RMSE, and coverage probability across scenarios (3×3) |
| `fig3_all_datasets.png` | Fitted density, empirical vs fitted CDF, and P-P plot for all three datasets (3×3) |
| `fig4_hazard.png` | Fitted hazard functions for all three datasets, side by side (1×3) |

Runtime is approximately 15–20 minutes depending on hardware (Monte Carlo simulation uses N = 2000 replications across 3 scenarios × 4 sample sizes).

---

## Simulation study details

The simulation study uses the following design, matching the paper exactly:

| Scenario | True α | True β | Hazard behaviour |
|---|---|---|---|
| Sc. 1 | 1.5 | 0.5 | Decreasing, unbounded initial hazard |
| Sc. 2 | 2.0 | 1.0 | Decreasing, finite initial hazard |
| Sc. 3 | 3.0 | 2.0 | Unimodal, zero initial hazard |

Sample sizes: n = 30, 50, 100, 200. Replications: N = 2000.

For each replication the script computes: Bias, RMSE, AvgSE (mean Hessian-based standard error), **CIW** (average 95% Wald interval width = 2 × 1.96 × AvgSE), and empirical coverage probability (CP). All metrics are reported separately for α̂ and β̂.

The MLE starting value grid `α₀ ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0}` is set **a priori** in the `qteg_mle()` function, before any simulation results are observed. The 100% convergence rate observed across all simulation scenarios provides post-hoc evidence that this grid was sufficient for the parameter settings examined.

---

## Model comparison

For each of the three benchmark datasets, QTEG is compared against eight competing distributions (Weibull, Gamma, Log-Normal, Exponential, EGD, Exponential-Gamma, Kumaraswamy-Gamma, Lindley) using five information criteria — AIC, BIC, AICc (small-sample corrected AIC), CAIC (Bozdogan's Consistent AIC), and HQIC — together with four goodness-of-fit measures: the Kolmogorov-Smirnov statistic and p-value, the Cramér-von Mises statistic, and the Anderson-Darling statistic.

---

## Datasets

Three publicly available benchmark lifetime datasets are used:

- **DS1** — Bladder cancer remission times (n = 128): Lee and Wang (2003)
- **DS2** — Boeing 720 air-conditioning system failures (n = 213): Proschan (1963)
- **DS3** — Malignant melanoma survival times (n = 205): Alizadeh et al. (2017)

No new data were collected or generated for this study. All datasets are embedded directly in the script.

---

## Citation

If you use this code, please cite the paper:

```
Ayeni, T.M., Odukoya, E.A. and Zhao, Y. (2026).
Quadratic Functional Variable Transformation of the Exponential-Gamma
Distribution for Flexible Lifetime Modelling.
Statistical Papers. [Under Review]
```

---

## Licence

MIT License. See `LICENSE` for details.

---

## Contact

Taiwo Michael Ayeni
Department of Mathematics and Statistics, Georgia State University
tayeni2@gsu.edu
