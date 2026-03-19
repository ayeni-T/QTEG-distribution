# QTEG Distribution — Analysis Code

Companion code for the paper:

> **Quadratic Functional Variable Transformation of the Exponential-Gamma Distribution for Flexible Lifetime Modelling**  
> Taiwo Michael Ayeni, Elijah Ayooluwa Odukoya, Yichuan Zhao  
> *Statistical Papers* (Preparing for submission)

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

where α > 0 is the shape parameter and β > 0 is the rate parameter. The transformation produces a tail decay exp(−β√y) that is strictly heavier than the Gamma tail exp(−βy), enabling flexible hazard shapes — decreasing, approximately constant, or unimodal — within a two-parameter family.

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

The script will create the following outputs in the same folder as the script:

| Output | Contents |
|---|---|
| `QTEG_results.txt` | All parameter estimates, goodness-of-fit tables, and simulation results (including CIW column) |
| `fig1_theoretical.png` | PDF, CDF, survival function, and hazard function across parameter scenarios (2×2) |
| `fig2_simulation.png` | Simulation diagnostics: \|Bias\|, RMSE, and coverage probability across scenarios (3×3) |
| `fig3_all_datasets.png` | Fitted density, empirical vs fitted CDF, and P-P plot for all three datasets (3×3) |
| `fig4_hazard.png` | Standalone fitted hazard functions for all three datasets (3×1) |

Runtime is approximately 5–10 minutes depending on hardware (Monte Carlo simulation uses N = 500 replications across 3 scenarios × 4 sample sizes).

---

## Simulation study details

The simulation study uses the following design, matching the paper exactly:

| Scenario | True α | True β | Hazard regime |
|---|---|---|---|
| Sc. 1 | 1.5 | 0.5 | Decreasing |
| Sc. 2 | 2.0 | 1.0 | Approximately constant |
| Sc. 3 | 3.0 | 2.0 | Unimodal |

Sample sizes: n = 30, 50, 100, 200. Replications: N = 500.

For each replication the script computes: Bias, RMSE, AvgSE (mean Hessian-based standard error), **CIW** (average 95% Wald interval width = 2 × 1.96 × AvgSE), and empirical coverage probability (CP). All metrics are reported separately for α̂ and β̂.

The MLE starting value grid `α₀ ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0}` is set **a priori** in the `qteg_mle()` function, before any simulation results are observed. The 100% convergence rate across all scenarios provides post-hoc validation that this grid is sufficient.

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
Statistical Papers. [Preparing for submission]
```

---

## Licence

MIT License. See `LICENSE` for details.

---

## Contact

Taiwo Michael Ayeni  
Department of Mathematics and Statistics, Georgia State University  
tayeni2@gsu.edu
