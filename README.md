# QTEG Distribution — Analysis Code

[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs00362--026--01908--w-blue)](https://doi.org/10.1007/s00362-026-01908-w)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Companion code for the published paper:

> **Quadratic functional variable transformation of the exponential-gamma distribution for flexible lifetime modelling**
> Taiwo Michael Ayeni, Elijah Ayooluwa Odukoya, Yichuan Zhao
> *Statistical Papers* (2026) **67**:140
> https://doi.org/10.1007/s00362-026-01908-w

The script reproduces all numerical tables and figures in the paper. For the derivations, proofs and discussion of results, please see the paper.

---

## The QTEG distribution

If $X \sim \mathrm{Gamma}(\alpha, \beta)$, then $Y = X^2$ follows the QTEG distribution, with density

```math
f_Y(y;\alpha,\beta) = \frac{\beta^{\alpha}}{2\,\Gamma(\alpha)}\, y^{(\alpha-2)/2}\, e^{-\beta\sqrt{y}}, \qquad y > 0,\ \alpha > 0,\ \beta > 0,
```

where $\alpha$ is the shape parameter and $\beta$ is the rate parameter.

---

## Repository contents

| File | Description |
|---|---|
| `QTEG_Full_Analysis_v3.py` | Complete Python script reproducing all tables and figures in the paper |

---

## Requirements

- Python 3.12 or later
- numpy, scipy, pandas, matplotlib

```bash
pip install numpy scipy pandas matplotlib
```

---

## Usage

```bash
python QTEG_Full_Analysis_v3.py
```

No arguments or input files are needed; the three datasets are embedded in the script. Runtime is about 15–20 minutes, most of it spent on the Monte Carlo simulation.

### What the script does

1. **Theoretical properties.** Computes moments and entropy for representative parameter values and plots the density, hazard, reversed hazard and cumulative hazard functions.
2. **Monte Carlo simulation.** Fits the MLE to simulated data under three scenarios ($\alpha, \beta$) = (1.5, 0.5), (2.0, 1.0), (3.0, 2.0), with $n = 30, 50, 100, 200$ and $N = 2000$ replications. It reports convergence rate, bias, RMSE, average standard error, average 95% Wald interval width and coverage probability for $\hat{\alpha}$ and $\hat{\beta}$.
3. **Real-data applications.** Fits QTEG and eight competing distributions (Weibull, Gamma, Log-Normal, Exponential, EGD, Exponentiated-Gamma, Kumaraswamy-Gamma, Lindley) to each dataset and compares them using AIC, BIC, AICc, CAIC, HQIC and the Kolmogorov-Smirnov, Cramér-von Mises and Anderson-Darling statistics.

Parameters are estimated by maximum likelihood with L-BFGS-B (`scipy.optimize.minimize`), using a fixed grid of starting values defined in `qteg_mle()`.

### Outputs

All outputs are written to the same folder as the script.

| Output | Contents |
|---|---|
| `QTEG_results.txt` | Moments, simulation results, parameter estimates and model-comparison tables |
| `fig1_theoretical.png` | Density, hazard, reversed hazard and cumulative hazard for the three simulation scenarios |
| `fig2_simulation.png` | Bias, RMSE and coverage probability across scenarios and sample sizes |
| `fig3_all_datasets.png` | Fitted density, empirical vs fitted CDF, and P-P plot for each dataset |
| `fig4_hazard.png` | Fitted hazard functions for each dataset |

---

## Datasets

Three publicly available benchmark lifetime datasets are embedded in the script:

| Dataset | n | Source |
|---|---|---|
| DS1 — Bladder cancer remission times (months) | 128 | Lee and Wang (2003) |
| DS2 — Boeing 720 air-conditioning system failures | 213 | Proschan (1963) |
| DS3 — Malignant melanoma survival times | 205 | Alizadeh et al. (2017) |

Full references are given in the paper.

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

## Licence

MIT License. See `LICENSE` for details.

---

## Contact

Taiwo Michael Ayeni
Department of Mathematics and Statistics, Georgia State University
tayeni2@gsu.edu
