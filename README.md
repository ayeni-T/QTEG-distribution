# QTEG Distribution — Analysis Code

Companion code for the paper:

> **Quadratic Functional Variable Transformation of the Exponential-Gamma Distribution for Flexible Lifetime Modelling**  
> Taiwo Michael Ayeni, Elijah Ayooluwa Odukoya, Yichuan Zhao  
> *Statistical Papers* (submitted)

---

## What this repository contains

| File | Description |
|---|---|
| `QTEG_Full_Analysis_v2.py` | Complete Python script reproducing all tables and figures in the paper |

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
python QTEG_Full_Analysis_v2.py
```

The script will create the following outputs in the same folder:

| Output | Contents |
|---|---|
| `QTEG_results.txt` | All parameter estimates, goodness-of-fit tables, and simulation results |
| `fig1_theoretical.png` | PDF, CDF, survival, and hazard functions across parameter scenarios |
| `fig2_simulation.png` | Simulation diagnostics: bias, RMSE, and coverage probability |
| `fig3_all_datasets.png` | Fitted density, CDF, P-P plot, and hazard for all three datasets |

Runtime is approximately 5–10 minutes depending on hardware (Monte Carlo simulation uses N = 500 replications).

---

## Datasets

Three publicly available benchmark lifetime datasets are used:

- **DS1** — Bladder cancer remission times (n = 128): Lee and Wang (2003)
- **DS2** — Boeing 720 air-conditioning failures (n = 213): Proschan (1963)  
- **DS3** — Malignant melanoma survival times (n = 205): Alizadeh et al. (2017)

No new data were collected or generated for this study.

---

## Citation

If you use this code, please cite the paper:

```
Ayeni, T.M., Odukoya, E.A. and Zhao, Y. (2026).
Quadratic Functional Variable Transformation of the Exponential-Gamma
Distribution for Flexible Lifetime Modelling.
Statistical Papers. [in review]
```

---

## Licence

MIT License. See `LICENSE` for details.

---

## Contact

Taiwo Michael Ayeni  
Department of Mathematics and Statistics, Georgia State University  
tayeni2.gsu.edu
