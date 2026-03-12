"""
================================================================================
QTEG Distribution -- Complete Analysis Script
================================================================================
Authors : Taiwo Michael Ayeni et al.
Purpose : Full reproduction of all tables and figures for the paper

CORRECT PDF:
    f_Y(y) = r^k / (2 * Gamma(k)) * y^((k-2)/2) * exp(-r * sqrt(y)),  y > 0

HOW TO RUN:
    1. Install dependencies (once):
       pip install numpy scipy pandas matplotlib

    2. Run this script:
       python QTEG_Full_Analysis.py

    3. Outputs created in the same folder:
       - QTEG_results.txt          (all tables, ready to paste into paper)
       - fig1_theoretical.png      (PDF / CDF / Survival / Hazard)
       - fig2_simulation_rmse.png  (RMSE convergence)
       - fig3_simulation_cp.png    (Coverage probability)
       - fig4_bias.png             (Absolute bias)
       - fig5_ds1_bladder.png
       - fig7_ds3_boeing.png
       - fig3_ds1_ds3_melanoma.png
================================================================================
"""

import numpy as np
import pandas as pd
from scipy.special import (gamma as G, gammaln, digamma,
                            gammainc, gammaincc, gammaincinv)
from scipy.optimize import minimize
from scipy.stats import (gamma as gamma_dist, weibull_min,
                         lognorm, expon, skew, kurtosis)
import matplotlib
matplotlib.use('Agg')          # no display needed; remove if you want pop-up windows
import matplotlib.pyplot as plt
import warnings, re, os, sys
warnings.filterwarnings('ignore')

np.random.seed(2024)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))   # same folder as this script
LOG_FILE   = os.path.join(OUTPUT_DIR, 'QTEG_results.txt')

# redirect print to both screen and file
# open with utf-8 encoding to handle all Unicode characters on Windows
class Tee:
    """Write to multiple files simultaneously.
    The log file (utf-8) gets all characters including Greek.
    The Windows terminal gets a safe ASCII fallback for any unencodable char.
    """
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except (UnicodeEncodeError, UnicodeDecodeError):
                # Terminal on Windows can't show Greek -- write safe version
                safe = obj.encode('ascii', 'replace').decode('ascii')
                try:
                    f.write(safe)
                    f.flush()
                except Exception:
                    pass
    def flush(self):
        for f in self.files:
            try: f.flush()
            except Exception: pass

log = open(LOG_FILE, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, log)

print("""
================================================================================
QTEG DISTRIBUTION -- COMPLETE COMPUTATIONAL RESULTS
================================================================================
""")

# ============================================================
# 1. CORE QTEG FUNCTIONS  (corrected PDF throughout)
# ============================================================

def qteg_pdf(y, alpha, beta):
    """Correct QTEG PDF: r^k/(2*Gamma(k)) * y^((k-2)/2) * exp(-r*sqrt(y))"""
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    m = y > 0
    out[m] = (beta**alpha / (2.0 * G(alpha))) * y[m]**((alpha - 2) / 2.0) * np.exp(-beta * np.sqrt(y[m]))
    return out

def qteg_logpdf(y, alpha, beta):
    """Log of correct QTEG PDF (vectorised, y must be > 0)"""
    return (alpha * np.log(beta) - np.log(2.0) - gammaln(alpha)
            + ((alpha - 2) / 2.0) * np.log(y) - beta * np.sqrt(y))

def qteg_cdf(y, alpha, beta):
    """F(y) = gamma(k, r*sqrt(y)) / Gamma(k)  [regularised lower incomplete gamma]"""
    return gammainc(alpha, beta * np.sqrt(np.maximum(y, 0.0)))

def qteg_sf(y, alpha, beta):
    """Survival = 1 - CDF"""
    return gammaincc(alpha, beta * np.sqrt(np.maximum(y, 1e-300)))

def qteg_hazard(y, alpha, beta):
    """Instantaneous hazard h(y) = f(y) / S(y)"""
    y  = np.atleast_1d(np.asarray(y, float))
    sf = qteg_sf(y, alpha, beta)
    return qteg_pdf(y, alpha, beta) / np.maximum(sf, 1e-300)

def qteg_quantile(p, alpha, beta):
    """Quantile Q(p) = (gammaincinv(k, p) / r)^2"""
    return (gammaincinv(alpha, p) / beta) ** 2

def qteg_moments(alpha, beta):
    """Return dict of key moments (all exact closed-form)"""
    mean  = alpha * (alpha + 1) / beta**2
    EY2   = G(alpha + 4) / (beta**4 * G(alpha))
    var   = EY2 - mean**2
    EY3   = G(alpha + 6) / (beta**6 * G(alpha))
    EY4   = G(alpha + 8) / (beta**8 * G(alpha))
    mu3   = EY3 - 3*mean*EY2 + 2*mean**3
    mu4   = EY4 - 4*mean*EY3 + 6*mean**2*EY2 - 3*mean**4
    skew_ = mu3 / var**1.5
    kurt_ = mu4 / var**2 - 3.0
    entropy = np.log(2*G(alpha)) + alpha - 2*np.log(beta) - (alpha - 2)*digamma(alpha)
    return dict(mean=mean, variance=var, std=np.sqrt(var),
                cv=np.sqrt(var)/mean, skewness=skew_, ex_kurtosis=kurt_,
                entropy=entropy)

# ============================================================
# 2. MLE WITH SEMI-CLOSED STARTING VALUE
# ============================================================

def qteg_mle(y, n_starts=8):
    """
    MLE for QTEG(alpha, beta).
    Starting value: beta0 = alpha0 / mean(sqrt(y))  [semi-closed from score eq. for beta].
    Returns dict with estimates, SEs, and information criteria.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)

    def neg_ll(params):
        alpha, beta = params
        if alpha <= 0.0 or beta <= 0.0:
            return np.inf
        return -np.sum(qteg_logpdf(y, alpha, beta))

    alpha_starts = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0]
    best_res = None
    best_val = np.inf

    for alpha0 in alpha_starts:
        beta0 = alpha0 / np.mean(np.sqrt(y))  # semi-closed starting value
        try:
            res = minimize(neg_ll, [alpha0, beta0],
                           method='L-BFGS-B',
                           bounds=[(1e-5, None), (1e-5, None)],
                           options={'ftol': 1e-13, 'gtol': 1e-9, 'maxiter': 5000})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_res = res
        except Exception:
            pass

    if best_res is None:
        return None

    alpha_hat, beta_hat = best_res.x
    logL = -best_val

    # Standard errors via finite-difference Hessian
    h_step = 1e-5
    try:
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                ei = np.zeros(2); ei[i] = h_step
                ej = np.zeros(2); ej[j] = h_step
                H[i, j] = (neg_ll(best_res.x + ei + ej)
                          - neg_ll(best_res.x + ei - ej)
                          - neg_ll(best_res.x - ei + ej)
                          + neg_ll(best_res.x - ei - ej)) / (4 * h_step**2)
        cov  = np.linalg.inv(H)
        se_alpha = float(np.sqrt(max(cov[0, 0], 0.0)))
        se_beta  = float(np.sqrt(max(cov[1, 1], 0.0)))
    except Exception:
        se_alpha = se_beta = float('nan')

    aic  = 4.0 - 2.0 * logL
    bic  = 2.0 * np.log(n) - 2.0 * logL
    caic = aic + 2.0 * 2.0 * 3.0 / max(n - 3, 1)   # 2k(k+1)/(n-k-1) with k=2
    hqic = -2.0 * logL + 4.0 * np.log(np.log(n))

    return dict(alpha=alpha_hat, beta=beta_hat, se_alpha=se_alpha, se_beta=se_beta,
                logL=logL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                n=n, converged=True)

# ============================================================
# 3. COMPETING MODEL FITTERS
# ============================================================

def _ic(logL, p, n):
    """Return (AIC, BIC, CAIC, HQIC)"""
    aic  = 2*p - 2*logL
    bic  = p*np.log(n) - 2*logL
    caic = aic + 2*p*(p+1) / max(n - p - 1, 1)
    hqic = -2*logL + 2*p*np.log(np.log(n))
    return aic, bic, caic, hqic

def fit_weibull(y):
    c, _, s = weibull_min.fit(y, floc=0)
    lL = float(np.sum(weibull_min.logpdf(y, c, scale=s)))
    aic, bic, caic, hqic = _ic(lL, 2, len(y))
    A  = '\u03b1'; B  = '\u03b2'
    return dict(label='Weibull', params=f'{A}={c:.4f}, {B}={s:.4f}',
                npar=2, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=lambda yy, c=c, s=s: weibull_min.cdf(yy, c, scale=s),
                pdf=lambda yy, c=c, s=s: weibull_min.pdf(yy, c, scale=s),
                haz=lambda yy, c=c, s=s: (weibull_min.pdf(yy,c,scale=s)
                                           /np.maximum(weibull_min.sf(yy,c,scale=s),1e-300)))

def fit_gamma(y):
    a, _, sc = gamma_dist.fit(y, floc=0)
    lL = float(np.sum(gamma_dist.logpdf(y, a, scale=sc)))
    aic, bic, caic, hqic = _ic(lL, 2, len(y))
    A  = '\u03b1'; B  = '\u03b2'
    return dict(label='Gamma', params=f'{A}={a:.4f}, {B}={1/sc:.4f}',
                npar=2, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=lambda yy, a=a, sc=sc: gamma_dist.cdf(yy, a, scale=sc),
                pdf=lambda yy, a=a, sc=sc: gamma_dist.pdf(yy, a, scale=sc),
                haz=lambda yy, a=a, sc=sc: (gamma_dist.pdf(yy,a,scale=sc)
                                             /np.maximum(gamma_dist.sf(yy,a,scale=sc),1e-300)))

def fit_lognormal(y):
    sig, _, scl = lognorm.fit(y, floc=0)
    lL = float(np.sum(lognorm.logpdf(y, sig, scale=scl)))
    aic, bic, caic, hqic = _ic(lL, 2, len(y))
    MU = '\u03bc'; SG = '\u03c3'
    return dict(label='Log-Normal', params=f'{MU}={np.log(scl):.4f}, {SG}={sig:.4f}',
                npar=2, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=lambda yy, sig=sig, scl=scl: lognorm.cdf(yy, sig, scale=scl),
                pdf=lambda yy, sig=sig, scl=scl: lognorm.pdf(yy, sig, scale=scl),
                haz=lambda yy, sig=sig, scl=scl: (lognorm.pdf(yy,sig,scale=scl)
                                                    /np.maximum(lognorm.sf(yy,sig,scale=scl),1e-300)))

def fit_exponential(y):
    rate = 1.0 / np.mean(y)
    lL   = float(np.sum(expon.logpdf(y, scale=1/rate)))
    aic, bic, caic, hqic = _ic(lL, 1, len(y))
    LA = '\u03bb'
    return dict(label='Exponential', params=f'{LA}={rate:.6f}',
                npar=1, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=lambda yy, rate=rate: expon.cdf(yy, scale=1/rate),
                pdf=lambda yy, rate=rate: expon.pdf(yy, scale=1/rate),
                haz=lambda yy, rate=rate: np.full_like(np.asarray(yy,float), rate))

def fit_exp_gamma(y):
    """Normalized Exponential-Gamma baseline = Gamma(k, r) -- same as Gamma fit,
    labelled separately to match the paper's Table structure."""
    a, _, sc = gamma_dist.fit(y, floc=0)
    lL = float(np.sum(gamma_dist.logpdf(y, a, scale=sc)))
    aic, bic, caic, hqic = _ic(lL, 2, len(y))
    A  = '\u03b1'; B  = '\u03b2'
    return dict(label='EGD (Baseline)', params=f'{A}={a:.4f}, {B}={1/sc:.4f}',
                npar=2, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=lambda yy, a=a, sc=sc: gamma_dist.cdf(yy, a, scale=sc),
                pdf=lambda yy, a=a, sc=sc: gamma_dist.pdf(yy, a, scale=sc))

def fit_exponentiated_gamma(y):
    """Exponentiated-Gamma: F(y) = [G(y; a, b)]^lambda -- 3 parameters"""
    def neg_ll(params):
        a, b, lam = params
        if a <= 0 or b <= 0 or lam <= 0: return np.inf
        Fv = np.clip(gamma_dist.cdf(y, a, scale=b), 1e-300, 1 - 1e-300)
        fv = np.maximum(gamma_dist.pdf(y, a, scale=b), 1e-300)
        return -float(np.sum(np.log(lam) + (lam - 1)*np.log(Fv) + np.log(fv)))

    best = None
    for a0, b0, l0 in [(1.5, np.mean(y)/1.5, 1.0),
                        (2.0, np.mean(y)/2.0, 0.5),
                        (0.8, np.mean(y)/0.8, 2.0),
                        (3.0, np.mean(y)/3.0, 1.5)]:
        try:
            res = minimize(neg_ll, [a0, b0, l0], method='L-BFGS-B',
                           bounds=[(1e-5,None)]*3,
                           options={'ftol':1e-13,'maxiter':5000})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass

    if best is None: return None
    a_, b_, lam_ = best.x
    lL = -best.fun
    aic, bic, caic, hqic = _ic(lL, 3, len(y))

    def eg_cdf(yy, a=a_, b=b_, lam=lam_):
        return np.clip(gamma_dist.cdf(yy, a, scale=b), 0, 1)**lam

    A  = '\u03b1'; B  = '\u03b2'; LA = '\u03bb'
    return dict(label='Exp-Gamma', params=f'{A}={a_:.4f}, {B}={b_:.4f}, {LA}={lam_:.4f}',
                npar=3, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=eg_cdf)

def fit_kumaraswamy_gamma(y):
    """Kumaraswamy-Gamma: F(y) = 1 - [1 - G(y)^a]^b -- 4 parameters"""
    def neg_ll(params):
        ak, bk, sh, sc = params
        if any(p <= 0 for p in params): return np.inf
        Gv = np.clip(gamma_dist.cdf(y, sh, scale=sc), 1e-300, 1 - 1e-300)
        gv = np.maximum(gamma_dist.pdf(y, sh, scale=sc), 1e-300)
        ll = np.sum(np.log(ak)+np.log(bk)+(ak-1)*np.log(Gv)
                    +(bk-1)*np.log(1-Gv**ak)+np.log(gv))
        return -float(ll)

    best = None
    for a0, b0, sh0, sc0 in [(1.5, 1.5, 2.0, np.mean(y)/2.0),
                               (2.0, 0.8, 1.5, np.mean(y)/1.5),
                               (0.8, 2.0, 3.0, np.mean(y)/3.0),
                               (1.0, 1.0, 1.0, np.mean(y))]:
        try:
            res = minimize(neg_ll, [a0, b0, sh0, sc0], method='L-BFGS-B',
                           bounds=[(1e-5,None)]*4,
                           options={'ftol':1e-13,'maxiter':5000})
            if res.success and (best is None or res.fun < best.fun):
                best = res
        except Exception:
            pass

    if best is None: return None
    ak_, bk_, sh_, sc_ = best.x
    lL = -best.fun
    aic, bic, caic, hqic = _ic(lL, 4, len(y))

    def kg_cdf(yy, ak=ak_, bk=bk_, sh=sh_, sc=sc_):
        Gv = np.clip(gamma_dist.cdf(yy, sh, scale=sc), 0, 1)
        return 1.0 - (1.0 - Gv**ak)**bk

    A  = '\u03b1'; B  = '\u03b2'
    return dict(label='Kum-Gamma',
                params=f'a={ak_:.4f}, b={bk_:.4f}, {A}={sh_:.4f}, {B}={sc_:.4f}',
                npar=4, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=kg_cdf)


def fit_lindley(y):
    """Lindley(theta): f(y) = theta^2/(1+theta) * (1+y) * exp(-theta*y), y>0"""
    from scipy.optimize import minimize_scalar
    def neg_ll(theta):
        if theta <= 0: return np.inf
        return -float(np.sum(2*np.log(theta) - np.log(1+theta)
                             + np.log(1+y) - theta*y))
    res = minimize_scalar(neg_ll, bounds=(1e-6, 100), method='bounded')
    theta = res.x
    lL = -res.fun
    n, p = len(y), 1
    aic, bic, caic, hqic = _ic(lL, p, n)
    def lindley_cdf(yy, th=theta):
        yy = np.asarray(yy, float)
        return 1.0 - (1.0 + th*yy/(1.0+th)) * np.exp(-th*yy)
    def lindley_pdf(yy, th=theta):
        yy = np.asarray(yy, float)
        return th**2/(1+th) * (1+yy) * np.exp(-th*yy)
    def lindley_haz(yy, th=theta):
        p = lindley_pdf(yy, th)
        s = np.maximum(1 - lindley_cdf(yy, th), 1e-300)
        return p / s
    TH = '\u03b8'
    return dict(label='Lindley', params=f'{TH}={theta:.4f}',
                npar=p, logL=lL, aic=aic, bic=bic, caic=caic, hqic=hqic,
                cdf=lindley_cdf, pdf=lindley_pdf, haz=lindley_haz)

# ============================================================
# 4. GOODNESS-OF-FIT TESTS
# ============================================================

def gof_tests(y, cdf_fn):
    """
    KS, Cramer-von Mises (CvM), Anderson-Darling (AD) statistics.
    KS p-value uses the two-sided Kolmogorov approximation.
    """
    ys = np.sort(y)
    n  = len(ys)
    Fn = np.clip(cdf_fn(ys), 1e-15, 1 - 1e-15)
    i  = np.arange(1, n + 1)

    # KS
    D_plus  = np.max(i / n - Fn)
    D_minus = np.max(Fn - (i - 1) / n)
    ks_stat = max(D_plus, D_minus)
    # Kolmogorov two-sided p-value approximation
    z      = (np.sqrt(n) + 0.12 + 0.11/np.sqrt(n)) * ks_stat
    ks_p   = float(np.clip(2 * sum((-1)**(j-1) * np.exp(-2*j**2*z**2)
                                    for j in range(1, 101)), 0, 1))

    # CvM: W^2 = sum[(F(x_i) - (2i-1)/(2n))^2] + 1/(12n)
    cvm_stat = float(np.sum((Fn - (2*i - 1) / (2*n))**2) + 1.0 / (12*n))

    # AD: A^2 = -n - (1/n) * sum[(2i-1)(log F(x_i) + log(1-F(x_{n+1-i})))]
    ad_stat = float(-n - np.sum((2*i - 1) / n
                                 * (np.log(Fn) + np.log(1.0 - Fn[::-1]))))

    return dict(ks=ks_stat, ks_p=ks_p, cvm=cvm_stat, ad=ad_stat)

# ============================================================
# 5. DATASETS
# ============================================================

def _parse(raw):
    return np.array([float(x) for x in re.findall(r'\d+\.?\d*', raw)])

DATA = {
    'DS1: Bladder Cancer Remission (n=128)': _parse(
        "0.08,2.09,3.48,4.87,6.94,8.66,13.11,23.63,0.20,2.23,3.52,4.98,"
        "6.97,9.02,13.29,0.40,2.26,3.57,5.06,7.09,9.22,13.80,25.74,0.50,"
        "2.46,3.64,5.09,7.26,9.47,14.24,25.82,0.51,2.54,3.70,5.17,7.28,"
        "9.74,14.76,26.31,0.81,2.62,3.82,5.32,7.32,10.06,14.77,32.15,2.64,"
        "11.79,18.10,1.46,4.40,5.85,8.26,11.98,19.13,1.76,3.25,4.50,6.25,"
        "8.37,12.02,2.02,3.31,4.51,6.54,8.53,12.03,20.28,2.02,3.36,6.76,"
        "12.07,21.73,2.0,3.36,6.93,8.65,12.63,22.69,3.88,5.32,7.39,10.34,"
        "14.83,34.26,0.90,2.69,4.18,5.34,7.59,10.66,15.96,36.66,1.05,2.69,"
        "4.23,5.41,7.62,10.75,16.62,43.01,1.19,2.75,4.26,5.41,7.63,17.12,"
        "46.12,1.26,2.83,4.33,5.49,7.66,11.25,17.14,79.05,1.35,2.87,5.62,"
        "7.87,11.64,17.36,1.40,3.02,4.34,5.71,7.93"),


    'DS2: Boeing 720 Failures (n=213)': _parse(
        "194,413,90,74,55,23,97,50,359,50,130,487,102,15,14,10,57,320,261,"
        "51,44,9,254,493,18,209,41,58,60,48,56,87,11,102,12,5,100,14,29,"
        "37,186,29,104,7,4,72,270,283,7,57,33,100,61,502,220,120,141,22,"
        "603,35,98,54,181,65,49,12,239,14,18,39,3,12,5,32,9,14,70,47,62,"
        "142,3,104,85,67,169,24,21,246,47,68,15,2,91,59,447,56,29,176,225,"
        "77,197,438,43,134,184,20,386,182,71,80,188,230,152,36,79,59,33,"
        "246,1,79,3,27,201,84,27,21,16,88,130,14,118,44,15,42,106,46,230,"
        "59,153,104,20,206,5,66,34,29,26,35,5,82,5,61,31,118,326,12,54,"
        "36,34,18,25,120,31,22,18,156,11,216,139,67,310,3,46,210,57,76,"
        "14,111,97,62,26,71,39,30,7,44,11,63,23,22,23,14,18,13,34,62,11,"
        "191,14,16,18,130,90,163,208,1,24,70,16,101,52,208,95"),

    'DS3: Malignant Melanoma (n=205)': _parse(
        "6.76,0.65,1.34,2.90,12.08,4.84,5.16,3.22,12.88,7.41,4.19,0.16,3.87,"
        "4.84,2.42,12.56,5.80,7.06,5.48,7.73,13.85,2.34,4.19,4.04,4.84,0.32,"
        "8.54,2.58,3.56,3.54,0.97,4.83,1.62,6.44,14.66,2.58,3.87,3.54,1.34,"
        "2.24,3.87,3.54,17.42,1.29,3.22,1.29,4.51,8.38,1.94,0.16,2.58,1.29,"
        "0.16,1.62,1.29,2.10,0.32,0.81,1.13,5.16,1.62,1.37,0.24,0.81,1.29,"
        "1.29,0.97,1.13,5.80,1.29,0.48,1.62,2.26,0.58,0.97,2.58,0.81,3.54,"
        "0.97,1.78,1.94,1.29,3.22,1.53,1.29,1.62,1.62,0.32,4.84,1.29,0.97,"
        "3.06,3.54,1.62,2.58,1.94,0.81,7.73,0.97,12.88,2.58,4.09,0.64,0.97,"
        "3.22,1.62,3.87,0.32,0.32,3.22,2.26,3.06,2.58,0.65,1.13,0.81,0.97,"
        "1.76,1.94,0.65,0.97,5.64,9.66,0.10,5.48,2.26,4.83,0.97,0.97,5.16,"
        "0.81,2.90,3.87,1.94,0.16,0.64,2.26,1.45,4.82,1.29,7.89,0.81,3.54,"
        "1.29,0.64,3.22,1.45,0.48,1.94,0.16,0.16,1.29,1.94,3.54,0.81,0.65,"
        "7.09,0.16,1.62,1.62,1.29,6.12,0.48,0.64,3.22,1.94,2.58,2.58,0.81,"
        "0.81,3.22,0.32,3.22,2.74,4.84,1.62,0.65,1.45,0.65,1.29,1.62,3.54,"
        "3.22,0.65,1.03,7.09,1.29,0.65,1.78,12.24,8.06,0.81,2.10,3.87,0.65,"
        "1.94,0.65,2.10,1.94,1.13,7.06,6.12,0.48,2.26,2.90"),
}

DATA_XLABEL = {
    'DS1: Bladder Cancer Remission (n=128)': 'Remission Time (months)',
    'DS2: Boeing 720 Failures (n=213)':      'Time Between Failures',
    'DS3: Malignant Melanoma (n=205)':       'Survival Time (years)',
}

# ============================================================
# 6.  DESCRIPTIVE STATISTICS TABLE
# ============================================================

print("="*80)
print("TABLE 1 -- DESCRIPTIVE STATISTICS")
print("="*80)
header = f"{'Statistic':<18}" + "".join(f"{k.split('(')[0].strip():>16}" for k in DATA)
print(header)
print("-"*80)

stat_fns = [
    ('n',            lambda x: len(x)),
    ('Min',          np.min),
    ('Max',          np.max),
    ('Mean',         np.mean),
    ('Median',       np.median),
    ('Std Dev',      lambda x: np.std(x, ddof=1)),
    ('CV',           lambda x: np.std(x, ddof=1) / np.mean(x)),
    ('Skewness',     skew),
    ('Ex. Kurtosis', kurtosis),
]

for sname, fn in stat_fns:
    vals = [fn(y) for y in DATA.values()]
    if sname == 'n':
        row = f"{'n':<18}" + "".join(f"{int(v):>16}" for v in vals)
    else:
        row = f"{sname:<18}" + "".join(f"{v:>16.4f}" for v in vals)
    print(row)

# ============================================================
# 7.  SIMULATION STUDY
# ============================================================

print("\n\n" + "="*80)
print("TABLE 6 -- MONTE CARLO SIMULATION STUDY  (Nsim=500, BFGS)")
print("="*80)

SCENARIOS = [
    (1.5, 0.5,  'Sc.1 (\u03b1=1.5, \u03b2=0.5)',  'Decreasing hazard'),
    (2.0, 1.0,  'Sc.2 (\u03b1=2.0, \u03b2=1.0)',  'Bathtub hazard'),
    (3.0, 2.0,  'Sc.3 (\u03b1=3.0, \u03b2=2.0)',  'Increasing then flat'),
]
SAMPLE_SIZES = [30, 50, 100, 200]
NSIM = 500

sim_rows = []

for alpha_true, beta_true, sc_label, _ in SCENARIOS:
    print(f"\n  {sc_label}  (true α={alpha_true}, β={beta_true})")
    print(f"  {'n':>5}  {'Conv%':>6}  {'Bias(\u03b1)':>9}  {'RMSE(\u03b1)':>9}  "
          f"{'AvgSE(\u03b1)':>9}  {'CP(\u03b1)%':>7}  {'Bias(\u03b2)':>9}  {'RMSE(\u03b2)':>9}  "
          f"{'AvgSE(\u03b2)':>9}  {'CP(\u03b2)%':>7}")
    print("  " + "-"*85)

    for n in SAMPLE_SIZES:
        alpha_hats, beta_hats, se_alphas, se_betas = [], [], [], []
        cover_k = cover_r = converged = 0

        for _ in range(NSIM):
            # Generate QTEG sample: Y = X^2, X ~ Gamma(alpha_true, beta_true)
            X = gamma_dist.rvs(alpha_true, scale=1.0 / beta_true, size=n)
            Y = X ** 2

            res = qteg_mle(Y)
            if res is None:
                continue
            converged += 1
            alpha_hats.append(res['alpha'])
            beta_hats.append(res['beta'])
            se_alphas.append(res['se_alpha'])
            se_betas.append(res['se_beta'])

            if not np.isnan(res['se_alpha']):
                lo = res['alpha'] - 1.96 * res['se_alpha']
                hi = res['alpha'] + 1.96 * res['se_alpha']
                if lo <= alpha_true <= hi:
                    cover_k += 1
            if not np.isnan(res['se_beta']):
                lo = res['beta'] - 1.96 * res['se_beta']
                hi = res['beta'] + 1.96 * res['se_beta']
                if lo <= beta_true <= hi:
                    cover_r += 1

        alpha_arr = np.array(alpha_hats)
        beta_arr  = np.array(beta_hats)
        valid = len(alpha_arr)

        bias_k  = float(np.mean(alpha_arr) - alpha_true)
        rmse_alpha  = float(np.sqrt(np.mean((alpha_arr - alpha_true)**2)))
        avgse_alpha = float(np.nanmean(se_alphas))
        cp_k    = 100.0 * cover_k / valid if valid > 0 else float('nan')

        bias_r  = float(np.mean(beta_arr) - beta_true)
        rmse_beta  = float(np.sqrt(np.mean((beta_arr - beta_true)**2)))
        avgse_beta = float(np.nanmean(se_betas))
        cp_r    = 100.0 * cover_r / valid if valid > 0 else float('nan')

        print(f"  {n:>5}  {100*converged/NSIM:>6.1f}"
              f"  {bias_k:>+9.4f}  {rmse_alpha:>9.4f}  {avgse_alpha:>9.4f}  {cp_k:>7.1f}"
              f"  {bias_r:>+9.4f}  {rmse_beta:>9.4f}  {avgse_beta:>9.4f}  {cp_r:>7.1f}")

        sim_rows.append(dict(scenario=sc_label, n=n,
                             conv=100*converged/NSIM,
                             bias_alpha=bias_k, rmse_alpha=rmse_alpha, avgse_alpha=avgse_alpha, cp_alpha=cp_k,
                             bias_beta=bias_r,  rmse_beta=rmse_beta,  avgse_beta=avgse_beta,  cp_beta=cp_r))

sim_df = pd.DataFrame(sim_rows)

# ============================================================
# 8.  THEORETICAL MOMENTS (for Section 2.5 of paper)
# ============================================================

print("\n\n" + "="*80)
print("TABLE 7 -- THEORETICAL MOMENTS FOR SELECTED PARAMETER VALUES")
print("="*80)
print(f"  {'α':>6}  {'β':>6}  {'Mean':>10}  {'Std':>10}  {'CV':>8}  "
      f"{'Skewness':>10}  {'Ex.Kurt':>10}  {'Entropy':>10}")
print("  " + "-"*70)
for alpha, beta in [(1.0,1.0),(1.5,1.0),(2.0,1.0),(3.0,1.0),(2.0,2.0),(4.0,1.5),(0.8,1.0)]:
    m = qteg_moments(alpha, beta)
    print(f"  {alpha:>4.1f}  {beta:>4.1f}  {m['mean']:>10.4f}  {m['std']:>10.4f}  "
          f"{m['cv']:>8.4f}  {m['skewness']:>10.4f}  {m['ex_kurtosis']:>10.4f}  "
          f"{m['entropy']:>10.4f}")

print("\n  Note: mean = α(α+1)/β², entropy = log(2Gamma(k)) + k - 2log(r) - (k-2)psi(k)")

# ============================================================
# 9.  MODEL FITTING -- REAL DATA -- ALL DATASETS
# ============================================================

print("\n\n" + "="*80)
print("TABLES 2-5 -- MODEL COMPARISON (ALL DATASETS)")
print("="*80)

ALL_FIT = {}   # store fits for figure generation

for ds_name, y in DATA.items():
    print(f"\n{'-'*80}")
    print(f"  {ds_name}")
    print(f"{'-'*80}")

    fits = {}

    # QTEG
    qr = qteg_mle(y)
    if qr:
        g = gof_tests(y, lambda yy: qteg_cdf(yy, qr['alpha'], qr['beta']))
        fits['QTEG'] = {**qr, **g,
                        'label': 'QTEG', 'npar': 2,
                        'params': (f"\u03b1={qr['alpha']:.4f} (SE={qr['se_alpha']:.4f}), "
                                   f"\u03b2={qr['beta']:.4f} (SE={qr['se_beta']:.4f})")}

    # Weibull
    w = fit_weibull(y)
    if w:
        g = gof_tests(y, w['cdf']); fits['Weibull'] = {**w, **g}

    # Gamma
    ga = fit_gamma(y)
    if ga:
        g = gof_tests(y, ga['cdf']); fits['Gamma'] = {**ga, **g}

    # Log-Normal
    ln = fit_lognormal(y)
    if ln:
        g = gof_tests(y, ln['cdf']); fits['LogNorm'] = {**ln, **g}

    # Exponential
    ex = fit_exponential(y)
    if ex:
        g = gof_tests(y, ex['cdf']); fits['Expon'] = {**ex, **g}

    # EGD baseline
    eg = fit_exp_gamma(y)
    if eg:
        g = gof_tests(y, eg['cdf']); fits['EGD'] = {**eg, **g}

    # Exponentiated-Gamma
    exg = fit_exponentiated_gamma(y)
    if exg:
        g = gof_tests(y, exg['cdf']); fits['ExpGamma'] = {**exg, **g}

    # Kumaraswamy-Gamma
    kg = fit_kumaraswamy_gamma(y)
    if kg:
        g = gof_tests(y, kg['cdf']); fits['KumGamma'] = {**kg, **g}

    # Lindley (named by Reviewer #2)
    li = fit_lindley(y)
    if li:
        g = gof_tests(y, li['cdf']); fits['Lindley'] = {**li, **g}

    ALL_FIT[ds_name] = fits

    # Print parameter estimates
    print(f"\n  Parameter Estimates:")
    for mname, r in fits.items():
        print(f"  {mname:<12}  {r['params']}")

    # Print criteria table
    print(f"\n  {'Model':<12} {'p':>2}  {'logL':>10}  {'AIC':>10}  {'BIC':>10}"
          f"  {'CAIC':>10}  {'HQIC':>10}  {'KS':>8}  {'KS p-val':>9}  {'CvM':>8}  {'AD':>8}")
    print("  " + "-"*100)
    for mname, r in fits.items():
        mark = " *" if mname == 'QTEG' else "  "
        print(f"  {mname:<12}{mark}{r['npar']:>2}  {r['logL']:>10.3f}  {r['aic']:>10.3f}"
              f"  {r['bic']:>10.3f}  {r['caic']:>10.3f}  {r['hqic']:>10.3f}"
              f"  {r['ks']:>8.4f}  {r['ks_p']:>9.4f}  {r['cvm']:>8.4f}  {r['ad']:>8.4f}")
    print("  * = proposed model")

    # Best per criterion
    crit = ['aic', 'bic', 'ks', 'cvm', 'ad']
    best = {c: min(fits, key=lambda m: fits[m][c]) for c in crit}
    print(f"\n  Best: AIC={best['aic']}  BIC={best['bic']}"
          f"  KS={best['ks']}  CvM={best['cvm']}  AD={best['ad']}")

# ============================================================
# 10.  FIGURES  (4 consolidated figures)
# ============================================================
# Figure 1 -- Theoretical properties  (2x2: PDF, CDF, Survival, Hazard)
# Figure 2 -- Simulation diagnostics  (3x3: Bias, RMSE, CP x 3 scenarios)
# Figure 3 -- Real data DS1 & DS2     (2x4: PDF+CDF+PP+Hazard per dataset)
# Figure 3 -- Real data DS1, DS2 & DS3 (3x4: PDF+CDF+PP+Hazard per dataset)
# ============================================================

# ── Colour / linestyle map ────────────────────────────────────────────────────
COLORS = {
    'QTEG':       ('#1f4e79', '-',           2.5),
    'Weibull':    ('#ed7d31', '--',          1.8),
    'Gamma':      ('#70ad47', '-.',          1.8),
    'LogNorm':    ('#7030a0', ':',           1.8),
    'Expon':      ('#c00000', (0,(3,1,1,1)), 1.5),
    'EGD':        ('#888888', '--',          1.2),
    'ExpGamma':   ('#00b0f0', '-.',          1.5),
    'KumGamma':   ('#ff7500', ':',           1.5),
    'Lindley':    ('#009b77', (0,(5,2)),     1.5),
}

# Greek letter shortcuts (Unicode -- render natively in matplotlib)
A  = '\u03b1'   # alpha
B  = '\u03b2'   # beta
TH = '\u03b8'   # theta
LA = '\u03bb'   # lambda
MU = '\u03bc'   # mu
SG = '\u03c3'   # sigma

print("\n\nGenerating figures ...")

# ── Shared helper: scenario title with Greek ──────────────────────────────────
def sc_title(sc_key):
    import re as _re
    return _re.sub(r'\u03b1=([0-9.]+), \u03b2=([0-9.]+)',
                   lambda m: f'{A}={m.group(1)}, {B}={m.group(2)}', sc_key)

# ── Shared helpers: legend labels with Greek ──────────────────────────────────
def qteg_label(qr):
    return f'QTEG  ({A}={qr["alpha"]:.3f}, {B}={qr["beta"]:.3f})'

def model_label(mname, fit):
    p = fit['params']
    p = (p.replace('alpha=', f'{A}=').replace('beta=',  f'{B}=')
          .replace('sigma=', f'{SG}=').replace('mu=',    f'{MU}=')
          .replace('lambda=',f'{LA}=').replace('theta=', f'{TH}='))
    return f'{mname}  ({p})'

HAZARD_MODELS = ['Weibull', 'Gamma', 'LogNorm', 'Expon', 'Lindley']

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1  Theoretical Properties  (2 x 2)
# ─────────────────────────────────────────────────────────────────────────────
param_sets = [
    (1.0, 1.0, f'{A}=1.0, {B}=1.0'),
    (2.0, 1.0, f'{A}=2.0, {B}=1.0'),
    (3.0, 1.0, f'{A}=3.0, {B}=1.0'),
    (2.0, 2.0, f'{A}=2.0, {B}=2.0'),
    (0.8, 1.0, f'{A}=0.8, {B}=1.0'),
    (4.0, 1.5, f'{A}=4.0, {B}=1.5'),
]
th_colors = ['#1f4e79','#2e75b6','#70ad47','#ed7d31','#c00000','#7030a0']
y_th = np.linspace(0.005, 15, 600)

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle('Figure 1: QTEG Distribution -- Theoretical Properties',
             fontsize=13, fontweight='bold')

panels = [
    (qteg_pdf,    '(a) Probability Density Function',     'f(y)',  None),
    (qteg_cdf,    '(b) Cumulative Distribution Function', 'F(y)',  None),
    (qteg_sf,     '(c) Survival Function',                'S(y)',  None),
    (qteg_hazard, '(d) Hazard Function',                  'h(y)',  3.0),
]
for ax, (func, title, ylabel, ylim_max) in zip(axes.flat, panels):
    for (a_val, b_val, lab), col in zip(param_sets, th_colors):
        vals = func(y_th, a_val, b_val)
        if ylim_max is not None:
            vals = np.minimum(vals, ylim_max)
        ax.plot(y_th, vals, color=col, lw=2, label=lab)
    ax.set_xlabel('y', fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(0, 15); ax.set_ylim(bottom=0)
    if ylim_max: ax.set_ylim(0, ylim_max)
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(OUTPUT_DIR, 'fig1_theoretical.png')
plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2  Simulation Diagnostics  (3 x 3)
# Row 1: |Bias|   Row 2: RMSE   Row 3: Coverage Probability
# Col 1-3: Scenarios 1-3
# ─────────────────────────────────────────────────────────────────────────────
sc_keys     = [r['scenario'] for r in sim_rows if r['n'] == 30]
alpha_trues = [1.5, 2.0, 3.0]
beta_trues  = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle('Figure 2: Monte Carlo Simulation Study  (N = 500 replications)',
             fontsize=13, fontweight='bold')

row_meta = [
    ('|Bias|',               'bias_alpha',  'bias_beta',
     lambda v: np.abs(v),   f'|Bias($\\hat{{{A}}}$)|',  f'|Bias($\\hat{{{B}}}$)|'),
    ('RMSE',                 'rmse_alpha',  'rmse_beta',
     lambda v: v,           f'RMSE($\\hat{{{A}}}$)',     f'RMSE($\\hat{{{B}}}$)'),
    ('Coverage Probability (%)', 'cp_alpha', 'cp_beta',
     lambda v: v,           f'CP($\\hat{{{A}}}$)',       f'CP($\\hat{{{B}}}$)'),
]

for row, (ylabel, col_a, col_b, tfm, lab_a, lab_b) in enumerate(row_meta):
    for col, (sc_key, kt, rt) in enumerate(zip(sc_keys, alpha_trues, beta_trues)):
        sub = sim_df[sim_df['scenario'] == sc_key]
        ax  = axes[row, col]

        ax.plot(sub['n'], tfm(sub[col_a]), 'o-',  color='#1f4e79', lw=2, ms=6,
                label=f'{lab_a}, true={kt}')
        ax.plot(sub['n'], tfm(sub[col_b]), 's--', color='#c00000', lw=2, ms=6,
                label=f'{lab_b}, true={rt}')

        if ylabel == 'Coverage Probability (%)':
            ax.axhline(95, color='gray', ls=':', lw=1.5, label='Nominal 95%')
            ax.set_ylim(60, 105)

        # Only label outer axes to save space
        if row == 2: ax.set_xlabel('Sample size n', fontsize=9)
        if col == 0: ax.set_ylabel(ylabel, fontsize=9)

        # Column title only on top row
        if row == 0:
            ax.set_title(sc_key, fontsize=9, fontweight='bold')

        ax.legend(fontsize=7.5, framealpha=0.85)
        ax.grid(alpha=0.3)
        ax.set_xticks(SAMPLE_SIZES)

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(OUTPUT_DIR, 'fig2_simulation.png')
plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()
print(f"  Saved: {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: draw one dataset's 4 panels into a 2x2 sub-block of axes
# ─────────────────────────────────────────────────────────────────────────────
def draw_dataset_panels(axes_block, ds_name, y, fits, xlabel, ds_label):
    """
    axes_block: 2x2 array of axes
    Panels: (a) PDF vs histogram  (b) Empirical vs Fitted CDF
            (c) P-P plot          (d) Hazard Function
    """
    if 'QTEG' not in fits:
        return
    qr        = fits['QTEG']
    alpha_fit = qr['alpha']; beta_fit = qr['beta']

    n       = len(y)
    ys      = np.sort(y)
    emp_cdf = np.arange(1, n + 1) / n
    y_plot  = np.linspace(max(y.min() * 0.02, 0.005), y.max() * 1.02, 500)

    plot_comps = {
        m: fits[m] for m in ['Weibull','Gamma','LogNorm','Expon','Lindley']
        if m in fits
    }

    # (a) PDF vs histogram
    ax = axes_block[0, 0]
    ax.hist(y, bins=min(20, n // 4), density=True, alpha=0.25,
            color='steelblue', edgecolor='white', label='Observed')
    ax.plot(y_plot, qteg_pdf(y_plot, alpha_fit, beta_fit),
            color='#1f4e79', lw=2.5, label=qteg_label(qr))
    for mname, mfit in plot_comps.items():
        col, ls, lw = COLORS[mname]
        ax.plot(y_plot, mfit['pdf'](y_plot), color=col, lw=lw, ls=ls,
                label=model_label(mname, mfit))
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.set_title(f'{ds_label}(a) Fitted PDF vs Histogram', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5, framealpha=0.85, loc='upper right'); ax.grid(alpha=0.3)

    # (b) CDF
    ax = axes_block[0, 1]
    ax.step(ys, emp_cdf, color='black', lw=1.8, where='post', label='Empirical CDF')
    ax.plot(y_plot, qteg_cdf(y_plot, alpha_fit, beta_fit),
            color='#1f4e79', lw=2.5, label=qteg_label(qr))
    for mname, mfit in plot_comps.items():
        col, ls, lw = COLORS[mname]
        ax.plot(y_plot, mfit['cdf'](y_plot), color=col, lw=lw, ls=ls,
                label=model_label(mname, mfit))
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('F(y)', fontsize=9)
    ax.set_title(f'{ds_label}(b) Empirical vs Fitted CDF', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6.5, framealpha=0.85, loc='lower right'); ax.grid(alpha=0.3)

    # (c) P-P plot
    ax = axes_block[1, 0]
    theo_p = qteg_cdf(ys, alpha_fit, beta_fit)
    ax.scatter(theo_p, emp_cdf, s=18, color='#1f4e79', alpha=0.8, zorder=3,
               label=qteg_label(qr))
    ax.plot([0, 1], [0, 1], 'r--', lw=1.5, label='Reference line')
    corr = float(np.corrcoef(theo_p, emp_cdf)[0, 1])
    ax.set_xlabel('Theoretical', fontsize=9); ax.set_ylabel('Empirical', fontsize=9)
    ax.set_title(f'{ds_label}(c) P-P Plot  (r = {corr:.4f})', fontsize=9, fontweight='bold')
    ax.legend(fontsize=7, framealpha=0.85); ax.grid(alpha=0.3)

    # (d) Hazard
    ax   = axes_block[1, 1]
    h_q  = qteg_hazard(y_plot, alpha_fit, beta_fit)
    h_med  = float(np.median(h_q[h_q > 0]))
    h_ceil = min(h_q.max() * 1.5, h_med * 20)
    ax.plot(y_plot, np.minimum(h_q, h_ceil),
            color='#1f4e79', lw=2.5, label=qteg_label(qr))
    for mname in HAZARD_MODELS:
        if mname not in plot_comps or 'haz' not in plot_comps[mname]:
            continue
        col, ls, lw = COLORS[mname]
        h_c = np.minimum(plot_comps[mname]['haz'](y_plot), h_ceil)
        ax.plot(y_plot, h_c, color=col, lw=lw, ls=ls,
                label=model_label(mname, plot_comps[mname]))
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel('h(y)', fontsize=9)
    ax.set_title(f'{ds_label}(d) Hazard Function', fontsize=9, fontweight='bold')
    ax.set_ylim(0, h_ceil); ax.legend(fontsize=6.5, framealpha=0.85); ax.grid(alpha=0.3)

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3  DS1 (Bladder Cancer) + DS2 (Boeing 720) + DS3 (Melanoma)
# Three datasets, each as a 1x4 row of panels (3 rows x 4 cols total)
# ─────────────────────────────────────────────────────────────────────────────
ds_triples = [
    ('DS1: Bladder Cancer Remission (n=128)', 'Remission Time (months)',  'DS1: '),
    ('DS2: Boeing 720 Failures (n=213)',      'Time Between Failures',    'DS2: '),
    ('DS3: Malignant Melanoma (n=205)',        'Survival Time (years)',    'DS3: '),
]

fig, axes = plt.subplots(3, 4, figsize=(22, 16))
fig.suptitle(
    'Figure 3: Empirical Applications -- DS1 Bladder Cancer, '
    'DS2 Boeing 720 Failures, DS3 Malignant Melanoma',
    fontsize=12, fontweight='bold')

for row, (ds_name, xlabel, ds_label) in enumerate(ds_triples):
    y    = DATA[ds_name]
    fits = ALL_FIT[ds_name]
    block = np.array([[axes[row, 0], axes[row, 1]],
                      [axes[row, 2], axes[row, 3]]])
    draw_dataset_panels(block, ds_name, y, fits, xlabel, ds_label)

plt.tight_layout(rect=[0, 0, 1, 0.97])
path = os.path.join(OUTPUT_DIR, 'fig3_all_datasets.png')
plt.savefig(path, dpi=180, bbox_inches='tight'); plt.close()
print(f"  Saved: {path}")

# ============================================================
# 11.  WRAP-UP
# ============================================================

log.close()
sys.stdout = sys.__stdout__

print(f"""
================================================================================
ALL DONE.

Files created in:  {OUTPUT_DIR}

  QTEG_results.txt      <- all tables (copy into paper)
  fig1_theoretical.png  <- Figure 1: Theoretical properties (2x2)
  fig2_simulation.png   <- Figure 2: Simulation diagnostics (3x3)
  fig3_all_datasets.png <- Figure 3: DS1 Bladder + DS2 Boeing + DS3 Melanoma (3x4)

Upload QTEG_results.txt + 4 PNG files here for confirmation.
================================================================================
""")
