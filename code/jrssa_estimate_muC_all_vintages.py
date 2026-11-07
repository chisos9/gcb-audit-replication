"""
Estimate the mu_C-augmented MDS-GCB (JRSSAClimateMuC) on every GCB vintage
2017-2025, common 1959-2016 sample (T=58), and test mu_C = 0 against the
stored baseline fits.

For each vintage v:
  parameters_jrssa_muC_{v}.npy     21-element parameter vector
  hessian_jrssa_muC_{v}_4h.npy     raw central-difference Hessian (21x21)
  cov_jrssa_muC_{v}_4h.npy         regularised inverse (21x21)
  jrssa_muC_{v}_results.npz        smoothed states + loglik

Summary table across vintages:
  jrssa_muC_vintages.csv           mu_C, SE, t, logL(base), logL(muC), LR, p

Baseline log-likelihoods are read from the existing jrssa_{v}_results.npz,
so the 20-parameter fits are not recomputed.
"""
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
RESULTS_DIR = os.path.abspath(os.path.join(CODE_DIR, os.pardir, 'results'))
os.chdir(RESULTS_DIR)

from jrssa_m2_climate_muC import JRSSAClimateMuC
from load_jrssa_vintage import load_vintage_for_jrssa
from estimation_utils import multistart, robust_hessian
from jrssa_estimate_muC_2025 import _neg_loglik

VINTAGES = ['2017', '2018', '2019', '2020', '2021', '2022', '2023',
            '2024', '2025']
MU_IDX = 20
YEAR_LO, YEAR_HI = 1959, 2016


def baseline_loglik(v):
    f = os.path.join(RESULTS_DIR, 'jrssa_%s_results.npz' % v)
    return float(np.load(f, allow_pickle=True)['loglik'])


def bim_mean(d):
    """Common-sample mean of the vintage's implied budget imbalance, used
    only as a starting value for mu_C."""
    try:
        return float(np.nanmean(d['bim']))
    except Exception:
        return 0.0


def estimate(v):
    print("\n%s\n  GCB %s vintage, %d-%d (T=%d)\n%s"
          % ('=' * 72, v, YEAR_LO, YEAR_HI, YEAR_HI - YEAR_LO + 1, '=' * 72))
    d = load_vintage_for_jrssa(v, YEAR_LO, YEAR_HI)
    m = JRSSAClimateMuC(d['mObs'], d['enso_shifted'], d['nao_shifted'],
                        d['pdo_shifted'], d['saod_shifted'])

    p_base = np.load(os.path.join(RESULTS_DIR, 'parameters_jrssa_%s.npy' % v))
    mu0 = -bim_mean(d)
    if not np.isfinite(mu0):
        mu0 = 0.0
    anchors = [('baseline+mu0', np.concatenate([p_base, [mu0]])),
               ('baseline+mu_zero', np.concatenate([p_base, [0.0]]))]
    print("  anchors from parameters_jrssa_%s.npy, mu_C init %+.4f and 0"
          % (v, mu0))

    t0 = time.time()
    best_ll, p_hat, modes = multistart(m, anchors=anchors, seed=42 + int(v),
                                       n_jitter=10, scale=0.4)
    print("  multistart: %d mode(s), best loglik=%.4f, t=%.1fs"
          % (len(modes), best_ll, time.time() - t0))
    sm = m.smooth(p_hat)

    H, cov, n_floored = robust_hessian(lambda p: -m.loglike(p), p_hat,
                                       param_names=list(m.param_names))
    eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
    se_mu = float(np.sqrt(max(cov[MU_IDX, MU_IDX], 0.0)))
    mu = float(p_hat[MU_IDX])

    ll_base = baseline_loglik(v)
    lr = 2.0 * (best_ll - ll_base)
    p_lr = float(chi2.sf(max(lr, 0.0), 1))
    disp = JRSSAClimateMuC.transform_params_display(p_hat)

    np.save('parameters_jrssa_muC_%s.npy' % v, p_hat)
    np.save('hessian_jrssa_muC_%s_4h.npy' % v, H)
    np.save('cov_jrssa_muC_%s_4h.npy' % v, cov)
    np.savez('jrssa_muC_%s_results.npz' % v, params=p_hat, loglik=best_ll,
             smoothed_state=sm.smoothed_state.T,
             smoothed_state_cov=sm.smoothed_state_cov, years=d['years'])

    print("  mu_C = %+.4f (%.4f)   t = %+.2f" % (mu, se_mu, mu / se_mu if se_mu else np.nan))
    print("  logL base = %.4f   logL muC = %.4f   LR = %.2f   p = %.4g"
          % (ll_base, best_ll, lr, p_lr))
    print("  beta1 = %.4f  beta2 = %.4f  phi1 = %.4f  phi3 = %.4f"
          % (disp['beta1'], disp['beta2'], disp['phi1'], disp['phi3']))
    print("  eig min = %+.3e  n_floored = %d" % (eigs.min(), n_floored))

    return dict(vintage=v, mu_C=mu, se_mu_C=se_mu,
                t=mu / se_mu if se_mu else np.nan,
                logL_base=ll_base, logL_muC=best_ll, LR=lr, p_LR=p_lr,
                beta1=disp['beta1'], beta2=disp['beta2'],
                phi1=disp['phi1'], phi3=disp['phi3'],
                n_floored=n_floored)


def summarise(vintages=VINTAGES):
    """Rebuild the summary table from saved per-vintage artifacts."""
    rows = []
    for v in vintages:
        try:
            p_hat = np.load('parameters_jrssa_muC_%s.npy' % v)
            cov = np.load('cov_jrssa_muC_%s_4h.npy' % v)
            ll = float(np.load('jrssa_muC_%s_results.npz' % v,
                               allow_pickle=True)['loglik'])
        except Exception as e:
            print("  %s: missing artifacts (%s)" % (v, e))
            continue
        mu = float(p_hat[MU_IDX])
        se = float(np.sqrt(max(cov[MU_IDX, MU_IDX], 0.0)))
        ll_b = baseline_loglik(v)
        lr = 2.0 * (ll - ll_b)
        disp = JRSSAClimateMuC.transform_params_display(p_hat)
        rows.append(dict(vintage=v, mu_C=mu, se_mu_C=se,
                         t=mu / se if se else np.nan,
                         logL_base=ll_b, logL_muC=ll, LR=lr,
                         p_LR=float(chi2.sf(max(lr, 0.0), 1)),
                         beta1=disp['beta1'], beta2=disp['beta2'],
                         phi1=disp['phi1'], phi3=disp['phi3']))
    return pd.DataFrame(rows)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    todo = args if args else VINTAGES
    if '--summary-only' not in sys.argv:
        for v in todo:
            try:
                estimate(v)
            except Exception as e:
                print("  VINTAGE %s FAILED: %s" % (v, e))
    df = summarise()
    if df.empty:
        print("no results yet"); return
    df.to_csv('jrssa_muC_vintages.csv', index=False)
    print("\n%s\n  SUMMARY\n%s" % ('=' * 72, '=' * 72))
    with pd.option_context('display.width', 200, 'display.max_columns', 30):
        print(df[['vintage', 'mu_C', 'se_mu_C', 't', 'logL_base', 'logL_muC',
                  'LR', 'p_LR', 'beta1', 'beta2']].to_string(
                      index=False, float_format=lambda x: '%8.4f' % x))
    print("\n  wrote jrssa_muC_vintages.csv")


if __name__ == '__main__':
    main()
