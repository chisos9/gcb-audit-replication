"""
Estimate the JRSSA-Climate model on every GCB vintage 2017-2025, on
the common 1959-2016 sample (T=58).

This version uses the X4-free specification: emissions measurement
error is plain Gaussian (eps_E ~ N(0, sigma2_epsE)), implemented via
obs_cov[1,1]. State vector is 12 elements (was 13); free parameters
are 20 (was 21).

For each vintage v, saves:
  parameters_jrssa_<v>.npy        20-element MLE parameter vector
  jrssa_<v>_results.npz           smoothed states, loglik, etc.

Per-vintage runtime: ~1 min (seeded multistart, about 12 polished
starts; see estimation_utils). Nine vintages serial: ~10 min.

Existing parameters_jrssa_{2024,2025}.npy and
jrssa_{2024,2025}_results.npz are on the 1959-2023 sample and will be
backed up to <name>_1959_2023_backup.npy / .npz before being
overwritten with the new 1959-2016 estimates. Set
BACKUP_EXISTING=False at the top of this file to skip the backup.
"""
import os
import sys
import shutil
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(PKG_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, CODE_DIR)
os.chdir(RESULTS_DIR)

from estimation_utils import multistart
from jrssa_m2_climate import JRSSAClimate
from load_jrssa_vintage import (
    load_vintage_for_jrssa, COMMON_LO_DEFAULT, COMMON_HI_DEFAULT
)
import load_vintage_panels as lvp

YEAR_LO = COMMON_LO_DEFAULT   # 1959
YEAR_HI = COMMON_HI_DEFAULT   # 2016
BACKUP_EXISTING = False


def _neg_loglik(params, model):
    return -model.loglike(params)


def estimate_vintage(vintage, year_lo=YEAR_LO, year_hi=YEAR_HI, seed=42):
    print(f"\n{'=' * 72}")
    print(f"  Estimating JRSSA-Climate on GCB {vintage} "
          f"({year_lo}-{year_hi}, T={year_hi - year_lo + 1})")
    print('=' * 72)
    d = load_vintage_for_jrssa(vintage, year_lo, year_hi)
    m = JRSSAClimate(d['mObs'],
                     d['enso_shifted'], d['nao_shifted'],
                     d['pdo_shifted'], d['saod_shifted'])

    # Seeded multistart (see estimation_utils): anchors are the model's
    # start_params and, when available, the previous vintage's fit, so
    # the chain of vintages shares information without hiding modes.
    anchors = []
    prev = str(int(vintage) - 1)
    prev_file = os.path.join(RESULTS_DIR, f'parameters_jrssa_{prev}.npy')
    if os.path.exists(prev_file):
        anchors.append((f'fit_{prev}', np.load(prev_file)))
    t0 = time.time()
    best_ll, p_hat, modes = multistart(m, anchors=anchors,
                                       seed=seed + int(vintage),
                                       n_jitter=10, scale=0.4)
    print(f"  multistart: {len(modes)} mode(s), best loglik={best_ll:.4f}, "
          f"t={time.time()-t0:.1f}s")
    res_sm = m.smooth(p_hat)
    alphahat = res_sm.smoothed_state.T            # (cT, cr)
    state_cov = res_sm.smoothed_state_cov         # (cr, cr, cT)

    disp = JRSSAClimate.transform_params_display(p_hat)
    print(f"\n  Final loglik: {best_ll:.4f}")
    print(f"  beta1 = {disp['beta1']:.4f}     beta2 = {disp['beta2']:.4f}")
    print(f"  c1 (smoothed last) = {alphahat[-1, 9]:+.4f}")
    print(f"  c2 (smoothed last) = {alphahat[-1, 10]:+.4f}")
    print(f"  d  (smoothed last) = {alphahat[-1, 11]:+.4f}")

    return p_hat, best_ll, alphahat, state_cov, d


def _backup(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"  backed up {src} -> {dst}")


def main(vintages=None):
    if vintages is None:
        vintages = lvp.VINTAGES   # ['2017', ..., '2025']

    if BACKUP_EXISTING:
        print('Backing up any existing 1959-2023 outputs to *_1959_2023_backup.*')
        for v in ('2024', '2025'):
            _backup(f'parameters_jrssa_{v}.npy',
                    f'parameters_jrssa_{v}_1959_2023_backup.npy')
            _backup(f'jrssa_{v}_results.npz',
                    f'jrssa_{v}_results_1959_2023_backup.npz')
            _backup(f'hessian_jrssa_{v}_4h.npy',
                    f'hessian_jrssa_{v}_4h_1959_2023_backup.npy')
            _backup(f'cov_jrssa_{v}_4h.npy',
                    f'cov_jrssa_{v}_4h_1959_2023_backup.npy')

    t_total = time.time()
    for v in vintages:
        p_hat, ll, st, V, d = estimate_vintage(v)
        np.save(f'parameters_jrssa_{v}.npy', p_hat)
        np.savez(f'jrssa_{v}_results.npz',
                 params=p_hat, loglik=ll,
                 smoothed_state=st, smoothed_state_cov=V,
                 years=d['years'],
                 year_lo=d['year_lo'], year_hi=d['year_hi'])

    print(f"\n{'=' * 72}")
    print(f"  Done all {len(vintages)} vintages in "
          f"{time.time() - t_total:.1f}s")
    print(f"  Files saved in {CODE_DIR}")
    print('=' * 72)


if __name__ == '__main__':
    main()
