"""
Multi-vintage AF/SR Monte Carlo using statsmodels' JRSSAClimate, with
multiprocessing across vintages.

For each GCB vintage 2017-2025 that has both a parameters file and a
covariance file on disk, this runs N_DRAWS smoother evaluations on
parameter draws from N(theta_hat, H^{-1}), computes the AF and SR
pointwise series from the smoothed states, and saves the 99% bands
plus point estimate.

We abandoned the experimental custom NumPy kernel
(jrssa_af_sr_fast.py) because the diffuse-like initial covariance
(P_1 with 1e6 on several diagonals) amplifies tiny per-step numerical
differences into O(1) divergences in the smoothed states. statsmodels
handles this internally via more careful diffuse-prior treatment.

Per-vintage runtime is ~2 minutes at N=10,000 draws (11 ms/draw).
On 8 cores in parallel, nine vintages complete in ~3 minutes wall time.

Inputs (per vintage v):
  parameters_jrssa_<v>.npy        MLE parameter vector
  hessian_jrssa_<v>_4h.npy        raw central-difference Hessian
  data/<vintage spreadsheet>
  data/climate_indices_annual_1959_2024.csv
  data/saod_annual_1959_2024.csv

Parameter draws are sampled from N(theta_hat, cov_proj) where cov_proj
is the inverse Hessian restricted to its well-identified subspace
(eigendirections with eigenvalue >= EIG_FLOOR). On boundary vintages
(e.g. 2018 with phi_1 at its sigmoid boundary) the spectral regulariser
in cov_jrssa_<v>_4h.npy floors a tiny eigenvalue to 1e-8 and the
inverted direction carries 1e+8 variance, which would otherwise dominate
the MC bands. Projecting those directions out leaves the bands reflecting
only the well-identified part of the likelihood. For non-boundary
vintages this is identical to drawing from the on-disk cov.

Outputs (per vintage v):
  jrssa_af_sr_bands_<v>.npz       point + pointwise bands

Run:
    python jrssa_af_sr_vintages.py
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
ROOT = os.path.abspath(os.path.join(CODE_DIR, os.pardir))
RESULTS_DIR = os.path.join(ROOT, 'results')
DATA = os.path.join(ROOT, 'data')

import load_vintage_panels as lvp
from load_jrssa_vintage import (
    load_vintage_for_jrssa, COMMON_LO_DEFAULT, COMMON_HI_DEFAULT,
)

N_DRAWS = 10_000
RNG_SEED_BASE = 12345
COMMON_LO = COMMON_LO_DEFAULT   # 1959
COMMON_HI = COMMON_HI_DEFAULT   # 2016
EIG_FLOOR = 1e-8                # matches jrssa_compute_hessian_all_vintages


def _well_identified_factor(H, eig_floor=EIG_FLOOR):
    """Square-root factor L such that L L^T = H^{-1} restricted to its
    well-identified subspace. Floored eigendirections receive zero
    variance. Returns (L, n_floored).
    """
    H_sym = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H_sym)
    keep = w >= eig_floor
    sd = np.zeros_like(w)
    sd[keep] = 1.0 / np.sqrt(w[keep])
    L = V * sd          # broadcasts: L[i,k] = V[i,k] * sd[k]
    return L, int(np.sum(~keep))


# ── Per-vintage worker (uses statsmodels) ────────────────────────────
def run_vintage(vintage, year_lo=COMMON_LO, year_hi=COMMON_HI,
                n_draws=N_DRAWS, seed=None):
    """AF/SR Monte Carlo on one vintage via statsmodels' JRSSAClimate.
    Saves jrssa_af_sr_bands_<vintage>.npz.
    """
    label = f'vintage {vintage}'
    if seed is None:
        seed = RNG_SEED_BASE + int(vintage)
    t0 = time.perf_counter()

    params_file = os.path.join(RESULTS_DIR, f'parameters_jrssa_{vintage}.npy')
    H_file = os.path.join(RESULTS_DIR, f'hessian_jrssa_{vintage}_4h.npy')
    if not (os.path.exists(params_file) and os.path.exists(H_file)):
        print(f'[{label}] skipping: missing parameters or hessian')
        return None

    p_hat = np.load(params_file)
    H = np.load(H_file)
    L, n_floored = _well_identified_factor(H)
    if n_floored:
        print(f'[{label}] projected out {n_floored} floored '
              f'eigendirection(s) from parameter-draw covariance')
    d = load_vintage_for_jrssa(vintage, year_lo, year_hi)
    cT = len(d['years'])

    # Import inside worker so each subprocess pays import cost only once
    from jrssa_m2_climate import JRSSAClimate, cGtC1750
    m = JRSSAClimate(d['mObs'],
                     d['enso_shifted'], d['nao_shifted'],
                     d['pdo_shifted'], d['saod_shifted'])

    # Point estimate
    res_pt = m.smooth(p_hat)
    st_pt = res_pt.smoothed_state
    af_pt = st_pt[1] / st_pt[8]
    sr_pt = (st_pt[2] + st_pt[3]) / (st_pt[0] - cGtC1750)

    # Parameter-draw factor L is the well-identified square root computed
    # from the Hessian above; floored eigendirections (if any) get zero
    # variance, so draws stay in the well-identified subspace.

    rng = np.random.default_rng(seed)
    AF = np.full((n_draws, cT), np.nan)
    SR = np.full((n_draws, cT), np.nan)
    n_ok = 0
    for k in range(n_draws):
        pk = p_hat + L @ rng.standard_normal(len(p_hat))
        try:
            r_k = m.smooth(pk)
            st_k = r_k.smoothed_state
            AF[k] = st_k[1] / st_k[8]
            SR[k] = (st_k[2] + st_k[3]) / (st_k[0] - cGtC1750)
            n_ok += 1
        except Exception:
            pass

    af_lo = np.nanpercentile(AF, 0.5, axis=0)
    af_hi = np.nanpercentile(AF, 99.5, axis=0)
    sr_lo = np.nanpercentile(SR, 0.5, axis=0)
    sr_hi = np.nanpercentile(SR, 99.5, axis=0)

    out_path = os.path.join(RESULTS_DIR, f'jrssa_af_sr_bands_{vintage}.npz')
    np.savez(out_path,
             years=d['years'],
             af_pt=af_pt, sr_pt=sr_pt,
             af_lo=af_lo, af_hi=af_hi,
             sr_lo=sr_lo, sr_hi=sr_hi,
             n_ok=n_ok, n_draws=n_draws,
             year_lo=year_lo, year_hi=year_hi)
    elapsed = time.perf_counter() - t0
    print(f'[{label}] {n_ok}/{n_draws} valid draws, T={cT}, '
          f'{elapsed:.1f}s  -> jrssa_af_sr_bands_{vintage}.npz')
    return dict(vintage=vintage, n_ok=n_ok, elapsed=elapsed)


def main(year_lo=COMMON_LO, year_hi=COMMON_HI, n_draws=N_DRAWS,
         vintages=None, n_workers=None):
    if vintages is None:
        vintages = lvp.VINTAGES   # ['2017', ..., '2025']
    if n_workers is None:
        n_workers = min(cpu_count(), len(vintages))

    print('Multi-vintage AF/SR MC (statsmodels + multiprocess)')
    print(f'  vintages: {vintages}')
    print(f'  sample:   {year_lo}-{year_hi}')
    print(f'  N draws:  {n_draws}')
    print(f'  workers:  {n_workers}')
    print()

    # Pre-filter vintages that have both files
    avail = []
    for v in vintages:
        p = os.path.join(RESULTS_DIR, f'parameters_jrssa_{v}.npy')
        h = os.path.join(RESULTS_DIR, f'hessian_jrssa_{v}_4h.npy')
        if os.path.exists(p) and os.path.exists(h):
            avail.append(v)
        else:
            print(f'  skipping {v}: parameters or hessian file missing')
    if not avail:
        print('No vintages available.')
        return
    print(f'  running on: {avail}\n')

    t0 = time.perf_counter()
    args = [(v, year_lo, year_hi, n_draws) for v in avail]
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.starmap(run_vintage, args)
    else:
        results = [run_vintage(*a) for a in args]
    elapsed = time.perf_counter() - t0
    print(f'\nAll done in {elapsed:.1f}s ({elapsed/60:.1f} min wall time)')
    return results


if __name__ == '__main__':
    main()
