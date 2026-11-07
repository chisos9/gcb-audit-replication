"""
Compute Hessians and regularised inverses for every JRSSA vintage MLE,
using a step-checked central-difference Hessian with a relative
eigenvalue floor (see estimation_utils.robust_hessian).

For each vintage v with parameters_jrssa_<v>.npy on disk:
  hessian_jrssa_<v>_4h.npy
  cov_jrssa_<v>_4h.npy   (= H^{-1}, spectrally regularised)

Per-vintage runtime: ~3-5 min (each off-diagonal Hessian entry needs
4 likelihood evaluations; 20x20/2 = 210 unique pairs × 4 evals × ~10 ms
≈ 10 sec? but with overhead ~3 min in practice based on existing logs).

Nine vintages: optional multiprocess across vintages via Pool. Set
N_WORKERS=1 below for serial, or leave as None for auto.
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
import numpy as np

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(CODE_DIR)
RESULTS_DIR = os.path.join(PKG_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, CODE_DIR)
os.chdir(RESULTS_DIR)

from jrssa_m2_climate import JRSSAClimate
from load_jrssa_vintage import (
    load_vintage_for_jrssa, COMMON_LO_DEFAULT, COMMON_HI_DEFAULT
)
import load_vintage_panels as lvp

N_WORKERS = None       # None -> auto (min(cpu_count, n_vintages))


from estimation_utils import robust_hessian


def run_vintage(vintage):
    params_file = f'parameters_jrssa_{vintage}.npy'
    if not os.path.exists(params_file):
        print(f'[vintage {vintage}] skipping: missing {params_file}')
        return None

    print(f"\n{'=' * 72}")
    print(f"  Hessian: GCB {vintage}")
    print('=' * 72)
    d = load_vintage_for_jrssa(vintage)
    m = JRSSAClimate(d['mObs'],
                     d['enso_shifted'], d['nao_shifted'],
                     d['pdo_shifted'], d['saod_shifted'])
    p_hat = np.load(params_file)
    print(f"  loaded {params_file}: shape={p_hat.shape}")

    def f(params):
        return -m.loglike(params)

    t0 = time.time()
    n = len(p_hat)
    H, cov, n_floored = robust_hessian(f, p_hat,
                                       param_names=list(m.param_names))
    print(f"  computed {n}x{n} Hessian (steps 4h and 2h) in "
          f"{time.time()-t0:.1f}s; f(MLE) = {f(p_hat):.6f}  "
          f"n_floored={n_floored}")
    eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
    print(f"  eigenvalues: min={eigs.min():+.3e} max={eigs.max():+.3e}")

    np.save(f'hessian_jrssa_{vintage}_4h.npy', H)
    np.save(f'cov_jrssa_{vintage}_4h.npy', cov)
    print(f"  saved hessian_jrssa_{vintage}_4h.npy and "
          f"cov_jrssa_{vintage}_4h.npy")
    return dict(vintage=vintage, n_floored=n_floored,
                eig_min=float(eigs.min()), eig_max=float(eigs.max()))


def main(vintages=None, n_workers=N_WORKERS):
    if vintages is None:
        vintages = lvp.VINTAGES
    if n_workers is None:
        n_workers = min(cpu_count(), len(vintages))

    print(f'Multi-vintage Hessian: vintages={vintages}, workers={n_workers}\n')
    t0 = time.time()
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(run_vintage, vintages)
    else:
        results = [run_vintage(v) for v in vintages]

    print(f"\n{'=' * 72}")
    print(f"  All done in {time.time() - t0:.1f}s "
          f"({(time.time() - t0)/60:.1f} min wall time)")
    print('=' * 72)
    # quick summary
    print('\nEigenvalue summary:')
    for r in results:
        if r is not None:
            print(f"  {r['vintage']}: min={r['eig_min']:+.3e} "
                  f"max={r['eig_max']:+.3e}  n_floored={r['n_floored']}")


if __name__ == '__main__':
    main()
