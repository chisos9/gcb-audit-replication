"""
Estimate the mu_C-augmented JRSSAClimate model on the 2025 vintage only,
on the common 1959-2016 sample (T=58).

Outputs:
  parameters_jrssa_muC_2025.npy   21-element parameter vector
  hessian_jrssa_muC_2025_4h.npy   raw central-difference Hessian (21x21)
  cov_jrssa_muC_2025_4h.npy       regularised inverse (21x21)
  jrssa_muC_2025_results.npz      smoothed states + loglik
"""
import os
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
RESULTS_DIR = os.path.abspath(os.path.join(CODE_DIR, os.pardir, 'results'))
os.chdir(RESULTS_DIR)

from jrssa_m2_climate_muC import JRSSAClimateMuC
from load_jrssa_vintage import load_vintage_for_jrssa


from estimation_utils import multistart, robust_hessian


def _neg_loglik(params, model):
    return -model.loglike(params)


def main(year_lo=1959, year_hi=2016):
    print(f"\n{'=' * 72}")
    print(f"  Estimating JRSSAClimateMuC on GCB 2025 vintage "
          f"({year_lo}-{year_hi}, T={year_hi - year_lo + 1})")
    print('=' * 72)

    d = load_vintage_for_jrssa('2025', year_lo, year_hi)
    m = JRSSAClimateMuC(d['mObs'],
                        d['enso_shifted'], d['nao_shifted'],
                        d['pdo_shifted'], d['saod_shifted'])

    # Anchor: baseline 2025 fit plus mu_C = -0.7 (approx. BIM + S_CARB - d)
    anchors = []
    baseline_path = os.path.join(RESULTS_DIR, 'parameters_jrssa_2025.npy')
    if os.path.exists(baseline_path):
        p_base = np.load(baseline_path)
        anchors.append(('baseline_2025+mu',
                        np.concatenate([p_base, [-0.7]])))
        print(f"  anchoring on parameters_jrssa_2025.npy "
              f"(len={len(p_base)} + 1 for mu_C)")

    t0 = time.time()
    best_ll, p_hat, modes = multistart(m, anchors=anchors, seed=42 + 2025,
                                       n_jitter=10, scale=0.4)
    print(f"  multistart: {len(modes)} mode(s), best loglik={best_ll:.4f}, "
          f"t={time.time()-t0:.1f}s")
    res_sm = m.smooth(p_hat)
    alphahat = res_sm.smoothed_state.T
    state_cov = res_sm.smoothed_state_cov

    disp = JRSSAClimateMuC.transform_params_display(p_hat)
    print(f"\n  Final loglik: {best_ll:.4f}")
    print(f"  beta1 = {disp['beta1']:.4f}     beta2 = {disp['beta2']:.4f}")
    print(f"  phi1  = {disp['phi1']:.4f}     phi3  = {disp['phi3']:.4f}     "
          f"phiE  = {disp['phiE']:.4f}")
    print(f"  sigma2_eta1 = {disp['sigma2_eta1']:.4f}   "
          f"sigma2_eta2 = {disp['sigma2_eta2']:.4f}")
    print(f"  sigma2_eta3 = {disp['sigma2_eta3']:.6f}   "
          f"sigma2_etaE = {disp['sigma2_etaE']:.6f}")
    print(f"  sigma2_epsE = {disp['sigma2_epsE']:.6f}")
    print(f"  mu_C        = {disp['mu_C']:+.6f}   (NEW)")
    print(f"  r12 = {disp['r12']:.4f}    r13 = {disp['r13']:.4f}")
    print(f"  b1_saod = {disp['b1_saod']:+.4f}     b2_saod = {disp['b2_saod']:+.4f}")
    print(f"  c1 = {alphahat[-1, 9]:+.4f}  c2 = {alphahat[-1, 10]:+.4f}  "
          f"d = {alphahat[-1, 11]:+.4f}")

    np.save('parameters_jrssa_muC_2025.npy', p_hat)
    np.savez('jrssa_muC_2025_results.npz',
             params=p_hat, loglik=best_ll,
             smoothed_state=alphahat, smoothed_state_cov=state_cov,
             years=d['years'])

    print("\n  Computing central-difference Hessian (21x21, steps 4h and 2h)...")
    t0 = time.time()
    H, cov, n_floored = robust_hessian(lambda p: -m.loglike(p), p_hat,
                                       param_names=list(m.param_names))
    print(f"  Hessian done in {time.time()-t0:.1f}s  n_floored={n_floored}")
    eigs = np.linalg.eigvalsh(0.5 * (H + H.T))
    print(f"  eigenvalues: min={eigs.min():+.3e} max={eigs.max():+.3e}")
    np.save('hessian_jrssa_muC_2025_4h.npy', H)
    np.save('cov_jrssa_muC_2025_4h.npy', cov)

    print(f"\n{'=' * 72}")
    print(f"  Done.")
    print('=' * 72)


if __name__ == '__main__':
    main()
