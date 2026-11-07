"""
Headline figure for the multi-vintage model-based audit.

Plots beta_1 and beta_2 across vintages 2017-2025, each with a
+/- 1.96 delta-method standard-error bar from the regularised
Hessian inverse. The 2025 vintage is highlighted in red.

beta_1 and beta_2 are reported in the "display" units used elsewhere in
the paper:
  beta_i = 10 * sigmoid(-p_beta_i)

The standard error on beta_i comes from the delta method:
  d beta_i / d p = -10 * sigmoid * (1 - sigmoid)
  Var(beta_i)    = (d beta_i / d p)^2 * Var(p_beta_i)

Reads parameters_jrssa_<v>.npy and hessian_jrssa_<v>_4h.npy for every
vintage that has both. For boundary cases where the spectral
regulariser floors a Hessian eigenvalue (e.g. the 2018 vintage with
phi_1 at its sigmoid boundary), the floored eigendirection's
contribution to the parameter variance is artificial and we project
it out before reading the diagonal. For vintages with no floored
directions the result is identical to taking cov[p_idx, p_idx] from
the regularised cov on disk.

Saves
  output/figures/figure_jrssa_beta_vintages.{pdf,png}
"""
import os
import sys
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
ROOT = os.path.abspath(os.path.join(CODE_DIR, os.pardir))
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(ROOT, 'output', 'figures')

import load_vintage_panels as lvp


def _sigmoid(x):
    return np.exp(-x) / (1.0 + np.exp(-x))


EIG_FLOOR = 1e-8  # matches jrssa_compute_hessian_all_vintages.spectral_regularise_inverse


def _well_identified_cov(H, eig_floor=EIG_FLOOR):
    """Inverse of H using only well-identified eigendirections.

    Eigenvalues of the symmetrised Hessian below `eig_floor` are treated
    as identifying noise (the spectral regulariser would otherwise floor
    them, producing 1/eig_floor = 1e+8 variance along those directions).
    Returns (cov, n_floored).
    """
    H_sym = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H_sym)
    keep = w >= eig_floor
    w_inv = np.zeros_like(w)
    w_inv[keep] = 1.0 / w[keep]
    cov = V @ np.diag(w_inv) @ V.T
    return cov, int(np.sum(~keep))


def _beta_and_se(p_idx, p_hat, cov):
    """Display value and SE for beta_i (i in {0, 1}) by delta method."""
    p = p_hat[p_idx]
    s = _sigmoid(p)
    beta = 10.0 * s
    # d beta / d p = -10 * s * (1 - s)
    dbeta_dp = -10.0 * s * (1.0 - s)
    var_p = cov[p_idx, p_idx]
    se_beta = abs(dbeta_dp) * np.sqrt(max(var_p, 0.0))
    return beta, se_beta


def collect():
    """Return list of dicts with keys vintage, beta1, se1, beta2, se2.

    SEs are computed from the well-identified eigendirections of the
    Hessian; floored directions (e.g. phi_1 boundary on 2018) are
    projected out before reading the cov diagonal.
    """
    out = []
    for v in lvp.VINTAGES:
        pfile = os.path.join(RESULTS_DIR, f'parameters_jrssa_{v}.npy')
        hfile = os.path.join(RESULTS_DIR, f'hessian_jrssa_{v}_4h.npy')
        if not (os.path.exists(pfile) and os.path.exists(hfile)):
            print(f'  skipping {v}: missing parameter or hessian file')
            continue
        p_hat = np.load(pfile)
        H = np.load(hfile)
        cov, n_floored = _well_identified_cov(H)
        if n_floored:
            print(f'  vintage {v}: projected out {n_floored} '
                  f'floored eigendirection(s)')
        b1, se1 = _beta_and_se(0, p_hat, cov)
        b2, se2 = _beta_and_se(1, p_hat, cov)
        out.append(dict(vintage=v, beta1=b1, se1=se1, beta2=b2, se2=se2))
    return out


def main():
    rows = collect()
    if not rows:
        print('No vintages available; run jrssa_estimate_all_vintages.py '
              'and jrssa_compute_hessian_all_vintages.py first.')
        return

    print(f'\n{"vintage":>8} {"beta1":>8} {"SE":>7}  {"beta2":>8} {"SE":>7}')
    for r in rows:
        print(f'{r["vintage"]:>8} {r["beta1"]:>8.3f} {r["se1"]:>7.3f}  '
              f'{r["beta2"]:>8.3f} {r["se2"]:>7.3f}')

    mpl.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'font.family': 'serif',
    })
    xs = [int(r['vintage']) for r in rows]
    b1 = np.array([r['beta1'] for r in rows])
    s1 = np.array([r['se1'] for r in rows])
    b2 = np.array([r['beta2'] for r in rows])
    s2 = np.array([r['se2'] for r in rows])
    colors = ['#1f4068' if v != 2025 else '#b32b1f' for v in xs]
    z = 1.96

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.errorbar(xs, b1, yerr=z * s1, fmt='none', ecolor='#888888',
                elinewidth=2.0, capsize=4, capthick=1.4, zorder=2)
    ax.scatter(xs, b1, c=colors, s=42, zorder=3,
               edgecolors='black', linewidths=0.6)
    ax.set_xticks(xs)
    ax.set_xlabel('GCB vintage')
    ax.set_ylabel(r'$\beta_1$  (land fertilisation slope, GtC per unit response)')
    ax.set_title('(a) Land fertilisation slope')
    ax.grid(True, axis='y', alpha=0.25)

    ax = axes[1]
    ax.errorbar(xs, b2, yerr=z * s2, fmt='none', ecolor='#888888',
                elinewidth=2.0, capsize=4, capthick=1.4, zorder=2)
    ax.scatter(xs, b2, c=colors, s=42, zorder=3,
               edgecolors='black', linewidths=0.6)
    ax.set_xticks(xs)
    ax.set_xlabel('GCB vintage')
    ax.set_ylabel(r'$\beta_2$  (ocean fertilisation slope)')
    ax.set_title('(b) Ocean fertilisation slope')
    ax.grid(True, axis='y', alpha=0.25)

    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    pdf_path = os.path.join(FIGDIR, 'figure_jrssa_beta_vintages.pdf')
    png_path = os.path.join(FIGDIR, 'figure_jrssa_beta_vintages.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=180)
    print(f'\nWrote {pdf_path}')
    print(f'Wrote {png_path}')


if __name__ == '__main__':
    main()
