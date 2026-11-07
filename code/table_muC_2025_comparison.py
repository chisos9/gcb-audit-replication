"""
Table 5: baseline MDS-GCB against the mu_C-augmented variant on the 2025
vintage, common 1959-2016 sample. Standard errors use the delta method on
the regularised inverse Hessian; those for c_1, c_2 are from the
smoothed-state covariance at the last observation.
Writes output/tables/table_muC_2025_comparison.tex.
"""
import os
import sys
import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
RES = os.path.join(ROOT, 'results')
from load_jrssa_vintage import load_vintage_for_jrssa


def _s(x):
    return np.exp(-x) / (1.0 + np.exp(-x))


def _ds(x):
    return abs(_s(x) * (1.0 - _s(x)))


def collect(tag, mObs):
    p = np.load(os.path.join(RES, f'parameters_jrssa_{tag}2025.npy'))
    cov = np.load(os.path.join(RES, f'cov_jrssa_{tag}2025_4h.npy'))
    r = np.load(os.path.join(RES, f'jrssa_{tag}2025_results.npz'), allow_pickle=True)
    st, V = r['smoothed_state'], r['smoothed_state_cov']
    out = dict(ll=float(r['loglik']))
    out['b1'] = (10 * _s(p[0]), 10 * _ds(p[0]) * np.sqrt(cov[0, 0]))
    out['b2'] = (10 * _s(p[1]), 10 * _ds(p[1]) * np.sqrt(cov[1, 1]))
    out['phi3'] = (_s(p[3]), _ds(p[3]) * np.sqrt(cov[3, 3]))
    out['c1'] = (st[-1, 9], np.sqrt(V[9, 9, -1]))
    out['c2'] = (st[-1, 10], np.sqrt(V[10, 10, -1]))
    out['gap_lnd'] = np.mean(mObs[:, 2] - st[:, 2])
    out['gap_ocn'] = np.mean(mObs[:, 3] - st[:, 3])
    if len(p) == 21:
        out['mu'] = (p[20], np.sqrt(cov[20, 20]))
    return out


def main():
    obs = load_vintage_for_jrssa('2025', 1959, 2016)['mObs']
    b, m = collect('', obs), collect('muC_', obs)
    lr = 2 * (m['ll'] - b['ll'])
    print(f'LR = {lr:.2f}, p = {chi2.sf(lr, 1):.3g}')
    rows = [('log-likelihood', f"${b['ll']:.3f}$", f"${m['ll']:.3f}$")]
    for k, lab, d in [('b1', r'$\beta_1$', 3), ('b2', r'$\beta_2$', 3), ('phi3', r'$\phi_3$', 4),
                      ('c1', r'$c_1$', 4), ('c2', r'$c_2$', 4)]:
        rows.append((lab, f"${b[k][0]:.{d}f}$ $({b[k][1]:.{d}f})$", f"${m[k][0]:.{d}f}$ $({m[k][1]:.{d}f})$"))
    rows.append((r'$\mu_C$', 'n.a.', f"${m['mu'][0]:.4f}$ $({m['mu'][1]:.4f})$"))
    rows.append((r'mean$(S^{LND}_{obs} - S^{LND*})$', f"${b['gap_lnd']:+.4f}$", f"${m['gap_lnd']:+.4f}$"))
    rows.append((r'mean$(S^{OCN}_{obs} - S^{OCN*})$', f"${b['gap_ocn']:+.4f}$", f"${m['gap_ocn']:+.4f}$"))
    with open(os.path.join(ROOT, 'output', 'tables', 'table_muC_2025_comparison.tex'), 'w') as fh:
        fh.write('  \\begin{tabular}{lcc}\n    \\hline\n     & Baseline (20p) & With $\\mu_C$ (21p) \\\\\n    \\hline\n')
        for r in rows:
            line = '    ' + ' & '.join(r) + ' \\\\'
            fh.write(line + '\n')
            print(line)
        fh.write('    \\hline\n  \\end{tabular}\n')


if __name__ == '__main__':
    main()
