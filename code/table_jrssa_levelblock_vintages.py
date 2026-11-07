"""
Level-block parameter table for the multi-vintage model-based audit.

For every vintage 2017-2025 with parameters_jrssa_<v>.npy and
cov_jrssa_<v>_4h.npy on disk, emits a LaTeX long table with rows
indexed by vintage and columns
    beta_1, beta_2, c_1, c_2, d
each followed by its standard error.

Standard errors:
  - beta_1, beta_2: delta method from the 20x20 Hessian inverse.
  - c_1, c_2, d:    diagonal of the smoothed_state_cov at the last
                     time step (constant states; their smoother variance
                     concentrates at the last observation).

The 2025 row is bolded.

Output:
  output/tables/table_jrssa_levelblock_vintages.tex
"""
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
ROOT = os.path.abspath(os.path.join(CODE_DIR, os.pardir))
RESULTS_DIR = os.path.join(ROOT, 'results')
TEX_OUT = os.path.join(ROOT, 'output', 'tables',
                       'table_jrssa_levelblock_vintages.tex')

import load_vintage_panels as lvp


def _sigmoid(x):
    return np.exp(-x) / (1.0 + np.exp(-x))


def _beta_se(p_idx, p_hat, cov):
    p = p_hat[p_idx]
    s = _sigmoid(p)
    beta = 10.0 * s
    dbeta_dp = -10.0 * s * (1.0 - s)
    se_beta = abs(dbeta_dp) * np.sqrt(max(cov[p_idx, p_idx], 0.0))
    return beta, se_beta


def collect():
    rows = []
    for v in lvp.VINTAGES:
        pfile = os.path.join(RESULTS_DIR, f'parameters_jrssa_{v}.npy')
        cfile = os.path.join(RESULTS_DIR, f'cov_jrssa_{v}_4h.npy')
        rfile = os.path.join(RESULTS_DIR, f'jrssa_{v}_results.npz')
        if not all(os.path.exists(p) for p in (pfile, cfile, rfile)):
            print(f'  skipping {v}: missing one of '
                  f'parameters/cov/results files')
            continue
        p_hat = np.load(pfile)
        cov = np.load(cfile)
        r = np.load(rfile)
        beta1, se1 = _beta_se(0, p_hat, cov)
        beta2, se2 = _beta_se(1, p_hat, cov)
        # Smoothed state at last time index: states 9 (c1), 10 (c2), 11 (d).
        # NB: estimation script saves smoothed_state as (T, cr), so the
        # last-time entry is st[-1, j], not st[j, -1].
        # smoothed_state_cov is saved as (cr, cr, T), so Vt[j, j, -1] is correct.
        st = r['smoothed_state']          # (T, cr)
        Vt = r['smoothed_state_cov']      # (cr, cr, T)
        c1 = float(st[-1, 9])
        c2 = float(st[-1, 10])
        d  = float(st[-1, 11])
        se_c1 = float(np.sqrt(max(Vt[9, 9, -1], 0.0)))
        se_c2 = float(np.sqrt(max(Vt[10, 10, -1], 0.0)))
        se_d  = float(np.sqrt(max(Vt[11, 11, -1], 0.0)))
        rows.append(dict(vintage=v,
                          beta1=beta1, se_beta1=se1,
                          beta2=beta2, se_beta2=se2,
                          c1=c1, se_c1=se_c1,
                          c2=c2, se_c2=se_c2,
                          d=d, se_d=se_d))
    return rows


def main():
    rows = collect()
    if not rows:
        print('No vintages available.')
        return

    print(f'{"vintage":>8}  {"beta1":>9} {"SE":>7}  {"beta2":>9} {"SE":>7}  '
          f'{"c1":>9} {"SE":>7}  {"c2":>9} {"SE":>7}  {"d":>9} {"SE":>7}')
    for r in rows:
        print(f'{r["vintage"]:>8}  '
              f'{r["beta1"]:>+9.3f} {r["se_beta1"]:>7.3f}  '
              f'{r["beta2"]:>+9.3f} {r["se_beta2"]:>7.3f}  '
              f'{r["c1"]:>+9.3f} {r["se_c1"]:>7.3f}  '
              f'{r["c2"]:>+9.3f} {r["se_c2"]:>7.3f}  '
              f'{r["d"]:>+9.3f} {r["se_d"]:>7.3f}')

    # LaTeX
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'  \centering')
    lines.append(r'  \footnotesize')
    lines.append(
        r'  \caption{Level-block parameters of the MDS-GCB model across '
        r'GCB vintages 2017--2025, on the common 1959--2016 sample. '
        r'$\beta_1$ and $\beta_2$ are in display units '
        r'($10 \times \mathrm{sigmoid}(-p)$); standard errors via '
        r'the delta method from the regularised $\hat H^{-1}$. '
        r'The constants $c_1$, $c_2$, $d$ are the smoothed values at '
        r'the last observation; standard errors are the diagonal of the '
        r'smoothed-state covariance at that time. The 2025 row is bolded.}'
    )
    lines.append(r'  \label{tab:jrssa_levelblock_vintages}')
    lines.append(r'  \begin{tabular}{lcccccccccc}')
    lines.append(r'    \toprule')
    lines.append(
        r'     & \multicolumn{2}{c}{$\beta_1$} '
        r'& \multicolumn{2}{c}{$\beta_2$} '
        r'& \multicolumn{2}{c}{$c_1$} '
        r'& \multicolumn{2}{c}{$c_2$} '
        r'& \multicolumn{2}{c}{$d$} \\'
    )
    lines.append(r'    \cmidrule(lr){2-3} \cmidrule(lr){4-5} '
                 r'\cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}')
    lines.append(r'    Vintage & est & SE & est & SE & est & SE & est & SE '
                 r'& est & SE \\')
    lines.append(r'    \midrule')
    for r in rows:
        em_l, em_r = (r'\bfseries{}', r'') if r['vintage'] == '2025' else ('', '')
        lines.append(
            f'    {em_l}{r["vintage"]}{em_r} & '
            f'${r["beta1"]:+.3f}$ & ${r["se_beta1"]:.3f}$ & '
            f'${r["beta2"]:+.3f}$ & ${r["se_beta2"]:.3f}$ & '
            f'${r["c1"]:+.3f}$ & ${r["se_c1"]:.3f}$ & '
            f'${r["c2"]:+.3f}$ & ${r["se_c2"]:.3f}$ & '
            f'${r["d"]:+.3f}$ & ${r["se_d"]:.3f}$ \\\\'
        )
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    with open(TEX_OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'\nWrote {TEX_OUT}')


if __name__ == '__main__':
    main()
