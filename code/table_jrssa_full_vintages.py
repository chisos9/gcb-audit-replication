"""
Full-parameter Appendix A table for the multi-vintage model-based audit.

For every vintage 2017-2025 with parameters and Hessian-inverse files
on disk, emits a LaTeX long table with all 20 free parameters in
display units, with delta-method standard errors. Also includes the
three constants c_1, c_2, d at the bottom. Emitted in landscape on its
own page (wide 9-vintage layout).

Layout: parameter rows, vintage columns (9). One table per
parameter-group block (level/persistence/disturbance variances/climate
loadings) so each fits on a page.

Output:
  output/tables/appendixA_full_parameters_vintages.tex
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
                       'appendixA_full_parameters_vintages.tex')

import load_vintage_panels as lvp


def _sigmoid(x):
    return np.exp(-x) / (1.0 + np.exp(-x))


def _tanh_half(x):
    return (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))


def _delta_se(deriv_at_p, var_p):
    return abs(deriv_at_p) * np.sqrt(max(var_p, 0.0))


def _vintage_disp(p_hat, cov):
    """Compute display values and delta-method SEs for the 20 parameters."""
    s = _sigmoid
    th = _tanh_half

    # link derivatives at the MLE
    def d_sigmoid(p):     # d sigmoid / dp = -s(p)*(1 - s(p))
        sp = s(p)
        return -sp * (1.0 - sp)

    def d_tanh_half(p):   # d tanh(p/2) / dp = (1 - tanh^2)/2
        tp = th(p)
        return 0.5 * (1.0 - tp * tp)

    def d_10sigmoid(p):
        return 10.0 * d_sigmoid(p)

    def d_exp(p):
        return np.exp(p)

    # 0,1   : beta_1, beta_2  -> 10*sigmoid(p)
    # 2,3,4 : phi_1, phi_3, phiE  -> sigmoid(p)
    # 5..8  : sigma^2_eta_*       -> exp(p)
    # 9,10  : r_12, r_13          -> tanh(p/2)
    # 11    : sigma^2_eps_E       -> exp(p)
    # 12..19: climate loadings (unconstrained, display = raw)

    val = np.empty(20); se = np.empty(20)
    val[0] = 10.0 * s(p_hat[0]); se[0] = _delta_se(d_10sigmoid(p_hat[0]), cov[0,0])
    val[1] = 10.0 * s(p_hat[1]); se[1] = _delta_se(d_10sigmoid(p_hat[1]), cov[1,1])
    val[2] = s(p_hat[2]);  se[2] = _delta_se(d_sigmoid(p_hat[2]), cov[2,2])
    val[3] = s(p_hat[3]);  se[3] = _delta_se(d_sigmoid(p_hat[3]), cov[3,3])
    val[4] = s(p_hat[4]);  se[4] = _delta_se(d_sigmoid(p_hat[4]), cov[4,4])
    for i in (5,6,7,8):
        val[i] = np.exp(p_hat[i]); se[i] = _delta_se(d_exp(p_hat[i]), cov[i,i])
    val[9]  = th(p_hat[9]);  se[9]  = _delta_se(d_tanh_half(p_hat[9]),  cov[9,9])
    val[10] = th(p_hat[10]); se[10] = _delta_se(d_tanh_half(p_hat[10]), cov[10,10])
    val[11] = np.exp(p_hat[11]); se[11] = _delta_se(d_exp(p_hat[11]), cov[11,11])
    for i in range(12, 20):
        val[i] = float(p_hat[i]); se[i] = float(np.sqrt(max(cov[i,i], 0.0)))
    return val, se


PARAM_LABELS = [
    (r'$\beta_1$',       'level'),
    (r'$\beta_2$',       'level'),
    (r'$\phi_1$',        'pers'),
    (r'$\phi_3$',        'pers'),
    (r'$\phi_E$',        'pers'),
    (r'$\sigma^2_{\eta_1}$', 'var'),
    (r'$\sigma^2_{\eta_2}$', 'var'),
    (r'$\sigma^2_{\eta_3}$', 'var'),
    (r'$\sigma^2_{\eta_E}$', 'var'),
    (r'$r_{12}$',        'corr'),
    (r'$r_{13}$',        'corr'),
    (r'$\sigma^2_{\varepsilon_E}$', 'var'),
    (r'$b_1^{\mathrm{ENSO}}$', 'clim'),
    (r'$b_2^{\mathrm{ENSO}}$', 'clim'),
    (r'$b_1^{\mathrm{NAO}}$',  'clim'),
    (r'$b_2^{\mathrm{NAO}}$',  'clim'),
    (r'$b_1^{\mathrm{PDO}}$',  'clim'),
    (r'$b_2^{\mathrm{PDO}}$',  'clim'),
    (r'$b_1^{\mathrm{SAOD}}$', 'clim'),
    (r'$b_2^{\mathrm{SAOD}}$', 'clim'),
]


def collect():
    out = []
    for v in lvp.VINTAGES:
        pfile = os.path.join(RESULTS_DIR, f'parameters_jrssa_{v}.npy')
        cfile = os.path.join(RESULTS_DIR, f'cov_jrssa_{v}_4h.npy')
        rfile = os.path.join(RESULTS_DIR, f'jrssa_{v}_results.npz')
        if not all(os.path.exists(p) for p in (pfile, cfile, rfile)):
            print(f'  skipping {v}: missing parameter/cov/results file')
            continue
        p_hat = np.load(pfile)
        cov = np.load(cfile)
        r = np.load(rfile)
        val, se = _vintage_disp(p_hat, cov)
        # smoothed_state is saved as (T, cr); smoothed_state_cov as (cr, cr, T).
        st = r['smoothed_state']         # (T, cr)
        Vt = r['smoothed_state_cov']     # (cr, cr, T)
        c1, c2, d = float(st[-1, 9]), float(st[-1, 10]), float(st[-1, 11])
        se_c1 = float(np.sqrt(max(Vt[9, 9, -1], 0.0)))
        se_c2 = float(np.sqrt(max(Vt[10, 10, -1], 0.0)))
        se_d  = float(np.sqrt(max(Vt[11, 11, -1], 0.0)))
        out.append(dict(vintage=v, val=val, se=se,
                        c1=c1, c2=c2, d=d,
                        se_c1=se_c1, se_c2=se_c2, se_d=se_d))
    return out


def _fmt(val, se, n=3):
    return f'${val:+.{n}f}$ ({se:.{n}f})'


def main():
    rows = collect()
    if not rows:
        print('No vintages available.')
        return
    vintages = [r['vintage'] for r in rows]

    lines = []
    lines.append(r'\begin{landscape}')
    lines.append(r'\begin{table}[p]')
    lines.append(r'  \centering')
    lines.append(r'  \scriptsize')
    lines.append(
        r'  \caption{Full MDS-GCB parameter table across GCB vintages '
        r'2017--2025, on the common 1959--2016 sample. Display units '
        r'with delta-method standard errors in parentheses. The '
        r'constants $c_1, c_2, d$ in the last three rows are smoothed at '
        r'the final observation with standard errors from the smoothed-'
        r'state covariance.}'
    )
    lines.append(r'  \label{tab:jrssa_full_vintages}')
    cols_spec = 'l' + 'c' * len(vintages)
    lines.append(r'  \begin{tabular}{' + cols_spec + r'}')
    lines.append(r'    \toprule')
    lines.append(
        r'    Parameter & ' + ' & '.join(vintages) + r' \\'
    )
    lines.append(r'    \midrule')
    for i, (label, _) in enumerate(PARAM_LABELS):
        cells = [_fmt(r['val'][i], r['se'][i]) for r in rows]
        line = f'    {label} & ' + ' & '.join(cells) + r' \\'
        # bold the 2025 cell? skipped for simplicity in this wide table
        lines.append(line)
    lines.append(r'    \midrule')
    # concentrated constants
    cells_c1 = [_fmt(r['c1'], r['se_c1']) for r in rows]
    cells_c2 = [_fmt(r['c2'], r['se_c2']) for r in rows]
    cells_d  = [_fmt(r['d'],  r['se_d'])  for r in rows]
    lines.append(r'    $c_1$ & ' + ' & '.join(cells_c1) + r' \\')
    lines.append(r'    $c_2$ & ' + ' & '.join(cells_c2) + r' \\')
    lines.append(r'    $d$   & ' + ' & '.join(cells_d)  + r' \\')
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')
    lines.append(r'\end{landscape}')

    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    with open(TEX_OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Wrote {TEX_OUT}')


if __name__ == '__main__':
    main()
