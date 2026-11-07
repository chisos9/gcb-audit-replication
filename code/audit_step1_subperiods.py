"""
Step 1 (second view): BIM mean / RMSE / MAE by period and vintage.

Reads bim_vintage_panel.csv (the wide panel produced by
bim_vintage_means.py) and reports, for each vintage 2017-2025 on its
own sample 1959..(v-1), the mean, RMSE, and MAE of BIM_raw on the full
sample and on successive ten-year windows, most recent first. Because
each vintage ends in a different year the decade windows shift by one
year; the earliest block is the pre-decade remainder.

The loss functions (RMSE, MAE) measure the size of the imbalance
without the sign cancellation that affects the mean; cf. Bennedsen
(2021), who shows an AR(1) model is adequate for the BIM.

Outputs:
  output/tables/table_step1_subperiods.tex  (via output/tables/)
"""
import os
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
TEX_OUT = os.path.join(ROOT, 'output', 'tables',
                       'table_step1_subperiods.tex')

LABELS = ['2017', '2018', '2019', '2020', '2021',
          '2022', '2023', '2024', '2025']
BLK = ['Full sample', 'Last decade', 'Second-to-last', 'Third-to-last',
       'Fourth-to-last', 'Fifth-to-last', 'Sixth-to-last']


def metrics(x):
    x = x[~np.isnan(x)]
    return x.mean(), np.sqrt(np.mean(x ** 2)), np.mean(np.abs(x))


def periods(end):
    P = [('Full sample', 1959, end)]
    lo = end
    while lo - 9 >= 1959:
        P.append(('dec', lo - 9, lo)); lo -= 10
    if lo >= 1959:
        P.append(('earliest', 1959, lo))
    return P


def blocks(end):
    P = periods(end)
    d = {'Full sample': P[0][1:]}
    decs = [p for p in P if p[0] == 'dec']
    ear = [p for p in P if p[0] == 'earliest']
    for i, p in enumerate(decs):
        d[BLK[1 + i]] = (p[1], p[2])
    if ear:
        d['Earliest years'] = (ear[0][1], ear[0][2])
    return d


def main():
    csv_in = os.path.join(ROOT, 'output', 'csv', 'bim_vintage_panel.csv')
    rows = list(csv.reader(open(csv_in)))
    rows = rows[1:]
    years = np.array([int(r[0]) for r in rows])
    data = {lab: np.array([float(r[i + 1]) if r[i + 1] else np.nan
                           for r in rows]) for i, lab in enumerate(LABELS)}
    ends = {lab: int(years[~np.isnan(data[lab])].max()) for lab in LABELS}
    bmaps = {lab: blocks(ends[lab]) for lab in LABELS}

    def sub(lab, lo, hi):
        m = (years >= lo) & (years <= hi)
        return data[lab][m]

    def fmt(v, sign, nd=3):
        return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")

    print('Full-sample mean / RMSE / MAE by vintage (GtC/yr):')
    for lab in LABELS:
        mm, rr, aa = metrics(sub(lab, 1959, ends[lab]))
        print(f'  {lab}: {mm:>+7.3f}  RMSE {rr:.3f}  MAE {aa:.3f}')

    L = [r'\begin{table}[htbp]', r'  \centering', r'  \scriptsize',
         r'  \caption{Budget-imbalance mean, RMSE, and MAE of '
         r'$\BIM^{\mathrm{GCB}}$ (GtC/yr) by period for each GCB vintage '
         r'2017--2025. For each period the mean is $\overline{\BIM}$, '
         r'$\mathrm{RMSE}=\sqrt{\overline{\BIM^{2}}}$, and '
         r'$\mathrm{MAE}=\overline{|\BIM|}$; the loss functions measure the '
         r'size of the imbalance without the sign cancellation that affects '
         r'the mean. Each vintage $v$ is evaluated on its own sample '
         r'$1959$--$(v{-}1)$; periods run from the full sample down through '
         r'successive ten-year windows, most recent first, so the windows '
         r'shift by one year from vintage to vintage; the pre-decade '
         r'remainder before 1959 is not shown. A dash marks a decade a '
         r'shorter vintage does not reach. The 2025 column is in bold.}',
         r'  \label{tab:bim_subperiods_vintages}',
         r'  \begin{tabular}{ll' + 'r' * len(LABELS) + r'}',
         r'    \toprule',
         r'    Period & & '
         + ' & '.join((r'\textbf{' + l + r'}' if l == '2025' else l)
                      for l in LABELS) + r' \\',
         r'    \midrule']
    for blab in BLK:
        for mi, mname in enumerate(['mean', 'RMSE', 'MAE']):
            cells = []
            for lab in LABELS:
                bm = bmaps[lab]
                if blab in bm:
                    lo, hi = bm[blab]
                    mm, rr, aa = metrics(sub(lab, lo, hi))
                    val = {'mean': mm, 'RMSE': rr, 'MAE': aa}[mname]
                    s = fmt(val, sign=(mname == "mean"))
                    c = (r'$\mathbf{' + s + r'}$' if lab == '2025'
                         else f'${s}$')
                else:
                    c = r'\textbf{--}' if lab == '2025' else r'--'
                cells.append(c)
            first = rf'\multirow{{3}}{{*}}{{{blab}}}' if mi == 0 else ''
            L.append(r'    ' + first + ' & ' + mname + ' & '
                     + ' & '.join(cells) + r' \\')
        L.append(r'    \addlinespace')
    L += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']

    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    with open(TEX_OUT, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print(f'\nWrote {TEX_OUT}')


if __name__ == '__main__':
    main()
