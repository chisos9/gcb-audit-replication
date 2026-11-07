"""
Step 1 of the multi-vintage panel audit.

Reads the "Global Carbon Budget" sheet from every GCB vintage in data/,
extracts the published budget-imbalance column (BIM_raw), computes its
sample mean and two 95% confidence intervals (Normal i.i.d. and AR(1)-
inflated SE) for each vintage, and writes:

  output/figures/figure_bim_vintage_means.pdf  -- 1x2 figure
  output/csv/bim_vintage_means.csv               -- vintage-by-sample stats
  output/csv/bim_vintage_panel.csv               -- wide BIM time series

The AR(1) SE inflation uses long-run variance s^2 * (1+rho)/(1-rho),
where rho is the lag-1 autocorrelation of the demeaned BIM series. This
is the parametric AR(1) analogue of the Newey-West / HAC adjustment for
serial correlation.

Run from code/:
    python bim_vintage_means.py
"""
import os
import csv
import warnings
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import matplotlib as mpl

warnings.filterwarnings('ignore')

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.abspath(os.path.join(HERE, os.pardir))
DATA   = os.path.join(ROOT, 'data')
FIGDIR = os.path.join(ROOT, 'output', 'figures')

# (file, header_row_1indexed, year_col_1indexed, bim_col_1indexed)
CONFIGS = [
    ('CarbonBudget_2017-Data-Global-1.xlsx', 20, 1, 7),
    ('CarbonBudget_2018-Global-1.xlsx',      20, 1, 7),
    ('CarbonBudget_2019-Data-Global-1.xlsx', 19, 2, 8),
    ('CarbonBudget_2020-Data-Global-1.xlsx', 21, 1, 8),
    ('CarbonBudget_2021-Data-Global-1.xlsx', 21, 1, 8),
    ('Global_Carbon_Budget_2022v1.0.xlsx',   21, 1, 8),
    ('Global_Carbon_Budget_2023v1.0.xlsx',   22, 1, 8),
    ('Global_Carbon_Budget_2024v1.0.xlsx',   22, 1, 8),
    ('Global_Carbon_Budget_2025_v0.7.xlsx',  22, 1, 8),
]
LABELS = ['2017', '2018', '2019', '2020', '2021',
          '2022', '2023', '2024', '2025']


def read_bim_series():
    """Return {vintage_label: (years_array, bim_array)} pulled from each
    vintage's published Global Carbon Budget sheet."""
    out = {}
    for (fname, hr, yc, bc), lab in zip(CONFIGS, LABELS):
        wb = openpyxl.load_workbook(os.path.join(DATA, fname),
                                    read_only=True, data_only=True)
        ws = wb['Global Carbon Budget']
        years, bims = [], []
        for row in ws.iter_rows(min_row=hr + 1, values_only=True):
            y = row[yc - 1]
            b = row[bc - 1]
            if isinstance(y, (int, float)) and isinstance(b, (int, float)):
                years.append(int(y))
                bims.append(float(b))
        wb.close()
        out[lab] = (np.array(years), np.array(bims))
    return out


def stats(x):
    """Mean, sd, lag-1 rho, Normal SE, AR(1)-inflated SE."""
    T = len(x)
    m = x.mean()
    s = x.std(ddof=1)
    e = x - m
    rho = np.sum(e[:-1] * e[1:]) / np.sum(e ** 2)
    se_n = s / np.sqrt(T)
    se_ar = se_n * np.sqrt(max((1 + rho) / (1 - rho), 1e-6))
    return dict(T=T, mean=m, sd=s, rho=rho, se_normal=se_n, se_ar1=se_ar)


def main():
    series = read_bim_series()

    # Three samples
    specs = [
        ('common_1959_2016', 1959, 2016),
        ('common_1959_2023', 1959, 2023),
        ('vintage_full',     None, None),
    ]
    table = {}
    for sname, lo, hi in specs:
        table[sname] = {}
        for lab, (yrs, bs) in series.items():
            if sname == 'vintage_full':
                x = bs
                lo2, hi2 = int(yrs.min()), int(yrs.max())
            else:
                mask = (yrs >= lo) & (yrs <= hi)
                x = bs[mask]
                lo2, hi2 = lo, hi
                if len(x) < 5:
                    continue
            st = stats(x)
            st['lo'] = lo2
            st['hi'] = hi2
            table[sname][lab] = st

    # Stats CSV
    fpath = os.path.join(ROOT, 'output', 'csv', 'bim_vintage_means.csv')
    with open(fpath, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sample', 'vintage', 'lo', 'hi', 'T',
                    'mean', 'sd', 'rho', 'se_normal', 'se_ar1'])
        for sname, by in table.items():
            for lab, st in by.items():
                w.writerow([sname, lab, st['lo'], st['hi'], st['T'],
                            '{:.6f}'.format(st['mean']),
                            '{:.6f}'.format(st['sd']),
                            '{:.6f}'.format(st['rho']),
                            '{:.6f}'.format(st['se_normal']),
                            '{:.6f}'.format(st['se_ar1'])])
    print('Wrote', fpath)

    # Wide panel CSV
    all_years = sorted(
        set().union(*[set(int(y) for y in yrs) for yrs, _ in series.values()])
    )
    fpath = os.path.join(ROOT, 'output', 'csv', 'bim_vintage_panel.csv')
    with open(fpath, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['year'] + LABELS)
        for y in all_years:
            row = [y]
            for lab in LABELS:
                yrs, bs = series[lab]
                idx = np.where(yrs == y)[0]
                row.append('{:.6f}'.format(bs[idx[0]]) if len(idx) else '')
            w.writerow(row)
    print('Wrote', fpath)

    # Figure
    mpl.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'font.family': 'serif',
    })
    vintage_int = [int(l) for l in LABELS]

    def collect(sample_key):
        return [table[sample_key][lab] for lab in LABELS]

    common = collect('common_1959_2016')
    full = collect('vintage_full')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    def draw(ax, data, title):
        xs = vintage_int
        m = np.array([d['mean'] for d in data])
        se_n = np.array([d['se_normal'] for d in data])
        se_ar = np.array([d['se_ar1'] for d in data])
        z = 1.96
        ax.errorbar(xs, m, yerr=z * se_ar, fmt='none', ecolor='#999999',
                    elinewidth=1.8, capsize=6, capthick=1.2,
                    label='95% CI (AR(1))', zorder=2)
        ax.errorbar(xs, m, yerr=z * se_n, fmt='none', ecolor='#2b5d8c',
                    elinewidth=2.6, capsize=4, capthick=1.6,
                    label='95% CI (Normal i.i.d.)', zorder=3)
        colors = ['#1f4068'] * 8 + ['#b32b1f']
        ax.scatter(xs, m, c=colors, s=42, zorder=4,
                   edgecolors='black', linewidths=0.6)
        ax.axhline(0, color='black', lw=0.6, ls='--', alpha=0.6, zorder=1)
        ax.set_xticks(xs)
        ax.set_xlabel('GCB vintage')
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.25)
        ax.set_xlim(2016.4, 2025.6)

    axes[0].set_ylabel(r'$\overline{\mathrm{BIM}}^{\,\mathrm{raw}}$  (GtC/yr)')
    draw(axes[0], common, '(a) Common sample 1959--2016 ($T=58$ each)')
    draw(axes[1], full,   '(b) Each vintage on its own full sample 1959--$v{-}1$')

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(FIGDIR, exist_ok=True)
    plt.savefig(os.path.join(FIGDIR, 'figure_bim_vintage_means.pdf'),
                bbox_inches='tight')
    plt.savefig(os.path.join(FIGDIR, 'figure_bim_vintage_means.png'),
                bbox_inches='tight', dpi=180)
    print('Wrote', os.path.join(FIGDIR, 'figure_bim_vintage_means.{pdf,png}'))


if __name__ == '__main__':
    main()
