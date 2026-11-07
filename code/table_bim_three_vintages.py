"""
Table 1: budget-imbalance mean and trend for GCB 2024, the consolidated
reanalysis of Friedlingstein et al. (2026) and GCB 2025.

The consolidated series is GCB 2024 plus the deposited corrections to
E_LUC, S_LND and S_OCN (Zenodo 10.5281/zenodo.16367993). The deposit omits
the lateral-carbon-export term.

Writes output/csv/bim_three_vintages_summary.csv and
output/tables/table_bim_three_vintages.tex.
"""
import os
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, 'data')


def stats_block(year, x):
    """Mean with Newey-West p-value; trend per decade with OLS p-value."""
    year = np.asarray(year, float)
    x = np.asarray(x, float)
    T = len(x)
    L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    m_mean = sm.OLS(x, np.ones((T, 1))).fit(cov_type='HAC', cov_kwds={'maxlags': L})
    m_trend = sm.OLS(x, sm.add_constant(year - year.mean())).fit()
    return m_mean.params[0], m_mean.pvalues[0], m_trend.params[1] * 10, m_trend.pvalues[1]


def load_series():
    g24 = pd.read_csv(os.path.join(DATA, 'Global_Carbon_Budget_2024_v1.0.csv'))
    g24.columns = [c.strip() for c in g24.columns]
    g24 = g24.rename(columns={'budget imbalance': 'BIM'})[['Year', 'BIM']].query('1959 <= Year <= 2023')

    adj = pd.read_csv(os.path.join(DATA, 'GCB_adjustments.csv'))
    adj.columns = [c.strip() for c in adj.columns]
    adj = adj.rename(columns={'sland_bias \u2013 mean': 'sland_bias',
                              'ELUC_bias \u2013 Dorgeist': 'eluc_bias',
                              'SOCEAN_Bias_Judith': 'socean_bias'})
    adj = adj[['Year', 'sland_bias', 'eluc_bias', 'socean_bias']]
    adj = adj[pd.to_numeric(adj['Year'], errors='coerce').notna()]
    adj['Year'] = adj['Year'].astype(int)
    for c in ['sland_bias', 'eluc_bias', 'socean_bias']:
        adj[c] = pd.to_numeric(adj[c], errors='coerce')
    nat = g24.merge(adj, on='Year')
    nat['BIM'] = nat['BIM'] + nat['eluc_bias'] + nat['sland_bias'] - nat['socean_bias']
    nat = nat[['Year', 'BIM']]

    g25 = pd.read_excel(os.path.join(DATA, 'Global_Carbon_Budget_2025_v0.7.xlsx'),
                        sheet_name='Global Carbon Budget', header=21)
    g25 = g25.rename(columns={'budget imbalance': 'BIM'})[['Year', 'BIM']].dropna()
    g25 = g25[pd.to_numeric(g25['Year'], errors='coerce').notna()]
    g25['Year'] = g25['Year'].astype(int)
    g25 = g25.query('1959 <= Year <= 2024')
    return [('GCB 2024', g24), ('Reanalysis', nat), ('GCB 2025', g25)]


def fmt_p(p, digits):
    return '<0.001' if p < 0.001 else f'{p:.{digits}f}'


def main():
    recs, lines = [], []
    pdig = {'GCB 2024': (2, 3), 'Reanalysis': (2, 2), 'GCB 2025': (2, 2)}
    for name, df in load_series():
        mean, mean_p, trend, trend_p = stats_block(df.Year, df.BIM)
        end = int(df.Year.max())
        last10 = df[df.Year >= end - 9].BIM.mean()
        recs.append(dict(series=name, sample_end=end, full_mean=mean, full_mean_p=mean_p,
                         last_decade_mean=last10, trend_per_decade=trend, trend_p=trend_p))
        d1, d2 = pdig[name]
        lines.append(f'    {name} & ${mean:+.2f}$ $({fmt_p(mean_p, d1)})$ & ${last10:+.2f}$ '
                     f'& ${trend:+.2f}$ $({fmt_p(trend_p, d2)})$ \\\\')
        print(lines[-1])
    os.makedirs(os.path.join(ROOT, 'output', 'csv'), exist_ok=True)
    pd.DataFrame(recs).to_csv(os.path.join(ROOT, 'output', 'csv', 'bim_three_vintages_summary.csv'), index=False)
    with open(os.path.join(ROOT, 'output', 'tables', 'table_bim_three_vintages.tex'), 'w') as fh:
        fh.write('  \\begin{tabular}{llll}\n    \\hline\n'
                 '     & Full-sample mean & Last decade mean & Trend / decade \\\\\n    \\hline\n')
        fh.write('\n'.join(lines) + '\n    \\hline\n  \\end{tabular}\n')


if __name__ == '__main__':
    main()
