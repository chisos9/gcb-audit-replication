"""
Table 6: test of mu_C = 0 across GCB vintages. Reads
results/jrssa_muC_vintages.csv (written by
jrssa_estimate_muC_all_vintages.py) and writes
output/tables/table_muC_vintages.tex.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))


def main():
    df = pd.read_csv(os.path.join(ROOT, 'results', 'jrssa_muC_vintages.csv'))
    with open(os.path.join(ROOT, 'output', 'tables', 'table_muC_vintages.tex'), 'w') as fh:
        fh.write('  \\begin{tabular}{lrrrrrr}\n    \\hline\n    Vintage & $\\hat\\mu_C$ & (SE) & $LR$ & $P$ & '
                 '$\\hat\\beta_1$ & $\\hat\\beta_2$ \\\\\n    \\hline\n')
        for r in df.itertuples():
            p = '<0.001' if r.p_LR < 0.001 else f'{r.p_LR:.3f}'
            line = (f'    {r.vintage} & ${r.mu_C:+.3f}$ & $({r.se_mu_C:.3f})$ & ${r.LR:.2f}$ & '
                    f'${p}$ & ${r.beta1:.2f}$ & ${r.beta2:.2f}$ \\\\')
            fh.write(line + '\n')
            print(line)
        fh.write('    \\hline\n  \\end{tabular}\n')


if __name__ == '__main__':
    main()
