"""
Table 4: Oaxaca-Blinder decomposition of the 2024 to 2025 transition.
Reads output/csv/audit_step4_appendixB.csv (written by
audit_step4_appendixB.py) and writes output/tables/table_ob_2024_2025.tex.
"""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
LABELS = [('BK', 'BK (LUC ensemble)'), ('DGVM_LUC', 'DGVM LUC'),
          ('DGVM_LND', r'DGVM $S_{\mathrm{LND}}$'), ('Ocean', 'Ocean')]


def main():
    df = pd.read_csv(os.path.join(ROOT, 'output', 'csv', 'audit_step4_appendixB.csv'))
    df = df[df.transition == '2024-2025'].set_index('panel')
    with open(os.path.join(ROOT, 'output', 'tables', 'table_ob_2024_2025.tex'), 'w') as fh:
        fh.write('  \\begin{tabular}{lrrrr}\n    \\hline\n    Ensemble & $\\Delta \\bar X$ & '
                 '$\\Delta_{\\mathrm{rev}}$ & $\\Delta_{\\mathrm{in}}$ & $-\\Delta_{\\mathrm{out}}$ \\\\\n    \\hline\n')
        for key, lab in LABELS:
            r = df.loc[key]
            # The CSV holds Delta_out; the table reports -Delta_out.
            vals = [r.delta_mean, r.delta_rev, r.delta_in, -r.delta_out]
            cells = ' & '.join('$%+.3f$' % (v + 0.0 if abs(v) >= 0.0005 else 0.0) for v in vals)
            line = f'    {lab} & {cells} \\\\'
            fh.write(line + '\n')
            print(line)
        fh.write('    \\hline\n  \\end{tabular}\n')


if __name__ == '__main__':
    main()
