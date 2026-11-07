"""
Step 2 of the multi-vintage panel audit: identity decomposition

  BIM_raw  =  BIM_panel  -  gap_LUC  +  gap_S_LND  +  gap_S_OCN

where  gap_X = mean_panel(X) - X_GCB_pref  for X in {LUC, S_LND, S_OCN},
and    BIM_panel uses panel means in place of GCB-preferred references.

Reports sample means on the common 1959-2016 sample for each vintage,
and writes:
  output/csv/audit_step2_vintages.csv      -- one row per vintage
  output/tables/table_step2_vintages.tex   -- LaTeX fragment
"""
import os
import csv
import numpy as np

import load_vintage_panels as lvp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
TEX_OUT = os.path.join(ROOT, 'output', 'tables', 'table_step2_vintages.tex')

COMMON_LO, COMMON_HI = 1959, 2016


def _mean_panel(model_dict):
    if not model_dict:
        return None
    M = np.column_stack(list(model_dict.values()))
    return np.nanmean(M, axis=1)


def run_vintage(v):
    g = lvp.load_gcb_aggregate(v)
    yL, gcb_L, _, mL = lvp.load_lnd_panel(v)
    yO, gcb_O, _, mO = lvp.load_ocn_panel(v)
    luc = lvp.load_luc_panel(v)

    # align years across sheets (should all match)
    assert (g['year'] == yL).all(), v
    assert (g['year'] == yO).all(), v
    assert (g['year'] == luc['year']).all(), v
    yrs = g['year']

    panel_L = _mean_panel(mL)
    panel_O = _mean_panel(mO)
    panel_LUC = _mean_panel(luc['bk'])      # BK ensemble

    gap_L = panel_L - gcb_L
    gap_O = panel_O - gcb_O
    gap_LUC = panel_LUC - g['E_LUC']        # GCB-published E_LUC

    bim_panel = (g['E_FF'] + panel_LUC - g['G_ATM']
                 - panel_L - panel_O - g['S_CARB'])

    mask = (yrs >= COMMON_LO) & (yrs <= COMMON_HI)
    out = dict(
        vintage=v,
        T=int(mask.sum()),
        BIM_raw   = float(np.nanmean(g['BIM_raw'][mask])),
        BIM_panel = float(np.nanmean(bim_panel[mask])),
        gap_LUC   = float(np.nanmean(gap_LUC[mask])),
        gap_LND   = float(np.nanmean(gap_L[mask])),
        gap_OCN   = float(np.nanmean(gap_O[mask])),
    )
    # identity check: BIM_raw should equal BIM_panel - gap_LUC + gap_LND + gap_OCN
    out['identity_residual'] = (out['BIM_raw']
        - (out['BIM_panel'] - out['gap_LUC'] + out['gap_LND'] + out['gap_OCN']))
    return out


def main():
    rows = [run_vintage(v) for v in lvp.VINTAGES]

    # Print
    print(f'Step 2 identity decomposition on common {COMMON_LO}-{COMMON_HI} sample')
    print(f'{"vintage":>7} {"T":>3} {"BIM_raw":>9} {"BIM_panel":>11} '
          f'{"gap_LUC":>9} {"gap_LND":>9} {"gap_OCN":>9} {"resid":>9}')
    for r in rows:
        print(f'{r["vintage"]:>7} {r["T"]:>3} '
              f'{r["BIM_raw"]:>+9.3f} {r["BIM_panel"]:>+11.3f} '
              f'{r["gap_LUC"]:>+9.3f} {r["gap_LND"]:>+9.3f} '
              f'{r["gap_OCN"]:>+9.3f} {r["identity_residual"]:>+9.4f}')

    # CSV
    csv_path = os.path.join(ROOT, 'output', 'csv', 'audit_step2_vintages.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['vintage', 'T_common',
                    'BIM_raw', 'BIM_panel', 'gap_LUC', 'gap_LND', 'gap_OCN',
                    'identity_residual'])
        for r in rows:
            w.writerow([r['vintage'], r['T'],
                        f'{r["BIM_raw"]:.6f}',
                        f'{r["BIM_panel"]:.6f}',
                        f'{r["gap_LUC"]:.6f}',
                        f'{r["gap_LND"]:.6f}',
                        f'{r["gap_OCN"]:.6f}',
                        f'{r["identity_residual"]:.8f}'])
    print(f'\nWrote {csv_path}')

    # LaTeX fragment
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Step 2 identity decomposition across vintages. '
                 r'Sample means in GtC/yr on the common '
                 r'$1959$--$2016$ sample ($T=58$). '
                 r'Row identity: '
                 r'$\BIM^{\mathrm{GCB}} = \BIM^{\mathrm{ens}}'
                 r' + \mathrm{gap}_{S_{\mathrm{LND}}}'
                 r' + \mathrm{gap}_{S_{\mathrm{OCN}}}$.}')
    lines.append(r'  \label{tab:audit_step2_vintages}')
    lines.append(r'  \begin{tabular}{lrrrr}')
    lines.append(r'    \toprule')
    lines.append(r'    Vintage & $\BIM^{\mathrm{GCB}}$ & $\BIM^{\mathrm{ens}}$ '
                 r'& $\mathrm{gap}_{S_{\mathrm{LND}}}$ '
                 r'& $\mathrm{gap}_{S_{\mathrm{OCN}}}$ \\')
    lines.append(r'    \midrule')
    for r in rows:
        # mark 2025 row in bold (text + math cells)
        if r['vintage'] == '2025':
            lines.append(
                f'    \\bfseries{{}}{r["vintage"]} & '
                f'$\\mathbf{{{r["BIM_raw"]:+.3f}}}$ & $\\mathbf{{{r["BIM_panel"]:+.3f}}}$ & '
                f'$\\mathbf{{{r["gap_LND"]:+.3f}}}$ & $\\mathbf{{{r["gap_OCN"]:+.3f}}}$ \\\\'
            )
        else:
            lines.append(
                f'    {r["vintage"]} & '
                f'${r["BIM_raw"]:+.3f}$ & ${r["BIM_panel"]:+.3f}$ & '
                f'${r["gap_LND"]:+.3f}$ & ${r["gap_OCN"]:+.3f}$ \\\\'
            )
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    with open(TEX_OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Wrote {TEX_OUT}')


if __name__ == '__main__':
    main()
