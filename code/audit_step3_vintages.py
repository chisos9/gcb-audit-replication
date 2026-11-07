"""
Step 3 of the multi-vintage panel audit: per-panel gaps.

For each vintage and each underlying panel, report the
  GCB-preferred-reference minus panel-mean gap

on the common 1959-2016 sample. Panels:
  - BK ensemble (LUC reference panel; gap = 0 by construction since
    GCB-published E_LUC IS the BK mean for every vintage; we still
    report it for completeness)
  - DGVM LUC ensemble (alternative LUC panel)
  - DGVM Terrestrial Sink panel (land sink)
  - Ocean ensemble (GOBMs + fCO2 products)

We follow the convention used in the 2024-vs-2025 Section 4.3 of
section4_audit_panel.tex: report (GCB-pref - panel mean), so a
negative entry means the GCB-preferred reference sits BELOW the
panel mean.

Outputs:
  output/csv/audit_step3_vintages.csv
  output/tables/table_step3_vintages.tex
"""
import os
import csv
import numpy as np
import load_vintage_panels as lvp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
TEX_OUT = os.path.join(ROOT, 'output', 'tables', 'table_step3_vintages.tex')

COMMON_LO, COMMON_HI = 1959, 2016


def _panel_mean(model_dict):
    if not model_dict:
        return None
    M = np.column_stack(list(model_dict.values()))
    return np.nanmean(M, axis=1)


def run_vintage(v):
    g = lvp.load_gcb_aggregate(v)
    yL, gcb_L, _, mL = lvp.load_lnd_panel(v)
    yO, gcb_O, _, mO = lvp.load_ocn_panel(v)
    luc = lvp.load_luc_panel(v)

    yrs = g['year']
    mask = (yrs >= COMMON_LO) & (yrs <= COMMON_HI)

    # BK ensemble mean
    bk_mean = _panel_mean(luc['bk'])
    # DGVM LUC ensemble mean
    dgvm_luc_mean = _panel_mean(luc['dgvm'])
    # DGVM Terrestrial Sink mean
    lnd_mean = _panel_mean(mL)
    # Ocean ensemble mean
    ocn_mean = _panel_mean(mO)

    # GCB-pref minus panel mean
    bk_gap        = float(np.nanmean(g['E_LUC'][mask] - bk_mean[mask]))
    dgvm_luc_gap  = float(np.nanmean(g['E_LUC'][mask] - dgvm_luc_mean[mask]))
    lnd_gap       = float(np.nanmean(gcb_L[mask] - lnd_mean[mask]))
    ocn_gap       = float(np.nanmean(gcb_O[mask] - ocn_mean[mask]))

    # Also report panel sample means themselves
    return dict(
        vintage=v,
        bk_mean   = float(np.nanmean(bk_mean[mask])),
        dgvm_luc_mean = float(np.nanmean(dgvm_luc_mean[mask])),
        lnd_mean  = float(np.nanmean(lnd_mean[mask])),
        ocn_mean  = float(np.nanmean(ocn_mean[mask])),
        bk_gap    = bk_gap,
        dgvm_luc_gap = dgvm_luc_gap,
        lnd_gap   = lnd_gap,
        ocn_gap   = ocn_gap,
    )


def main():
    rows = [run_vintage(v) for v in lvp.VINTAGES]

    print(f'Step 3 panel-vs-GCB gaps on common {COMMON_LO}-{COMMON_HI} sample')
    print(f'{"vintage":>7} {"BK_gap":>8} {"DGVM_LUC":>9} {"LND_gap":>8} {"OCN_gap":>8}')
    for r in rows:
        print(f'{r["vintage"]:>7} {r["bk_gap"]:>+8.3f} '
              f'{r["dgvm_luc_gap"]:>+9.3f} '
              f'{r["lnd_gap"]:>+8.3f} {r["ocn_gap"]:>+8.3f}')

    print('\nPanel sample means (GtC/yr), common sample')
    print(f'{"vintage":>7} {"BK":>7} {"DGVM_LUC":>9} {"DGVM_LND":>9} {"Ocean":>7}')
    for r in rows:
        print(f'{r["vintage"]:>7} {r["bk_mean"]:>7.3f} '
              f'{r["dgvm_luc_mean"]:>9.3f} '
              f'{r["lnd_mean"]:>9.3f} {r["ocn_mean"]:>7.3f}')

    csv_path = os.path.join(ROOT, 'output', 'csv', 'audit_step3_vintages.csv')
    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['vintage',
                    'bk_mean', 'dgvm_luc_mean', 'lnd_mean', 'ocn_mean',
                    'bk_gap', 'dgvm_luc_gap', 'lnd_gap', 'ocn_gap'])
        for r in rows:
            w.writerow([r['vintage'],
                        f'{r["bk_mean"]:.6f}',
                        f'{r["dgvm_luc_mean"]:.6f}',
                        f'{r["lnd_mean"]:.6f}',
                        f'{r["ocn_mean"]:.6f}',
                        f'{r["bk_gap"]:.6f}',
                        f'{r["dgvm_luc_gap"]:.6f}',
                        f'{r["lnd_gap"]:.6f}',
                        f'{r["ocn_gap"]:.6f}'])
    print(f'\nWrote {csv_path}')

    # LaTeX fragment
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'  \centering')
    lines.append(r'  \caption{Step 3: GCB value minus ensemble sample '
                 r'mean across vintages, in GtC/yr, common $1959$--$2016$ sample. '
                 r'Negative entries mean the GCB value sits below the '
                 r'ensemble mean. The DGVM-LUC column compares the GCB '
                 r'$E^{LUC}$ against the DGVM LUC ensemble (alternative ensemble).}')
    lines.append(r'  \label{tab:audit_step3_vintages}')
    lines.append(r'  \begin{tabular}{lrrr}')
    lines.append(r'    \toprule')
    lines.append(r'    Vintage & DGVM LUC (alt) '
                 r'& DGVM $S_{\mathrm{LND}}$ & Ocean \\')
    lines.append(r'    \midrule')
    for r in rows:
        if r['vintage'] == '2025':
            lines.append(
                f'    \\bfseries{{}}{r["vintage"]} & '
                f'$\\mathbf{{{r["dgvm_luc_gap"]:+.3f}}}$ & '
                f'$\\mathbf{{{r["lnd_gap"]:+.3f}}}$ & '
                f'$\\mathbf{{{r["ocn_gap"]:+.3f}}}$ \\\\'
            )
        else:
            lines.append(
                f'    {r["vintage"]} & '
                f'${r["dgvm_luc_gap"]:+.3f}$ & '
                f'${r["lnd_gap"]:+.3f}$ & ${r["ocn_gap"]:+.3f}$ \\\\'
            )
    lines.append(r'    \bottomrule')
    lines.append(r'  \end{tabular}')
    lines.append(r'\end{table}')

    with open(TEX_OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Wrote {TEX_OUT}')


if __name__ == '__main__':
    main()
