"""
Print per-member 1959-2016 sample means for the four panels (BK, DGVM
LUC, DGVM Terrestrial Sink, Ocean) on the 2024 and 2025 vintages.

Reproduces the individual-model figures cited in Section 4.4 of the
paper (e.g., VISIT 1.65 -> 0.10 GtC/yr for DGVM LUC, ELM-FATES 1.05
GtC/yr 1959-2016 mean for DGVM Terrestrial Sink, etc.).

Run:
    python member_level_means.py

This is a read-only diagnostic; no files are written.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from audit_step4_appendixB import load_all_panels, canonical, COMMON_LO, COMMON_HI

def mn(arr, yrs, lo=COMMON_LO, hi=COMMON_HI):
    m = (yrs >= lo) & (yrs <= hi)
    return float(np.nanmean(arr[m]))


def panel_means(panel_dict, years, label):
    print(f'\n=== {label} (common {COMMON_LO}-{COMMON_HI}) ===')
    rows = []
    for nm, arr in panel_dict.items():
        rows.append((mn(arr, years), nm))
    rows.sort()
    for v, n in rows:
        print(f'  {n:35s}  {v:+8.4f}')


def main():
    pan = load_all_panels()
    print('=' * 60)
    print('Per-member 1959-2016 sample means by panel and vintage')
    print('=' * 60)
    for v in ('2024', '2025'):
        for p, ykey in [('BK', 'year_LUC'),
                        ('DGVM_LUC', 'year_LUC'),
                        ('DGVM_LND', 'year_L'),
                        ('Ocean', 'year_O')]:
            panel_means(pan[v][p], pan[v][ykey], f'{v} {p}')

if __name__ == '__main__':
    main()
