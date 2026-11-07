"""
Model-free analogue of figure_jrssa_af_sr_vintages: airborne fraction
and sink rate computed directly from the GCB-published series, no
state-space model in between.

E_t is defined here as the net anthropogenic emissions
  E_t = E_FF_t + E_LUC_t - S_CARB_t,
so that cement-carbonation absorption is subsumed in the emissions
term and the budget identity reduces to
  BIM_t = E_t - G_ATM_t - S_LND_t - S_OCN_t.

Panel (a):  AF_t = G_ATM_t / E_t
Panel (b):  SR_t = (S_LND_t + S_OCN_t) / (C_t - C_1750)
Panel (c):  2025 only, two sink-rate definitions:
              SR^1_t = (S_LND_t + S_OCN_t)  / (C_t - C_1750)
              SR^2_t = (E_t - G_ATM_t)      / (C_t - C_1750)

Subtracting, SR^2 - SR^1 = BIM_t / (C_t - C_1750) exactly: with E_t
defined as net of cement carbonation, the gap between the two
sink-rate definitions equals the budget-imbalance term divided by
the post-1750 atmospheric stock, with no residual S_CARB term.

All series on the common 1959--2016 sample for consistency with the
other multi-vintage figures.

Output:
  output/figures/figure_panel_af_sr_vintages.{pdf,png}
"""
import os
import sys
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
ROOT = os.path.abspath(os.path.join(CODE_DIR, os.pardir))
FIGDIR = os.path.join(ROOT, 'output', 'figures')

import load_vintage_panels as lvp

COMMON_LO = 1959
COMMON_HI = 2016
cGtC1959 = 2.127 * 315.39
cGtC1750 = 2.127 * 279.0


def collect():
    """Per-vintage dict: years, AF, SR1, SR2."""
    out = []
    for v in lvp.VINTAGES:
        g = lvp.load_gcb_aggregate(v)
        yr = g['year']
        mask = (yr >= COMMON_LO) & (yr <= COMMON_HI)
        years = yr[mask]
        E_FF = g['E_FF'][mask]
        E_LUC = g['E_LUC'][mask]
        G_ATM = g['G_ATM'][mask]
        S_LND = g['S_LND'][mask]
        S_OCN = g['S_OCN'][mask]
        S_CARB = g['S_CARB'][mask]    # zero for 2017-2019 (pre-cement) vintages
        # Net anthropogenic emissions: cement-carbonation absorption
        # subsumed into the emissions term so that
        # SR^2 - SR^1 = BIM / (C - C_1750) exactly.
        E = E_FF + E_LUC - S_CARB
        C = cGtC1959 + np.cumsum(G_ATM)
        denom = C - cGtC1750
        AF = G_ATM / E
        SR1 = (S_LND + S_OCN) / denom
        SR2 = (E - G_ATM) / denom
        out.append(dict(vintage=v, years=years,
                        AF=AF, SR1=SR1, SR2=SR2))
    return out


def main():
    rows = collect()

    # 8 blue shades for 2017-2024, red for 2025
    blue_palette = plt.cm.Blues(np.linspace(0.35, 0.85, 8))
    color_map = {}
    bi = 0
    for r in rows:
        if r['vintage'] == '2025':
            color_map[r['vintage']] = '#b32b1f'
        else:
            color_map[r['vintage']] = blue_palette[bi]
            bi += 1

    mpl.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
        'font.family': 'serif',
    })

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # Panel (a): AF across vintages
    ax = axes[0]
    ordered = sorted(rows, key=lambda r: (r['vintage'] == '2025',
                                          int(r['vintage'])))
    for r in ordered:
        col = color_map[r['vintage']]
        is25 = r['vintage'] == '2025'
        ax.plot(r['years'], r['AF'], color=col,
                linewidth=1.0 if not is25 else 2.0,
                label=r['vintage'], zorder=3 if not is25 else 5)
    ax.set_title(r'(a) Airborne fraction  $G_{ATM}/E$,'
                  '\n'
                  r' $E=E^{FF}+E^{LUC}-S^{CARB}$')
    ax.set_xlabel('year')
    ax.set_ylabel(r'$\mathrm{AF}_t$')
    ax.grid(True, alpha=0.25)

    # Panel (b): SR1 across vintages
    ax = axes[1]
    for r in ordered:
        col = color_map[r['vintage']]
        is25 = r['vintage'] == '2025'
        ax.plot(r['years'], r['SR1'], color=col,
                linewidth=1.0 if not is25 else 2.0,
                label=r['vintage'], zorder=3 if not is25 else 5)
    ax.set_title(r'(b) Sink rate  $(S_{LND}+S_{OCN})/(C-C_{1750})$')
    ax.set_xlabel('year')
    ax.set_ylabel(r'$\mathrm{SR}_t$')
    ax.grid(True, alpha=0.25)

    # Panel (c): SR1 vs SR2 for 2025 only
    ax = axes[2]
    r25 = [r for r in rows if r['vintage'] == '2025'][0]
    ax.plot(r25['years'], r25['SR1'], color='#2b5d8c', linewidth=1.8,
            label=r'$\mathrm{SR}^{1}=(S_{LND}+S_{OCN})/(C-C_{1750})$')
    ax.plot(r25['years'], r25['SR2'], color='#b32b1f', linewidth=1.8,
            label=r'$\mathrm{SR}^{2}=(E-G_{ATM})/(C-C_{1750})$',
            linestyle='--')
    ax.set_title(r'(c) Two sink-rate definitions, 2025 vintage')
    ax.set_xlabel('year')
    ax.set_ylabel(r'$\mathrm{SR}_t$')
    ax.legend(loc='upper right', frameon=False)
    ax.grid(True, alpha=0.25)

    # Single shared legend for panels (a) and (b), at bottom
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=len(rows),
               frameon=False, bbox_to_anchor=(0.34, -0.05),
               handlelength=1.5, columnspacing=0.9,
               title='vintage')
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(FIGDIR, exist_ok=True)
    pdf_path = os.path.join(FIGDIR, 'figure_panel_af_sr_vintages.pdf')
    png_path = os.path.join(FIGDIR, 'figure_panel_af_sr_vintages.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=180)
    print(f'Wrote {pdf_path}')
    print(f'Wrote {png_path}')


if __name__ == '__main__':
    main()
