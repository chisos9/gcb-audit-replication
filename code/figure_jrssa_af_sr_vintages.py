"""
Multi-vintage AF/SR overlay figure (headline visual for Section 5.2).

For each vintage v with jrssa_af_sr_bands_<v>.npz on disk:
  - draws the point-estimate AF and SR series as a line
  - draws the pointwise 99% band as a translucent ribbon

The 2025 vintage is highlighted in red; the 2017-2024 vintages share a
neutral blue palette so the reader sees at a glance whether 2025 is the
sole outlier.

Output:
  output/figures/figure_jrssa_af_sr_vintages.{pdf,png}
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
RESULTS_DIR = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(ROOT, 'output', 'figures')

import load_vintage_panels as lvp


def collect():
    out = []
    for v in lvp.VINTAGES:
        path = os.path.join(RESULTS_DIR, f'jrssa_af_sr_bands_{v}.npz')
        if not os.path.exists(path):
            print(f'  skipping {v}: missing {path}')
            continue
        d = np.load(path)
        out.append(dict(vintage=v,
                        years=d['years'],
                        af_pt=d['af_pt'], af_lo=d['af_lo'], af_hi=d['af_hi'],
                        sr_pt=d['sr_pt'], sr_lo=d['sr_lo'], sr_hi=d['sr_hi']))
    return out


def main():
    rows = collect()
    if not rows:
        print('No band files found; run jrssa_af_sr_vintages.py first.')
        return

    mpl.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 8,
        'font.family': 'serif',
    })

    # 8 blue shades for 2017-2024, red for 2025
    blue_palette = plt.cm.Blues(np.linspace(0.35, 0.85, 8))
    color_map = {}
    blue_idx = 0
    for r in rows:
        if r['vintage'] == '2025':
            color_map[r['vintage']] = '#b32b1f'
        else:
            color_map[r['vintage']] = blue_palette[blue_idx]
            blue_idx += 1

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    def draw_panel(ax, key_pt, key_lo, key_hi, title, ylabel):
        # Draw older vintages first (thin lines, light fill),
        # 2025 last (thick, red fill on top).
        ordered = sorted(rows, key=lambda r: (r['vintage'] == '2025',
                                              int(r['vintage'])))
        for r in ordered:
            col = color_map[r['vintage']]
            yrs = r['years']
            is25 = r['vintage'] == '2025'
            ax.fill_between(yrs, r[key_lo], r[key_hi],
                            facecolor=col, alpha=0.13 if not is25 else 0.22,
                            edgecolor='none', zorder=2 if not is25 else 4)
            ax.plot(yrs, r[key_pt], color=col,
                    linewidth=1.0 if not is25 else 2.0,
                    label=r['vintage'], zorder=3 if not is25 else 5)
        ax.set_title(title)
        ax.set_xlabel('year')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    draw_panel(axes[0], 'af_pt', 'af_lo', 'af_hi',
               '(a) Airborne fraction', r'$\mathrm{AF}_t$')
    draw_panel(axes[1], 'sr_pt', 'sr_lo', 'sr_hi',
               '(b) Sink rate', r'$\mathrm{SR}_t$')
    # legend in figure-level position
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=len(rows),
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               handlelength=1.5, columnspacing=0.9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    os.makedirs(FIGDIR, exist_ok=True)
    pdf_path = os.path.join(FIGDIR, 'figure_jrssa_af_sr_vintages.pdf')
    png_path = os.path.join(FIGDIR, 'figure_jrssa_af_sr_vintages.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, bbox_inches='tight', dpi=180)
    print(f'Wrote {pdf_path}')
    print(f'Wrote {png_path}')


if __name__ == '__main__':
    main()
