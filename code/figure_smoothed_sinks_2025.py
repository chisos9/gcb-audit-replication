"""
Diagnostic 2x2 panel: smoothed S^{LND*} and S^{OCN*} vs the GCB-preferred
published series for the 2025 vintage, under both the baseline JRSSAClimate
(top row, 20 parameters) and the mu_C-augmented JRSSAClimateMuC (bottom
row, 21 parameters).

Top-left  (a) : Land sink, baseline
Top-right (b) : Ocean sink, baseline
Bot-left  (c) : Land sink, with mu_C
Bot-right (d) : Ocean sink, with mu_C

The discrepancy mean(obs - smoothed) is printed in each panel title.
"""
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
ROOT = os.path.abspath(os.path.join(CODE_DIR, os.pardir))
FIGDIR = os.path.join(ROOT, 'output', 'figures')

from jrssa_m2_climate import JRSSAClimate
from jrssa_m2_climate_muC import JRSSAClimateMuC
from load_jrssa_vintage import load_vintage_for_jrssa


def main():
    d = load_vintage_for_jrssa('2025', 1959, 2016)
    SLND_obs = d['mObs'][:, 2]; SOCN_obs = d['mObs'][:, 3]
    years = d['years']

    mb = JRSSAClimate(d['mObs'], d['enso_shifted'], d['nao_shifted'],
                      d['pdo_shifted'], d['saod_shifted'])
    pb = np.load(os.path.join(ROOT, 'results', 'parameters_jrssa_2025.npy'))
    sb = mb.smooth(pb).smoothed_state

    mc = JRSSAClimateMuC(d['mObs'], d['enso_shifted'], d['nao_shifted'],
                         d['pdo_shifted'], d['saod_shifted'])
    pc = np.load(os.path.join(ROOT, 'results', 'parameters_jrssa_muC_2025.npy'))
    sc = mc.smooth(pc).smoothed_state

    mpl.rcParams.update({
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'font.family': 'serif',
    })

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

    def panel(ax, obs, smoothed, title):
        ax.plot(years, obs, color='#1f4068', linewidth=1.6,
                label='GCB-preferred', zorder=3)
        ax.plot(years, smoothed, color='#b32b1f', linewidth=1.6,
                label=r'smoothed $S^*$', zorder=4)
        ax.fill_between(years, obs, smoothed,
                        where=(smoothed >= obs),
                        facecolor='#b32b1f', alpha=0.15,
                        edgecolor='none', zorder=2)
        ax.fill_between(years, obs, smoothed,
                        where=(smoothed < obs),
                        facecolor='#1f4068', alpha=0.15,
                        edgecolor='none', zorder=2)
        gap = (obs - smoothed).mean()
        ax.set_title(f'{title}\n(mean(obs $-$ smoothed) $= {gap:+.3f}$)')
        ax.set_xlabel('year')
        ax.set_ylabel('GtC/yr')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper left', frameon=False)

    panel(axes[0, 0], SLND_obs, sb[2], '(a) Land sink, baseline (20p)')
    panel(axes[0, 1], SOCN_obs, sb[3], '(b) Ocean sink, baseline (20p)')
    panel(axes[1, 0], SLND_obs, sc[2],
          r'(c) Land sink, with $\mu_C$ (21p)')
    panel(axes[1, 1], SOCN_obs, sc[3],
          r'(d) Ocean sink, with $\mu_C$ (21p)')

    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    pdf = os.path.join(FIGDIR, 'figure_smoothed_sinks_2025.pdf')
    png = os.path.join(FIGDIR, 'figure_smoothed_sinks_2025.png')
    plt.savefig(pdf, bbox_inches='tight')
    plt.savefig(png, bbox_inches='tight', dpi=180)
    print(f'wrote {pdf}')
    print(f'wrote {png}')


if __name__ == '__main__':
    main()
