"""
Drift-diagnostic figures for the mu_C model (X^(1) intact).

Produces three PDFs in output/figures/:
  figure_X1_all_vintages.pdf      baseline X^(1), all nine vintages, 1959-2016
  figure_muC_full_states_2x2.pdf  2x2 data vs smoothed states, mu_C on full 1959-2024  (appendix)
  figure_muC_full_X1.pdf          X^(1) residual, mu_C on full 1959-2024  (appendix)

The 1959-2016 fit uses the canonical parameters_jrssa_muC_2025.npy.
The full-sample fit is re-estimated here (warm-started from the canonical
fit) and its parameters saved to parameters_jrssa_muC_2025_full1959_2024.npy.
State indices: C*=0, E*=8, S_LND*=2, S_OCN*=3, X^(1)=4.
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIG = os.path.abspath(os.path.join(HERE, os.pardir, 'output', 'figures'))
RES = os.path.abspath(os.path.join(HERE, os.pardir, 'results'))
from estimation_utils import multistart
from jrssa_m2_climate_muC import JRSSAClimateMuC
from jrssa_m2_climate import JRSSAClimate
from load_jrssa_vintage import load_vintage_for_jrssa

RED = '#c1272d'
_SIG = lambda x: np.exp(-x) / (1.0 + np.exp(-x))


def _sd(res, st):
    V = res.smoothed_state_cov
    if V.shape[0] == st.shape[0] and V.shape[1] == st.shape[0]:
        return np.sqrt(np.clip(np.array([V[j, j, :] for j in range(st.shape[0])]), 0, None))
    return np.sqrt(np.clip(np.array([V[:, j, j] for j in range(st.shape[0])]), 0, None))


def _fit(year_lo, year_hi, x0):
    d = load_vintage_for_jrssa('2025', year_lo, year_hi)
    m = JRSSAClimateMuC(d['mObs'], d['enso_shifted'], d['nao_shifted'],
                        d['pdo_shifted'], d['saod_shifted'])
    ll, p_hat, modes = multistart(m, anchors=[('canonical', x0)], seed=7,
                                  n_jitter=8, scale=0.4)
    return d, m, p_hat, ll


def _x1_fig(d, res, st, out, title):
    yr = d['years']; X1 = st[4]; sd = _sd(res, st)[4]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.axhline(0, color='0.6', lw=0.8)
    ax.fill_between(yr, X1 - 1.96 * sd, X1 + 1.96 * sd, color=RED, alpha=0.18, lw=0, label='95% band')
    ax.plot(yr, X1, '-o', color=RED, ms=3, lw=1.8, label=r'$X^{(1)}$ smoothed')
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel('Year'); ax.set_ylabel(r'$X^{(1)} = C - C^{*}$  (GtC)')
    ax.grid(alpha=0.25); ax.set_xlim(yr[0], yr[-1]); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print('wrote', out, '  X1 range=[%.2f, %.2f] GtC' % (X1.min(), X1.max()))


def _states_2x2(d, res, st, out, title):
    yr = d['years']; sd = _sd(res, st)
    panels = [(0, d['mObs'][:, 0], r'Cumulative atmospheric CO$_2$ stock $C$ (GtC)'),
              (8, d['mObs'][:, 1], r'Net anthropogenic emissions $E$ (GtC/yr)'),
              (2, d['mObs'][:, 2], r'Land sink $S^{LND}$ (GtC/yr)'),
              (3, d['mObs'][:, 3], r'Ocean sink $S^{OCN}$ (GtC/yr)')]
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6)); axs = axs.ravel()
    for ax, (j, obs, ttl) in zip(axs, panels):
        s = st[j]; b = 1.96 * sd[j]
        ax.fill_between(yr, s - b, s + b, color=RED, alpha=0.18, lw=0, label='95% state band')
        ax.plot(yr, obs, 'o', ms=3.2, color='0.25', label='GCB data', zorder=3)
        ax.plot(yr, s, '-', color=RED, lw=1.9, label='smoothed latent state', zorder=4)
        ax.set_title(ttl, fontsize=10.5); ax.set_xlabel('Year', fontsize=9)
        ax.grid(alpha=0.25); ax.tick_params(labelsize=8.5); ax.set_xlim(yr[0], yr[-1])
    axs[0].legend(fontsize=8.5, loc='upper left', framealpha=0.9)
    fig.suptitle(title, fontsize=11.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out); plt.close(fig)
    print('wrote', out)


def _x1_vintages(out):
    """Overlay baseline X^(1) for all nine vintages 2017-2025 on 1959-2016
    (Section 5.2), no confidence bands."""
    vints = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025']
    blues = plt.cm.Blues(np.linspace(0.35, 0.9, 8))
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.axhline(0, color='0.6', lw=0.8, zorder=1)
    for i, v in enumerate(vints):
        d = load_vintage_for_jrssa(v, 1959, 2016); yr = d['years']
        m = JRSSAClimate(d['mObs'], d['enso_shifted'], d['nao_shifted'],
                         d['pdo_shifted'], d['saod_shifted'])
        p = np.load(os.path.join(RES, f'parameters_jrssa_{v}.npy'))
        st = m.smooth(p).smoothed_state
        X1 = st[4]; phi1 = _SIG(p[2])
        col, lw, z = ('#c1272d', 2.4, 5) if v == '2025' else (blues[i], 1.6, 3)
        ax.plot(yr, X1, '-', color=col, lw=lw, zorder=z,
                label=r'%s ($\hat\phi_1$=%.2f)' % (v, phi1))
    ax.set_title(r'Cumulative-CO$_2$ discrepancy $X^{(1)}=C-C^{*}$, baseline MDS-GCB, '
                 r'all vintages 2017--2025 (1959--2016)', fontsize=10.5)
    ax.set_xlabel('Year'); ax.set_ylabel(r'$X^{(1)}$  (GtC)')
    ax.grid(alpha=0.25); ax.set_xlim(1959, 2016)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc='lower left')
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print('wrote', out)


def main():
    os.makedirs(FIG, exist_ok=True)
    canon = np.load(os.path.join(RES, 'parameters_jrssa_muC_2025.npy'))

    # (1) baseline X^(1) overlay for all nine vintages, 1959-2016
    _x1_vintages(os.path.join(FIG, 'figure_X1_all_vintages.pdf'))

    # (2) full 1959-2024 fit (re-estimated): 2x2 states + X^(1)
    t0 = time.time()
    dF, mF, pF, llF = _fit(1959, 2024, canon)
    print('full-sample fit: loglik=%.4f  t=%.1fs' % (llF, time.time() - t0))
    disp = JRSSAClimateMuC.transform_params_display(pF)
    print('  beta1=%.3f beta2=%.3f phi1=%.4f phi3=%.4f mu_C=%+.4f'
          % (disp['beta1'], disp['beta2'], disp['phi1'], disp['phi3'], disp['mu_C']))
    np.save(os.path.join(RES, 'parameters_jrssa_muC_2025_full1959_2024.npy'), pF)
    resF = mF.smooth(pF); stF = resF.smoothed_state
    _states_2x2(dF, resF, stF,
                os.path.join(FIG, 'figure_muC_full_states_2x2.pdf'),
                r'MDS-GCB with $\mu_C$: data vs smoothed latent states, 2025 vintage full sample 1959--2024')
    _x1_fig(dF, resF, stF,
            os.path.join(FIG, 'figure_muC_full_X1.pdf'),
            r'Cumulative-CO$_2$ discrepancy $X^{(1)}$, $\mu_C$ fit, 2025 vintage full sample 1959--2024')


if __name__ == '__main__':
    main()
