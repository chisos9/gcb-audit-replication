"""
Appendix B of the multi-vintage panel audit: Oaxaca-Blinder
decomposition of each consecutive vintage transition 2017->2018
through 2023->2024.

For each panel (BK, DGVM LUC, DGVM Terrestrial Sink, Ocean) and each
transition v->w, decompose the panel-mean shift as

  ΔX̄  =  Δ_rev  +  Δ_in  -  Δ_out

where
  Δ_rev   = m_C(w) - m_C(v)
  Δ_in    = (N_in / N_w) * (m_in - m_C(w))
  Δ_out   = (N_out / N_v) * (m_out - m_C(v))

m_C is the panel mean across continuing members; m_in (m_out) are the
means across newly-added (removed) members. The decomposition is on the
common 1959-2016 sample.

Model-name canonicalization handles known renames (CABLE / CABLE-POP,
CLM4.5(BGC) / CLM5.0 / CLM6.0, CLASS-CTEM / CLASSIC, JULES / JULES-ES,
OCN / OCNv2, LPX-Bern / LPX-Bern_DGVM). The headline 2024->2025
transition is omitted from this appendix because it appears in the main
text (Section 4.4 of section4_audit_panel.tex).

Outputs:
  output/csv/audit_step4_appendixB.csv
  output/tables/table_step4_appendixB.tex
"""
import os
import csv
import numpy as np

import load_vintage_panels as lvp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
TEX_OUT = os.path.join(ROOT, 'output', 'tables',
                       'table_step4_appendixB.tex')

COMMON_LO, COMMON_HI = 1959, 2016


# Canonical-name aliases: each list is treated as one model across
# vintages. The canonical name is the first entry.
ALIASES = [
    ['CABLE-POP', 'CABLE'],
    ['CLM6.0', 'CLM5.0', 'CLM4.5(BGC)'],
    ['CLASSIC', 'CLASS-CTEM'],
    ['JULES-ES', 'JULES'],
    ['OCN', 'OCNv2'],
    ['LPX-Bern', 'LPX-Bern_DGVM', 'LPX'],
    # Ocean side renames
    ['NEMO3.6-PISCESv2-gas (CNRM)', 'NEMO-PISCES (CNRM)', 'CNRM',
     'NEMO-PISCES (CNRM)_c7'],
    ['NEMO-PISCES (IPSL)', 'IPSL'],
    ['FESOM-REcoM', 'FESOM-2.1-REcoM3', 'FESOM2-RECOM2', 'FESOM2',
     'MITgcm-REcoM2', 'FESOM'],
    ['MOM6-COBALT (Princeton)', 'PRINCETON', 'MOM6-COBALT', 'Princeton'],
    ['MICOM-HAMOCC (NorESM1-OCv1.2)', 'NorESM-OC', 'NORESM', 'NorESM',
     'MICOM-HAMOCC'],
    ['NEMO-PlankTOM12', 'NEMO-PlankTOM5', 'PlankTOM'],
    ['CESM-ETHZ', 'CESM-ETH', 'CESM2'],
    ['MPIOM-HAMOCC6', 'MPIOM-HAMOCC', 'MPI', 'MPIOM'],
    ['CSIR-ML6', 'CSIR'],
    ['ACCESS (CSIRO)', 'ACCESS', 'CSIRO'],
    ['MRI-ESM2-21', 'MRI-ESM2', 'MRI'],
]
_ALIAS_MAP = {}
for grp in ALIASES:
    canon = grp[0]
    for nm in grp:
        _ALIAS_MAP[nm] = canon


def canonical(name):
    return _ALIAS_MAP.get(name, name)


def _panel_dict_to_canon(d):
    """Map {model_name: arr} -> {canonical_name: arr} (collapsing aliases)."""
    out = {}
    for k, v in d.items():
        out[canonical(k)] = v   # if duplicates, last wins; should not happen
                                  # within a single vintage
    return out


def _restrict_mean(arr, years, lo, hi):
    mask = (years >= lo) & (years <= hi)
    return float(np.nanmean(arr[mask]))


def ob_decompose(d_v, d_w, years_v, years_w, lo, hi):
    """Oaxaca-Blinder decomposition of panel-mean shift between vintages
    v and w.

    d_v, d_w: {canonical_name: array} dictionaries for the two vintages.
    Returns dict with delta_mean, delta_rev, delta_in, delta_out, counts.
    """
    names_v = set(d_v.keys())
    names_w = set(d_w.keys())
    cont   = sorted(names_v & names_w)
    out    = sorted(names_v - names_w)
    inn    = sorted(names_w - names_v)

    N_v = len(names_v)
    N_w = len(names_w)
    N_c = len(cont)
    N_in = len(inn)
    N_out = len(out)

    def panel_mean_subset(d, names, years):
        if not names:
            return float('nan')
        cols = [_restrict_mean(d[nm], years, lo, hi) for nm in names]
        # ignore members with no data on the common sample
        return float(np.nanmean(cols))

    m_C_v   = panel_mean_subset(d_v, cont, years_v)
    m_C_w   = panel_mean_subset(d_w, cont, years_w)
    m_out_v = panel_mean_subset(d_v, out, years_v) if out else float('nan')
    m_in_w  = panel_mean_subset(d_w, inn, years_w) if inn else float('nan')

    mean_v = panel_mean_subset(d_v, list(names_v), years_v)
    mean_w = panel_mean_subset(d_w, list(names_w), years_w)
    delta_mean = mean_w - mean_v

    delta_rev = m_C_w - m_C_v
    delta_in  = ((N_in / N_w) * (m_in_w  - m_C_w)) if N_in else 0.0
    delta_out = ((N_out / N_v) * (m_out_v - m_C_v)) if N_out else 0.0

    # Identity: ΔX̄ = Δ_rev + Δ_in - Δ_out
    resid = delta_mean - (delta_rev + delta_in - delta_out)
    return dict(
        delta_mean=delta_mean,
        delta_rev=delta_rev,
        delta_in=delta_in,
        delta_out=delta_out,
        residual=resid,
        N_v=N_v, N_w=N_w, N_c=N_c, N_in=N_in, N_out=N_out,
        names_in=inn, names_out=out,
    )


def load_all_panels():
    """Return per-vintage dicts {panel_name: (years, canon_model_dict)}."""
    out = {}
    for v in lvp.VINTAGES:
        g = lvp.load_gcb_aggregate(v)
        yL, gcb_L, _, mL = lvp.load_lnd_panel(v)
        yO, gcb_O, _, mO = lvp.load_ocn_panel(v)
        luc = lvp.load_luc_panel(v)
        out[v] = dict(
            year_g=g['year'], year_L=yL, year_O=yO, year_LUC=luc['year'],
            BK=_panel_dict_to_canon(luc['bk']),
            DGVM_LUC=_panel_dict_to_canon(luc['dgvm']),
            DGVM_LND=_panel_dict_to_canon(mL),
            Ocean=_panel_dict_to_canon(mO),
        )
    return out


def main():
    pan = load_all_panels()
    transitions = list(zip(lvp.VINTAGES[:-1], lvp.VINTAGES[1:]))

    panel_names = ['BK', 'DGVM_LUC', 'DGVM_LND', 'Ocean']
    panel_labels = {'BK': 'BK',
                    'DGVM_LUC': 'DGVM LUC',
                    'DGVM_LND': 'DGVM $S_{\\mathrm{LND}}$',
                    'Ocean': 'Ocean'}
    year_key = {'BK': 'year_LUC', 'DGVM_LUC': 'year_LUC',
                'DGVM_LND': 'year_L', 'Ocean': 'year_O'}

    rows = []   # (transition, panel, deltas...)
    print(f'Oaxaca-Blinder for consecutive vintage transitions, '
          f'common {COMMON_LO}-{COMMON_HI} sample')
    for v, w in transitions:
        is_headline = (w == '2025')
        tag = f'{v}->{w}' + ('  (HEADLINE / main text)' if is_headline else '')
        print('\n', '=' * 70, '\n ', tag)
        for pname in panel_names:
            d_v = pan[v][pname]; d_w = pan[w][pname]
            yv  = pan[v][year_key[pname]]
            yw  = pan[w][year_key[pname]]
            r = ob_decompose(d_v, d_w, yv, yw, COMMON_LO, COMMON_HI)
            print(f'   {pname:>10}  N {r["N_v"]:>2}->{r["N_w"]:>2}  '
                  f'(C={r["N_c"]}, in={r["N_in"]}, out={r["N_out"]})  '
                  f'Δ={r["delta_mean"]:+.3f}  rev={r["delta_rev"]:+.3f}  '
                  f'in={r["delta_in"]:+.3f}  out={r["delta_out"]:+.3f}  '
                  f'(resid {r["residual"]:+.4f})')
            rows.append(dict(trans=f'{v}-{w}', panel=pname,
                             headline=is_headline, **r))

    # CSV
    csv_path = os.path.join(ROOT, 'output', 'csv', 'audit_step4_appendixB.csv')
    with open(csv_path, 'w', newline='') as fh:
        w_csv = csv.writer(fh)
        w_csv.writerow(['transition', 'panel', 'N_prev', 'N_next',
                        'N_continuing', 'N_in', 'N_out',
                        'delta_mean', 'delta_rev', 'delta_in', 'delta_out',
                        'residual', 'names_in', 'names_out'])
        for r in rows:
            w_csv.writerow([r['trans'], r['panel'],
                            r['N_v'], r['N_w'], r['N_c'], r['N_in'], r['N_out'],
                            f'{r["delta_mean"]:.6f}',
                            f'{r["delta_rev"]:.6f}',
                            f'{r["delta_in"]:.6f}',
                            f'{r["delta_out"]:.6f}',
                            f'{r["residual"]:.6f}',
                            ';'.join(r['names_in']),
                            ';'.join(r['names_out'])])
    print(f'\nWrote {csv_path}')

    # LaTeX fragment: one table per panel, 7 transitions x 4 components
    os.makedirs(os.path.dirname(TEX_OUT), exist_ok=True)
    lines = []
    # Drop the headline 2024->2025 row
    non_headline = [r for r in rows if not r['headline']]
    for pname in panel_names:
        lines.append(r'\begin{table}[h]')
        lines.append(r'  \centering')
        lines.append(
            r'  \caption{Appendix B. Oaxaca-Blinder decomposition of the '
            + panel_labels[pname]
            + r' ensemble-mean change across consecutive non-headline GCB '
              r'vintages. Common $1959$--$2016$ sample. '
              r'Components: revision on continuing members, composition '
              r'inflow from new members, composition outflow from removed '
              r'members. Decomposition identity '
              r'$\Delta \bar X = \Delta_{\mathrm{rev}} + \Delta_{\mathrm{in}}'
              r' - \Delta_{\mathrm{out}}$.}'
        )
        lines.append(r'  \label{tab:appB_' + pname.lower() + '}')
        lines.append(r'  \begin{tabular}{lrrrrrrr}')
        lines.append(r'    \toprule')
        lines.append(r'    Transition & $N_v\!\to\! N_w$ & $N_{\mathrm{in}}$ '
                     r'& $N_{\mathrm{out}}$ & $\Delta \bar X$ '
                     r'& $\Delta_{\mathrm{rev}}$ & $\Delta_{\mathrm{in}}$ '
                     r'& $-\Delta_{\mathrm{out}}$ \\')
        lines.append(r'    \midrule')
        for r in [r for r in non_headline if r['panel'] == pname]:
            lines.append(
                f'    {r["trans"]} & ${r["N_v"]}\\!\\to\\!{r["N_w"]}$ & '
                f'{r["N_in"]} & {r["N_out"]} & '
                f'${r["delta_mean"]:+.3f}$ & ${r["delta_rev"]:+.3f}$ & '
                f'${r["delta_in"]:+.3f}$ & ${-r["delta_out"]:+.3f}$ \\\\'
            )
        lines.append(r'    \bottomrule')
        lines.append(r'  \end{tabular}')
        lines.append(r'\end{table}')
        lines.append('')

    with open(TEX_OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Wrote {TEX_OUT}')

    # Summary statement: largest |delta_X| and largest |component| across
    # all 7 non-headline transitions, by panel
    print('\nSummary of largest absolute components across non-headline transitions:')
    for pname in panel_names:
        sub = [r for r in non_headline if r['panel'] == pname]
        max_dx = max((abs(r['delta_mean']) for r in sub), default=0)
        max_comp = max((max(abs(r['delta_rev']), abs(r['delta_in']),
                            abs(r['delta_out'])) for r in sub), default=0)
        print(f'  {pname:>10}  max |ΔX̄| = {max_dx:.3f}, '
              f'max |component| = {max_comp:.3f}')


if __name__ == '__main__':
    main()
