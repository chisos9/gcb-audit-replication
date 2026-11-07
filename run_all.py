"""
run_all.py: reproduce the tables and figures of the paper.

Usage:
    python run_all.py                     # everything
    python run_all.py --from 5_tables     # use the fits shipped in results/
    python run_all.py --to 1_panel        # model-free audit only

Stages run in order; the script stops at the first failure.
"""
import argparse
import os
import subprocess
import sys
import time

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(PKG_DIR, 'code')
PRE = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']

STAGES = [
    ('1_panel', [
        ('bim_vintage_means.py', 'Figure 1'),
        ('table_bim_three_vintages.py', 'Table 1'),
        ('audit_step1_subperiods.py', 'Table 2'),
        ('figure_panel_af_sr_vintages.py', 'Figure 2'),
        ('audit_step2_vintages.py', 'Table 3'),
        ('audit_step4_appendixB.py', 'Tables A1-A4, input to Table 4'),
        ('table_ob_2024_2025.py', 'Table 4'),
        ('member_level_means.py', 'Per-member means cited in the text'),
        ('audit_step5_hc_counterfactual.py', 'H&C2023 counterfactual'),
    ]),
    ('2_estimate', [('jrssa_estimate_all_vintages.py', 'Baseline MDS-GCB, all vintages')]),
    ('3_hessian', [('jrssa_compute_hessian_all_vintages.py', 'Baseline Hessians')]),
    ('4_muC', [
        ('jrssa_estimate_muC_2025.py', 'mu_C model, 2025 vintage'),
        ('jrssa_estimate_muC_all_vintages.py', 'mu_C model, 2017-2024, and summary', PRE),
    ]),
    ('5_tables', [
        ('table_muC_2025_comparison.py', 'Table 5'),
        ('table_muC_vintages.py', 'Table 6'),
        ('table_jrssa_full_vintages.py', 'Table C1'),
    ]),
    ('6_figures', [
        ('figure_jrssa_beta_vintages.py', 'Figure 3'),
        ('figure_smoothed_sinks_2025.py', 'Figure 4'),
        ('appendix_muC_drift_figures.py', 'Figures 5, D1, D2'),
    ]),
    ('7_supplementary', [
        ('audit_step3_vintages.py', 'Panel-versus-published gaps (not in the paper)'),
        ('table_jrssa_levelblock_vintages.py', 'Level-block table (not in the paper)'),
        ('jrssa_af_sr_vintages.py', 'AF/SR bands (not in the paper)'),
        ('figure_jrssa_af_sr_vintages.py', 'AF/SR figure (not in the paper)'),
    ]),
]


def main():
    keys = [s[0] for s in STAGES]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='start', choices=keys, default=keys[0])
    ap.add_argument('--to', dest='stop', choices=keys, default=keys[-1])
    args = ap.parse_args()
    for d in ('results', 'output/tables', 'output/figures', 'output/csv'):
        os.makedirs(os.path.join(PKG_DIR, d), exist_ok=True)
    for key, items in STAGES[keys.index(args.start):keys.index(args.stop) + 1]:
        print(f'\n### Stage {key} ###', flush=True)
        for item in items:
            script, label = item[0], item[1]
            extra = item[2] if len(item) > 2 else []
            t0 = time.time()
            print(f'\n>>> {script}  ({label})', flush=True)
            rc = subprocess.call([sys.executable, os.path.join(CODE_DIR, script)] + extra)
            if rc:
                sys.exit(f'ERROR: {script} returned exit code {rc}.')
            print(f'<<< {script} done in {time.time() - t0:.0f}s', flush=True)
    print('\nAll requested stages completed.')


if __name__ == '__main__':
    main()
