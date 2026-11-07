"""
Step 5: counterfactual restoration of H&C2023 to the GCB2025 bookkeeping
ensemble, on the common 1959-2016 sample.

H&C2023 (Houghton and Castanho, 2023) was dropped from the bookkeeping
ensemble in GCB2025 because it does not use transient carbon densities
and its data end in 2020. It is therefore not available at 2025 vintage.
On the common 1959-2016 sample this does not matter: the model covers the
window in full, and we hold it at the values published in GCB2024. The
script first quantifies how much that freezing assumption can cost, by
measuring the model's own GCB2023 to GCB2024 revision on the same window.

The counterfactual adds H&C2023 back as a fourth member of the GCB2025
bookkeeping ensemble, recomputes the ensemble mean (which is the GCB's
published E_LUC), and passes the difference through the budget identity
to the imbalance. Every other component is held at its GCB2025 value, so
the exercise answers "what would the 2025 imbalance have been had this one
model still been in the ensemble", not "what would the GCB have done".

Outputs:
  hc_counterfactual.csv   one row per quantity reported below
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
CSV_DIR = os.path.abspath(os.path.join(CODE_DIR, os.pardir, 'output', 'csv'))
os.makedirs(CSV_DIR, exist_ok=True)
os.chdir(CSV_DIR)

import load_vintage_panels as lvp

YEAR_LO, YEAR_HI = 1959, 2016
HC = 'H&C2023'


def window(year, x):
    year = np.asarray(year)
    m = (year >= YEAR_LO) & (year <= YEAR_HI)
    return np.asarray(x, dtype=float)[m]


def hc_series(vintage):
    p = lvp.load_luc_panel(vintage)
    return window(p['year'], p['bk'][HC])


def ar1_se(x):
    """Standard error of the mean inflated for first-order serial
    correlation, as in Step 1."""
    x = np.asarray(x, dtype=float)
    T = len(x)
    s = np.nanstd(x, ddof=1)
    dev = x - np.nanmean(x)
    rho = float(np.sum(dev[1:] * dev[:-1]) / np.sum(dev ** 2))
    return s / np.sqrt(T), s * np.sqrt((1.0 + rho) / (1.0 - rho) / T), rho


def main():
    rows = []

    hc23, hc24 = hc_series('2023'), hc_series('2024')
    rev = hc24 - hc23
    print("=" * 74)
    print("  Freezing assumption: H&C2023 revision between its last two vintages")
    print("=" * 74)
    print("  as published in GCB2023 : mean %+.4f" % np.nanmean(hc23))
    print("  as published in GCB2024 : mean %+.4f" % np.nanmean(hc24))
    print("  revision                : mean %+.4f  sd %.4f  max|.| %.4f"
          % (np.nanmean(rev), np.nanstd(rev, ddof=1), np.nanmax(np.abs(rev))))
    rows += [dict(quantity='H&C2023 mean, GCB2023 vintage', value=np.nanmean(hc23)),
             dict(quantity='H&C2023 mean, GCB2024 vintage', value=np.nanmean(hc24)),
             dict(quantity='H&C2023 revision, mean', value=np.nanmean(rev)),
             dict(quantity='H&C2023 revision, max abs', value=np.nanmax(np.abs(rev)))]

    p25 = lvp.load_luc_panel('2025')
    members = {k: window(p25['year'], v) for k, v in p25['bk'].items()}
    bk3 = np.nanmean(np.vstack(list(members.values())), axis=0)
    bk4 = np.nanmean(np.vstack(list(members.values()) + [hc24]), axis=0)

    print()
    print("=" * 74)
    print("  Counterfactual bookkeeping ensemble, %d-%d" % (YEAR_LO, YEAR_HI))
    print("=" * 74)
    for k, v in members.items():
        print("  %-10s mean %+.4f" % (k, np.nanmean(v)))
        rows.append(dict(quantity='BK member %s mean' % k, value=np.nanmean(v)))
    print("  %-10s mean %+.4f   (frozen at GCB2024)" % (HC, np.nanmean(hc24)))
    print("  BK mean, 3 members (published GCB2025) : %+.4f" % np.nanmean(bk3))
    print("  BK mean, 4 members (counterfactual)    : %+.4f" % np.nanmean(bk4))
    print("  difference in E_LUC                    : %+.4f"
          % (np.nanmean(bk4) - np.nanmean(bk3)))
    rows += [dict(quantity='BK mean 3 members (published)', value=np.nanmean(bk3)),
             dict(quantity='BK mean 4 members (counterfactual)', value=np.nanmean(bk4)),
             dict(quantity='E_LUC difference', value=np.nanmean(bk4) - np.nanmean(bk3))]

    agg = lvp.load_gcb_aggregate('2025')
    eluc = window(agg['year'], agg['E_LUC'])
    bim = window(agg['year'], agg['BIM_raw'])
    resid = np.nanmax(np.abs(eluc - bk3))
    if resid > 5e-3:
        print("  WARNING: published E_LUC differs from the 3-member mean by "
              "up to %.4f" % resid)

    bim_cf = bim + (bk4 - bk3)
    se_n, se_ar, rho = ar1_se(bim_cf)
    m_cf = np.nanmean(bim_cf)

    print()
    print("=" * 74)
    print("  Budget imbalance, %d-%d" % (YEAR_LO, YEAR_HI))
    print("=" * 74)
    print("  published GCB2025            : %+.4f" % np.nanmean(bim))
    print("  counterfactual with %s   : %+.4f" % (HC, m_cf))
    print("  change                       : %+.4f" % (m_cf - np.nanmean(bim)))
    print("  Normal 95%% CI                : [%+.3f, %+.3f]"
          % (m_cf - 1.96 * se_n, m_cf + 1.96 * se_n))
    print("  AR(1) 95%% CI                  : [%+.3f, %+.3f]  (rho = %.3f)"
          % (m_cf - 1.96 * se_ar, m_cf + 1.96 * se_ar, rho))
    rows += [dict(quantity='BIM mean, published GCB2025', value=np.nanmean(bim)),
             dict(quantity='BIM mean, counterfactual', value=m_cf),
             dict(quantity='BIM counterfactual, Normal CI lo', value=m_cf - 1.96 * se_n),
             dict(quantity='BIM counterfactual, Normal CI hi', value=m_cf + 1.96 * se_n),
             dict(quantity='BIM counterfactual, AR(1) CI lo', value=m_cf - 1.96 * se_ar),
             dict(quantity='BIM counterfactual, AR(1) CI hi', value=m_cf + 1.96 * se_ar),
             dict(quantity='BIM counterfactual, AR(1) rho', value=rho)]

    n_bk = len(members) + 1
    ob = (np.nanmean(bk3) - np.nanmean(hc24)) / n_bk
    print()
    print("  for reference, the Oaxaca-Blinder leaver contribution in Step 3 is")
    print("  computed with the continuing members at their 2024 values; the same")
    print("  expression at their 2025 values gives %+.4f" % ob)
    rows.append(dict(quantity='leaver term at 2025 member values', value=ob))

    pd.DataFrame(rows).to_csv('hc_counterfactual.csv', index=False)
    print("\n  wrote hc_counterfactual.csv")


if __name__ == '__main__':
    main()
