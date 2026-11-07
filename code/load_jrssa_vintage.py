"""
Shared loader for the multi-vintage JRSSA model-based audit.

Composes
  load_vintage_panels.load_gcb_aggregate(v)   - GCB aggregates
with the climate-covariate CSVs in data/ to return the same dict
shape as jrssa_m2_climate.load_vintage_2024/2025, for any vintage
v in {2017, ..., 2025}.

The headline sample for the multi-vintage extension is 1959-2016
(T=58), the longest common window across all nine vintages.
"""
import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
DATA = os.path.join(ROOT, 'data')

import load_vintage_panels as lvp

COMMON_LO_DEFAULT = 1959
COMMON_HI_DEFAULT = 2016


def _load_climate_shifted(years_arr):
    """Return (enso, nao, pdo, saod) each of length len(years_arr).

    Convention: out[i] is the value at calendar year (years_arr[i] + 1),
    matching the +1-year shift used throughout the JRSSA / SBM pipeline.
    """
    df = pd.read_csv(os.path.join(DATA, 'climate_indices_annual_1959_2024.csv'))
    clim_years = df['Year'].values.astype(int)
    enso_all = df['ENSO_Nino34'].values
    nao_all = df['NAO'].values
    pdo_all = df['PDO'].values

    sdf = pd.read_csv(os.path.join(DATA, 'saod_annual_1959_2024.csv'))
    # column may be 'Year' or 'year' depending on vintage of the CSV
    year_col = 'Year' if 'Year' in sdf.columns else 'year'
    sdf_years = sdf[year_col].values.astype(int)
    saod_all = sdf['SAOD'].values

    def _shifted(arr, source_years):
        out = np.empty(len(years_arr))
        for i, y in enumerate(years_arr):
            yshift = int(y) + 1
            j = np.where(source_years == yshift)[0]
            if len(j) == 0:
                j = [len(source_years) - 1]   # extrapolate by repeat
            out[i] = arr[j[0]]
        return out

    return (_shifted(enso_all, clim_years),
            _shifted(nao_all, clim_years),
            _shifted(pdo_all, clim_years),
            _shifted(saod_all, sdf_years))


def load_vintage_for_jrssa(vintage,
                            year_lo=COMMON_LO_DEFAULT,
                            year_hi=COMMON_HI_DEFAULT):
    """Load GCB aggregates + shifted climate covariates for one vintage.

    vintage: string in {'2017', ..., '2025'} (matches load_vintage_panels.VINTAGES).
    year_lo, year_hi: inclusive sample window. Default is the common
    1959-2016 window.

    Returns a dict with keys
      mObs           (T, 4) observations  (vCreg, E, S_LND, S_OCN)
      years          (T,) array of integer years
      E_FF, E_LUC, G_ATM, S_LND, S_OCN, vCreg
      enso_shifted, nao_shifted, pdo_shifted, saod_shifted
      cT             T

    Identical signature to jrssa_m2_climate.load_vintage_2024.
    """
    g = lvp.load_gcb_aggregate(vintage)
    yr_all = g['year']
    mask = (yr_all >= year_lo) & (yr_all <= year_hi)
    if mask.sum() == 0:
        raise ValueError(f'vintage {vintage}: no observations in '
                          f'{year_lo}-{year_hi}')
    years = yr_all[mask]
    E_FF = g['E_FF'][mask]
    E_LUC = g['E_LUC'][mask]
    G_ATM = g['G_ATM'][mask]
    S_LND = g['S_LND'][mask]
    S_OCN = g['S_OCN'][mask]
    S_CARB = g['S_CARB'][mask]   # cement carbonation sink (0 for 2017-2019)

    cGtC1959 = 2.127 * 315.39
    E = E_FF + E_LUC - S_CARB     # net anthropogenic emissions (matches model-free part)
    vCreg = cGtC1959 + np.cumsum(G_ATM)
    mObs = np.column_stack([vCreg, E, S_LND, S_OCN])

    enso, nao, pdo, saod = _load_climate_shifted(years)

    return dict(
        mObs=mObs, years=years,
        E_FF=E_FF, E_LUC=E_LUC, G_ATM=G_ATM,
        S_LND=S_LND, S_OCN=S_OCN, S_CARB=S_CARB, vCreg=vCreg,
        enso_shifted=enso, nao_shifted=nao,
        pdo_shifted=pdo, saod_shifted=saod,
        cT=int(mask.sum()),
        year_lo=year_lo, year_hi=year_hi,
    )


if __name__ == '__main__':
    print(f'Smoke test: shared loader on common '
          f'{COMMON_LO_DEFAULT}-{COMMON_HI_DEFAULT} sample\n')
    print(f'{"vintage":>8}  {"T":>3}  {"mObs.shape":>12}  '
          f'{"first year":>10}  {"last year":>10}')
    for v in lvp.VINTAGES:
        d = load_vintage_for_jrssa(v)
        print(f'{v:>8}  {d["cT"]:>3}  {str(d["mObs"].shape):>12}  '
              f'{d["years"][0]:>10}  {d["years"][-1]:>10}')
