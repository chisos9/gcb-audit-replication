"""
Multi-vintage panel loader for the model-free audit.

For each GCB vintage 2017-2025, returns:
  - the published GCB-preferred BIM, S_LND, S_OCN, E_LUC series
  - the underlying DGVM land-sink panel (per-model time series)
  - the underlying ocean panel (per-model time series, GOBMs + fCO2)
  - the underlying bookkeeping LUC panel (per-model)
  - the underlying DGVM LUC panel (per-model, where available)

Each panel sheet has its own header row offset and column layout per
vintage. Configs are hardcoded below from manual inspection.

Notation:
  GCB-published series (col 2 of each panel sheet) is the GCB "preferred"
  reference cited in the audit identity. In 2025 the GCB column is the
  adjusted value; col 3 ("GCB unadjusted") equals the MMM.

Sample windows:
  - 2017 vintage covers 1959-2016 (T=58)
  - each subsequent vintage covers 1959 to (vintage_year - 1)
  - common headline sample: 1959-2016 (T=58)
"""
import os
import warnings
import numpy as np
import openpyxl

warnings.filterwarnings('ignore')

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = os.path.abspath(os.path.join(HERE, os.pardir))
DATA   = os.path.join(ROOT, 'data')

# ──────────────────────────────────────────────────────────────────────
# File / header configs per vintage
# ──────────────────────────────────────────────────────────────────────
VINTAGES = ['2017', '2018', '2019', '2020', '2021',
            '2022', '2023', '2024', '2025']

GCB_FILES = {
    '2017': 'CarbonBudget_2017-Data-Global-1.xlsx',
    '2018': 'CarbonBudget_2018-Global-1.xlsx',
    '2019': 'CarbonBudget_2019-Data-Global-1.xlsx',
    '2020': 'CarbonBudget_2020-Data-Global-1.xlsx',
    '2021': 'CarbonBudget_2021-Data-Global-1.xlsx',
    '2022': 'Global_Carbon_Budget_2022v1.0.xlsx',
    '2023': 'Global_Carbon_Budget_2023v1.0.xlsx',
    '2024': 'Global_Carbon_Budget_2024v1.0.xlsx',
    '2025': 'Global_Carbon_Budget_2025_v0.7.xlsx',
}

# (header_row, year_col, bim_col) for the Global Carbon Budget sheet
GCB_SHEET_CONFIG = {
    '2017': (20, 1, 7),
    '2018': (20, 1, 7),
    '2019': (19, 2, 8),
    '2020': (21, 1, 8),
    '2021': (21, 1, 8),
    '2022': (21, 1, 8),
    '2023': (22, 1, 8),
    '2024': (22, 1, 8),
    '2025': (22, 1, 8),
}

# header_row for Terrestrial Sink, Ocean Sink (single-row headers)
TLND_HEADER = {
    '2017': 23, '2018': 24, '2019': 24, '2020': 25, '2021': 25,
    '2022': 24, '2023': 28, '2024': 28, '2025': 30,
}
OCN_HEADER = {
    '2017': 21, '2018': 20, '2019': 23, '2020': 25, '2021': 28,
    '2022': 31, '2023': 31, '2024': 32, '2025': 32,
}

# Land-Use Change Emissions sheet: explicit column maps because the
# layout has Net + Gross Sink + Gross Source sub-headers from 2021
# onwards, and peat columns from 2023 onwards.
# LUC_CONFIG[vintage] = dict with keys
#   header_row : main header row (Year row)
#   data_row   : first row of numeric data (header_row + skip + 1)
#   year_col, gcb_col : column indices for Year and GCB-published
#   bk_cols    : {model_name: col_index for Net column}
#   peat_cols  : {model_name: col_index} or None if vintage predates peat panel
#   dgvm_cols  : {model_name: col_index for Net column}
LUC_CONFIG = {
    '2017': dict(
        header_row=24, data_row=25, year_col=1, gcb_col=2,
        bk_cols={'H&N': 4, 'BLUE': 5},
        peat_cols=None,
        dgvm_cols={'CABLE':7, 'CLASS-CTEM':8, 'CLM4.5(BGC)':9, 'DLEM':10,
                   'ISAM':11, 'JULES':12, 'LPJ':13, 'LPX-Bern':14,
                   'OCN':15, 'ORCHIDEE':16, 'ORCHIDEE-MICT':17, 'SDGVM':18},
    ),
    '2018': dict(
        header_row=28, data_row=29, year_col=1, gcb_col=2,
        bk_cols={'H&N': 4, 'BLUE': 5},
        peat_cols=None,
        dgvm_cols={'CABLE-POP':7, 'CLASS-CTEM':8, 'CLM5.0':9, 'DLEM':10,
                   'ISAM':11, 'JSBACH':12, 'JULES':13, 'LPJ-GUESS':14,
                   'LPJ':15, 'LPX-Bern':16, 'OCN':17, 'ORCHIDEE-CNP':18,
                   'ORCHIDEE-Trunk':19, 'SDGVM':20, 'SURFEX':21, 'VISIT':22},
    ),
    '2019': dict(
        header_row=26, data_row=27, year_col=1, gcb_col=2,
        bk_cols={'H&N': 4, 'BLUE': 5},
        peat_cols=None,
        dgvm_cols={'CABLE':7, 'CLASS-CTEM':8, 'CLM5.0':9, 'DLEM':10,
                   'ISAM':11, 'ISBA-CTRIP':12, 'JSBACH':13, 'JULES':14,
                   'LPJ-GUESS':15, 'LPJ':16, 'LPX':17, 'OCN':18,
                   'ORCHIDEE-CNP':19, 'ORCHIDEE-Trunk':20, 'SDGVM':21,
                   'VISIT':22},
    ),
    '2020': dict(
        header_row=29, data_row=30, year_col=1, gcb_col=2,
        bk_cols={'H&N': 4, 'BLUE': 5, 'OSCAR': 6},
        peat_cols=None,
        dgvm_cols={'CABLE-POP':8, 'CLASSIC':9, 'CLM5.0':10, 'DLEM':11,
                   'IBIS':12, 'ISAM':13, 'ISBA-CTRIP':14, 'JSBACH':15,
                   'JULES-ES':16, 'LPJ-GUESS':17, 'LPJ':18, 'LPX-Bern':19,
                   'OCN':20, 'ORCHIDEE-v3':21, 'SDGVM':22, 'VISIT':23,
                   'YIBs':24},
    ),
    # 2021: Net+Gross Sink+Gross Source sub-headers, so BK Net cols are
    # at 6, 9, 12 (each spans 3 sub-cols). DGVMs are single Net col each.
    # Data starts 2 rows after Year header (because sub-headers occupy
    # row hr+1). data_row = header_row + 2.
    '2021': dict(
        header_row=30, data_row=32, year_col=1, gcb_col=2,
        bk_cols={'H&N': 6, 'BLUE': 9, 'OSCAR': 12},
        peat_cols=None,
        dgvm_cols={'CABLE-POP':16, 'CLASSIC':17, 'CLM5.0':18, 'DLEM':19,
                   'IBIS':20, 'ISAM':21, 'ISBA-CTRIP':22, 'JSBACH':23,
                   'JULES-ES':24, 'LPJ-GUESS':25, 'LPJ':26, 'LPX-Bern':27,
                   'OCN':28, 'ORCHIDEE-v3':29, 'SDGVM':30, 'VISIT':31,
                   'YIBs':32},
    ),
    '2022': dict(
        header_row=29, data_row=31, year_col=1, gcb_col=2,
        bk_cols={'H&N': 6, 'BLUE': 9, 'OSCAR': 12},
        peat_cols=None,
        dgvm_cols={'CABLE-POP':16, 'CLASSIC':17, 'CLM5.0':18, 'DLEM':19,
                   'IBIS':20, 'ISAM':21, 'JSBACH':22, 'JULES-ES':23,
                   'LPJ-GUESS':24, 'LPJ':25, 'LPX-Bern':26, 'OCN':27,
                   'ORCHIDEE-v3':28, 'SDGVM':29, 'VISIT':30, 'YIBs':31},
        dgvm_luc_sign=-1,   # 2022 reports DGVM net land-to-atm flux with
                             # opposite sign convention to BK methods.
    ),
    # 2023+: also has a peat block, then DGVM "Individual models (NET)"
    '2023': dict(
        header_row=37, data_row=39, year_col=1, gcb_col=2,
        bk_cols={'BLUE': 9, 'H&C2023': 14, 'OSCAR': 19},
        peat_cols={'FAO':25, 'LPX-Bern':26, 'ORCHIDEE':27},
        dgvm_cols={'CABLE-POP':29, 'CLASSIC':30, 'CLM5.0':31, 'DLEM':32,
                   'ED':33, 'ELM':34, 'IBIS':35, 'ISAM':36, 'ISBA-CTRIP':37,
                   'JSBACH':38, 'JULES':39, 'LPJ-GUESS':40, 'LPJ':41,
                   'LPJwsl':42, 'LPX-Bern_DGVM':43, 'OCNv2':44, 'ORCHIDEE':45,
                   'SDGVM':46, 'VISIT':47, 'YIBs':48},
        dgvm_luc_sign=-1,   # 2023 same sign convention as 2022.
    ),
    '2024': dict(
        header_row=38, data_row=40, year_col=1, gcb_col=2,
        bk_cols={'BLUE': 9, 'H&C2023': 14, 'OSCAR': 19, 'LUCE': 24},
        peat_cols={'FAO':30, 'LPX-Bern':31, 'ORCHIDEE':32},
        dgvm_cols={'CABLE-POP':34, 'CLASSIC':35, 'CLM6.0':36, 'DLEM':37,
                   'ED':38, 'ELM':39, 'IBIS':40, 'iMAPLE':41, 'ISAM':42,
                   'ISBA-CTRIP':43, 'JSBACH':44, 'JULES':45, 'LPJ-GUESS':46,
                   'LPJ':47, 'LPJwsl':48, 'LPX-Bern_DGVM':49, 'OCN':50,
                   'ORCHIDEE':51, 'SDGVM':52, 'VISIT':53},
    ),
    '2025': dict(
        header_row=39, data_row=41, year_col=1, gcb_col=2,
        bk_cols={'BLUE': 9, 'OSCAR': 14, 'LUCE': 19},
        peat_cols={'FAO':25, 'LPX-Bern':26, 'ORCHIDEE':27},
        dgvm_cols={'CABLE-POP':29, 'CLASSIC':30, 'CLM6.0':31, 'CLM-FATES':32,
                   'DLEM':33, 'ED':34, 'ELM':35, 'ELM-FATES':36, 'IBIS':37,
                   'iMAPLE':38, 'ISAM':39, 'ISBA-CTRIP':40, 'JSBACH':41,
                   'JULES-ES':42, 'LPJ-GUESS':43, 'LPJ':44, 'LPJwsl':45,
                   'LPX-Bern_DGVM':46, 'OCN':47, 'ORCHIDEE':48,
                   'SDGVM':49, 'VISIT':50, 'VISIT-UT':51},
    ),
}

# Skip-column names for the Terrestrial and Ocean panels: anything that
# is not a model time series.
NONMODEL_COLS = {
    'Year', 'year', 'GCB', 'GCB unadjusted', '1 sigma uncertainty',
    'Multi-model mean', 'MMM', 'MMM (multi-model mean)',
    'Model Spread (sd)', 'Model spread (sd)', 'stdev',
    'Ensemble mean', 'MME', 'ave',
    'land sink', 'Land Sink', 'Land sink',
    'ocean sink', 'Ocean Sink',
    # Per-year aggregate columns that some GCB vintages place alongside
    # the individual data products on the Ocean Sink sheet. Excluding
    # these from the model list keeps the panel-mean computation honest.
    'Average data-products (excl. Watson.)',
    'sd data-products (excl. Watson.)',
    'Average fCO2-products',
    'sd fCO2-product',
    'sd fCO2-products',
    'MPM (multi-product mean)',
    'mean data-products (excl. Watson.)',
}


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────
def _open_sheet(vintage, sheet_name):
    fpath = os.path.join(DATA, GCB_FILES[vintage])
    wb = openpyxl.load_workbook(fpath, data_only=True)
    if sheet_name not in wb.sheetnames:
        raise KeyError(f'{vintage}: sheet {sheet_name!r} not present '
                       f'(have: {wb.sheetnames})')
    return wb, wb[sheet_name]


def _read_columns(ws, header_row, year_col, want_cols, last_year=None):
    """Read year + the requested columns from ws starting at header_row + 1.
    Returns (years, {colname: array}). Stops at first all-empty data row.
    """
    out = {name: [] for name in want_cols}
    years = []
    r = header_row + 1
    while True:
        y = ws.cell(r, year_col).value
        if not isinstance(y, (int, float)):
            # check next row; if also empty, terminate
            yn = ws.cell(r + 1, year_col).value
            if not isinstance(yn, (int, float)):
                break
            r += 1
            continue
        y = int(y)
        if last_year is not None and y > last_year:
            break
        years.append(y)
        for name, col in want_cols.items():
            v = ws.cell(r, col).value
            out[name].append(float(v) if isinstance(v, (int, float)) else np.nan)
        r += 1
        if r > 300:
            break
    return np.array(years), {k: np.array(v) for k, v in out.items()}


def _scan_header_models(ws, header_row, year_col,
                        skip_cols=NONMODEL_COLS, max_col=80):
    """Return {model_name: col_index} for all columns in header_row that
    are not in skip_cols and not blank."""
    out = {}
    for c in range(year_col, max_col + 1):
        v = ws.cell(header_row, c).value
        if v is None:
            continue
        name = str(v).strip()
        if not name:
            continue
        if name in skip_cols:
            continue
        if name.lower().startswith('unnamed'):
            continue
        # avoid duplicates: keep first
        if name in out:
            # disambiguate by appending col index
            name = f'{name}_c{c}'
        out[name] = c
    return out


# ──────────────────────────────────────────────────────────────────────
# Public loaders
# ──────────────────────────────────────────────────────────────────────
def load_gcb_aggregate(vintage):
    """Return dict of pandas-style 1d arrays: year, E_FF, E_LUC,
    G_ATM, S_OCN, S_LND, S_CARB (may be NaN for 2017-2019),
    BIM_raw (col 7 for 2017-2019, col 8 from 2020 on)."""
    hr, yc, bc = GCB_SHEET_CONFIG[vintage]
    wb, ws = _open_sheet(vintage, 'Global Carbon Budget')
    # column layout in header row:
    #   2017-2019: Year, E_FF, E_LUC, G_ATM, S_OCN, S_LND, BIM
    #   2020+:     Year, E_FF, E_LUC, G_ATM, S_OCN, S_LND, S_CARB, BIM
    is_pre_cement = vintage in ('2017', '2018', '2019')
    if is_pre_cement:
        want = {'E_FF': yc + 1, 'E_LUC': yc + 2, 'G_ATM': yc + 3,
                'S_OCN': yc + 4, 'S_LND': yc + 5, 'BIM_raw': yc + 6}
    else:
        want = {'E_FF': yc + 1, 'E_LUC': yc + 2, 'G_ATM': yc + 3,
                'S_OCN': yc + 4, 'S_LND': yc + 5,
                'S_CARB': yc + 6, 'BIM_raw': yc + 7}
    years, cols = _read_columns(ws, hr, yc, want)
    wb.close()
    out = {'year': years, **cols}
    if is_pre_cement:
        out['S_CARB'] = np.zeros_like(out['E_FF'])  # absent → treat as 0
    return out


def load_lnd_panel(vintage):
    """Return (years, GCB_published, GCB_unadjusted_or_None,
    {model_name: array}) for the Terrestrial Sink sheet."""
    hr = TLND_HEADER[vintage]
    wb, ws = _open_sheet(vintage, 'Terrestrial Sink')
    # locate year column (always col 1 in our data; verified)
    yc = 1
    # col 2 = 'GCB'; for 2025 col 3 = 'GCB unadjusted'
    want_fixed = {'GCB': 2}
    has_unadj = False
    v3 = ws.cell(hr, 3).value
    if isinstance(v3, str) and 'unadjusted' in v3.lower():
        want_fixed['GCB_unadjusted'] = 3
        has_unadj = True
    # discover model columns
    skip = NONMODEL_COLS | {'GCB unadjusted'}
    model_cols = _scan_header_models(ws, hr, yc, skip_cols=skip)
    want = {**want_fixed, **model_cols}
    years, cols = _read_columns(ws, hr, yc, want)
    wb.close()
    gcb = cols['GCB']
    unadj = cols.get('GCB_unadjusted', None)
    models = {k: v for k, v in cols.items() if k not in ('GCB', 'GCB_unadjusted')}
    return years, gcb, unadj, models


def load_ocn_panel(vintage):
    """Return (years, GCB_published, GCB_unadjusted_or_None,
    {model_name: array}) for the Ocean Sink sheet."""
    hr = OCN_HEADER[vintage]
    wb, ws = _open_sheet(vintage, 'Ocean Sink')
    yc = 1
    want_fixed = {'GCB': 2}
    v3 = ws.cell(hr, 3).value
    if isinstance(v3, str) and 'unadjusted' in v3.lower():
        want_fixed['GCB_unadjusted'] = 3
    skip = NONMODEL_COLS | {'GCB unadjusted', '1 sigma uncertainty'}
    model_cols = _scan_header_models(ws, hr, yc, skip_cols=skip)
    want = {**want_fixed, **model_cols}
    years, cols = _read_columns(ws, hr, yc, want)
    wb.close()
    gcb = cols['GCB']
    unadj = cols.get('GCB_unadjusted', None)
    models = {k: v for k, v in cols.items()
              if k not in ('GCB', 'GCB_unadjusted')}
    return years, gcb, unadj, models


def load_luc_panel(vintage):
    """Return dict with keys
      year:   array
      gcb:    GCB-published E_LUC series
      bk:     {model_name: array}   bookkeeping ensemble
      peat:   {model_name: array}   peat panel (None if not present)
      dgvm:   {model_name: array}   DGVM-LUC alternative ensemble
    """
    cfg = LUC_CONFIG[vintage]
    wb, ws = _open_sheet(vintage, 'Land-Use Change Emissions')
    want = {'gcb': cfg['gcb_col']}
    for nm, c in cfg['bk_cols'].items():
        want[f'bk:{nm}'] = c
    if cfg['peat_cols']:
        for nm, c in cfg['peat_cols'].items():
            want[f'peat:{nm}'] = c
    for nm, c in cfg['dgvm_cols'].items():
        want[f'dgvm:{nm}'] = c

    # data starts at cfg['data_row']; _read_columns starts at
    # header_row + 1 so we shift effective header_row.
    eff_hr = cfg['data_row'] - 1
    years, cols = _read_columns(ws, eff_hr, cfg['year_col'], want)
    wb.close()
    # Apply DGVM sign convention flip (2022 and 2023 report
    # net land-atm flux with opposite sign).
    dgvm_sign = cfg.get('dgvm_luc_sign', 1)
    dgvm_dict = {k.split(':',1)[1]: dgvm_sign * v
                 for k, v in cols.items() if k.startswith('dgvm:')}
    out = {
        'year': years,
        'gcb': cols['gcb'],
        'bk':   {k.split(':',1)[1]: v for k, v in cols.items()
                 if k.startswith('bk:')},
        'peat': ({k.split(':',1)[1]: v for k, v in cols.items()
                  if k.startswith('peat:')} if cfg['peat_cols'] else None),
        'dgvm': dgvm_dict,
    }
    return out


def panel_mean(model_dict, restrict_to_year=None, years=None):
    """Compute the cross-model mean (axis=1) of a {name: array} dict.
    Missing values are nan-handled. If restrict_to_year supplied, also
    return the year-by-year mean only over years in restrict_to_year.
    """
    if not model_dict:
        return None
    M = np.column_stack(list(model_dict.values()))   # T × N_models
    return np.nanmean(M, axis=1)


# ──────────────────────────────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Smoke test on all 9 vintages:\n')
    print(f'{"vintage":>8} {"T_gcb":>5} {"T_lnd":>5} {"T_ocn":>5} {"T_luc":>5}  '
          f'{"#DGVM_LND":>9}  {"#BK":>4} {"#DGVM_LUC":>9} {"peat":>4}')
    for v in VINTAGES:
        try:
            g = load_gcb_aggregate(v)
            yL, gL, uL, mL = load_lnd_panel(v)
            yO, gO, uO, mO = load_ocn_panel(v)
            luc = load_luc_panel(v)
            print(f'{v:>8} {len(g["year"]):>5} {len(yL):>5} {len(yO):>5} '
                  f'{len(luc["year"]):>5}  '
                  f'{len(mL):>9}  {len(luc["bk"]):>4} '
                  f'{len(luc["dgvm"]):>9}  '
                  f'{"yes" if luc["peat"] else "no":>4}')
        except Exception as e:
            print(f'{v:>8}  ERROR: {e}')
