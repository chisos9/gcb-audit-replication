"""
JRSSA 4-series model with SBM-style climate covariates.

This is the model used in the model-based audit.
Built on the lean variant of the Bennedsen, Hillebrand, Koopman (JRSSA 2023)
state-space framework: 4 measurement series (cumulative atmospheric CO2,
total emissions, land sink, ocean sink), 12-state vector following Model 1
of the JRSSA paper (not Model 2 - GDP loading and 1991/1997 dummies are
dropped, as decided 2026-05-26).

Climate covariates replace the single SOI loading of the JRSSA paper with
the four-variable set from the SBM: ENSO 3.4, NAO, PDO, and SAOD. Each
enters the land- and ocean-sink state equations with its own loading.

State vector (12 elements):
   0: C*       atmospheric CO2 stock (latent)
   1: G_ATM*   atmospheric growth (latent)
   2: S_LND*   land sink (latent)
   3: S_OCN*   ocean sink (latent)
   4: X1       AR(1) residual in C equation
   5: X2       zero-mean residual in S_LND equation
   6: X3       AR(1) residual in S_OCN equation
   7: XE       AR(1) residual in E* trend
   8: E*       emissions trend (random walk with drift)
   9: c1       constant state, land-sink intercept
  10: c2       constant state, ocean-sink intercept
  11: d        constant state, emissions drift

Measurement equations:
   C_reg_t = C*_t + X1_t
   E_t     = E*_t + eps_E_t,    eps_E_t ~ N(0, sigma2_epsE) i.i.d.
   S_LND_t = S_LND*_t + X2_t
   S_OCN_t = S_OCN*_t + X3_t

(The AR(1) measurement residual X4 on E that was in the previous version
has been removed; emissions measurement error is now plain Gaussian via
obs_cov[1,1].)

Sink state equations (climate forcing enters at t+1):
   S_LND*_{t+1} = c1 + beta1*C*_{t+1}/cGtC1750
                + b1_enso*ENSO_{t+1} + b1_nao*NAO_{t+1}
                + b1_pdo*PDO_{t+1}  + b1_saod*SAOD_{t+1}
   S_OCN*_{t+1} = c2 + beta2*C*_{t+1}/cGtC1750
                + b2_enso*ENSO_{t+1} + b2_nao*NAO_{t+1}
                + b2_pdo*PDO_{t+1}  + b2_saod*SAOD_{t+1}

Free parameters (20 total):
  0: p_beta1   -> beta1 = 10*sigmoid(-p) / cGtC1750
  1: p_beta2   -> beta2 = 10*sigmoid(-p) / cGtC1750
  2: p_phi1    -> phi1  = sigmoid(-p)
  3: p_phi3    -> phi3  = sigmoid(-p)
  4: p_phiE    -> phiE  = sigmoid(-p)
  5: p_sig1    -> sigma2_eta1 = exp(p)
  6: p_sig2    -> sigma2_eta2 = exp(p)
  7: p_sig3    -> sigma2_eta3 = exp(p)
  8: p_sigE    -> sigma2_etaE = exp(p)
  9: p_r12     -> r12 = tanh(p/2)
 10: p_r13     -> r13 = tanh(p/2)
 11: p_sigepsE -> sigma2_epsE = exp(p)
 12: b1_enso   (unconstrained)
 13: b2_enso   (unconstrained)
 14: b1_nao    (unconstrained)
 15: b2_nao    (unconstrained)
 16: b1_pdo    (unconstrained)
 17: b2_pdo    (unconstrained)
 18: b1_saod   (unconstrained)
 19: b2_saod   (unconstrained)
"""

import os
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.mlemodel import MLEModel
import statsmodels.tsa.statespace.kalman_filter as kf_const

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CODE_DIR, '..', 'data')

cGtC1959 = 2.127 * 315.39
cGtC1750 = 2.127 * 279.0

cN = 4    # measurement variables
cr = 12   # state variables
N_CLIMATE = 4   # ENSO, NAO, PDO, SAOD


# ─── Data loaders ──────────────────────────────────────────────────────

def _load_climate_shifted(years_t, sample_end_year):
    """Load the four climate covariates at year t+1 for each t in years_t.

    Parameters
    ----------
    years_t : array of int, length cT
        Sample years for the GCB observations (e.g., 1959..2023).
    sample_end_year : int
        Last GCB year. SAOD must cover sample_end_year + 1.
    """
    shifted = (years_t + 1).astype(int)
    needed_lo, needed_hi = shifted[0], shifted[-1]

    # Climate aggregation:
    # ENSO Niño 3.4 as Sep(t-1)–Aug(t) mean (4-month lag relative to the
    # calendar year), harmonised with the convention used in our companion work
    # and the companion natural-science submission. This captures the
    # lagged CO2 growth-rate response to ENSO; it replaces the earlier
    # Jul(t-1)-Jun(t) (6-month-lag) literature-standard window (the
    # Jul-Jun version is retained at climate_indices_annual_litstd_*.csv.jul_jun_backup).
    # NAO as DJFM (Dec(t-1)+Jan(t)+Feb(t)+Mar(t)), the winter NAO of
    # Hurrell (1995) -- deliberately kept distinct from the companion work's
    # calendar-year NAO.
    # PDO and SOI: calendar-year mean (PDO is decadal so window choice
    # is immaterial; SOI kept for backward compatibility).
    climate = pd.read_csv(os.path.join(DATA_DIR, 'climate_indices_annual_litstd_1959_2024.csv'))
    cy = climate['Year'].values.astype(int)
    yr_to_idx = {int(y): i for i, y in enumerate(cy)}
    if not (needed_lo in yr_to_idx and needed_hi in yr_to_idx):
        raise ValueError(f"Climate indices CSV missing range {needed_lo}-{needed_hi}")
    enso = np.array([climate['ENSO_Nino34'].values[yr_to_idx[y]] for y in shifted])
    nao  = np.array([climate['NAO'].values[yr_to_idx[y]] for y in shifted])
    pdo  = np.array([climate['PDO'].values[yr_to_idx[y]] for y in shifted])

    saod = pd.read_csv(os.path.join(DATA_DIR, 'saod_annual_1959_2024.csv'))
    sy = saod['year'].values.astype(int)
    sy_to_idx = {int(y): i for i, y in enumerate(sy)}
    if not (needed_lo in sy_to_idx and needed_hi in sy_to_idx):
        raise ValueError(f"SAOD CSV missing range {needed_lo}-{needed_hi}")
    saod_arr = np.array([saod['SAOD'].values[sy_to_idx[y]] for y in shifted])

    return enso, nao, pdo, saod_arr


def load_vintage_2024(year_end=2023):
    """Load GCB 2024 v1.0 four-series observations, plus climate shifted by +1.

    Estimation sample is 1959 to year_end (default 2023, T=65).
    """
    xls = os.path.join(DATA_DIR, 'Global_Carbon_Budget_2024v1.0.xlsx')
    gcb = pd.read_excel(xls, sheet_name='Global Carbon Budget', header=21)
    gcb = gcb[(gcb['Year'] >= 1959) & (gcb['Year'] <= year_end)
              ].reset_index(drop=True)
    E_FF = gcb['fossil emissions excluding carbonation'].values.astype(float)
    E_LUC = gcb['land-use change emissions'].values.astype(float)
    G_ATM = gcb['atmospheric growth'].values.astype(float)
    S_LND = gcb['land sink'].values.astype(float)
    S_OCN = gcb['ocean sink'].values.astype(float)
    years = gcb['Year'].values.astype(int)

    S_CARB = gcb['cement carbonation sink'].values.astype(float)
    E = E_FF + E_LUC - S_CARB
    vCreg = cGtC1959 + np.cumsum(G_ATM)
    mObs = np.column_stack([vCreg, E, S_LND, S_OCN])

    enso, nao, pdo, saod = _load_climate_shifted(years, year_end)
    return {
        'mObs': mObs,
        'years': years,
        'E_FF': E_FF, 'E_LUC': E_LUC, 'G_ATM': G_ATM,
        'S_LND': S_LND, 'S_OCN': S_OCN, 'S_CARB': S_CARB, 'vCreg': vCreg,
        'enso_shifted': enso, 'nao_shifted': nao,
        'pdo_shifted': pdo, 'saod_shifted': saod,
        'cT': len(gcb),
    }


def load_vintage_2025(year_end=2023):
    """Load GCB 2025 v0.7 four-series observations, plus climate shifted by +1.

    Estimation sample defaults to 1959-2023 for direct comparability with the
    2024 vintage. Pass year_end=2024 to use the full 2025 sample (T=66).
    """
    xls = os.path.join(DATA_DIR, 'Global_Carbon_Budget_2025_v0.7.xlsx')
    gcb = pd.read_excel(xls, sheet_name='Global Carbon Budget', header=21)
    gcb = gcb[(gcb['Year'] >= 1959) & (gcb['Year'] <= year_end)
              ].reset_index(drop=True)
    E_FF = gcb['fossil emissions excluding carbonation'].values.astype(float)
    E_LUC = gcb['land-use change emissions'].values.astype(float)
    G_ATM = gcb['atmospheric growth'].values.astype(float)
    S_LND = gcb['land sink'].values.astype(float)   # GCB-pref (adjusted)
    S_OCN = gcb['ocean sink'].values.astype(float)
    years = gcb['Year'].values.astype(int)

    S_CARB = gcb['cement carbonation sink'].values.astype(float)
    E = E_FF + E_LUC - S_CARB
    vCreg = cGtC1959 + np.cumsum(G_ATM)
    mObs = np.column_stack([vCreg, E, S_LND, S_OCN])

    enso, nao, pdo, saod = _load_climate_shifted(years, year_end)
    return {
        'mObs': mObs,
        'years': years,
        'E_FF': E_FF, 'E_LUC': E_LUC, 'G_ATM': G_ATM,
        'S_LND': S_LND, 'S_OCN': S_OCN, 'S_CARB': S_CARB, 'vCreg': vCreg,
        'enso_shifted': enso, 'nao_shifted': nao,
        'pdo_shifted': pdo, 'saod_shifted': saod,
        'cT': len(gcb),
    }


# ─── Model class ───────────────────────────────────────────────────────

class JRSSAClimate(MLEModel):
    """JRSSA 4-series model with SBM-style climate covariates."""

    def __init__(self, endog, enso_shifted, nao_shifted, pdo_shifted, saod_shifted):
        k_states = cr
        k_posdef = 4   # 4 state disturbances: eta1, eta2, eta3, etaE

        super().__init__(
            endog,
            k_states=k_states,
            k_posdef=k_posdef,
            initialization='known',
            initial_state=np.zeros(k_states),
            initial_state_cov=self._build_init_cov(k_states),
        )

        self.cGtC1750 = cGtC1750
        self.enso_shifted = np.asarray(enso_shifted, dtype=float)
        self.nao_shifted = np.asarray(nao_shifted, dtype=float)
        self.pdo_shifted = np.asarray(pdo_shifted, dtype=float)
        self.saod_shifted = np.asarray(saod_shifted, dtype=float)

        # Selection matrix (k_states x k_posdef) maps 4 disturbances
        # to states 4, 5, 6, 7 (X1, X2, X3, XE)
        self['selection'] = np.zeros((k_states, k_posdef))
        self['selection', 4, 0] = 1.0
        self['selection', 5, 1] = 1.0
        self['selection', 6, 2] = 1.0
        self['selection', 7, 3] = 1.0

        # obs_cov[1,1] = sigma2_epsE is set in update() from the free parameter.
        self['obs_cov'] = np.zeros((cN, cN))
        self.ssm.set_filter_method(kf_const.FILTER_UNIVARIATE)

    @staticmethod
    def _build_init_cov(k_states):
        P1 = 1_000_000.0 * np.eye(k_states)
        P1[4, 4] = 1.5
        P1[5, 5] = 0.50
        P1[6, 6] = 0.02
        P1[7, 7] = 3.0   # PinitXE
        return P1

    @property
    def param_names(self):
        return [
            'p_beta1', 'p_beta2', 'p_phi1', 'p_phi3', 'p_phiE',
            'p_sig1', 'p_sig2', 'p_sig3', 'p_sigE',
            'p_r12', 'p_r13', 'p_sigepsE',
            'b1_enso', 'b2_enso', 'b1_nao', 'b2_nao',
            'b1_pdo', 'b2_pdo', 'b1_saod', 'b2_saod',
        ]

    @property
    def start_params(self):
        # Reasonable starting values; will be refined by optimization.
        return np.array([
            -0.4, -0.2,                 # p_beta1, p_beta2
            -2.5, -0.8, 5.0,            # p_phi1, p_phi3, p_phiE
            -0.1, -0.4, -4.4, -3.5,     # p_sig1, p_sig2, p_sig3, p_sigE
            -1.8, -0.4,                 # p_r12, p_r13
            -3.0,                       # p_sigepsE -> sigma2_epsE ~ 0.05
            0.0, 0.0, 0.0, 0.0,         # b1_enso, b2_enso, b1_nao, b2_nao
            0.0, 0.0, 0.0, 0.0,         # b1_pdo, b2_pdo, b1_saod, b2_saod
        ])

    @staticmethod
    def transform_params_ssm(pars, cGtC1750):
        sigmoid = lambda x: np.exp(-x) / (1.0 + np.exp(-x))
        tanh_half = lambda x: (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))
        return {
            'beta1': 10.0 * sigmoid(pars[0]) / cGtC1750,
            'beta2': 10.0 * sigmoid(pars[1]) / cGtC1750,
            'phi1': sigmoid(pars[2]),
            'phi3': sigmoid(pars[3]),
            'phiE': sigmoid(pars[4]),
            'sigma2_eta1': np.exp(pars[5]),
            'sigma2_eta2': np.exp(pars[6]),
            'sigma2_eta3': np.exp(pars[7]),
            'sigma2_etaE': np.exp(pars[8]),
            'r12': tanh_half(pars[9]),
            'r13': tanh_half(pars[10]),
            'sigma2_epsE': np.exp(pars[11]),
        }

    @staticmethod
    def transform_params_display(pars):
        sigmoid = lambda x: np.exp(-x) / (1.0 + np.exp(-x))
        tanh_half = lambda x: (1.0 - np.exp(-x)) / (1.0 + np.exp(-x))
        d = {
            'beta1': 10.0 * sigmoid(pars[0]),
            'beta2': 10.0 * sigmoid(pars[1]),
            'phi1': sigmoid(pars[2]),
            'phi3': sigmoid(pars[3]),
            'phiE': sigmoid(pars[4]),
            'sigma2_eta1': np.exp(pars[5]),
            'sigma2_eta2': np.exp(pars[6]),
            'sigma2_eta3': np.exp(pars[7]),
            'sigma2_etaE': np.exp(pars[8]),
            'r12': tanh_half(pars[9]),
            'r13': tanh_half(pars[10]),
            'sigma2_epsE': np.exp(pars[11]),
        }
        if len(pars) >= 20:
            d['b1_enso'] = float(np.real(pars[12]))
            d['b2_enso'] = float(np.real(pars[13]))
            d['b1_nao']  = float(np.real(pars[14]))
            d['b2_nao']  = float(np.real(pars[15]))
            d['b1_pdo']  = float(np.real(pars[16]))
            d['b2_pdo']  = float(np.real(pars[17]))
            d['b1_saod'] = float(np.real(pars[18]))
            d['b2_saod'] = float(np.real(pars[19]))
        return d

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        params = np.real(params)

        p = self.transform_params_ssm(params[:12], self.cGtC1750)
        beta1 = p['beta1']; beta2 = p['beta2']
        phi1 = p['phi1'];   phi3 = p['phi3']
        phiE = p['phiE']
        sigma2_1 = p['sigma2_eta1']; sigma2_2 = p['sigma2_eta2']
        sigma2_3 = p['sigma2_eta3']; sigma2_E = p['sigma2_etaE']
        r12 = p['r12'];     r13 = p['r13']
        sigma2_epsE = p['sigma2_epsE']

        b1_enso = params[12]; b2_enso = params[13]
        b1_nao  = params[14]; b2_nao  = params[15]
        b1_pdo  = params[16]; b2_pdo  = params[17]
        b1_saod = params[18]; b2_saod = params[19]

        cInv = 1.0 / (1.0 + beta1 + beta2)

        # ----- Design matrix Z (cN x cr) -----
        Z = np.zeros((cN, cr))
        Z[0, 0] = 1.0; Z[0, 4] = 1.0    # C_reg = C* + X1
        Z[1, 8] = 1.0                   # E     = E* + eps_E  (eps_E in obs_cov)
        Z[2, 2] = 1.0; Z[2, 5] = 1.0    # S_LND = S_LND* + X2
        Z[3, 3] = 1.0; Z[3, 6] = 1.0    # S_OCN = S_OCN* + X3
        self['design'] = Z

        # ----- Observation covariance H (cN x cN) -----
        H = np.zeros((cN, cN))
        H[1, 1] = sigma2_epsE
        self['obs_cov'] = H

        # ----- Transition matrix T (cr x cr) -----
        T = np.zeros((cr, cr))
        # Row 0: C*_{t+1} = cInv * (C*_t + XE_t + E*_t + d - c1 - c2)
        T[0, 0] = cInv;  T[0, 7] = cInv;  T[0, 8] = cInv
        T[0, 9] = -cInv; T[0, 10] = -cInv; T[0, 11] = cInv
        # Row 1: G_ATM*_{t+1}
        T[1, 0] = -(beta1 + beta2) * cInv
        T[1, 7] = cInv;  T[1, 8] = cInv;  T[1, 11] = cInv
        T[1, 9] = -cInv; T[1, 10] = -cInv
        # Row 2: S_LND*_{t+1}
        T[2, 0] = beta1 * cInv
        T[2, 7] = beta1 * cInv; T[2, 8] = beta1 * cInv; T[2, 11] = beta1 * cInv
        T[2, 9] = (1.0 + beta2) * cInv
        T[2, 10] = -beta1 * cInv
        # Row 3: S_OCN*_{t+1}
        T[3, 0] = beta2 * cInv
        T[3, 7] = beta2 * cInv; T[3, 8] = beta2 * cInv; T[3, 11] = beta2 * cInv
        T[3, 9] = -beta2 * cInv
        T[3, 10] = (1.0 + beta1) * cInv
        # Rows 4-6: measurement residuals
        T[4, 4] = phi1
        # T[5, :] = 0  (X2 is white noise)
        T[6, 6] = phi3
        # Row 7: XE_{t+1} = phiE * XE_t + etaE
        T[7, 7] = phiE
        # Row 8: E*_{t+1} = E*_t + XE_t + d
        T[8, 7] = 1.0; T[8, 8] = 1.0; T[8, 11] = 1.0
        # Constant states
        T[9, 9] = 1.0; T[10, 10] = 1.0; T[11, 11] = 1.0
        self['transition'] = T

        # ----- State disturbance covariance Q (4 x 4) -----
        Q = np.zeros((4, 4))
        Q[0, 0] = sigma2_1
        Q[1, 1] = sigma2_2
        Q[2, 2] = sigma2_3
        Q[3, 3] = sigma2_E
        Q[0, 1] = Q[1, 0] = r12 * np.sqrt(sigma2_1 * sigma2_2)
        Q[0, 2] = Q[2, 0] = r13 * np.sqrt(sigma2_1 * sigma2_3)
        self['state_cov'] = Q

        # ----- Time-varying state intercept from climate covariates -----
        def _climate_coeff(b1, b2):
            coeff = np.zeros(cr)
            B = b1 + b2
            coeff[0] = -cInv * B
            coeff[1] = -cInv * B
            coeff[2] = cInv * ((1.0 + beta2) * b1 - beta1 * b2)
            coeff[3] = cInv * (-beta2 * b1 + (1.0 + beta1) * b2)
            return coeff

        nobs = self.nobs
        coeff_enso = _climate_coeff(b1_enso, b2_enso)
        coeff_nao  = _climate_coeff(b1_nao,  b2_nao)
        coeff_pdo  = _climate_coeff(b1_pdo,  b2_pdo)
        coeff_saod = _climate_coeff(b1_saod, b2_saod)

        state_intercept = (
            np.outer(coeff_enso, self.enso_shifted[:nobs]) +
            np.outer(coeff_nao,  self.nao_shifted[:nobs]) +
            np.outer(coeff_pdo,  self.pdo_shifted[:nobs]) +
            np.outer(coeff_saod, self.saod_shifted[:nobs])
        )
        self['state_intercept'] = state_intercept


# ─── Quick smoke test ──────────────────────────────────────────────────

if __name__ == '__main__':
    print("Smoke test: load both vintages and verify model construction.\n")
    for label, loader in [('2024', load_vintage_2024), ('2025', load_vintage_2025)]:
        d = loader(year_end=2023)
        print(f"  {label}: cT={d['cT']}, years {d['years'][0]}-{d['years'][-1]}, "
              f"E mean={d['mObs'][:,1].mean():.3f}, "
              f"SAOD mean (shifted)={d['saod_shifted'].mean():.4f}")
        m = JRSSAClimate(d['mObs'],
                          d['enso_shifted'], d['nao_shifted'],
                          d['pdo_shifted'], d['saod_shifted'])
        ll = m.loglike(m.start_params)
        print(f"        loglik at start_params = {ll:.4f}")
    print("\nOK.")
