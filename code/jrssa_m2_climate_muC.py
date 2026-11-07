"""
JRSSA 4-series model with SBM-style climate covariates AND a constant
offset mu_C on the cumulative-CO2 (budget-identity) state equation.
Vintage-specific variant for the 2025 audit.

Motivation
----------
The baseline JRSSAClimate imposes the carbon-budget identity exactly as a
deterministic state equation on C*. When the published vintage carries a
persistent budget imbalance (BIM != 0 on average, as the 2025 vintage does
with BIM_GCB = +0.61 GtC/yr on 1959-2016), the smoother is forced to
deform the latent states to make the budget close. In the baseline 2025
fit this routes the BIM almost entirely through the ocean-sink AR(1)
measurement residual: phi_3 is driven to its sigmoid boundary (0.998) and
the smoothed S^{OCN*} sits 0.57 GtC/yr above the published ocean panel
across the whole sample.

This variant adds a constant offset mu_C to the C state equation:

    C*_{t+1} = C*_t + X^E_t + E*_t + d - S^{LND*}_{t+1} - S^{OCN*}_{t+1}
               + mu_C

mu_C is unrestricted (any sign). The sink equations and all other
transitions are unchanged. With mu_C present, the budget identity
becomes "closes up to a constant offset"; the smoother allocates the
persistent BIM mean to mu_C and leaves the sink measurement residuals
free to do their original job.

We tried earlier a variant that added both mu_C and a Gaussian innovation
eta^C ~ N(0, sigma^2_etaC) to the C state equation. On the 2025 vintage
the optimiser drove sigma^2_etaC -> 8.5e-8 (essentially zero), indicating
that once the BIM mean is taken out by mu_C, no year-to-year variance is
left for eta^C to absorb. We therefore drop eta^C and keep only the
constant.

Parameter vector
----------------
Length 21 (was 20). One new parameter appended:

  0..11  same as JRSSAClimate
  12..19 climate loadings (same as JRSSAClimate)
  20     mu_C (unconstrained constant)

Selection matrix is unchanged (k_posdef=4); mu_C enters only through
the state_intercept. State-disturbance cov Q remains 4x4.
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
N_CLIMATE = 4


class JRSSAClimateMuC(MLEModel):
    """JRSSAClimate with a constant offset mu_C on the C state equation."""

    def __init__(self, endog, enso_shifted, nao_shifted,
                 pdo_shifted, saod_shifted):
        k_states = cr
        k_posdef = 4   # same as baseline: eta1, eta2, eta3, etaE

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

        # Same selection as baseline.
        self['selection'] = np.zeros((k_states, k_posdef))
        self['selection', 4, 0] = 1.0
        self['selection', 5, 1] = 1.0
        self['selection', 6, 2] = 1.0
        self['selection', 7, 3] = 1.0

        self['obs_cov'] = np.zeros((cN, cN))
        self.ssm.set_filter_method(kf_const.FILTER_UNIVARIATE)

    @staticmethod
    def _build_init_cov(k_states):
        P1 = 1_000_000.0 * np.eye(k_states)
        P1[4, 4] = 1.5
        P1[5, 5] = 0.50
        P1[6, 6] = 0.02
        P1[7, 7] = 3.0
        return P1

    @property
    def param_names(self):
        return [
            'p_beta1', 'p_beta2', 'p_phi1', 'p_phi3', 'p_phiE',
            'p_sig1', 'p_sig2', 'p_sig3', 'p_sigE',
            'p_r12', 'p_r13', 'p_sigepsE',
            'b1_enso', 'b2_enso', 'b1_nao', 'b2_nao',
            'b1_pdo', 'b2_pdo', 'b1_saod', 'b2_saod',
            'mu_C',
        ]

    @property
    def start_params(self):
        return np.array([
            -0.4, -0.2,                 # p_beta1, p_beta2
            -2.5, -0.8, 5.0,            # p_phi1, p_phi3, p_phiE
            -0.1, -0.4, -4.4, -3.5,     # p_sig1, p_sig2, p_sig3, p_sigE
            -1.8, -0.4,                 # p_r12, p_r13
            -3.0,                       # p_sigepsE
            0.0, 0.0, 0.0, 0.0,         # b1_enso, b2_enso, b1_nao, b2_nao
            0.0, 0.0, 0.0, 0.0,         # b1_pdo, b2_pdo, b1_saod, b2_saod
            0.0,                        # mu_C
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
            'mu_C': float(np.real(pars[20])),
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
        if len(pars) >= 21:
            d['mu_C'] = float(np.real(pars[20]))
        return d

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        params = np.real(params)

        p = self.transform_params_ssm(params[:21], self.cGtC1750)
        beta1 = p['beta1']; beta2 = p['beta2']
        phi1 = p['phi1'];   phi3 = p['phi3']
        phiE = p['phiE']
        sigma2_1 = p['sigma2_eta1']; sigma2_2 = p['sigma2_eta2']
        sigma2_3 = p['sigma2_eta3']; sigma2_E = p['sigma2_etaE']
        mu_C = p['mu_C']
        r12 = p['r12'];     r13 = p['r13']
        sigma2_epsE = p['sigma2_epsE']

        b1_enso = params[12]; b2_enso = params[13]
        b1_nao  = params[14]; b2_nao  = params[15]
        b1_pdo  = params[16]; b2_pdo  = params[17]
        b1_saod = params[18]; b2_saod = params[19]

        cInv = 1.0 / (1.0 + beta1 + beta2)

        # ----- Design matrix Z (unchanged) -----
        Z = np.zeros((cN, cr))
        Z[0, 0] = 1.0; Z[0, 4] = 1.0
        Z[1, 8] = 1.0
        Z[2, 2] = 1.0; Z[2, 5] = 1.0
        Z[3, 3] = 1.0; Z[3, 6] = 1.0
        self['design'] = Z

        H = np.zeros((cN, cN))
        H[1, 1] = sigma2_epsE
        self['obs_cov'] = H

        # ----- Transition matrix T (unchanged) -----
        T = np.zeros((cr, cr))
        T[0, 0] = cInv;  T[0, 7] = cInv;  T[0, 8] = cInv
        T[0, 9] = -cInv; T[0, 10] = -cInv; T[0, 11] = cInv
        T[1, 0] = -(beta1 + beta2) * cInv
        T[1, 7] = cInv;  T[1, 8] = cInv;  T[1, 11] = cInv
        T[1, 9] = -cInv; T[1, 10] = -cInv
        T[2, 0] = beta1 * cInv
        T[2, 7] = beta1 * cInv; T[2, 8] = beta1 * cInv; T[2, 11] = beta1 * cInv
        T[2, 9] = (1.0 + beta2) * cInv
        T[2, 10] = -beta1 * cInv
        T[3, 0] = beta2 * cInv
        T[3, 7] = beta2 * cInv; T[3, 8] = beta2 * cInv; T[3, 11] = beta2 * cInv
        T[3, 9] = -beta2 * cInv
        T[3, 10] = (1.0 + beta1) * cInv
        T[4, 4] = phi1
        T[6, 6] = phi3
        T[7, 7] = phiE
        T[8, 7] = 1.0; T[8, 8] = 1.0; T[8, 11] = 1.0
        T[9, 9] = 1.0; T[10, 10] = 1.0; T[11, 11] = 1.0
        self['transition'] = T

        # ----- State-disturbance covariance Q (4x4, unchanged) -----
        Q = np.zeros((4, 4))
        Q[0, 0] = sigma2_1
        Q[1, 1] = sigma2_2
        Q[2, 2] = sigma2_3
        Q[3, 3] = sigma2_E
        Q[0, 1] = Q[1, 0] = r12 * np.sqrt(sigma2_1 * sigma2_2)
        Q[0, 2] = Q[2, 0] = r13 * np.sqrt(sigma2_1 * sigma2_3)
        self['state_cov'] = Q

        # ----- Time-varying state intercept from climate covariates,
        #       plus constant contribution from mu_C -----
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

        # mu_C enters (C*, G_ATM*, S^LND*, S^OCN*) with the same
        # cInv * (1, 1, beta1, beta2) algebra as E*.
        if mu_C != 0.0:
            state_intercept[0, :] += cInv * mu_C
            state_intercept[1, :] += cInv * mu_C
            state_intercept[2, :] += beta1 * cInv * mu_C
            state_intercept[3, :] += beta2 * cInv * mu_C

        self['state_intercept'] = state_intercept


# ─── Quick smoke test ──────────────────────────────────────────────────

if __name__ == '__main__':
    from load_jrssa_vintage import load_vintage_for_jrssa
    print("Smoke test: load 2025 vintage and verify mu_C variant.\n")
    d = load_vintage_for_jrssa('2025', 1959, 2016)
    m = JRSSAClimateMuC(d['mObs'],
                        d['enso_shifted'], d['nao_shifted'],
                        d['pdo_shifted'], d['saod_shifted'])
    print(f"  cT={d['cT']}, params length={len(m.start_params)}")
    print(f"  loglik at start_params = {m.loglike(m.start_params):.4f}")
    print("OK.")
