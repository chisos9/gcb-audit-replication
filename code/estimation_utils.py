"""
Shared estimation machinery for the MDS-GCB fits: a seeded multistart
with a degeneracy screen, and a step-checked numerical Hessian with a
relative eigenvalue floor.

Why a multistart. The likelihood is multimodal in the persistence
phi_1 for some vintages (2024 and 2025 in particular), with modes less
than one log-likelihood unit apart. A single warm start lands in one
mode or the other depending on machine and library versions. The
multistart polishes a fixed, seeded set of starting points, discards
degenerate fits, keeps the best survivor, and prints every distinct
mode it saw, so multimodality is visible in the log rather than
discovered by a referee.

Why the degeneracy screen. An apparent optimum near loglik -49 exists
where the Kalman filter breaks down: most observations contribute
exactly zero to the log-likelihood and the one-step forecast errors
explode to ~1e6 GtC. It beats every genuine fit on raw likelihood and
must be rejected on filter diagnostics, not likelihood value.

Why the guarded Hessian. The central-difference Hessian can have one
near-zero eigenvalue whose estimate varies across BLAS builds. With an
absolute eigenvalue floor, that direction turns into variance 1e8 on
one machine and something sane on another (observed: SE(mu_C) = 18.7
versus 0.08 for the 2021 vintage). The guard floors eigenvalues
relative to the largest one, compares two step sizes, and names any
parameter whose standard error rests on a floored direction.
"""
import numpy as np
from scipy.optimize import minimize

__all__ = ['multistart', 'is_degenerate', 'polish', 'robust_hessian',
           'hessian_fd', 'spectral_regularise_inverse', 'get_epsilon']


def _neg_loglik(params, model):
    return -model.loglike(params)


def polish(model, x0):
    """BFGS followed by a Nelder-Mead polish; returns (loglik, params)."""
    res = minimize(_neg_loglik, np.asarray(x0, dtype=float), args=(model,),
                   method='BFGS',
                   options={'maxiter': 4000, 'disp': False, 'gtol': 1e-6})
    nm = minimize(_neg_loglik, res.x, args=(model,), method='Nelder-Mead',
                  options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 20000,
                           'maxfev': 40000, 'adaptive': True, 'disp': False})
    best = nm if nm.fun < res.fun else res
    return -best.fun, best.x


def is_degenerate(model, params, fe_std_cap=50.0):
    """True when the Kalman filter has broken down at `params`.

    Two symptoms, either one disqualifies the fit:
    - one-step forecast-error standard deviations (after the diffuse
      burn-in, t >= 3) above `fe_std_cap` GtC, orders of magnitude
      beyond the data;
    - per-observation log-likelihood contributions that are exactly
      zero, which a Gaussian likelihood on real data never produces.
    """
    try:
        r = model.smooth(np.asarray(params, dtype=float))
    except Exception:
        return True
    fe = np.asarray(r.forecasts_error)
    if fe.shape[1] > 3 and np.nanstd(fe[:, 3:], axis=1).max() > fe_std_cap:
        return True
    if np.any(np.abs(np.asarray(r.llf_obs)) < 1e-12):
        return True
    return False


def multistart(model, anchors=(), seed=0, n_jitter=10, scale=0.4,
               verbose=True):
    """Polish a fixed, seeded set of starts; keep the best healthy fit.

    Starts are the model's own start_params, each anchor (a tuple of
    (name, parameter vector); anchors that are None are skipped), and
    `n_jitter` Gaussian perturbations (sd `scale`) cycling over those
    bases. The RNG is seeded, so every machine polishes the same
    starts. Degenerate fits (see is_degenerate) are rejected.

    Returns (loglik, params, modes) where modes lists the distinct
    local maxima found among healthy fits as (loglik, params), best
    first. More than one mode means the data cannot cleanly separate
    them; the gap between the top two says how much likelihood the
    lower mode gives up.
    """
    rng = np.random.default_rng(seed)
    bases = [('start_params', np.asarray(model.start_params, dtype=float))]
    for name, p in anchors:
        if p is not None:
            bases.append((name, np.asarray(p, dtype=float)))
    starts = list(bases)
    k = len(bases[0][1])
    for i in range(n_jitter):
        name, p = bases[i % len(bases)]
        starts.append((f'{name}+j{i}', p + rng.normal(scale=scale, size=k)))

    healthy = []
    for name, x0 in starts:
        try:
            ll, x = polish(model, x0)
        except Exception as e:
            if verbose:
                print(f'    start {name:<20} FAILED: {e}')
            continue
        deg = is_degenerate(model, x)
        if verbose:
            tag = '  DEGENERATE (rejected)' if deg else ''
            print(f'    start {name:<20} loglik={ll:10.4f}{tag}')
        if not deg:
            healthy.append((ll, x))
    if not healthy:
        raise RuntimeError('multistart: every start failed or was '
                           'degenerate; inspect the model or data')

    healthy.sort(key=lambda t: -t[0])
    modes = []
    for ll, x in healthy:
        # Optimiser scatter within one basin is far below 0.01 loglik
        # units here; distinct basins can sit 0.03 apart (2018), so
        # cluster on loglik at 0.01 and on parameter distance.
        if not any(abs(ll - ll0) < 0.01 and
                   np.max(np.abs(x - x0)) < 0.05
                   for ll0, x0 in modes):
            modes.append((ll, x))
    if verbose and len(modes) > 1:
        gaps = ', '.join(f'{ll:.3f}' for ll, _ in modes)
        print(f'    NOTE: {len(modes)} distinct modes found '
              f'(logliks {gaps}); keeping the highest. The data do not '
              f'cleanly separate these fits.')
    return modes[0][0], modes[0][1], modes


def get_epsilon(x):
    macheps = np.finfo(float).eps
    base = (macheps / 6.0) ** (1.0 / 3.0)
    return base * np.maximum(np.abs(x), 1.0)


def hessian_fd(f, x, step_mult=4.0):
    """Central-difference Hessian with step = step_mult * get_epsilon."""
    n = len(x)
    h = step_mult * get_epsilon(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x_pp = x.copy(); x_pp[i] += h[i]; x_pp[j] += h[j]
            x_pm = x.copy(); x_pm[i] += h[i]; x_pm[j] -= h[j]
            x_mp = x.copy(); x_mp[i] -= h[i]; x_mp[j] += h[j]
            x_mm = x.copy(); x_mm[i] -= h[i]; x_mm[j] -= h[j]
            v = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * h[i] * h[j])
            H[i, j] = v
            H[j, i] = v
    return H


def spectral_regularise_inverse(H, rel_floor=1e-6):
    """Eigendecompose, floor eigenvalues RELATIVE to the largest, invert.

    An absolute floor makes the implied variances depend on the overall
    likelihood scale and on which side of the floor a noisy eigenvalue
    lands; a relative floor does not.
    """
    H_sym = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(H_sym)
    floor = max(float(w.max()), 0.0) * rel_floor
    if floor <= 0.0:
        floor = 1e-8
    w_safe = np.maximum(w, floor)
    cov = V @ np.diag(1.0 / w_safe) @ V.T
    floored = w < floor
    return cov, w, V, floored


def robust_hessian(f, x, param_names=None, verbose=True):
    """Hessian of f at x with a step-size check and SE warnings.

    The Hessian is computed at step 4h (the convention used throughout
    the paper) and at 8h. When the 4h Hessian has floored eigenvalues
    and the 8h one does not, the 4h finite differences are in their
    noise floor and the 8h Hessian is kept; this is the failure mode
    that produced SE(mu_C) = 18.7 on one machine against 0.08 on
    another for the 2021 vintage. Otherwise the 4h Hessian is kept.
    Standard errors that still disagree by more than 20% between the
    two steps, or that rest mostly on floored directions, are named in
    a warning: those SEs are not identified by the curvature.

    Returns (H, cov, n_floored) for the chosen step.
    """
    runs = {}
    for mult in (4.0, 8.0):
        H = hessian_fd(f, x, step_mult=mult)
        cov, w, V, floored = spectral_regularise_inverse(H)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        runs[mult] = dict(H=H, cov=cov, w=w, V=V, floored=floored, se=se)

    n4 = int(runs[4.0]['floored'].sum())
    n8 = int(runs[8.0]['floored'].sum())
    pick = 4.0
    if n4 > n8:
        pick = 8.0
        if verbose:
            print(f'    NOTE: the 4h Hessian has {n4} floored '
                  f'eigenvalue(s) and the 8h Hessian has {n8}; keeping '
                  f'the 8h step. The 4h finite differences are in their '
                  f'noise floor at this optimum.')

    se4, se8 = runs[4.0]['se'], runs[8.0]['se']
    denom = np.maximum(np.maximum(se4, se8), 1e-300)
    rel = np.abs(se4 - se8) / denom
    bad = np.where(rel > 0.20)[0]
    # When 4h was escalated away for being in its noise floor, the
    # disagreement against it carries no extra information.
    if verbose and bad.size and pick == 4.0:
        names = ', '.join(param_names[i] if param_names else f'param {i}'
                          for i in bad)
        print(f'    WARNING: SEs at steps 4h and 8h disagree by more '
              f'than 20% for: {names} (worst '
              f'{100 * float(np.nanmax(rel)):.0f}%). Treat those SEs '
              f'with caution; the estimates themselves are unaffected.')

    r = runs[pick]
    if r['floored'].any():
        w_safe = np.maximum(r['w'], max(float(r['w'].max()), 0.0) * 1e-6)
        var_total = (r['V'] ** 2 / w_safe).sum(axis=1)
        var_floor = (r['V'][:, r['floored']] ** 2
                     / w_safe[r['floored']]).sum(axis=1)
        frac = var_floor / np.maximum(var_total, 1e-300)
        for i in np.where(frac > 0.5)[0]:
            nm = param_names[i] if param_names else f'parameter {i}'
            if verbose:
                print(f'    WARNING: the SE of {nm} rests on a floored '
                      f'Hessian direction ({100 * frac[i]:.0f}% of its '
                      f'variance); it is not identified by the '
                      f'curvature and should not be quoted.')
    return r['H'], r['cov'], int(r['floored'].sum())
