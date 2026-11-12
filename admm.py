"""Deterministic consensus ADMM for the elastic St-MP-DCOPF model.

Only the initial generation block is learned. Exact and hybrid runs return
raw terminal states and complete residual histories for independent replay.
"""
import time
import numpy as np
from qp_ipm import BatchedIPM


class Problem:
    def __init__(self, case, scen, S, T, rho):
        self.case, self.S, self.T, self.rho = case, S, T, rho
        self.G = G = len(case["Pmax"])
        self.D = scen["load"][:S].sum(axis=2).T.copy()
        self.W = scen["wind"][:S].T.copy()
        self.p_prev = scen["p_prev"]
        self.mask = np.arange(G)
        self.Gc = G
        ramp = 0.3 * case["Pmax"]
        pi = 1.0 / S
        self.base_w = np.tile(2.0 * pi * case["c2"], T)
        w = self.base_w.copy()
        w[:G] += rho
        lo = np.tile(case["Pmin"], T)
        hi = np.tile(case["Pmax"], T)
        lo[:G] = np.maximum(case["Pmin"], scen["p_prev"] - ramp)
        hi[:G] = np.minimum(case["Pmax"], scen["p_prev"] + ramp)
        self.q_cost = np.tile(pi * case["c1"], T)
        self.c_sh_abs, self.c_cur_abs = 1.0e4, 1.0e2
        # The implemented objective is expected generation cost plus the sum
        # of absolute per-scenario elastic penalties. Thus each scenario has
        # the same marginal VOLL/curtailment penalty; these terms are not
        # probability-weighted. The manuscript states this convention exactly.
        self.qp = BatchedIPM(G, T, w, lo, hi, ramp,
                             c_sh=self.c_sh_abs, c_cur=self.c_cur_abs)
        self.x_lo, self.x_hi = lo[:G].copy(), hi[:G].copy()
        self.ramp = ramp
        self.load = scen["load"][:S]
        self.last_qp_info = None

    def z_solve(self, x, lam, state0=None, ret_state=False):
        S, G = self.S, self.G
        q = np.repeat(self.q_cost[:, None], S, axis=1)
        q[:G] = q[:G] - lam.T - self.rho * x[:, None]
        if ret_state:
            p, info, state = self.qp.solve(q, self.D, self.D, self.W,
                                           tol_mu=1e-11, tol_res=1e-9,
                                           max_iter=100, state0=state0,
                                           ret_state=True)
            self.last_qp_info = info
            return p, info, state
        p, info = self.qp.solve(q, self.D, self.D, self.W, state0=state0,
                                tol_mu=1e-11, tol_res=1e-9, max_iter=100)
        self.last_qp_info = info
        return p


def x_update(prob, z0, lam):
    return np.clip((z0 - lam / prob.rho).mean(axis=0), prob.x_lo, prob.x_hi)


def residuals(prob, x_new, x_old, z0_new):
    r = np.linalg.norm(z0_new - x_new[None, :], axis=1).max()
    s = prob.rho * np.linalg.norm(x_new - x_old) if x_old is not None else np.inf
    return float(r), float(s)


def run_exact(prob, x0=None, lam0=None, eps_r=1e-4, eps_s=1e-4,
              max_iter=2000, hist=True, z0_init=None, verbose=False,
              warm=False):
    S = prob.S
    x = prob.x_lo + 0.5 * (prob.x_hi - prob.x_lo) if x0 is None else x0.copy()
    lam = np.zeros((S, prob.Gc)) if lam0 is None else lam0.copy()
    z0 = np.repeat(x[None, :], S, axis=0) if z0_init is None else z0_init.copy()
    t0 = time.perf_counter()
    history, x_prev, state, qp_infos = [], None, None, []
    r = s = np.inf
    p = np.empty((prob.G * prob.T, S))
    qp_info = None
    for k in range(1, max_iter + 1):
        x_prev = x
        x = x_update(prob, z0, lam)
        if warm:
            p, qp_info, state = prob.z_solve(x, lam, state0=state, ret_state=True)
        else:
            p = prob.z_solve(x, lam)
            qp_info = prob.last_qp_info
        qp_infos.append(dict(qp_info))
        z0 = p[:prob.G].T
        lam = lam + prob.rho * (x[None, :] - z0)
        r, s = residuals(prob, x, x_prev, z0)
        if hist:
            history.append((r, s))
        if verbose and (k % 10 == 0 or k < 4):
            print(f"    admm k={k:4d} r={r:.3e} s={s:.3e}", flush=True)
        if r <= eps_r and s <= eps_s:
            break
    wall = time.perf_counter() - t0
    converged = bool(r <= eps_r and s <= eps_s)
    return {"x": x, "x_prev": x_prev, "z0": z0, "lam": lam, "p": p, "r": r, "s": s,
            "K": k, "wall": wall, "history": np.asarray(history),
            "converged": converged, "qp_info": qp_info,
            "qp_infos": qp_infos}


def run_ml(prob, predict_fn, x0=None, lam0=None, theta=0.01, tau=5,
           K_budget=100, eps=1e-4, hist=True, z0_init=None, verbose=False):
    x = prob.x_lo + 0.5 * (prob.x_hi - prob.x_lo) if x0 is None else x0.copy()
    lam = np.zeros((prob.S, prob.Gc)) if lam0 is None else lam0.copy()
    z0 = np.repeat(x[None, :], prob.S, axis=0) if z0_init is None else z0_init.copy()
    r_best, n_stall = np.inf, 0
    component_rows, history = [], []
    t_start = time.perf_counter()
    r = s = np.inf
    stop_reason = "budget"
    for k in range(1, K_budget + 1):
        x_prev = x
        t_c0 = time.perf_counter()
        x = x_update(prob, z0, lam)
        z0, t_fwd, t_frl = predict_fn(prob, x, lam)
        lam = lam + prob.rho * (x[None, :] - z0)
        t_cons = time.perf_counter() - t_c0 - t_fwd - t_frl
        component_rows.append((t_fwd, t_frl, t_cons))
        r, s = residuals(prob, x, x_prev, z0)
        if hist:
            history.append((r, s))
        if verbose:
            print(f"    ml k={k:3d} r={r:.3e} s={s:.3e}", flush=True)
        if r < (1 - theta) * r_best:
            r_best, n_stall = r, 0
        else:
            n_stall += 1
        if n_stall >= tau:
            stop_reason = "stall"
            break
        if r <= eps:
            stop_reason = "primal_tolerance_only"
            break
    wall = time.perf_counter() - t_start
    components = np.asarray(component_rows)
    means = components.mean(axis=0)
    return {"x": x, "x_prev": x_prev, "z0": z0, "lam": lam, "r_plateau": r,
            "s_plateau": s, "K_ml": k, "wall": wall,
            "t_fwd": float(means[0]), "t_FRL": float(means[1]),
            "t_cons": float(means[2]),
            "component_times": components, "history": np.asarray(history),
            "stop_reason": stop_reason,
            "t_other": float(wall - components.sum())}


def run_hybrid(prob, predict_fn, eps=1e-4, eps_s=1e-4, warm=False, **kw):
    ml_keys = ("theta", "tau", "K_budget", "hist", "verbose")
    ml = run_ml(prob, predict_fn, **{k: v for k, v in kw.items() if k in ml_keys})
    tail = run_exact(prob, x0=ml["x"], lam0=ml["lam"], z0_init=ml["z0"],
                     eps_r=eps, eps_s=eps_s, max_iter=2000, warm=warm,
                     hist=True, verbose=kw.get("verbose", False))
    return {"ml": ml, "tail": tail, "r": tail["r"], "s": tail["s"],
            "wall": ml["wall"] + tail["wall"], "K_ml": ml["K_ml"]}
