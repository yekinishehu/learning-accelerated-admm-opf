"""Independent Lagrangian and pinned-recourse certificate evaluation.

Every scenario dual value is computed from nonnegative bound multipliers and
an unrestricted balance multiplier. It is a lower bound for its QP even when
the IPM stationarity residual is nonzero. Floating-point values are reported
with their primal-dual and feasibility diagnostics; no exact-real-arithmetic
claim is made.
"""
import time
import numpy as np
from qp_ipm import BatchedIPM


def qp_values(solver, p, state, q, d, sh_max, cur_max):
    """Return primal objective and a rigorous-in-real-arithmetic QP dual value."""
    G, T = solver.G, solver.T
    Rp = np.tile(solver.R, T - 1)
    sh, cur, nu = state["sh"], state["cur"], state["nu"]
    primal = (0.5 * solver.w[:, None] * p * p + q * p).sum(axis=0)
    primal += (0.5 * solver.eps_q * sh * sh + solver.c_sh * sh).sum(axis=0)
    primal += (0.5 * solver.eps_q * cur * cur + solver.c_cur * cur).sum(axis=0)

    dm = state["mu_ru"] - state["mu_rl"]
    vp = q + state["mu_bu"] - state["mu_bl"]
    vp[G:] += dm
    vp[:-G] -= dm
    vp = vp.reshape(T, G, -1) + nu[:, None, :]
    vsh = solver.c_sh + state["mu_su"] - state["mu_sl"] + nu
    vcur = solver.c_cur + state["mu_cu"] - state["mu_cl"] - nu
    dual = -0.5 * ((vp * vp / solver.w.reshape(T, G, 1)).sum(axis=(0, 1))
                   + (vsh * vsh + vcur * vcur).sum(axis=0) / solver.eps_q)
    dual -= (nu * d).sum(axis=0)
    dual -= (state["mu_bu"] * solver.hi[:, None]
             - state["mu_bl"] * solver.lo[:, None]).sum(axis=0)
    # Upper ramp: -mu_ru R. Lower ramp (-R <= dp): -mu_rl R.
    dual -= ((state["mu_ru"] + state["mu_rl"]) * Rp[:, None]).sum(axis=0)
    dual -= (state["mu_su"] * sh_max + state["mu_cu"] * cur_max).sum(axis=0)
    return primal, dual


def feasibility(solver, p, state, d, sh_max, cur_max):
    G, T = solver.G, solver.T
    balance = p.reshape(T, G, -1).sum(axis=1) + state["sh"] - state["cur"] - d
    dp = p[G:] - p[:-G]
    ramp = np.tile(solver.R, T - 1)[:, None]
    return {
        "balance_inf": float(np.abs(balance).max()),
        "box_violation": float(max(0.0, (solver.lo[:, None] - p).max(),
                                   (p - solver.hi[:, None]).max())),
        "ramp_violation": float(max(0.0, (np.abs(dp) - ramp).max())),
        "slack_bound_violation": float(max(
            0.0, (-state["sh"]).min(), (state["sh"] - sh_max).max(),
            (-state["cur"]).min(), (state["cur"] - cur_max).max())),
    }


def evaluate_certificate(prob, x, lam):
    """Evaluate q(lambda) and V(x) independently of the ADMM run."""
    S, G, T = prob.S, prob.G, prob.T
    start = time.perf_counter()
    # Scenario part of q(lambda): original objective, no ADMM penalty.
    cert_qp = BatchedIPM(G, T, prob.base_w, prob.qp.lo, prob.qp.hi,
                         prob.ramp, c_sh=prob.qp.c_sh, c_cur=prob.qp.c_cur)
    q = np.repeat(prob.q_cost[:, None], S, axis=1)
    q[:G] -= lam.T
    p, info, state = cert_qp.solve(q, prob.D, prob.D, prob.W,
                                   tol_mu=1e-11, tol_res=1e-9, max_iter=100,
                                   ret_state=True)
    primal, dual = qp_values(cert_qp, p, state, q, prob.D, prob.D, prob.W)
    coeff = lam.sum(axis=0)
    x_arg = np.where(coeff > 0, prob.x_lo, prob.x_hi)
    first_stage = float(coeff @ x_arg)
    p_low = float(dual.sum() + first_stage)
    q_primal = float(primal.sum() + first_stage)
    q_feas = feasibility(cert_qp, p, state, prob.D, prob.D, prob.W)

    # V(x): eliminate t=0 exactly. Its elastic variables are determined by
    # balance; all remaining variables solve a (T-1)-period QP whose first
    # ramp window is relative to x.
    total0 = float(x.sum())
    sh0 = np.maximum(prob.D[0] - total0, 0.0)
    cur0 = np.maximum(total0 - prob.D[0], 0.0)
    if np.any(sh0 > prob.D[0] + 1e-8) or np.any(cur0 > prob.W[0] + 1e-8):
        raise ValueError("Terminal x is outside the elastic t=0 feasible range")
    # base_w and q_cost already contain pi_s. As in every other scenario
    # objective, the common t=0 generation contribution must be summed over
    # all S scenarios (sum_s pi_s = 1), not evaluated only once with pi_s.
    fixed0 = S * (0.5 * prob.base_w[:G] * x * x + prob.q_cost[:G] * x).sum()
    fixed0 += (0.5 * cert_qp.eps_q * sh0 * sh0 + prob.qp.c_sh * sh0).sum()
    fixed0 += (0.5 * cert_qp.eps_q * cur0 * cur0 + prob.qp.c_cur * cur0).sum()
    if T > 1:
        lo = prob.qp.lo[G:].copy()
        hi = prob.qp.hi[G:].copy()
        lo[:G] = np.maximum(lo[:G], x - prob.ramp)
        hi[:G] = np.minimum(hi[:G], x + prob.ramp)
        if np.any(hi <= lo):
            raise ValueError("Terminal x leaves an empty first ramp window")
        pin_qp = BatchedIPM(G, T - 1, prob.base_w[G:], lo, hi, prob.ramp,
                            c_sh=prob.qp.c_sh, c_cur=prob.qp.c_cur)
        qpin = np.repeat(prob.q_cost[G:, None], S, axis=1)
        pp, pinfo, pstate = pin_qp.solve(qpin, prob.D[1:], prob.D[1:],
                                         prob.W[1:], tol_mu=1e-11,
                                         tol_res=1e-9, max_iter=100,
                                         ret_state=True)
        pprimal, pdual = qp_values(pin_qp, pp, pstate, qpin, prob.D[1:],
                                   prob.D[1:], prob.W[1:])
        V = float(fixed0 + pprimal.sum())
        V_dual = float(fixed0 + pdual.sum())
        pin_feas = feasibility(pin_qp, pp, pstate, prob.D[1:],
                               prob.D[1:], prob.W[1:])
        pin_p = pp
    else:
        V = V_dual = float(fixed0)
        pinfo = {"iters": 0, "res": 0.0, "mu": 0.0}
        pin_feas = {"balance_inf": 0.0, "box_violation": 0.0,
                    "ramp_violation": 0.0, "slack_bound_violation": 0.0}
        pin_p = np.empty((0, S))
    gap = V - p_low
    result = {
        "p_low": p_low,
        "q_scenario_primal_plus_first_stage": q_primal,
        "q_primal_dual_spread": q_primal - p_low,
        "V_pinned": V,
        "V_pinned_dual_lower": V_dual,
        "V_primal_dual_spread": V - V_dual,
        "absolute_gap": float(gap),
        "relative_gap": float(gap / p_low) if p_low > 0 else None,
        "relative_gap_valid": bool(p_low > 0 and gap >= -1e-6),
        "first_stage_inf": first_stage,
        "dual_scenario_min_margin": float((primal - dual).min()),
        "dual_solver_info": info,
        "pinned_solver_info": pinfo,
        "dual_feasibility": q_feas,
        "pinned_feasibility": pin_feas,
        "t0_balance_inf": float(np.abs(total0 + sh0 - cur0 - prob.D[0]).max()),
        "wall": time.perf_counter() - start,
    }
    arrays = {"dual_p": p, "dual_sh": state["sh"], "dual_cur": state["cur"],
              "pinned_tail_p": pin_p, "x": x, "lam": lam,
              "dual_values": dual, "dual_primal_values": primal}
    return result, arrays
