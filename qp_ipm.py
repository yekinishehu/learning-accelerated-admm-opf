"""
qp_ipm.py — Tailored primal-dual interior-point QP solver for the per-scenario
multi-period DC-OPF subproblem (Eq. (13b) of the manuscript), batched over
scenarios.

Problem per scenario (variables p in R^{G*T} time-major, plus elastic slacks
sh, cur in R^T):
    min  1/2 p' diag(w) p + q'p  +  sum_t [ c_sh*sh_t + c_cur*cur_t
                                           + (eps_q/2)(sh_t^2 + cur_t^2) ]
    s.t. sum_g p_{t,g} + sh_t - cur_t = d_t   (elastic balance, t = 0..T-1)
         lo_i <= p_i <= hi_i                  (box; t=0 ramp folded into lo/hi)
         -R_g <= p_{t,g} - p_{t-1,g} <= R_g   (ramp, t = 1..T-1)
         0 <= sh_t <= sh_max_{t,s}            (shedding cannot exceed net load)
         0 <= cur_t <= cur_max_{t,s}          (curtailment bounded by wind)

The elastic terms keep every scenario subproblem feasible for ANY consensus
point x and scenario-dependent initial-period net load (complete recourse);
without them the consensus formulation is infeasible whenever D_{s,0} differs
across scenarios. eps_q > 0 is a small strict-convexity regulariser.

Structure exploited: after eliminating inequality slacks, the IPM Newton
matrix is [H  B'; B  0] with H block-diagonal: per-generator tridiagonal T x T
blocks (diagonal + ramp chains), plus a 2T diagonal block for (sh, cur).
Schur-complementing the T balance multipliers gives an SPD T x T system
    Sigma = sum_g H_g^{-1} + diag(1/h_sh,t + 1/h_cur,t),
assembled by inverting G tridiagonal T x T systems (batched Thomas,
O(G T^2) per scenario). Everything is vectorised over the scenario batch.

Mehrotra predictor-corrector. The iterates stop at tol_mu/tol_res (looser
than machine precision on purpose: at extreme centrality the Newton system
becomes numerically singular, well past the accuracy the ADMM outer loop
needs); the best iterate seen is tracked and returned as a fallback.
"""

import numpy as np


class BatchedIPM:
    def __init__(self, G: int, T: int, w: np.ndarray, lo: np.ndarray,
                 hi: np.ndarray, ramp: np.ndarray,
                 c_sh: float = 1000.0, c_cur: float = 1.0, eps_q: float = 1e-6):
        """
        w    : (G*T,) Hessian diagonal for generation (time-major), > 0
        lo/hi: (G*T,) generation box bounds (t=0 ramp folded in), lo < hi
        ramp : (G,) ramp limits R_g > 0
        c_sh, c_cur : linear penalties on shedding / curtailment ($/MWh)
        eps_q: quadratic regularisation of the slack variables
        """
        self.G, self.T = G, T
        self.n = n = G * T
        self.w = np.asarray(w, float)
        assert self.w.shape == (n,) and (self.w > 0).all()
        self.lo = np.asarray(lo, float).copy()
        self.hi = np.asarray(hi, float).copy()
        assert (self.hi > self.lo).all()
        self.R = np.asarray(ramp, float)
        assert self.R.shape == (G,) and (self.R > 0).all()
        self.c_sh, self.c_cur, self.eps_q = c_sh, c_cur, eps_q
        self.pmid = 0.5 * (self.lo[:G] + self.hi[:G])   # strict t0 interior

    # ------------------------------------------------------------------ #
    # batched tridiagonal inverse (SPD Thomas), O(T^2) per matrix        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tri_inv(a, b):
        """a: (..., T) diagonal, b: (..., T-1) off-diagonal (symmetric).
        Returns (..., T, T) explicit inverse."""
        T = a.shape[-1]
        dp = np.empty_like(a)
        cp = np.empty_like(b)
        dp[..., 0] = a[..., 0]
        for i in range(T - 1):
            cp[..., i] = b[..., i] / dp[..., i]
            dp[..., i + 1] = a[..., i + 1] - b[..., i] * cp[..., i]
        Hinv = np.empty(a.shape + (T,))
        for j in range(T):
            y = np.zeros_like(a)
            y[..., j] = 1.0 / dp[..., j]
            for i in range(j + 1, T):
                y[..., i] = -b[..., i - 1] * y[..., i - 1] / dp[..., i]
            x = np.empty_like(a)
            x[..., T - 1] = y[..., T - 1]
            for i in range(T - 2, -1, -1):
                x[..., i] = y[..., i] - cp[..., i] * x[..., i + 1]
            Hinv[..., :, j] = x
        return Hinv

    # ------------------------------------------------------------------ #
    # main entry                                                          #
    # ------------------------------------------------------------------ #
    def solve(self, q, d, sh_max, cur_max, tol_mu=1e-9, tol_res=1e-7,
              max_iter=60, chunk=200, verbose=False, p0=None,
              state0=None, ret_state=False):
        """q: (n,S) linear term; d: (T,S) balance RHS;
        sh_max, cur_max: (T,S) scenario-dependent slack bounds.
        p0: optional (n,S) primal warm start (previous solution,
        strictly-interiorised).  state0: optional full primal-dual IPM
        state dict from a previous solve (all arrays with last axis S);
        takes precedence over p0.  With ret_state=True the final
        primal-dual state is returned as a third value for reuse as
        state0 in the next call.  The converged solution is independent
        of the starting point up to the solver tolerances.
        Returns (p (n,S), info) or (p, info, state)."""
        n, S = self.n, q.shape[1]
        out = np.empty((n, S))
        worst = {"iters": 0, "res": 0.0, "mu": 0.0}
        states = []
        for s0 in range(0, S, chunk):
            sl = slice(s0, min(S, s0 + chunk))
            st0 = None
            if state0 is not None:
                st0 = {k: v[:, sl] for k, v in state0.items()}
            res = self._solve_chunk(
                q[:, sl], d[:, sl], sh_max[:, sl], cur_max[:, sl],
                tol_mu, tol_res, max_iter, verbose,
                None if p0 is None else p0[:, sl],
                state0=st0, ret_state=ret_state)
            if ret_state:
                out[:, sl], info, st = res
                states.append(st)
            else:
                out[:, sl], info = res
            worst["iters"] = max(worst["iters"], info["iters"])
            worst["res"] = max(worst["res"], info["res"])
            worst["mu"] = max(worst["mu"], info["mu"])
        if ret_state:
            if any(st is None for st in states):
                state = None            # non-finite final iterate somewhere
            else:
                state = {k: np.concatenate([st[k] for st in states], axis=-1)
                         for k in states[0]}
            return out, worst, state
        return out, worst

    def _solve_chunk(self, q, d, sh_max, cur_max, tol_mu, tol_res,
                     max_iter, verbose, p0=None, state0=None,
                     ret_state=False):
        n, G, T = self.n, self.G, self.T
        S = q.shape[1]
        w3 = self.w.reshape(T, G, 1)                   # (T,G,1)
        lo, hi = self.lo, self.hi
        Rp = np.tile(self.R, T - 1)                    # (n-G,)
        eq = self.eps_q

        # ---- starting point: full-state warm restart (previous IPM
        #      primal-dual state, positivity-floored), else cold (or
        #      primal-only warm via p0).
        if state0 is not None:
            p     = state0["p"].copy()
            nu    = state0["nu"].copy()
            sh    = state0["sh"].copy()
            cur   = state0["cur"].copy()
            s_bu  = np.maximum(state0["s_bu"], 1e-4)
            s_bl  = np.maximum(state0["s_bl"], 1e-4)
            s_ru  = np.maximum(state0["s_ru"], 1e-4)
            s_rl  = np.maximum(state0["s_rl"], 1e-4)
            s_su  = np.maximum(state0["s_su"], 1e-4)
            s_sl_ = np.maximum(state0["s_sl"], 1e-4)
            s_cu  = np.maximum(state0["s_cu"], 1e-4)
            s_cl  = np.maximum(state0["s_cl"], 1e-4)
            mu_bu = np.maximum(state0["mu_bu"], 1e-4)
            mu_bl = np.maximum(state0["mu_bl"], 1e-4)
            mu_ru = np.maximum(state0["mu_ru"], 1e-4)
            mu_rl = np.maximum(state0["mu_rl"], 1e-4)
            mu_su = np.maximum(state0["mu_su"], 1e-4)
            mu_sl = np.maximum(state0["mu_sl"], 1e-4)
            mu_cu = np.maximum(state0["mu_cu"], 1e-4)
            mu_cl = np.maximum(state0["mu_cl"], 1e-4)
            M = float(2 * n + 2 * (n - G) + 4 * T)
        else:
            if p0 is None:
                p = np.tile(self.pmid, T)[:, None] * np.ones((1, S))
            else:
                mrg = 1e-6 * np.maximum(1.0, hi - lo)
                p = np.minimum(np.maximum(p0, (lo + mrg)[:, None]),
                               (hi - mrg)[:, None])
            nu = np.zeros((T, S))
            psum0 = p.reshape(T, G, S).sum(axis=1)     # (T,S)
            sh = np.clip(d - psum0, 1e-3, np.maximum(sh_max - 1e-3, 1e-3))
            cur = np.clip(psum0 - d, 1e-3, np.maximum(cur_max - 1e-3, 1e-3))
            if p0 is None:
                sh = np.minimum(sh, sh_max * 0.5 + 1e-3)
                cur = np.minimum(cur, cur_max * 0.5 + 1e-3)

            # inequality slacks, all strictly positive at the start
            s_bu = hi[:, None] - p                      # (n,S)
            s_bl = p - lo[:, None]
            if p0 is None:
                s_ru = Rp[:, None] * np.ones((1, S))    # constant-in-time p
                s_rl = Rp[:, None] * np.ones((1, S))
            else:
                dpt0 = p[G:] - p[:-G]
                s_ru = np.maximum(Rp[:, None] - dpt0, 1e-6)
                s_rl = np.maximum(Rp[:, None] + dpt0, 1e-6)
            s_su = sh_max - sh; s_sl_ = sh              # sh in (0, sh_max)
            s_cu = cur_max - cur; s_cl = cur
            mu_bu = np.ones_like(p); mu_bl = np.ones_like(p)
            mu_ru = np.ones((n - G, S)); mu_rl = np.ones((n - G, S))
            mu_su = np.ones((T, S)); mu_sl = np.ones((T, S))
            mu_cu = np.ones((T, S)); mu_cl = np.ones((T, S))
            M = float(2 * n + 2 * (n - G) + 4 * T)

        it, res_inf, mu_bar = 0, np.inf, np.inf
        best_p, best_res, best_mu = p.copy(), np.inf, np.inf
        best_dual = None
        for it in range(1, max_iter + 1):
            def residuals():
                rq = (self.w[:, None] * p + q
                      + mu_bu - mu_bl)
                dm = mu_ru - mu_rl
                rq[G:] += dm
                rq[:-G] -= dm
                rq3 = rq.reshape(T, G, S)
                rq3 += nu[:, None, :]
                rq_sh = eq * sh + self.c_sh + mu_su - mu_sl + nu
                rq_cur = eq * cur + self.c_cur + mu_cu - mu_cl - nu
                rb = p.reshape(T, G, S).sum(axis=1) + sh - cur - d
                vals = [np.abs(rq).max(), np.abs(rb).max(),
                        np.abs(rq_sh).max(), np.abs(rq_cur).max()]
                if not all(np.isfinite(v) for v in vals):
                    return np.inf
                return max(vals)

            comp = (np.sum(s_bu * mu_bu) + np.sum(s_bl * mu_bl)
                    + np.sum(s_ru * mu_ru) + np.sum(s_rl * mu_rl)
                    + np.sum(s_su * mu_su) + np.sum(s_sl_ * mu_sl)
                    + np.sum(s_cu * mu_cu) + np.sum(s_cl * mu_cl))
            mu_bar = float(comp / M / S)
            res_inf = residuals()
            if not np.isfinite(mu_bar):
                mu_bar = np.inf
            if res_inf < best_res and np.isfinite(res_inf) and np.isfinite(mu_bar):
                best_res, best_p, best_mu = res_inf, p.copy(), mu_bar
                # Keep the primal-dual pair itself. The old implementation
                # returned best_p with the final (possibly non-finite) dual
                # arrays, which is not a valid warm state or dual certificate.
                best_dual = {k: v.copy() for k, v in {
                    "nu": nu, "mu_bu": mu_bu, "mu_bl": mu_bl,
                    "mu_ru": mu_ru, "mu_rl": mu_rl,
                    "mu_su": mu_su, "mu_sl": mu_sl,
                    "mu_cu": mu_cu, "mu_cl": mu_cl}.items()}
            if verbose:
                print(f"    ipm it={it:3d}  mu_bar={mu_bar:.2e}  res={res_inf:.2e}")
            if mu_bar < tol_mu and res_inf < tol_res:
                break
            if not np.isfinite(res_inf) or not np.isfinite(mu_bar):
                it = it - 1
                break
            if mu_bar < 1e-11 and best_res < tol_res:
                # Deep in the numerically singular regime with an already
                # converged best iterate: further Newton steps only
                # thrash (residual grows as mu -> 0).  Stop and return
                # the best iterate.  Note: acceptance is tied to tol_res
                # itself -- a looser threshold here biases warm-started
                # solves toward the stale state and breaks the exactness
                # of the ADMM z-oracle (observed as a limit cycle).
                break

            # ---- weights: H diagonal/tridiagonal pieces -----------------
            w_bu = mu_bu / s_bu; w_bl = mu_bl / s_bl
            w_ru = mu_ru / s_ru; w_rl = mu_rl / s_rl
            w_su = mu_su / s_su; w_sl = mu_sl / s_sl_
            w_cu = mu_cu / s_cu; w_cl = mu_cl / s_cl
            wbox3 = (w_bu + w_bl).reshape(T, G, S)
            wr3 = (w_ru + w_rl).reshape(T - 1, G, S)
            a3 = w3 + wbox3
            a3[1:] += wr3
            a3[:-1] += wr3
            b3 = -wr3
            Hinv = self._tri_inv(np.transpose(a3, (1, 2, 0)),
                                 np.transpose(b3, (1, 2, 0)))   # (G,S,T,T)
            h_sh = eq + w_su + w_sl                         # (T,S) diagonal
            h_cu = eq + w_cu + w_cl
            Sigma = Hinv.sum(axis=0)                        # (S,T,T)
            sidx = np.arange(T)
            Sigma[:, sidx, sidx] += (1.0 / h_sh + 1.0 / h_cu).T

            def step_from_tau(tau_arr, corr_term=None):
                """Newton step with complementarity target tau (+corrector)."""
                def phi(mu_, s_, c_=None):
                    v = tau_arr / s_ - mu_
                    if c_ is not None:
                        v = v - c_ / s_
                    return v
                if corr_term is None:
                    cbu = cbl = cru = crl = csu = csl = ccu = ccl = None
                else:
                    (cbu, cbl, cru, crl, csu, csl, ccu, ccl) = corr_term
                phi_bu = phi(mu_bu, s_bu, cbu)
                phi_bl = phi(mu_bl, s_bl, cbl)
                phi_ru = phi(mu_ru, s_ru, cru)
                phi_rl = phi(mu_rl, s_rl, crl)
                phi_su = phi(mu_su, s_su, csu)
                phi_sl = phi(mu_sl, s_sl_, csl)
                phi_cu = phi(mu_cu, s_cu, ccu)
                phi_cl = phi(mu_cl, s_cl, ccl)
                # stationarity residuals
                rq = (self.w[:, None] * p + q + mu_bu - mu_bl)
                dm = mu_ru - mu_rl
                rq[G:] += dm
                rq[:-G] -= dm
                rq3 = rq.reshape(T, G, S) + nu[:, None, :]
                rq_sh = eq * sh + self.c_sh + mu_su - mu_sl + nu
                rq_cu = eq * cur + self.c_cur + mu_cu - mu_cl - nu
                # g = -r_q - C' phi
                Cphi = phi_bu - phi_bl
                dmr = phi_ru - phi_rl
                Cphi[G:] += dmr
                Cphi[:-G] -= dmr
                gq3 = -(rq3 + Cphi.reshape(T, G, S))        # (T,G,S)
                gq_sh = -(rq_sh + phi_su - phi_sl)          # (T,S)
                gq_cu = -(rq_cu + phi_cu - phi_cl)
                h = d - (p.reshape(T, G, S).sum(axis=1) + sh - cur)  # -r_b
                # Schur RHS: B H^{-1} gq
                v = np.einsum("gstu,gsu->gst", Hinv,
                              np.transpose(gq3, (1, 2, 0)))   # (G,S,T)
                BHg = v.sum(axis=0).T + gq_sh / h_sh - gq_cu / h_cu  # (T,S)
                dnu = np.linalg.solve(Sigma, (BHg - h).T[:, :, None])[..., 0].T
                # recover generation block
                gm = np.transpose(gq3, (1, 2, 0)) - dnu.T[None, :, :]
                dp = np.transpose(np.einsum("gstu,gsu->gst", Hinv, gm),
                                  (2, 0, 1)).reshape(n, S)
                dsh = (gq_sh - dnu) / h_sh
                dcur = (gq_cu + dnu) / h_cu
                # inequality-slack directions
                ds_bu = -dp; ds_bl = dp
                dpair = dp[G:] - dp[:-G]
                ds_ru = -dpair; ds_rl = dpair
                ds_su = -dsh; ds_sl = dsh
                ds_cu = -dcur; ds_cl = dcur
                dmu_bu = phi_bu + w_bu * dp
                dmu_bl = phi_bl - w_bl * dp
                dmu_ru = phi_ru + w_ru * dpair
                dmu_rl = phi_rl - w_rl * dpair
                dmu_su = phi_su + w_su * dsh
                dmu_sl = phi_sl - w_sl * dsh
                dmu_cu = phi_cu + w_cu * dcur
                dmu_cl = phi_cl - w_cl * dcur
                return (dp, dnu, dsh, dcur,
                        ds_bu, ds_bl, ds_ru, ds_rl, ds_su, ds_sl, ds_cu, ds_cl,
                        dmu_bu, dmu_bl, dmu_ru, dmu_rl,
                        dmu_su, dmu_sl, dmu_cu, dmu_cl)

            zero_tau = np.zeros((1, S))
            (dp_a, dnu_a, dsh_a, dcur_a,
             ds_bu_a, ds_bl_a, ds_ru_a, ds_rl_a,
             ds_su_a, ds_sl_a, ds_cu_a, ds_cl_a,
             dmu_bu_a, dmu_bl_a, dmu_ru_a, dmu_rl_a,
             dmu_su_a, dmu_sl_a, dmu_cu_a, dmu_cl_a) = step_from_tau(zero_tau)

            def frac(s_, ds_):
                neg = ds_ < 0
                if not neg.any():
                    return 1.0
                return min(1.0, float((s_[neg] / -ds_[neg]).min()))

            a_aff = min(frac(s_bu, ds_bu_a), frac(s_bl, ds_bl_a),
                        frac(s_ru, ds_ru_a), frac(s_rl, ds_rl_a),
                        frac(s_su, ds_su_a), frac(s_sl_, ds_sl_a),
                        frac(s_cu, ds_cu_a), frac(s_cl, ds_cl_a),
                        frac(mu_bu, dmu_bu_a), frac(mu_bl, dmu_bl_a),
                        frac(mu_ru, dmu_ru_a), frac(mu_rl, dmu_rl_a),
                        frac(mu_su, dmu_su_a), frac(mu_sl, dmu_sl_a),
                        frac(mu_cu, dmu_cu_a), frac(mu_cl, dmu_cl_a))
            mu_aff = float((
                np.sum((s_bu + a_aff * ds_bu_a) * (mu_bu + a_aff * dmu_bu_a))
                + np.sum((s_bl + a_aff * ds_bl_a) * (mu_bl + a_aff * dmu_bl_a))
                + np.sum((s_ru + a_aff * ds_ru_a) * (mu_ru + a_aff * dmu_ru_a))
                + np.sum((s_rl + a_aff * ds_rl_a) * (mu_rl + a_aff * dmu_rl_a))
                + np.sum((s_su + a_aff * ds_su_a) * (mu_su + a_aff * dmu_su_a))
                + np.sum((s_sl_ + a_aff * ds_sl_a) * (mu_sl + a_aff * dmu_sl_a))
                + np.sum((s_cu + a_aff * ds_cu_a) * (mu_cu + a_aff * dmu_cu_a))
                + np.sum((s_cl + a_aff * ds_cl_a) * (mu_cl + a_aff * dmu_cl_a))
            ) / M / S)
            ratio = mu_aff / max(mu_bar, 1e-300)
            if not np.isfinite(ratio) or ratio >= 1.0:
                sigma_c = 1.0
            elif ratio <= 0.0:
                sigma_c = 0.0
            else:
                sigma_c = ratio * ratio * ratio
            tau = sigma_c * mu_bar * np.ones((1, S))

            corr = (ds_bu_a * dmu_bu_a, ds_bl_a * dmu_bl_a,
                    ds_ru_a * dmu_ru_a, ds_rl_a * dmu_rl_a,
                    ds_su_a * dmu_su_a, ds_sl_a * dmu_sl_a,
                    ds_cu_a * dmu_cu_a, ds_cl_a * dmu_cl_a)
            (dp, dnu, dsh, dcur,
             ds_bu, ds_bl, ds_ru, ds_rl, ds_su, ds_sl, ds_cu, ds_cl,
             dmu_bu, dmu_bl, dmu_ru, dmu_rl,
             dmu_su, dmu_sl, dmu_cu, dmu_cl) = step_from_tau(tau, corr)

            eta = 0.995
            aps = eta * min(frac(s_bu, ds_bu), frac(s_bl, ds_bl),
                            frac(s_ru, ds_ru), frac(s_rl, ds_rl),
                            frac(s_su, ds_su), frac(s_sl_, ds_sl),
                            frac(s_cu, ds_cu), frac(s_cl, ds_cl))
            apd = eta * min(frac(mu_bu, dmu_bu), frac(mu_bl, dmu_bl),
                            frac(mu_ru, dmu_ru), frac(mu_rl, dmu_rl),
                            frac(mu_su, dmu_su), frac(mu_sl, dmu_sl),
                            frac(mu_cu, dmu_cu), frac(mu_cl, dmu_cl))
            p = p + aps * dp
            nu = nu + apd * dnu
            sh = sh + aps * dsh
            cur = cur + aps * dcur
            mu_bu = mu_bu + apd * dmu_bu
            mu_bl = mu_bl + apd * dmu_bl
            mu_ru = mu_ru + apd * dmu_ru
            mu_rl = mu_rl + apd * dmu_rl
            mu_su = mu_su + apd * dmu_su
            mu_sl = mu_sl + apd * dmu_sl
            mu_cu = mu_cu + apd * dmu_cu
            mu_cl = mu_cl + apd * dmu_cl
            # recompute inequality slacks exactly from the new primal point
            s_bu = hi[:, None] - p
            s_bl = p - lo[:, None]
            dpt = p[G:] - p[:-G]
            s_ru = Rp[:, None] - dpt
            s_rl = Rp[:, None] + dpt
            s_su = sh_max - sh
            s_sl_ = sh
            s_cu = cur_max - cur
            s_cl = cur
        info = {"iters": it, "res": float(best_res), "mu": float(best_mu)}
        if not ret_state:
            return best_p, info
        state = None
        if np.all(np.isfinite(best_p)) and best_dual is not None:
            # Sanitise the dual/slack part of the state: final-iterate
            # arrays may contain non-finite entries on the fallback path.
            def clean_mu(m):
                return np.clip(np.nan_to_num(m, nan=1.0, posinf=1.0,
                                             neginf=1.0), 1e-10, 1e6)
            pb = best_p
            psb = pb.reshape(T, G, S).sum(axis=1)
            shb = np.clip(d - psb, 1e-10, np.maximum(sh_max - 1e-10, 1e-10))
            curb = np.clip(psb - d, 1e-10, np.maximum(cur_max - 1e-10, 1e-10))
            dpb = pb[G:] - pb[:-G]
            state = dict(
                p=pb.copy(), nu=best_dual["nu"], sh=shb, cur=curb,
                s_bu=np.maximum(hi[:, None] - pb, 1e-10),
                s_bl=np.maximum(pb - lo[:, None], 1e-10),
                s_ru=np.maximum(Rp[:, None] - dpb, 1e-10),
                s_rl=np.maximum(Rp[:, None] + dpb, 1e-10),
                s_su=np.maximum(sh_max - shb, 1e-10), s_sl=shb,
                s_cu=np.maximum(cur_max - curb, 1e-10), s_cl=curb,
                **{k: clean_mu(best_dual[k]) for k in best_dual if k != "nu"})
        return best_p, info, state

    # ------------------------------------------------------------------ #
    # independent KKT verification (used by tests)                        #
    # ------------------------------------------------------------------ #
    def kkt_check(self, p, q, d):
        n, G, T = self.n, self.G, self.T
        S = p.shape[1]
        rb = np.abs(p.reshape(T, G, S).sum(axis=1) - d).max()
        viol_box = max((self.lo[:, None] - p).max(), (p - self.hi[:, None]).max())
        dpt = p[G:] - p[:-G]
        Rfull = np.tile(self.R, T - 1)[:, None]
        viol_ramp = (np.abs(dpt) - Rfull).max()
        return {"res_balance_noSlack": float(rb),
                "viol_box": float(max(viol_box, 0.0)),
                "viol_ramp": float(max(viol_ramp, 0.0))}
