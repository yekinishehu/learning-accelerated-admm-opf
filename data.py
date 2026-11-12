"""
data.py — IEEE 118-bus data loading and scenario generation for the
St-MP-DCOPF rebuild (ELSCAM-D-26-03479R2).

Scenario generation: the R1 manuscript describes seasonal ARIMA models fitted
to ENTSO-E historical data. The historical series are not available in this
rebuild; scenarios are simulated from a seasonal ARIMA-type process whose
parameters are calibrated to standard daily load / wind shapes (double-peak
load profile, anticorrelated diurnal wind). Per-scenario multiplicative noise
is uniform +/-10% as in the manuscript. All parameters are explicit below and
the generator is seeded for reproducibility.
"""

import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _parse_matrix(path: Path, start_marker: str) -> np.ndarray:
    text = path.read_text()
    start = text.index(start_marker) + len(start_marker)
    end = text.index("];", start)
    rows = []
    for line in text[start:end].splitlines():
        line = line.strip().rstrip(";").strip()
        if line:
            rows.append([float(v) for v in re.split(r"\s+", line)])
    return np.array(rows)


def load_case118():
    bus = _parse_matrix(HERE / "case118_bus.m", "mpc.bus = [")
    gen, gencost = None, None
    text = (HERE / "case118_gen.m").read_text()
    b = text.index("gen = [")
    e = text.index("];", b)
    gen = np.array([[float(v) for v in re.split(r"\s+", ln.strip().rstrip(";").strip())]
                    for ln in text[b + len("gen = ["):e].splitlines() if ln.strip()])
    b = text.index("gencost = [")
    e = text.index("];", b)
    gencost = np.array([[float(v) for v in re.split(r"\s+", ln.strip().rstrip(";").strip())]
                        for ln in text[b + len("gencost = ["):e].splitlines() if ln.strip()])
    assert bus.shape == (118, 13), bus.shape
    assert gen.shape == (54, 10), gen.shape
    assert gencost.shape == (54, 7), gencost.shape
    # known IEEE-118 sanity checks
    assert abs(bus[:, 2].sum() - 4242.0) < 1.0, bus[:, 2].sum()
    assert abs(bus[:, 3].sum() - 1438.0) < 1.0, bus[:, 3].sum()
    on = gen[:, 7] > 0
    assert on.sum() == 54
    return {
        "baseMVA": 100.0,
        "Pd": bus[:, 2],                       # MW, per bus
        "Qd": bus[:, 3],
        "gen_bus": gen[:, 0].astype(int) - 1,  # 0-based
        "Pmax": gen[:, 8],
        "Pmin": gen[:, 9],
        "c2": gencost[:, 4],
        "c1": gencost[:, 5],
        "c0": gencost[:, 6],
    }


# --- scenario generation -------------------------------------------------

# standard double-peak daily load shape (fraction of daily peak), hours 0..23
LOAD_SHAPE = np.array([
    0.67, 0.63, 0.60, 0.59, 0.60, 0.64, 0.74, 0.83,
    0.90, 0.95, 0.97, 0.96, 0.95, 0.95, 0.93, 0.94,
    0.99, 1.00, 0.99, 0.96, 0.91, 0.83, 0.75, 0.70])
# diurnal wind shape (fraction of capacity), anticorrelated with load
WIND_SHAPE = np.array([
    0.62, 0.65, 0.67, 0.68, 0.66, 0.62, 0.55, 0.47,
    0.40, 0.34, 0.30, 0.28, 0.27, 0.28, 0.31, 0.35,
    0.40, 0.45, 0.50, 0.54, 0.57, 0.60, 0.61, 0.62])

WIND_BUSES = np.array([58, 115, 89, 79, 61])   # 0-based: buses 59, 116, 90, 80, 62
WIND_CAP = 50.0                                # MW nominal per farm (manuscript)


def _ar2_innov(rng, n, phi1=0.75, phi2=-0.15, sigma=0.02):
    """AR(2) multiplicative noise path of length n, unit asymptotic variance-ish."""
    e = rng.normal(0.0, sigma, n)
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = phi1 * x[t - 1] + phi2 * x[t - 2] + e[t]
    return x


def make_scenarios(case, S: int, T: int = 24, seed: int = 0):
    """
    Returns dict with
      load : (S, T, 118) net-load-per-bus trajectory in MW (load minus wind
             potential at wind buses), always >= small positive floor,
      wind : (S, T) total available wind generation (MW) per scenario/time
             (upper bound for the curtailment slack),
      p_prev : (54,) initial dispatch (previous hour), MW.
    A seasonal-ARIMA-type simulation: deterministic daily shapes times a
    per-scenario AR(2) multiplicative noise path, plus a per-scenario uniform
    +/-10% scaling (manuscript convention).
    """
    rng = np.random.default_rng(seed)
    Pd = case["Pd"]
    load = np.empty((S, T, 118))
    wind_tot = np.empty((S, T))
    for s in range(S):
        scale = 1.0 + rng.uniform(-0.10, 0.10)
        noise = _ar2_innov(rng, T)
        prof = scale * LOAD_SHAPE * (1.0 + noise)                    # (T,)
        load_s = Pd[None, :] * prof[:, None]                         # (T,118)
        # wind subtracted at wind buses
        wnoise = _ar2_innov(rng, T, sigma=0.05)
        wind = WIND_CAP * WIND_SHAPE * (1.0 + wnoise)                # (T,)
        load_s[:, WIND_BUSES] -= wind[:, None]
        wind_tot[s] = len(WIND_BUSES) * wind
        load[s] = load_s
    # here-and-now convention (manuscript Eq. (st-mp-opf)): the t=0 state is
    # the observed present, common to all scenarios; uncertainty applies to
    # t >= 1.  (Scenario-varying t0 net load would make the hard consensus
    # p_{g,t0,s} = x_g incompatible with per-scenario balance.)
    load[:, 0, :] = load[:, 0, :].mean(axis=0, keepdims=True)
    wind_tot[:, 0] = wind_tot[:, 0].mean()
    np.clip(load, 0.05 * Pd[None, :], None, out=load)
    # initial dispatch: cheapest-first merit order at mean t=0 net load
    p_prev = economic_dispatch(case, load[:, 0, :].mean(axis=0).sum())
    return {"load": load, "wind": wind_tot, "p_prev": p_prev}


def economic_dispatch(case, total_mw):
    """Merit-order dispatch at the given total load (for initial conditions)."""
    Pmin, Pmax, c1 = case["Pmin"], case["Pmax"], case["c1"]
    order = np.argsort(c1)
    p = Pmin.copy()
    remaining = total_mw - Pmin.sum()
    for g in order:
        add = min(Pmax[g] - Pmin[g], max(remaining, 0.0))
        p[g] += add
        remaining -= add
    return p
