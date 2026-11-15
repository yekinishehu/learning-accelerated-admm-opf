"""Generate manuscript tables, figures, and numeric macros from verified runs."""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "manuscript" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
PROTOCOL = json.loads((ROOT / "strengthening_protocol.json").read_text())
verification = json.loads((ROOT / "results" / "verification.json").read_text())
assert verification["verified_runs"] == verification["expected_runs"] == PROTOCOL["expected_run_count"]
assert json.loads((ROOT / "results" / "training_verification.json").read_text())["status"] == "pass"
INSTANCES = [(S, seed) for S in PROTOCOL["scenario_counts"] for seed in PROTOCOL["seeds"]]
MAIN = [(S, 99) for S in PROTOCOL["scenario_counts"]]
ALGOS = ("exact", "ml", "hybrid")


def load(S, seed, algo):
    return json.loads((RUNS / f"{algo}_S{S}_seed{seed}.json").read_text())


def sci(x, digits=3):
    if x == 0:
        return "0"
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"


def pct(x):
    return sci(100 * x, 2)


def main():
    allruns = {(S, seed, a): load(S, seed, a) for S, seed in INSTANCES for a in ALGOS}
    ratios = {(S, seed): allruns[S, seed, "hybrid"]["algorithm_wall_measured"] /
              allruns[S, seed, "exact"]["algorithm_wall_measured"] for S, seed in INSTANCES}
    speedups = {key: 100 * (1 - value) for key, value in ratios.items()}
    certified = [r for (S, seed, a), r in allruns.items() if a in ("exact", "hybrid")]
    mlruns = [r for (S, seed, a), r in allruns.items() if a == "ml"]
    maxabs = max(r["certificate"]["absolute_gap"] for r in certified)
    maxrelpct = max(100 * r["certificate"]["relative_gap"] for r in certified)
    train_log = (ROOT / "results" / "train.log").read_text()
    val = float(re.search(r"val L2 err: mean ([0-9.]+)", train_log).group(1))
    valp90 = float(re.search(r"p90 ([0-9.]+)", train_log).group(1))
    dg_wall = float(re.search(r"WALL_SECONDS ([0-9.]+)",
                              (ROOT / "results" / "datagen.log").read_text()).group(1))
    tr_wall = float(re.search(r"WALL_SECONDS ([0-9.]+)", train_log).group(1))
    worst_inner = max(r["inner_qp_worst"]["res"] for r in certified)
    worst_mu = max(r["inner_qp_worst"]["mu"] for r in certified)
    worst_iters = max(r["inner_qp_worst"]["iters"] for r in certified)
    max_qspread = max(r["certificate"]["q_primal_dual_spread"] for r in allruns.values())
    max_vspread = max(r["certificate"]["V_primal_dual_spread"] for r in allruns.values())
    by_s = {S: [speedups[S, seed] for seed in PROTOCOL["seeds"]]
            for S in PROTOCOL["scenario_counts"]}
    labels = {20: "Twenty", 50: "Fifty", 100: "Hundred"}
    macros = {
        "RunCount": str(len(allruns)),
        "InstanceCount": str(len(INSTANCES)),
        "CertifiedRunCount": str(len(certified)),
        "HybridOverheadTwenty": f"{-speedups[20, 99]:.2f}",
        "HybridSpeedupFifty": f"{speedups[50, 99]:.2f}",
        "HybridSpeedupHundred": f"{speedups[100, 99]:.2f}",
        "HybridSpeedupMin": f"{min(speedups.values()):.2f}",
        "HybridSpeedupMax": f"{max(speedups.values()):.2f}",
        "MaxCertifiedAbsGap": f"{maxabs:.6f}",
        "MaxCertifiedRelGapPct": f"{maxrelpct:.6f}",
        "MLResidualMin": f"{min(r['r_terminal'] for r in mlruns):.3f}",
        "MLResidualMax": f"{max(r['r_terminal'] for r in mlruns):.3f}",
        "ValidationMeanMW": f"{val:.3f}",
        "ValidationP90MW": f"{valp90:.3f}",
        "OfflineWall": f"{dg_wall + tr_wall:.2f}",
        "DatagenWall": f"{dg_wall:.2f}",
        "TrainingWall": f"{tr_wall:.2f}",
        "WorstInnerResidual": sci(worst_inner, 3).replace("$", ""),
        "WorstInnerMu": sci(worst_mu, 3).replace("$", ""),
        "WorstInnerIterations": str(worst_iters),
        "MaxDualPrimalSpread": f"{max_qspread:.6f}",
        "MaxPinnedPrimalDualSpread": f"{max_vspread:.6f}",
    }
    for S, vals in by_s.items():
        label = labels[S]
        macros[f"HybridSpeedupMin{label}"] = f"{min(vals):.2f}"
        macros[f"HybridSpeedupMax{label}"] = f"{max(vals):.2f}"
        macros[f"HybridSpeedupMedian{label}"] = f"{np.median(vals):.2f}"
        macros[f"HybridFasterCount{label}"] = str(sum(v > 0 for v in vals))
        macros[f"HybridSlowerCount{label}"] = str(sum(v < 0 for v in vals))
    (OUT / "numbers.tex").write_text("\n".join(
        f"\\newcommand{{\\Num{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    lines = [r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
             r"$S$ & Method & $K_{\rm ML}$ & $K_{\rm ex}$ & $r^K$ & $s^K$ & Time (s) & Abs.\ gap & Rel.\ gap (\%) \\",
             r"\midrule"]
    for S, seed in MAIN:
        for algo, label in (("exact", "ADMM"), ("ml", "ADMM-ML"), ("hybrid", "ADMM-ML-H")):
            r = allruns[S, seed, algo]
            c = r["certificate"]
            lines.append(f"{S} & {label} & {r['K_ml']} & {r['K_tail']} & {sci(r['r_terminal'],2)} & {sci(r['s_terminal'],2)} & {r['algorithm_wall_measured']:.3f} & {c['absolute_gap']:.6f} & {pct(c['relative_gap'])} \\\\")
        if (S, seed) != MAIN[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "main_table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$S$ & Seed & Exact (s) & Hybrid (s) & Hybrid/exact & Speedup (\%) & ML $r^K$ \\",
             r"\midrule"]
    for S, seed in INSTANCES:
        ex, hy, ml = (allruns[S, seed, a] for a in ("exact", "hybrid", "ml"))
        lines.append(f"{S} & {seed} & {ex['algorithm_wall_measured']:.3f} & {hy['algorithm_wall_measured']:.3f} & {ratios[S,seed]:.4f} & {speedups[S,seed]:.2f} & {ml['r_terminal']:.3f} \\\\")
        if seed != PROTOCOL["seeds"][-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "robustness_table.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$S$ & Forward (ms) & FRL (ms) & Consensus (ms) & Other total (s) & ML total (s) & Certificate (s) \\",
             r"\midrule"]
    for S, seed in MAIN:
        r = allruns[S, seed, "ml"]
        ms = [1000 * x for x in r["ml_component_mean_seconds"]]
        lines.append(f"{S} & {ms[0]:.3f} & {ms[1]:.3f} & {ms[2]:.3f} & {r['ml_other_seconds']:.6f} & {r['algorithm_wall_measured']:.3f} & {r['certificate']['wall']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "cost_table.tex").write_text("\n".join(lines) + "\n")

    # Long table: every run carrying a certified label, with no selective rows.
    lines = [r"\begin{longtable}{rrrrrrrr}",
             r"\caption{Terminal evidence for every exact and hybrid run. Absolute gap is $V(x^K)-p_{\rm low}$; relative gaps use $p_{\rm low}>0$. The final column is the worst returned inner-QP stationarity/balance residual during that algorithm run.}\label{tab:termination}\\",
             r"\toprule",
             r"$S$ & Seed & Method & $r^K$ & $s^K$ & Abs.\ gap & Rel.\ gap (\%) & Worst inner res.\ \\",
             r"\midrule\endfirsthead",
             r"\multicolumn{8}{l}{Table \ref*{tab:termination}, continued}\\",
             r"\toprule",
             r"$S$ & Seed & Method & $r^K$ & $s^K$ & Abs.\ gap & Rel.\ gap (\%) & Worst inner res.\ \\",
             r"\midrule\endhead",
             r"\midrule\multicolumn{8}{r}{Continued on the next page}\\\endfoot",
             r"\bottomrule\endlastfoot"]
    for S, seed in INSTANCES:
        for algo, label in (("exact", "ADMM"), ("hybrid", "ADMM-ML-H")):
            r = allruns[S, seed, algo]
            c = r["certificate"]
            lines.append(f"{S} & {seed} & {label} & {sci(r['r_terminal'],2)} & {sci(r['s_terminal'],2)} & {c['absolute_gap']:.6f} & {pct(c['relative_gap'])} & {sci(r['inner_qp_worst']['res'],2)} \\\\")
        if (S, seed) != INSTANCES[-1]:
            lines.append(r"\midrule")
    lines += [r"\end{longtable}"]
    (OUT / "termination_table.tex").write_text("\n".join(lines) + "\n")

    ex = np.load(RUNS / "exact_S50_seed99.npz")["history"]
    ml = np.load(RUNS / "ml_S50_seed99.npz")["history"]
    hyb_np = np.load(RUNS / "hybrid_S50_seed99.npz")
    hy = np.vstack([hyb_np["ml_history"], hyb_np["history"]])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for hist, label, style in ((ex, "ADMM", "-"), (ml, "ADMM-ML", "--"), (hy, "ADMM-ML-H", "-.")):
        k = np.arange(1, len(hist) + 1)
        ax.semilogy(k, np.maximum(hist[:, 0], 1e-16), style, label=label + " primal")
        ax.semilogy(k, np.maximum(hist[:, 1], 1e-16), style, alpha=.45, label=label + " dual")
    ax.axhline(1e-4, color="black", linewidth=.8, label="tolerance")
    ax.axvline(len(ml), color="gray", linewidth=.8, linestyle=":")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual")
    ax.grid(True, which="both", alpha=.2)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "residuals_S50.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.9, 4.2))
    x = np.arange(len(MAIN))
    width = .25
    for j, (algo, label) in enumerate((("exact", "ADMM"), ("ml", "ADMM-ML"), ("hybrid", "ADMM-ML-H"))):
        vals = [allruns[S, 99, algo]["algorithm_wall_measured"] for S, _ in MAIN]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.3g}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"S={S}" for S, _ in MAIN])
    ax.set_ylabel("Algorithm wall time (s, log scale)")
    ax.grid(True, axis="y", which="both", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "timings.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    for i, S in enumerate(PROTOCOL["scenario_counts"]):
        vals = np.asarray(by_s[S])
        offsets = np.linspace(-.12, .12, len(vals))
        ax.scatter(np.full(len(vals), i) + offsets, vals, s=28, alpha=.85,
                   color="tab:blue", zorder=3)
        ax.scatter([i], [np.median(vals)], marker="D", s=55, color="black",
                   label="Median" if i == 0 else None, zorder=4)
    ax.axhline(0, color="black", linewidth=.9)
    ax.set_xticks(range(len(by_s)), [f"S={S}" for S in by_s])
    ax.set_ylabel("Hybrid speedup over exact ADMM (%)")
    ax.set_xlabel("Scenario count (five paired seeds per point set)")
    ax.grid(True, axis="y", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "speedups.pdf")
    plt.close(fig)

    numeric = {
        "protocol": PROTOCOL["protocol_version"],
        "ratios_hybrid_over_exact": {f"S{S}_seed{seed}": v for (S, seed), v in ratios.items()},
        "speedups_percent": {f"S{S}_seed{seed}": v for (S, seed), v in speedups.items()},
        "speedups_by_scenario_count": by_s,
        "macros": macros,
        "runs": {f"{a}_S{S}_seed{seed}": r for (S, seed, a), r in allruns.items()},
    }
    (ROOT / "results" / "analysis.json").write_text(json.dumps(numeric, indent=2))
    print(json.dumps(macros, indent=2))


if __name__ == "__main__":
    main()
