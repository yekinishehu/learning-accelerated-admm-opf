# Final verification report — ELSCAM-D-26-03479R2

**Manuscript:** *Learning-Accelerated ADMM: Convergence Theory and Application to Stochastic Optimal Power Flow*  
**Final numerical protocol:** `strengthened-cold-v1`  
**Protocol SHA-256:** `cf6aee6e81de097e5e46d771689d3a1bba2bbe69e45cf3d38ecae26f6b40a0e5`

## 1. Final referee-comment disposition

| Comment | Verdict | Final evidence |
|---|---|---|
| R1.1 — Global DRS contraction and linear convergence | **Addressed** | Global linear convergence is restricted to the globally smooth strongly convex partial-value regime. The half-line counterexample remains, and the unsupported polyhedral-QP/activity-identification claims are absent. |
| R1.2 — FRL frequency and computational cost | **Addressed** | FRL is applied and timed on every learned iteration. Fresh raw component arrays passed exact mean/sum accounting for all 30 ML/hybrid runs. Forward, FRL, consensus, other overhead, ML wall time, and certificate time are reported separately. |
| R1.3 — Full-scenario convergence and finite-run certification | **Addressed** | The learned theory concerns initial blocks only. All 45 certificates and terminal residuals were independently recomputed; all 30 exact/hybrid runs meet both residual tolerances. Scenario-dual and pinned-recourse algebra and solver diagnostics are included. |
| R1.4 — Formulation imprecision | **Addressed** | Convex-set/objective hypotheses and the implemented elastic copper-plate DC dispatch model, penalties, ramps, scenario generator, and scope restrictions are explicit. |
| R2.1 — Error monitoring and weaker conditions | **Addressed** | Sampled audits and validation diagnostics are not called summability certificates. The exact tail supplies the unconditional guarantee. The final theory pass also corrected the realized-residual perturbation coefficient in Lemma 3.4 and Theorem 3.5. |
| R2.2 — Scenario-count dependence | **Addressed** | The scaling condition is correctly stated as constant \(\rho S\), hence \(\rho\propto1/S\); weighted/max residual conversion is treated separately. |
| R2.3 — Zero-padding, graphs, and bus ordering | **Addressed** | Fixed-order/shared-label limitations, lack of permutation invariance, graph alternatives, and spectral-embedding limitations are stated without untested transfer claims. |
| R2.4 — Notation | **Addressed** | Exact proximal minimizers, learned/restored initial blocks, optimal recourse vectors, residuals, DRS objects, and certificate quantities are distinguished. |
| R2.5 — Exact versus learned updates | **Addressed** | The fixed proxy remains an uncertified heuristic; exact-tail convergence is assigned only to the hybrid. All ML stalls and all unfavorable hybrid timings remain visible. Bitwise standalone/hybrid ML identity, full ML replay, and cold exact-tail replay passed for all 15 instances. |

**Count:** nine addressed, zero partial, zero unverifiable within the explicitly stated theoretical and numerical scope. This is an artifact-based compliance assessment, not a prediction of editorial acceptance.

## 2. Final 45-run evidence gate

- Complete grid: \(S\in\{20,50,100\}\), seeds 99–103, algorithms exact/ML/hybrid; **45/45 runs verified**.
- Cold scenario solves: every exact ADMM iteration and every exact tail; \(\rho=0.01\).
- Source, configuration, raw-NPZ, and summary hashes: **pass**.
- ZIP CRC, required-array schema, and finiteness: **pass for all 45 NPZ archives**.
- Terminal residual replay: **exact pass**.
- Certificate replay: **pass**, with zero maximum absolute difference in all compared certificate scalars.
- Standalone-ML versus hybrid-ML identity: **bitwise pass for all 15 instances**.
- Full ML replay from the executed checkpoint: **bitwise pass for all 15 instances**.
- Cold exact-tail replay: **bitwise pass for all 15 instances**.
- ML component mean/sum accounting: **exact pass for all 30 ML/hybrid runs**.
- Training replay: **bitwise pass for all 30 shards and every checkpoint tensor/normalization array**.
- Stable-copy I/O retries: **10**, all completed with unchanged numerical settings; all ten failure records are retained.

The older twelve-run `final-cold-v1` study is not included in the final-grid archive and supplied no final timing value.

## 3. Regenerated numerical outcomes

All tables, figures, numeric macros, and the 45-row grid report were regenerated only after the evidence gates passed.

- Fixed-proxy ADMM-ML: all 15 runs stall; terminal maximum-scenario consensus residuals span **2.282–8.181 MW**. These runs are not certified.
- Exact and hybrid runs: all **30** meet \(r^K\leq10^{-4}\) and \(s^K\leq10^{-4}\).
- Maximum exact/hybrid absolute gap: **8.089959**.
- Maximum exact/hybrid relative gap: **0.000346%**.
- Signed hybrid speedup at \(S=20\): range **−74.16% to −25.30%**, median **−41.42%**; **0/5 faster, 5/5 slower**.
- Signed hybrid speedup at \(S=50\): range **11.31% to 16.10%**, median **14.74%**; **5/5 faster**.
- Signed hybrid speedup at \(S=100\): range **12.12% to 19.87%**, median **18.06%**; **5/5 faster**.
- Worst returned inner-IPM residual: \(1.392\times10^{-6}\); worst complementarity measure: \(1.127\times10^{-7}\); largest returned inner iteration count: **48**.

The timing sample is descriptive, not confirmatory. No universal acceleration claim is made. Divide-by-zero/invalid-value warnings near singular Newton systems remain disclosed; the solver returns its best finite matched primal–dual pair, and the manuscript does not call every inner solve machine-exact.

## 4. Final theory correction

The last proof audit identified an omitted learned reflected-block perturbation in the residual-bound proof. The realized residual is

\[
\widehat a^k-\widehat b^k=(a^k-b^k)+(\widehat a^k-a^k)-\delta_0^k.
\]

Since \(\|\widehat a^k-a^k\|_\pi\leq2\|\delta_0^k\|_\pi\), Lemma 3.4 now correctly uses

\[
r_\pi^{k+1}\leq4\|\widehat u^k-u^\star\|_\pi+3\delta_\pi^k.
\]

Consequently, Theorem 3.5 now uses the residual constant \(12/(1-\gamma)+3\), not \(12/(1-\gamma)+1\). Both statements and the proof were corrected before final compilation. No numerical result changed.

## 5. Document QA

- Clean manuscript: **23 pages**.
- Highlighted manuscript: **23 pages**, with page-by-page extracted text identical to the clean copy.
- Response to reviewers: **4 pages**, identified as manuscript **ELSCAM-D-26-03479R2**.
- All **50 pages** were rendered and visually inspected; corrected theory and the complete termination table received additional page-level inspection.
- No clipping, missing figure, broken table continuation, placeholder, unresolved reference/citation, duplicate label/anchor, overfull box, or LaTeX/package warning remains in the final compile logs.
- The response cites Section 4.3 for certificates and uses the fresh retry count and regenerated per-scenario timing outcomes.

## 6. Principal verification-record hashes

- `results/verification.json`: `ad81e8c286b0df88bd969599fb992c5c632da850e069bf10c18bea7e713d5463`
- `results/structural_verification.json`: `9176428fee38ed09c08f84522d6c9ad32bb8f54611400187b4b6b884fc05114c`
- `results/training_verification.json`: `d890211c4d14572c8039f0c96d331c2f9c64e30378cc3b96db5104918218233a`
- `results/analysis.json`: `96890d6912ca756352c64f9f051cfc759b6a38b2fbad8e564b17c58be4ef6e67`
- Clean manuscript PDF: `110bfa59a3fb4f867373c29293a68dcfc9cb90c0c602515d32c5b5a6cac18264`
- Highlighted manuscript PDF: `9405095e9e248bb3ff47a45413a73c46d2b0e2a08e0c8e92d6b0c67b9fc9f366`
- Response PDF: `6aac08a6d13b6db296ce19aa47b61754ba7abc812f9a3a6206e46e73a446110c`

The external delivery manifest and the two archive-internal manifests were generated only after these checks and all source edits were complete.
