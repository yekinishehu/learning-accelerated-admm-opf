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
| R2.1 — Error monitoring and weaker conditions | **Addressed** | Sampled audits and validation diagnostics are not called summability certificates. The exact tail supplies the unconditional guarantee. The final theory pass corrected the ADMM/DRS residual indexing and rederived the learned-residual bound. |
| R2.2 — Scenario-count dependence | **Addressed** | The scaling condition is constant \(\rho S\), hence \(\rho\propto1/S\). A new remark separates triangle-inequality conservativeness from the sharp scalar-quadratic value of \(\kappa_\rho\) and the sharp weighted/max norm conversion. |
| R2.3 — Zero-padding, graphs, and bus ordering | **Addressed** | Fixed-order/shared-label limitations, lack of permutation invariance, graph alternatives, and spectral-embedding limitations are stated without untested transfer claims. |
| R2.4 — Notation | **Addressed** | Exact proximal minimizers, learned/restored initial blocks, optimal recourse vectors, residuals, DRS objects, and certificate quantities are distinguished. |
| R2.5 — Exact versus learned updates | **Addressed** | Algorithm 2 now consistently indexes the learned/restored block, multiplier, and residuals by \(k+1\). The fixed proxy remains an uncertified heuristic; exact-tail convergence is assigned only to the hybrid. All ML stalls and all unfavorable hybrid timings remain visible. Bitwise standalone/hybrid ML identity, full ML replay, and cold exact-tail replay passed for all 15 instances. |

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

**Fresh replay on 2026-11-15:** the unchanged `code/verify_final.py`, `experiments/verify_structural.py`, and `experiments/verify_training_replay.py` drivers were run sequentially with OpenBLAS, OpenMP, and MKL limited to one thread. All three completed successfully. Their regenerated JSON reports are byte-for-byte identical to the archived reports hashed in Section 6: 45/45 residual/certificate verifications passed, all structural and ML/exact-tail replay checks passed, and all 30 training shards plus every checkpoint tensor/normalization array replayed bitwise. The disclosed QP solver warnings remained in the replay output and are not hidden or reclassified as failures.

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

## 4. Final theory and algorithm corrections

The last proof audit identified an ADMM/DRS residual-identity mismatch. Lemma 2.1 gives \(b^k=z_0^{k+1}\) and \(a^k=x^{k+2}\), so the exact consensus residual at step \(k+1\) is

\[
a^{k-1}-b^k=x^{k+1}-z_0^{k+1},
\]

not \(a^k-b^k\). For the learned iteration, the corresponding realized residual is \(\widehat a^{k-1}-\widehat b^k\). With exact comparison blocks evaluated at the learned states and \(d_k=\|\widehat u^k-u^\star\|_\pi\), Lemma 3.4 now proves, for every \(k\geq1\),

\[
r_\pi^{k+1}\leq3d_{k-1}+d_k+2\delta_\pi^{k-1}+\delta_\pi^k,
\qquad
r^{k+1}\leq\sqrt S\,r_\pi^{k+1}.
\]

The four contributions are \(3d_{k-1}\), \(d_k\), \(2\delta_\pi^{k-1}\), and \(\delta_\pi^k\). Theorem 3.5 sums their limsup bounds and retains the residual constant \(12/(1-\gamma)+3\), now with a valid derivation. Theorem 3.8 inserts its linear distance bound at indices \(k-1\) and \(k\), including the error terms \(Cq^{k-1}\) and \(Cq^k\); the stated \(O(\max\{\gamma,q\}^k)\) residual consequence therefore includes the additional factor \(k\) when \(\gamma=q\). The initial residual \(r^1\) is treated separately. No numerical result changed.

Algorithm 2 now runs its displayed iteration for \(k=0,\ldots,K_{\mathrm{budget}}-1\), produces \(\widehat z_{s,0}^{k+1}\) and \(\lambda_s^{k+1}\), and computes \(r^{k+1}\) and \(s^{k+1}\). Improvement, stall, and exit tests use those same indices. The primal-residual exit remains explicitly uncertified.

A short remark after Proposition 3.10 now distinguishes sharpness from conservativeness. The triangle-inequality coefficients in the inexact-DRS and residual bounds, and the resulting ultimate constant, are not claimed to be sharp. By contrast, \(\kappa_\rho\) is sharp for an unconstrained scalar quadratic, and the weighted/max conversion is attained by a vector supported on a scenario attaining \(\pi_{\min}\); under uniform weights this gives \(\|u\|_{\max}=\sqrt S\,\|u\|_\pi\).

As a separate sanity check—not a substitute for the archived 45-run evidence—a deterministic scalar learned-DRS test with seed 20261115, \(\mu=2\), and \(\rho=0.7\) checked the repaired identity and inequality at 29 nontrivial iterations. All 29 checks passed; the maximum observed residual/RHS ratio was 0.7532601128746527.

## 5. Document QA

- Clean manuscript: **24 pages**.
- Highlighted manuscript: **24 pages**, with page-by-page extracted text identical to the clean copy.
- Response to reviewers: **4 pages**, identified as manuscript **ELSCAM-D-26-03479R2**.
- All **52 pages** were rendered and visually inspected; corrected theory, Algorithm 2, and the complete termination table received additional page-level inspection.
- No clipping, missing figure, broken table continuation, placeholder, unresolved reference/citation, duplicate label/anchor, overfull box, or LaTeX/package warning remains in the final compile logs.
- The response cites Section 4.3 for certificates, records the corrected residual identity and Algorithm 2 indexing, and uses the fresh retry count and regenerated per-scenario timing outcomes.

## 6. Principal verification-record hashes

- `results/verification.json`: `ad81e8c286b0df88bd969599fb992c5c632da850e069bf10c18bea7e713d5463`
- `results/structural_verification.json`: `9176428fee38ed09c08f84522d6c9ad32bb8f54611400187b4b6b884fc05114c`
- `results/training_verification.json`: `d890211c4d14572c8039f0c96d331c2f9c64e30378cc3b96db5104918218233a`
- `results/analysis.json`: `96890d6912ca756352c64f9f051cfc759b6a38b2fbad8e564b17c58be4ef6e67`
- Clean manuscript PDF: `6bab27c20023535667c0be7324d6235b2b1bc94d4906b04f8491321924a38544`
- Highlighted manuscript PDF: `83f3faa17cfe66d7187db22deaf5fa447afe5dc5c4a177ff941f2e89cd6fe4e0`
- Response PDF: `53bb7d24c5fbb075fcedf3cfa8e37eec736883cd2cf2c715bbba9bd521df71db`

The repository `SHA256SUMS` manifest was refreshed after the document and metadata edits. It includes the corrected PDF/TeX documents and `restore_layout.py` as well as the archived evidence and verification reports. The restoration utility reconstructs missing canonical paths without deleting flattened files or rewriting existing files. No fingerprinted executed source, protocol file, raw NPZ archive, run summary, or per-run manifest was modified during this theory/documentation repair.
