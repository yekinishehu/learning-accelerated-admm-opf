# ELSCAM-D-26-03479R2 — Verified 45-run final grid

Protocol: `strengthened-cold-v1`; cold scenario-QP solves; $\rho=0.01$; seeds 99–103; $S\in\{20,50,100\}$. All 45 rows passed source/raw-hash, residual, and certificate verification. Standalone/hybrid ML identity, full ML replay, and cold exact-tail replay passed bitwise for all 15 instances. Ten I/O-incomplete standalone-ML attempts were rerun with unchanged numerical settings and stable-copy handling; all failure records are retained.

Times are freshly measured algorithm wall times and exclude certificate evaluation. Signed speedup is $100(1-T_{\rm hybrid}/T_{\rm exact})$; negative values are slowdowns. The companion CSV contains complete certificate, solver, component-timing, provenance-hash, and retry fields.

| S | Seed | Method | ML iters | Tail/total iters | Final r (MW) | Final s (MW) | Stop | Wall (s) | Abs. gap | Rel. gap (%) | Hybrid speedup (%) | Verification |
|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 99 | ADMM | 0 | 141/141 | 9.957360e-05 | 7.643440e-12 | converged | 38.396191 | 0.001371 | 6.070408e-08 |  | pass |
| 20 | 99 | ADMM-ML | 18 | 0/18 | 2.282156e+00 | 1.488773e-02 | stall | 0.117881 | 69416.961575 | 3.073393e+00 |  | pass |
| 20 | 99 | ADMM-ML-H | 18 | 189/207 | 8.044510e-05 | 1.520786e-09 | converged | 54.300763 | 7.291976 | 3.227745e-04 | -41.422266 | pass |
| 20 | 100 | ADMM | 0 | 141/141 | 9.802167e-05 | 9.110086e-12 | converged | 38.231499 | 0.001382 | 5.822819e-08 |  | pass |
| 20 | 100 | ADMM-ML | 21 | 0/21 | 3.263502e+00 | 4.092065e-03 | stall | 0.116107 | 5008053.419259 | 2.110030e+02 |  | pass |
| 20 | 100 | ADMM-ML-H | 21 | 180/201 | 9.841539e-05 | 5.582334e-11 | converged | 50.284868 | 0.511409 | 2.154450e-05 | -31.527325 | pass |
| 20 | 101 | ADMM | 0 | 143/143 | 9.662231e-05 | 6.325610e-12 | converged | 38.994302 | 0.001375 | 5.885736e-08 |  | pass |
| 20 | 101 | ADMM-ML | 22 | 0/22 | 3.771897e+00 | 8.166991e-04 | stall | 0.121732 | 2040328.634554 | 8.732255e+01 |  | pass |
| 20 | 101 | ADMM-ML-H | 22 | 213/235 | 5.965757e-05 | 1.308779e-09 | converged | 57.092493 | 8.089959 | 3.461933e-04 | -46.412398 | pass |
| 20 | 102 | ADMM | 0 | 145/145 | 9.863342e-05 | 4.760176e-12 | converged | 37.975044 | 0.001438 | 6.115080e-08 |  | pass |
| 20 | 102 | ADMM-ML | 23 | 0/23 | 3.759290e+00 | 1.089654e-03 | stall | 0.169734 | 4482973.062111 | 1.907254e+02 |  | pass |
| 20 | 102 | ADMM-ML-H | 23 | 187/210 | 9.842917e-05 | 1.065979e-08 | converged | 47.583419 | 0.004533 | 1.928493e-07 | -25.301814 | pass |
| 20 | 103 | ADMM | 0 | 141/141 | 9.881464e-05 | 7.820040e-12 | converged | 37.732911 | 0.001564 | 6.792053e-08 |  | pass |
| 20 | 103 | ADMM-ML | 25 | 0/25 | 3.948813e+00 | 1.519023e-03 | stall | 0.139510 | 64672.692828 | 2.809586e+00 |  | pass |
| 20 | 103 | ADMM-ML-H | 25 | 234/259 | 9.992618e-05 | 9.636244e-10 | converged | 65.714996 | 6.186885 | 2.687455e-04 | -74.158299 | pass |
| 50 | 99 | ADMM | 0 | 110/110 | 9.374486e-05 | 1.336005e-05 | converged | 68.672567 | 0.101036 | 4.395796e-06 |  | pass |
| 50 | 99 | ADMM-ML | 23 | 0/23 | 4.244578e+00 | 1.249649e-03 | stall | 0.186349 | 140143.152309 | 6.111327e+00 |  | pass |
| 50 | 99 | ADMM-ML-H | 23 | 93/116 | 9.835640e-05 | 1.251566e-05 | converged | 60.908980 | 0.093257 | 4.057328e-06 | 11.305224 | pass |
| 50 | 100 | ADMM | 0 | 109/109 | 9.892903e-05 | 1.409776e-05 | converged | 64.292758 | 0.106581 | 4.629408e-06 |  | pass |
| 50 | 100 | ADMM-ML | 22 | 0/22 | 4.354190e+00 | 9.753152e-04 | stall | 0.123521 | 160022.013654 | 6.966147e+00 |  | pass |
| 50 | 100 | ADMM-ML-H | 22 | 92/114 | 9.969649e-05 | 1.286182e-05 | converged | 55.613082 | 0.096454 | 4.189530e-06 | 13.500239 | pass |
| 50 | 101 | ADMM | 0 | 110/110 | 9.713076e-05 | 1.383886e-05 | converged | 72.670786 | 0.104302 | 4.519173e-06 |  | pass |
| 50 | 101 | ADMM-ML | 15 | 0/15 | 6.512828e+00 | 4.956195e-02 | stall | 0.085151 | 97999.552585 | 4.255759e+00 |  | pass |
| 50 | 101 | ADMM-ML-H | 15 | 93/108 | 9.331250e-05 | 1.294922e-05 | converged | 60.971717 | 0.094960 | 4.114398e-06 | 16.098723 | pass |
| 50 | 102 | ADMM | 0 | 110/110 | 9.803588e-05 | 1.396665e-05 | converged | 66.693220 | 0.105984 | 4.566951e-06 |  | pass |
| 50 | 102 | ADMM-ML | 23 | 0/23 | 3.733472e+00 | 1.027623e-03 | stall | 0.132367 | 63758.381307 | 2.753653e+00 |  | pass |
| 50 | 102 | ADMM-ML-H | 23 | 93/116 | 9.842860e-05 | 1.292961e-05 | converged | 56.865570 | 0.095914 | 4.133007e-06 | 14.735605 | pass |
| 50 | 103 | ADMM | 0 | 110/110 | 9.778708e-05 | 1.393211e-05 | converged | 74.305832 | 0.104975 | 4.541616e-06 |  | pass |
| 50 | 103 | ADMM-ML | 23 | 0/23 | 3.841436e+00 | 1.153694e-03 | stall | 0.130770 | 76604.482274 | 3.321817e+00 |  | pass |
| 50 | 103 | ADMM-ML-H | 23 | 93/116 | 9.953727e-05 | 1.318293e-05 | converged | 62.481852 | 0.096771 | 4.186659e-06 | 15.912587 | pass |
| 100 | 99 | ADMM | 0 | 175/175 | 9.956668e-05 | 2.839109e-05 | converged | 232.552654 | 0.416828 | 1.806492e-05 |  | pass |
| 100 | 99 | ADMM-ML | 23 | 0/23 | 4.462712e+00 | 1.181359e-03 | stall | 0.183387 | 237672.811971 | 1.039152e+01 |  | pass |
| 100 | 99 | ADMM-ML-H | 23 | 142/165 | 9.872285e-05 | 2.739057e-05 | converged | 190.925184 | 0.411760 | 1.784530e-05 | 17.900235 | pass |
| 100 | 100 | ADMM | 0 | 176/176 | 9.801725e-05 | 2.793470e-05 | converged | 239.554967 | 0.409780 | 1.767672e-05 |  | pass |
| 100 | 100 | ADMM-ML | 24 | 0/24 | 4.822055e+00 | 1.128198e-03 | stall | 0.190541 | 176389.168006 | 7.675863e+00 |  | pass |
| 100 | 100 | ADMM-ML-H | 24 | 143/167 | 9.710043e-05 | 2.699727e-05 | converged | 196.291104 | 0.403958 | 1.742556e-05 | 18.060099 | pass |
| 100 | 101 | ADMM | 0 | 176/176 | 9.824124e-05 | 2.799338e-05 | converged | 221.313438 | 0.411712 | 1.779617e-05 |  | pass |
| 100 | 101 | ADMM-ML | 15 | 0/15 | 8.180806e+00 | 4.664150e-02 | stall | 0.116846 | 189288.202300 | 8.253508e+00 |  | pass |
| 100 | 101 | ADMM-ML-H | 15 | 143/158 | 9.704313e-05 | 2.704493e-05 | converged | 180.012976 | 0.405691 | 1.753593e-05 | 18.661525 | pass |
| 100 | 102 | ADMM | 0 | 177/177 | 9.759147e-05 | 2.778778e-05 | converged | 238.049090 | 0.407047 | 1.750385e-05 |  | pass |
| 100 | 102 | ADMM-ML | 23 | 0/23 | 3.758098e+00 | 9.431760e-04 | stall | 0.178492 | 73880.033385 | 3.204767e+00 |  | pass |
| 100 | 102 | ADMM-ML-H | 23 | 143/166 | 9.910732e-05 | 2.762870e-05 | converged | 190.748675 | 0.412176 | 1.772437e-05 | 19.870025 | pass |
| 100 | 103 | ADMM | 0 | 177/177 | 9.902301e-05 | 2.817956e-05 | converged | 233.820832 | 0.411135 | 1.764916e-05 |  | pass |
| 100 | 103 | ADMM-ML | 23 | 0/23 | 4.341781e+00 | 1.051866e-03 | stall | 0.186949 | 38558.305180 | 1.669789e+00 |  | pass |
| 100 | 103 | ADMM-ML-H | 23 | 144/167 | 9.852014e-05 | 2.749253e-05 | converged | 205.481722 | 0.404666 | 1.737145e-05 | 12.120010 | pass |

## Mixed timing outcomes retained

- $S=20$: min -74.158299%, median -41.422266%, max -25.301814%; faster 0/5, slower 5/5.
- $S=50$: min 11.305224%, median 14.735605%, max 16.098723%; faster 5/5, slower 0/5.
- $S=100$: min 12.120010%, median 18.060099%, max 19.870025%; faster 5/5, slower 0/5.

## Evidence gate

- Verified runs: 45/45.
- Source/raw hashes: pass.
- Residual replay: exact pass.
- Certificate replay: pass; maximum replay difference 0.
- Standalone/hybrid ML identity: bitwise pass for all 15 instances.
- Full ML replay: bitwise pass for all 15 instances.
- Exact-tail replay: bitwise pass for all 15 instances.
- Stable-copy I/O retries completed: 10.
