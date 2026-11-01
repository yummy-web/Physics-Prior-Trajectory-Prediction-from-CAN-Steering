# Lambda calibration regret audit

This is a descriptive audit of frozen results; it does not retrain or select models.

| Dataset | N | Full FDE | Shrunk FDE | Oracle FDE | Shrunk gain vs full | Full-to-oracle gap closed |
|---|---:|---:|---:|---:|---:|---:|
| comma2k19 | 3 | 2.103 | 1.824 | 1.809 | 0.279 | 0.948 |
| comma2k19 | 5 | 1.833 | 1.749 | 1.727 | 0.084 | 0.789 |
| comma2k19 | 10 | 1.689 | 1.686 | 1.665 | 0.002 | 0.104 |
| comma2k19 | 20 | 1.587 | 1.588 | 1.582 | -0.001 | -0.147 |
| nuScenes | 3 | 4.146 | 2.483 | 2.451 | 1.663 | 0.981 |
| nuScenes | 5 | 3.405 | 2.445 | 2.428 | 0.960 | 0.983 |
| nuScenes | 10 | 2.844 | 2.425 | 2.422 | 0.420 | 0.994 |
| nuScenes | 20 | 2.489 | 2.352 | 2.334 | 0.137 | 0.885 |

Interpretation:
- The oracle uses test-set information and is only a descriptive upper bound.
- Regret and fraction-closed values are point estimates; they are not significance claims.
- The source nuScenes JSON contains scene-clustered CIs for shrink-vs-full, but not for every regret quantity.
