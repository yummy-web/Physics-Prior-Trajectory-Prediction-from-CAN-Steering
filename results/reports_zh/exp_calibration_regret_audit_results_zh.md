# Lambda calibration regret audit

This is a descriptive audit of frozen results; it does not retrain or select models.

| Dataset | N | Full FDE | Shrunk FDE | Oracle FDE | Shrunk gain vs full | Full-to-oracle gap closed |
|---|---:|---:|---:|---:|---:|---:|
| comma2k19 | 3 | 2.195 | 1.868 | 1.858 | 0.327 | 0.971 |
| comma2k19 | 5 | 1.854 | 1.799 | 1.772 | 0.055 | 0.667 |
| comma2k19 | 10 | 1.733 | 1.747 | 1.708 | -0.014 | -0.538 |
| comma2k19 | 20 | 1.580 | 1.584 | 1.578 | -0.004 | -2.098 |
| nuScenes | 3 | 4.653 | 2.868 | 2.832 | 1.785 | 0.980 |
| nuScenes | 5 | 3.769 | 2.822 | 2.800 | 0.947 | 0.977 |
| nuScenes | 10 | 3.318 | 2.792 | 2.772 | 0.526 | 0.964 |
| nuScenes | 20 | 2.850 | 2.686 | 2.662 | 0.164 | 0.869 |

Interpretation:
- The oracle uses test-set information and is only a descriptive upper bound.
- Regret and fraction-closed values are point estimates; they are not significance claims.
- The source nuScenes JSON contains scene-clustered CIs for shrink-vs-full, but not for every regret quantity.
