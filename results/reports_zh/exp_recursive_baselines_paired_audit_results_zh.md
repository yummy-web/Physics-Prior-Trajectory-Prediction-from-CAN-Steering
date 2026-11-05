# RLS and steering-input EKF paired audit

The audit uses the same route/scene groups as the baseline experiment.

| Dataset | Group unit | Fold | Groups | RLS-EKF mean difference | 95% CI | RLS better fraction | permutation p |
|---|---|---:|---:|---:|---|---:|---:|
| comma | route | 1 | 14 | 0.133 | [-0.038, 0.336] | 0.429 | 0.231 |
| comma | route | 2 | 14 | -0.058 | [-0.271, 0.193] | 0.571 | 0.6425 |
| comma | route | 3 | 14 | -0.162 | [-0.367, 0.001] | 0.714 | 0.0983 |
| nuscenes | scene | 1 | 67 | -0.793 | [-0.975, -0.626] | 0.881 | 5e-05 |
| nuscenes | scene | 2 | 67 | -0.762 | [-0.903, -0.614] | 0.940 | 5e-05 |
| nuscenes | scene | 3 | 66 | -0.670 | [-0.840, -0.497] | 0.848 | 5e-05 |

Interpretation: negative RLS-EKF means RLS has lower FDE. This is only a paired comparison between the two supplementary baselines.
