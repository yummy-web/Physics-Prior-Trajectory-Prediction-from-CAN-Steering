"""
run_all_experiments.py
Run the full paper experiment suite (loader + comma2k19), generate all figures
and result JSONs.

Usage:
    conda activate loader_paper
    python run_all_experiments.py

Outputs:
    figures/    all paper figures (PNG)
    results/    all numeric results (JSON)

Note: exp4 (bias-variance) and exp7 (GRU) train several models; a few minutes on CPU.
"""
import time
import importlib

EXPERIMENTS = [
    ("exp1_steering_identification", "Exp1: steering->curvature identification (Innovation 1)"),
    ("exp2_prediction_and_crossover", "Exp2: prediction + crossover + anisotropy (Innovation 2/3)"),
    ("exp3_baselines_ci", "Exp3: full baselines + bootstrap CI + significance"),
    ("exp4_bias_variance", "Exp4: bias-variance theory of the crossover"),
    ("exp5_ablation", "Exp5: physics-model ablation"),
    ("exp6_robustness", "Exp6: CAN robustness (physics vs data-driven)"),
    ("exp7_deep_baseline", "Exp7: deep GRU baseline vs physics (needs torch; auto-skips)"),
    ("exp8_grouped_comparison", "Exp8: grouped baselines (fair) + ADE + lat/lon decomposition"),
    ("exp9_stratified", "Exp9: stratified validation (speed x turn)"),
    ("exp10_hybrid", "Exp10: improved physics-data hybrid (method contribution)"),
    ("exp11_cross_dataset", "Exp11: cross-DATASET validation (loader+comma2k19+nuScenes)"),
    ("exp12_pi_deep_comparison", "Exp12: vs physics-informed deep baselines (PERL-style/gated hybrid)"),
]


def run(module_name, title):
    print("\n" + "#" * 76)
    print("#  " + title)
    print("#" * 76)
    t0 = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    print("[%s] took %.1fs" % (title, time.time() - t0))


if __name__ == "__main__":
    t0 = time.time()
    for mod, title in EXPERIMENTS:
        run(mod, title)
    print("\n" + "=" * 76)
    print("  ALL EXPERIMENTS DONE. total %.1fs" % (time.time() - t0))
    print("  figures/ and results/ populated.")
    print("=" * 76)
