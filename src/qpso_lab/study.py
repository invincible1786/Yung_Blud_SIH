import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tqdm import tqdm

from src.algorithms.aco_mmas import ACOMMAS
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.nn_clarke_wright import NearestNeighborClarkeWright
from src.algorithms.qpso import QPSO
from src.algorithms.qpso_optimized import QPSOOptimized
from src.algorithms.standard_pso import StandardPSO
from src.benchmark.runner import get_algo_key
from src.qpso_lab.evaluation import UniformEvaluator, reference_summary
from src.utils.graph_utils import load_instance_resources

OUT_DIR = os.path.join("results", "qpso_study")
INSTANCES_DIR = os.path.join("data", "instances")
ALL_INSTANCES = ["instance_n20", "instance_n50", "instance_n100"]


def _safe_out(path: str) -> str:
    """Refuses to write anywhere outside results/qpso_study/ -- a mechanical
    guard, not just a convention, against ever touching results/logs/."""
    allowed_root = os.path.abspath(OUT_DIR)
    target = os.path.abspath(path)
    if os.path.commonpath([target, allowed_root]) != allowed_root:
        raise RuntimeError(f"Refusing to write outside {OUT_DIR}: {path}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return target


def _load_instance(name: str) -> Dict[str, Any]:
    with open(os.path.join(INSTANCES_DIR, f"{name}.json"), "r") as f:
        return json.load(f)


def _seed_all(seed: int) -> None:
    """Seeds both np.random and stdlib random. The frozen
    src/benchmark/runner.py only seeds np.random, but DEAP's cxOrdered /
    selTournament (used by the frozen GeneticAlgorithm) draw from stdlib
    `random`, which the frozen runner leaves unseeded -- so its own GA results
    are not actually seed-reproducible. Seeding both here fixes that for this
    study without touching runner.py."""
    np.random.seed(seed)
    random.seed(seed)


# ----------------------------------------------------------------------
# Solver registry
# ----------------------------------------------------------------------


def _baseline_solvers(max_iter: int) -> List[Tuple[str, Any, Dict[str, Any]]]:
    """Configs copied verbatim from src/benchmark/runner.py's
    `algorithms_config` so the comparison is like-for-like with the frozen
    suite's own settings."""
    return [
        ("Nearest Neighbor", NearestNeighborClarkeWright, {"method": "nn", "max_iter": max_iter}),
        ("Clarke-Wright", NearestNeighborClarkeWright, {"method": "clarke_wright", "max_iter": max_iter}),
        ("Standard PSO", StandardPSO, {"swarm_size": 30, "max_iter": max_iter, "w": 0.7, "c1": 1.5, "c2": 1.5}),
        ("QPSO (baseline)", QPSO, {"swarm_size": 30, "max_iter": max_iter, "beta_start": 1.0, "beta_end": 0.5}),
        (
            "Genetic Algorithm",
            GeneticAlgorithm,
            {
                "population_size": 30,
                "max_generations": max_iter,
                "crossover_rate": 0.8,
                "mutation_rate": 0.2,
                "apply_2opt": True,
            },
        ),
        (
            "Ant Colony (MMAS)",
            ACOMMAS,
            {"num_ants": 15, "max_iter": max_iter, "alpha": 1.0, "beta": 3.0, "evaporation_rate": 0.1},
        ),
    ]


# Component ablation ladder -- each rung adds one enhancement on top of the
# last, isolating that enhancement's individual contribution. V5 (full
# solver) uses QPSOOptimized's own defaults (fleet-bounded Split, Lamarckian
# local search, elitism/restarts, FFD seeding, reflect bounds).
ABLATION_VARIANTS: List[Tuple[str, Dict[str, Any]]] = [
    (
        "V1_prins_split_unconstrained",
        dict(
            fleet_bounded=False,
            use_local_search=False,
            elitism_restarts=False,
            seed_mode="random",
            bounds_mode="clip",
        ),
    ),
    (
        "V2_fleet_bounded_split",
        dict(
            fleet_bounded=True,
            use_local_search=False,
            elitism_restarts=False,
            seed_mode="random",
            bounds_mode="clip",
        ),
    ),
    (
        "V3_local_search_baldwinian",
        dict(
            fleet_bounded=True,
            use_local_search=True,
            lamarckian_writeback=False,
            elitism_restarts=False,
            seed_mode="random",
            bounds_mode="clip",
        ),
    ),
    (
        "V4_lamarckian_writeback",
        dict(
            fleet_bounded=True,
            use_local_search=True,
            lamarckian_writeback=True,
            elitism_restarts=False,
            seed_mode="random",
            bounds_mode="clip",
        ),
    ),
    ("V5_full_solver", dict()),
]


# ----------------------------------------------------------------------
# Core run loop (shared by the comparison and the ablation study)
# ----------------------------------------------------------------------


def _run_variant_over_seeds(
    algo_name: str,
    algo_cls: Any,
    cfg: Dict[str, Any],
    instance: Dict[str, Any],
    evaluator: UniformEvaluator,
    num_seeds: int,
    instance_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], Optional[Dict[str, Any]]]:
    rows = []
    histories: Dict[str, List[float]] = {}
    seed_results = []

    for seed in tqdm(range(1, num_seeds + 1), desc=f"{instance_name}/{algo_name}", leave=False):
        _seed_all(seed)
        try:
            solver = algo_cls(instance, cfg)
            t0 = time.time()
            result = solver.solve()
            runtime = result.runtime_seconds if result.runtime_seconds else time.time() - t0
        except Exception as exc:  # noqa: BLE001 -- study must keep going if one solver/seed errors
            print(f"  [error] {algo_name} seed={seed} on {instance_name}: {exc}")
            continue

        evaluation, reported_minus_eval = evaluator.evaluate_result(result)
        min_cost = min(result.convergence_history) if result.convergence_history else evaluation.penalized_cost
        converged_iter = result.convergence_history.index(min_cost) if result.convergence_history else 0

        rows.append(
            {
                "instance_id": instance_name,
                "algorithm": algo_name,
                "seed": seed,
                "reported_cost": float(result.total_cost),
                "eval_distance": evaluation.distance,
                "eval_num_routes": evaluation.num_routes,
                "eval_penalty": evaluation.penalty,
                "eval_penalized_cost": evaluation.penalized_cost,
                "reported_minus_eval": reported_minus_eval,
                "feasible_capacity": evaluation.feasible_capacity,
                "feasible_coverage": evaluation.feasible_coverage,
                "feasible_fleet": evaluation.feasible_fleet,
                "lower_bound_vehicles": evaluation.lower_bound_vehicles,
                "vehicles_allowed": evaluation.vehicles_allowed,
                "runtime_seconds": runtime,
                "converged_iteration": converged_iter,
            }
        )
        histories[f"{instance_name}_{algo_name}_{seed}"] = [float(c) for c in result.convergence_history]
        seed_results.append((evaluation, result))

    best_entry = None
    if seed_results:
        best_eval, best_result = min(seed_results, key=lambda pair: (pair[0].num_routes, pair[0].distance))
        best_entry = {
            # Some solvers (e.g. ACOMMAS, which samples node ids via
            # np.random.choice) return numpy int64 node ids rather than plain
            # Python ints, which json.dump cannot serialize -- cast explicitly.
            "routes": [[int(node) for node in route] for route in best_result.routes],
            "distance": best_eval.distance,
            "num_routes": best_eval.num_routes,
            "penalized_cost": best_eval.penalized_cost,
        }
    return rows, histories, best_entry


# ----------------------------------------------------------------------
# Public entry points
# ----------------------------------------------------------------------


def run_comparison(
    num_seeds: int = 10, max_iter: int = 100, instances: Optional[List[str]] = None, quick: bool = False
) -> pd.DataFrame:
    instances = instances or list(ALL_INSTANCES)
    if quick:
        num_seeds = 2
        max_iter = min(max_iter, 20)
        instances = instances[:1]

    all_rows: List[Dict[str, Any]] = []
    all_histories: Dict[str, List[float]] = {}
    best_routes: Dict[str, Dict[str, Any]] = {}

    for instance_name in instances:
        instance = _load_instance(instance_name)
        distance_matrix, node_id_map = load_instance_resources(instance)
        evaluator = UniformEvaluator(instance, distance_matrix, node_id_map)
        best_routes[instance_name] = {}

        solvers = _baseline_solvers(max_iter) + [
            ("QPSO-Optimized", QPSOOptimized, {"swarm_size": 30, "max_iter": max_iter})
        ]

        print(f"\n=== {instance_name} (lower bound vehicles: {evaluator.lower_bound_vehicles}, "
              f"declared fleet: {evaluator.num_vehicles}) ===")

        for algo_name, algo_cls, cfg in solvers:
            rows, histories, best_entry = _run_variant_over_seeds(
                algo_name, algo_cls, cfg, instance, evaluator, num_seeds, instance_name
            )
            all_rows.extend(rows)
            all_histories.update(histories)
            if best_entry is not None:
                best_routes[instance_name][get_algo_key(algo_name)] = best_entry

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(_safe_out(os.path.join(OUT_DIR, "results_raw.csv")), index=False)

    with open(_safe_out(os.path.join(OUT_DIR, "convergence.json")), "w") as f:
        json.dump(all_histories, f, indent=2)
    with open(_safe_out(os.path.join(OUT_DIR, "best_routes.json")), "w") as f:
        json.dump(best_routes, f, indent=2)

    df_summary = _summarize(df_raw)
    df_summary.to_csv(_safe_out(os.path.join(OUT_DIR, "results_summary.csv")), index=False)

    leaderboard_rows = []
    for instance_name, algos in best_routes.items():
        for algo_key, entry in algos.items():
            leaderboard_rows.append(
                {
                    "instance_id": instance_name,
                    "algorithm": algo_key,
                    "num_routes": entry["num_routes"],
                    "distance": entry["distance"],
                    "penalized_cost": entry["penalized_cost"],
                }
            )
    df_leaderboard = pd.DataFrame(leaderboard_rows)
    if not df_leaderboard.empty:
        df_leaderboard = df_leaderboard.sort_values(by=["instance_id", "num_routes", "distance"])
    df_leaderboard.to_csv(_safe_out(os.path.join(OUT_DIR, "leaderboard.csv")), index=False)

    print(f"\nSaved comparison outputs to '{OUT_DIR}/'.")
    return df_summary


def run_ablation(num_seeds: int = 10, max_iter: int = 100, instances: Optional[List[str]] = None) -> pd.DataFrame:
    instances = instances or list(ALL_INSTANCES)
    all_rows: List[Dict[str, Any]] = []

    for instance_name in instances:
        instance = _load_instance(instance_name)
        distance_matrix, node_id_map = load_instance_resources(instance)
        evaluator = UniformEvaluator(instance, distance_matrix, node_id_map)

        for variant_name, flags in ABLATION_VARIANTS:
            cfg = {"swarm_size": 30, "max_iter": max_iter, **flags}
            rows, _, _ = _run_variant_over_seeds(
                variant_name, QPSOOptimized, cfg, instance, evaluator, num_seeds, instance_name
            )
            all_rows.extend(rows)

    df_raw = pd.DataFrame(all_rows)
    df_raw.to_csv(_safe_out(os.path.join(OUT_DIR, "ablation_raw.csv")), index=False)
    df_summary = _summarize(df_raw)
    df_summary.to_csv(_safe_out(os.path.join(OUT_DIR, "ablation_summary.csv")), index=False)
    print(f"\nSaved ablation outputs to '{OUT_DIR}/'.")
    return df_summary


def run_reference(time_limit_seconds: int = 30, instances: Optional[List[str]] = None) -> Dict[str, Any]:
    instances = instances or list(ALL_INSTANCES)
    results = {}
    for instance_name in instances:
        instance = _load_instance(instance_name)
        distance_matrix, node_id_map = load_instance_resources(instance)
        print(f"Solving OR-Tools reference for {instance_name} (time limit {time_limit_seconds}s x2)...")
        results[instance_name] = reference_summary(instance, distance_matrix, node_id_map, time_limit_seconds)

    serializable = {
        name: {k: v for k, v in r.items() if k not in ("declared_routes", "relaxed_routes")}
        for name, r in results.items()
    }
    with open(_safe_out(os.path.join(OUT_DIR, "ortools_reference.json")), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved OR-Tools reference summary to '{OUT_DIR}/ortools_reference.json'.")
    return results


def _summarize(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return df_raw
    grouped = df_raw.groupby(["instance_id", "algorithm"])
    records = []
    for (instance_id, algorithm), group in grouped:
        records.append(
            {
                "instance_id": instance_id,
                "algorithm": algorithm,
                "mean_distance": group["eval_distance"].mean(),
                "std_distance": group["eval_distance"].std() or 0.0,
                "best_distance": group["eval_distance"].min(),
                "mean_num_routes": group["eval_num_routes"].mean(),
                "min_num_routes": group["eval_num_routes"].min(),
                "mean_penalized_cost": group["eval_penalized_cost"].mean(),
                "std_penalized_cost": group["eval_penalized_cost"].std() or 0.0,
                "mean_runtime_sec": group["runtime_seconds"].mean(),
                "mean_converged_iteration": group["converged_iteration"].mean(),
                "feasible_fleet_rate": group["feasible_fleet"].mean(),
                "mean_reported_minus_eval": group["reported_minus_eval"].mean(),
            }
        )
    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------


def generate_report() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    summary_path = os.path.join(OUT_DIR, "results_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing '{summary_path}'. Run `study.py run` first.")
    df_summary = pd.read_csv(summary_path)

    convergence_path = os.path.join(OUT_DIR, "convergence.json")
    histories = {}
    if os.path.exists(convergence_path):
        with open(convergence_path, "r") as f:
            histories = json.load(f)

    plots_dir = os.path.join(OUT_DIR, "plots")
    sns.set_theme(style="whitegrid")

    for inst in df_summary["instance_id"].unique():
        # Convergence
        plt.figure(figsize=(10, 6))
        for algo in df_summary[df_summary["instance_id"] == inst]["algorithm"].unique():
            runs = [histories[k] for k in histories if k.startswith(f"{inst}_{algo}_")]
            if runs:
                min_len = min(len(r) for r in runs)
                avg = np.mean([r[:min_len] for r in runs], axis=0)
                plt.plot(avg, label=algo, linewidth=2)
        plt.title(f"Convergence (penalized cost) -- {inst}")
        plt.xlabel("Iteration")
        plt.ylabel("Penalized cost")
        plt.yscale("log")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(_safe_out(os.path.join(plots_dir, f"convergence_{inst}.png")), dpi=200)
        plt.close()

        # Distance + route count, side by side
        df_inst = df_summary[df_summary["instance_id"] == inst].sort_values("mean_num_routes")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.barplot(data=df_inst, x="algorithm", y="mean_distance", hue="algorithm", legend=False, ax=axes[0])
        axes[0].set_title(f"Mean distance -- {inst}")
        axes[0].tick_params(axis="x", rotation=30)
        sns.barplot(data=df_inst, x="algorithm", y="mean_num_routes", hue="algorithm", legend=False, ax=axes[1])
        axes[1].set_title(f"Mean vehicles used -- {inst}")
        axes[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        plt.savefig(_safe_out(os.path.join(plots_dir, f"distance_routes_{inst}.png")), dpi=200)
        plt.close()

    _write_markdown_report(df_summary)
    print(f"Generated report at '{OUT_DIR}/qpso_study_report.md'.")


def _write_markdown_report(df_summary: pd.DataFrame) -> None:
    report_path = _safe_out(os.path.join(OUT_DIR, "qpso_study_report.md"))

    leaderboard_path = os.path.join(OUT_DIR, "leaderboard.csv")
    df_leaderboard = pd.read_csv(leaderboard_path) if os.path.exists(leaderboard_path) else pd.DataFrame()

    ablation_path = os.path.join(OUT_DIR, "ablation_summary.csv")
    df_ablation = pd.read_csv(ablation_path) if os.path.exists(ablation_path) else None

    reference_path = os.path.join(OUT_DIR, "ortools_reference.json")
    reference = None
    if os.path.exists(reference_path):
        with open(reference_path, "r") as f:
            reference = json.load(f)

    with open(report_path, "w") as f:
        f.write("# Optimized QPSO Study\n\n")
        f.write(
            "This report re-scores every solver's `.routes` through one uniform evaluator "
            "(src/qpso_lab/evaluation.py) instead of trusting each solver's self-reported "
            "`total_cost`. Two defects in the frozen benchmark make this necessary: "
            "genetic_algorithm.py drops the fleet penalty after 2-opt, and the n=50/n=100 "
            "instances need more vehicles than their declared fleet just to cover total "
            "demand (a bin-packing lower bound), so every solution to those instances is "
            "necessarily penalized. Neither frozen file was modified.\n\n"
        )

        f.write("## Leaderboard (ranked by num_routes, then distance)\n\n")
        if not df_leaderboard.empty:
            f.write(df_leaderboard.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Summary statistics (10-seed mean unless noted)\n\n")
        f.write(df_summary.to_markdown(index=False))
        f.write("\n\n")

        if df_ablation is not None:
            f.write("## Component ablation (QPSO-Optimized)\n\n")
            f.write(
                "Each rung adds one enhancement on top of the last; V3 vs V4 isolates the "
                "Lamarckian write-back specifically (Baldwinian control vs. the real thing).\n\n"
            )
            f.write(df_ablation.to_markdown(index=False))
            f.write("\n\n")

        if reference is not None:
            f.write("## OR-Tools reference\n\n")
            f.write("| instance | lower bound vehicles | declared fleet feasible? | relaxed fleet | relaxed distance |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- |\n")
            for inst, r in reference.items():
                relaxed_distance = f"{r['relaxed_distance']:.2f}" if r["relaxed_distance"] is not None else "infeasible within time limit"
                f.write(
                    f"| {inst} | {r['lower_bound_vehicles']} | {r['declared_feasible']} "
                    f"| {r['relaxed_fleet']} | {relaxed_distance} |\n"
                )
            f.write("\n")

        f.write("## Plots\n\n")
        for inst in df_summary["instance_id"].unique():
            f.write(f"### {inst}\n\n")
            f.write(f"![Convergence](plots/convergence_{inst}.png)\n\n")
            f.write(f"![Distance and routes](plots/distance_routes_{inst}.png)\n\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized QPSO comparison study")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run the full solver comparison")
    p_run.add_argument("--seeds", type=int, default=10)
    p_run.add_argument("--max-iter", type=int, default=100)
    p_run.add_argument("--quick", action="store_true", help="2 seeds, n20 only, reduced iterations")

    p_abl = sub.add_parser("ablation", help="Run the component ablation ladder")
    p_abl.add_argument("--seeds", type=int, default=10)
    p_abl.add_argument("--max-iter", type=int, default=100)

    p_ref = sub.add_parser("reference", help="Run the OR-Tools reference solver")
    p_ref.add_argument("--time-limit", type=int, default=30)

    sub.add_parser("report", help="Generate plots + markdown report from saved CSVs")

    args = parser.parse_args()

    if args.command == "run":
        run_comparison(num_seeds=args.seeds, max_iter=args.max_iter, quick=args.quick)
    elif args.command == "ablation":
        run_ablation(num_seeds=args.seeds, max_iter=args.max_iter)
    elif args.command == "reference":
        run_reference(time_limit_seconds=args.time_limit)
    elif args.command == "report":
        generate_report()


if __name__ == "__main__":
    main()
