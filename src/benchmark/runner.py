import os
import json
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.algorithms.nn_clarke_wright import NearestNeighborClarkeWright
from src.algorithms.standard_pso import StandardPSO
from src.algorithms.qpso import QPSO
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.aco_mmas import ACOMMAS

def run_benchmarks(num_seeds: int = 10, max_iter: int = 100):
    instances_dir = os.path.join("data", "instances")
    results_dir = os.path.join("results", "logs")
    os.makedirs(results_dir, exist_ok=True)

    instance_files = [f for f in os.listdir(instances_dir) if f.endswith(".json")]
    instance_files.sort()

    print(f"Found {len(instance_files)} instances: {instance_files}")

    raw_results = []
    convergence_histories = {}

    # Define algorithms to run
    # Format: (algo_name, class_reference, config_dict)
    algorithms_config = [
        ("Nearest Neighbor", NearestNeighborClarkeWright, {"method": "nn", "max_iter": max_iter}),
        ("Clarke-Wright", NearestNeighborClarkeWright, {"method": "clarke_wright", "max_iter": max_iter}),
        ("Standard PSO", StandardPSO, {"swarm_size": 30, "max_iter": max_iter, "w": 0.7, "c1": 1.5, "c2": 1.5}),
        ("QPSO", QPSO, {"swarm_size": 30, "max_iter": max_iter, "beta_start": 1.0, "beta_end": 0.5}),
        ("Genetic Algorithm", GeneticAlgorithm, {"population_size": 30, "max_generations": max_iter, "crossover_rate": 0.8, "mutation_rate": 0.2, "apply_2opt": True}),
        ("Ant Colony (MMAS)", ACOMMAS, {"num_ants": 15, "max_iter": max_iter, "alpha": 1.0, "beta": 3.0, "evaporation_rate": 0.1})
    ]

    for inst_file in instance_files:
        inst_path = os.path.join(instances_dir, inst_file)
        with open(inst_path, "r") as f:
            instance = json.load(f)

        instance_id = instance["instance_id"]
        print(f"\n==========================================")
        print(f"BENCHMARKING INSTANCE: {instance_id}")
        print(f"==========================================")

        for algo_name, algo_class, config in algorithms_config:
            print(f"Running {algo_name}...")
            
            # Constructive heuristics are deterministic, we only need to run once, but we replicate for seed logging simplicity
            is_deterministic = algo_name in ["Nearest Neighbor", "Clarke-Wright"]
            
            for seed in tqdm(range(1, num_seeds + 1), desc=f"{algo_name} seeds"):
                # Fix seed for reproducibility
                np.random.seed(seed)
                
                # Instantiate and solve
                # GA needs its creator settings, which are handled inside.
                # If deterministic, we can reuse seed 1 result or run normally
                if is_deterministic and seed > 1:
                    # Duplicate seed 1 results to keep format consistent
                    prev_res = [r for r in raw_results if r["instance_id"] == instance_id and r["algorithm"] == algo_name and r["seed"] == 1][0]
                    raw_results.append({
                        "instance_id": instance_id,
                        "algorithm": algo_name,
                        "seed": seed,
                        "total_cost": prev_res["total_cost"],
                        "runtime_seconds": prev_res["runtime_seconds"],
                        "converged_iteration": prev_res["converged_iteration"]
                    })
                    history_key = f"{instance_id}_{algo_name}_{seed}"
                    convergence_histories[history_key] = convergence_histories[f"{instance_id}_{algo_name}_1"]
                    continue
                
                # Run the algorithm solver
                try:
                    solver = algo_class(instance, config)
                    result = solver.solve()
                    
                    # Compute iteration at which the minimum cost was first achieved
                    min_cost = min(result.convergence_history)
                    converged_iter = result.convergence_history.index(min_cost)
                    
                    raw_results.append({
                        "instance_id": instance_id,
                        "algorithm": algo_name,
                        "seed": seed,
                        "total_cost": result.total_cost,
                        "runtime_seconds": result.runtime_seconds,
                        "converged_iteration": converged_iter
                    })
                    
                    # Store convergence history
                    history_key = f"{instance_id}_{algo_name}_{seed}"
                    convergence_histories[history_key] = [float(c) for c in result.convergence_history]
                    
                except Exception as e:
                    print(f"\nError running {algo_name} on seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()

    # Save raw results to CSV
    df_raw = pd.DataFrame(raw_results)
    csv_path = os.path.join(results_dir, "results_raw.csv")
    df_raw.to_csv(csv_path, index=False)
    print(f"\nSaved raw results to '{csv_path}'.")

    # Save convergence histories to JSON
    json_path = os.path.join(results_dir, "convergence_histories.json")
    with open(json_path, "w") as f:
        json.dump(convergence_histories, f, indent=4)
    print(f"Saved convergence histories to '{json_path}'.")

if __name__ == "__main__":
    run_benchmarks()
