import os
import pandas as pd
import numpy as np

def calculate_benchmark_metrics():
    results_dir = os.path.join("results", "logs")
    raw_csv_path = os.path.join(results_dir, "results_raw.csv")
    
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Missing raw results: '{raw_csv_path}'. Run runner.py first.")

    df_raw = pd.read_csv(raw_csv_path)

    # 1. Group by instance and algorithm to compute aggregates
    grouped = df_raw.groupby(["instance_id", "algorithm"])
    
    summary_records = []
    
    # Calculate best found cost per instance across all runs to compute gaps
    best_costs = df_raw.groupby("instance_id")["total_cost"].min().to_dict()

    for (instance_id, algorithm), group in grouped:
        mean_cost = group["total_cost"].mean()
        std_cost = group["total_cost"].std()
        best_cost_in_group = group["total_cost"].min()
        mean_runtime = group["runtime_seconds"].mean()
        mean_conv_iter = group["converged_iteration"].mean()
        
        # Calculate optimality gap based on mean cost vs overall best-known cost for this instance
        best_known = best_costs[instance_id]
        if best_known == 0:
            gap = 0.0
        else:
            gap = ((mean_cost - best_known) / best_known) * 100.0

        summary_records.append({
            "instance_id": instance_id,
            "algorithm": algorithm,
            "mean_cost": float(mean_cost),
            "std_cost": float(std_cost) if not pd.isna(std_cost) else 0.0,
            "best_cost": float(best_cost_in_group),
            "mean_runtime_sec": float(mean_runtime),
            "mean_conv_iter": float(mean_conv_iter),
            "optimality_gap_percent": float(gap)
        })

    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(results_dir, "results_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Calculated aggregated metrics and saved to '{summary_path}'.")
    return df_summary

if __name__ == "__main__":
    calculate_benchmark_metrics()
