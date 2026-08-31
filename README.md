# QPSO VRP Benchmarking Suite

This repository implements a self-contained, reproducible benchmark harness to compare 5 Vehicle Routing Problem (VRP) baseline algorithms against a Quantum-behaved Particle Swarm Optimization (QPSO) implementation. 

The evaluation runs on real Indian road networks (OSM) fetched for **Kharagpur, West Bengal**, combined with real delivery logistic details (Kaggle) and postal depots (India Post PIN directory).

---

## 1. Repository Structure

```
SIH/
├── data/
│   ├── raw/
│   │   ├── osm/                   # Cached .graphml road networks
│   │   ├── kaggle_delivery.csv    # Delivery records (Kaggle)
│   │   └── india_pincode.csv      # Indian Pincode directory (GitHub mirror)
│   ├── processed/
│   │   ├── distance_matrix.npy    # Computed shortest path distances
│   │   ├── travel_time_matrix.npy # Computed shortest path travel times
│   │   ├── demand_vector.csv      # Customer nodes mapped to package demands
│   │   └── depot_nodes.csv        # Snapped pincode depots
│   └── instances/
│       ├── instance_n20.json      # VRP Instance: N=20 customers
│       ├── instance_n50.json      # VRP Instance: N=50 customers
│       └── instance_n100.json     # VRP Instance: N=100 customers
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_osm.py           # Road graph download, SCC extraction & Dijkstra
│   │   ├── fetch_demand.py        # Kaggle CSV mapping & node association
│   │   ├── fetch_depots.py        # Snap PIN locations to road nodes
│   │   └── build_instance.py      # Compose problem JSON configurations
│   ├── algorithms/
│   │   ├── base.py                # Abstract ABC & SolutionResult wrapper
│   │   ├── nn_clarke_wright.py    # NN + Clarke-Wright Savings
│   │   ├── standard_pso.py        # Standard PSO (Continuous random key)
│   │   ├── qpso.py                # Quantum PSO VRP Optimizer (Ablation)
│   │   ├── genetic_algorithm.py   # Permutation GA via DEAP + 2-opt refinement
│   │   └── aco_mmas.py            # Max-Min Ant System (MMAS) VRP
│   ├── benchmark/
│   │   ├── runner.py              # Executes runs over 10 seeds
│   │   ├── metrics.py             # Computes gaps, means, stds, and statistics
│   │   └── report.py              # Generates Seaborn plots and comparison report
│   └── utils/
│       └── graph_utils.py         # Routing distance and map lookups
├── results/
│   ├── logs/                      # Raw execution CSV and history JSON
│   ├── plots/                     # Output comparison and convergence plots
│   └── comparison_report.md       # Auto-generated markdown report summary
├── tests/
│   └── sanity_check.py            # Unit-like tests validating routing correctness
├── requirements.txt               # Pin dependencies
└── README.md                      # This documentation file
```

---

## 2. Installation & Quickstart

### Prerequisites
Make sure Python 3.10+ is installed on your system.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Data & Benchmarking Pipeline

The suite is fully automated. You can run individual stages or execute the full workflow sequence.

#### 1. Fetch & Build Data (Pre-processing)
Before running benchmarks, you need to prepare the road network and instances:
```bash
# 1. Fetch and process OSM road network
python src/data_pipeline/fetch_osm.py

# 2. Extract and link Kaggle delivery demands
python src/data_pipeline/fetch_demand.py

# 3. Download and snap Indian post office depots
python src/data_pipeline/fetch_depots.py

# 4. Generate the N=20, N=50, N=100 problem instances
python src/data_pipeline/build_instance.py
```

#### 2. Run Sanity Check Tests
Verify that all algorithms compile, execute, and return valid VRP solutions (respecting vehicle capacities and visiting all customers exactly once):
```bash
python tests/sanity_check.py
```

#### 3. Execute Full Benchmark Harness
Runs all 5 algorithms + QPSO on all generated instances over 10 different random seeds:
```bash
python src/benchmark/runner.py
```

#### 4. Compile Metrics & Generate Visual Report
Aggregates logs, calculates gaps, creates convergence/scalability curves, and compiles the final report:
```bash
# Compute summary metrics
python src/benchmark/metrics.py

# Generate comparison plots and comparison_report.md
python src/benchmark/report.py
```

All figures will be saved in `results/plots/` and the final Markdown table will be generated at `results/comparison_report.md`.

---

## 3. Implemented Algorithms

1. **Nearest Neighbor**: Simple greedy constructive heuristic.
2. **Clarke-Wright Savings**: Classic greedy route-merging heuristic based on savings estimations.
3. **Standard PSO**: Classic Particle Swarm Optimization using Random-Key mapping (positions in `[0,1]` sorted to derive permutation, split into vehicle routes greedily).
4. **QPSO**: Quantum-behaved PSO utilizing a delta potential well position update rule for enhanced exploration capabilities. Fits directly into the same random-key decoder for a valid ablation study.
5. **Genetic Algorithm**: Permutation representation initialized via DEAP, utilizing Order Crossover (OX1), Swap Mutation, Tournament Selection, and final 2-opt local search route refinement.
6. **Max-Min Ant System (MMAS)**: Ant Colony Optimization where pheromone values are restricted between dynamically updated bounds $[\tau_{min}, \tau_{max}]$, and only the best ant deposits pheromones.
