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

## 2. Viewing the Results Dashboard

All benchmark results, live Folium maps, 2×4 comparison grids, and ablation tables are compiled into a single presentation-ready dashboard:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/index.html` in any web browser. The dashboard requires the `results/` directory populated (already committed and verified — no need to re-run the pipeline to view it).

---

## 3. Installation & Quickstart

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

## 4. Implemented Algorithms

1. **Nearest Neighbor**: Simple greedy constructive heuristic.
2. **Clarke-Wright Savings**: Classic greedy route-merging heuristic based on savings estimations.
3. **Standard PSO**: Classic Particle Swarm Optimization using Random-Key mapping (positions in `[0,1]` sorted to derive permutation, split into vehicle routes greedily).
4. **QPSO (Baseline)**: Quantum-behaved PSO utilizing a delta potential well position update rule for exploration. Fits into the same random-key decoder for baseline comparison.
5. **Genetic Algorithm**: Permutation representation initialized via DEAP, utilizing Order Crossover (OX1), Swap Mutation, Tournament Selection, and final 2-opt local search route refinement.
6. **Max-Min Ant System (MMAS)**: Ant Colony Optimization where pheromone values are restricted between dynamically updated bounds $[\tau_{min}, \tau_{max}]$, and only the best ant deposits pheromones.
7. **QPSO-Optimized**: Production-grade memetic QPSO architecture combining Prins Split dynamic programming, fleet bounding, FFD bin-packing seeding, Lamarckian writeback, and inter-route 2-opt* + intra-route Or-opt local search.

---

## 5. QPSO-Optimized & Optimization Study (`src/qpso_lab/`)

While baseline QPSO used naive greedy tour splitting that inflated route counts, **`QPSOOptimized`** (`src/algorithms/qpso_optimized.py`) establishes a mathematically rigorous solver:
- **Optimal Route Decoding**: Prins Split algorithm finding the shortest path through an auxiliary DAG in $O(n \cdot B)$ time.
- **Fleet-Bounded Split**: Multi-layer dynamic programming enforcing exact vehicle limits $K \le K_{max}$.
- **FFD Seed Particle**: Uses a First-Fit-Decreasing bin-packing seed to guarantee the swarm explores the true minimum vehicle count on every run.
- **Hybrid Local Search**: Intra-route Or-opt string relocation combined with inter-route 2-opt\* segment exchange.
- **Lamarckian Write-back**: Inverts improved customer orders back into continuous quantum particle coordinates.

### Verified Benchmark Standing
- **$N=20$:** **Tied for #1** (58,381.85 m, matches OR-Tools reference optimum).
- **$N=50$:** **#2** (187,720.08 m, matches minimum fleet of 10 vehicles).
- **$N=100$:** **#1** (225,422.01 m, 17 vehicles — outperforming ACO MMAS's 248,214.48 m by 22.8 km and sitting within **1.19%** of the OR-Tools MIP reference).

### Running the QPSO Lab Study
```bash
# Run 10-seed comparison across all algorithms with unified scoring
python src/qpso_lab/study.py compare --seeds 10

# Run 6-rung ablation study (V1 Prins -> V6 Inter-route 2-opt*)
python src/qpso_lab/study.py ablation --seeds 10

# Generate study report and plots
python src/qpso_lab/study.py report

# Run independent OR-Tools MIP reference
python src/qpso_lab/study.py reference
```
Detailed findings are documented in [`QPSO_OPTIMIZATION_REPORT.md`](file:///c:/Users/Ruhaan%20Kakar/Desktop/Yung_Blud_SIH/QPSO_OPTIMIZATION_REPORT.md).

---

## 6. Route Visualization Suite (`src/visualization/`)

The repository includes a comprehensive visualization suite rendering real road-following routes on OpenStreetMap:

### 1. Static Road Network Maps (OSMnx)
Renders high-resolution vector PNG maps tracing exact road network geometry:
```bash
python src/visualization/static_map.py --instance instance_n20 --algorithm "QPSO-Optimized"
python src/visualization/static_map.py --instance instance_n50 --algorithm "QPSO-Optimized"
python src/visualization/static_map.py --instance instance_n100 --algorithm "QPSO-Optimized"
```
Outputs: `results/route_maps/static/instance_{n20,n50,n100}_{algo_slug}.png`

### 2. Interactive Folium Maps (HTML / Leaflet)
Generates interactive, zoomable HTML maps with customer demand markers, depot star, vehicle layer controls, and turn-by-turn route paths:
```bash
python src/visualization/interactive_map.py --instance instance_n20 --algorithm "QPSO-Optimized"
python src/visualization/interactive_map.py --instance instance_n50 --algorithm "QPSO-Optimized"
python src/visualization/interactive_map.py --instance instance_n100 --algorithm "QPSO-Optimized"
```
Outputs: `results/route_maps/interactive/instance_{n20,n50,n100}_{algo_slug}.html`

### 3. Side-by-Side Comparison Grids
Generates a $2 \times 4$ Matplotlib subplot figure comparing all 7 algorithms under identical road networks, demand colormaps, vehicle legends, and optimality gaps:
```bash
# Generate for a single instance
python src/visualization/comparison_grid.py --instance instance_n50

# Batch generate for all instances (n20, n50, n100)
python src/visualization/comparison_grid.py --all
```
Outputs: `results/route_maps/comparison_grids/comparison_grid_{instance}.png`

---

## 7. Testing & Validation

Run the complete test suite:
```bash
# Baseline sanity check (algorithms, visualizers, data pipeline)
python tests/sanity_check.py

# Comprehensive QPSO-Optimized unit & ablation tests (20 tests)
python tests/test_qpso_optimized.py
```
