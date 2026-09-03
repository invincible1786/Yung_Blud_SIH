# Implementation Details & System Architecture: QPSO-VRP Benchmarking Suite

This document provides a comprehensive technical overview of everything implemented in the **QPSO-VRP Benchmarking Suite** as of today. 

## 1. Project Overview & Problem Statement

The goal of this project is to benchmark a **Quantum-behaved Particle Swarm Optimization (QPSO)** algorithm against established heuristic and metaheuristic baselines for the **Capacitated Vehicle Routing Problem (CVRP)** under realistic operating conditions.

### Core Distinctions
- **Real-world Road Topology**: Instead of Euclidean 2D distance approximations, actual road networks from OpenStreetMap (OSM) for **Kharagpur, West Bengal, India** are utilized. Edge travel times and one-way constraints are modeled using Dijkstra all-pairs shortest path matrices.
- **Real Delivery Demand Distribution**: Real package weights from Kaggle delivery logistics data are converted to demand payloads.
- **Authentic Postal Depots**: Depot locations are derived from the India Post PIN code directory and snapped to the nearest road network junctions.
- **Fair Ablation Benchmark**: QPSO and Standard PSO share the exact same Random-Key priority decoder, representation, and evaluation metrics to ensure a mathematically rigorous comparison.

---

## 2. Implemented Repository Architecture

```
SIH/
├── data/
│   ├── raw/
│   │   ├── osm/                          # Cached OpenStreetMap road networks (.graphml)
│   │   ├── kaggle_delivery.csv           # Kaggle delivery logistics dataset
│   │   └── india_pincode.csv             # Indian Postal PIN directory mirror
│   ├── processed/
│   │   ├── distance_matrix.npy           # (N x N) All-pairs shortest path distance matrix (meters)
│   │   ├── travel_time_matrix.npy        # (N x N) All-pairs shortest travel time matrix (seconds)
│   │   ├── demand_vector.csv             # Cleaned customer demands mapped to graph nodes
│   │   ├── depot_nodes.csv               # Snapped India Post depot locations
│   │   ├── node_id_map.json              # OSM Node ID to matrix index mapping
│   │   └── nodes_metadata.json           # Geographic (lat, lon) coordinates of all road junctions
│   └── instances/
│       ├── instance_n20.json             # VRP benchmark instance: N=20 customers, 5 vehicles
│       ├── instance_n50.json             # VRP benchmark instance: N=50 customers, 8 vehicles
│       └── instance_n100.json            # VRP benchmark instance: N=100 customers, 15 vehicles
├── src/
│   ├── data_pipeline/
│   │   ├── fetch_osm.py                  # Downloads OSM drivable graph, extracts SCC, computes Dijkstra
│   │   ├── fetch_demand.py               # Processes Kaggle logistics data and maps to nodes
│   │   ├── fetch_depots.py               # Filters Kharagpur PIN codes and snaps depots via OSMnx
│   │   └── build_instance.py             # Generates JSON instances with depot, customers & constraints
│   ├── algorithms/
│   │   ├── base.py                       # Abstract base class `RoutingAlgorithm` & `SolutionResult`
│   │   ├── nn_clarke_wright.py           # Nearest Neighbor & Clarke-Wright Savings constructive heuristics
│   │   ├── standard_pso.py               # Standard PSO with continuous Random-Key permutation decoder
│   │   ├── qpso.py                       # Quantum-behaved PSO (Delta-potential well update rule)
│   │   ├── genetic_algorithm.py          # DEAP-based Permutation GA with Order Crossover & 2-opt refinement
│   │   └── aco_mmas.py                   # Max-Min Ant System (MMAS) with clamped pheromone bounds
│   ├── benchmark/
│   │   ├── runner.py                     # Executes 10 random seeds per algorithm across all instances
│   │   ├── metrics.py                    # Aggregates mean, std, runtime, convergence, and optimality gaps
│   │   └── report.py                     # Generates Seaborn/Matplotlib comparison plots and markdown report
│   └── utils/
│       └── graph_utils.py                # Matrix lookup utilities and route distance calculators
├── results/
│   ├── logs/
│   │   ├── results_raw.csv               # Per-seed raw execution logs for all runs
│   │   ├── results_summary.csv           # Statistical aggregations across seeds
│   │   └── convergence_histories.json    # Iteration-by-iteration cost convergence trajectories
│   ├── plots/
│   │   ├── convergence_instance_n20.png  # Convergence curve for N=20
│   │   ├── convergence_instance_n50.png  # Convergence curve for N=50
│   │   ├── convergence_instance_n100.png # Convergence curve for N=100
│   │   ├── cost_comparison_*.png         # Bar charts of routing costs
│   │   ├── runtime_comparison_*.png      # Bar charts of algorithm runtimes
│   │   ├── scalability_cost.png          # Scalability line plot (Cost vs N)
│   │   └── scalability_runtime.png       # Scalability line plot (Runtime vs N)
│   └── comparison_report.md              # Auto-generated markdown benchmark summary
├── tests/
│   └── sanity_check.py                   # Unit tests validating capacity & single-visit correctness
├── osm.py                                # Standalone OSM network fetcher and visualizer
├── requirements.txt                      # Project dependency specification
└── README.md                             # Repository introduction and execution guide
```

---

## 3. Data Pipeline Implementation

The data pipeline converts raw geographic and logistical data into validated VRP problem instances:

### 1. Road Network Extraction (`src/data_pipeline/fetch_osm.py`)
- **OSMnx Integration**: Fetches drivable street network within a 3,000-meter radius of Kharagpur (`network_type="drive"`).
- **Graph Caching**: Saves the network in GraphML format (`data/raw/osm/kharagpur_west_bengal_india.graphml`) to prevent repeated remote queries.
- **Strongly Connected Component (SCC)**: Drops isolated subgraphs and cul-de-sacs without exits by extracting the largest SCC via `networkx.strongly_connected_components`.
- **Dijkstra All-Pairs Shortest Path**: Runs single-source Dijkstra sweeps across all SCC nodes to construct:
  - `distance_matrix.npy`: Physical road distances in meters.
  - `travel_time_matrix.npy`: Free-flow traversal times in seconds (fallback speed limit of 30 km/h).
- **Coordinate Lookup**: Saves `nodes_metadata.json` with latitude/longitude for every intersection.

### 2. Delivery Demand Association (`src/data_pipeline/fetch_demand.py`)
- Ingests delivery logistics dataset (`data/raw/kaggle_delivery.csv`).
- Converts `package_weight_kg` into integer demands ($\text{ceil}(\text{weight})$ clipped at minimum 1 kg).
- Maps sampled delivery orders uniformly onto unique non-depot SCC road nodes, writing `data/processed/demand_vector.csv`.

### 3. Depot Snapping (`src/data_pipeline/fetch_depots.py`)
- Fetches Indian Postal PIN code directory containing postal branch geolocations.
- Filters for postal stations within **Paschim Medinipur / Kharagpur** (e.g., Kharagpur Technology B.O, Kharagpur Town S.O, Nimpura S.O).
- Uses `ox.nearest_nodes` for vectorized Euclidean-to-network projection, snapping depots to valid SCC junction IDs (`data/processed/depot_nodes.csv`).

### 4. Instance Composition (`src/data_pipeline/build_instance.py`)
Composes reproducible JSON problem definitions for three scale benchmarks:
- **`instance_n20.json`**: $N=20$ customer nodes, 5 vehicles, capacity = 100 kg.
- **`instance_n50.json`**: $N=50$ customer nodes, 8 vehicles, capacity = 120 kg.
- **`instance_n100.json`**: $N=100$ customer nodes, 15 vehicles, capacity = 150 kg.
- Each instance links to the precomputed distance matrix and node ID index.

---

## 4. Optimization Algorithms Implemented

All algorithms inherit from the base class `RoutingAlgorithm` in `src/algorithms/base.py` and return a standardized `SolutionResult(routes, total_cost, convergence_history, runtime_seconds)`.

### 1. Nearest Neighbor Heuristic (`src/algorithms/nn_clarke_wright.py`)
- **Type**: Greedy constructive heuristic.
- **Mechanism**: Iteratively selects the closest unvisited customer that satisfies the remaining vehicle capacity. When no eligible customer fits, the vehicle returns to the depot, and a new vehicle route begins.
- **Runtime**: Extremely fast ($\approx 1\text{ ms}$), providing an immediate baseline.

### 2. Clarke-Wright Savings Algorithm (`src/algorithms/nn_clarke_wright.py`)
- **Type**: Route-merging constructive heuristic.
- **Mechanism**: 
  - Initializes with $N$ individual back-and-forth routes: $[\text{depot}, i, \text{depot}]$.
  - Computes savings for every customer pair $(i, j)$:
    $$S_{ij} = d(0, i) + d(0, j) - d(i, j)$$
  - Sorts savings in descending order and greedily merges routes at endpoints if vehicle capacity permits.

### 3. Standard Particle Swarm Optimization (PSO) (`src/algorithms/standard_pso.py`)
- **Representation**: Continuous Random-Key vector $\mathbf{x}_i \in [0, 1]^N$.
- **Decoder**: Sorting particle coordinate values produces a customer visiting permutation. A greedy splitter sections the permutation into vehicle routes whenever capacity is exceeded.
- **Dynamics**: Velocity and position updates using inertia weight $w = 0.7$ and cognitive/social coefficients $c_1 = 1.5, c_2 = 1.5$:
  $$\mathbf{v}_i^{(t+1)} = w \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i^{(t)})$$
  $$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$
- **Constraint Handling**: Imposes a heavy penalty ($10^6 \times \Delta_{\text{vehicles}}$) when generated routes exceed available vehicle fleet size.

### 4. Quantum-behaved Particle Swarm Optimization (QPSO) (`src/algorithms/qpso.py`)
- **Theoretical Basis**: In quantum space, particles move according to wave-packet probability distributions rather than Newtonian velocity trajectories, governed by a delta potential well.
- **Mean Best ($m_{\text{best}}$)**: A central attractor computed as the centroid of all personal best positions:
  $$m_{\text{best}} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{pbest}_i$$
- **Local Attractor ($p_i$)**:
  $$p_{i, d} = \phi_d \cdot \mathbf{pbest}_{i, d} + (1 - \phi_d) \cdot \mathbf{gbest}_d, \quad \phi_d \sim U(0, 1)$$
- **Quantum Position Update**:
  $$x_{i, d}^{(t+1)} = p_{i, d} \pm \beta \left| m_{\text{best}, d} - x_{i, d}^{(t)} \right| \ln\left(\frac{1}{u_d}\right), \quad u_d \sim U(0, 1)$$
- **Contraction-Expansion ($\beta$)**: Linearly annealed from $\beta_{\text{start}} = 1.0$ (global exploration) to $\beta_{\text{end}} = 0.5$ (local exploitation).
- **Ablation Validity**: Shares the identical random-key permutation decoder and penalty function as Standard PSO.

### 5. Genetic Algorithm (GA) (`src/algorithms/genetic_algorithm.py`)
- **Library**: Implemented using DEAP (`Distributed Evolutionary Algorithms in Python`).
- **Chromosome**: Direct permutation of customer indices $[0, 1, \dots, N-1]$.
- **Operators**:
  - **Order Crossover (OX1)**: Preserves relative sequence order without producing duplicate customer visits.
  - **Shuffle Mutation**: Randomly transposes customer indices ($p_m = 0.2$).
  - **Tournament Selection**: Tournament size $k = 3$.
- **Local Search Refinement**: Applies a 2-opt edge-exchange post-processor on the best chromosome routes to untangle crossover paths.

### 6. Max-Min Ant System (MMAS) (`src/algorithms/aco_mmas.py`)
- **Representation**: Pheromone matrix $\tau_{ij}$ defined over all network nodes.
- **Transition Probability**: Ant chooses next customer $j$ from unvisited set with probability:
  $$P_{ij} = \frac{\tau_{ij}^\alpha \cdot \eta_{ij}^\beta}{\sum_{k} \tau_{ik}^\alpha \cdot \eta_{ik}^\beta}$$
  where $\eta_{ij} = 1 / d_{ij}$ is visibility, $\alpha = 1.0$, and $\beta = 3.0$.
- **MMAS Bounding**: Pheromone levels are strictly clamped to $[\tau_{\min}, \tau_{\max}]$:
  $$\tau_{\max} = \frac{1}{\rho \cdot C_{\text{best}}}, \quad \tau_{\min} = \frac{\tau_{\max}}{2N}$$
- **Pheromone Deposit**: Only the iteration or global best ant deposits pheromones, accelerating convergence while bounds prevent stagnation.

---

## 5. Benchmarking & Evaluation Suite

### 1. Harness Execution (`src/benchmark/runner.py`)
- Runs all 6 algorithm configurations across all 3 benchmark instances.
- Replicates stochastic algorithms over **10 random seeds** (seeds 1 to 10) for statistical significance.
- Records per-run execution time, total routing cost, and exact iteration of best convergence.
- Dumps data to `results/logs/results_raw.csv` and iteration curves to `results/logs/convergence_histories.json`.

### 2. Statistical Aggregator (`src/benchmark/metrics.py`)
Calculates summary statistics across seeds:
- **Mean Cost** and **Standard Deviation** of routing distance.
- **Mean Solving Runtime** (seconds).
- **Mean Convergence Iteration**.
- **Optimality Gap (%)**: Percentage deviation from the best-known solution for that instance:
  $$\text{Gap} = \frac{\text{Mean Cost} - \text{Best Cost}}{\text{Best Cost}} \times 100\%$$

### 3. Visualization & Reporting (`src/benchmark/report.py`)
Generates publication-quality charts using Matplotlib and Seaborn, saved to `results/plots/`:
- **Convergence Curves** (`convergence_instance_n*.png`): Log-scale iteration vs. cost progression.
- **Cost Comparisons** (`cost_comparison_instance_n*.png`): Bar charts contrasting algorithm solution quality.
- **Runtime Comparisons** (`runtime_comparison_instance_n*.png`): Log-scale execution speed comparison.
- **Scalability Curves** (`scalability_cost.png` and `scalability_runtime.png`): Cost and runtime trajectories as $N$ scales from 20 to 100.
- **Markdown Report** (`results/comparison_report.md`): Formatted table and embedded plots summarizing results.

---

## 6. Current Benchmark Results Summary

The table below reflects the actual benchmark runs stored in `results/logs/results_summary.csv`:

| Instance ID | Algorithm | Mean Cost (m) | Std Cost | Mean Runtime (s) | Conv. Iteration | Optimality Gap (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **instance_n20** | **Ant Colony (MMAS)** | **58,709.33** | 911.46 | 4.9176 | 73.6 | **0.56% (Best)** |
| instance_n20 | Clarke-Wright | 64,161.05 | 0.00 | 0.0065 | 0.0 | 9.90% |
| instance_n20 | Genetic Algorithm | 69,483.51 | 1,667.63 | 0.7390 | 99.0 | 19.02% |
| instance_n20 | Standard PSO | 70,822.61 | 2,515.94 | 0.2445 | 72.9 | 21.31% |
| instance_n20 | QPSO | 71,599.49 | 3,984.84 | 0.7928 | 89.9 | 22.64% |
| instance_n20 | Nearest Neighbor | 85,440.06 | 0.00 | 0.0010 | 0.0 | 46.35% |
| **instance_n50** | **Genetic Algorithm** | **190,606.71** | 4,643.21 | 1.4967 | 99.0 | **3.49% (Best)** |
| instance_n50 | Ant Colony (MMAS) | 2,169,348.50 | 8,993.02 | 10.6776 | 83.2 | 1077.85% |
| instance_n50 | Clarke-Wright | 3,149,027.94 | 0.00 | 0.0085 | 0.0 | 1609.77% |
| instance_n50 | Standard PSO | 3,214,001.06 | 8,066.93 | 0.2894 | 86.9 | 1645.05% |
| instance_n50 | QPSO | 3,225,133.37 | 3,916.95 | 1.1526 | 52.5 | 1651.10% |
| instance_n50 | Nearest Neighbor | 3,184,609.02 | 0.00 | 0.0011 | 0.0 | 1629.09% |
| **instance_n100**| **Genetic Algorithm** | **342,041.04** | 14,392.33 | 3.2367 | 99.0 | **7.18% (Best)** |
| instance_n100 | Ant Colony (MMAS) | 2,259,572.86 | 5,313.06 | 36.7424 | 54.3 | 608.07% |
| instance_n100 | Nearest Neighbor | 2,266,701.90 | 0.00 | 0.0020 | 0.0 | 610.30% |
| instance_n100 | Clarke-Wright | 3,228,951.41 | 0.00 | 0.0780 | 0.0 | 911.83% |
| instance_n100 | Standard PSO | 3,429,316.32 | 11,125.50 | 1.4394 | 87.3 | 974.62% |
| instance_n100 | QPSO | 3,443,906.69 | 7,260.20 | 1.8511 | 51.3 | 979.19% |

*(Note: In larger instances $N \ge 50$, standard PSO, QPSO, and constructive heuristics incur vehicle capacity penalty costs ($10^6$ per excess vehicle needed), whereas GA with 2-opt successfully packs routes to satisfy vehicle fleet limits).*

---

## 7. Testing & Verification

- **Correctness Suite (`tests/sanity_check.py`)**:
  - Confirms every route starts and ends at depot `depot_node_id`.
  - Confirms customer sets are partitioned with zero omissions and zero duplicates.
  - Validates route load $\le \text{vehicle capacity}$.
  - Confirms execution runtime and convergence histories are recorded properly.

---

## 8. Potential Next Enhancements

1. **VRP Route Network Visualizer**: Plotting calculated vehicle routes (lines with arrows and distinct colors per vehicle) over the OSM road map or generating an interactive HTML Folium map.
2. **QPSO Decoder Improvements**: Integrating Split-algorithm (Prins' optimal split) or 2-opt local search into the QPSO decoder to dramatically improve performance on larger instances ($N \ge 50$).
3. **Time Window Support (VRPTW)**: Extending `travel_time_matrix.npy` to test customer delivery time windows.
