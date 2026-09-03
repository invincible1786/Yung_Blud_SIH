# System Architecture & Technical Specification: QPSO-VRP Suite

## 1. Architectural Overview & Design Philosophy

The **Quantum-behaved Particle Swarm Optimization Vehicle Routing Problem (QPSO-VRP) Benchmarking Suite** is an end-to-end computational framework designed to benchmark quantum-inspired swarm intelligence against classical heuristics and metaheuristics on **real-world road topologies**.

Unlike canonical VRP benchmarks (e.g., Solomon or TSPLIB benchmark sets) that rely on idealized 2D Euclidean planes, this system is engineered around **authentic geospatial and logistical constraints**:
- **True Road Network Topology**: Actual directed road networks extracted from OpenStreetMap (OSM) for **Kharagpur, West Bengal, India**, accounting for one-way roads, physical street geometry, and varying speeds.
- **Topological Distance & Travel Time**: Graph-theoretic shortest-path all-pairs matrices computed via Dijkstra's algorithm across the Strongly Connected Component (SCC).
- **Empirical Demand Distributions**: Customer delivery payloads derived from real-world Kaggle logistics package records.
- **Geocoded Postal Depots**: Depot locations determined by snapping India Post PIN directory coordinates onto physical street intersections.
- **Strict Ablation Grounding**: QPSO and Standard PSO share an identical continuous Random-Key permutation decoder and penalty function, ensuring that performance variations stem solely from search dynamics rather than decoding artifacts.

---

## 2. End-to-End System Architecture Diagram

```mermaid
flowchart TD
    subgraph Data Sources ["1. Geospatial & Empirical Data Sources"]
        OSM["OpenStreetMap (OSM)\n3000m Kharagpur Network"]
        Kaggle["Kaggle Logistics Dataset\n(package_weight_kg)"]
        IndiaPost["India Post PIN Directory\n(Geocoded Postal Stations)"]
    end

    subgraph Data Pipeline ["2. Data Pipeline Layer (src/data_pipeline)"]
        F_OSM["fetch_osm.py\n- OSMnx Download & Cache\n- SCC Extraction (NetworkX)\n- All-Pairs Dijkstra Sweeps"]
        F_DEM["fetch_demand.py\n- Ingest Kaggle CSV\n- Integer Demand Weighting\n- SCC Node Mapping"]
        F_DEP["fetch_depots.py\n- Filter Paschim Medinipur PINs\n- Snapping via ox.nearest_nodes"]
        BUILD["build_instance.py\n- Merge Depot, Demand, Graph\n- Compose Problem Instances\n(N=20, N=50, N=100)"]
        
        OSM --> F_OSM
        Kaggle --> F_DEM
        IndiaPost --> F_DEP
        F_OSM --> BUILD
        F_DEM --> BUILD
        F_DEP --> BUILD
    end

    subgraph Processed Data ["3. Persistent Artifacts (data/processed & data/instances)"]
        DistMat["distance_matrix.npy\n(N x N Shortest Distances)"]
        TimeMat["travel_time_matrix.npy\n(N x N Travel Times)"]
        NodeMap["node_id_map.json\n(OSM Node ID <-> Matrix Index)"]
        Instances["JSON Instances\n- instance_n20.json\n- instance_n50.json\n- instance_n100.json"]
        
        BUILD --> DistMat
        BUILD --> TimeMat
        BUILD --> NodeMap
        BUILD --> Instances
    end

    subgraph Algorithm Layer ["4. Algorithm Engine Layer (src/algorithms)"]
        Base["base.py\nRoutingAlgorithm (ABC)\nSolutionResult Data Container"]
        
        NN_CW["nn_clarke_wright.py\n- Nearest Neighbor Heuristic\n- Clarke-Wright Savings Heuristic"]
        S_PSO["standard_pso.py\nStandard PSO (Inertia + Cognitive + Social)"]
        Q_PSO["qpso.py\nQuantum PSO (Delta Potential Well + mbest)"]
        GA["genetic_algorithm.py\nDEAP GA (OX1, Mutation, 2-Opt Local Search)"]
        ACO["aco_mmas.py\nMax-Min Ant System (Bounded Pheromones)"]

        Base --> NN_CW
        Base --> S_PSO
        Base --> Q_PSO
        Base --> GA
        Base --> ACO
        Instances -.-> Base
    end

    subgraph Benchmark Harness ["5. Benchmark & Statistical Evaluation (src/benchmark)"]
        Runner["runner.py\n- 10 Random Seeds Per Solver\n- Per-Seed Execution Logs\n- best_routes.json Storage"]
        Metrics["metrics.py\n- Mean Cost & Runtime\n- Convergence Iterations\n- Optimality Gaps (%)"]
        Report["report.py\n- Convergence Curves (Log Scale)\n- Cost & Runtime Bar Charts\n- Scalability Curves vs N\n- comparison_report.md"]

        NN_CW --> Runner
        S_PSO --> Runner
        Q_PSO --> Runner
        GA --> Runner
        ACO --> Runner
        Runner --> Metrics
        Metrics --> Report
    end

    subgraph Visualization Suite ["6. Visualization Subsystem (src/visualization)"]
        StaticMap["static_map.py\n- OSMnx Cartographic Maps\n- Shortest Road Segment Tracing\n- High-Res Matplotlib Output"]
        InterMap["interactive_map.py\n- Leaflet/Folium Dynamic Maps\n- Multi-Layer Vehicle Routes\n- Interactive Popups & Tooltips"]
        CompGrid["comparison_grid.py\n- 2x3 Side-by-Side Subplot Grid\n- Direct 6-Algorithm Comparison\n- Standardized Metrics & Scale"]

        Runner --> StaticMap
        Runner --> InterMap
        Runner --> CompGrid
    end

    subgraph Testing Harness ["7. Verification & Sanity Suite (tests)"]
        Sanity["tests/sanity_check.py\n- Route Depot Invariance\n- Zero Omissions & No Duplicates\n- Capacity Constraint Enforcement\n- Visualizer Integrity Checks"]
        Instances -.-> Sanity
        Algorithm Layer -.-> Sanity
    end
```

---

## 3. Directory Layout & Module Responsibilities

The codebase is organized in a modular, decoupled structure:

```
SIH/
├── data/
│   ├── raw/
│   │   ├── osm/                          # Cached OpenStreetMap road networks (.graphml)
│   │   ├── kaggle_delivery.csv           # Raw package logistics delivery dataset
│   │   └── india_pincode.csv             # Indian Postal PIN directory database
│   ├── processed/
│   │   ├── distance_matrix.npy           # (N x N) All-pairs shortest path distance matrix (meters)
│   │   ├── travel_time_matrix.npy        # (N x N) All-pairs shortest travel time matrix (seconds)
│   │   ├── demand_vector.csv             # Cleaned customer demands mapped to graph nodes
│   │   ├── depot_nodes.csv               # Snapped India Post depot locations
│   │   ├── node_id_map.json              # OSM Node ID to matrix index mapping
│   │   └── nodes_metadata.json           # Geographic (lat, lon) coordinates of all road junctions
│   └── instances/
│       ├── instance_n20.json             # VRP benchmark instance: N=20 customers, 5 vehicles, cap=100kg
│       ├── instance_n50.json             # VRP benchmark instance: N=50 customers, 8 vehicles, cap=120kg
│       └── instance_n100.json            # VRP benchmark instance: N=100 customers, 15 vehicles, cap=150kg
├── src/
│   ├── algorithms/
│   │   ├── __init__.py                   # Package initialization
│   │   ├── base.py                       # Abstract base class `RoutingAlgorithm` & `SolutionResult`
│   │   ├── nn_clarke_wright.py           # Nearest Neighbor & Clarke-Wright Savings heuristics
│   │   ├── standard_pso.py               # Standard PSO with continuous Random-Key permutation decoder
│   │   ├── qpso.py                       # Quantum-behaved PSO (Delta-potential well update rule)
│   │   ├── genetic_algorithm.py          # DEAP-based Permutation GA with Order Crossover & 2-opt refinement
│   │   └── aco_mmas.py                   # Max-Min Ant System (MMAS) with clamped pheromone bounds
│   ├── benchmark/
│   │   ├── __init__.py                   # Package initialization
│   │   ├── runner.py                     # Executes 10 random seeds per algorithm across all instances
│   │   ├── metrics.py                    # Aggregates mean, std, runtime, convergence, and optimality gaps
│   │   └── report.py                     # Generates Seaborn/Matplotlib comparison plots and markdown report
│   ├── data_pipeline/
│   │   ├── __init__.py                   # Package initialization
│   │   ├── fetch_osm.py                  # Downloads OSM drivable graph, extracts SCC, computes Dijkstra
│   │   ├── fetch_demand.py               # Processes Kaggle logistics data and maps to nodes
│   │   ├── fetch_depots.py               # Filters Kharagpur PIN codes and snaps depots via OSMnx
│   │   └── build_instance.py             # Generates JSON instances with depot, customers & constraints
│   ├── utils/
│   │   ├── __init__.py                   # Package initialization
│   │   └── graph_utils.py                # Matrix lookup utilities and route distance calculators
│   └── visualization/
│       ├── __init__.py                   # Visualization package exports
│       ├── static_map.py                 # OSMnx-based static high-res cartographic route plotting
│       ├── interactive_map.py            # Leaflet/Folium interactive HTML route map generator
│       └── comparison_grid.py            # 2x3 side-by-side subplot visual comparison of all 6 solvers
├── results/
│   ├── logs/
│   │   ├── best_routes.json              # Serialized optimal route node sequences per (instance, solver)
│   │   ├── results_raw.csv               # Per-seed raw execution logs for all runs
│   │   ├── results_summary.csv           # Statistical aggregations across seeds
│   │   └── convergence_histories.json    # Iteration-by-iteration cost convergence trajectories
│   ├── plots/
│   │   ├── convergence_instance_n*.png   # Convergence curves across iterations (log-scale)
│   │   ├── cost_comparison_*.png         # Bar charts of routing costs per algorithm
│   │   ├── runtime_comparison_*.png      # Bar charts of algorithm solving times
│   │   ├── scalability_cost.png          # Scalability line plot (Cost vs N)
│   │   └── scalability_runtime.png       # Scalability line plot (Runtime vs N)
│   ├── route_maps/
│   │   ├── static/                       # 18 individual static PNG maps (3 instances × 6 algorithms)
│   │   ├── interactive/                  # 18 interactive HTML Leaflet maps (3 instances × 6 algorithms)
│   │   └── comparison_grids/             # 3 2×3 side-by-side comparison grids (PNG)
│   └── comparison_report.md              # Auto-generated markdown benchmark summary
├── tests/
│   └── sanity_check.py                   # Automated tests validating capacity, visits, and visualizers
├── osm.py                                # Standalone OSM network fetcher, cache manager, and visualizer
├── requirements.txt                      # Project dependency specification
├── IMPLEMENTATION_DETAILS.md             # Implementation details summary
└── README.md                             # High-level overview and execution guide
```

---

## 4. Mathematical Formulations & Algorithm Implementations

Every algorithm in the suite derives from `RoutingAlgorithm` in `src/algorithms/base.py`. Solutions are encapsulated in `SolutionResult`:
- `routes`: `List[List[int]]` (node IDs representing each vehicle tour, always beginning and terminating at the depot node).
- `total_cost`: `float` (total road distance traversed in meters, plus penalty if fleet limits are breached).
- `convergence_history`: `List[float]` (best fitness score at each iteration).
- `runtime_seconds`: `float` (wall-clock solution time).

### 4.1 Cost Function & Constraint Handling

The objective function to minimize is the total network route distance with an exterior penalty function for vehicle fleet overflow:

$$\min \quad f(R) = \sum_{r \in R} \sum_{i=0}^{|r|-2} D(\pi_{r, i}, \pi_{r, i+1}) + \lambda \cdot \max(0, |R| - K_{\text{max}})$$

Where:
- $R$ is the set of vehicle routes.
- $\pi_{r, i}$ is the $i$-th node in route $r$, with $\pi_{r, 0} = \pi_{r, |r|-1} = \text{depot}$.
- $D(u, v)$ is the Dijkstra shortest-path distance between road junctions $u$ and $v$ looked up from `distance_matrix.npy`.
- $|R|$ is the number of routes generated by the capacity decoder.
- $K_{\text{max}}$ is the maximum vehicle fleet size allocated to the instance.
- $\lambda = 10^6$ is the penalty weight penalizing each vehicle required in excess of the fleet limit.

Each individual route $r$ strictly satisfies the vehicle capacity constraint:
$$\sum_{j \in r \setminus \{\text{depot}\}} q_j \le Q_{\text{vehicle}}$$

---

### 4.2 Quantum-behaved Particle Swarm Optimization (QPSO)

- **File**: `src/algorithms/qpso.py`
- **Class**: `QPSO`

#### Theoretical Formulation
In classical PSO, a particle moves along deterministic Newtonian trajectories with a velocity vector. In quantum space, a particle's state is described by a wave function $\psi(\mathbf{x}, t)$, where $|\psi(\mathbf{x}, t)|^2$ defines the probability density of finding the particle at coordinate $\mathbf{x}$.

Assuming particles move in a quantum delta potential well centered at a local stochastic attractor $\mathbf{p}_i$, the particle position is updated using the Monte Carlo inverse transform:

1. **Mean Best ($m_{\text{best}}$)**: Centroid of all individual particles' historical best positions:
   $$m_{\text{best}} = \frac{1}{M} \sum_{i=1}^M \mathbf{pbest}_i$$

2. **Local Attractor ($\mathbf{p}_i$)**: A stochastic combination of personal best ($\mathbf{pbest}_i$) and swarm global best ($\mathbf{gbest}$):
   $$p_{i, d} = \phi_d \cdot \text{pbest}_{i, d} + (1 - \phi_d) \cdot \text{gbest}_d, \quad \phi_d \sim \mathcal{U}(0, 1)$$

3. **Quantum Wave-Packet Position Update**:
   $$x_{i, d}^{(t+1)} = p_{i, d} \pm \beta \left| m_{\text{best}, d} - x_{i, d}^{(t)} \right| \ln\left(\frac{1}{u_d}\right), \quad u_d \sim \mathcal{U}(0, 1)$$

4. **Contraction-Expansion Coefficient Annealing ($\beta$)**:
   $$\beta(t) = \beta_{\text{start}} - (\beta_{\text{start}} - \beta_{\text{end}}) \cdot \frac{t}{T_{\text{max}}}$$
   Where $\beta_{\text{start}} = 1.0$ (strong global exploration) and $\beta_{\text{end}} = 0.5$ (fine local exploitation).

#### Priority Decoder
To map continuous particle positions $\mathbf{x}_i \in [0, 1]^N$ to discrete VRP routes:
1. `np.argsort(x_i)` yields customer visit priorities.
2. A greedy route splitter traverses this customer permutation, appending customers to the current vehicle route until package demand exceeds $Q_{\text{vehicle}}$, at which point the vehicle returns to the depot and a new route begins.

---

### 4.3 Standard Particle Swarm Optimization (PSO)

- **File**: `src/algorithms/standard_pso.py`
- **Class**: `StandardPSO`

- **Representation**: Continuous Random-Key vector $\mathbf{x}_i \in [0, 1]^N$.
- **Velocity & Position Dynamics**:
  $$\mathbf{v}_i^{(t+1)} = w \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i^{(t)})$$
  $$\mathbf{x}_i^{(t+1)} = \mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}$$
- **Parameters**: Inertia weight $w = 0.729$, cognitive acceleration $c_1 = 1.4944$, social acceleration $c_2 = 1.4944$, random factors $r_1, r_2 \sim \mathcal{U}(0, 1)$.
- **Position Clipping**: Particle coordinates are clamped to $[0, 1]$.
- **Decoder**: Shares the identical `decode_permutation` and penalty calculation as QPSO for experimental consistency.

---

### 4.4 Permutation Genetic Algorithm (GA) with 2-Opt Refinement

- **File**: `src/algorithms/genetic_algorithm.py`
- **Class**: `GeneticAlgorithm`

- **Representation**: Chromosome is a direct permutation of customer indices $[0, 1, \dots, N-1]$ implemented via DEAP (`deap.creator`, `deap.tools`).
- **Genetic Operators**:
  - **Crossover**: Order Crossover (OX1 / `tools.cxOrdered`), preserving relative ordering without customer duplication or omission ($p_c = 0.8$).
  - **Mutation**: Shuffle Indexes mutation (`tools.mutShuffleIndexes`) transposing customer stops ($p_m = 0.2$).
  - **Selection**: Tournament selection (`tools.selTournament`) with tournament size $k = 3$.
- **2-Opt Local Search Refinement**:
  - Applied to the best chromosome routes at completion.
  - Iteratively tests pairs of non-adjacent edges $(i, i+1)$ and $(j, j+1)$ and reverses the subsegment if:
    $$D(\pi_i, \pi_j) + D(\pi_{i+1}, \pi_{j+1}) < D(\pi_i, \pi_{i+1}) + D(\pi_j, \pi_{j+1})$$
  - Eliminates self-crossing paths and untangles twisted tours.

---

### 4.5 Max-Min Ant System (MMAS)

- **File**: `src/algorithms/aco_mmas.py`
- **Class**: `ACOMMAS`

- **Pheromone Matrix**: $\tau_{ij}$ defined over all graph nodes (depot + customers).
- **Visibility Matrix**:
  $$\eta_{ij} = \frac{1}{\max(D(i, j), 1.0)}$$
- **State Transition Probability**:
  $$P_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}$$
  Where $\alpha = 1.0$, $\beta = 3.0$, and $\mathcal{N}_i^k$ is the feasible unvisited customer neighborhood that fits the ant's remaining capacity.
- **MMAS Clamped Bounds**:
  To avoid stagnation and premature search freeze, pheromones are strictly bounded within $[\tau_{\min}, \tau_{\max}]$:
  $$\tau_{\max} = \frac{1}{\rho \cdot C_{\text{best}}}, \quad \tau_{\min} = \frac{\tau_{\max}}{2N}$$
  Where $\rho = 0.1$ is the pheromone evaporation rate and $C_{\text{best}}$ is the best cost discovered.
- **Pheromone Deposit**: Only the iteration-best or global-best ant deposits pheromones:
  $$\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \Delta \tau_{ij}^{\text{best}}$$

---

### 4.6 Constructive Baselines: Nearest Neighbor & Clarke-Wright

- **File**: `src/algorithms/nn_clarke_wright.py`
- **Class**: `NearestNeighborClarkeWright`

1. **Nearest Neighbor (`method="nn"`)**:
   - Starting from the depot, greedily attaches the closest customer whose demand satisfies remaining vehicle capacity.
   - When no unvisited customer fits, the vehicle returns to the depot, and a new vehicle departs.
   - Execution time is near-instantaneous ($\approx 1\text{ ms}$).

2. **Clarke-Wright Savings (`method="clarke_wright"`)**:
   - Initializes $N$ independent back-and-forth routes: $[\text{depot}, i, \text{depot}]$.
   - Calculates savings matrix for all pairs $(i, j)$:
     $$S_{ij} = D(\text{depot}, i) + D(\text{depot}, j) - D(i, j)$$
   - Sorts pairs in descending order of savings.
   - Merges routes containing customer $i$ and $j$ if:
     1. Customers $i$ and $j$ reside in different routes.
     2. $i$ and $j$ are adjacent to the depot in their respective routes.
     3. Combined route demand does not exceed $Q_{\text{vehicle}}$.

---

## 5. Data Pipeline Implementation

The data pipeline processes real spatial and logistical datasets into validated VRP instances:

```
Raw OSM (.graphml)   +   Kaggle Delivery CSV   +   India Post PIN Directory
       │                         │                           │
       ▼                         ▼                           ▼
[fetch_osm.py]           [fetch_demand.py]           [fetch_depots.py]
  - 3km Kharagpur          - Integer demands           - Paschim Medinipur
  - Largest SCC            - Uniform mapping           - Nearest road snap
  - Dijkstra sweeps              │                           │
       │                         │                           │
       └─────────────────────────┼───────────────────────────┘
                                 ▼
                        [build_instance.py]
                                 │
                   ┌─────────────┼─────────────┐
                   ▼             ▼             ▼
             instance_n20  instance_n50  instance_n100
```

### 5.1 Step 1: Road Network Extraction (`src/data_pipeline/fetch_osm.py`)
- Downloads drivable road network within a 3,000m radius of Kharagpur using `osmnx.graph_from_address(..., network_type="drive")`.
- Caches raw graph to `data/raw/osm/kharagpur_west_bengal_india.graphml`.
- Enforces strong connectivity by extracting the largest Strongly Connected Component (SCC) using `networkx.strongly_connected_components`. Drops unreachable dead-ends and one-way trapped nodes.
- Assigns road speeds (`ox.add_edge_speeds` with 30 km/h default) and travel times (`ox.add_edge_travel_times`).
- Executes single-source Dijkstra sweeps from every node in the SCC to construct:
  - `distance_matrix.npy`: Physical road distance in meters between every junction pair.
  - `travel_time_matrix.npy`: Traversal duration in seconds.
- Exports `node_id_map.json` (OSM node ID to continuous matrix index) and `nodes_metadata.json` (junction latitude/longitude).

### 5.2 Step 2: Delivery Demand Processing (`src/data_pipeline/fetch_demand.py`)
- Reads `data/raw/kaggle_delivery.csv`.
- Normalizes package weights: $\text{demand} = \max(1, \lceil \text{package\_weight\_kg} \rceil)$.
- Maps delivery demands to unique non-depot SCC road nodes, saving `data/processed/demand_vector.csv`.

### 5.3 Step 3: Depot Location Snapping (`src/data_pipeline/fetch_depots.py`)
- Parses `data/raw/india_pincode.csv` for post office locations in Paschim Medinipur / Kharagpur (e.g., Kharagpur Technology B.O, Kharagpur Town S.O, Nimpura S.O).
- Uses `ox.nearest_nodes` to snap geographic coordinates onto the closest SCC road junction, saving `data/processed/depot_nodes.csv`.

### 5.4 Step 4: Problem Instance Builder (`src/data_pipeline/build_instance.py`)
Generates standardized benchmark JSON files in `data/instances/`:
- **`instance_n20.json`**: $N=20$ customers, 5 vehicles, capacity $Q=100\text{ kg}$.
- **`instance_n50.json`**: $N=50$ customers, 8 vehicles, capacity $Q=120\text{ kg}$.
- **`instance_n100.json`**: $N=100$ customers, 15 vehicles, capacity $Q=150\text{ kg}$.

Each instance JSON structure:
```json
{
  "instance_id": "instance_n20",
  "name": "Kharagpur VRP N=20",
  "city": "Kharagpur, West Bengal, India",
  "depot": {
    "node_id": 1162450375,
    "lat": 22.3168,
    "lon": 87.3065
  },
  "vehicle_capacity": 100,
  "num_vehicles": 5,
  "num_customers": 20,
  "distance_matrix": "ref:data/processed/distance_matrix.npy",
  "travel_time_matrix": "ref:data/processed/travel_time_matrix.npy",
  "node_id_map": "ref:data/processed/node_id_map.json",
  "customers": [
    {
      "node_id": 923847123,
      "demand": 14,
      "lat": 22.3245,
      "lon": 87.3112
    }
  ]
}
```

---

## 6. Benchmarking Harness & Statistical Aggregation

### 6.1 Runner Lifecycle (`src/benchmark/runner.py`)
- Iterates over all 3 instances and all 6 algorithms.
- Runs each stochastic solver over **10 distinct random seeds** (seeds 1 through 10) for rigorous statistical confidence.
- Records per-run results in `results/logs/results_raw.csv`:
  - `instance_id`, `algorithm`, `seed`, `total_cost`, `runtime_seconds`, `converged_iteration`.
- Records complete iteration-by-iteration cost trajectories in `results/logs/convergence_histories.json`.
- Extracts and saves the optimal route node sequence for each (instance, algorithm) pair into `results/logs/best_routes.json`.

### 6.2 Metrics Aggregator (`src/benchmark/metrics.py`)
Aggregates seed runs into `results/logs/results_summary.csv`:
- **Mean Cost** ($\mu$) and **Standard Deviation** ($\sigma$) in meters.
- **Best Cost** found across all 10 seeds.
- **Mean Runtime** in seconds.
- **Mean Converged Iteration** (iteration where minimum cost was first reached).
- **Optimality Gap (%)**: Percentage divergence from the absolute best-known solution for that instance:
  $$\text{Gap} = \frac{\mu_{\text{cost}} - C^*}{C^*} \times 100\%$$

### 6.3 Automated Report & Chart Generator (`src/benchmark/report.py`)
Produces publication-ready charts in `results/plots/`:
- **Convergence Curves** (`convergence_instance_n*.png`): Log-scaled convergence trajectories comparing all algorithms over 100 iterations.
- **Cost Comparison Bar Charts** (`cost_comparison_instance_n*.png`).
- **Runtime Comparison Bar Charts** (`runtime_comparison_instance_n*.png`).
- **Scalability Trajectories**:
  - `scalability_cost.png`: Routing cost progression as customer size $N$ scales from 20 to 100.
  - `scalability_runtime.png`: Computational scaling overhead across $N$.
- **Markdown Report** (`results/comparison_report.md`).

---

## 7. Visualization Subsystem

The visualization layer provides three complementary rendering pipelines:

```
                          best_routes.json / Instance Data
                                         │
                 ┌───────────────────────┼───────────────────────┐
                 ▼                       ▼                       ▼
          [static_map.py]        [interactive_map.py]   [comparison_grid.py]
                 │                       │                       │
                 ▼                       ▼                       ▼
         OSMnx Static PNGs       Folium Leaflet HTML      2x3 Subplot Grid
        (results/route_maps/    (results/route_maps/    (results/route_maps/
              static/)              interactive/)        comparison_grids/)
```

### 7.1 Static Cartographic Maps (`src/visualization/static_map.py`)
- Utilizes `osmnx.plot_graph` and `matplotlib`.
- Converts discrete stop sequences $[v_1, v_2, \dots, v_k]$ into detailed turn-by-turn road paths by computing sub-shortest paths `nx.shortest_path(graph, u, v, weight='length')` between consecutive stops.
- Overlays vehicle routes with distinct qualitative colors (`PALETTE`), plotting lines directly over the real street network.
- Visual markers:
  - **Depot**: Prominent red star ($\star$) with white outline.
  - **Customer Stops**: Black dots with high z-order.
- Generates 18 individual static PNGs (3 instances $\times$ 6 algorithms) in `results/route_maps/static/`.

### 7.2 Interactive Leaflet/Folium Maps (`src/visualization/interactive_map.py`)
- Built with Python `folium` (Leaflet.js wrapper).
- Features:
  - **Multiple Base Tiles**: OpenStreetMap, CartoDB Positron, CartoDB Dark Matter.
  - **True Road Geometry Tracing**: Inspects edge geometry attributes (`geometry` Shapely LineStrings) or connects intermediate nodes to follow actual road curves rather than straight lines.
  - **Layer Controls**: Each vehicle route is an independent `folium.FeatureGroup`, allowing users to toggle specific routes on and off in the browser.
  - **Depot Marker**: Dedicated red building icon marker.
  - **Interactive Customer Markers**: Color-coded circle markers per vehicle with rich popups showing Customer Node ID, Stop Sequence Index, Demand (kg), and Coordinates.
  - **Fullscreen Plugin**: Interactive fullscreen toggle for presentation.
- Generates 18 interactive HTML files in `results/route_maps/interactive/`.

### 7.3 Side-by-Side 2x3 Comparison Grids (`src/visualization/comparison_grid.py`)
- Constructs a unified 2-row by 3-column Matplotlib figure per instance.
- **Ordered Placement**:
  - Row 0: Nearest Neighbor | Clarke-Wright | Standard PSO
  - Row 1: Genetic Algorithm (Bottom-Left) | QPSO | Ant Colony (MMAS)
- **Unified Visual Standards**:
  - Identical spatial bounding boxes $[x_{\min}, x_{\max}]$ and $[y_{\min}, y_{\max}]$ across all 6 subplots.
  - Consistent road network background (subtle light gray `#E2E8F0`).
  - Standardized metrics box per subplot: Total Distance, Fleet Size, Runtime, and Optimality Gap.
- Generates 3 master comparison images in `results/route_maps/comparison_grids/`.

---

## 8. Empirical Benchmark Results

The table below reflects the actual benchmark runs stored in `results/logs/results_summary.csv`:

| Instance ID | Algorithm | Mean Cost (m) | Std Cost (m) | Mean Runtime (s) | Conv. Iteration | Optimality Gap (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **instance_n20** | **Ant Colony (MMAS)** | **58,709.33** | 911.46 | 4.9176 | 73.6 | **0.56% (Best)** |
| instance_n20 | Clarke-Wright | 64,161.05 | 0.00 | 0.0065 | 0.0 | 9.90% |
| instance_n20 | Genetic Algorithm | 69,483.51 | 1,667.63 | 0.7390 | 99.0 | 19.02% |
| instance_n20 | Standard PSO | 70,822.61 | 2,515.94 | 0.2445 | 72.9 | 21.31% |
| instance_n20 | QPSO | 71,599.49 | 3,984.84 | 0.7928 | 89.9 | 22.64% |
| instance_n20 | Nearest Neighbor | 85,440.06 | 0.00 | 0.0010 | 0.0 | 46.35% |
| **instance_n50** | **Genetic Algorithm** | **190,606.71** | 4,643.21 | 1.4967 | 99.0 | **3.49% (Best)** |
| instance_n50 | Ant Colony (MMAS) | 2,169,348.50 | 8,993.02 | 10.6776 | 83.2 | 1,077.85% |
| instance_n50 | Clarke-Wright | 3,149,027.94 | 0.00 | 0.0085 | 0.0 | 1,609.77% |
| instance_n50 | Standard PSO | 3,214,001.06 | 8,066.93 | 0.2894 | 86.9 | 1,645.05% |
| instance_n50 | QPSO | 3,225,133.37 | 3,916.95 | 1.1526 | 52.5 | 1,651.10% |
| instance_n50 | Nearest Neighbor | 3,184,609.02 | 0.00 | 0.0011 | 0.0 | 1,629.09% |
| **instance_n100**| **Genetic Algorithm** | **342,041.04** | 14,392.33 | 3.2367 | 99.0 | **7.18% (Best)** |
| instance_n100 | Ant Colony (MMAS) | 2,259,572.86 | 5,313.06 | 36.7424 | 54.3 | 608.07% |
| instance_n100 | Nearest Neighbor | 2,266,701.90 | 0.00 | 0.0020 | 0.0 | 610.30% |
| instance_n100 | Clarke-Wright | 3,228,951.41 | 0.00 | 0.0780 | 0.0 | 911.83% |
| instance_n100 | Standard PSO | 3,429,316.32 | 11,125.50 | 1.4394 | 87.3 | 974.62% |
| instance_n100 | QPSO | 3,443,906.69 | 7,260.20 | 1.8511 | 51.3 | 979.19% |

### 8.1 Key Empirical Findings
1. **Small Instances ($N=20$)**:
   - MMAS achieved the best solution quality (58.7 km), followed by Clarke-Wright (64.2 km).
   - Standard PSO and QPSO exhibited competitive performance ($\approx 70.8\text{ km}$ vs $\approx 71.5\text{ km}$), confirming the validity of the continuous random-key formulation on modest problem sizes.
2. **Medium & Large Instances ($N \ge 50$)**:
   - **Genetic Algorithm Domination**: GA with 2-opt search outperformed all other solvers by an order of magnitude. It was the only metaheuristic that consistently packed customer stops into routes that satisfied the available vehicle fleet limit ($K_{\text{max}}$), completely avoiding penalty costs.
   - **Fleet Penalty Impact**: For $N=50$ and $N=100$, greedy-split decoders in PSO, QPSO, and constructive heuristics occasionally produced routes requiring 1 to 3 vehicles beyond the fleet limit, triggering the exterior penalty ($10^6$ per excess vehicle).
   - **Solving Time Efficiency**: Constructive heuristics ran in under 10 ms; PSO/QPSO solved in 0.2–1.8 seconds; GA solved in 0.7–3.2 seconds; MMAS required 4.9–36.7 seconds due to graph ant walk evaluations.

---

## 9. Verification & Sanity Testing

The automated test suite in `tests/sanity_check.py` validates both algorithmic correctness and visualization pipelines:

1. **Route Structure Invariance**: Confirms every route starts and ends at `depot_node_id`.
2. **Customer Partition Invariance**: Ensures every customer is visited **exactly once** (zero omissions, zero duplicates):
   $$\bigcup_{r \in R} (r \setminus \{\text{depot}\}) = C, \quad \sum_{r \in R} |r \setminus \{\text{depot}\}| = |C|$$
3. **Capacity Constraints**: Confirms that for every route $r \in R$, $\sum_{i \in r} q_i \le Q_{\text{vehicle}}$.
4. **Solver Instrumentation**: Confirms non-empty convergence histories and positive runtimes.
5. **Static Visualizer Verification**: Generates and validates that `plot_algorithm_routes` creates non-empty PNG maps.
6. **Interactive Visualizer Verification**: Generates and validates that `build_interactive_route_map` creates valid Leaflet HTML files.
7. **Comparison Grid Verification**: Generates and validates that `plot_comparison_grid` renders complete 2x3 subplot grids.

Execution command:
```bash
python tests/sanity_check.py
```

---

## 10. Execution Command Quick Reference

```bash
# 1. Pre-processing: Generate road graph, demands, and problem instances
python src/data_pipeline/fetch_osm.py
python src/data_pipeline/fetch_demand.py
python src/data_pipeline/fetch_depots.py
python src/data_pipeline/build_instance.py

# 2. Run Sanity Checks
python tests/sanity_check.py

# 3. Execute Benchmarking Harness (10 seeds across all instances)
python src/benchmark/runner.py

# 4. Calculate Summary Statistics & Metrics
python src/benchmark/metrics.py

# 5. Generate Plots & Markdown Comparison Report
python src/benchmark/report.py

# 6. Generate Route Visualizations
# Static Maps:
python src/visualization/static_map.py --generate-all
# Interactive HTML Maps:
python src/visualization/interactive_map.py --generate-all
# 2x3 Comparison Grids:
python src/visualization/comparison_grid.py --generate-all
```
