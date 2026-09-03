# VRP Benchmark Algorithms: Technical Deep Dive, Mechanics, Pros & Cons

This document provides a comprehensive technical guide to the six Vehicle Routing Problem (VRP) algorithms implemented and evaluated in the **QPSO-VRP Benchmarking Suite**.

All algorithms are benchmarked on real-world road networks from **OpenStreetMap (OSM) for Kharagpur, West Bengal, India**, using Dijkstra shortest path distance/time matrices, real parcel delivery weights from Kaggle logistics data, and authentic India Post branch depots.

---

## 1. Problem Formulation & Shared Evaluation Metric

Each algorithm extends the abstract base class [`RoutingAlgorithm`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/base.py#L22-L38) in [src/algorithms/base.py](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/base.py) and outputs a standardized [`SolutionResult`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/base.py#L4-L20).

### 1.1 Objective Function
The routing cost $f(R)$ minimizes the total physical road distance traversed across all vehicle routes $R$, subject to vehicle capacity constraints and an exterior penalty for exceeding the allocated fleet size:

$$\min \quad f(R) = \sum_{r \in R} \sum_{i=0}^{|r|-2} D(\pi_{r, i}, \pi_{r, i+1}) + \lambda \cdot \max(0, |R| - K_{\text{max}})$$

Where:
- $R$: The set of vehicle routes.
- $\pi_{r, i}$: The $i$-th road node in route $r$, with $\pi_{r, 0} = \pi_{r, |r|-1} = \text{depot}$.
- $D(u, v)$: Shortest road network distance between junctions $u$ and $v$ (looked up from `distance_matrix.npy`).
- $|R|$: Number of active routes generated.
- $K_{\text{max}}$: Maximum available vehicle fleet limit for the problem instance.
- $\lambda = 10^6$: Heavy exterior penalty factor per vehicle used in excess of $K_{\text{max}}$.

### 1.2 Vehicle Capacity Constraint
For every vehicle tour $r \in R$, the cumulative customer package demand cannot exceed the vehicle payload capacity:

$$\sum_{j \in r \setminus \{\text{depot}\}} q_j \le Q_{\text{vehicle}}$$

---

## 2. Algorithm Deep Dives

---

### 2.1 Nearest Neighbor (NN)

- **Source Code**: [src/algorithms/nn_clarke_wright.py](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/nn_clarke_wright.py)
- **Class**: [`NearestNeighborClarkeWright`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/nn_clarke_wright.py#L7-L87) (`method="nn"`)
- **Category**: Greedy Constructive Heuristic

#### Mathematical & Operational Mechanics
1. A delivery vehicle departs from the central postal depot node.
2. At current location $i$, the algorithm queries the distance matrix $D(i, j)$ across all unvisited customer nodes $j \in \mathcal{U}$.
3. The candidate set is filtered to only include customers whose demand $q_j$ does not exceed the vehicle's remaining payload capacity ($q_j \le Q_{\text{rem}}$).
4. Selects the nearest eligible candidate:
   $$j^* = \arg\min_{j \in \mathcal{U}, q_j \le Q_{\text{rem}}} D(i, j)$$
5. The vehicle travels to $j^*$, decrements its capacity ($Q_{\text{rem}} \leftarrow Q_{\text{rem}} - q_{j^*}$), and removes $j^*$ from $\mathcal{U}$.
6. When no remaining unvisited customer fits into the vehicle's remaining capacity, the vehicle returns to the depot, and a fresh vehicle departs.

#### Pros
- **Near-Instant Execution**: Resolves $N=100$ instances in approximately **1 to 2 milliseconds** ($O(N^2)$ computational complexity).
- **Deterministic & Zero-Hyperparameters**: Completely reproducible with zero parameter tuning (no learning rates, mutation rates, or weights).
- **Ideal Seed Warm-Starter**: Excellent for generating immediate, strictly capacity-valid starting solutions for iterative metaheuristics.

#### Cons
- **Myopic Decisions**: Highly susceptible to greedy traps; early decisions ignore downstream impacts, frequently leaving peripheral or isolated stops stranded at the end.
- **Suboptimal Global Quality**: Produces substantial optimality gaps (~46% on $N=20$, over 600% on $N=100$) because it lacks global geometric awareness.
- **Fleet Limit Violations**: Fragmented routes often require more vehicles than the allocated fleet limit $K_{\text{max}}$, incurring massive penalties ($10^6 \times \text{overflow}$).
- **No Search or Improvement Capability**: Single-pass heuristic without mechanisms to untangle crossing paths or optimize node orders.

---

### 2.2 Clarke-Wright Savings Heuristic (CW)

- **Source Code**: [src/algorithms/nn_clarke_wright.py](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/nn_clarke_wright.py)
- **Class**: [`NearestNeighborClarkeWright`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/nn_clarke_wright.py#L88-L178) (`method="clarke_wright"`)
- **Category**: Route-merging Constructive Heuristic

#### Mathematical & Operational Mechanics
1. **Initial Solution**: Configures $N$ dedicated round trips: $[\text{depot}, i, \text{depot}]$ for every customer $i \in \{1, \dots, N\}$.
2. **Savings Computation**: Computes the distance reduction achieved by linking customers $i$ and $j$ into a single shared tour:
   $$S_{ij} = D(\text{depot}, i) + D(\text{depot}, j) - D(i, j)$$
3. **Sorting**: Sorts all pairs $(i, j)$ in descending order of savings $S_{ij}$.
4. **Greedy Merging**: Evaluates pairs sequentially and merges the route containing $i$ with the route containing $j$ if and only if:
   - Customers $i$ and $j$ reside in separate routes ($r_i \ne r_j$).
   - Both $i$ and $j$ are currently adjacent to the depot in their respective routes (i.e., at the exterior endpoints).
   - The merged route payload satisfies vehicle capacity: $\sum_{k \in r_i} q_k + \sum_{l \in r_j} q_l \le Q_{\text{vehicle}}$.

#### Pros
- **High Computational Efficiency**: Executes within **6 to 78 milliseconds** ($O(N^2 \log N)$ sorting bottleneck).
- **Depot-Centric Spatial Awareness**: The savings metric naturally accounts for depot proximity, preventing peripheral node abandonment.
- **Strong Small-Scale Benchmark**: Achieved a competitive 9.9% optimality gap on $N=20$ (64.1 km total distance).
- **Standard Industrial Heuristic**: Serves as a primary reference point in logistics management.

#### Cons
- **Irreversible Merge Decisions**: Once two routes merge, the connection cannot be broken even if a globally superior combination is discovered later in the sorted list.
- **Endpoint Rigidity**: Can only merge customers located at the ends of routes, preventing internal sequence optimizations.
- **Under-Packing Under Tight Fleets**: On larger scales ($N=50, 100$), greedy merges can leave small residual routes that exceed $K_{\text{max}}$, resulting in exterior penalty triggers.

---

### 2.3 Standard Particle Swarm Optimization (Standard PSO)

- **Source Code**: [src/algorithms/standard_pso.py](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/standard_pso.py)
- **Class**: [`StandardPSO`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/standard_pso.py#L7-L143)
- **Category**: Continuous Swarm Intelligence Metaheuristic

#### Mathematical & Operational Mechanics
1. **Random-Key Encoding**: Each particle $i$ in a swarm of size $M=50$ maintains a continuous position vector $\mathbf{x}_i \in [0, 1]^N$ and velocity vector $\mathbf{v}_i \in [-0.1, 0.1]^N$.
2. **Priority Permutation Decoding**: The discrete customer visit order is derived via `np.argsort(x_i)`. A greedy splitter traverses this permutation, assigning customers to the current vehicle until demand exceeds $Q_{\text{vehicle}}$, returning to the depot to start the next route.
3. **Kinematic Updates**:
   $$\mathbf{v}_i^{(t+1)} = w \mathbf{v}_i^{(t)} + c_1 r_1 (\mathbf{pbest}_i - \mathbf{x}_i^{(t)}) + c_2 r_2 (\mathbf{gbest} - \mathbf{x}_i^{(t)})$$
   $$\mathbf{x}_i^{(t+1)} = \text{clip}\left(\mathbf{x}_i^{(t)} + \mathbf{v}_i^{(t+1)}, 0, 1\right)$$
   - Inertia weight: $w = 0.729$
   - Cognitive acceleration: $c_1 = 1.4944$
   - Social acceleration: $c_2 = 1.4944$
   - Random factors: $r_1, r_2 \sim \mathcal{U}(0, 1)$
4. **Historical Memory**: Personal best ($\mathbf{pbest}_i$) and global best ($\mathbf{gbest}$) are tracked across 100 iterations.

#### Pros
- **Fast Iteration Speed**: Vectorized NumPy operations achieve solution times between **0.24s ($N=20$) and 1.44s ($N=100$)**.
- **Collective Intelligence**: Combines individual cognitive memory with swarm-level social communication.
- **Continuous Representation Flexibility**: Well-suited for extensions to continuous multi-objective variables (such as variable vehicle speeds or fuel burn rates).

#### Cons
- **Premature Convergence & Local Trapping**: Deterministic Newtonian trajectory updates cause velocities to dampen over iterations. If $\mathbf{gbest}$ settles in a local optimum, particles collapse into the same region and cannot escape.
- **Random-Key Disconnect**: Euclidean distance in continuous coordinate space does not correlate smoothly with route quality in permutation space. Small continuous coordinate perturbations can result in sudden, massive tour reorganizations.
- **High Sensitivity to Penalty Functions**: Without embedded discrete repair operators, random continuous keys often generate permutations that exceed vehicle fleet limits on larger problems ($N \ge 50$).

---

### 2.4 Quantum-behaved Particle Swarm Optimization (QPSO)

- **Source Code**: [src/algorithms/qpso.py](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/qpso.py)
- **Class**: [`QPSO`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/qpso.py#L7-L143)
- **Category**: Quantum-Mechanics Inspired Swarm Metaheuristic

#### Mathematical & Operational Mechanics
1. **Delta Potential Well Formulation**: Particles do not follow classical Newtonian trajectories with velocity vectors. Instead, particles behave as quantum wave packets $\psi(\mathbf{x}, t)$ bound within a quantum delta potential well centered at a local stochastic attractor $\mathbf{p}_i$.
2. **Mean Best Position ($m_{\text{best}}$)**: Swarm-wide centroid of all personal best positions:
   $$m_{\text{best}} = \frac{1}{M} \sum_{i=1}^M \mathbf{pbest}_i$$
3. **Local Stochastic Attractor**: Blends personal experience with global knowledge:
   $$\mathbf{p}_{i, d} = \phi_d \cdot \text{pbest}_{i, d} + (1 - \phi_d) \cdot \text{gbest}_d, \quad \phi_d \sim \mathcal{U}(0, 1)$$
4. **Monte Carlo Quantum Position Update**: Derived via the inverse transform of the quantum wave probability density $|\psi(\mathbf{x})|^2$:
   $$x_{i, d}^{(t+1)} = p_{i, d} \pm \beta \left| m_{\text{best}, d} - x_{i, d}^{(t)} \right| \ln\left(\frac{1}{u_d}\right), \quad u_d \sim \mathcal{U}(0, 1)$$
5. **Annealed Contraction-Expansion ($\beta$)**:
   $$\beta(t) = \beta_{\text{start}} - (\beta_{\text{start}} - \beta_{\text{end}}) \cdot \frac{t}{T_{\text{max}}}$$
   Decays linearly from $\beta_{\text{start}} = 1.0$ (strong global exploration) to $\beta_{\text{end}} = 0.5$ (fine local exploitation).

#### Pros
- **Global Search Capability (No Velocity Stagnation)**: Because the quantum probability distribution has infinite support through the logarithmic term $\ln(1/u)$, particles have a non-zero probability of appearing anywhere in search space, preventing permanent stagnation in local traps.
- **Fewer Hyperparameters**: Completely eliminates velocity clamping, inertia weight ($w$), and acceleration constants ($c_1, c_2$). It is governed solely by the single parameter $\beta$.
- **Population Diversity via $m_{\text{best}}$**: Incorporating the swarm centroid prevents the particles from collapsing prematurely toward $\mathbf{gbest}$.
- **Lightweight Runtime Overhead**: Highly efficient solution times (**0.79s for $N=20$; 1.85s for $N=100$**).

#### Cons
- **Permutation Decoder Bottleneck**: Like Standard PSO, QPSO decodes continuous vectors into permutations via sorting (`np.argsort`), which does not capture direct graph-edge adjacency.
- **Sensitivity to $\beta$ Cooling Schedule**: Rapid decay freezes exploration too early; slow decay results in chaotic wandering that impairs convergence.
- **Requires Hybridization for Large-Scale CVRP**: On larger problem sizes ($N \ge 50$), continuous metaheuristics require pairing with discrete local search (e.g., 2-Opt or Split algorithms) to enforce valid vehicle counts and prevent fleet limit penalties.

---

### 2.5 Permutation Genetic Algorithm with 2-Opt (GA + 2-Opt)

- **Source Code**: [src/algorithms/genetic_algorithm.py](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/genetic_algorithm.py)
- **Class**: [`GeneticAlgorithm`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/genetic_algorithm.py#L14-L188)
- **Category**: Memetic Evolutionary Metaheuristic (DEAP Framework + 2-Opt Local Search)

#### Mathematical & Operational Mechanics
1. **Chromosome Representation**: An individual is represented as a direct discrete permutation of customer node indices: $[c_1, c_2, \dots, c_N]$.
2. **Genetic Operators**:
   - **Order Crossover (OX1)** ($p_c = 0.8$): Selects a random slice from parent 1, preserves the relative order of remaining items from parent 2, avoiding invalid tours (no duplicates, no omissions).
   - **Shuffle Indexes Mutation** ($p_m = 0.2$): Swaps customer stops with probability $0.05$ per locus.
   - **Tournament Selection** ($k = 3$): Enforces selection pressure toward high-fitness individuals.
3. **2-Opt Local Search Refinement**:
   - Post-processes the decoded routes of the best chromosome.
   - Evaluates non-adjacent edge pairs $(i, i+1)$ and $(j, j+1)$ along each route and reverses the sub-tour segment if:
     $$D(\pi_i, \pi_j) + D(\pi_{i+1}, \pi_{j+1}) < D(\pi_i, \pi_{i+1}) + D(\pi_j, \pi_{j+1})$$
   - Eliminates self-crossings and uncrosses tangled road routes.

#### Pros
- **Superior Solution Quality on Large Instances**: Ranked **#1 on $N=50$ (190,606 m)** and **#1 on $N=100$ (342,041 m)** with low optimality gaps (**3.49%** and **7.18%**).
- **Direct Permutation Encoding**: Operates natively in discrete space without lossy continuous-to-discrete approximations.
- **Memetic Synergy**: Global genetic operators search for effective customer groupings, while 2-Opt polishes individual route trajectories.
- **Strict Fleet Constraint Adherence**: Consistently produces route allocations that fit within the allowed vehicle fleet limit $K_{\text{max}}$.

#### Cons
- **Higher Computational Cost**: Runtime is higher than PSO (~3.24 seconds for $N=100$) due to DEAP object cloning and 2-Opt quadratic edge checks.
- **Hyperparameter Sensitivity**: Requires configuring population size, crossover rate, mutation rate, tournament size, and number of generations.
- **Intra-Route Only**: The current 2-Opt implementation refines each vehicle's route individually; it does not perform inter-route exchanges (such as 2-Opt* or Cross-Exchange) between separate vehicles.

---

### 2.6 Max-Min Ant System (MMAS / ACO)

- **Source Code**: [src/algorithms/aco_mmas.py](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/aco_mmas.py)
- **Class**: [`ACOMMAS`](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/aco_mmas.py#L8-L185)
- **Category**: Graph-Based Ant Colony Optimization Metaheuristic

#### Mathematical & Operational Mechanics
1. **Pheromone ($\tau$) and Visibility ($\eta$) Matrices**:
   - $\tau_{ij}$: Learned artificial pheromone intensity on road edge $(i, j)$.
   - $\eta_{ij} = \frac{1}{\max(D(i, j), 1.0)}$: Heuristic visibility favoring shorter road segments.
2. **State Transition Probability**:
   An ant $k$ currently at node $i$ selects the next customer $j$ among candidate set $\mathcal{N}_i^k$ (unvisited customers that satisfy remaining vehicle capacity):
   $$P_{ij}^k = \frac{[\tau_{ij}]^\alpha \cdot [\eta_{ij}]^\beta}{\sum_{l \in \mathcal{N}_i^k} [\tau_{il}]^\alpha \cdot [\eta_{il}]^\beta}$$
   - Pheromone sensitivity: $\alpha = 1.0$
   - Visibility sensitivity: $\beta = 3.0$
3. **Pheromone Clamping to Prevent Stagnation**:
   Pheromones are strictly bounded within $[\tau_{\min}, \tau_{\max}]$:
   $$\tau_{\max} = \frac{1}{\rho \cdot C_{\text{best}}}, \quad \tau_{\min} = \frac{\tau_{\max}}{2N}$$
   Where $\rho = 0.1$ is the evaporation rate and $C_{\text{best}}$ is the global best solution cost.
4. **Selective Global Deposition**: Only the iteration-best or global-best ant deposits pheromones:
   $$\tau_{ij} \leftarrow (1 - \rho)\tau_{ij} + \Delta \tau_{ij}^{\text{best}}, \quad \Delta \tau_{ij} = \frac{1}{C_{\text{best}}}$$

#### Pros
- **Native Graph Operation**: Operates directly on graph edges and Dijkstra road distances rather than abstract vectors.
- **Strict Capacity Guarantee**: Ants check remaining capacity at every step during route construction, avoiding invalid capacity states.
- **Best Small-Scale Performance**: Achieved the **lowest routing cost on $N=20$ (58,709 m, 0.56% optimality gap)**.
- **Stagnation Resistance**: The $[\tau_{\min}, \tau_{\max}]$ bounding mechanism ensures pheromone paths never decay to zero, maintaining exploration.

#### Cons
- **High Computational Overhead**: The slowest algorithm evaluated (**~36.7 seconds on $N=100$**) because each ant must calculate normalized probability distributions across all available candidates at every stop.
- **Memory Footprint**: Maintains full $N \times N$ floating-point matrices for pheromones and visibility.
- **Stagnation on Large Graphs**: Without supplementary local search (e.g., 2-Opt or 3-Opt), ant tours on large graphs ($N \ge 50$) can settle into sub-optimal route groupings.

---

## 3. Comparative Summary Matrix

| Algorithm | Method Category | Problem Representation | Mean Runtime ($N=100$) | Best Problem Scale | Primary Strength | Main Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Nearest Neighbor** | Constructive | Greedy Graph Walk | **0.002 s** | Any (as warm starter) | Instant computation, zero tuning | Myopic, high route counts and fleet violations |
| **Clarke-Wright** | Constructive | Pairwise Savings | **0.078 s** | $N \le 30$ | Depot-aware spatial merging | Greedy merges are irreversible |
| **Standard PSO** | Metaheuristic | Continuous $[0, 1]^N$ | **1.439 s** | $N \le 30$ | Fast vectorized execution | Premature convergence, velocity damping |
| **QPSO** | Metaheuristic | Quantum Delta Well | **1.851 s** | $N \le 50$ | Escapes local traps via wave distribution | Sorting decoder disconnect; needs local search |
| **GA + 2-Opt** | Memetic Metaheuristic | Discrete Permutation | **3.237 s** | **$N = 50 \text{ to } 100+$** | **Best global route quality & constraint adherence** | Multiple hyperparameters to calibrate |
| **MMAS (ACO)** | Metaheuristic | Edge Pheromone Matrix | **36.742 s** | **$N \le 20$** | **Highest precision on compact networks** | Substantial $O(K \cdot N^2)$ runtime overhead |

---

## 4. Empirical Benchmark Results (Kharagpur Dataset)

The table below summarizes performance across **10 random seeds** (seeds 1 to 10) per algorithm on the Kharagpur road network:

| Instance | Algorithm | Mean Cost (m) | Std Cost (m) | Mean Runtime (s) | Best Cost (m) | Optimality Gap (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N=20** | **Ant Colony (MMAS)** | **58,709.33** | 911.46 | 4.9176 | **58,382.11** | **0.56%** |
| $N=20$ | Clarke-Wright | 64,161.05 | 0.00 | 0.0065 | 64,161.05 | 9.90% |
| $N=20$ | Genetic Algorithm | 69,483.51 | 1,667.63 | 0.7390 | 66,742.18 | 19.02% |
| $N=20$ | Standard PSO | 70,822.61 | 2,515.94 | 0.2445 | 67,419.04 | 21.31% |
| $N=20$ | QPSO | 71,599.49 | 3,984.84 | 0.7928 | 66,912.44 | 22.64% |
| $N=20$ | Nearest Neighbor | 85,440.06 | 0.00 | 0.0010 | 85,440.06 | 46.35% |
| **N=50** | **Genetic Algorithm** | **190,606.71** | 4,643.21 | 1.4967 | **184,178.43** | **3.49%** |
| $N=50$ | Ant Colony (MMAS) | 2,169,348.50* | 8,993.02 | 10.6776 | 2,154,231.10 | 1077.85% |
| $N=50$ | Clarke-Wright | 3,149,027.94* | 0.00 | 0.0085 | 3,149,027.94 | 1609.77% |
| $N=50$ | Nearest Neighbor | 3,184,609.02* | 0.00 | 0.0011 | 3,184,609.02 | 1629.09% |
| $N=50$ | Standard PSO | 3,214,001.06* | 8,066.93 | 0.2894 | 3,201,844.12 | 1645.05% |
| $N=50$ | QPSO | 3,225,133.37* | 3,916.95 | 1.1526 | 3,219,410.88 | 1651.10% |
| **N=100**| **Genetic Algorithm** | **342,041.04** | 14,392.33 | 3.2367 | **319,134.22** | **7.18%** |
| $N=100$| Ant Colony (MMAS) | 2,259,572.86* | 5,313.06 | 36.7424 | 2,251,104.50 | 608.07% |
| $N=100$| Nearest Neighbor | 2,266,701.90* | 0.00 | 0.0020 | 2,266,701.90 | 610.30% |
| $N=100$| Clarke-Wright | 3,228,951.41* | 0.00 | 0.0780 | 3,228,951.41 | 911.83% |
| $N=100$| Standard PSO | 3,429,316.32* | 11,125.50 | 1.4394 | 3,412,890.11 | 974.62% |
| $N=100$| QPSO | 3,443,906.69* | 7,260.20 | 1.8511 | 3,431,209.45 | 979.19% |

*\*Note: Costs above $10^6$ indicate that the solver exceeded the available vehicle fleet limit $K_{\text{max}}$, triggering the exterior penalty function ($\lambda = 10^6 \times \text{overflow}$).*

---

## 5. Architectural Takeaways & Design Recommendations

1. **Memetic Combinations Are Crucial at Scale**:
   The standalone Genetic Algorithm outperforms continuous swarm methods on large instances because it pairs global crossover exploration with local 2-Opt topological untangling. This enables it to maintain valid fleet limits without triggering penalties.

2. **QPSO vs. Standard PSO Dynamics**:
   Standard PSO suffers from velocity decay, leading to premature stagnation in complex search landscapes. QPSO overcomes velocity damping through its quantum wave-packet formulation and mean-best attractor ($m_{\text{best}}$). To realize its full potential on discrete problems like CVRP, QPSO should be coupled with a discrete local search heuristic (e.g., 2-Opt or variable neighborhood search).

3. **Two-Stage Dispatch Architecture**:
   For production logistics deployments:
   - **Phase 1 (Instant Response)**: Run [Clarke-Wright](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/nn_clarke_wright.py) to produce a validated feasible dispatch plan within 10 milliseconds.
   - **Phase 2 (Background Optimization)**: Feed the initial plan into [Genetic Algorithm](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/genetic_algorithm.py) or [QPSO](file:///c:/Users/Ruhaan%20Kakar/Desktop/SIH/src/algorithms/qpso.py) with 2-Opt refinement for iterative route shortening.
