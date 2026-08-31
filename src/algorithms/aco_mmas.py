import time
import numpy as np
from typing import Dict, Any, List
from src.algorithms.base import RoutingAlgorithm, SolutionResult
from src.utils.graph_utils import load_instance_resources, calculate_total_distance
from src.algorithms.nn_clarke_wright import NearestNeighborClarkeWright

class ACOMMAS(RoutingAlgorithm):
    """
    Implements Max-Min Ant System (MMAS) Ant Colony Optimization for CVRP.
    Uses pheromone matrix constrained within [tau_min, tau_max] limits.
    """
    def __init__(self, instance: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(instance, config)
        self.num_ants = config.get("num_ants", 20)
        self.max_iter = config.get("max_iter", 100)
        self.alpha = config.get("alpha", 1.0)           # Pheromone importance
        self.beta = config.get("beta", 3.0)             # Visibility importance
        self.rho = config.get("evaporation_rate", 0.1)  # Evaporation rate
        self.p_best = config.get("p_best", 0.05)         # MMAS specific parameter
        
        # Load resources
        self.distance_matrix, self.node_id_map = load_instance_resources(self.instance)
        
        # Parse VRP details
        self.customers = self.instance["customers"]
        self.customer_ids = [c["node_id"] for c in self.customers]
        self.demands = {c["node_id"]: c["demand"] for c in self.customers}
        self.vehicle_capacity = self.instance["vehicle_capacity"]
        self.num_vehicles = self.instance["num_vehicles"]
        self.N = len(self.customers)
        
        # Total nodes (customers + depot)
        self.node_ids = [self.depot_id] + self.customer_ids
        self.total_nodes = len(self.node_ids)
        self.node_index_to_id = {i: nid for i, nid in enumerate(self.node_ids)}
        self.node_id_to_index = {nid: i for i, nid in enumerate(self.node_ids)}

        # Visibility matrix: eta_ij = 1/d_ij
        self.eta = np.zeros((self.total_nodes, self.total_nodes), dtype=np.float32)
        for i, u in enumerate(self.node_ids):
            for j, v in enumerate(self.node_ids):
                if i != j:
                    dist = float(self.distance_matrix[self.node_id_map[u], self.node_id_map[v]])
                    # Prevent divide by zero
                    self.eta[i, j] = 1.0 / max(dist, 1.0)
                else:
                    self.eta[i, j] = 0.0

        # Precompute eta raised to the beta power
        self.eta_beta = self.eta ** self.beta

    def _get_initial_best_cost(self) -> float:
        """Runs Clarke-Wright to estimate a good initial best cost for pheromone scaling."""
        try:
            cw = NearestNeighborClarkeWright(self.instance, {"method": "clarke_wright", "max_iter": 1})
            res = cw.solve()
            return res.total_cost
        except Exception:
            # Fallback to random tour estimation
            return 1000.0 * self.N

    def solve(self) -> SolutionResult:
        start_time = time.time()
        
        # 1. Initialize Pheromone matrix and bounds
        init_cost = self._get_initial_best_cost()
        tau_max = 1.0 / (self.rho * init_cost)
        tau_min = tau_max / (2.0 * self.N)
        tau = np.full((self.total_nodes, self.total_nodes), tau_max, dtype=np.float32)
        
        gbest_routes = None
        gbest_cost = float('inf')
        
        convergence_history = []
        
        # 2. Main Iteration Loop
        for iteration in range(self.max_iter):
            iteration_solutions = []
            iteration_costs = []
            
            for ant in range(self.num_ants):
                routes, cost = self._construct_solution(tau)
                iteration_solutions.append(routes)
                iteration_costs.append(cost)
                
            # Find iteration best
            ibest_idx = np.argmin(iteration_costs)
            ibest_routes = iteration_solutions[ibest_idx]
            ibest_cost = iteration_costs[ibest_idx]
            
            # Update global best
            if ibest_cost < gbest_cost:
                gbest_cost = ibest_cost
                gbest_routes = ibest_routes
                
                # Dynamically update pheromone limits based on new global best
                tau_max = 1.0 / (self.rho * gbest_cost)
                tau_min = tau_max / (2.0 * self.N)
                
            convergence_history.append(gbest_cost)
            
            # 3. Pheromone Evaporation
            tau = (1.0 - self.rho) * tau
            
            # 4. Pheromone Deposit (MMAS: only best ant deposits pheromone)
            # We use global best ant for deposition
            deposit_amount = 1.0 / gbest_cost
            for route in gbest_routes:
                for idx in range(len(route) - 1):
                    u = route[idx]
                    v = route[idx+1]
                    u_idx = self.node_id_to_index[u]
                    v_idx = self.node_id_to_index[v]
                    
                    tau[u_idx, v_idx] += deposit_amount
                    # Also deposit on reverse edge to support symmetric pathways
                    tau[v_idx, u_idx] += deposit_amount
                    
            # 5. Pheromone Clamping
            tau = np.clip(tau, tau_min, tau_max)
            
        runtime = time.time() - start_time
        return SolutionResult(gbest_routes, gbest_cost, convergence_history, runtime)

    def _construct_solution(self, tau: np.ndarray) -> tuple:
        """Constructs a single ant's solution respecting VRP constraints."""
        unvisited = set(self.customer_ids)
        routes = []
        
        depot_idx = self.node_id_to_index[self.depot_id]
        
        while unvisited:
            route = [self.depot_id]
            current_load = 0
            current_node = self.depot_id
            current_idx = depot_idx
            
            while unvisited:
                # Find candidates that fit the capacity limit
                candidates = [c for c in unvisited if current_load + self.demands[c] <= self.vehicle_capacity]
                
                if not candidates:
                    # No customer fits, return to depot
                    break
                    
                # Calculate selection probability for each candidate using vectorized operations
                cand_indices = [self.node_id_to_index[c] for c in candidates]
                t_vals = tau[current_idx, cand_indices]
                if self.alpha != 1.0:
                    t_vals = t_vals ** self.alpha
                e_vals = self.eta_beta[current_idx, cand_indices]
                
                probs = t_vals * e_vals
                probs = np.array(probs, dtype=np.float64)
                sum_probs = np.sum(probs)
                
                if sum_probs == 0:
                    probs = np.ones(len(candidates)) / len(candidates)
                else:
                    probs = probs / sum_probs
                    
                # Sample next customer
                next_cust = np.random.choice(candidates, p=probs)
                route.append(next_cust)
                current_load += self.demands[next_cust]
                unvisited.remove(next_cust)
                current_node = next_cust
                current_idx = self.node_id_to_index[next_cust]
                
            route.append(self.depot_id)
            routes.append(route)
            
        # Calculate solution cost
        distance = calculate_total_distance(routes, self.distance_matrix, self.node_id_map)
        
        # Penalize if it exceeds the number of vehicles
        num_routes = len(routes)
        penalty = 0.0
        if num_routes > self.num_vehicles:
            penalty = 1e6 * (num_routes - self.num_vehicles)
            
        cost = distance + penalty
        return routes, cost
