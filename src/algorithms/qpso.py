import time
import numpy as np
from typing import Dict, Any, List
from src.algorithms.base import RoutingAlgorithm, SolutionResult
from src.utils.graph_utils import load_instance_resources, calculate_total_distance

class QPSO(RoutingAlgorithm):
    """
    Implements Quantum-behaved Particle Swarm Optimization (QPSO) using Random-Key representation.
    Shares the exact same priority-permutation mapping and evaluation as Standard PSO.
    """
    def __init__(self, instance: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(instance, config)
        self.swarm_size = config.get("swarm_size", 50)
        self.max_iter = config.get("max_iter", 100)
        
        # QPSO specific: contraction-expansion coefficients
        self.beta_start = config.get("beta_start", 1.0)
        self.beta_end = config.get("beta_end", 0.5)
        
        # Load resources
        self.distance_matrix, self.node_id_map = load_instance_resources(self.instance)
        
        # Parse VRP details
        self.customers = self.instance["customers"]
        self.customer_ids = [c["node_id"] for c in self.customers]
        self.demands = {c["node_id"]: c["demand"] for c in self.customers}
        self.vehicle_capacity = self.instance["vehicle_capacity"]
        self.num_vehicles = self.instance["num_vehicles"]
        self.N = len(self.customers)

    def decode_permutation(self, permutation: List[int]) -> List[List[int]]:
        """Splits a customer permutation into routes based on capacity constraints."""
        routes = []
        current_route = [self.depot_id]
        current_load = 0
        
        for cust in permutation:
            demand = self.demands[cust]
            if current_load + demand <= self.vehicle_capacity:
                current_route.append(cust)
                current_load += demand
            else:
                current_route.append(self.depot_id)
                routes.append(current_route)
                current_route = [self.depot_id, cust]
                current_load = demand
                
        if len(current_route) > 1:
            current_route.append(self.depot_id)
            routes.append(current_route)
            
        return routes

    def evaluate_fitness(self, position: np.ndarray) -> tuple:
        """Decodes position vector and calculates route distance + vehicle penalty."""
        # Sort indices based on position priorities
        sorted_indices = np.argsort(position)
        permutation = [self.customer_ids[idx] for idx in sorted_indices]
        
        routes = self.decode_permutation(permutation)
        
        # Calculate raw distance
        distance = calculate_total_distance(routes, self.distance_matrix, self.node_id_map)
        
        # Apply penalty if routes exceed available vehicles
        num_routes = len(routes)
        penalty = 0.0
        if num_routes > self.num_vehicles:
            penalty = 1e6 * (num_routes - self.num_vehicles)
            
        fitness = distance + penalty
        return fitness, routes

    def solve(self) -> SolutionResult:
        start_time = time.time()
        
        # 1. Initialize Swarm Positions
        positions = np.random.uniform(0.0, 1.0, (self.swarm_size, self.N))
        
        pbest_positions = np.copy(positions)
        pbest_fitnesses = np.zeros(self.swarm_size)
        pbest_routes = [None] * self.swarm_size
        
        # Initial evaluation
        for i in range(self.swarm_size):
            pbest_fitnesses[i], pbest_routes[i] = self.evaluate_fitness(positions[i])
            
        # Find global best
        gbest_idx = np.argmin(pbest_fitnesses)
        gbest_position = np.copy(pbest_positions[gbest_idx])
        gbest_fitness = pbest_fitnesses[gbest_idx]
        gbest_routes = pbest_routes[gbest_idx]
        
        convergence_history = []
        
        # 2. Main Optimization Loop
        for iteration in range(self.max_iter):
            # Linearly decreasing beta (contraction-expansion coefficient)
            beta = self.beta_start - (self.beta_start - self.beta_end) * (iteration / self.max_iter)
            
            # Compute Mean Best (mbest) position: average of all pbest positions
            mbest = np.mean(pbest_positions, axis=0)
            
            for i in range(self.swarm_size):
                # Update for each particle dimension
                phi = np.random.uniform(0.0, 1.0, self.N)
                u = np.random.uniform(0.0, 1.0, self.N)
                
                # Local attractor p_i
                p = phi * pbest_positions[i] + (1.0 - phi) * gbest_position
                
                # Quantum position update:
                # x = p +/- beta * |mbest - x| * ln(1/u)
                sign = np.random.choice([-1.0, 1.0], size=self.N)
                
                # Update positions
                positions[i] = p + sign * beta * np.abs(mbest - positions[i]) * np.log(1.0 / u)
                
                # Clamp positions to [0, 1]
                positions[i] = np.clip(positions[i], 0.0, 1.0)
                
                # Evaluate fitness
                fit, routes = self.evaluate_fitness(positions[i])
                
                # Update personal best
                if fit < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit
                    pbest_positions[i] = np.copy(positions[i])
                    pbest_routes[i] = routes
                    
            # Update global best
            best_idx = np.argmin(pbest_fitnesses)
            if pbest_fitnesses[best_idx] < gbest_fitness:
                gbest_fitness = pbest_fitnesses[best_idx]
                gbest_position = np.copy(pbest_positions[best_idx])
                gbest_routes = pbest_routes[best_idx]
                
            convergence_history.append(gbest_fitness)
            
        runtime = time.time() - start_time
        return SolutionResult(gbest_routes, gbest_fitness, convergence_history, runtime)
