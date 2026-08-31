import time
import numpy as np
from typing import Dict, Any
from src.algorithms.base import RoutingAlgorithm, SolutionResult
from src.utils.graph_utils import load_instance_resources, calculate_total_distance

class NearestNeighborClarkeWright(RoutingAlgorithm):
    """
    Implements Nearest Neighbor and Clarke-Wright Savings constructive heuristics.
    Exposes a config option 'method' ("nn" or "clarke_wright").
    """
    def __init__(self, instance: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(instance, config)
        self.method = config.get("method", "clarke_wright")
        self.max_iter = config.get("max_iter", 100)  # for formatting convergence history
        
        # Load resources
        self.distance_matrix, self.node_id_map = load_instance_resources(self.instance)
        
        # Parse customer demands and coordinate mapping
        self.customers = self.instance["customers"]
        self.demands = {c["node_id"]: c["demand"] for c in self.customers}
        self.vehicle_capacity = self.instance["vehicle_capacity"]
        self.num_vehicles = self.instance["num_vehicles"]

    def _get_distance(self, u: int, v: int) -> float:
        u_idx = self.node_id_map[u]
        v_idx = self.node_id_map[v]
        return float(self.distance_matrix[u_idx, v_idx])

    def solve(self) -> SolutionResult:
        start_time = time.time()
        
        if self.method == "nn":
            routes = self._solve_nearest_neighbor()
        else:
            routes = self._solve_clarke_wright()

        total_cost = calculate_total_distance(routes, self.distance_matrix, self.node_id_map)
        
        # Penalize if it exceeds the number of vehicles
        if len(routes) > self.num_vehicles:
            penalty = 1e6 * (len(routes) - self.num_vehicles)
            total_cost += penalty

        runtime = time.time() - start_time
        
        # Flat history for construction heuristic
        convergence_history = [total_cost] * self.max_iter

        return SolutionResult(routes, total_cost, convergence_history, runtime)

    def _solve_nearest_neighbor(self) -> list:
        unvisited = set(self.demands.keys())
        routes = []
        
        while unvisited:
            route = [self.depot_id]
            current_capacity = self.vehicle_capacity
            current_node = self.depot_id
            
            while unvisited:
                # Find nearest unvisited customer that fits in capacity
                nearest_cust = None
                min_dist = float('inf')
                
                for cust in unvisited:
                    if self.demands[cust] <= current_capacity:
                        d = self._get_distance(current_node, cust)
                        if d < min_dist:
                            min_dist = d
                            nearest_cust = cust
                
                if nearest_cust is not None:
                    route.append(nearest_cust)
                    current_capacity -= self.demands[nearest_cust]
                    unvisited.remove(nearest_cust)
                    current_node = nearest_cust
                else:
                    # No customer fits the capacity, return to depot
                    break
                    
            route.append(self.depot_id)
            routes.append(route)
            
        return routes

    def _solve_clarke_wright(self) -> list:
        # 1. Initialize with single-customer routes: [depot, customer, depot]
        routes = {c["node_id"]: [self.depot_id, c["node_id"], self.depot_id] for c in self.customers}
        
        # 2. Compute savings: S_ij = d(0, i) + d(0, j) - d(i, j)
        savings = []
        c_ids = list(self.demands.keys())
        n_cust = len(c_ids)
        
        for i in range(n_cust):
            for j in range(i + 1, n_cust):
                ci = c_ids[i]
                cj = c_ids[j]
                
                # Asymmetric matrix: take average or symmetric distance for savings estimation
                d_0_i = self._get_distance(self.depot_id, ci)
                d_0_j = self._get_distance(self.depot_id, cj)
                d_i_j = self._get_distance(ci, cj)
                
                s = d_0_i + d_0_j - d_i_j
                savings.append((s, ci, cj))
                
        # Sort savings in descending order
        savings.sort(key=lambda x: x[0], reverse=True)
        
        # Helper to get route load
        def get_route_load(route_list):
            return sum(self.demands[c] for c in route_list if c != self.depot_id)

        # 3. Merge routes greedily
        for s, ci, cj in savings:
            # Find routes containing ci and cj
            route_i = None
            route_j = None
            
            for r_id, r in routes.items():
                if ci in r:
                    route_i = r
                if cj in r:
                    route_j = r
                    
            if route_i is None or route_j is None or route_i == route_j:
                continue
                
            # Verify capacity constraint
            load_i = get_route_load(route_i)
            load_j = get_route_load(route_j)
            if load_i + load_j > self.vehicle_capacity:
                continue
                
            # Check merge condition (both must be adjacent to depot in their respective routes)
            # Route representation: [depot, c1, c2, ..., ck, depot]
            # ci is adjacent to depot in route_i if ci is route_i[1] (first customer) or route_i[-2] (last customer)
            # cj is adjacent to depot in route_j if cj is route_j[1] (first customer) or route_j[-2] (last customer)
            is_ci_first = (route_i[1] == ci)
            is_ci_last = (route_i[-2] == ci)
            is_cj_first = (route_j[1] == cj)
            is_cj_last = (route_j[-2] == cj)
            
            if not ((is_ci_first or is_ci_last) and (is_cj_first or is_cj_last)):
                continue
                
            # Perform merge
            new_route = None
            if is_ci_last and is_cj_first:
                # Merge route_i and route_j: [depot, ..., ci] + [cj, ..., depot]
                new_route = route_i[:-1] + route_j[1:]
            elif is_cj_last and is_ci_first:
                # Merge route_j and route_i: [depot, ..., cj] + [ci, ..., depot]
                new_route = route_j[:-1] + route_i[1:]
            elif is_ci_first and is_cj_first:
                # Reverse route_i and merge: [depot, ci, ..., depot] -> [depot, ..., ci]
                # Then merge with route_j: [depot, ..., ci] + [cj, ..., depot]
                rev_i = [self.depot_id] + list(reversed(route_i[1:-1])) + [self.depot_id]
                new_route = rev_i[:-1] + route_j[1:]
            elif is_ci_last and is_cj_last:
                # Reverse route_j and merge: [depot, ..., ci] + [depot, ..., cj] -> [depot, ..., ci] + [cj, ..., depot]
                rev_j = [self.depot_id] + list(reversed(route_j[1:-1])) + [self.depot_id]
                new_route = route_i[:-1] + rev_j[1:]
                
            if new_route is not None:
                # Update routes
                # Remove old routes
                for k in list(routes.keys()):
                    if routes[k] == route_i or routes[k] == route_j:
                        del routes[k]
                # Add new route keyed by its first customer
                routes[new_route[1]] = new_route
                
        return list(routes.values())
