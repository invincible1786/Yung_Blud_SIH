import time
import numpy as np
from typing import Dict, Any, List
from deap import base, creator, tools, algorithms
from src.algorithms.base import RoutingAlgorithm, SolutionResult
from src.utils.graph_utils import load_instance_resources, calculate_total_distance

# Safe initialization of DEAP types to prevent re-declaration errors in multiple runs
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

class GeneticAlgorithm(RoutingAlgorithm):
    """
    Implements a Genetic Algorithm for CVRP using DEAP.
    Individual representation: permutation of customer indices [0, ..., N-1].
    Operators: Order Crossover (OX1), Swap Mutation, Tournament Selection.
    """
    def __init__(self, instance: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(instance, config)
        self.pop_size = config.get("population_size", 50)
        self.cx_pb = config.get("crossover_rate", 0.8)
        self.mut_pb = config.get("mutation_rate", 0.2)
        self.max_gen = config.get("max_generations", 100)
        self.tourn_size = config.get("tournament_size", 3)
        self.apply_2opt = config.get("apply_2opt", True)
        
        # Load resources
        self.distance_matrix, self.node_id_map = load_instance_resources(self.instance)
        
        # Parse VRP details
        self.customers = self.instance["customers"]
        self.customer_ids = [c["node_id"] for c in self.customers]
        self.demands = {c["node_id"]: c["demand"] for c in self.customers}
        self.vehicle_capacity = self.instance["vehicle_capacity"]
        self.num_vehicles = self.instance["num_vehicles"]
        self.N = len(self.customers)

        # Setup DEAP Toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("indices", lambda: list(np.random.permutation(self.N)))
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.indices)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Register genetic operators
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tourn_size)
        self.toolbox.register("evaluate", self._evaluate_individual)

    def _get_distance(self, u: int, v: int) -> float:
        u_idx = self.node_id_map[u]
        v_idx = self.node_id_map[v]
        return float(self.distance_matrix[u_idx, v_idx])

    def decode_permutation(self, permutation: List[int]) -> List[List[int]]:
        """Splits a customer index permutation into VRP routes."""
        routes = []
        current_route = [self.depot_id]
        current_load = 0
        
        for idx in permutation:
            cust = self.customer_ids[idx]
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

    def _evaluate_individual(self, individual: List[int]) -> tuple:
        """Calculates distance cost + vehicle constraint penalty of individual."""
        routes = self.decode_permutation(individual)
        distance = calculate_total_distance(routes, self.distance_matrix, self.node_id_map)
        
        num_routes = len(routes)
        penalty = 0.0
        if num_routes > self.num_vehicles:
            penalty = 1e6 * (num_routes - self.num_vehicles)
            
        return (distance + penalty,)

    def _run_2opt(self, route: List[int]) -> List[int]:
        """Refines a single route using 2-opt local search."""
        if len(route) <= 3:
            return route
            
        best_route = list(route)
        best_dist = sum(self._get_distance(best_route[i], best_route[i+1]) for i in range(len(best_route)-1))
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    # Swap the edge
                    new_route = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    new_dist = sum(self._get_distance(new_route[k], new_route[k+1]) for k in range(len(new_route)-1))
                    
                    if new_dist < best_dist:
                        best_dist = new_dist
                        best_route = new_route
                        improved = True
                        
        return best_route

    def solve(self) -> SolutionResult:
        start_time = time.time()
        
        # Initialize population
        pop = self.toolbox.population(n=self.pop_size)
        
        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        gbest_ind = tools.selBest(pop, 1)[0]
        gbest_fitness = gbest_ind.fitness.values[0]
        
        convergence_history = [gbest_fitness]
        
        # Main Evolution Loop
        for gen in range(1, self.max_gen):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.uniform(0, 1) < self.cx_pb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.uniform(0, 1) < self.mut_pb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluate offspring with invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            # Replace population with offspring
            pop[:] = offspring
            
            # Record current generation's best
            current_best = tools.selBest(pop, 1)[0]
            current_best_fit = current_best.fitness.values[0]
            
            if current_best_fit < gbest_fitness:
                gbest_fitness = current_best_fit
                gbest_ind = self.toolbox.clone(current_best)
                
            convergence_history.append(gbest_fitness)
            
        # Decode best permutation to final routes
        best_routes = self.decode_permutation(gbest_ind)
        
        # Apply 2-opt local search refinement to best routes if requested
        if self.apply_2opt:
            refined_routes = []
            for r in best_routes:
                refined_routes.append(self._run_2opt(r))
            best_routes = refined_routes
            gbest_fitness = calculate_total_distance(best_routes, self.distance_matrix, self.node_id_map)
            
            # Update last history value
            convergence_history[-1] = gbest_fitness

        runtime = time.time() - start_time
        return SolutionResult(best_routes, gbest_fitness, convergence_history, runtime)
