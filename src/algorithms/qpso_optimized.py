import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.algorithms.base import RoutingAlgorithm, SolutionResult
from src.algorithms.qpso_components import (
    or_opt,
    split_fleet_bounded,
    split_optimal,
    two_opt,
    write_back_keys,
)
from src.utils.graph_utils import calculate_total_distance, load_instance_resources


def _reflect_bounds(x: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """
    Reflects values back into [lo, hi] instead of clipping them onto the
    boundary. Hard clipping (as in the frozen qpso.py) sends every particle
    component that overshoots to exactly `lo` or `hi`; with enough particles
    and iterations many components saturate at the same value, and
    np.argsort's tie-breaking -- not the search -- then decides those
    customers' relative order. Reflection keeps the same [lo, hi] support
    without creating that point mass.
    """
    span = hi - lo
    if span <= 0:
        return np.clip(x, lo, hi)
    y = np.mod(x - lo, 2 * span)
    y = np.where(y > span, 2 * span - y, y)
    return y + lo


def _ffd_bins(customer_ids: List[int], demands: Dict[int, int], capacity: float) -> List[List[int]]:
    """First-fit-decreasing bin packing over customer demands."""
    order = sorted(customer_ids, key=lambda c: demands[c], reverse=True)
    bins: List[List[Any]] = []  # each entry: [load, [customer_ids]]
    for c in order:
        d = demands[c]
        for b in bins:
            if b[0] + d <= capacity:
                b[0] += d
                b[1].append(c)
                break
        else:
            bins.append([d, [c]])
    return [b[1] for b in bins]


def _canonical_keys(tour: List[int], customer_ids: List[int]) -> np.ndarray:
    """
    Evenly spaced ascending key values (already sorted), assigned to `tour`'s
    visiting order via the same rank-remap `write_back_keys` uses for local
    search results -- so argsort of the returned vector reproduces `tour`
    exactly.
    """
    n = len(tour)
    base = (np.arange(n) + 0.5) / n
    return write_back_keys(base, customer_ids, tour)


class QPSOOptimized(RoutingAlgorithm):
    """
    Optimized QPSO for CVRP. See module docstring for the list of enhancements;
    each is independently switchable via `config` for ablation studies.
    """

    def __init__(self, instance: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(instance, config)
        cfg = config or {}

        # Core swarm parameters (defaults match the frozen suite's swarm/iteration
        # budget so comparisons are like-for-like).
        self.swarm_size = cfg.get("swarm_size", 30)
        self.max_iter = cfg.get("max_iter", 100)
        self.alpha_start = cfg.get("alpha_start", cfg.get("beta_start", 1.0))
        self.alpha_end = cfg.get("alpha_end", cfg.get("beta_end", 0.4))
        self.penalty_weight = cfg.get("penalty_weight", 1e6)

        # Ablation switches.
        self.use_split = cfg.get("use_split", True)
        self.fleet_bounded = cfg.get("fleet_bounded", True)
        self.use_local_search = cfg.get("use_local_search", True)
        self.lamarckian_writeback = cfg.get("lamarckian_writeback", True)
        self.elitism_restarts = cfg.get("elitism_restarts", True)
        self.bounds_mode = cfg.get("bounds_mode", "reflect")  # "reflect" | "clip"
        self.seed_mode = cfg.get("seed_mode", "ffd")  # "ffd" | "random"

        # Local search scheduling.
        self.ls_interval = cfg.get("ls_interval", 5)
        self.ls_fraction = cfg.get("ls_fraction", 0.2)

        # Stagnation handling.
        self.stagnation_limit = cfg.get("stagnation_limit", 15)
        self.restart_fraction = cfg.get("restart_fraction", 0.3)
        self.reheat_interval = cfg.get("reheat_interval", 5)
        self.reheat_factor = cfg.get("reheat_factor", 1.5)

        # Warm start (dynamic re-optimization hook).
        self.warm_start_routes = cfg.get("warm_start_routes", None)
        self.warm_start_fraction = cfg.get("warm_start_fraction", 0.3)
        self.warm_start_noise = cfg.get("warm_start_noise", 0.05)

        self.final_polish = cfg.get("final_polish", True)

        # Resources.
        self.distance_matrix, self.node_id_map = load_instance_resources(self.instance)

        self.customers = self.instance["customers"]
        self.customer_ids = [c["node_id"] for c in self.customers]
        self.demands = {c["node_id"]: c["demand"] for c in self.customers}
        self.vehicle_capacity = self.instance["vehicle_capacity"]
        self.num_vehicles = self.instance["num_vehicles"]
        self.N = len(self.customers)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _decode_greedy(self, tour: List[int]) -> List[List[int]]:
        """Baseline-equivalent greedy decoder, retained only as an ablation fallback
        (config: use_split=False) so V0 can reproduce src/algorithms/qpso.py."""
        routes: List[List[int]] = []
        current_route = [self.depot_id]
        current_load = 0
        for cust in tour:
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

    def evaluate_fitness(self, position: np.ndarray) -> Tuple[float, List[List[int]], List[int]]:
        """Decodes a position vector and returns (penalized_fitness, routes, tour)."""
        sorted_indices = np.argsort(position, kind="stable")
        tour = [self.customer_ids[idx] for idx in sorted_indices]

        if not self.use_split:
            routes = self._decode_greedy(tour)
            distance = calculate_total_distance(routes, self.distance_matrix, self.node_id_map)
            num_routes = len(routes)
        elif self.fleet_bounded:
            routes, distance, num_routes = split_fleet_bounded(
                tour,
                self.demands,
                self.depot_id,
                self.distance_matrix,
                self.node_id_map,
                self.vehicle_capacity,
                self.num_vehicles,
                penalty_weight=self.penalty_weight,
            )
        else:
            routes, distance = split_optimal(
                tour, self.demands, self.depot_id, self.distance_matrix, self.node_id_map, self.vehicle_capacity
            )
            num_routes = len(routes)

        penalty = self.penalty_weight * max(0, num_routes - self.num_vehicles)
        fitness = distance + penalty
        return fitness, routes, tour

    # ------------------------------------------------------------------
    # Local search / Lamarckian education
    # ------------------------------------------------------------------

    def _educate(self, routes: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """Applies 2-opt + Or-opt to a decoded solution; returns the improved
        routes and the corresponding flattened customer tour."""
        improved = [two_opt(r, self.distance_matrix, self.node_id_map)[0] for r in routes]
        improved = or_opt(
            improved,
            self.demands,
            self.depot_id,
            self.distance_matrix,
            self.node_id_map,
            self.vehicle_capacity,
        )
        improved_tour = [c for r in improved for c in r if c != self.depot_id]
        return improved, improved_tour

    def _score_routes(self, routes: List[List[int]]) -> float:
        distance = calculate_total_distance(routes, self.distance_matrix, self.node_id_map)
        penalty = self.penalty_weight * max(0, len(routes) - self.num_vehicles)
        return distance + penalty

    # ------------------------------------------------------------------
    # Swarm initialization
    # ------------------------------------------------------------------

    def _init_positions(self) -> np.ndarray:
        positions = np.random.uniform(0.0, 1.0, (self.swarm_size, self.N))
        if self.seed_mode == "ffd" and self.N > 0:
            bins = _ffd_bins(self.customer_ids, self.demands, self.vehicle_capacity)
            ffd_tour = [c for b in bins for c in b]
            positions[0] = _canonical_keys(ffd_tour, self.customer_ids)
        if self.warm_start_routes:
            positions = self._apply_warm_start(positions)
        return positions

    def _apply_warm_start(self, positions: np.ndarray) -> np.ndarray:
        """Seeds a fraction of the swarm from an incumbent solution plus Laplace
        noise, for dynamic re-optimization when e.g. live travel times change --
        supports a warm-started re-solve instead of starting from scratch."""
        incumbent_tour = [c for r in self.warm_start_routes for c in r if c != self.depot_id]
        if set(incumbent_tour) != set(self.customer_ids):
            # Incumbent doesn't match this instance's customer set; ignore it
            # rather than producing an invalid seed.
            return positions
        incumbent_keys = _canonical_keys(incumbent_tour, self.customer_ids)
        warm_count = max(1, int(self.swarm_size * self.warm_start_fraction))
        for i in range(warm_count):
            noise = np.random.laplace(0.0, self.warm_start_noise, self.N)
            positions[i] = _reflect_bounds(incumbent_keys + noise)
        return positions

    def warm_start(self, incumbent_routes: List[List[int]], **overrides: Any) -> SolutionResult:
        """Convenience entry point: re-solve this instance seeded from
        `incumbent_routes` (e.g. the previous solve's output) instead of a cold
        random start. Any keyword overrides temporarily replace this solver's
        current config values for the duration of the call."""
        original = {k: getattr(self, k) for k in overrides}
        self.warm_start_routes = incumbent_routes
        for k, v in overrides.items():
            setattr(self, k, v)
        try:
            return self.solve()
        finally:
            self.warm_start_routes = None
            for k, v in original.items():
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def solve(self) -> SolutionResult:
        start_time = time.time()

        positions = self._init_positions()

        pbest_positions = np.copy(positions)
        pbest_fitnesses = np.zeros(self.swarm_size)
        pbest_routes: List[Optional[List[List[int]]]] = [None] * self.swarm_size

        for i in range(self.swarm_size):
            pbest_fitnesses[i], pbest_routes[i], _ = self.evaluate_fitness(positions[i])

        gbest_idx = int(np.argmin(pbest_fitnesses))
        gbest_position = np.copy(pbest_positions[gbest_idx])
        gbest_fitness = float(pbest_fitnesses[gbest_idx])
        gbest_routes = pbest_routes[gbest_idx]

        convergence_history: List[float] = []
        stagnation_counter = 0

        for iteration in range(self.max_iter):
            base_alpha = self.alpha_start - (self.alpha_start - self.alpha_end) * (iteration / max(1, self.max_iter))
            if stagnation_counter > 0 and stagnation_counter % self.reheat_interval == 0:
                alpha = min(self.alpha_start, base_alpha * self.reheat_factor)
            else:
                alpha = base_alpha

            mbest = np.mean(pbest_positions, axis=0)

            for i in range(self.swarm_size):
                phi = np.random.uniform(0.0, 1.0, self.N)
                u = np.maximum(np.random.uniform(0.0, 1.0, self.N), 1e-12)
                sign = np.random.choice([-1.0, 1.0], size=self.N)

                p = phi * pbest_positions[i] + (1.0 - phi) * gbest_position
                new_position = p + sign * alpha * np.abs(mbest - positions[i]) * np.log(1.0 / u)

                if self.bounds_mode == "reflect":
                    new_position = _reflect_bounds(new_position)
                else:
                    new_position = np.clip(new_position, 0.0, 1.0)

                positions[i] = new_position

                fit, routes, _ = self.evaluate_fitness(positions[i])
                if fit < pbest_fitnesses[i]:
                    pbest_fitnesses[i] = fit
                    pbest_positions[i] = np.copy(positions[i])
                    pbest_routes[i] = routes

            # Periodic Lamarckian/Baldwinian local search on the elite fraction
            # of the swarm's personal bests.
            if self.use_local_search and (iteration % self.ls_interval == 0):
                elite_count = max(1, int(self.swarm_size * self.ls_fraction))
                elite_idx = np.argsort(pbest_fitnesses)[:elite_count]
                for idx in elite_idx:
                    current_routes = pbest_routes[idx]
                    if current_routes is None:
                        continue
                    improved_routes, improved_tour = self._educate(current_routes)
                    improved_fitness = self._score_routes(improved_routes)

                    if improved_fitness < pbest_fitnesses[idx] - 1e-9:
                        pbest_fitnesses[idx] = improved_fitness
                        pbest_routes[idx] = improved_routes

                        if self.lamarckian_writeback:
                            new_keys = write_back_keys(pbest_positions[idx], self.customer_ids, improved_tour)
                            pbest_positions[idx] = new_keys
                            positions[idx] = np.copy(new_keys)
                        # Baldwinian mode (lamarckian_writeback=False): the improved
                        # fitness/routes are kept for selection purposes (so gbest can
                        # still improve), but pbest_positions[idx] and positions[idx]
                        # are deliberately left unchanged -- the swarm's genotype does
                        # not learn the improvement, only its recorded fitness does.
                        # This is the V3-vs-V4 ablation contrast.

                        if improved_fitness < gbest_fitness:
                            gbest_fitness = improved_fitness
                            gbest_routes = improved_routes
                            if self.lamarckian_writeback:
                                gbest_position = np.copy(pbest_positions[idx])
                            stagnation_counter = 0

            best_idx = int(np.argmin(pbest_fitnesses))
            if pbest_fitnesses[best_idx] < gbest_fitness - 1e-9:
                gbest_fitness = float(pbest_fitnesses[best_idx])
                gbest_position = np.copy(pbest_positions[best_idx])
                gbest_routes = pbest_routes[best_idx]
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if self.elitism_restarts and stagnation_counter >= self.stagnation_limit:
                restart_count = max(1, int(self.swarm_size * self.restart_fraction))
                worst_idx = np.argsort(pbest_fitnesses)[-restart_count:]
                for idx in worst_idx:
                    positions[idx] = np.random.uniform(0.0, 1.0, self.N)
                stagnation_counter = 0

            convergence_history.append(gbest_fitness)

        if self.final_polish and gbest_routes is not None:
            polished_routes, polished_tour = self._educate(gbest_routes)
            polished_fitness = self._score_routes(polished_routes)
            if polished_fitness < gbest_fitness:
                gbest_fitness = polished_fitness
                gbest_routes = polished_routes
                convergence_history[-1] = gbest_fitness

        runtime = time.time() - start_time
        return SolutionResult(gbest_routes, gbest_fitness, convergence_history, runtime)
