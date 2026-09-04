"""
Uniform re-scoring of any solver's routes (UniformEvaluator), plus an OR-Tools
CVRP reference solver used only to give the study a near-optimal yardstick.
Kept as one file since both exist purely to support src/qpso_lab/study.py's
comparison and neither is large enough to justify a separate module.

Why UniformEvaluator exists
----------------------------
Every solver in src/algorithms/ is supposed to report `total_cost = distance +
1e6 * max(0, num_routes - num_vehicles)` (see ARCHITECTURE.md section 4.1). One
of them doesn't: src/algorithms/genetic_algorithm.py line 181 recomputes
`gbest_fitness = calculate_total_distance(...)` *after* 2-opt, which silently
drops the fleet penalty every other solver includes. Concretely, at n=50 the
stored best GA solution in results/logs/best_routes.json uses 11 routes
against a fleet limit of 8 (3 excess vehicles, i.e. 3e6 of penalty that should
be there) yet reports a cost of 189,552 -- as if it had zero excess vehicles.
GA is not outperforming everyone else by an order of magnitude; it is
reporting a different quantity than everyone else.

We do not edit genetic_algorithm.py (it belongs to another team member and is
frozen). Instead, every solver's `.routes` -- the actual node sequences, which
are trustworthy regardless of what `.total_cost` says -- are re-scored here by
one function, so every algorithm in a comparison is judged by the same
yardstick without a single existing file changing.

This also surfaces a second, structural issue: the n=50 and n=100 instances
need more vehicles than their declared fleet just to cover total demand (a
bin-packing lower bound, independent of routing quality), so every solution to
those instances is necessarily penalized. `Evaluation.lower_bound_vehicles` and
`.instance_fleet_infeasible` make that fact a queryable field instead of a
buried assumption.

Ranking convention
-------------------
Because the penalty (1e6/vehicle) dwarfs any plausible distance difference
between solutions (distances here are 1e4-1e6 m), two solutions that both
incur penalty are not comparable on `penalized_cost` alone in a way that
reflects routing skill -- the one using fewer vehicles will essentially always
look "better" even if its actual route distance is worse. `leaderboard()`
therefore ranks lexicographically by (num_routes, distance); that ordering,
not a bare penalized-cost sort, is the honest one.

Why the OR-Tools reference exists
------------------------------------
Not part of the QPSO ablation and not presented as a competing algorithm --
its only purpose is to give a near-optimal reference cost so an "optimality
gap %" means something, instead of being measured against best-of-our-own-runs
the way src/benchmark/metrics.py currently computes it (a number that shrinks
or grows depending only on who else happened to be in the comparison, not on
distance from a real optimum). Requires the `ortools` package
(requirements-qpso.txt); imported lazily inside the functions that need it so
importing this module doesn't hard-fail if ortools isn't installed and only
the evaluator half is wanted.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.graph_utils import calculate_total_distance

# ======================================================================
# UniformEvaluator
# ======================================================================


@dataclass(frozen=True)
class Evaluation:
    distance: float
    num_routes: int
    vehicles_allowed: int
    excess_vehicles: int
    penalty: float
    penalized_cost: float
    max_route_load: int
    capacity_violations: List[Tuple[int, int]]
    missing_customers: List[int]
    duplicate_customers: List[int]
    malformed_routes: List[int]
    empty_routes: int
    feasible_capacity: bool
    feasible_coverage: bool
    feasible_fleet: bool
    feasible: bool
    lower_bound_vehicles: int
    instance_fleet_infeasible: bool

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["capacity_violations"] = list(self.capacity_violations)
        d["missing_customers"] = list(self.missing_customers)
        d["duplicate_customers"] = list(self.duplicate_customers)
        d["malformed_routes"] = list(self.malformed_routes)
        return d


class UniformEvaluator:
    """Re-scores any solver's `.routes` against one instance with one consistent
    cost function, independent of whatever that solver itself reported."""

    def __init__(
        self,
        instance: Dict[str, Any],
        distance_matrix: np.ndarray,
        node_id_map: Dict[int, int],
        penalty_weight: float = 1e6,
    ):
        self.instance = instance
        self.distance_matrix = distance_matrix
        self.node_id_map = node_id_map
        self.penalty_weight = penalty_weight

        self.depot_id = instance["depot"]["node_id"]
        self.demands = {c["node_id"]: c["demand"] for c in instance["customers"]}
        self.customer_ids = set(self.demands.keys())
        self.vehicle_capacity = instance["vehicle_capacity"]
        self.num_vehicles = instance["num_vehicles"]

        total_demand = sum(self.demands.values())
        self.lower_bound_vehicles = int(np.ceil(total_demand / self.vehicle_capacity)) if self.vehicle_capacity else 0
        self.instance_fleet_infeasible = self.lower_bound_vehicles > self.num_vehicles

    def evaluate(self, routes: List[List[int]]) -> Evaluation:
        malformed = []
        capacity_violations = []
        empty_routes = 0
        visited: List[int] = []
        max_load = 0

        for idx, route in enumerate(routes):
            if len(route) < 2 or route[0] != self.depot_id or route[-1] != self.depot_id:
                malformed.append(idx)
                continue
            if len(route) == 2:
                empty_routes += 1
                continue
            load = 0
            for node in route[1:-1]:
                visited.append(node)
                load += self.demands.get(node, 0)
            max_load = max(max_load, load)
            if load > self.vehicle_capacity:
                capacity_violations.append((idx, load - self.vehicle_capacity))

        seen = set()
        duplicates = []
        for node in visited:
            if node in seen:
                duplicates.append(node)
            else:
                seen.add(node)
        missing = sorted(self.customer_ids - seen)

        real_routes = [r for r in routes if len(r) > 2 and r[0] == self.depot_id and r[-1] == self.depot_id]
        distance = calculate_total_distance(real_routes, self.distance_matrix, self.node_id_map)
        num_routes = len(real_routes)
        excess = max(0, num_routes - self.num_vehicles)
        penalty = self.penalty_weight * excess
        penalized_cost = distance + penalty

        feasible_capacity = len(capacity_violations) == 0
        feasible_coverage = (not missing) and (not duplicates) and (not malformed)
        feasible_fleet = excess == 0
        feasible = feasible_capacity and feasible_coverage and feasible_fleet

        return Evaluation(
            distance=float(distance),
            num_routes=num_routes,
            vehicles_allowed=self.num_vehicles,
            excess_vehicles=excess,
            penalty=float(penalty),
            penalized_cost=float(penalized_cost),
            max_route_load=int(max_load),
            capacity_violations=capacity_violations,
            missing_customers=missing,
            duplicate_customers=duplicates,
            malformed_routes=malformed,
            empty_routes=empty_routes,
            feasible_capacity=feasible_capacity,
            feasible_coverage=feasible_coverage,
            feasible_fleet=feasible_fleet,
            feasible=feasible,
            lower_bound_vehicles=self.lower_bound_vehicles,
            instance_fleet_infeasible=self.instance_fleet_infeasible,
        )

    def evaluate_result(self, result: Any) -> Tuple[Evaluation, float]:
        """Evaluates a SolutionResult and also returns
        `reported_cost - evaluation.penalized_cost` (for the GA at n>=50 this
        is a large negative number: exactly the missing fleet penalty)."""
        evaluation = self.evaluate(result.routes)
        delta = float(result.total_cost) - evaluation.penalized_cost
        return evaluation, delta

    @staticmethod
    def leaderboard(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        """Ranks a list of {"algorithm": ..., "evaluation": Evaluation, ...} rows
        lexicographically by (num_routes, distance) -- see module docstring for
        why penalized_cost alone is not the right sort key once several entries
        carry nonzero penalty."""
        records = []
        for row in rows:
            ev: Evaluation = row["evaluation"]
            rec = {k: v for k, v in row.items() if k != "evaluation"}
            rec.update(
                {
                    "num_routes": ev.num_routes,
                    "distance": ev.distance,
                    "excess_vehicles": ev.excess_vehicles,
                    "penalized_cost": ev.penalized_cost,
                    "feasible": ev.feasible,
                }
            )
            records.append(rec)
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values(by=["num_routes", "distance"]).reset_index(drop=True)
            df.insert(0, "rank", np.arange(1, len(df) + 1))
        return df


# ======================================================================
# OR-Tools reference solver
# ======================================================================


def solve_cvrp_reference(
    instance: Dict[str, Any],
    distance_matrix: np.ndarray,
    node_id_map: Dict[int, int],
    num_vehicles: Optional[int] = None,
    time_limit_seconds: int = 30,
) -> Tuple[Optional[List[List[int]]], Optional[float], bool]:
    """
    Solves the instance's CVRP with Google OR-Tools' routing library.

    `num_vehicles` overrides the instance's declared fleet size. On this
    project's n=50/n=100 instances the declared fleet is smaller than the
    bin-packing lower bound on vehicles needed just to cover demand, so
    passing the declared size through unchanged will correctly come back with
    found=False -- itself an independent confirmation of the structural
    infeasibility documented above. Pass a fleet size at or above the lower
    bound to get a genuine near-optimal reference.

    Returns (routes, distance, found); routes/distance are None when OR-Tools
    could not find any feasible solution within the time limit.
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    depot_id = instance["depot"]["node_id"]
    customers = instance["customers"]
    customer_ids = [c["node_id"] for c in customers]
    demands = {c["node_id"]: c["demand"] for c in customers}
    capacity = instance["vehicle_capacity"]
    fleet = num_vehicles if num_vehicles is not None else instance["num_vehicles"]

    node_ids = [depot_id] + customer_ids
    n = len(node_ids)
    idx_of = {nid: node_id_map[nid] for nid in node_ids}

    # OR-Tools requires integer arc costs; rounding to the nearest metre loses
    # negligible precision on this network's distance scale.
    dist_int = np.array(
        [[int(round(float(distance_matrix[idx_of[u], idx_of[v]]))) for v in node_ids] for u in node_ids],
        dtype=np.int64,
    )
    demand_list = [0] + [demands[c] for c in customer_ids]

    manager = pywrapcp.RoutingIndexManager(n, fleet, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        f = manager.IndexToNode(from_index)
        t = manager.IndexToNode(to_index)
        return int(dist_int[f, t])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return int(demand_list[node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [int(capacity)] * fleet, True, "Capacity")

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_seconds)

    solution = routing.SolveWithParameters(search_parameters)
    if solution is None:
        return None, None, False

    routes: List[List[int]] = []
    for vehicle_id in range(fleet):
        index = routing.Start(vehicle_id)
        route_nodes = []
        while not routing.IsEnd(index):
            route_nodes.append(node_ids[manager.IndexToNode(index)])
            index = solution.Value(routing.NextVar(index))
        route_nodes.append(node_ids[manager.IndexToNode(index)])
        if len(route_nodes) > 2:
            routes.append(route_nodes)

    distance = calculate_total_distance(routes, distance_matrix, node_id_map)
    return routes, float(distance), True


def reference_summary(
    instance: Dict[str, Any],
    distance_matrix: np.ndarray,
    node_id_map: Dict[int, int],
    time_limit_seconds: int = 30,
) -> Dict[str, Any]:
    """
    Runs OR-Tools twice: once at the instance's declared fleet size (expected
    to fail on n50/n100 -- that failure is itself the reference-solver
    confirmation of the bin-packing infeasibility finding) and once at
    max(declared, lower_bound), which gives the actual near-optimal reference
    distance the study's gap percentages are measured against.
    """
    total_demand = sum(c["demand"] for c in instance["customers"])
    capacity = instance["vehicle_capacity"]
    lower_bound_vehicles = int(np.ceil(total_demand / capacity)) if capacity else 0
    declared_fleet = instance["num_vehicles"]

    declared_routes, declared_distance, declared_found = solve_cvrp_reference(
        instance, distance_matrix, node_id_map, num_vehicles=declared_fleet, time_limit_seconds=time_limit_seconds
    )

    relaxed_fleet = max(declared_fleet, lower_bound_vehicles)
    relaxed_routes, relaxed_distance, relaxed_found = solve_cvrp_reference(
        instance, distance_matrix, node_id_map, num_vehicles=relaxed_fleet, time_limit_seconds=time_limit_seconds
    )

    return {
        "lower_bound_vehicles": lower_bound_vehicles,
        "declared_fleet": declared_fleet,
        "declared_feasible": declared_found,
        "declared_routes": declared_routes,
        "declared_distance": declared_distance,
        "relaxed_fleet": relaxed_fleet,
        "relaxed_feasible": relaxed_found,
        "relaxed_routes": relaxed_routes,
        "relaxed_distance": relaxed_distance,
    }
