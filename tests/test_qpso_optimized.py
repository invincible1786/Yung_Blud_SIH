import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.algorithms.qpso_components import (
    _greedy_min_routes,
    or_opt,
    route_cost,
    split_fleet_bounded,
    split_optimal,
    two_opt,
    two_opt_star,
    write_back_keys,
)
from src.algorithms.qpso_optimized import QPSOOptimized, _canonical_keys, _ffd_bins
from src.qpso_lab.evaluation import UniformEvaluator
from src.utils.graph_utils import calculate_total_distance, load_instance_resources

INSTANCES_DIR = os.path.join("data", "instances")


# ----------------------------------------------------------------------
# Synthetic asymmetric fixture (small enough to brute force exhaustively)
# ----------------------------------------------------------------------


def _make_synthetic_instance(seed: int = 7, n: int = 6):
    """A tiny hand-rolled, deliberately ASYMMETRIC instance (d(u,v) != d(v,u)),
    identity node_id_map, used to brute-force-verify Split without depending on
    the real (much larger) road network data."""
    rng = np.random.RandomState(seed)
    depot_id = 0
    customer_ids = list(range(1, n + 1))
    size = n + 1
    distance_matrix = rng.uniform(1.0, 20.0, size=(size, size))
    np.fill_diagonal(distance_matrix, 0.0)
    node_id_map = {i: i for i in range(size)}
    demands = {c: int(rng.randint(1, 5)) for c in customer_ids}
    capacity = 8
    return depot_id, customer_ids, demands, capacity, distance_matrix, node_id_map


def _brute_force_route_cost(depot_id, segment, distance_matrix, node_id_map):
    route = [depot_id] + list(segment) + [depot_id]
    return route_cost(route, distance_matrix, node_id_map)


def _brute_force_split(tour, demands, depot_id, distance_matrix, node_id_map, capacity, exact_k=None):
    """Exhaustively enumerates every way to cut `tour` into contiguous,
    capacity-feasible segments (2^(n-1) cut patterns) and returns the minimum
    total distance, optionally restricted to exactly `exact_k` segments."""
    n = len(tour)
    best = None
    for mask in range(1 << (n - 1)):
        cuts = [i for i in range(n - 1) if (mask >> i) & 1]
        if exact_k is not None and len(cuts) != exact_k - 1:
            continue
        segments = []
        start = 0
        for c in cuts:
            segments.append(tour[start : c + 1])
            start = c + 1
        segments.append(tour[start:n])

        feasible = True
        total = 0.0
        for seg in segments:
            if sum(demands[c] for c in seg) > capacity:
                feasible = False
                break
            total += _brute_force_route_cost(depot_id, seg, distance_matrix, node_id_map)
        if feasible and (best is None or total < best):
            best = total
    return best


# ----------------------------------------------------------------------
# Split correctness
# ----------------------------------------------------------------------


def test_split_optimal_matches_bruteforce():
    depot_id, customer_ids, demands, capacity, D, node_id_map = _make_synthetic_instance()
    for perm_seed in range(5):
        rng = np.random.RandomState(perm_seed)
        tour = list(rng.permutation(customer_ids))
        routes, dist = split_optimal(tour, demands, depot_id, D, node_id_map, capacity)
        expected = _brute_force_split(tour, demands, depot_id, D, node_id_map, capacity)
        assert expected is not None, "brute force found no feasible partition"
        assert abs(dist - expected) < 1e-6, f"split_optimal={dist} brute_force={expected} tour={tour}"

        # Reconstructed routes must actually cost what was reported and must
        # cover the tour exactly once, in order.
        recombined = [c for r in routes for c in r if c != depot_id]
        assert recombined == tour
        recomputed = sum(route_cost(r, D, node_id_map) for r in routes)
        assert abs(recomputed - dist) < 1e-6


def test_split_fleet_bounded_zero_penalty_when_feasible():
    depot_id, customer_ids, demands, capacity, D, node_id_map = _make_synthetic_instance()
    tour = list(np.random.RandomState(3).permutation(customer_ids))
    _, unconstrained_dist = split_optimal(tour, demands, depot_id, D, node_id_map, capacity)

    routes, dist, num_routes = split_fleet_bounded(
        tour, demands, depot_id, D, node_id_map, capacity, max_vehicles=len(tour)
    )
    # A generous fleet limit must reproduce the unconstrained optimum exactly.
    assert abs(dist - unconstrained_dist) < 1e-6
    assert num_routes == len(routes)


def test_split_fleet_bounded_matches_bruteforce_at_forced_k():
    depot_id, customer_ids, demands, capacity, D, node_id_map = _make_synthetic_instance()
    tour = list(np.random.RandomState(11).permutation(customer_ids))
    k_min = _greedy_min_routes(tour, demands, capacity)

    # Force max_vehicles below k_min: the DP must fall back to exactly k_min
    # routes (the tour's true minimum), matching brute force at that k.
    routes, dist, num_routes = split_fleet_bounded(
        tour, demands, depot_id, D, node_id_map, capacity, max_vehicles=max(1, k_min - 1), penalty_weight=1e6
    )
    assert num_routes == k_min
    expected = _brute_force_split(tour, demands, depot_id, D, node_id_map, capacity, exact_k=k_min)
    assert expected is not None
    assert abs(dist - expected) < 1e-6


def test_prins_never_worse_than_greedy_decode():
    """Prins' optimal Split must never produce a worse distance than the naive
    greedy decoder (src/algorithms/qpso.py's decode_permutation) for the SAME
    tour, since greedy is just one feasible partition among the ones Split
    searches exhaustively."""
    depot_id, customer_ids, demands, capacity, D, node_id_map = _make_synthetic_instance(seed=21, n=8)
    for perm_seed in range(8):
        tour = list(np.random.RandomState(perm_seed).permutation(customer_ids))

        # Reimplementation of qpso.py's greedy decoder for comparison.
        greedy_routes = []
        current = [depot_id]
        load = 0
        for cust in tour:
            d = demands[cust]
            if load + d <= capacity:
                current.append(cust)
                load += d
            else:
                current.append(depot_id)
                greedy_routes.append(current)
                current = [depot_id, cust]
                load = d
        if len(current) > 1:
            current.append(depot_id)
            greedy_routes.append(current)
        greedy_dist = sum(route_cost(r, D, node_id_map) for r in greedy_routes)

        _, split_dist = split_optimal(tour, demands, depot_id, D, node_id_map, capacity)
        assert split_dist <= greedy_dist + 1e-9


# ----------------------------------------------------------------------
# Local search correctness
# ----------------------------------------------------------------------


def test_two_opt_never_worsens_a_route():
    depot_id, customer_ids, demands, capacity, D, node_id_map = _make_synthetic_instance(seed=42, n=10)
    tour = list(np.random.RandomState(5).permutation(customer_ids))
    route = [depot_id] + tour + [depot_id]
    before = route_cost(route, D, node_id_map)
    improved, after = two_opt(route, D, node_id_map)
    assert after <= before + 1e-9
    assert abs(route_cost(improved, D, node_id_map) - after) < 1e-6
    assert sorted(improved) == sorted(route)  # same multiset of nodes, no loss/duplication
    assert improved[0] == depot_id and improved[-1] == depot_id


def test_or_opt_preserves_coverage_and_capacity_and_never_worsens():
    depot_id, customer_ids, demands, capacity, D, node_id_map = _make_synthetic_instance(seed=99, n=12)
    tour = list(np.random.RandomState(2).permutation(customer_ids))
    routes, _ = split_optimal(tour, demands, depot_id, D, node_id_map, capacity)
    before = sum(route_cost(r, D, node_id_map) for r in routes)

    improved = or_opt(routes, demands, depot_id, D, node_id_map, capacity)
    after = sum(route_cost(r, D, node_id_map) for r in improved)
    assert after <= before + 1e-6

    visited = [c for r in improved for c in r if c != depot_id]
    assert sorted(visited) == sorted(customer_ids)
    for r in improved:
        assert r[0] == depot_id and r[-1] == depot_id
        load = sum(demands[c] for c in r if c != depot_id)
        assert load <= capacity


def test_or_opt_can_eliminate_a_route():
    """Two nearly-empty routes that fit into one should collapse to one route
    -- this is the mechanism by which local search reduces vehicle count below
    what a single Split call on one fixed tour can achieve."""
    depot_id = 0
    node_id_map = {i: i for i in range(4)}
    demands = {1: 2, 2: 2}
    capacity = 10
    # Depot at origin, both customers essentially co-located and cheap to combine.
    D = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.1, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    routes = [[depot_id, 1, depot_id], [depot_id, 2, depot_id]]
    improved = or_opt(routes, demands, depot_id, D, node_id_map, capacity)
    assert len(improved) == 1
    visited = [c for r in improved for c in r if c != depot_id]
    assert sorted(visited) == [1, 2]


def test_two_opt_star_preserves_coverage_and_capacity_and_never_worsens():
    depot_id, customer_ids, demands, capacity, D, node_id_map = _make_synthetic_instance(seed=42, n=6)
    routes = [[depot_id, 1, 2, 3, depot_id], [depot_id, 4, 5, 6, depot_id]]
    before = sum(route_cost(r, D, node_id_map) for r in routes)
    improved = two_opt_star(routes, demands, depot_id, D, node_id_map, capacity)
    after = sum(route_cost(r, D, node_id_map) for r in improved)

    assert after <= before + 1e-6
    visited = [c for r in improved for c in r if c != depot_id]
    assert sorted(visited) == sorted([1, 2, 3, 4, 5, 6])
    for r in improved:
        assert r[0] == depot_id and r[-1] == depot_id
        load = sum(demands[c] for c in r if c != depot_id)
        assert load <= capacity


def test_two_opt_star_cross_improves_synthetic_crossing():
    depot_id = 0
    node_id_map = {i: i for i in range(5)}
    demands = {1: 1, 2: 1, 3: 1, 4: 1}
    capacity = 10
    coords = {0: (0.0, 0.0), 1: (0.0, 1.0), 2: (0.0, 2.0), 3: (2.0, 1.0), 4: (2.0, 2.0)}
    D = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            D[i, j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])

    crossing_routes = [[0, 1, 4, 0], [0, 3, 2, 0]]
    cost_before = sum(route_cost(r, D, node_id_map) for r in crossing_routes)

    improved = two_opt_star(crossing_routes, demands, depot_id, D, node_id_map, capacity)
    cost_after = sum(route_cost(r, D, node_id_map) for r in improved)

    assert cost_after < cost_before - 1e-3
    visited = [c for r in improved for c in r if c != depot_id]
    assert sorted(visited) == [1, 2, 3, 4]


def test_two_opt_star_can_eliminate_a_route():
    depot_id = 0
    node_id_map = {i: i for i in range(4)}
    demands = {1: 1, 2: 1}
    capacity = 10
    D = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.1, 0.0],
            [1.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    routes = [[depot_id, 1, depot_id], [depot_id, 2, depot_id]]
    improved = two_opt_star(routes, demands, depot_id, D, node_id_map, capacity)
    assert len(improved) == 1
    visited = [c for r in improved for c in r if c != depot_id]
    assert sorted(visited) == [1, 2]


# ----------------------------------------------------------------------
# Lamarckian write-back invariants
# ----------------------------------------------------------------------


def test_write_back_keys_preserves_multiset_and_reproduces_order():
    rng = np.random.RandomState(17)
    customer_ids = list(range(100, 110))
    keys = rng.uniform(0.0, 1.0, size=len(customer_ids))
    improved_tour = list(rng.permutation(customer_ids))

    new_keys = write_back_keys(keys, customer_ids, improved_tour)

    assert np.allclose(np.sort(new_keys), np.sort(keys)), "key multiset must be preserved"

    order = np.argsort(new_keys, kind="stable")
    decoded = [customer_ids[i] for i in order]
    assert decoded == improved_tour, "argsort(new_keys) must reproduce the improved tour exactly"


def test_write_back_keys_handles_duplicate_input_keys():
    # Saturated/clipped keys can tie exactly; write-back must still round-trip.
    customer_ids = [1, 2, 3, 4]
    keys = np.array([0.0, 0.0, 1.0, 1.0])
    improved_tour = [3, 1, 4, 2]
    new_keys = write_back_keys(keys, customer_ids, improved_tour)
    assert np.allclose(np.sort(new_keys), np.sort(keys))
    order = np.argsort(new_keys, kind="stable")
    assert [customer_ids[i] for i in order] == improved_tour


# ----------------------------------------------------------------------
# FFD seeding
# ----------------------------------------------------------------------


def test_ffd_seed_reaches_bin_packing_lower_bound_on_real_instances():
    for size in (20, 50, 100):
        path = os.path.join(INSTANCES_DIR, f"instance_n{size}.json")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        with open(path, "r") as f:
            instance = json.load(f)

        customer_ids = [c["node_id"] for c in instance["customers"]]
        demands = {c["node_id"]: c["demand"] for c in instance["customers"]}
        capacity = instance["vehicle_capacity"]
        total_demand = sum(demands.values())
        lower_bound = int(np.ceil(total_demand / capacity))

        bins = _ffd_bins(customer_ids, demands, capacity)
        ffd_tour = [c for b in bins for c in b]
        k_min = _greedy_min_routes(ffd_tour, demands, capacity)

        assert k_min == lower_bound, (
            f"instance_n{size}: FFD-seeded tour needs {k_min} routes, "
            f"bin-packing lower bound is {lower_bound}"
        )
        assert len(bins) == lower_bound

        canonical = _canonical_keys(ffd_tour, customer_ids)
        decoded_order = [customer_ids[i] for i in np.argsort(canonical, kind="stable")]
        assert decoded_order == ffd_tour


# ----------------------------------------------------------------------
# QPSOOptimized: solution invariants across ablation flag combinations
# ----------------------------------------------------------------------


def _load_instance(name: str):
    path = os.path.join(INSTANCES_DIR, f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)


def _assert_valid_solution(instance, result, label):
    depot_id = instance["depot"]["node_id"]
    customer_ids = {c["node_id"] for c in instance["customers"]}
    demands = {c["node_id"]: c["demand"] for c in instance["customers"]}
    capacity = instance["vehicle_capacity"]

    assert result.routes is not None, f"[{label}] solver returned no routes"
    visited = []
    for r in result.routes:
        assert r[0] == depot_id and r[-1] == depot_id, f"[{label}] route must start/end at depot"
        load = sum(demands[n] for n in r if n != depot_id)
        assert load <= capacity, f"[{label}] route exceeds capacity: {load} > {capacity}"
        visited.extend(n for n in r if n != depot_id)

    assert sorted(visited) == sorted(customer_ids), f"[{label}] coverage mismatch"
    assert len(result.convergence_history) > 0, f"[{label}] empty convergence history"
    assert result.runtime_seconds >= 0, f"[{label}] negative runtime"


def test_qpso_optimized_solution_invariants_across_ablation_flags():
    if not os.path.exists(os.path.join(INSTANCES_DIR, "instance_n20.json")):
        print("  [skip] instance_n20.json not found")
        return
    instance = _load_instance("instance_n20")

    configs = {
        "full (V6)": dict(swarm_size=8, max_iter=8),
        "greedy decode, no split (V0-ish)": dict(
            swarm_size=8, max_iter=8, use_split=False, use_local_search=False,
            lamarckian_writeback=False, elitism_restarts=False, seed_mode="random",
        ),
        "unconstrained split (V1-ish)": dict(swarm_size=8, max_iter=8, fleet_bounded=False, use_local_search=False),
        "baldwinian LS (V3-ish)": dict(swarm_size=8, max_iter=8, lamarckian_writeback=False),
        "no restarts": dict(swarm_size=8, max_iter=8, elitism_restarts=False),
        "clip bounds": dict(swarm_size=8, max_iter=8, bounds_mode="clip"),
    }
    for label, cfg in configs.items():
        np.random.seed(123)
        solver = QPSOOptimized(instance, cfg)
        result = solver.solve()
        _assert_valid_solution(instance, result, label)


def test_qpso_optimized_determinism():
    if not os.path.exists(os.path.join(INSTANCES_DIR, "instance_n20.json")):
        print("  [skip] instance_n20.json not found")
        return
    instance = _load_instance("instance_n20")
    cfg = dict(swarm_size=6, max_iter=6)

    np.random.seed(999)
    result_a = QPSOOptimized(instance, cfg).solve()
    np.random.seed(999)
    result_b = QPSOOptimized(instance, cfg).solve()

    assert result_a.total_cost == result_b.total_cost
    assert result_a.routes == result_b.routes


def test_warm_start_reencoding_never_worsens_an_arbitrary_incumbent():
    """
    The guaranteed property warm-starting relies on: re-encoding ANY existing
    solution's flattened customer order as canonical keys and re-decoding it
    through evaluate_fitness (i.e. through split_fleet_bounded) can only match
    or improve on that solution's own cost. This holds unconditionally because
    Split finds the OPTIMAL partition of a fixed tour, and the incumbent's own
    routes are themselves just one (not necessarily optimal) partition of their
    own concatenated tour. This is a deterministic property, unlike comparing
    two independent stochastic multi-iteration solves (which can legitimately
    diverge once extra random draws for warm-start noise shift one run's RNG
    stream away from the other's -- not a meaningful "warm start helps" check).
    """
    if not os.path.exists(os.path.join(INSTANCES_DIR, "instance_n20.json")):
        print("  [skip] instance_n20.json not found")
        return
    instance = _load_instance("instance_n20")
    solver = QPSOOptimized(instance, dict(swarm_size=5, max_iter=5))

    # An arbitrary (deliberately not Split-optimal) incumbent solution.
    tour = list(np.random.RandomState(0).permutation(solver.customer_ids))
    incumbent_routes = solver._decode_greedy(tour)
    incumbent_fitness = solver._score_routes(incumbent_routes)

    incumbent_tour = [c for r in incumbent_routes for c in r if c != solver.depot_id]
    canonical = _canonical_keys(incumbent_tour, solver.customer_ids)
    reevaluated_fitness, _, _ = solver.evaluate_fitness(canonical)

    assert reevaluated_fitness <= incumbent_fitness + 1e-6


def test_warm_start_produces_a_valid_solution():
    """Smoke test for the full warm_start() entry point end-to-end."""
    if not os.path.exists(os.path.join(INSTANCES_DIR, "instance_n20.json")):
        print("  [skip] instance_n20.json not found")
        return
    instance = _load_instance("instance_n20")

    np.random.seed(5)
    cold_result = QPSOOptimized(instance, dict(swarm_size=10, max_iter=10)).solve()

    warm_solver = QPSOOptimized(instance, dict(swarm_size=10, max_iter=10))
    warm_result = warm_solver.warm_start(cold_result.routes, max_iter=10)
    _assert_valid_solution(instance, warm_result, "warm_start")


# ----------------------------------------------------------------------
# Evaluator correctness and the two documented benchmark defects
# ----------------------------------------------------------------------


def test_evaluator_distance_matches_graph_utils():
    if not os.path.exists(os.path.join(INSTANCES_DIR, "instance_n20.json")):
        print("  [skip] instance_n20.json not found")
        return
    instance = _load_instance("instance_n20")
    distance_matrix, node_id_map = load_instance_resources(instance)
    evaluator = UniformEvaluator(instance, distance_matrix, node_id_map)

    depot_id = instance["depot"]["node_id"]
    customers = [c["node_id"] for c in instance["customers"]]
    routes = [[depot_id, customers[0], customers[1], depot_id], [depot_id, customers[2], depot_id]]

    evaluation = evaluator.evaluate(routes)
    expected = calculate_total_distance(routes, distance_matrix, node_id_map)
    assert abs(evaluation.distance - expected) < 1e-6


def test_evaluator_flags_structural_fleet_infeasibility():
    for size, expect_infeasible in ((20, False), (50, True), (100, True)):
        path = os.path.join(INSTANCES_DIR, f"instance_n{size}.json")
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        instance = _load_instance(f"instance_n{size}")
        distance_matrix, node_id_map = load_instance_resources(instance)
        evaluator = UniformEvaluator(instance, distance_matrix, node_id_map)
        assert evaluator.instance_fleet_infeasible == expect_infeasible, (
            f"instance_n{size}: expected instance_fleet_infeasible={expect_infeasible}, "
            f"got {evaluator.instance_fleet_infeasible} (lower_bound={evaluator.lower_bound_vehicles}, "
            f"declared={evaluator.num_vehicles})"
        )


def test_evaluator_exposes_ga_reporting_gap_on_stored_best_routes():
    """Regression guard for the documented defect: genetic_algorithm.py line 181
    recomputes gbest_fitness from distance alone AFTER 2-opt, silently dropping
    the fleet penalty every other solver includes. This reads
    results/logs/best_routes.json READ-ONLY (never written to) and checks that
    re-scoring the GA's own stored routes at n=50/n=100 does not match its
    self-reported cost, by roughly the missing penalty."""
    best_routes_path = os.path.join("results", "logs", "best_routes.json")
    if not os.path.exists(best_routes_path):
        print(f"  [skip] {best_routes_path} not found")
        return

    with open(best_routes_path, "r") as f:
        best_routes = json.load(f)

    for size in (50, 100):
        instance_id = f"instance_n{size}"
        inst_path = os.path.join(INSTANCES_DIR, f"{instance_id}.json")
        if instance_id not in best_routes or "genetic_algorithm" not in best_routes[instance_id]:
            print(f"  [skip] no stored GA result for {instance_id}")
            continue
        if not os.path.exists(inst_path):
            print(f"  [skip] {inst_path} not found")
            continue

        instance = _load_instance(instance_id)
        distance_matrix, node_id_map = load_instance_resources(instance)
        evaluator = UniformEvaluator(instance, distance_matrix, node_id_map)

        entry = best_routes[instance_id]["genetic_algorithm"]
        stored_cost = entry["cost"]
        evaluation = evaluator.evaluate(entry["routes"])

        assert evaluation.excess_vehicles > 0, (
            f"{instance_id}: expected the stored GA solution to exceed the fleet limit "
            f"(it uses {evaluation.num_routes} routes against {evaluation.vehicles_allowed} vehicles)"
        )
        # The stored cost should read as (approximately) distance-only, i.e. it
        # is missing the fleet penalty the evaluator restores.
        assert abs(stored_cost - evaluation.distance) < 1.0, (
            f"{instance_id}: stored GA cost {stored_cost} does not match distance-only "
            f"{evaluation.distance} -- has the frozen genetic_algorithm.py changed?"
        )
        assert evaluation.penalized_cost - stored_cost > 0.9 * evaluation.penalty


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


def run_all():
    tests = [
        test_split_optimal_matches_bruteforce,
        test_split_fleet_bounded_zero_penalty_when_feasible,
        test_split_fleet_bounded_matches_bruteforce_at_forced_k,
        test_prins_never_worse_than_greedy_decode,
        test_two_opt_never_worsens_a_route,
        test_or_opt_preserves_coverage_and_capacity_and_never_worsens,
        test_or_opt_can_eliminate_a_route,
        test_two_opt_star_preserves_coverage_and_capacity_and_never_worsens,
        test_two_opt_star_cross_improves_synthetic_crossing,
        test_two_opt_star_can_eliminate_a_route,
        test_write_back_keys_preserves_multiset_and_reproduces_order,
        test_write_back_keys_handles_duplicate_input_keys,
        test_ffd_seed_reaches_bin_packing_lower_bound_on_real_instances,
        test_qpso_optimized_solution_invariants_across_ablation_flags,
        test_qpso_optimized_determinism,
        test_warm_start_reencoding_never_worsens_an_arbitrary_incumbent,
        test_warm_start_produces_a_valid_solution,
        test_evaluator_distance_matches_graph_utils,
        test_evaluator_flags_structural_fleet_infeasibility,
        test_evaluator_exposes_ga_reporting_gap_on_stored_best_routes,
    ]
    for t in tests:
        print(f"Running {t.__name__} ...")
        t()
        print(f"  PASSED")

    print("\n==========================================")
    print("ALL QPSO-OPTIMIZED TESTS PASSED SUCCESSFULLY!")
    print("==========================================")


if __name__ == "__main__":
    run_all()
