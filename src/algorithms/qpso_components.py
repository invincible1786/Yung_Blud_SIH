from typing import Dict, List, Sequence, Tuple

import numpy as np

# ======================================================================
# Part 1: Prins' optimal Split
# ======================================================================


def _greedy_min_routes(tour: List[int], demands: Dict[int, int], vehicle_capacity: float) -> int:
    """
    Minimum number of contiguous segments needed to partition `tour` so that no
    segment's demand exceeds capacity, IGNORING distance.

    For a fixed left-to-right order, greedily filling each segment as full as
    possible before starting the next one is optimal for minimizing segment
    count (a standard interval-partitioning result). This is exactly the naive
    decoder used by src/algorithms/qpso.py -- reused here only to compute a
    route-count *lower bound for this specific tour*, not to decide the routes
    Split ultimately returns.
    """
    if not tour:
        return 0
    count = 1
    load = 0
    for cust in tour:
        d = demands[cust]
        if d > vehicle_capacity:
            raise ValueError(f"Customer {cust} demand {d} exceeds vehicle capacity {vehicle_capacity}")
        if load + d <= vehicle_capacity:
            load += d
        else:
            count += 1
            load = d
    return count


def split_optimal(
    tour: List[int],
    demands: Dict[int, int],
    depot_id: int,
    distance_matrix: np.ndarray,
    node_id_map: Dict[int, int],
    vehicle_capacity: float,
) -> Tuple[List[List[int]], float]:
    """
    Unconstrained optimal Split: the minimum-distance partition of `tour` into
    capacity-feasible contiguous vehicle routes, with no limit on how many
    vehicles are used.

    Builds V[j] = min over feasible i of V[i] + cost(i, j), where arc (i, j)
    represents one vehicle serving tour[i:j] (0-indexed, half-open), and solves
    it via one forward sweep. cost(i, j) is extended incrementally as j grows
    for a fixed i, so the whole procedure is O(N * b), b = max customers a
    single vehicle's capacity allows -- not O(N^2).

    Returns (routes, total_distance); total_distance excludes any fleet
    penalty (see split_fleet_bounded for that).
    """
    n = len(tour)
    if n == 0:
        return [], 0.0

    depot_idx = node_id_map[depot_id]
    INF = float("inf")
    V = [INF] * (n + 1)
    parent = [-1] * (n + 1)
    V[0] = 0.0

    for i in range(n):
        if V[i] == INF:
            continue
        load = 0
        route_cost_acc = 0.0
        for j in range(i + 1, n + 1):
            cust = tour[j - 1]
            load += demands[cust]
            if load > vehicle_capacity:
                break
            cust_idx = node_id_map[cust]
            if j == i + 1:
                route_cost_acc = distance_matrix[depot_idx, cust_idx] + distance_matrix[cust_idx, depot_idx]
            else:
                prev_idx = node_id_map[tour[j - 2]]
                route_cost_acc = (
                    route_cost_acc
                    - distance_matrix[prev_idx, depot_idx]
                    + distance_matrix[prev_idx, cust_idx]
                    + distance_matrix[cust_idx, depot_idx]
                )
            candidate = V[i] + route_cost_acc
            if candidate < V[j]:
                V[j] = candidate
                parent[j] = i

    if V[n] == INF:
        raise RuntimeError("split_optimal: no feasible partition found (a single customer may exceed capacity)")

    routes: List[List[int]] = []
    j = n
    breakpoints = []
    while j > 0:
        i = parent[j]
        breakpoints.append((i, j))
        j = i
    breakpoints.reverse()
    for i, j in breakpoints:
        routes.append([depot_id] + tour[i:j] + [depot_id])

    return routes, float(V[n])


def split_fleet_bounded(
    tour: List[int],
    demands: Dict[int, int],
    depot_id: int,
    distance_matrix: np.ndarray,
    node_id_map: Dict[int, int],
    vehicle_capacity: float,
    max_vehicles: int,
    penalty_weight: float = 1e6,
) -> Tuple[List[List[int]], float, int]:
    """
    Fleet-bounded optimal Split: the minimum-(distance + fleet penalty)
    partition of `tour`, where using more than `max_vehicles` routes costs
    `penalty_weight` per excess vehicle -- matching the penalty convention
    every existing solver in src/algorithms/ uses (see e.g. qpso.py's
    evaluate_fitness). Needed because on this project's n=50/n=100 instances
    the fleet limit is smaller than the bin-packing lower bound on vehicles
    needed, so an unconstrained decoder is not answering the question that
    matters there.

    Search strategy: let k_min = the minimum number of contiguous segments
    this specific tour can be split into at all (via `_greedy_min_routes`,
    O(N)).
    - k_min <= max_vehicles (a zero-penalty partition exists): search every
      route count K in [k_min, max_vehicles] and keep the best distance found.
    - k_min > max_vehicles (this tour cannot be served without breaching the
      fleet limit -- true whenever total demand exceeds
      max_vehicles * vehicle_capacity, independent of algorithm quality): the
      1e6 penalty per vehicle dwarfs any plausible distance saving, so the
      optimal choice is always exactly k_min, and only that single K needs
      solving.

    Either way the number of route-count layers actually computed stays small,
    keeping this close to O(N * b) rather than O(max_vehicles * N * b).

    Returns (routes, total_distance, num_routes); total_distance excludes the
    penalty term -- callers combine
    `total_distance + penalty_weight * max(0, num_routes - max_vehicles)`
    themselves, mirroring how every other solver in this suite reports
    distance and penalty separately.
    """
    n = len(tour)
    if n == 0:
        return [], 0.0, 0

    k_min = _greedy_min_routes(tour, demands, vehicle_capacity)
    k_hi = k_min if k_min > max_vehicles else max(max_vehicles, k_min)

    depot_idx = node_id_map[depot_id]
    INF = float("inf")

    # V[k][j]: min distance serving the first j customers of `tour` using exactly k routes.
    V = [[INF] * (n + 1) for _ in range(k_hi + 1)]
    parent = [[-1] * (n + 1) for _ in range(k_hi + 1)]
    V[0][0] = 0.0

    for k in range(1, k_hi + 1):
        prev_layer = V[k - 1]
        for i in range(n):
            if prev_layer[i] == INF:
                continue
            load = 0
            route_cost_acc = 0.0
            for j in range(i + 1, n + 1):
                cust = tour[j - 1]
                load += demands[cust]
                if load > vehicle_capacity:
                    break
                cust_idx = node_id_map[cust]
                if j == i + 1:
                    route_cost_acc = distance_matrix[depot_idx, cust_idx] + distance_matrix[cust_idx, depot_idx]
                else:
                    prev_idx = node_id_map[tour[j - 2]]
                    route_cost_acc = (
                        route_cost_acc
                        - distance_matrix[prev_idx, depot_idx]
                        + distance_matrix[prev_idx, cust_idx]
                        + distance_matrix[cust_idx, depot_idx]
                    )
                candidate = prev_layer[i] + route_cost_acc
                if candidate < V[k][j]:
                    V[k][j] = candidate
                    parent[k][j] = i

    best_k = None
    best_total = INF
    for k in range(k_min, k_hi + 1):
        if V[k][n] == INF:
            continue
        penalty = penalty_weight * max(0, k - max_vehicles)
        total = V[k][n] + penalty
        if total < best_total:
            best_total = total
            best_k = k

    if best_k is None:
        raise RuntimeError("split_fleet_bounded: no feasible partition found for any route count tried")

    routes: List[List[int]] = []
    k, j = best_k, n
    while k > 0:
        i = parent[k][j]
        routes.append([depot_id] + tour[i:j] + [depot_id])
        j = i
        k -= 1
    routes.reverse()

    return routes, float(V[best_k][n]), best_k


# ======================================================================
# Part 2: local search + Lamarckian write-back
# ======================================================================


def route_cost(route: Sequence[int], distance_matrix: np.ndarray, node_id_map: Dict[int, int]) -> float:
    """Full-precision cost of a single [depot, ..., depot] route, direction-aware."""
    total = 0.0
    for i in range(len(route) - 1):
        u = node_id_map[route[i]]
        v = node_id_map[route[i + 1]]
        total += distance_matrix[u, v]
    return float(total)


def two_opt(
    route: List[int],
    distance_matrix: np.ndarray,
    node_id_map: Dict[int, int],
    max_passes: int = 20,
) -> Tuple[List[int], float]:
    """
    First-improvement 2-opt on a single route (depot fixed at both ends). Every
    candidate reversal is scored by fully recomputing the route's cost (see
    module docstring for why the O(1) symmetric delta cannot be used here).
    Route lengths in this project's instances are small, so this stays cheap.
    """
    best = list(route)
    best_cost = route_cost(best, distance_matrix, node_id_map)
    n = len(best)
    if n <= 3:
        return best, best_cost

    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                candidate = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                cand_cost = route_cost(candidate, distance_matrix, node_id_map)
                if cand_cost < best_cost - 1e-9:
                    best = candidate
                    best_cost = cand_cost
                    improved = True
    return best, best_cost


def _route_load(route: Sequence[int], demands: Dict[int, int], depot_id: int) -> int:
    return sum(demands[c] for c in route if c != depot_id)


def or_opt(
    routes: List[List[int]],
    demands: Dict[int, int],
    depot_id: int,
    distance_matrix: np.ndarray,
    node_id_map: Dict[int, int],
    vehicle_capacity: float,
    segment_lengths: Sequence[int] = (1, 2, 3),
    max_passes: int = 5,
) -> List[List[int]]:
    """
    Or-opt: relocate contiguous segments of 1-3 customers to a better position,
    either within the same route or into a different route, subject to
    capacity. This is also the mechanism that can shrink the number of routes
    used: if a route's customers are all relocated elsewhere it becomes empty
    and is dropped -- the only way to beat the route count a single Split call
    can produce for one fixed tour is to move customers between routes, which
    is exactly what this operator does.

    Every candidate move is scored by fully recomputing the cost of the (at
    most two) affected routes, correct under the asymmetric matrix exactly
    like `two_opt` above.
    """
    routes = [list(r) for r in routes]

    def cost(r: Sequence[int]) -> float:
        return route_cost(r, distance_matrix, node_id_map)

    def load(r: Sequence[int]) -> int:
        return _route_load(r, demands, depot_id)

    passes = 0
    improved = True
    while improved and passes < max_passes:
        improved = False
        passes += 1

        for src_idx in range(len(routes)):
            src = routes[src_idx]
            src_inner = src[1:-1]
            if not src_inner:
                continue

            move_applied = False
            for seg_len in segment_lengths:
                if seg_len > len(src_inner):
                    continue
                for start in range(len(src_inner) - seg_len + 1):
                    segment = src_inner[start : start + seg_len]
                    seg_demand = sum(demands[c] for c in segment)
                    reduced_inner = src_inner[:start] + src_inner[start + seg_len :]
                    reduced_src = [depot_id] + reduced_inner + [depot_id]
                    src_cost_before = cost(src)
                    src_cost_after_removal = cost(reduced_src)

                    best_delta = -1e-6
                    best_move = None  # (tgt_idx, new_route)

                    for tgt_idx in range(len(routes)):
                        if tgt_idx == src_idx:
                            base_inner = reduced_inner
                            base_load = load(reduced_src)
                        else:
                            base_inner = routes[tgt_idx][1:-1]
                            base_load = load(routes[tgt_idx])

                        if base_load + seg_demand > vehicle_capacity:
                            continue

                        tgt_cost_before = src_cost_before if tgt_idx == src_idx else cost(routes[tgt_idx])

                        for pos in range(len(base_inner) + 1):
                            new_inner = base_inner[:pos] + segment + base_inner[pos:]
                            new_route = [depot_id] + new_inner + [depot_id]
                            new_cost = cost(new_route)

                            if tgt_idx == src_idx:
                                delta = new_cost - src_cost_before
                            else:
                                delta = (src_cost_after_removal + new_cost) - (
                                    src_cost_before + tgt_cost_before
                                )

                            if delta < best_delta:
                                best_delta = delta
                                best_move = (tgt_idx, new_route)

                    if best_move is not None:
                        tgt_idx, new_route = best_move
                        if tgt_idx == src_idx:
                            routes[src_idx] = new_route
                        else:
                            routes[src_idx] = reduced_src
                            routes[tgt_idx] = new_route
                        improved = True
                        move_applied = True
                        break
                if move_applied:
                    break

    # Drop routes left with no customers (depot -> depot only) -- this is how
    # or_opt actually reduces the vehicle count.
    routes = [r for r in routes if len(r) > 2]
    return routes


def _break_ties(sorted_vals: np.ndarray) -> np.ndarray:
    """
    Nudges a sorted array so it is strictly increasing, leaving values that
    were already distinct untouched. Exact ties can only reach here from
    degenerate input (e.g. several key components saturated at the same
    clipped boundary value); argsort cannot distinguish equal floats by rank,
    so without this a tied group's relative order in the rewritten key vector
    would be arbitrary instead of matching `improved_tour`. The nudge is at
    the 1e-9 scale -- far below anything that affects swarm dynamics -- and is
    the standard practical fix for random-key ties.
    """
    out = sorted_vals.astype(np.float64, copy=True)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = np.nextafter(out[i - 1], np.inf)
    return out


def write_back_keys(keys: np.ndarray, customer_ids: List[int], improved_tour: List[int]) -> np.ndarray:
    """
    Lamarckian write-back: remap a particle's own key VALUES onto the visiting
    order that local search found, so the swarm actually learns from local
    search instead of being blind to it.

    Take the particle's key values, sort them ascending (breaking any exact
    ties -- see `_break_ties`), and hand the r-th smallest value to whichever
    customer sits at rank r in `improved_tour`. Two invariants follow directly
    and are what make this safe to use inside the swarm:

      1. multiset(new_keys) ~= multiset(keys) -- the particle's position in the
         swarm's continuous key distribution (its relation to mbest, its
         spread) is unchanged up to tie-breaking noise at the 1e-9 scale; only
         which customer holds which rank moves.
      2. argsort(new_keys) == improved_tour -- decoding the rewritten particle
         reproduces the improved order exactly, including when the input had
         exact ties.
    """
    sorted_keys = _break_ties(np.sort(keys))
    id_to_pos = {cid: pos for pos, cid in enumerate(customer_ids)}
    new_keys = np.empty_like(keys, dtype=np.float64)
    for rank, cust_id in enumerate(improved_tour):
        new_keys[id_to_pos[cust_id]] = sorted_keys[rank]
    return new_keys
