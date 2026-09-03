import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.algorithms.nn_clarke_wright import NearestNeighborClarkeWright
from src.algorithms.standard_pso import StandardPSO
from src.algorithms.qpso import QPSO
from src.algorithms.genetic_algorithm import GeneticAlgorithm
from src.algorithms.aco_mmas import ACOMMAS
from src.utils.graph_utils import load_instance_resources

def run_sanity_tests():
    instances_dir = os.path.join("data", "instances")
    instance_file = os.path.join(instances_dir, "instance_n20.json")

    if not os.path.exists(instance_file):
        print(f"Skipping tests: instance file {instance_file} not found. Build instances first.")
        return

    print(f"Loading test instance: {instance_file}")
    with open(instance_file, "r") as f:
        instance = json.load(f)

    depot_id = instance["depot"]["node_id"]
    customers = instance["customers"]
    customer_ids = {c["node_id"] for c in customers}
    demands = {c["node_id"]: c["demand"] for c in customers}
    capacity = instance["vehicle_capacity"]

    # Load matrix to verify mapping
    distance_matrix, node_id_map = load_instance_resources(instance)

    # Initialize all solvers with a fast config
    solvers = {
        "Nearest Neighbor": NearestNeighborClarkeWright(instance, {"method": "nn", "max_iter": 5}),
        "Clarke-Wright": NearestNeighborClarkeWright(instance, {"method": "clarke_wright", "max_iter": 5}),
        "Standard PSO": StandardPSO(instance, {"swarm_size": 5, "max_iter": 5}),
        "QPSO": QPSO(instance, {"swarm_size": 5, "max_iter": 5}),
        "Genetic Algorithm": GeneticAlgorithm(instance, {"population_size": 5, "max_generations": 5, "apply_2opt": True}),
        "Ant Colony (MMAS)": ACOMMAS(instance, {"num_ants": 5, "max_iter": 5})
    }

    for name, solver in solvers.items():
        print(f"\nTesting algorithm: {name}")
        res = solver.solve()
        
        # Test 1: Routes structure check
        routes = res.routes
        assert isinstance(routes, list), f"[{name}] Routes should be a list"
        
        visited_customers = []
        for r in routes:
            assert isinstance(r, list), f"[{name}] Individual route should be a list"
            assert r[0] == depot_id, f"[{name}] Route must start at depot"
            assert r[-1] == depot_id, f"[{name}] Route must end at depot"
            
            # Check capacity constraint for this route
            route_load = sum(demands[node] for node in r if node != depot_id)
            assert route_load <= capacity, f"[{name}] Route exceeds capacity: {route_load} > {capacity}"
            
            # Record visited customers
            for node in r[1:-1]:
                assert node in customer_ids, f"[{name}] Invalid customer node visited: {node}"
                visited_customers.append(node)
                
        # Test 2: Coverage check - all customers visited exactly once
        assert len(visited_customers) == len(customer_ids), f"[{name}] Visited customer count mismatch: {len(visited_customers)} vs {len(customer_ids)}"
        assert set(visited_customers) == customer_ids, f"[{name}] Some customers were missed or duplicated: {set(visited_customers)} vs {customer_ids}"
        
        # Test 3: Convergence logs check
        assert len(res.convergence_history) > 0, f"[{name}] Convergence history is empty"
        assert res.runtime_seconds >= 0, f"[{name}] Solver runtime should be non-negative"
        
        print(f"[{name}] All checks passed successfully! Cost: {res.total_cost:.2f}, Routes count: {len(routes)}, Time: {res.runtime_seconds:.4f}s")

    # Test 4: Static route map visualizer test
    print("\nTesting static route map visualizer (plot_algorithm_routes)...")
    from src.visualization.static_map import plot_algorithm_routes, load_default_graph
    graph = load_default_graph()
    test_out = os.path.join("results", "route_maps", "static", "test_sanity_nn.png")
    nn_solver = solvers["Nearest Neighbor"]
    nn_res = nn_solver.solve()
    plot_algorithm_routes(graph, instance, nn_res, "Nearest Neighbor", test_out)
    assert os.path.exists(test_out), "Static route map was not created"
    assert os.path.getsize(test_out) > 10000, "Static route map file is too small or empty"
    if os.path.exists(test_out):
        os.remove(test_out)
    print("Static route map visualizer passed successfully!")

    # Test 5: Interactive route map visualizer test
    print("\nTesting interactive route map visualizer (build_interactive_route_map)...")
    from src.visualization.interactive_map import build_interactive_route_map
    test_html = os.path.join("results", "route_maps", "interactive", "test_sanity_nn.html")
    build_interactive_route_map(graph, instance, nn_res, "Nearest Neighbor", test_html)
    assert os.path.exists(test_html), "Interactive route map HTML was not created"
    assert os.path.getsize(test_html) > 5000, "Interactive HTML file is too small"
    if os.path.exists(test_html):
        os.remove(test_html)
    print("Interactive route map visualizer passed successfully!")

    # Test 6: Comparison grid visualizer test
    print("\nTesting comparison grid visualizer (plot_comparison_grid)...")
    from src.visualization.comparison_grid import plot_comparison_grid
    test_grid = os.path.join("results", "route_maps", "comparison_grids", "test_sanity_grid.png")
    plot_comparison_grid(instance_id="instance_n20", graph=graph, instance=instance, save_path=test_grid)
    assert os.path.exists(test_grid), "Comparison grid was not created"
    assert os.path.getsize(test_grid) > 50000, "Comparison grid image file is too small"
    if os.path.exists(test_grid):
        os.remove(test_grid)
    print("Comparison grid visualizer passed successfully!")

    print("\n==========================================")
    print("ALL SANITY TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    run_sanity_tests()



