import os
import json
import numpy as np
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
        assert res.runtime_seconds > 0, f"[{name}] Solver runtime should be positive"
        
        print(f"[{name}] All checks passed successfully! Cost: {res.total_cost:.2f}, Routes count: {len(routes)}, Time: {res.runtime_seconds:.4f}s")

    print("\n==========================================")
    print("ALL SANITY TESTS PASSED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    run_sanity_tests()
