from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SolutionResult:
    """Standardized result structure for all VRP routing algorithms."""
    def __init__(self, routes: List[List[int]], total_cost: float, convergence_history: List[float], runtime_seconds: float):
        # A list of routes, where each route is a list of node IDs starting and ending at the depot
        self.routes = routes
        self.total_cost = total_cost
        # Best cost per iteration (or generation)
        self.convergence_history = convergence_history
        self.runtime_seconds = runtime_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "routes": self.routes,
            "total_cost": self.total_cost,
            "convergence_history": self.convergence_history,
            "runtime_seconds": self.runtime_seconds
        }

class RoutingAlgorithm(ABC):
    """Abstract base class that all algorithms must implement."""
    def __init__(self, instance: Dict[str, Any], config: Dict[str, Any]):
        self.instance = instance
        self.config = config
        self.depot_id = instance["depot"]["node_id"]
        
        # Load distance matrix and index maps if necessary
        # Usually initialized in subclass or helper
        self.distance_matrix = None
        self.node_id_map = None

    @abstractmethod
    def solve(self) -> SolutionResult:
        """Solves the VRP instance and returns a SolutionResult."""
        pass
