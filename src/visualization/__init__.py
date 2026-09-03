"""
Visualization package for VRP routing algorithms and road networks.
"""
from .static_map import (
    plot_algorithm_routes,
    load_default_graph,
    load_solution_for_pair,
    generate_all_static_maps,
)
from .interactive_map import (
    build_interactive_route_map,
    generate_all_interactive_maps,
    load_nodes_metadata,
)
from .comparison_grid import (
    plot_comparison_grid,
    generate_all_comparison_grids,
    load_optimality_gaps,
)

__all__ = [
    "plot_algorithm_routes",
    "load_default_graph",
    "load_solution_for_pair",
    "generate_all_static_maps",
    "build_interactive_route_map",
    "generate_all_interactive_maps",
    "load_nodes_metadata",
    "plot_comparison_grid",
    "generate_all_comparison_grids",
    "load_optimality_gaps",
]


