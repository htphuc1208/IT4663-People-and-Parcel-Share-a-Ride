"""
ortools_solver.py - Optional Google OR-Tools solver for SARP min-max instances.

This module models the current ProblemData as an OR-Tools routing problem and
converts the resulting routes back to the repository's Solution1D encoding.
The dependency is intentionally imported lazily, so GA/Tabu/ALNS keep working
without installing OR-Tools.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Tuple

from .encoding_and_read import ProblemData, Solution1D
from .fitness import evaluate


@dataclass
class ORToolsConfig:
    """Configuration for the optional OR-Tools routing solver."""

    time_limit_seconds: float = 30.0
    first_solution_strategy: str = "PATH_CHEAPEST_ARC"
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH"


def run_ortools(
    problem: ProblemData,
    config: ORToolsConfig | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Solve one instance with Google OR-Tools and return a Solution1D.

    Passenger drop-offs are explicit in the OR-Tools routing model, with a
    direct ``pickup -> dropoff`` constraint. They are removed again when the
    route is converted back to the repository's chromosome representation.
    """
    if config is None:
        config = ORToolsConfig()

    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise RuntimeError(
            "Google OR-Tools is not installed. Install the optional dependency "
            "with: pip install ortools"
        ) from exc

    node_count = len(problem.distance_matrix)
    if node_count == 0:
        raise ValueError("Problem has an empty distance matrix")

    manager = pywrapcp.RoutingIndexManager(node_count, problem.K, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(round(problem.dist(from_node, to_node)))

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    max_arc = max(int(round(value)) for row in problem.distance_matrix for value in row)
    max_route_distance = max(1, max_arc * max(1, node_count + 1))
    routing.AddDimension(
        distance_callback_index,
        0,
        max_route_distance,
        True,
        "Distance",
    )
    distance_dimension = routing.GetDimensionOrDie("Distance")
    distance_dimension.SetGlobalSpanCostCoefficient(max_route_distance + 1)

    def demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return _parcel_demand_delta(problem, node)

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [problem.capacity_for_vehicle(vehicle) for vehicle in range(problem.K)],
        True,
        "ParcelCapacity",
    )

    solver = routing.solver()

    for passenger_id in range(1, problem.N + 1):
        pickup_node = passenger_id
        dropoff_node = passenger_id + problem.N + problem.M
        pickup_index = manager.NodeToIndex(pickup_node)
        dropoff_index = manager.NodeToIndex(dropoff_node)
        solver.Add(routing.NextVar(pickup_index) == dropoff_index)

    for parcel_id in range(1, problem.M + 1):
        pickup_node = problem.N + parcel_id
        dropoff_node = 2 * problem.N + problem.M + parcel_id
        pickup_index = manager.NodeToIndex(pickup_node)
        dropoff_index = manager.NodeToIndex(dropoff_node)
        routing.AddPickupAndDelivery(pickup_index, dropoff_index)
        solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(dropoff_index))
        solver.Add(
            distance_dimension.CumulVar(pickup_index)
            <= distance_dimension.CumulVar(dropoff_index)
        )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = _enum_value(
        routing_enums_pb2.FirstSolutionStrategy,
        config.first_solution_strategy,
        "first_solution_strategy",
    )
    search_parameters.local_search_metaheuristic = _enum_value(
        routing_enums_pb2.LocalSearchMetaheuristic,
        config.local_search_metaheuristic,
        "local_search_metaheuristic",
    )
    search_parameters.time_limit.FromSeconds(_time_limit_seconds(config.time_limit_seconds))

    assignment = routing.SolveWithParameters(search_parameters)
    if assignment is None:
        raise RuntimeError("OR-Tools did not find a feasible routing solution")

    routes = _extract_routes(problem, routing, manager, assignment)
    solution = _routes_to_solution(problem, routes)
    fitness = evaluate(solution)
    return solution, fitness, [fitness]


def _parcel_demand_delta(problem: ProblemData, node: int) -> int:
    if problem.N < node <= problem.N + problem.M:
        parcel_index = node - problem.N - 1
        return problem.parcels[parcel_index].demand
    if 2 * problem.N + problem.M < node <= 2 * problem.N + 2 * problem.M:
        parcel_index = node - (2 * problem.N + problem.M) - 1
        return -problem.parcels[parcel_index].demand
    return 0


def _extract_routes(
    problem: ProblemData,
    routing: Any,
    manager: Any,
    assignment: Any,
) -> List[List[int]]:
    routes: List[List[int]] = []
    for vehicle_id in range(problem.K):
        index = routing.Start(vehicle_id)
        route: List[int] = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = assignment.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        routes.append(route)
    return routes


def _routes_to_solution(problem: ProblemData, routes: List[List[int]]) -> Solution1D:
    chromosome: List[int] = []
    passenger_dropoffs = set(
        range(problem.N + problem.M + 1, 2 * problem.N + problem.M + 1)
    )

    for vehicle_id, route in enumerate(routes):
        interior = route[1:-1]
        index = 0
        while index < len(interior):
            node = interior[index]
            if 1 <= node <= problem.N:
                expected_dropoff = node + problem.N + problem.M
                next_node = interior[index + 1] if index + 1 < len(interior) else None
                if next_node != expected_dropoff:
                    raise RuntimeError(
                        "OR-Tools returned a passenger route that is not direct: "
                        f"pickup={node}, next={next_node}, expected={expected_dropoff}"
                    )
                chromosome.append(node)
                index += 2
                continue
            if node in passenger_dropoffs:
                raise RuntimeError(
                    "OR-Tools returned a passenger drop-off without its direct pickup: "
                    f"node={node}"
                )
            chromosome.append(node)
            index += 1

        if vehicle_id < problem.K - 1:
            chromosome.append(0)

    solution = Solution1D(chromosome=chromosome, problem=problem)
    is_valid, message = solution.validate()
    if not is_valid:
        raise RuntimeError(f"OR-Tools produced an invalid chromosome: {message}")
    return solution


def _enum_value(enum_container: Any, name: str, field_name: str) -> int:
    try:
        return getattr(enum_container, name)
    except AttributeError as exc:
        raise ValueError(f"Unknown OR-Tools {field_name}: {name}") from exc


def _time_limit_seconds(value: float) -> int:
    if math.isinf(value):
        return 30
    return max(1, int(math.ceil(value)))
