"""
Tabu Search for the People and Parcel Share-a-Ride problem.

The search starts from the common initial solution and improves one current
solution at a time. Neighbours are request-aware: a passenger is represented by
its pickup node, while a parcel is handled as a pickup/drop-off pair so parcel
precedence is not broken by the local moves.

Objective: minimize the maximum route cost among all taxis. The cost and all
constraint penalties are computed by fitness.evaluate().
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .encoding_and_read import ProblemData, Solution1D
from .fitness import evaluate

try:
    from .init import init_solution as build_initial_solution
except ImportError:
    from .init import greedy_init as build_initial_solution


logger = logging.getLogger(__name__)

Edge = Tuple[int, int]
MoveInfo = Tuple[Solution1D, Set[Edge], Set[Edge], str]
MoveGenerator = Callable[[Solution1D], Optional[MoveInfo]]


@dataclass
class TabuConfig:
    """Configuration for Adaptive Tabu Search."""

    max_iterations: int = 1000
    neighbourhood_sample_size: int = 50

    tenure_init: int = 10
    tenure_min: int = 4
    tenure_max: int = 50
    tenure_increase: int = 3
    tenure_decrease: int = 1
    stagnation_limit: int = 25

    seed: int | None = None
    time_limit_seconds: float = float("inf")


class TabuList:
    """
    Short-term memory over route segments.

    A route segment is represented by an edge (u, v) in a raw route wrapped by
    depot nodes. After a move, removed edges are made tabu. A future neighbour
    that tries to add any of those edges is forbidden unless aspiration accepts
    it because it improves the global best solution.
    """

    def __init__(self, tenure: int) -> None:
        self.tenure = tenure
        self._expires_at: Dict[Edge, int] = {}

    def add(self, edges: Iterable[Edge], iteration: int) -> None:
        expiry = iteration + self.tenure
        for edge in edges:
            self._expires_at[edge] = expiry
            self._expires_at[(edge[1], edge[0])] = expiry

    def is_tabu(self, added_edges: Iterable[Edge], iteration: int) -> bool:
        self.purge(iteration)
        return any(self._expires_at.get(edge, -1) > iteration for edge in added_edges)

    def purge(self, iteration: int) -> None:
        expired = [edge for edge, expiry in self._expires_at.items() if expiry <= iteration]
        for edge in expired:
            del self._expires_at[edge]


def _flatten_routes(routes: List[List[int]]) -> List[int]:
    chromosome: List[int] = []
    for idx, route in enumerate(routes):
        chromosome.extend(route)
        if idx < len(routes) - 1:
            chromosome.append(0)
    return chromosome


def _route_edges(route: List[int]) -> Set[Edge]:
    wrapped = [0] + route + [0]
    return {(wrapped[i], wrapped[i + 1]) for i in range(len(wrapped) - 1)}


def _all_edges(routes: List[List[int]]) -> Set[Edge]:
    edges: Set[Edge] = set()
    for route in routes:
        edges.update(_route_edges(route))
    return edges


def _make_move(
    solution: Solution1D,
    routes: List[List[int]],
    before_edges: Set[Edge],
    name: str,
) -> MoveInfo:
    neighbour = Solution1D(_flatten_routes(routes), solution.problem)
    after_edges = _all_edges(routes)
    removed_edges = before_edges - after_edges
    added_edges = after_edges - before_edges
    return neighbour, added_edges, removed_edges, name


def _parcel_nodes(problem: ProblemData, parcel_index: int) -> Tuple[int, int]:
    pickup = problem.N + parcel_index
    dropoff = 2 * problem.N + problem.M + parcel_index
    return pickup, dropoff


def _find_parcel(
    routes: List[List[int]],
    problem: ProblemData,
    parcel_index: int,
) -> Optional[Tuple[int, int, int]]:
    pickup, dropoff = _parcel_nodes(problem, parcel_index)
    for route_idx, route in enumerate(routes):
        if pickup in route and dropoff in route:
            pickup_pos = route.index(pickup)
            dropoff_pos = route.index(dropoff)
            if pickup_pos < dropoff_pos:
                return route_idx, pickup_pos, dropoff_pos
    return None


def _passenger_locations(routes: List[List[int]], problem: ProblemData) -> List[Tuple[int, int]]:
    locations: List[Tuple[int, int]] = []
    for route_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            if 1 <= node <= problem.N:
                locations.append((route_idx, pos))
    return locations


def generate_parcel_transfer(solution: Solution1D) -> Optional[MoveInfo]:
    """
    Move one complete parcel pickup/drop-off pair to another taxi.

    Why this operator: the objective is min-max, so moving parcel work away from
    a long route is the most direct way to rebalance the bottleneck taxi. The
    pickup and drop-off are moved together and reinserted in precedence order.
    """

    problem = solution.problem
    if problem.M == 0 or problem.K < 2:
        return None

    routes = [list(route) for route in solution.get_routes()]
    before_edges = _all_edges(routes)

    parcel_ids = list(range(1, problem.M + 1))
    random.shuffle(parcel_ids)

    selected: Optional[Tuple[int, int, int, int]] = None
    for parcel_idx in parcel_ids:
        found = _find_parcel(routes, problem, parcel_idx)
        if found is not None:
            selected = (parcel_idx, found[0], found[1], found[2])
            break

    if selected is None:
        return None

    parcel_idx, src_route, pickup_pos, dropoff_pos = selected
    pickup, dropoff = _parcel_nodes(problem, parcel_idx)
    target_routes = [idx for idx in range(problem.K) if idx != src_route]
    if not target_routes:
        return None

    dst_route = random.choice(target_routes)
    for pos in sorted((pickup_pos, dropoff_pos), reverse=True):
        routes[src_route].pop(pos)

    insert_pickup = random.randint(0, len(routes[dst_route]))
    routes[dst_route].insert(insert_pickup, pickup)
    insert_dropoff = random.randint(insert_pickup + 1, len(routes[dst_route]))
    routes[dst_route].insert(insert_dropoff, dropoff)

    return _make_move(solution, routes, before_edges, "parcel_transfer")


def generate_parcel_swap(solution: Solution1D) -> Optional[MoveInfo]:
    """
    Swap the positions of two parcel requests.

    Why this operator: it changes parcel sequencing while keeping the number
    of parcel jobs on each selected taxi stable. This is useful when the route
    is feasible but the local order creates long detours.
    """

    problem = solution.problem
    if problem.M < 2:
        return None

    routes = [list(route) for route in solution.get_routes()]
    before_edges = _all_edges(routes)
    parcel_ids = list(range(1, problem.M + 1))
    random.shuffle(parcel_ids)

    found: List[Tuple[int, int, int, int]] = []
    for parcel_idx in parcel_ids:
        loc = _find_parcel(routes, problem, parcel_idx)
        if loc is not None:
            found.append((parcel_idx, loc[0], loc[1], loc[2]))
            if len(found) == 2:
                break

    if len(found) < 2:
        return None

    first, second = found
    p1, r1, p1_pos, d1_pos = first
    p2, r2, p2_pos, d2_pos = second
    p1_pick, p1_drop = _parcel_nodes(problem, p1)
    p2_pick, p2_drop = _parcel_nodes(problem, p2)

    routes[r1][p1_pos], routes[r1][d1_pos] = p2_pick, p2_drop
    routes[r2][p2_pos], routes[r2][d2_pos] = p1_pick, p1_drop

    return _make_move(solution, routes, before_edges, "parcel_swap")


def generate_passenger_relocate(solution: Solution1D) -> Optional[MoveInfo]:
    """
    Move one passenger pickup node to another route position.

    Why this operator: passengers are direct trips in the decoder, so relocating
    the pickup is enough to transfer the whole direct passenger ride. It gives
    the search a small, precise load-balancing move beside the parcel moves.
    """

    problem = solution.problem
    routes = [list(route) for route in solution.get_routes()]
    locations = _passenger_locations(routes, problem)
    if len(locations) < 1:
        return None

    before_edges = _all_edges(routes)
    src_route, src_pos = random.choice(locations)
    node = routes[src_route].pop(src_pos)

    dst_route = random.randrange(problem.K)
    insert_pos = random.randint(0, len(routes[dst_route]))
    routes[dst_route].insert(insert_pos, node)

    return _make_move(solution, routes, before_edges, "passenger_relocate")


def generate_intra_route_reorder(solution: Solution1D) -> Optional[MoveInfo]:
    """
    Reverse a short block inside one taxi route.

    Why this operator: after assignments are roughly balanced, most remaining
    improvement comes from route order. Reversing a local block is a 2-opt-like
    exploitation move. Reversals that break parcel precedence are skipped.
    """

    problem = solution.problem
    routes = [list(route) for route in solution.get_routes()]
    candidates = [idx for idx, route in enumerate(routes) if len(route) >= 3]
    if not candidates:
        return None

    before_edges = _all_edges(routes)
    route_idx = random.choice(candidates)
    route = routes[route_idx]
    left, right = sorted(random.sample(range(len(route)), 2))
    if left == right:
        return None

    route[left : right + 1] = reversed(route[left : right + 1])

    for parcel_idx in range(1, problem.M + 1):
        found = _find_parcel(routes, problem, parcel_idx)
        if found is None:
            return None

    return _make_move(solution, routes, before_edges, "intra_route_reorder")


def _is_structurally_valid(solution: Solution1D) -> bool:
    valid, _ = solution.validate()
    return valid


def run_tabu(
    problem: ProblemData,
    config: TabuConfig | None = None,
    initial_solution: Solution1D | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Execute Adaptive Tabu Search.

    Returns (best_solution, best_fitness, history). Lower fitness is better.
    The aspiration criterion allows a tabu move when it creates a new global
    best solution.
    """

    if config is None:
        config = TabuConfig()
    if config.seed is not None:
        random.seed(config.seed)

    start_time = time.time()
    current = initial_solution.copy() if initial_solution is not None else build_initial_solution(problem)
    current_fitness = evaluate(current)

    best_solution = current.copy()
    best_fitness = current_fitness
    history: List[float] = [best_fitness]

    tabu = TabuList(config.tenure_init)
    stagnation_counter = 0
    generators: List[MoveGenerator] = [
        generate_parcel_transfer,
        generate_parcel_swap,
        generate_passenger_relocate,
        generate_intra_route_reorder,
    ]

    logger.info(
        "Tabu Search started | max_iter=%d | sample=%d | tenure=%d",
        config.max_iterations,
        config.neighbourhood_sample_size,
        tabu.tenure,
    )

    for iteration in range(1, config.max_iterations + 1):
        if time.time() - start_time >= config.time_limit_seconds:
            logger.info("Tabu time limit reached at iteration %d", iteration)
            break

        candidates: List[Tuple[Solution1D, float, Set[Edge], Set[Edge], str]] = []
        attempts = max(config.neighbourhood_sample_size * 3, 10)
        while len(candidates) < config.neighbourhood_sample_size and attempts > 0:
            attempts -= 1
            move = random.choice(generators)(current)
            if move is None:
                continue
            neighbour, added_edges, removed_edges, name = move
            if not _is_structurally_valid(neighbour):
                continue
            candidates.append((neighbour, evaluate(neighbour), added_edges, removed_edges, name))

        if not candidates:
            logger.warning("No valid Tabu neighbours generated at iteration %d", iteration)
            break

        candidates.sort(key=lambda item: item[1])
        chosen: Optional[Tuple[Solution1D, float, Set[Edge], Set[Edge], str]] = None

        for candidate in candidates:
            neighbour, fit, added_edges, _removed_edges, _name = candidate
            is_tabu = tabu.is_tabu(added_edges, iteration)
            aspiration = fit < best_fitness
            if not is_tabu or aspiration:
                chosen = candidate
                break

        if chosen is None:
            chosen = candidates[0]

        current, current_fitness, added_edges, removed_edges, move_name = chosen
        tabu.add(removed_edges, iteration)

        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current.copy()
            stagnation_counter = 0
            tabu.tenure = max(config.tenure_min, tabu.tenure - config.tenure_decrease)
        else:
            stagnation_counter += 1

        if stagnation_counter >= config.stagnation_limit:
            tabu.tenure = min(config.tenure_max, tabu.tenure + config.tenure_increase)
            stagnation_counter = 0
            logger.debug("Iter %d: stagnation, increase tenure to %d", iteration, tabu.tenure)

        history.append(best_fitness)

        if iteration == 1 or iteration % 100 == 0:
            logger.info(
                "Iter %4d | best=%.4f | current=%.4f | tenure=%d | move=%s",
                iteration,
                best_fitness,
                current_fitness,
                tabu.tenure,
                move_name,
            )

    logger.info("Tabu Search finished | best_fitness=%.4f", best_fitness)
    return best_solution, best_fitness, history
