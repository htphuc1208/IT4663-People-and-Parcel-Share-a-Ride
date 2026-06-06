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
from .fitness import evaluate, _route_distance, _evaluate_violations

from .init import greedy_init, minmax_greedy_init, savings_init


logger = logging.getLogger(__name__)

Edge = Tuple[int, int]
# A move is reported as the modified K route segments plus the tabu edge deltas,
# the move name, and the set of route indices it touched. Returning the raw
# routes (instead of a built Solution1D) lets the main loop run delta evaluation
# on the changed routes only.
MoveInfo = Tuple[List[List[int]], Set[Edge], Set[Edge], str, Set[int]]
MoveGenerator = Callable[[Solution1D], Optional[MoveInfo]]


@dataclass
class TabuConfig:
    """Configuration for Adaptive Tabu Search."""

    max_iterations: int = 1000
    neighbourhood_sample_size: int = 50

    tenure_init: int = -1   # -1 = auto-scale: max(7,  min(30,  (N+M)//10))
    tenure_min: int  = -1   # -1 = auto-scale: max(3,  min(10,  (N+M)//20))
    tenure_max: int  = -1   # -1 = auto-scale: max(20, min(100, (N+M)//5))
    tenure_increase: int = 3
    tenure_decrease: int = 1
    stagnation_limit: int = 25

    # ── Diversification (ILS-style restart from best) ────────────────────
    # After this many consecutive stagnation-driven tenure bumps without any
    # global improvement, perturb the incumbent best and restart the search.
    restart_after_bumps: int = 3
    perturbation_strength: int = -1   # -1 = auto: max(2, (N+M)//20)

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
    routes: List[List[int]],
    before_edges: Set[Edge],
    name: str,
    changed: Set[int],
) -> MoveInfo:
    after_edges = _all_edges(routes)
    removed_edges = before_edges - after_edges
    added_edges = after_edges - before_edges
    return routes, added_edges, removed_edges, name, changed


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║          DELTA EVALUATION — per-route distance/penalty caching           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# evaluate() decodes and measures every one of the K routes. Because each move
# touches only one or two routes, we cache (distance, penalty) per route and
# recompute just the changed ones. fitness = max(distances) + sum(penalties),
# kept identical to fitness.evaluate() by reusing its distance/penalty helpers.

RouteMetric = Tuple[float, float]   # (distance, penalty)


def _segment_cost(segment: List[int], problem: ProblemData, capacity: int) -> RouteMetric:
    """Decode one raw route segment and return its (distance, penalty)."""
    N, M = problem.N, problem.M
    route: List[int] = [0]
    for node in segment:
        if 1 <= node <= N:
            route.append(node)
            route.append(node + N + M)
        else:
            route.append(node)
    route.append(0)
    dist = _route_distance(route, problem)
    penalty = _evaluate_violations(
        route, capacity, problem.demand_by_pickup, problem.pickup_by_dropoff
    )
    return dist, penalty


def _full_metrics(solution: Solution1D) -> List[RouteMetric]:
    prob = solution.problem
    return [
        _segment_cost(seg, prob, prob.vehicle_capacities[k])
        for k, seg in enumerate(solution.get_routes())
    ]


def _fitness_from_metrics(metrics: List[RouteMetric]) -> float:
    return max(m[0] for m in metrics) + sum(m[1] for m in metrics)


def _delta_fitness(
    routes: List[List[int]],
    changed: Set[int],
    base_metrics: List[RouteMetric],
    problem: ProblemData,
) -> Tuple[float, List[RouteMetric]]:
    """Recompute only the changed routes; reuse cached metrics for the rest."""
    metrics = list(base_metrics)
    for idx in changed:
        metrics[idx] = _segment_cost(routes[idx], problem, problem.vehicle_capacities[idx])
    return _fitness_from_metrics(metrics), metrics


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 CHEAPEST-INSERTION (marginal distance)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def _phys_last(token: int, problem: ProblemData) -> int:
    """Physical node the vehicle is at after serving a raw token (passenger →
    its auto drop-off; parcel pickup/drop-off → itself)."""
    return token + problem.N + problem.M if 1 <= token <= problem.N else token


def _insertion_marginal(
    seg: List[int], i: int, expand: List[int], problem: ProblemData
) -> float:
    """Added decoded distance from inserting `expand` (the physical nodes of one
    raw token) into the raw segment `seg` at gap index `i`."""
    left = 0 if i == 0 else _phys_last(seg[i - 1], problem)
    right = 0 if i == len(seg) else seg[i]   # decoded route enters a token at its pickup id
    cost = problem.dist(left, expand[0])
    for a, b in zip(expand, expand[1:]):
        cost += problem.dist(a, b)
    cost += problem.dist(expand[-1], right) - problem.dist(left, right)
    return cost


def _best_insert_position(
    seg: List[int], expand: List[int], problem: ProblemData, lo: int = 0
) -> int:
    """Gap index in [lo, len(seg)] minimising the marginal insertion distance."""
    best_i = lo
    best_c = float("inf")
    for i in range(lo, len(seg) + 1):
        c = _insertion_marginal(seg, i, expand, problem)
        if c < best_c:
            best_c, best_i = c, i
    return best_i


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


def _route_precedence_ok(route: List[int], problem: ProblemData) -> bool:
    """True iff every parcel drop-off in this route appears after its pickup."""
    pickup_by_dropoff = problem.pickup_by_dropoff
    seen: Set[int] = set()
    for node in route:
        if node in pickup_by_dropoff and pickup_by_dropoff[node] not in seen:
            return False
        seen.add(node)
    return True


def _passenger_locations(routes: List[List[int]], problem: ProblemData) -> List[Tuple[int, int]]:
    locations: List[Tuple[int, int]] = []
    for route_idx, route in enumerate(routes):
        for pos, node in enumerate(route):
            if 1 <= node <= problem.N:
                locations.append((route_idx, pos))
    return locations


def _route_distances(solution: Solution1D) -> List[float]:
    """Khoảng cách thực của từng route (depot→...→depot)."""
    prob = solution.problem
    dists: List[float] = []
    for seg in solution.get_routes():
        if not seg:
            dists.append(0.0)
            continue
        d = prob.dist(0, seg[0])
        for i in range(len(seg) - 1):
            d += prob.dist(seg[i], seg[i + 1])
        d += prob.dist(seg[-1], 0)
        dists.append(d)
    return dists


def _parcels_on_route(route: List[int], problem: ProblemData) -> List[int]:
    """Trả về danh sách parcel ID (1-based) có pickup node nằm trong route."""
    return [node - problem.N for node in route
            if problem.N < node <= problem.N + problem.M]


def generate_parcel_transfer(solution: Solution1D) -> Optional[MoveInfo]:
    """
    Move one parcel pickup/drop-off pair to another taxi.

    Bottleneck bias (70 %): lấy parcel từ xe DÀI nhất, chèn vào xe NGẮN nhất.
    Phần còn lại (30 %): chọn ngẫu nhiên để giữ diversity.
    """
    problem = solution.problem
    if problem.M == 0 or problem.K < 2:
        return None

    routes = [list(route) for route in solution.get_routes()]
    before_edges = _all_edges(routes)
    dists = _route_distances(solution)
    bottleneck_idx = dists.index(max(dists))

    # ── Chọn nguồn ──────────────────────────────────────────────────────
    selected: Optional[Tuple[int, int, int, int]] = None

    if random.random() < 0.70:
        bn_parcels = _parcels_on_route(routes[bottleneck_idx], problem)
        if bn_parcels:
            pid = random.choice(bn_parcels)
            found = _find_parcel(routes, problem, pid)
            if found is not None:
                selected = (pid, found[0], found[1], found[2])

    if selected is None:          # fallback: random
        parcel_ids = list(range(1, problem.M + 1))
        random.shuffle(parcel_ids)
        for pid in parcel_ids:
            found = _find_parcel(routes, problem, pid)
            if found is not None:
                selected = (pid, found[0], found[1], found[2])
                break

    if selected is None:
        return None

    parcel_idx, src_route, pickup_pos, dropoff_pos = selected
    pickup, dropoff = _parcel_nodes(problem, parcel_idx)
    target_routes = [i for i in range(problem.K) if i != src_route]
    if not target_routes:
        return None

    # ── Chọn đích ───────────────────────────────────────────────────────
    shortest_target = min(target_routes, key=lambda i: dists[i])
    dst_route = shortest_target if random.random() < 0.70 else random.choice(target_routes)

    for pos in sorted((pickup_pos, dropoff_pos), reverse=True):
        routes[src_route].pop(pos)

    # Cheapest sequential insertion: place the pickup at its best gap, then the
    # drop-off at its best gap after the pickup (keeps precedence).
    dst = routes[dst_route]
    insert_pickup = _best_insert_position(dst, [pickup], problem)
    dst.insert(insert_pickup, pickup)
    insert_dropoff = _best_insert_position(dst, [dropoff], problem, lo=insert_pickup + 1)
    dst.insert(insert_dropoff, dropoff)

    return _make_move(routes, before_edges, "parcel_transfer", {src_route, dst_route})


def generate_parcel_swap(solution: Solution1D) -> Optional[MoveInfo]:
    """
    Swap hai parcel requests (đổi vị trí trong chromosome).

    Bottleneck bias: ưu tiên lấy parcel 1 từ xe dài nhất, parcel 2 từ xe ngắn nhất
    để đổi công việc giữa hai đầu cực — cân bằng trực tiếp hơn swap ngẫu nhiên.
    """
    problem = solution.problem
    if problem.M < 2:
        return None

    routes = [list(route) for route in solution.get_routes()]
    before_edges = _all_edges(routes)
    dists = _route_distances(solution)
    bottleneck_idx = dists.index(max(dists))

    found: List[Tuple[int, int, int, int]] = []

    # Parcel 1: 70% từ bottleneck
    if random.random() < 0.70:
        bn_parcels = _parcels_on_route(routes[bottleneck_idx], problem)
        if bn_parcels:
            pid = random.choice(bn_parcels)
            loc = _find_parcel(routes, problem, pid)
            if loc is not None:
                found.append((pid, loc[0], loc[1], loc[2]))

    # Parcel 2 (và fallback parcel 1): từ random
    parcel_ids = list(range(1, problem.M + 1))
    random.shuffle(parcel_ids)
    for pid in parcel_ids:
        if len(found) >= 2:
            break
        if found and found[0][0] == pid:
            continue
        loc = _find_parcel(routes, problem, pid)
        if loc is not None:
            found.append((pid, loc[0], loc[1], loc[2]))

    if len(found) < 2:
        return None

    first, second = found[0], found[1]
    p1, r1, p1_pos, d1_pos = first
    p2, r2, p2_pos, d2_pos = second
    p1_pick, p1_drop = _parcel_nodes(problem, p1)
    p2_pick, p2_drop = _parcel_nodes(problem, p2)

    routes[r1][p1_pos], routes[r1][d1_pos] = p2_pick, p2_drop
    routes[r2][p2_pos], routes[r2][d2_pos] = p1_pick, p1_drop

    return _make_move(routes, before_edges, "parcel_swap", {r1, r2})


def generate_passenger_relocate(solution: Solution1D) -> Optional[MoveInfo]:
    """
    Di chuyển một passenger pickup sang route khác.

    Bottleneck bias (70 %): lấy passenger từ xe DÀI nhất,
    chèn vào xe NGẮN nhất để cân bằng tải.
    """
    problem = solution.problem
    routes = [list(route) for route in solution.get_routes()]
    before_edges = _all_edges(routes)
    dists = _route_distances(solution)
    bottleneck_idx = dists.index(max(dists))

    # ── Chọn nguồn ──────────────────────────────────────────────────────
    src_route: int
    src_pos: int

    if random.random() < 0.70:
        bn_locs = [(bottleneck_idx, pos)
                   for pos, node in enumerate(routes[bottleneck_idx])
                   if 1 <= node <= problem.N]
        if bn_locs:
            src_route, src_pos = random.choice(bn_locs)
        else:
            all_locs = _passenger_locations(routes, problem)
            if not all_locs:
                return None
            src_route, src_pos = random.choice(all_locs)
    else:
        all_locs = _passenger_locations(routes, problem)
        if not all_locs:
            return None
        src_route, src_pos = random.choice(all_locs)

    node = routes[src_route].pop(src_pos)

    # ── Chọn đích ───────────────────────────────────────────────────────
    target_routes = list(range(problem.K))
    shortest_dst = min(target_routes, key=lambda i: dists[i])
    dst_route = shortest_dst if random.random() < 0.70 else random.randrange(problem.K)

    # Cheapest insertion: the decoded passenger occupies [pickup, dropoff], so
    # estimate the marginal over that pair even though only the pickup is stored.
    dropoff = node + problem.N + problem.M
    insert_pos = _best_insert_position(routes[dst_route], [node, dropoff], problem)
    routes[dst_route].insert(insert_pos, node)

    return _make_move(routes, before_edges, "passenger_relocate", {src_route, dst_route})


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

    # A reversal can only break parcel precedence inside the reversed route, so
    # checking that single route (O(len route)) is enough — no need to scan all
    # M parcels across every route.
    if not _route_precedence_ok(route, problem):
        return None

    return _make_move(routes, before_edges, "intra_route_reorder", {route_idx})


def _perturb(solution: Solution1D, strength: int) -> Solution1D:
    """
    ILS-style diversification kick: apply `strength` random relocations to a copy
    of `solution`. Passengers move as a single pickup token; parcels move as a
    pickup/drop-off pair (drop-off placed right after the pickup) so structural
    validity and precedence are preserved by construction.
    """
    problem = solution.problem
    K = problem.K
    routes = [list(route) for route in solution.get_routes()]

    for _ in range(strength):
        non_empty = [i for i, r in enumerate(routes) if r]
        if not non_empty:
            break
        src = random.choice(non_empty)
        node = random.choice(routes[src])

        if 1 <= node <= problem.N:                      # passenger pickup
            routes[src].remove(node)
            dst = random.randrange(K)
            routes[dst].insert(random.randint(0, len(routes[dst])), node)
        elif problem.N < node <= problem.N + problem.M:  # parcel pickup
            pid = node - problem.N
            found = _find_parcel(routes, problem, pid)
            if found is None:
                continue
            r, pp, dp = found
            pickup, dropoff = _parcel_nodes(problem, pid)
            for pos in sorted((pp, dp), reverse=True):
                routes[r].pop(pos)
            dst = random.randrange(K)
            ip = random.randint(0, len(routes[dst]))
            routes[dst].insert(ip, pickup)
            routes[dst].insert(random.randint(ip + 1, len(routes[dst])), dropoff)
        # parcel drop-off tokens are handled via their pickup; skip otherwise.

    return Solution1D(_flatten_routes(routes), problem)


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

    # ── Auto-scale tenure based on problem size ──────────────────────────
    R: int = problem.N + problem.M
    eff_tenure_init: int = (
        config.tenure_init if config.tenure_init > 0
        else max(7, min(30, R // 10))
    )
    eff_tenure_min: int = (
        config.tenure_min if config.tenure_min > 0
        else max(3, min(10, R // 20))
    )
    eff_tenure_max: int = (
        config.tenure_max if config.tenure_max > 0
        else max(20, min(100, R // 5))
    )
    eff_perturbation: int = (
        config.perturbation_strength if config.perturbation_strength > 0
        else max(2, R // 20)
    )

    if initial_solution is not None:
        current = initial_solution.copy()
    else:
        _s = config.seed if config.seed is not None else 0
        _cands = [
            greedy_init(problem),
            greedy_init(problem, seed=_s),
            minmax_greedy_init(problem),
            savings_init(problem),
        ]
        _fits = [evaluate(c) for c in _cands]
        current = _cands[int(min(range(4), key=lambda i: _fits[i]))]
        logger.info("Tabu init | best-of-4 fitness=%.4f", min(_fits))

    current_metrics = _full_metrics(current)
    current_fitness = _fitness_from_metrics(current_metrics)

    best_solution = current.copy()
    best_fitness = current_fitness
    history: List[float] = [best_fitness]

    tabu = TabuList(eff_tenure_init)
    stagnation_counter = 0
    bumps_without_improvement = 0   # consecutive stagnation bumps since last gain
    generators: List[MoveGenerator] = [
        generate_parcel_transfer,
        generate_parcel_swap,
        generate_passenger_relocate,
        generate_intra_route_reorder,
    ]

    logger.info(
        "Tabu Search started | max_iter=%d | sample=%d | tenure=%d (min=%d max=%d)",
        config.max_iterations,
        config.neighbourhood_sample_size,
        eff_tenure_init, eff_tenure_min, eff_tenure_max,
    )

    for iteration in range(1, config.max_iterations + 1):
        if time.time() - start_time >= config.time_limit_seconds:
            logger.info("Tabu time limit reached at iteration %d", iteration)
            break

        # Purge expired tabu entries once per iteration instead of once per candidate.
        tabu.purge(iteration)

        # ── Generate candidates with delta evaluation ────────────────────────
        # Generators return raw route lists; we rebuild Solution1D only for the
        # accepted move, not for every candidate.
        candidates: List = []
        attempts = max(config.neighbourhood_sample_size * 3, 10)
        while len(candidates) < config.neighbourhood_sample_size and attempts > 0:
            attempts -= 1
            move = random.choice(generators)(current)
            if move is None:
                continue
            n_routes, added_edges, removed_edges, name, changed = move
            fit, new_metrics = _delta_fitness(n_routes, changed, current_metrics, problem)
            candidates.append((n_routes, fit, added_edges, removed_edges, name, changed, new_metrics))

        if not candidates:
            logger.warning("No valid Tabu neighbours generated at iteration %d", iteration)
            break

        candidates.sort(key=lambda item: item[1])

        # ── Tabu filtering with aspiration ───────────────────────────────────
        chosen = None
        for candidate in candidates:
            _, fit, added_edges, _, _, _, _ = candidate
            if not tabu.is_tabu(added_edges, iteration) or fit < best_fitness:
                chosen = candidate
                break
        if chosen is None:
            chosen = candidates[0]

        n_routes, current_fitness, added_edges, removed_edges, move_name, _, current_metrics = chosen
        # Build Solution1D once per accepted move (not per candidate).
        current = Solution1D(_flatten_routes(n_routes), problem)
        tabu.add(removed_edges, iteration)

        # ── Update best and adaptive tenure ─────────────────────────────────
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current.copy()
            stagnation_counter = 0
            bumps_without_improvement = 0
            tabu.tenure = max(eff_tenure_min, tabu.tenure - config.tenure_decrease)
        else:
            stagnation_counter += 1

        if stagnation_counter >= config.stagnation_limit:
            tabu.tenure = min(eff_tenure_max, tabu.tenure + config.tenure_increase)
            stagnation_counter = 0
            bumps_without_improvement += 1
            logger.debug("Iter %d: stagnation, tenure→%d", iteration, tabu.tenure)

            # ── Diversification: ILS-style restart from best ─────────────────
            # After restart_after_bumps consecutive bumps with no global gain,
            # perturb the best known solution and reset tabu memory. This escapes
            # deep local optima that tenure increase alone cannot escape.
            if bumps_without_improvement >= config.restart_after_bumps:
                current = _perturb(best_solution, eff_perturbation)
                current_metrics = _full_metrics(current)
                current_fitness = _fitness_from_metrics(current_metrics)
                tabu = TabuList(eff_tenure_init)
                stagnation_counter = 0
                bumps_without_improvement = 0
                logger.info(
                    "Iter %d: restart | strength=%d | restart_fitness=%.4f | best=%.4f",
                    iteration, eff_perturbation, current_fitness, best_fitness,
                )

        history.append(best_fitness)

        if iteration == 1 or iteration % 100 == 0:
            logger.info(
                "Iter %4d | best=%.4f | current=%.4f | tenure=%d | move=%s",
                iteration, best_fitness, current_fitness, tabu.tenure, move_name,
            )

    logger.info("Tabu Search finished | best_fitness=%.4f", best_fitness)
    return best_solution, best_fitness, history
