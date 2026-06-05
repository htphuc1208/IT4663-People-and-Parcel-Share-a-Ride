"""
init.py — Greedy Initialisations for the Unified 1D Solution
=============================================================

Four initialisation strategies are provided, all returning a valid Solution1D:

  greedy_init           — round-robin passengers, best-fit parcels by distance.
  proximity_sweep_init  — sort requests by distance from depot, capacity-
                          proportional sector split, nearest-neighbor ordering.
  minmax_greedy_init    — assign longest trips first to the shortest vehicle,
                          targeting the min-max objective directly.
  regret_init           — regret-based greedy insertion (O((N+M)^2*K)).
  random_init           — random structurally-valid fallback.

Capacity notes
--------------
  Q[k] (vehicle_capacities[k]) applies to PARCEL demand only.
  Passengers have no capacity constraint — they are always assignable.
  All init functions track only parcel load against Q[k].
"""

from __future__ import annotations

import random
from typing import List, Tuple

from .encoding_and_read import ProblemData, Request, RequestType, Solution1D


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GREEDY INITIALISATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def greedy_init(problem: ProblemData, seed: int | None = None) -> Solution1D:
    """
    Build an initial solution using a greedy heuristic.

    Phase 1 — Passengers (round-robin):
        Assign each passenger pickup to vehicles in round-robin order.
        Passengers do not consume parcel capacity.

    Phase 2 — Parcels (best-fit decreasing):
        Sort parcels by demand descending.  Assign each parcel to the vehicle
        with the shortest accumulated route distance that still has parcel
        capacity.

    Phase 3 — Assemble:
        Concatenate route segments separated by 0 (depot).
    """
    if seed is not None:
        random.seed(seed)

    K: int = problem.K

    vehicle_routes: List[List[int]] = [[] for _ in range(K)]
    vehicle_dist: List[float] = [0.0] * K
    vehicle_parcel_load: List[int] = [0] * K

    # Phase 1: passengers round-robin
    for idx, pax in enumerate(problem.passengers):
        v: int = idx % K
        vehicle_routes[v].append(pax.pickup_node)
        vehicle_dist[v] += (
            problem.dist(0, pax.pickup_node)
            + problem.dist(pax.pickup_node, pax.dropoff_node)
            + problem.dist(pax.dropoff_node, 0)
        )

    # Phase 2: parcels best-fit decreasing
    for parcel in sorted(problem.parcels, key=lambda r: r.demand, reverse=True):
        best_v: int = -1
        best_d: float = float("inf")
        for v in range(K):
            if vehicle_parcel_load[v] + parcel.demand <= problem.vehicle_capacities[v]:
                if vehicle_dist[v] < best_d:
                    best_d = vehicle_dist[v]
                    best_v = v
        if best_v == -1:
            best_v = min(range(K), key=lambda v: vehicle_dist[v])

        vehicle_routes[best_v].append(parcel.pickup_node)
        vehicle_routes[best_v].append(parcel.dropoff_node)
        vehicle_parcel_load[best_v] += parcel.demand
        vehicle_dist[best_v] += (
            problem.dist(0, parcel.pickup_node)
            + problem.dist(parcel.pickup_node, parcel.dropoff_node)
            + problem.dist(parcel.dropoff_node, 0)
        )

    # Phase 3: assemble chromosome
    chromosome: List[int] = []
    for v in range(K):
        chromosome.extend(vehicle_routes[v])
        if v < K - 1:
            chromosome.append(0)

    solution = Solution1D(chromosome=chromosome, problem=problem)
    is_valid, msg = solution.validate()
    if not is_valid:
        raise RuntimeError(
            f"greedy_init produced an invalid chromosome: {msg}\n"
            f"  Expected={problem.chromosome_length}, Got={len(chromosome)}"
        )
    return solution


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  PROXIMITY SWEEP INITIALISATION                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def proximity_sweep_init(
    problem: ProblemData,
    vehicle_capacities: List[int] | None = None,
    seed: int | None = None,
) -> Solution1D:
    """
    Build an initial solution via proximity sweep.

    Replaces the coordinate-based angular sweep with a distance-matrix-based
    sweep usable when only d(i,j) is available.

    Step 1 — Sort requests by pickup distance from depot.
              A random pivot shifts the sort key for diversity across seeds.
    Step 2 — Assign proportionally to vehicle capacity thresholds.
              Parcel capacity Q[k] is enforced; passengers are unconstrained.
    Step 3 — Nearest-neighbor ordering within each vehicle.
    Step 4 — Assemble chromosome.
    """
    if seed is not None:
        random.seed(seed)

    K: int = problem.K
    if vehicle_capacities is None:
        vehicle_capacities = problem.vehicle_capacities

    all_requests: List[Request] = problem.passengers + problem.parcels
    if not all_requests:
        return Solution1D(chromosome=[0] * (K - 1), problem=problem)

    # Step 1: sort with random pivot for diversity
    max_dist: float = max(problem.dist(0, r.pickup_node) for r in all_requests)
    if seed is not None:
        pivot = random.choice(all_requests)
        offset: float = problem.dist(0, pivot.pickup_node)
        sorted_requests: List[Request] = sorted(
            all_requests,
            key=lambda r: (problem.dist(0, r.pickup_node) - offset) % (max_dist + 1.0),
        )
    else:
        sorted_requests = sorted(
            all_requests, key=lambda r: problem.dist(0, r.pickup_node)
        )

    # Step 2: capacity-proportional assignment
    total_cap: int = sum(vehicle_capacities)
    n_req: int = len(all_requests)
    thresholds: List[float] = []
    acc: int = 0
    for k in range(K):
        acc += vehicle_capacities[k]
        thresholds.append(acc / total_cap * n_req)

    vehicle_requests: List[List[Request]] = [[] for _ in range(K)]
    vehicle_parcel_load: List[int] = [0] * K
    current_v: int = 0

    for req in sorted_requests:
        while current_v < K - 1 and len(vehicle_requests[current_v]) >= thresholds[current_v]:
            current_v += 1

        if req.request_type == RequestType.PASSENGER:
            vehicle_requests[current_v].append(req)
        else:
            if vehicle_parcel_load[current_v] + req.demand <= vehicle_capacities[current_v]:
                vehicle_requests[current_v].append(req)
                vehicle_parcel_load[current_v] += req.demand
            else:
                placed: bool = False
                for v in range(K):
                    if vehicle_parcel_load[v] + req.demand <= vehicle_capacities[v]:
                        vehicle_requests[v].append(req)
                        vehicle_parcel_load[v] += req.demand
                        placed = True
                        break
                if not placed:
                    vehicle_requests[current_v].append(req)
                    vehicle_parcel_load[current_v] += req.demand

    chromosome: List[int] = _nn_order_and_assemble(problem, vehicle_requests)
    solution = Solution1D(chromosome=chromosome, problem=problem)
    is_valid, msg = solution.validate()
    if not is_valid:
        raise RuntimeError(
            f"proximity_sweep_init produced an invalid chromosome: {msg}\n"
            f"  Expected={problem.chromosome_length}, Got={len(chromosome)}"
        )
    return solution


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper: nearest-neighbor ordering + chromosome assembly
# ─────────────────────────────────────────────────────────────────────────────

def _nn_order_and_assemble(
    problem: ProblemData,
    vehicle_requests: List[List[Request]],
) -> List[int]:
    """
    Nearest-neighbor ordering within each vehicle then flat chromosome assembly.

    Starting from depot, repeatedly pick the unvisited request whose pickup is
    closest to the current position; cursor advances to its dropoff node.
    """
    K: int = problem.K
    chromosome: List[int] = []

    for v in range(K):
        remaining: List[Request] = list(vehicle_requests[v])
        route: List[int] = []
        current_node: int = 0

        while remaining:
            best_idx: int = min(
                range(len(remaining)),
                key=lambda i: problem.dist(current_node, remaining[i].pickup_node),
            )
            req = remaining.pop(best_idx)
            if req.request_type == RequestType.PASSENGER:
                route.append(req.pickup_node)
                current_node = req.dropoff_node
            else:
                route.append(req.pickup_node)
                route.append(req.dropoff_node)
                current_node = req.dropoff_node

        chromosome.extend(route)
        if v < K - 1:
            chromosome.append(0)

    return chromosome


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              MIN-MAX BALANCED GREEDY INITIALISATION                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def minmax_greedy_init(
    problem: ProblemData,
    vehicle_capacities: List[int] | None = None,
    seed: int | None = None,
) -> Solution1D:
    """
    Build an initial solution targeting the min-max objective directly.

    Step 1 — Sort ALL requests by direct trip distance descending.
    Step 2 — Assign each to the vehicle with minimum accumulated distance
              that still has parcel capacity (passengers unconstrained).
    Step 3 — Nearest-neighbor ordering + assemble.
    """
    _ = seed
    K: int = problem.K
    if vehicle_capacities is None:
        vehicle_capacities = problem.vehicle_capacities

    all_requests: List[Request] = sorted(
        problem.passengers + problem.parcels,
        key=lambda r: problem.dist(r.pickup_node, r.dropoff_node),
        reverse=True,
    )

    vehicle_requests: List[List[Request]] = [[] for _ in range(K)]
    vehicle_parcel_load: List[int] = [0] * K
    vehicle_dist: List[float] = [0.0] * K

    for req in all_requests:
        best_v: int = -1
        best_d: float = float("inf")

        if req.request_type == RequestType.PASSENGER:
            for v in range(K):
                if vehicle_dist[v] < best_d:
                    best_d = vehicle_dist[v]
                    best_v = v
        else:
            for v in range(K):
                if vehicle_parcel_load[v] + req.demand <= vehicle_capacities[v]:
                    if vehicle_dist[v] < best_d:
                        best_d = vehicle_dist[v]
                        best_v = v
            if best_v == -1:
                best_v = min(range(K), key=lambda v: vehicle_dist[v])

        vehicle_requests[best_v].append(req)
        if req.request_type == RequestType.PARCEL:
            vehicle_parcel_load[best_v] += req.demand
        vehicle_dist[best_v] += (
            problem.dist(0, req.pickup_node)
            + problem.dist(req.pickup_node, req.dropoff_node)
            + problem.dist(req.dropoff_node, 0)
        )

    chromosome: List[int] = _nn_order_and_assemble(problem, vehicle_requests)
    solution = Solution1D(chromosome=chromosome, problem=problem)
    is_valid, msg = solution.validate()
    if not is_valid:
        raise RuntimeError(
            f"minmax_greedy_init produced an invalid chromosome: {msg}\n"
            f"  Expected={problem.chromosome_length}, Got={len(chromosome)}"
        )
    return solution


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  REGRET-BASED INSERTION INITIALISATION                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def regret_init(
    problem: ProblemData,
    vehicle_capacities: List[int] | None = None,
    seed: int | None = None,
) -> Solution1D:
    """
    Regret-based greedy insertion.

    regret(r) = cost(r, 2nd-best vehicle) - cost(r, best vehicle)

    High regret → commit first.  Insertion cost = dist(last[v], pickup) + dist(pickup, dropoff).
    Complexity: O((N+M)^2 * K).  Prefer minmax_greedy_init for large instances.
    """
    _ = seed
    K: int = problem.K
    if vehicle_capacities is None:
        vehicle_capacities = problem.vehicle_capacities

    vehicle_routes: List[List[int]] = [[] for _ in range(K)]
    vehicle_parcel_load: List[int] = [0] * K
    vehicle_last: List[int] = [0] * K

    remaining: List[Request] = list(problem.passengers + problem.parcels)

    while remaining:
        best_regret: float = -1.0
        chosen_idx: int = 0
        chosen_req: Request = remaining[0]
        chosen_v: int = 0

        for i, req in enumerate(remaining):
            feasible: List[Tuple[float, int]] = []
            for v in range(K):
                if req.request_type == RequestType.PASSENGER:
                    c: float = (
                        problem.dist(vehicle_last[v], req.pickup_node)
                        + problem.dist(req.pickup_node, req.dropoff_node)
                    )
                    feasible.append((c, v))
                elif vehicle_parcel_load[v] + req.demand <= vehicle_capacities[v]:
                    c = (
                        problem.dist(vehicle_last[v], req.pickup_node)
                        + problem.dist(req.pickup_node, req.dropoff_node)
                    )
                    feasible.append((c, v))

            if not feasible:
                regret = float("inf")
                best_v_for_req = min(range(K), key=lambda v: vehicle_parcel_load[v])
            elif len(feasible) == 1:
                regret = float("inf")
                best_v_for_req = feasible[0][1]
            else:
                feasible.sort()
                regret = feasible[1][0] - feasible[0][0]
                best_v_for_req = feasible[0][1]

            if regret > best_regret:
                best_regret = regret
                chosen_idx = i
                chosen_req = req
                chosen_v = best_v_for_req

        # O(1) swap-and-pop
        remaining[chosen_idx] = remaining[-1]
        remaining.pop()

        if chosen_req.request_type == RequestType.PASSENGER:
            vehicle_routes[chosen_v].append(chosen_req.pickup_node)
            vehicle_last[chosen_v] = chosen_req.dropoff_node
        else:
            vehicle_routes[chosen_v].append(chosen_req.pickup_node)
            vehicle_routes[chosen_v].append(chosen_req.dropoff_node)
            vehicle_parcel_load[chosen_v] += chosen_req.demand
            vehicle_last[chosen_v] = chosen_req.dropoff_node

    chromosome: List[int] = []
    for v in range(K):
        chromosome.extend(vehicle_routes[v])
        if v < K - 1:
            chromosome.append(0)

    solution = Solution1D(chromosome=chromosome, problem=problem)
    is_valid, msg = solution.validate()
    if not is_valid:
        raise RuntimeError(
            f"regret_init produced an invalid chromosome: {msg}\n"
            f"  Expected={problem.chromosome_length}, Got={len(chromosome)}"
        )
    return solution


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    RANDOM INITIALISATION (FALLBACK)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def random_init(problem: ProblemData, seed: int | None = None) -> Solution1D:
    """
    Random structurally-valid solution.

    Shuffles all non-zero nodes and places (K-1) depot separators at random
    positions.  Likely infeasible; relies on penalty-driven fitness.
    """
    if seed is not None:
        random.seed(seed)

    nodes: List[int] = []
    for pax in problem.passengers:
        nodes.append(pax.pickup_node)
    for parcel in problem.parcels:
        nodes.append(parcel.pickup_node)
        nodes.append(parcel.dropoff_node)

    random.shuffle(nodes)

    K: int = problem.K
    positions: List[int] = sorted(random.sample(range(1, len(nodes) + 1), K - 1))
    chromosome: List[int] = []
    node_idx: int = 0
    for pos in positions:
        while node_idx < pos:
            chromosome.append(nodes[node_idx])
            node_idx += 1
        chromosome.append(0)
    while node_idx < len(nodes):
        chromosome.append(nodes[node_idx])
        node_idx += 1

    return Solution1D(chromosome=chromosome, problem=problem)
