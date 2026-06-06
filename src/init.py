"""
init.py — Greedy Initialisations for the Unified 1D Solution
=============================================================

Three initialisation strategies + one random fallback:

  greedy_init         — round-robin passengers + best-fit parcels.
                        Seed shuffles passengers and adds epsilon-greedy
                        perturbation for population diversity.  O((N+M)·K).

  minmax_greedy_init  — sort all requests by trip length descending, assign
                        each to the shortest vehicle.  Seed adds ±5 % noise
                        to sort key for diversity.  O((N+M)·K).

  savings_init        — Clarke-Wright savings algorithm.  Builds geographically-
                        coherent chains and assigns them to vehicles targeting
                        the min-max objective.  O((N+M)² log(N+M)) — no K
                        factor, replaces the O((N+M)²·K) regret approach.

  random_init         — random structurally-valid fallback for GA diversity.

─────────────────────────────────────────────────────────────
CONSTRAINT GUARANTEES
─────────────────────────────────────────────────────────────
  Direct-trip (passenger):
    Only the PICKUP node of each passenger appears in the chromosome.
    The decoder (fitness.decode_routes) auto-inserts the dropoff node
    (pickup + N + M) immediately after — no other stop can appear between
    them by construction.

  Parcel precedence:
    _nn_order_and_assemble always appends parcel pickup then dropoff as a
    pair, ensuring pickup precedes dropoff in the chromosome segment.

  Parcel capacity:
    Every function tracks vehicle_parcel_load[v] against Q[v] = vehicle_
    capacities[v].  Passengers never consume parcel capacity.

  Structural validity:
    chromosome length = N + 2M + (K-1), exactly (K-1) zero separators,
    no duplicate non-zero node IDs — checked by Solution1D.validate().
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from .encoding_and_read import ProblemData, Request, RequestType, Solution1D


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GREEDY INITIALISATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def greedy_init(problem: ProblemData, seed: int | None = None) -> Solution1D:
    """
    Phase 1 — Passengers (round-robin):
        If seed is set, shuffle passengers first so different seeds yield
        different vehicle assignments → population diversity for GA.

    Phase 2 — Parcels (best-fit decreasing):
        Sort parcels by demand descending.  Assign each to the feasible
        vehicle with the shortest accumulated distance.
        If seed is set, 15 % of assignments are drawn from the top-3
        feasible vehicles (epsilon-greedy) instead of always the best.

    Complexity: O((N+M) · K)
    """
    if seed is not None:
        random.seed(seed)

    K: int = problem.K
    vehicle_routes: List[List[int]] = [[] for _ in range(K)]
    vehicle_dist: List[float] = [0.0] * K
    vehicle_parcel_load: List[int] = [0] * K

    # ── Phase 1: passengers ───────────────────────────────────────────────
    passengers = list(problem.passengers)
    if seed is not None:
        random.shuffle(passengers)

    for idx, pax in enumerate(passengers):
        v: int = idx % K
        vehicle_routes[v].append(pax.pickup_node)
        vehicle_dist[v] += (
            problem.dist(0, pax.pickup_node)
            + problem.dist(pax.pickup_node, pax.dropoff_node)
            + problem.dist(pax.dropoff_node, 0)
        )

    # ── Phase 2: parcels ──────────────────────────────────────────────────
    for parcel in sorted(problem.parcels, key=lambda r: r.demand, reverse=True):
        feasible = [
            v for v in range(K)
            if vehicle_parcel_load[v] + parcel.demand <= problem.vehicle_capacities[v]
        ]

        if not feasible:
            # Prefer vehicles whose capacity can at least hold this single parcel
            can_hold = [v for v in range(K) if problem.vehicle_capacities[v] >= parcel.demand]
            pool = can_hold if can_hold else list(range(K))
            best_v: int = min(pool, key=lambda v: vehicle_dist[v])
        elif seed is not None and len(feasible) > 1 and random.random() < 0.15:
            top3 = sorted(feasible, key=lambda v: vehicle_dist[v])[:3]
            best_v = random.choice(top3)
        else:
            best_v = min(feasible, key=lambda v: vehicle_dist[v])

        vehicle_routes[best_v].append(parcel.pickup_node)
        vehicle_routes[best_v].append(parcel.dropoff_node)
        vehicle_parcel_load[best_v] += parcel.demand
        vehicle_dist[best_v] += (
            problem.dist(0, parcel.pickup_node)
            + problem.dist(parcel.pickup_node, parcel.dropoff_node)
            + problem.dist(parcel.dropoff_node, 0)
        )

    # ── Assemble chromosome ───────────────────────────────────────────────
    chromosome: List[int] = []
    for v in range(K):
        chromosome.extend(vehicle_routes[v])
        if v < K - 1:
            chromosome.append(0)

    solution = Solution1D(chromosome=chromosome, problem=problem)
    is_valid, msg = solution.validate()
    if not is_valid:
        raise RuntimeError(
            f"greedy_init: {msg} | len={len(chromosome)}, "
            f"expected={problem.chromosome_length}"
        )
    return solution


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║              MIN-MAX BALANCED GREEDY INITIALISATION                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def minmax_greedy_init(
    problem: ProblemData,
    vehicle_capacities: List[int] | None = None,
    seed: int | None = None,
) -> Solution1D:
    """
    Sort all requests by direct-trip distance descending (longest trips first),
    then assign each to the feasible vehicle with the shortest accumulated
    route distance — directly targeting the min-max objective.

    When seed is provided ±5 % noise is added to the sort key, generating
    different request orderings while preserving the min-max focus.

    Complexity: O((N+M) · K)  +  O((N+M)²/K) for NN ordering within vehicles.
    """
    if seed is not None:
        random.seed(seed)

    K: int = problem.K
    if vehicle_capacities is None:
        vehicle_capacities = problem.vehicle_capacities

    raw = problem.passengers + problem.parcels

    if seed is not None:
        all_requests: List[Request] = sorted(
            raw,
            key=lambda r: problem.dist(r.pickup_node, r.dropoff_node)
                          * (1.0 + random.uniform(-0.05, 0.05)),
            reverse=True,
        )
    else:
        all_requests = sorted(
            raw,
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
                # Prefer vehicles whose capacity can at least hold this single parcel
                can_hold = [v for v in range(K) if vehicle_capacities[v] >= req.demand]
                pool = can_hold if can_hold else list(range(K))
                best_v = min(pool, key=lambda v: vehicle_dist[v])

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
        raise RuntimeError(f"minmax_greedy_init: {msg}")
    return solution


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                CLARKE-WRIGHT SAVINGS INITIALISATION                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def savings_init(
    problem: ProblemData,
    vehicle_capacities: List[int] | None = None,
    seed: int | None = None,
) -> Solution1D:
    """
    Clarke-Wright savings algorithm.

    Starts with each request in its own depot-to-depot route and iteratively
    merges pairs where the combined route is shorter:

        savings(i, j) = d(0, last_i) + d(0, pickup_j) - d(last_i, pickup_j)

    where last_i = dropoff_node of request i (the actual last physical stop).

    Merging is allowed when:
      • i is currently the tail of its chain and j is the head of another chain.
      • The combined parcel demand does not exceed the maximum vehicle capacity.

    After chaining, chains are assigned to vehicles using a min-max strategy
    (longest chain to the shortest vehicle), then NN-ordered within each vehicle.

    Seed randomises tie-breaking in the savings sort to generate diverse solutions.

    Complexity: O((N+M)² log(N+M)) — no K factor.
                ~100× faster than regret_init for large K.
    """
    if seed is not None:
        random.seed(seed)

    K: int = problem.K
    if vehicle_capacities is None:
        vehicle_capacities = problem.vehicle_capacities

    all_requests: List[Request] = problem.passengers + problem.parcels
    R: int = len(all_requests)

    if R == 0:
        return Solution1D(chromosome=[0] * (K - 1), problem=problem)

    max_cap: int = max(vehicle_capacities)

    def parcel_dem(req: Request) -> int:
        return req.demand if req.request_type == RequestType.PARCEL else 0

    # ── Compute savings ───────────────────────────────────────────────────
    # savings(i, j): benefit of visiting j immediately after i in the same route.
    # last_phys(i) = i.dropoff_node (where the vehicle physically ends up after i).
    savings: List[Tuple[float, int, int]] = []
    for i in range(R):
        last_i: int = all_requests[i].dropoff_node
        for j in range(R):
            if i == j:
                continue
            first_j: int = all_requests[j].pickup_node
            s: float = (
                problem.dist(0, last_i)
                + problem.dist(0, first_j)
                - problem.dist(last_i, first_j)
            )
            if s > 0:
                savings.append((s, i, j))

    # Sort descending; add tiny noise to break ties when seed is set
    if seed is not None:
        savings.sort(key=lambda x: -x[0] + random.uniform(-1e-9, 1e-9))
    else:
        savings.sort(reverse=True)

    # ── Chain data structures ─────────────────────────────────────────────
    # Each request starts as its own single-request chain.
    # is_head[i]: request i is the first in its chain (can receive a predecessor).
    # is_tail[i]: request i is the last in its chain (can receive a successor).
    # chain_id[i]: index into `chains` for the active chain containing i.
    # chains[cid]: ordered list of request indices, or None if merged away.
    # chain_dem[cid]: total parcel demand of chain cid.

    is_head: List[bool] = [True] * R
    is_tail: List[bool] = [True] * R
    chain_id: List[int] = list(range(R))
    chains: List[Optional[List[int]]] = [[i] for i in range(R)]
    chain_dem: List[int] = [parcel_dem(all_requests[i]) for i in range(R)]

    # ── Greedy merging ────────────────────────────────────────────────────
    for _, i, j in savings:
        if not is_tail[i] or not is_head[j]:
            continue

        ci: int = chain_id[i]
        cj: int = chain_id[j]
        if ci == cj:
            continue

        merged_dem: int = chain_dem[ci] + chain_dem[cj]
        if merged_dem > max_cap:
            continue

        # Merge chain_cj onto the tail of chain_ci
        merged: List[int] = chains[ci] + chains[cj]  # type: ignore[operator]
        new_id: int = len(chains)

        for req_idx in merged:
            chain_id[req_idx] = new_id

        chains.append(merged)
        chain_dem.append(merged_dem)
        chains[ci] = None
        chains[cj] = None

        is_tail[i] = False   # i now has a successor
        is_head[j] = False   # j now has a predecessor

    # ── Collect active chains ─────────────────────────────────────────────
    # A chain is active iff its first element is still a head.
    active: List[Tuple[List[int], int]] = []   # (req_indices, parcel_demand)
    for i in range(R):
        if is_head[i]:
            cid = chain_id[i]
            active.append((chains[cid], chain_dem[cid]))  # type: ignore[arg-type]

    # ── Assign chains to vehicles (min-max strategy) ──────────────────────
    def chain_dist_est(req_indices: List[int]) -> float:
        """Estimated route length for a chain (depot → requests → depot)."""
        length: float = 0.0
        prev: int = 0
        for idx in req_indices:
            r = all_requests[idx]
            length += problem.dist(prev, r.pickup_node)
            length += problem.dist(r.pickup_node, r.dropoff_node)
            prev = r.dropoff_node
        return length + problem.dist(prev, 0)

    # Sort chains longest-first so large chains are distributed first
    active.sort(key=lambda x: chain_dist_est(x[0]), reverse=True)

    vehicle_requests: List[List[Request]] = [[] for _ in range(K)]
    vehicle_parcel_load: List[int] = [0] * K
    vehicle_dist_est: List[float] = [0.0] * K

    for req_indices, dem in active:
        best_v: int = -1
        best_d: float = float("inf")
        for v in range(K):
            if vehicle_parcel_load[v] + dem <= vehicle_capacities[v]:
                if vehicle_dist_est[v] < best_d:
                    best_d = vehicle_dist_est[v]
                    best_v = v
        if best_v == -1:
            # Prefer vehicles whose capacity can at least hold this chain's total demand
            can_hold = [v for v in range(K) if vehicle_capacities[v] >= dem]
            pool = can_hold if can_hold else list(range(K))
            best_v = min(pool, key=lambda v: vehicle_dist_est[v])

        vehicle_requests[best_v].extend(all_requests[idx] for idx in req_indices)
        vehicle_parcel_load[best_v] += dem
        vehicle_dist_est[best_v] += chain_dist_est(req_indices)

    # ── NN ordering + assemble ────────────────────────────────────────────
    chromosome: List[int] = _nn_order_and_assemble(problem, vehicle_requests)
    solution = Solution1D(chromosome=chromosome, problem=problem)
    is_valid, msg = solution.validate()
    if not is_valid:
        raise RuntimeError(
            f"savings_init: {msg} | len={len(chromosome)}, "
            f"expected={problem.chromosome_length}"
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
    Apply nearest-neighbor ordering within each vehicle's request list, then
    assemble and return the flat 1D chromosome.

    Starting from the depot, repeatedly pick the unvisited request whose
    pickup is closest to the current position.  Cursor advances to the
    request's dropoff node (the physical last stop for both types).

    Constraint guarantee:
      • Passengers: only pickup_node is appended; decoder inserts dropoff
        immediately after — direct-trip constraint is upheld.
      • Parcels: pickup_node then dropoff_node appended as a unit — parcel
        precedence (pickup before dropoff) is always satisfied.
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
                current_node = req.dropoff_node   # physical cursor to dropoff
            else:
                route.append(req.pickup_node)
                route.append(req.dropoff_node)
                current_node = req.dropoff_node

        chromosome.extend(route)
        if v < K - 1:
            chromosome.append(0)

    return chromosome


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    RANDOM INITIALISATION (FALLBACK)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def random_init(problem: ProblemData, seed: int | None = None) -> Solution1D:
    """
    Random structurally-valid solution for GA population diversity.

    Shuffles all non-zero chromosome positions and places (K-1) depot
    separators at random positions.  Likely infeasible; the penalty-driven
    fitness function guides the search toward feasibility.
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
