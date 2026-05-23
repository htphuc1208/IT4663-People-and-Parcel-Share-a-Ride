"""
init.py — Greedy Initialisation for the Unified 1D Solution
============================================================

Constructs a feasible (or near-feasible) starting solution by:
  1. Assigning passenger requests first — each passenger is a direct trip
     (pickup only in the chromosome; drop-off is implicit).
  2. Inserting parcel requests — both pickup and drop-off node IDs are placed
     into the route of the least-loaded vehicle that still has capacity.
  3. Connecting the K route segments with (K−1) zero separators.

The output is always a valid `Solution1D` whose chromosome has length
  N + 2M + (K − 1).

─────────────────────────────────────────────────────────────
STRATEGY
─────────────────────────────────────────────────────────────
  • Passengers are sorted by earliest pickup time and assigned round-robin
    to balance vehicle load.
  • Parcels are sorted by demand (heaviest first) and inserted into the
    vehicle whose current accumulated distance is shortest (greedy
    load-balancing for the min-max objective).
  • After building per-vehicle node lists, the chromosome is assembled by
    concatenating: route_0 + [0] + route_1 + [0] + … + route_{K−1}.
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .encoding_and_read import ProblemData, Request, RequestType, Solution1D


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     GREEDY INITIALISATION                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def greedy_init(problem: ProblemData, seed: int | None = None) -> Solution1D:
    """
    Build an initial solution using a greedy heuristic.

    Parameters
    ----------
    problem : ProblemData
        Parsed SARP instance.
    seed : int | None
        Optional random seed for reproducibility (used for tie-breaking).

    Returns
    -------
    Solution1D
        A complete 1D-encoded solution with length N + 2M + (K − 1).

    Algorithm
    ---------
    Phase 1 — Assign passengers (direct trips):
        Sort passengers by earliest_pickup.
        Round-robin assign each passenger's **pickup node only** to vehicles.
        (Drop-off is omitted from the chromosome; it's reconstructed in decode.)

    Phase 2 — Insert parcels (pickup + drop-off):
        Sort parcels by demand descending (best-fit decreasing heuristic).
        For each parcel, pick the vehicle with the shortest current route
        distance.  Append **both** pickup and drop-off nodes to that vehicle.

    Phase 3 — Assemble chromosome:
        Concatenate route segments separated by `0` (depot).
    """
    if seed is not None:
        random.seed(seed)

    K: int = problem.K
    N: int = problem.N
    M: int = problem.M

    # Per-vehicle route buffers (each holds node IDs, NO zeros)
    vehicle_routes: List[List[int]] = [[] for _ in range(K)]

    # Track cumulative distance per vehicle (for greedy load-balancing)
    vehicle_dist: List[float] = [0.0] * K

    # Track current load per vehicle
    vehicle_load: List[int] = [0] * K

    # ── Phase 1: Assign passengers ──────────────────────────────────────
    sorted_passengers: List[Request] = sorted(
        problem.passengers,
        key=lambda r: r.earliest_pickup,
    )

    for idx, pax in enumerate(sorted_passengers):
        # Round-robin vehicle assignment
        v: int = idx % K
        # Only the pickup node goes into the chromosome
        vehicle_routes[v].append(pax.pickup_node)
        vehicle_load[v] += pax.demand

        # Update distance estimate: depot → pickup → dropoff → depot
        d_pickup: float = problem.dist(0, pax.pickup_node)
        d_trip: float = problem.dist(pax.pickup_node, pax.dropoff_node)
        d_return: float = problem.dist(pax.dropoff_node, 0)
        vehicle_dist[v] += d_pickup + d_trip + d_return

    # ── Phase 2: Insert parcels ─────────────────────────────────────────
    sorted_parcels: List[Request] = sorted(
        problem.parcels,
        key=lambda r: r.demand,
        reverse=True,  # heaviest first → best-fit decreasing
    )

    for parcel in sorted_parcels:
        # Pick the vehicle with the shortest accumulated distance
        # that still has capacity for this parcel
        best_vehicle: int = -1
        best_dist: float = float("inf")

        for v in range(K):
            if vehicle_load[v] + parcel.demand <= problem.vehicle_capacity:
                if vehicle_dist[v] < best_dist:
                    best_dist = vehicle_dist[v]
                    best_vehicle = v

        # Fallback: if no vehicle has capacity, assign to least loaded anyway
        # (the penalty system in fitness.py will handle the infeasibility)
        if best_vehicle == -1:
            best_vehicle = min(range(K), key=lambda v: vehicle_dist[v])

        # Both pickup AND drop-off node IDs go into the chromosome
        vehicle_routes[best_vehicle].append(parcel.pickup_node)
        vehicle_routes[best_vehicle].append(parcel.dropoff_node)
        vehicle_load[best_vehicle] += parcel.demand

        # Update distance estimate
        d_pickup = problem.dist(0, parcel.pickup_node)
        d_trip = problem.dist(parcel.pickup_node, parcel.dropoff_node)
        d_return = problem.dist(parcel.dropoff_node, 0)
        vehicle_dist[best_vehicle] += d_pickup + d_trip + d_return

    # ── Phase 3: Assemble the 1D chromosome ─────────────────────────────
    chromosome: List[int] = []
    for v in range(K):
        chromosome.extend(vehicle_routes[v])
        if v < K - 1:
            chromosome.append(0)  # depot separator

    solution = Solution1D(chromosome=chromosome, problem=problem)

    # Validate structural integrity
    is_valid, msg = solution.validate()
    if not is_valid:
        raise RuntimeError(
            f"greedy_init produced an invalid chromosome: {msg}\n"
            f"  Expected length = {problem.chromosome_length}, "
            f"  Got = {len(chromosome)}\n"
            f"  Chromosome = {chromosome}"
        )

    return solution


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    RANDOM INITIALISATION (FALLBACK)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def random_init(problem: ProblemData, seed: int | None = None) -> Solution1D:
    """
    Build a random (but structurally valid) initial solution.

    This is a fallback / diversity-injection method.  It shuffles all non-zero
    nodes randomly and interleaves (K−1) zero separators at random positions.

    Parameters
    ----------
    problem : ProblemData
        Parsed SARP instance.
    seed : int | None
        Optional random seed.

    Returns
    -------
    Solution1D
        A random 1D-encoded solution.

    Warning
    -------
    The solution is structurally valid (correct length, correct separator
    count) but is very likely to violate capacity and precedence constraints.
    It relies on the penalty-driven fitness function to guide the search
    toward feasibility.
    """
    if seed is not None:
        random.seed(seed)

    N: int = problem.N
    M: int = problem.M
    K: int = problem.K

    # Collect all non-zero node IDs that belong in the chromosome
    nodes: List[int] = []

    # Passenger pickup nodes: 1 .. N  (drop-offs are implicit)
    for pax in problem.passengers:
        nodes.append(pax.pickup_node)

    # Parcel nodes: both pickup AND drop-off
    for parcel in problem.parcels:
        nodes.append(parcel.pickup_node)
        nodes.append(parcel.dropoff_node)

    random.shuffle(nodes)

    # Insert (K−1) zeros at random positions
    positions: List[int] = sorted(
        random.sample(range(1, len(nodes) + 1), K - 1)
    )
    chromosome: List[int] = []
    node_idx: int = 0
    for pos in positions:
        while node_idx < pos:
            chromosome.append(nodes[node_idx])
            node_idx += 1
        chromosome.append(0)
    # Append remaining nodes
    while node_idx < len(nodes):
        chromosome.append(nodes[node_idx])
        node_idx += 1

    return Solution1D(chromosome=chromosome, problem=problem)
