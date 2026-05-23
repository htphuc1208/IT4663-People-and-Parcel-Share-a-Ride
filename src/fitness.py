"""
fitness.py — Decoding Logic & Min-Max Objective Calculation
===========================================================

This module takes a `Solution1D` object, decodes it into full vehicle routes
(reinserting implicit passenger drop-offs), and computes the **Min-Max**
objective: minimise the MAXIMUM route distance across all K taxis.

Heavy penalties are applied for:
  • Capacity violations   (exceeding Q at any point along a route)
  • Precedence violations (parcel drop-off appears before its pickup)
  • Time-window violations (arriving outside the allowed window)
  • Ride-time violations   (passenger/parcel spends too long in the vehicle)

─────────────────────────────────────────────────────────────
DECODING PROCEDURE
─────────────────────────────────────────────────────────────
  1. Split the 1D chromosome by `0` separators → K route segments.
  2. For each segment, iterate over node IDs:
       a. If the node is a **passenger pickup** (node ∈ [1..N]):
          → automatically append the corresponding drop-off node
            (drop-off ID = pickup_id + N + M) immediately after the pickup.
       b. If the node is a **parcel pickup or drop-off** (node ∈ [N+1..N+2M]
          or node ∈ [2N+M+1..2N+2M]):
          → append it as-is (both pickup & drop-off are explicit in the chromosome).
  3. Wrap each decoded segment with depot → … → depot.
  4. Evaluate distance, capacity, and constraint violations along the route.
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .encoding_and_read import ProblemData, RequestType, Solution1D


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           CONSTANTS                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Penalty weights — tune these to guide the search toward feasibility.
PENALTY_CAPACITY: float = 1_000.0    # per unit of capacity overflow
PENALTY_PRECEDENCE: float = 10_000.0 # per precedence violation (parcel drop before pick)
PENALTY_TIME_WINDOW: float = 500.0   # per unit of time-window violation
PENALTY_RIDE_TIME: float = 500.0     # per unit of max-ride-time violation


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      ROUTE DECODING                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def decode_routes(solution: Solution1D) -> List[List[int]]:
    """
    Decode the 1D chromosome into K full vehicle routes.

    For each route segment (obtained by splitting on `0` separators):
      • Passenger pickup nodes get their drop-off auto-appended immediately.
      • Parcel nodes pass through unchanged (pickup AND drop-off already present).
      • Each route is wrapped:  depot(0) → [nodes] → depot(0).

    Parameters
    ----------
    solution : Solution1D
        The unified 1D-encoded solution.

    Returns
    -------
    List[List[int]]
        K routes, each as a list of node IDs starting and ending at depot (0).

    Example
    -------
    >>> # N=2, M=1, K=2  →  chromosome = [1, 3, 6, 0, 2]
    >>> # Passenger 1: pickup=1, dropoff=1+2+1=4
    >>> # Passenger 2: pickup=2, dropoff=2+2+1=5
    >>> # Parcel 1:    pickup=3, dropoff=6  (both explicit)
    >>> # Route 1: [0, 1, 4, 3, 6, 0]   (passenger 1 direct, then parcel)
    >>> # Route 2: [0, 2, 5, 0]          (passenger 2 direct)
    """
    prob: ProblemData = solution.problem
    N: int = prob.N  # number of passengers
    M: int = prob.M  # number of parcels

    raw_segments: List[List[int]] = solution.get_routes()
    full_routes: List[List[int]] = []

    for segment in raw_segments:
        route: List[int] = [0]  # start at depot

        for node in segment:
            if 1 <= node <= N:
                # ── Passenger pickup → auto-append drop-off ──────────
                # Drop-off node ID = pickup + N + M
                dropoff: int = node + N + M
                route.append(node)
                route.append(dropoff)
            else:
                # ── Parcel node (pickup or drop-off): append as-is ───
                route.append(node)

        route.append(0)  # return to depot
        full_routes.append(route)

    return full_routes


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     ROUTE DISTANCE CALCULATION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _route_distance(route: List[int], prob: ProblemData) -> float:
    """
    Compute the total Euclidean travel distance along a single route.

    Parameters
    ----------
    route : List[int]
        Fully decoded route [depot, ..., depot].
    prob : ProblemData
        Instance data with pre-computed distance matrix.

    Returns
    -------
    float
        Sum of edge distances along the route.
    """
    total: float = 0.0
    for i in range(len(route) - 1):
        total += prob.dist(route[i], route[i + 1])
    return total


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  CONSTRAINT VIOLATION EVALUATION                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _evaluate_violations(
    route: List[int],
    prob: ProblemData,
) -> float:
    """
    Walk along a fully decoded route and compute the total penalty for:
      1. Capacity violations  — current load exceeds vehicle_capacity.
      2. Precedence violations — a parcel drop-off is visited before its pickup.
      3. Time-window violations — arrival is outside [earliest, latest].
      4. Ride-time violations  — a request stays in the vehicle too long.

    Parameters
    ----------
    route : List[int]
        Fully decoded route [depot, ..., depot].
    prob : ProblemData
        Instance data.

    Returns
    -------
    float
        Sum of weighted penalty values (0 if the route is fully feasible).
    """
    penalty: float = 0.0
    N: int = prob.N
    M: int = prob.M
    Q: int = prob.vehicle_capacity

    # ── Build a quick lookup: node_id → Request ──────────────────────────
    request_by_pickup: Dict[int, "Request"] = {}   # type: ignore[name-defined]
    request_by_dropoff: Dict[int, "Request"] = {}  # type: ignore[name-defined]
    for req in prob.all_requests():
        request_by_pickup[req.pickup_node] = req
        request_by_dropoff[req.dropoff_node] = req

    # ── Simulate the route ───────────────────────────────────────────────
    current_load: int = 0
    current_time: float = 0.0
    visited_pickups: set = set()     # track which pickups have been visited
    pickup_times: Dict[int, float] = {}  # pickup_node → time picked up

    for idx in range(1, len(route) - 1):  # skip leading & trailing depot
        prev_node: int = route[idx - 1]
        node: int = route[idx]

        # Travel time ≈ travel distance (speed = 1)
        travel: float = prob.dist(prev_node, node)
        current_time += travel

        # ── Check if this is a pickup node ──────────────────────────────
        if node in request_by_pickup:
            req = request_by_pickup[node]

            # Apply time window on pickup
            if current_time < req.earliest_pickup:
                current_time = req.earliest_pickup  # wait
            if current_time > req.latest_pickup:
                penalty += PENALTY_TIME_WINDOW * (current_time - req.latest_pickup)

            current_time += req.service_time_pickup
            current_load += req.demand
            visited_pickups.add(node)
            pickup_times[node] = current_time

            # Capacity check
            if current_load > Q:
                penalty += PENALTY_CAPACITY * (current_load - Q)

        # ── Check if this is a drop-off node ────────────────────────────
        elif node in request_by_dropoff:
            req = request_by_dropoff[node]

            # Precedence: pickup must have been visited first
            if req.pickup_node not in visited_pickups:
                penalty += PENALTY_PRECEDENCE

            # Apply time window on dropoff
            if current_time < req.earliest_dropoff:
                current_time = req.earliest_dropoff
            if current_time > req.latest_dropoff:
                penalty += PENALTY_TIME_WINDOW * (current_time - req.latest_dropoff)

            current_time += req.service_time_dropoff
            current_load -= req.demand

            # Ride time check
            if req.pickup_node in pickup_times:
                ride: float = current_time - pickup_times[req.pickup_node]
                if ride > req.max_ride_time:
                    penalty += PENALTY_RIDE_TIME * (ride - req.max_ride_time)

    return penalty


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    MIN-MAX OBJECTIVE FUNCTION                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def evaluate(solution: Solution1D) -> float:
    """
    Compute the **Min-Max** fitness of a Solution1D.

    Objective = max over all K routes { route_distance + route_penalty }

    A lower value is better. Penalties inflate the objective to drive the
    search away from infeasible regions.

    Parameters
    ----------
    solution : Solution1D
        The unified 1D-encoded solution.

    Returns
    -------
    float
        The fitness value (max route cost across all vehicles).
    """
    prob: ProblemData = solution.problem
    routes: List[List[int]] = decode_routes(solution)

    route_costs: List[float] = []
    for route in routes:
        dist: float = _route_distance(route, prob)
        penalty: float = _evaluate_violations(route, prob)
        route_costs.append(dist + penalty)

    # Min-Max objective: the bottleneck (longest) route determines fitness
    return max(route_costs) if route_costs else 0.0


def evaluate_detailed(
    solution: Solution1D,
) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Like `evaluate`, but also returns per-route breakdowns.

    Returns
    -------
    (fitness, distances, penalties, total_costs)
        fitness     — max(total_costs)
        distances   — [dist_route_0, dist_route_1, ...]
        penalties   — [penalty_route_0, penalty_route_1, ...]
        total_costs — [dist+pen for each route]
    """
    prob: ProblemData = solution.problem
    routes: List[List[int]] = decode_routes(solution)

    distances: List[float] = []
    penalties: List[float] = []
    total_costs: List[float] = []

    for route in routes:
        d: float = _route_distance(route, prob)
        p: float = _evaluate_violations(route, prob)
        distances.append(d)
        penalties.append(p)
        total_costs.append(d + p)

    fitness: float = max(total_costs) if total_costs else 0.0
    return fitness, distances, penalties, total_costs
