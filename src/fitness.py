"""
fitness.py — Decoding Logic & Min-Max Objective Calculation
===========================================================

Decodes a Solution1D into full vehicle routes and computes the Min-Max
objective: minimise the MAXIMUM route distance across all K taxis.

Penalties are applied for:
  • Capacity violations   — parcel load exceeds Q[k] at any point
  • Precedence violations — a parcel drop-off is visited before its pickup

Passengers have no capacity constraint.  Their direct-trip constraint is
enforced by the decoder (dropoff auto-inserted immediately after pickup).

─────────────────────────────────────────────────────────────
DECODING PROCEDURE
─────────────────────────────────────────────────────────────
  1. Split chromosome by `0` separators → K route segments.
  2. For each segment:
       Passenger pickup (node 1..N)   → auto-append dropoff (pickup + N + M).
       Parcel node                    → append as-is.
  3. Wrap each segment: depot → [...] → depot.
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .encoding_and_read import ProblemData, Solution1D


PENALTY_CAPACITY: float = 1_000.0
PENALTY_PRECEDENCE: float = 10_000.0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      ROUTE DECODING                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def decode_routes(solution: Solution1D) -> List[List[int]]:
    """
    Decode the 1D chromosome into K full vehicle routes.

    Passenger pickup nodes (1..N) get their dropoff auto-appended immediately,
    enforcing the direct-trip constraint.  Each route is wrapped depot→…→depot.
    """
    prob: ProblemData = solution.problem
    N: int = prob.N
    M: int = prob.M

    full_routes: List[List[int]] = []
    for segment in solution.get_routes():
        route: List[int] = [0]
        for node in segment:
            if 1 <= node <= N:
                route.append(node)
                route.append(node + N + M)
            else:
                route.append(node)
        route.append(0)
        full_routes.append(route)

    return full_routes


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     ROUTE DISTANCE                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _route_distance(route: List[int], prob: ProblemData) -> float:
    total: float = 0.0
    for i in range(len(route) - 1):
        total += prob.dist(route[i], route[i + 1])
    return total


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  CONSTRAINT VIOLATION EVALUATION                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _build_parcel_tables(
    prob: ProblemData,
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build parcel lookup tables once per evaluate call."""
    demand_by_pickup: Dict[int, int] = {}
    pickup_by_dropoff: Dict[int, int] = {}
    for req in prob.parcels:
        demand_by_pickup[req.pickup_node] = req.demand
        pickup_by_dropoff[req.dropoff_node] = req.pickup_node
    return demand_by_pickup, pickup_by_dropoff


def _evaluate_violations(
    route: List[int],
    vehicle_capacity: int,
    demand_by_pickup: Dict[int, int],
    pickup_by_dropoff: Dict[int, int],
) -> float:
    """
    Compute penalty for a single decoded route.

    Checks:
      1. Parcel capacity — running parcel load vs Q[k].
      2. Precedence     — parcel dropoff visited before its pickup.
    """
    penalty: float = 0.0
    parcel_load: int = 0
    visited_pickups: Set[int] = set()

    for node in route[1:-1]:
        if node in demand_by_pickup:
            parcel_load += demand_by_pickup[node]
            visited_pickups.add(node)
            if parcel_load > vehicle_capacity:
                penalty += PENALTY_CAPACITY * (parcel_load - vehicle_capacity)
        elif node in pickup_by_dropoff:
            pickup_node = pickup_by_dropoff[node]
            if pickup_node not in visited_pickups:
                penalty += PENALTY_PRECEDENCE
            parcel_load -= demand_by_pickup.get(pickup_node, 0)

    return penalty


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    MIN-MAX OBJECTIVE FUNCTION                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def evaluate(solution: Solution1D) -> float:
    """
    Min-Max fitness: max over K routes of (route_distance + route_penalty).
    Lower is better.
    """
    prob: ProblemData = solution.problem
    routes: List[List[int]] = decode_routes(solution)
    demand_by_pickup, pickup_by_dropoff = _build_parcel_tables(prob)

    route_costs: List[float] = []
    for k, route in enumerate(routes):
        d: float = _route_distance(route, prob)
        p: float = _evaluate_violations(
            route, prob.vehicle_capacities[k], demand_by_pickup, pickup_by_dropoff
        )
        route_costs.append(d + p)

    return max(route_costs) if route_costs else 0.0


def evaluate_detailed(
    solution: Solution1D,
) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Like evaluate(), but returns (fitness, distances, penalties, total_costs).
    """
    prob: ProblemData = solution.problem
    routes: List[List[int]] = decode_routes(solution)
    demand_by_pickup, pickup_by_dropoff = _build_parcel_tables(prob)

    distances: List[float] = []
    penalties: List[float] = []
    total_costs: List[float] = []

    for k, route in enumerate(routes):
        d: float = _route_distance(route, prob)
        p: float = _evaluate_violations(
            route, prob.vehicle_capacities[k], demand_by_pickup, pickup_by_dropoff
        )
        distances.append(d)
        penalties.append(p)
        total_costs.append(d + p)

    fitness: float = max(total_costs) if total_costs else 0.0
    return fitness, distances, penalties, total_costs
