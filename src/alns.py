"""
alns.py — Adaptive Large Neighbourhood Search for the SARP (Min-Max)
=====================================================================

This module implements ALNS, which iteratively destroys and repairs the
current solution using a portfolio of operators.  Operator selection is
governed by an **adaptive roulette-wheel** mechanism that rewards operators
producing improvements.

═══════════════════════════════════════════════════════════════════════════
DESIGN NOTES (revised) — REQUEST-LEVEL, MIN-MAX-AWARE OPERATORS
═══════════════════════════════════════════════════════════════════════════
Unlike a textbook total-distance VRP, this problem has three peculiarities
that the operators below are built around:

  1. MIN-MAX objective.  We minimise the LONGEST route, not the sum.  So
     destroy/repair are scored by the resulting *maximum* route length, with
     a bias toward off-loading work onto the currently shortest routes.

  2. PASSENGER direct-trip.  A passenger gene `i` decodes to the pair
     (pickup i → dropoff i+N+M).  All cost estimates expand the gene to its
     two real nodes (prev→pickup→dropoff→next), never treat it as one node.

  3. PARCEL coupling + capacity.  A parcel owns two genes (pickup, dropoff)
     and a demand q.  Operators work on *requests* (logical units), so a
     parcel's two genes are always removed together and re-inserted as a
     precedence-safe, capacity-feasible pair.  This eliminates the 10 000 /
     1 000 penalties at the source instead of paying them and repairing.

WORKING REPRESENTATION
----------------------
Destroy/repair operate on `routes : List[List[int]]` — the K chromosome
*segments* (non-zero genes only), obtained from `Solution1D.get_routes()`.
After an operator runs, `_routes_to_chromosome` rejoins them with the K−1
zero separators.  Zeros are therefore never touched explicitly: an empty
route is simply an empty inner list.

  • Destroy:  routes  -> (routes', removed: List[Request])
  • Repair :  routes', removed -> routes''   (all removed requests re-inserted)

ADAPTIVE ROULETTE-WHEEL SELECTION (unchanged in spirit)
-------------------------------------------------------
Each operator has a weight; selection is fitness-proportionate.  After a
(destroy, repair) pair we award a reward (σ₁ best / σ₂ improve / σ₃ accepted
/ 0 reject) and, at each segment boundary, blend it into the weight:
      w_new = (1 − r) · w_old + r · (π / θ).
Rewards are kept on the SAME numeric scale as the initial weights (≈1) so the
roulette wheel stays balanced from the first segment.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .encoding_and_read import ProblemData, Request, RequestType, Solution1D
from .fitness import evaluate
from .init import greedy_init, minmax_greedy_init, savings_init

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         ALNS PARAMETERS                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ALNSConfig:
    """Configuration for ALNS."""
    max_iterations: int = 2000
    segment_length: int = 100       # iterations per weight-update segment

    # Destruction degree: fraction of REQUESTS (N+M) to remove
    destroy_fraction_min: float = 0.10
    destroy_fraction_max: float = 0.40

    # ── Adaptive weight parameters ───────────────────────────────────────
    # Rewards are normalised to ~[0, 1] so they share the scale of the
    # uniformly-initialised weights (1.0) — keeps the roulette wheel balanced.
    reaction_factor: float = 0.15   # r : learning rate for weight update
    sigma_1: float = 1.00           # reward: new global best
    sigma_2: float = 0.40           # reward: improves current
    sigma_3: float = 0.15           # reward: accepted (worse but accepted)
    weight_min: float = 0.05        # floor so no operator is ever starved

    # ── Simulated Annealing acceptance ───────────────────────────────────
    sa_start_temperature: float = 100.0
    sa_cooling_rate: float = 0.9995

    # ── Restart / re-heat ────────────────────────────────────────────────
    # If no new global best for this many segments, snap `current` back to the
    # best solution and re-heat the temperature to intensify around the best.
    restart_segments: int = 15
    reheat_factor: float = 0.5      # T <- max(T, reheat_factor * T0) on restart

    seed: int | None = None
    time_limit_seconds: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 LOW-LEVEL ROUTE / GENE GEOMETRY HELPERS                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _gene_head(gene: int) -> int:
    """First decoded node when entering a gene (== the gene itself)."""
    return gene


def _gene_tail(gene: int, N: int, M: int) -> int:
    """
    Last decoded node when leaving a gene.

    Passenger pickup `i` decodes to (i, i+N+M) → tail is the dropoff i+N+M.
    Parcel pickup/dropoff genes decode to themselves → tail is the gene.
    """
    return gene + N + M if 1 <= gene <= N else gene


def _segment_distance(segment: List[int], prob: ProblemData) -> float:
    """Full decoded distance of one route segment, depot → … → depot."""
    N, M = prob.N, prob.M
    prev: int = 0
    total: float = 0.0
    for g in segment:
        if 1 <= g <= N:                      # passenger expands to pickup,dropoff
            drop = g + N + M
            total += prob.dist(prev, g) + prob.dist(g, drop)
            prev = drop
        else:
            total += prob.dist(prev, g)
            prev = g
    total += prob.dist(prev, 0)
    return total


def _routes_distances(routes: List[List[int]], prob: ProblemData) -> List[float]:
    """Per-route decoded distance."""
    return [_segment_distance(seg, prob) for seg in routes]


def _routes_to_chromosome(routes: List[List[int]]) -> List[int]:
    """Rejoin K segments with K−1 zero separators (empty route → '' between zeros)."""
    chrom: List[int] = []
    for i, seg in enumerate(routes):
        if i > 0:
            chrom.append(0)
        chrom.extend(seg)
    return chrom


def _top_two(dists: List[float]) -> Tuple[float, int, float]:
    """Return (max value, its index, second-max value).  Used for O(1) max-excluding-k."""
    max1v: float = -math.inf
    max1i: int = -1
    max2v: float = -math.inf
    for i, d in enumerate(dists):
        if d > max1v:
            max2v, max1v, max1i = max1v, d, i
        elif d > max2v:
            max2v = d
    return max1v, max1i, max2v


def _request_gene_set(req: Request) -> set:
    """Genes a request occupies in the chromosome (1 for passenger, 2 for parcel)."""
    if req.request_type == RequestType.PARCEL:
        return {req.pickup_node, req.dropoff_node}
    return {req.pickup_node}


def _segment_requests(segment: List[int], prob: ProblemData) -> List[Request]:
    """Distinct requests appearing in a segment (a parcel counted once)."""
    seen: set = set()
    out: List[Request] = []
    for g in segment:
        req = prob.request_by_node[g]
        key = (req.request_type, req.request_id)
        if key not in seen:
            seen.add(key)
            out.append(req)
    return out


def _remove_requests(
    routes: List[List[int]], reqs: List[Request], prob: ProblemData,
) -> None:
    """Strip every gene of every request in `reqs` from `routes` (in place)."""
    genes: set = set()
    for req in reqs:
        genes |= _request_gene_set(req)
    if not genes:
        return
    for k in range(len(routes)):
        routes[k] = [g for g in routes[k] if g not in genes]


def _request_savings(segment: List[int], prob: ProblemData) -> List[Tuple[float, Request]]:
    """
    For each request in a segment, the distance that would be saved by removing
    it (sum of its genes' local removal deltas).  Passenger genes account for
    the implicit dropoff; parcel genes are summed independently.
    """
    N, M, n = prob.N, prob.M, len(segment)
    agg: dict = {}
    for idx in range(n):
        g = segment[idx]
        prev = _gene_tail(segment[idx - 1], N, M) if idx > 0 else 0
        nxt = _gene_head(segment[idx + 1]) if idx + 1 < n else 0
        if 1 <= g <= N:
            drop = g + N + M
            sv = (prob.dist(prev, g) + prob.dist(g, drop)
                  + prob.dist(drop, nxt) - prob.dist(prev, nxt))
        else:
            sv = prob.dist(prev, g) + prob.dist(g, nxt) - prob.dist(prev, nxt)
        req = prob.request_by_node[g]
        key = (req.request_type, req.request_id)
        if key in agg:
            agg[key] = (agg[key][0] + sv, req)
        else:
            agg[key] = (sv, req)
    return list(agg.values())


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        DESTROY OPERATORS                                 ║
# ║   signature: (routes, problem, num_remove) -> (routes', removed reqs)    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def destroy_random(
    routes: List[List[int]], prob: ProblemData, num_remove: int,
) -> Tuple[List[List[int]], List[Request]]:
    """Random Removal — remove `num_remove` randomly chosen *requests*."""
    routes = [list(r) for r in routes]
    all_reqs: List[Request] = [req for seg in routes for req in _segment_requests(seg, prob)]
    k = min(num_remove, len(all_reqs))
    if k == 0:
        return routes, []
    chosen = random.sample(all_reqs, k)
    _remove_requests(routes, chosen, prob)
    return routes, chosen


def destroy_worst(
    routes: List[List[int]], prob: ProblemData, num_remove: int, p: float = 4.0,
) -> Tuple[List[List[int]], List[Request]]:
    """
    Worst Removal — remove the requests whose removal saves the most distance.
    A randomisation exponent `p` (Shaw-style) avoids always picking the same
    extreme genes: index = floor(len * U^p).
    """
    routes = [list(r) for r in routes]
    scored: List[Tuple[float, Request]] = []
    for seg in routes:
        scored.extend(_request_savings(seg, prob))
    if not scored:
        return routes, []
    scored.sort(key=lambda x: x[0], reverse=True)

    k = min(num_remove, len(scored))
    chosen: List[Request] = []
    pool = scored
    for _ in range(k):
        idx = int(len(pool) * (random.random() ** p))
        chosen.append(pool.pop(idx)[1])
    _remove_requests(routes, chosen, prob)
    return routes, chosen


def _relatedness(a: Request, b: Request, prob: ProblemData) -> float:
    """Shaw relatedness — lower means more related (closer to remove together)."""
    d = prob.dist(a.pickup_node, b.pickup_node)
    demand_diff = abs(a.demand - b.demand)
    type_pen = 0.0 if a.request_type == b.request_type else 50.0
    return d + 0.5 * demand_diff + type_pen


def destroy_related(
    routes: List[List[int]], prob: ProblemData, num_remove: int, p: float = 4.0,
) -> Tuple[List[List[int]], List[Request]]:
    """Shaw / Related Removal — grow a cluster of mutually-related requests."""
    routes = [list(r) for r in routes]
    all_reqs: List[Request] = [req for seg in routes for req in _segment_requests(seg, prob)]
    target = min(num_remove, len(all_reqs))
    if target == 0:
        return routes, []

    seed = random.choice(all_reqs)
    chosen: List[Request] = [seed]
    remaining = [r for r in all_reqs if r is not seed]

    while len(chosen) < target and remaining:
        ref = random.choice(chosen)
        remaining.sort(key=lambda r: _relatedness(ref, r, prob))
        idx = int(len(remaining) * (random.random() ** p))
        chosen.append(remaining.pop(idx))

    _remove_requests(routes, chosen, prob)
    return routes, chosen


def destroy_bottleneck(
    routes: List[List[int]], prob: ProblemData, num_remove: int,
) -> Tuple[List[List[int]], List[Request]]:
    """
    Min-Max specific — peel the worst-contributing requests off the LONGEST
    route(s) first, directly lowering the objective's ceiling.  Spills onto the
    next-longest route if the bottleneck does not hold `num_remove` requests.
    """
    routes = [list(r) for r in routes]
    dists = _routes_distances(routes, prob)
    order = sorted(range(len(routes)), key=lambda k: dists[k], reverse=True)

    chosen: List[Request] = []
    need = num_remove
    for k in order:
        if need <= 0:
            break
        ranked = _request_savings(routes[k], prob)
        ranked.sort(key=lambda x: x[0], reverse=True)   # worst (most saving) first
        for _, req in ranked:
            if need <= 0:
                break
            chosen.append(req)
            need -= 1
    if not chosen:
        return routes, []
    _remove_requests(routes, chosen, prob)
    return routes, chosen


def destroy_route(
    routes: List[List[int]], prob: ProblemData, num_remove: int,
) -> Tuple[List[List[int]], List[Request]]:
    """
    Whole-route Removal — empty one taxi entirely and dump all its requests
    into the pool.  Excellent for re-balancing a skewed Min-Max assignment.
    Longer routes are preferred (weighted random).  `num_remove` is ignored.
    """
    routes = [list(r) for r in routes]
    nonempty = [k for k in range(len(routes)) if routes[k]]
    if not nonempty:
        return routes, []
    dists = _routes_distances(routes, prob)
    k = random.choices(nonempty, weights=[dists[i] + 1.0 for i in nonempty])[0]
    chosen = _segment_requests(routes[k], prob)
    routes[k] = []
    return routes, chosen


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                  INSERTION GEOMETRY (shared by repairs)                  ║
# ║  A "move" encodes where a request goes:                                  ║
# ║     ('P', gap)        passenger pickup at gene-gap `gap`                 ║
# ║     ('C', i, j)       parcel pickup at gap i, dropoff at gap j  (i <= j) ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _best_in_route(
    segment: List[int],
    req: Request,
    prob: ProblemData,
    capacity: int,
) -> Optional[Tuple[float, tuple]]:
    """
    Cheapest *feasible* insertion of `req` into a single segment.

    Returns (delta_distance, move) or None if the request cannot be placed in
    this route without violating capacity (only possible for parcels).

    • Passenger: expands to pickup→dropoff; every gap is feasible.
    • Parcel:    searches (i, j) gap pairs with i ≤ j so the pickup precedes the
      dropoff; prunes positions where the running parcel load would exceed Q.
    """
    N, M = prob.N, prob.M
    n = len(segment)

    # Pre-compute the decoded boundary nodes around every gap once.
    prevs = [(_gene_tail(segment[g - 1], N, M) if g > 0 else 0) for g in range(n + 1)]
    nexts = [(_gene_head(segment[g]) if g < n else 0) for g in range(n + 1)]

    if req.request_type == RequestType.PASSENGER:
        pk = req.pickup_node
        dr = pk + N + M
        best: Optional[Tuple[float, tuple]] = None
        for g in range(n + 1):
            a, b = prevs[g], nexts[g]
            delta = (prob.dist(a, pk) + prob.dist(pk, dr)
                     + prob.dist(dr, b) - prob.dist(a, b))
            if best is None or delta < best[0]:
                best = (delta, ('P', g))
        return best

    # ── Parcel: coupled, capacity-feasible (i, j) search ─────────────────
    pk, dr, q = req.pickup_node, req.dropoff_node, req.demand
    if q > capacity:
        return None

    # Running parcel load right before each gap: load[p] = load after genes[:p].
    load = [0] * (n + 1)
    cur = 0
    demand_by_pickup = prob.demand_by_pickup
    pickup_by_dropoff = prob.pickup_by_dropoff
    for idx in range(n):
        g = segment[idx]
        if g in demand_by_pickup:
            cur += demand_by_pickup[g]
        elif g in pickup_by_dropoff:
            cur -= demand_by_pickup[pickup_by_dropoff[g]]
        load[idx + 1] = cur

    # Independent single-gap insertion deltas (valid & additive when i < j).
    dp = [prob.dist(prevs[g], pk) + prob.dist(pk, nexts[g]) - prob.dist(prevs[g], nexts[g])
          for g in range(n + 1)]
    dd = [prob.dist(prevs[g], dr) + prob.dist(dr, nexts[g]) - prob.dist(prevs[g], nexts[g])
          for g in range(n + 1)]

    best = None
    for i in range(n + 1):
        if load[i] + q > capacity:          # carrying region starts here
            continue
        region_max = load[i]
        for j in range(i, n + 1):
            region_max = max(region_max, load[j])
            if region_max + q > capacity:   # load only grows with j → stop
                break
            if i == j:                      # pickup & dropoff adjacent: one gap
                a, b = prevs[i], nexts[i]
                delta = (prob.dist(a, pk) + prob.dist(pk, dr)
                         + prob.dist(dr, b) - prob.dist(a, b))
            else:                           # two disjoint gaps → additive
                delta = dp[i] + dd[j]
            if best is None or delta < best[0]:
                best = (delta, ('C', i, j))
    return best


def _force_insert_move(
    segment: List[int], req: Request, prob: ProblemData,
) -> Tuple[float, tuple]:
    """
    Last-resort insertion ignoring capacity (keeps every request present so the
    chromosome stays complete).  Parcels go in adjacent (precedence still safe);
    the penalty in `evaluate` will discourage such placements.
    """
    N, M = prob.N, prob.M
    n = len(segment)
    prevs = [(_gene_tail(segment[g - 1], N, M) if g > 0 else 0) for g in range(n + 1)]
    nexts = [(_gene_head(segment[g]) if g < n else 0) for g in range(n + 1)]

    if req.request_type == RequestType.PASSENGER:
        pk, dr = req.pickup_node, req.pickup_node + N + M
    else:
        pk, dr = req.pickup_node, req.dropoff_node

    best: Optional[Tuple[float, tuple]] = None
    for g in range(n + 1):
        a, b = prevs[g], nexts[g]
        delta = prob.dist(a, pk) + prob.dist(pk, dr) + prob.dist(dr, b) - prob.dist(a, b)
        if best is None or delta < best[0]:
            best = (delta, ('P', g) if req.request_type == RequestType.PASSENGER
                    else ('C', g, g))
    return best  # type: ignore[return-value]


def _apply_move(segment: List[int], req: Request, move: tuple) -> None:
    """Mutate `segment` to realise `move` (insert dropoff first to keep indices valid)."""
    if move[0] == 'P':
        segment.insert(move[1], req.pickup_node)
    else:
        _, i, j = move
        segment.insert(j, req.dropoff_node)   # j >= i, so inserting it first
        segment.insert(i, req.pickup_node)     # leaves index i untouched


def _minmax_cost(dist_after_k: float, k: int, max1v: float, max1i: int, max2v: float) -> float:
    """Resulting maximum route length if route k becomes `dist_after_k`."""
    other_max = max2v if k == max1i else max1v
    return dist_after_k if dist_after_k > other_max else other_max


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         REPAIR OPERATORS                                 ║
# ║   signature: (routes, removed reqs, problem) -> routes'                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _insert_request_minmax(
    routes: List[List[int]], dists: List[float], req: Request, prob: ProblemData,
) -> None:
    """
    Greedily insert one request at the position minimising the resulting MAX
    route length (tie-break: smallest distance delta).  Updates `dists`.
    """
    max1v, max1i, max2v = _top_two(dists)
    best: Optional[Tuple[Tuple[float, float], int, float, tuple]] = None

    for k in range(len(routes)):
        res = _best_in_route(routes[k], req, prob, prob.vehicle_capacities[k])
        if res is None:
            continue
        delta, move = res
        cand_max = _minmax_cost(dists[k] + delta, k, max1v, max1i, max2v)
        key = (cand_max, delta)
        if best is None or key < best[0]:
            best = (key, k, delta, move)

    if best is None:   # no capacity-feasible route → forced placement
        k = max(range(len(routes)), key=lambda i: prob.vehicle_capacities[i])
        delta, move = _force_insert_move(routes[k], req, prob)
        best = ((math.inf, delta), k, delta, move)

    _, k, delta, move = best
    _apply_move(routes[k], req, move)
    dists[k] += delta


def repair_greedy(
    routes: List[List[int]], removed: List[Request], prob: ProblemData,
) -> List[List[int]]:
    """
    Min-Max Greedy Repair — re-insert requests one at a time, each at its
    globally best (min resulting maximum) feasible position.  Parcels go in
    coupled & capacity-checked; passengers expand to their direct trip.
    """
    routes = [list(r) for r in routes]
    dists = _routes_distances(routes, prob)
    pool = list(removed)
    random.shuffle(pool)
    for req in pool:
        _insert_request_minmax(routes, dists, req, prob)
    return routes


def repair_random(
    routes: List[List[int]], removed: List[Request], prob: ProblemData,
) -> List[List[int]]:
    """
    Random Repair — drop each request into a random route at a random gap.
    Parcels are inserted ADJACENT (pickup immediately followed by dropoff) so
    precedence is always respected even though the position is random.
    """
    routes = [list(r) for r in routes]
    pool = list(removed)
    random.shuffle(pool)
    for req in pool:
        k = random.randrange(len(routes))
        seg = routes[k]
        if req.request_type == RequestType.PASSENGER:
            seg.insert(random.randint(0, len(seg)), req.pickup_node)
        else:
            gap = random.randint(0, len(seg))
            seg.insert(gap, req.dropoff_node)
            seg.insert(gap, req.pickup_node)   # → [..., pickup, dropoff, ...]
    return routes


def repair_regret(
    routes: List[List[int]], removed: List[Request], prob: ProblemData, k_regret: int = 3,
) -> List[List[int]]:
    """
    Min-Max Regret-k Repair — at each step insert the request we would most
    "regret" leaving out: the one whose 2nd…kth best min-max costs exceed its
    best by the largest margin.  Greedy ties are broken by lower best cost.

    Efficiency: the per-(request, route) insertion delta is cached and only the
    single route changed by an insertion is recomputed, so a full fill is
    O(pool · (pool + work_on_changed_route)) rather than O(pool² · L).
    """
    routes = [list(r) for r in routes]
    dists = _routes_distances(routes, prob)
    pool = list(removed)
    nroutes = len(routes)

    def key_of(r: Request) -> tuple:
        return (r.request_type, r.request_id)

    # cache[reqkey][k] = (delta, move) or None  (None = infeasible in route k)
    cache: dict = {}
    for req in pool:
        cache[key_of(req)] = [
            _best_in_route(routes[k], req, prob, prob.vehicle_capacities[k])
            for k in range(nroutes)
        ]

    while pool:
        max1v, max1i, max2v = _top_two(dists)
        best_req: Optional[Request] = None
        best_score: Optional[tuple] = None
        best_choice: Optional[tuple] = None     # (delta, k, move)

        for req in pool:
            entries = cache[key_of(req)]
            costs: List[Tuple[float, float, int, tuple]] = []
            for k, res in enumerate(entries):
                if res is None:
                    continue
                delta, move = res
                cand_max = _minmax_cost(dists[k] + delta, k, max1v, max1i, max2v)
                costs.append((cand_max, delta, k, move))

            if not costs:   # infeasible everywhere → force into roomiest route
                kk = max(range(nroutes), key=lambda i: prob.vehicle_capacities[i])
                delta, move = _force_insert_move(routes[kk], req, prob)
                costs = [(math.inf, delta, kk, move)]

            costs.sort(key=lambda x: (x[0], x[1]))
            best_cost = costs[0][0]
            regret = sum(costs[t][0] - best_cost
                         for t in range(1, min(k_regret, len(costs))))
            # maximise regret; tie-break: prefer the cheaper best insertion
            score = (regret, -best_cost)
            if best_score is None or score > best_score:
                best_score = score
                best_req = req
                best_choice = (costs[0][1], costs[0][2], costs[0][3])

        assert best_req is not None and best_choice is not None
        delta, k, move = best_choice
        _apply_move(routes[k], best_req, move)
        dists[k] += delta
        pool.remove(best_req)
        del cache[key_of(best_req)]

        # only route k changed → refresh that column of the cache
        for req in pool:
            cache[key_of(req)][k] = _best_in_route(
                routes[k], req, prob, prob.vehicle_capacities[k]
            )

    return routes


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    OPERATOR WEIGHT MANAGEMENT                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class OperatorStats:
    """Track score and usage count for one operator within a segment."""
    score: float = 0.0
    uses: int = 0


def roulette_wheel_select(weights: List[float]) -> int:
    """Fitness-proportionate selection of an index."""
    total: float = sum(weights)
    if total <= 0:
        return random.randrange(len(weights))
    r: float = random.uniform(0, total)
    cumulative: float = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if cumulative >= r:
            return i
    return len(weights) - 1


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          ALNS MAIN LOOP                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

DestroyOp = Callable[[List[List[int]], ProblemData, int], Tuple[List[List[int]], List[Request]]]
RepairOp = Callable[[List[List[int]], List[Request], ProblemData], List[List[int]]]


def run_alns(
    problem: ProblemData,
    config: ALNSConfig | None = None,
    initial_solution: Solution1D | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Execute the ALNS algorithm.

    Returns (best_solution, best_fitness, history).
    """
    if config is None:
        config = ALNSConfig()
    if config.seed is not None:
        random.seed(config.seed)

    start_time: float = time.time()

    # ── Initial solution ─────────────────────────────────────────────────
    if initial_solution is None:
        _seed0: int = config.seed if config.seed is not None else 0
        _candidates: List[Solution1D] = [
            greedy_init(problem),
            greedy_init(problem, seed=_seed0),
            minmax_greedy_init(problem),
            savings_init(problem),
        ]
        _fits: List[float] = [evaluate(s) for s in _candidates]
        _best_idx: int = _fits.index(min(_fits))
        current: Solution1D = _candidates[_best_idx]
        logger.info(
            "ALNS init | 4 heuristics evaluated | best_init_fitness=%.4f", _fits[_best_idx]
        )
    else:
        current = initial_solution.copy()

    current_fitness: float = evaluate(current)
    best_solution: Solution1D = current.copy()
    best_fitness: float = current_fitness
    history: List[float] = [best_fitness]

    total_requests: int = problem.N + problem.M

    # ── Register operators ───────────────────────────────────────────────
    destroy_ops: List[DestroyOp] = [
        destroy_random,
        destroy_worst,
        destroy_related,
        destroy_bottleneck,
        destroy_route,
    ]
    repair_ops: List[RepairOp] = [
        repair_greedy,
        repair_random,
        repair_regret,
    ]
    destroy_names: List[str] = ["random", "worst", "related", "bottleneck", "route"]
    repair_names: List[str] = ["greedy", "random", "regret-3"]

    num_destroy: int = len(destroy_ops)
    num_repair: int = len(repair_ops)
    destroy_weights: List[float] = [1.0] * num_destroy
    repair_weights: List[float] = [1.0] * num_repair

    destroy_stats: List[OperatorStats] = [OperatorStats() for _ in range(num_destroy)]
    repair_stats: List[OperatorStats] = [OperatorStats() for _ in range(num_repair)]

    temperature: float = config.sa_start_temperature
    last_improve_iter: int = 0
    restart_patience: int = config.restart_segments * config.segment_length

    logger.info(
        "ALNS started | max_iter=%d | segment=%d | T0=%.2f | requests=%d | "
        "destroy=%s | repair=%s",
        config.max_iterations, config.segment_length, temperature, total_requests,
        destroy_names, repair_names,
    )

    # ── Main loop ────────────────────────────────────────────────────────
    for iteration in range(1, config.max_iterations + 1):
        if time.time() - start_time >= config.time_limit_seconds:
            logger.info("ALNS time limit reached at iteration %d", iteration)
            break

        # Destruction degree (as a number of REQUESTS).
        num_remove: int = random.randint(
            max(1, int(total_requests * config.destroy_fraction_min)),
            max(1, int(total_requests * config.destroy_fraction_max)),
        )

        d_idx: int = roulette_wheel_select(destroy_weights)
        r_idx: int = roulette_wheel_select(repair_weights)

        routes: List[List[int]] = current.get_routes()
        partial_routes, removed = destroy_ops[d_idx](routes, problem, num_remove)
        repaired_routes: List[List[int]] = repair_ops[r_idx](partial_routes, removed, problem)

        candidate = Solution1D(_routes_to_chromosome(repaired_routes), problem)
        candidate_fitness: float = evaluate(candidate)

        # ── Acceptance & scoring ─────────────────────────────────────────
        reward: float = 0.0
        if candidate_fitness < best_fitness:
            best_fitness = candidate_fitness
            best_solution = candidate.copy()
            current = candidate
            current_fitness = candidate_fitness
            reward = config.sigma_1
            last_improve_iter = iteration
        elif candidate_fitness < current_fitness:
            current = candidate
            current_fitness = candidate_fitness
            reward = config.sigma_2
        else:
            delta_f: float = candidate_fitness - current_fitness
            if temperature > 1e-12 and random.random() < math.exp(-delta_f / temperature):
                current = candidate
                current_fitness = candidate_fitness
                reward = config.sigma_3

        destroy_stats[d_idx].score += reward
        destroy_stats[d_idx].uses += 1
        repair_stats[r_idx].score += reward
        repair_stats[r_idx].uses += 1

        history.append(best_fitness)
        temperature *= config.sa_cooling_rate

        # ── Segment boundary: weights + restart check ────────────────────
        if iteration % config.segment_length == 0:
            r: float = config.reaction_factor
            for i in range(num_destroy):
                if destroy_stats[i].uses > 0:
                    avg = destroy_stats[i].score / destroy_stats[i].uses
                    destroy_weights[i] = max(
                        config.weight_min, (1 - r) * destroy_weights[i] + r * avg
                    )
                destroy_stats[i] = OperatorStats()
            for i in range(num_repair):
                if repair_stats[i].uses > 0:
                    avg = repair_stats[i].score / repair_stats[i].uses
                    repair_weights[i] = max(
                        config.weight_min, (1 - r) * repair_weights[i] + r * avg
                    )
                repair_stats[i] = OperatorStats()

            # Restart: no global-best progress for restart_segments → intensify.
            if iteration - last_improve_iter >= restart_patience:
                current = best_solution.copy()
                current_fitness = best_fitness
                temperature = max(temperature, config.reheat_factor * config.sa_start_temperature)
                last_improve_iter = iteration
                logger.debug("Iter %d | restart to best | T re-heated to %.3f",
                             iteration, temperature)

            logger.debug(
                "Iter %d | weights | destroy=%s | repair=%s",
                iteration,
                [f"{w:.2f}" for w in destroy_weights],
                [f"{w:.2f}" for w in repair_weights],
            )

        if iteration % 200 == 0 or iteration == 1:
            logger.info(
                "Iter %4d | best=%.4f | current=%.4f | T=%.4f | d_wt=%s | r_wt=%s",
                iteration, best_fitness, current_fitness, temperature,
                [f"{w:.2f}" for w in destroy_weights],
                [f"{w:.2f}" for w in repair_weights],
            )

    logger.info("ALNS finished | best_fitness=%.4f", best_fitness)
    return best_solution, best_fitness, history
