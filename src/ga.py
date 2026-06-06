"""
ga.py — Memetic Genetic Algorithm for the Share-a-Ride Problem (Min-Max)
========================================================================

This is a rewrite of the original permutation-GA. Empirical profiling of the
old Order-Crossover GA showed it was destructive: 94-96% of children were worse
than their best parent and 83-90% carried penalties (split parcels) even after
repair. The crossover treated the multi-route chromosome as one flat TSP, so it
never inherited good *vehicle clusters* — the actual building blocks of a
min-max partitioning problem.

This version fixes that with four changes:

  1. REQUEST-LEVEL, ATOMIC-PARCEL REPRESENTATION
     Operators work on `List[List[Request]]` — one ordered request list per
     vehicle. A parcel is an atomic unit (pickup immediately followed by its
     drop-off). Consequences:
       • Parcels can never be split across routes  → no precedence penalty.
       • At most one parcel is on board at a time   → capacity is satisfied as
         long as each parcel sits on a vehicle with Q[k] ≥ demand (guaranteed
         by every operator). So penalties are structurally zero and fitness
         reduces to the pure min-max distance the problem actually asks for.

  2. ROUTE-AWARE CROSSOVER (improvement #1)
     `_route_crossover` inherits whole routes from one parent, removes the
     requests of one donor route taken from the other parent, then re-inserts
     them by cheapest, min-max-aware insertion. Good vehicle clusters survive.

  3. BOTTLENECK-DIRECTED MUTATION (improvement #2)
     `_mut_bottleneck` pulls a request off the longest route and re-inserts it
     into the shortest feasible route — a move aimed straight at the min-max
     objective. It sits alongside relocate / swap / inversion.

  4. MEMETIC LOCAL SEARCH (improvement #3)
     `_local_search` polishes a solution with inter-route relocate (off the
     bottleneck) plus intra-route 2-opt on the bottleneck route.

  + CONVERGENCE & POPULATION MANAGEMENT (improvement #4)
     Duplicate-free seeding biased toward greedy/min-max/savings variants,
     smaller tournaments, and early stop on convergence instead of a fixed
     generation count.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from .encoding_and_read import ProblemData, Request, RequestType, Solution1D
from .fitness import evaluate
from .init import greedy_init, minmax_greedy_init, savings_init, random_init

logger = logging.getLogger(__name__)

# A working solution is K ordered request lists, one per vehicle.
Routes = List[List[Request]]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         GA PARAMETERS                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class GAConfig:
    """Configuration for the memetic Genetic Algorithm."""
    population_size: int = -1   # -1 = auto-scale ≈ max(40, min(200, N+M+20))
    max_generations: int = 2000
    max_no_improve: int = 250   # early stop after this many stagnant generations
    elite_count: int = 4
    tournament_size: int = 3    # lowered from 5 → less premature convergence

    # ── Adaptive mutation ────────────────────────────────────────────────
    mutation_rate_init: float = 0.20
    mutation_rate_min: float = 0.05
    mutation_rate_max: float = 0.70
    mutation_rate_increase: float = 0.05
    mutation_rate_decay: float = 0.985
    stagnation_limit: int = 25            # generations before a mutation bump

    crossover_rate: float = 0.90

    # ── Memetic local search ─────────────────────────────────────────────
    ls_rate: float = 0.20       # probability of polishing a child
    ls_passes: int = 8          # local-search passes per call

    seed: Optional[int] = None
    time_limit_seconds: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 ROUTES ⇄ SOLUTION1D CONVERSION                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _solution_to_routes(solution: Solution1D) -> Routes:
    """
    Parse a chromosome into K request lists. Parcel drop-off nodes are skipped
    (re-derived on assembly), collapsing every parcel to an atomic unit.
    """
    prob = solution.problem
    by_pickup = {r.pickup_node: r for r in prob.passengers}
    by_pickup.update({r.pickup_node: r for r in prob.parcels})

    routes: Routes = [[]]
    for gene in solution.chromosome:
        if gene == 0:
            routes.append([])
        elif gene in by_pickup:        # pickup node → its request
            routes[-1].append(by_pickup[gene])
        # else: a drop-off node — implied, skip
    return routes


def _routes_to_solution(routes: Routes, problem: ProblemData) -> Solution1D:
    """Assemble K request lists back into a structurally-valid chromosome."""
    chrom: List[int] = []
    last = len(routes) - 1
    for v, route in enumerate(routes):
        for req in route:
            chrom.append(req.pickup_node)
            if req.request_type == RequestType.PARCEL:
                chrom.append(req.dropoff_node)
        if v < last:
            chrom.append(0)
    return Solution1D(chrom, problem)


def _clone(routes: Routes) -> Routes:
    """Deep-enough copy: Request is immutable, so copy the inner lists only."""
    return [list(r) for r in routes]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 DISTANCE / FITNESS HELPERS                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _route_distance(route: List[Request], prob: ProblemData) -> float:
    """depot → (pickup → dropoff)* → depot for one request list."""
    total = 0.0
    prev = 0
    for req in route:
        total += prob.dist(prev, req.pickup_node)
        total += prob.dist(req.pickup_node, req.dropoff_node)
        prev = req.dropoff_node
    total += prob.dist(prev, 0)
    return total


def _route_dists(routes: Routes, prob: ProblemData) -> List[float]:
    return [_route_distance(r, prob) for r in routes]


def _minmax(routes: Routes, prob: ProblemData) -> float:
    """Internal objective: the longest route distance (penalties are 0 here)."""
    return max(_route_distance(r, prob) for r in routes)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 CAPACITY-FEASIBLE CHEAPEST INSERTION                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _feasible_vehicles(req: Request, prob: ProblemData) -> List[int]:
    """Vehicles that can carry this request (Q ≥ demand for parcels)."""
    if req.request_type == RequestType.PASSENGER:
        return list(range(prob.K))
    elig = [v for v in range(prob.K) if prob.vehicle_capacities[v] >= req.demand]
    return elig if elig else list(range(prob.K))


def _best_position(route: List[Request], req: Request, prob: ProblemData) -> Tuple[int, float]:
    """
    Cheapest insertion position of req into one route; returns (pos, delta).

    Uses the O(1)-per-position insertion delta instead of rebuilding the route:
        delta = d(left, pu) + d(pu, do) + d(do, right) − d(left, right)
    where (left, right) are the boundary nodes around the insertion point and
    (pu, do) are the request's pickup/drop-off. O(n) overall — this is the hot
    path for crossover and local search.
    """
    pu, do = req.pickup_node, req.dropoff_node
    seg = prob.dist(pu, do)            # internal pickup→dropoff, always traversed
    left = 0                            # depot before the first request
    best_pos, best_delta = 0, float("inf")
    n = len(route)
    for pos in range(n + 1):
        right = route[pos].pickup_node if pos < n else 0   # depot after the last
        delta = prob.dist(left, pu) + seg + prob.dist(do, right) - prob.dist(left, right)
        if delta < best_delta:
            best_delta, best_pos = delta, pos
        if pos < n:
            left = route[pos].dropoff_node
    return best_pos, best_delta


def _insert_best(
    routes: Routes,
    req: Request,
    prob: ProblemData,
    avoid: Optional[int] = None,
) -> None:
    """
    Insert req at the position that minimises the *resulting route length* over
    all feasible vehicles. Minimising the resulting length (not just the delta)
    naturally steers requests toward shorter routes — the min-max objective.
    """
    choices = [v for v in _feasible_vehicles(req, prob) if v != avoid]
    if not choices:
        choices = _feasible_vehicles(req, prob)

    best_v, best_pos, best_len = choices[0], 0, float("inf")
    for v in choices:
        base = _route_distance(routes[v], prob)
        pos, delta = _best_position(routes[v], req, prob)
        result_len = base + delta
        if result_len < best_len:
            best_v, best_pos, best_len = v, pos, result_len
    routes[best_v].insert(best_pos, req)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 ROUTE-AWARE CROSSOVER (improvement #1)                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _route_crossover(parent_a: Routes, parent_b: Routes, prob: ProblemData) -> Routes:
    """
    Inherit parent A's routes, then re-cluster the requests of one donor route
    taken from parent B via cheapest min-max-aware insertion. Whole good
    vehicle-clusters survive intact — unlike the old flat Order-Crossover.
    """
    child = _clone(parent_a)

    donor_candidates = [r for r in parent_b if r]
    if not donor_candidates:
        return child
    donor = random.choice(donor_candidates)
    donor_ids = {req.pickup_node for req in donor}

    # Remove donor requests from the child, remembering them.
    removed: List[Request] = []
    for v in range(len(child)):
        kept = []
        for req in child[v]:
            if req.pickup_node in donor_ids:
                removed.append(req)
            else:
                kept.append(req)
        child[v] = kept

    # Re-insert (shuffled so insertion order varies between calls).
    random.shuffle(removed)
    for req in removed:
        _insert_best(child, req, prob)
    return child


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          MUTATION OPERATORS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _mut_relocate(routes: Routes, prob: ProblemData) -> Routes:
    """Move one random request to its cheapest feasible position elsewhere."""
    out = _clone(routes)
    nonempty = [v for v in range(len(out)) if out[v]]
    if not nonempty:
        return out
    v = random.choice(nonempty)
    req = out[v].pop(random.randrange(len(out[v])))
    _insert_best(out, req, prob)
    return out


def _mut_swap(routes: Routes, prob: ProblemData) -> Routes:
    """Swap two requests between two routes, respecting parcel capacity."""
    out = _clone(routes)
    nonempty = [v for v in range(len(out)) if out[v]]
    if len(nonempty) < 2:
        return out
    va, vb = random.sample(nonempty, 2)
    ia, ib = random.randrange(len(out[va])), random.randrange(len(out[vb]))
    ra, rb = out[va][ia], out[vb][ib]

    # Capacity guard keeps the atomic-parcel feasibility invariant.
    if ra.request_type == RequestType.PARCEL and prob.vehicle_capacities[vb] < ra.demand:
        return out
    if rb.request_type == RequestType.PARCEL and prob.vehicle_capacities[va] < rb.demand:
        return out

    out[va][ia], out[vb][ib] = rb, ra
    return out


def _mut_inversion(routes: Routes, prob: ProblemData) -> Routes:
    """Reverse a sub-segment within one route (always feasible)."""
    out = _clone(routes)
    candidates = [v for v in range(len(out)) if len(out[v]) >= 2]
    if not candidates:
        return out
    v = random.choice(candidates)
    i, j = sorted(random.sample(range(len(out[v])), 2))
    out[v][i:j + 1] = reversed(out[v][i:j + 1])
    return out


def _mut_bottleneck(routes: Routes, prob: ProblemData) -> Routes:
    """
    Min-max-directed: take a request off the longest route and re-insert it into
    the best *other* feasible (shortest) route. Directly attacks the objective.
    """
    out = _clone(routes)
    dists = _route_dists(out, prob)
    bidx = max(range(len(out)), key=lambda i: dists[i])
    if not out[bidx]:
        return _mut_relocate(routes, prob)
    req = out[bidx].pop(random.randrange(len(out[bidx])))
    _insert_best(out, req, prob, avoid=bidx)
    return out


_MUTATORS: List[Callable[[Routes, ProblemData], Routes]] = [
    _mut_relocate, _mut_swap, _mut_inversion, _mut_bottleneck,
]
_MUT_WEIGHTS = [0.25, 0.20, 0.20, 0.35]   # bias toward the bottleneck operator


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 MEMETIC LOCAL SEARCH (improvement #3)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _two_opt_once(route: List[Request], prob: ProblemData) -> List[Request]:
    """Return the best single 2-opt segment reversal of one route."""
    n = len(route)
    if n < 3:
        return route
    best_route, best_d = route, _route_distance(route, prob)
    for i in range(n - 1):
        for j in range(i + 1, n):
            cand = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
            d = _route_distance(cand, prob)
            if d < best_d - 1e-9:
                best_d, best_route = d, cand
    return best_route


def _local_search(routes: Routes, prob: ProblemData, max_pass: int) -> Routes:
    """
    Polish a solution: relocate requests off the bottleneck route when that
    lowers the global max, then 2-opt the bottleneck route. Repeat until no
    improvement or max_pass reached.
    """
    out = _clone(routes)
    for _ in range(max_pass):
        dists = _route_dists(out, prob)
        bidx = max(range(len(out)), key=lambda i: dists[i])
        cur_max = dists[bidx]
        improved = False

        # (a) inter-route relocate off the bottleneck
        broute = out[bidx]
        for i in range(len(broute)):
            req = broute[i]
            remainder = broute[:i] + broute[i + 1:]
            rem_len = _route_distance(remainder, prob)
            if rem_len >= cur_max:
                continue  # removing this request alone can't lower the max
            best = None
            for v in _feasible_vehicles(req, prob):
                if v == bidx:
                    continue
                base = _route_distance(out[v], prob)
                pos, delta = _best_position(out[v], req, prob)
                new_len = base + delta
                if new_len < cur_max and (best is None or new_len < best[2]):
                    best = (v, pos, new_len)
            if best is not None and max(rem_len, best[2]) < cur_max - 1e-9:
                v, pos, _ = best
                out[bidx] = remainder
                out[v].insert(pos, req)
                improved = True
                break

        if improved:
            continue

        # (b) intra-route 2-opt on the bottleneck
        polished = _two_opt_once(out[bidx], prob)
        if _route_distance(polished, prob) < cur_max - 1e-9:
            out[bidx] = polished
            improved = True

        if not improved:
            break
    return out


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       SELECTION: TOURNAMENT                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _tournament(population: List[Routes], fitnesses: List[float], k: int) -> Routes:
    size = min(k, len(population))
    idx = random.sample(range(len(population)), size)
    best = min(idx, key=lambda i: fitnesses[i])
    return _clone(population[best])


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 POPULATION SEEDING (improvement #4)                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _seed_population(problem: ProblemData, pop_size: int, base_seed: int) -> List[Routes]:
    """
    Build a duplicate-free population biased toward structured constructions
    (greedy / min-max / savings variants), padded with random solutions.
    """
    population: List[Routes] = []
    seen: set = set()

    def add(sol: Solution1D) -> None:
        routes = _solution_to_routes(sol)
        key = tuple(_routes_to_solution(routes, problem).chromosome)
        if key not in seen:
            seen.add(key)
            population.append(routes)

    # Deterministic structured seeds first.
    add(greedy_init(problem))
    add(minmax_greedy_init(problem))
    add(savings_init(problem))

    # ~70% structured (perturbed) variants.
    structured_target = max(6, int(pop_size * 0.70))
    s = base_seed
    while len(population) < structured_target and s < base_seed + pop_size * 4:
        add(greedy_init(problem, seed=s))
        add(minmax_greedy_init(problem, seed=s))
        add(savings_init(problem, seed=s))
        s += 1

    # Remainder: random diversity.
    tries = 0
    while len(population) < pop_size and tries < pop_size * 5:
        add(random_init(problem, seed=base_seed + 1000 + tries))
        tries += 1

    # Hard pad if duplicates blocked us from reaching pop_size.
    while len(population) < pop_size:
        population.append(_solution_to_routes(random_init(problem)))

    return population[:pop_size]


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           GA MAIN LOOP                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_ga(
    problem: ProblemData,
    config: GAConfig | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Execute the memetic GA.

    Returns
    -------
    (best_solution, best_fitness, history)
    """
    if config is None:
        config = GAConfig()
    if config.seed is not None:
        random.seed(config.seed)

    start = time.time()
    R = problem.N + problem.M
    pop_size = (
        config.population_size if config.population_size > 0
        else max(40, min(200, R + 20))
    )
    base_seed = config.seed if config.seed is not None else 0

    # ── Initial population ───────────────────────────────────────────────
    population: List[Routes] = _seed_population(problem, pop_size, base_seed)
    fitnesses: List[float] = [_minmax(r, problem) for r in population]

    best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
    best_routes = _clone(population[best_idx])
    best_fit = fitnesses[best_idx]
    # Polish the incumbent once up front.
    polished = _local_search(best_routes, problem, config.ls_passes)
    if _minmax(polished, problem) < best_fit:
        best_routes, best_fit = polished, _minmax(polished, problem)
    history: List[float] = [best_fit]

    mutation_rate = config.mutation_rate_init
    stagnation = 0           # for adaptive mutation bumps
    no_improve = 0           # for early-stop

    logger.info(
        "Memetic GA started | pop=%d | max_gen=%d | init_best=%.1f",
        pop_size, config.max_generations, best_fit,
    )

    for gen in range(1, config.max_generations + 1):
        if time.time() - start >= config.time_limit_seconds:
            logger.info("GA time limit reached at generation %d", gen)
            break

        # Elitism.
        order = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        new_pop: List[Routes] = [_clone(population[i]) for i in order[:config.elite_count]]

        # Offspring.
        while len(new_pop) < pop_size:
            pa = _tournament(population, fitnesses, config.tournament_size)
            pb = _tournament(population, fitnesses, config.tournament_size)

            if random.random() < config.crossover_rate:
                child = _route_crossover(pa, pb, problem)
            else:
                child = pa

            if random.random() < mutation_rate:
                mutator = random.choices(_MUTATORS, weights=_MUT_WEIGHTS, k=1)[0]
                child = mutator(child, problem)

            if random.random() < config.ls_rate:
                child = _local_search(child, problem, config.ls_passes)

            new_pop.append(child)

        population = new_pop
        fitnesses = [_minmax(r, problem) for r in population]

        gen_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        gen_fit = fitnesses[gen_idx]

        if gen_fit < best_fit - 1e-9:
            # Polish the new champion before committing.
            cand = _local_search(population[gen_idx], problem, config.ls_passes)
            cand_fit = _minmax(cand, problem)
            if cand_fit <= gen_fit:
                best_routes, best_fit = _clone(cand), cand_fit
            else:
                best_routes, best_fit = _clone(population[gen_idx]), gen_fit
            stagnation = 0
            no_improve = 0
            logger.debug("Gen %4d: new best = %.1f", gen, best_fit)
        else:
            stagnation += 1
            no_improve += 1

        history.append(best_fit)

        # Adaptive mutation.
        if stagnation >= config.stagnation_limit:
            mutation_rate = min(mutation_rate + config.mutation_rate_increase,
                                config.mutation_rate_max)
            stagnation = 0
        else:
            mutation_rate = max(mutation_rate * config.mutation_rate_decay,
                                config.mutation_rate_min)

        # Early stop on convergence.
        if no_improve >= config.max_no_improve:
            logger.info("GA converged: no improvement for %d generations (gen %d)",
                        config.max_no_improve, gen)
            break

        if gen % 50 == 0 or gen == 1:
            logger.info("Gen %4d | best=%.1f | mr=%.3f | no_improve=%d",
                        gen, best_fit, mutation_rate, no_improve)

    # Final polish + official evaluation.
    final_routes = _local_search(best_routes, problem, config.ls_passes * 2)
    if _minmax(final_routes, problem) < best_fit:
        best_routes = final_routes

    best_solution = _routes_to_solution(best_routes, problem)
    final_fitness = evaluate(best_solution)
    logger.info("Memetic GA finished | best fitness = %.1f", final_fitness)
    return best_solution, final_fitness, history
