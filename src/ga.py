"""
ga.py — Genetic Algorithm for the Share-a-Ride Problem (Min-Max)
================================================================

This module implements a Genetic Algorithm (GA) that operates EXCLUSIVELY
on the unified 1D chromosome representation defined in `Solution1D`.

═══════════════════════════════════════════════════════════════════════════
CRITICAL: 1D ENCODING MANIPULATION RULES
═══════════════════════════════════════════════════════════════════════════
All genetic operators (crossover, mutation, repair) MUST obey these rules:

  1. NEVER duplicate or remove `0` separators.
     • The chromosome always contains exactly (K − 1) zeros.
     • Operators must treat zeros as immovable "walls" — they partition the
       chromosome into K route segments.  Swapping, inserting, or deleting
       a zero would change the number of vehicles, which is invalid.

  2. Operate ONLY on non-zero genes.
     • Crossover and mutation should identify the positions of non-zero
       entries, manipulate those entries (swap, relocate, reverse), and
       then write them back into the chromosome at the non-zero positions,
       leaving every zero in its original index.

  3. Preserve the set of non-zero node IDs.
     • After any operator, the chromosome must contain EXACTLY the same
       set of non-zero integers as before — no duplicates, no omissions.
     • A repair step should verify and fix any violations.

  4. Parcel precedence is enforced by the penalty function (fitness.py),
     NOT by the genetic operators.  This keeps operators simple and fast;
     the penalty-driven fitness naturally steers the population toward
     feasibility.

  5. ADAPTIVE MUTATION RATE:
     • The mutation probability starts at an initial value (e.g. 0.15).
     • If the population's best fitness stagnates for `stagnation_limit`
       generations, the mutation rate is INCREASED (up to a cap) to inject
       diversity.
     • If improvement resumes, the mutation rate decays back toward the
       baseline to allow fine-tuning.
     • This feedback loop prevents premature convergence without requiring
       manual tuning.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .encoding_and_read import ProblemData, Solution1D
from .fitness import evaluate
from .init import greedy_init, random_init

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         GA PARAMETERS                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class GAConfig:
    """Configuration for the Genetic Algorithm."""
    population_size: int = 50
    max_generations: int = 500
    elite_count: int = 2
    tournament_size: int = 5

    # ── Adaptive mutation ────────────────────────────────────────────────
    mutation_rate_init: float = 0.15
    mutation_rate_min: float = 0.05
    mutation_rate_max: float = 0.60
    mutation_rate_increase: float = 0.05  # bump when stagnating
    mutation_rate_decay: float = 0.98     # decay when improving
    stagnation_limit: int = 15            # generations without improvement

    crossover_rate: float = 0.85
    seed: int | None = None
    time_limit_seconds: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       HELPER: EXTRACT / INJECT NON-ZEROS                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _non_zero_positions(chromosome: List[int]) -> List[int]:
    """Return the indices of all non-zero genes."""
    return [i for i, g in enumerate(chromosome) if g != 0]


def _extract_non_zeros(chromosome: List[int]) -> Tuple[List[int], List[int]]:
    """
    Split the chromosome into (non_zero_values, zero_skeleton).
    `zero_skeleton` is a copy of the chromosome with non-zeros replaced by -1,
    preserving the positions of all zeros.
    """
    positions: List[int] = _non_zero_positions(chromosome)
    values: List[int] = [chromosome[i] for i in positions]
    return values, positions


def _inject_non_zeros(
    chromosome: List[int],
    values: List[int],
    positions: List[int],
) -> List[int]:
    """Write `values` back into `chromosome` at `positions`."""
    new_chrom: List[int] = list(chromosome)
    for pos, val in zip(positions, values):
        new_chrom[pos] = val
    return new_chrom


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     CROSSOVER: ORDER CROSSOVER (OX)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def order_crossover(
    parent_a: Solution1D,
    parent_b: Solution1D,
) -> Tuple[Solution1D, Solution1D]:
    """
    Order Crossover (OX) that operates ONLY on the non-zero genes.

    ── HOW IT WORKS ON THE 1D ENCODING ──
    1. Extract the ordered list of non-zero values from each parent.
       (Zeros stay pinned — they are NOT part of the crossover.)
    2. Apply classical OX on these two value sequences.
    3. Inject the resulting child sequences back into the chromosome
       template (which retains zeros at their original positions).

    This guarantees:
      • Exactly (K−1) zeros remain untouched.
      • Every non-zero node ID appears exactly once in each child.
    """
    prob: ProblemData = parent_a.problem
    chrom_a: List[int] = parent_a.chromosome
    chrom_b: List[int] = parent_b.chromosome

    vals_a, positions = _extract_non_zeros(chrom_a)
    vals_b, _         = _extract_non_zeros(chrom_b)
    n: int = len(vals_a)

    if n < 2:
        return parent_a.copy(), parent_b.copy()

    # Select two random cut points
    cx1, cx2 = sorted(random.sample(range(n), 2))

    # ── Build child 1 ────────────────────────────────────────────────────
    child_vals_1: List[int] = [-1] * n
    child_vals_1[cx1:cx2 + 1] = vals_a[cx1:cx2 + 1]
    fill_order: List[int] = [v for v in vals_b if v not in child_vals_1[cx1:cx2 + 1]]
    fill_idx: int = 0
    for i in range(n):
        if child_vals_1[i] == -1:
            child_vals_1[i] = fill_order[fill_idx]
            fill_idx += 1

    # ── Build child 2 (symmetric) ────────────────────────────────────────
    child_vals_2: List[int] = [-1] * n
    child_vals_2[cx1:cx2 + 1] = vals_b[cx1:cx2 + 1]
    fill_order = [v for v in vals_a if v not in child_vals_2[cx1:cx2 + 1]]
    fill_idx = 0
    for i in range(n):
        if child_vals_2[i] == -1:
            child_vals_2[i] = fill_order[fill_idx]
            fill_idx += 1

    new_chrom_1: List[int] = _inject_non_zeros(chrom_a, child_vals_1, positions)
    new_chrom_2: List[int] = _inject_non_zeros(chrom_b, child_vals_2, positions)

    return (
        Solution1D(new_chrom_1, prob),
        Solution1D(new_chrom_2, prob),
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          MUTATION OPERATORS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def swap_mutation(solution: Solution1D) -> Solution1D:
    """
    Swap two randomly chosen non-zero genes.

    ── 1D ENCODING SAFETY ──
    • Only non-zero positions are candidates for swapping.
    • Zeros (separators) are never moved.
    • The set of non-zero IDs is preserved.
    """
    chrom: List[int] = list(solution.chromosome)
    positions: List[int] = _non_zero_positions(chrom)
    if len(positions) < 2:
        return solution.copy()

    i, j = random.sample(positions, 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    return Solution1D(chrom, solution.problem)


def inversion_mutation(solution: Solution1D) -> Solution1D:
    """
    Reverse a sub-segment of non-zero genes within one route segment.

    ── 1D ENCODING SAFETY ──
    • Picks a random route segment (between two consecutive zeros / edges).
    • Reverses the non-zero portion of that segment in-place.
    • Zeros are untouched; no node IDs are added or removed.
    """
    chrom: List[int] = list(solution.chromosome)
    routes: List[List[int]] = solution.get_routes()

    # Pick a non-empty route to reverse a sub-segment
    non_empty_indices: List[int] = [
        k for k, r in enumerate(routes) if len(r) >= 2
    ]
    if not non_empty_indices:
        return solution.copy()

    route_idx: int = random.choice(non_empty_indices)

    # Locate the start position of this route in the chromosome
    pos: int = 0
    for k in range(route_idx):
        pos += len(routes[k]) + 1  # +1 for the zero separator

    route: List[int] = routes[route_idx]
    if len(route) < 2:
        return solution.copy()

    i, j = sorted(random.sample(range(len(route)), 2))
    route[i:j + 1] = reversed(route[i:j + 1])

    # Write back
    for offset, node in enumerate(route):
        chrom[pos + offset] = node

    return Solution1D(chrom, solution.problem)


def relocate_mutation(solution: Solution1D) -> Solution1D:
    """
    Remove a random non-zero gene and reinsert it at a random non-zero position.

    ── 1D ENCODING SAFETY ──
    • Only non-zero genes are candidates.
    • The gene is removed from its current position and inserted elsewhere.
    • The zero separators stay pinned; this effectively moves a node from
      one vehicle's route segment to another (or within the same route).
    • The set of non-zero node IDs is preserved (no duplicates, no loss).
    """
    chrom: List[int] = list(solution.chromosome)
    positions: List[int] = _non_zero_positions(chrom)
    if len(positions) < 2:
        return solution.copy()

    # Pick a random non-zero to remove
    src: int = random.choice(positions)
    gene: int = chrom.pop(src)

    # Recompute non-zero positions after removal, then pick insertion point
    # We insert BEFORE a non-zero position or at the end of the chromosome
    # but we must avoid inserting at a zero's position
    possible_inserts: List[int] = [
        i for i, g in enumerate(chrom) if g != 0
    ]
    # Also allow inserting at the very end if the last element is not 0
    if chrom and chrom[-1] != 0:
        possible_inserts.append(len(chrom))
    elif not chrom:
        possible_inserts.append(0)

    if not possible_inserts:
        # Fallback: insert before first separator or at start
        chrom.insert(0, gene)
    else:
        dst: int = random.choice(possible_inserts)
        chrom.insert(dst, gene)

    return Solution1D(chrom, solution.problem)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       SELECTION: TOURNAMENT                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def tournament_selection(
    population: List[Solution1D],
    fitnesses: List[float],
    tournament_size: int,
) -> Solution1D:
    """Select the best individual from a random tournament of given size."""
    indices: List[int] = random.sample(range(len(population)), tournament_size)
    best_idx: int = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           GA MAIN LOOP                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_ga(
    problem: ProblemData,
    config: GAConfig | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Execute the full Genetic Algorithm.

    Parameters
    ----------
    problem : ProblemData
        Parsed SARP instance.
    config : GAConfig | None
        Algorithm hyper-parameters (uses defaults if None).

    Returns
    -------
    (best_solution, best_fitness, history)
        best_solution — the best Solution1D found.
        best_fitness  — its Min-Max objective value.
        history       — best fitness per generation for plotting.
    """
    if config is None:
        config = GAConfig()
    if config.seed is not None:
        random.seed(config.seed)

    start_time: float = time.time()

    # ── Initialise population ────────────────────────────────────────────
    population: List[Solution1D] = [greedy_init(problem)]
    for _ in range(config.population_size - 1):
        population.append(random_init(problem))

    fitnesses: List[float] = [evaluate(sol) for sol in population]

    best_idx: int = int(min(range(len(fitnesses)), key=lambda i: fitnesses[i]))
    best_solution: Solution1D = population[best_idx].copy()
    best_fitness: float = fitnesses[best_idx]
    history: List[float] = [best_fitness]

    # ── Adaptive mutation state ──────────────────────────────────────────
    mutation_rate: float = config.mutation_rate_init
    stagnation_counter: int = 0

    mutators: List[Callable[[Solution1D], Solution1D]] = [
        swap_mutation,
        inversion_mutation,
        relocate_mutation,
    ]

    logger.info(
        "GA started | pop=%d | max_gen=%d | init_mr=%.3f",
        config.population_size, config.max_generations, mutation_rate,
    )

    # ── Evolution loop ───────────────────────────────────────────────────
    for gen in range(1, config.max_generations + 1):
        elapsed: float = time.time() - start_time
        if elapsed >= config.time_limit_seconds:
            logger.info("GA time limit reached at generation %d", gen)
            break

        new_population: List[Solution1D] = []

        # Elitism: keep the top `elite_count` individuals
        elite_indices: List[int] = sorted(
            range(len(fitnesses)), key=lambda i: fitnesses[i]
        )[:config.elite_count]
        for ei in elite_indices:
            new_population.append(population[ei].copy())

        # Fill the rest via crossover + mutation
        while len(new_population) < config.population_size:
            # ── Selection ────────────────────────────────────────────
            parent_a: Solution1D = tournament_selection(
                population, fitnesses, config.tournament_size,
            )
            parent_b: Solution1D = tournament_selection(
                population, fitnesses, config.tournament_size,
            )

            # ── Crossover ────────────────────────────────────────────
            if random.random() < config.crossover_rate:
                child_a, child_b = order_crossover(parent_a, parent_b)
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()

            # ── Mutation ─────────────────────────────────────────────
            if random.random() < mutation_rate:
                mutator = random.choice(mutators)
                child_a = mutator(child_a)
            if random.random() < mutation_rate:
                mutator = random.choice(mutators)
                child_b = mutator(child_b)

            new_population.append(child_a)
            if len(new_population) < config.population_size:
                new_population.append(child_b)

        # ── Evaluate new population ──────────────────────────────────────
        population = new_population
        fitnesses = [evaluate(sol) for sol in population]

        gen_best_idx: int = int(
            min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        )
        gen_best_fitness: float = fitnesses[gen_best_idx]

        # ── Update global best ───────────────────────────────────────────
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_solution = population[gen_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        history.append(best_fitness)

        # ── Adaptive mutation rate adjustment ────────────────────────────
        # If the population stagnates → INCREASE mutation to inject diversity.
        # If improving → DECAY mutation to allow exploitation / fine-tuning.
        if stagnation_counter >= config.stagnation_limit:
            mutation_rate = min(
                mutation_rate + config.mutation_rate_increase,
                config.mutation_rate_max,
            )
            stagnation_counter = 0  # reset after bump
            logger.debug(
                "Gen %d: stagnation → mutation_rate ↑ %.3f", gen, mutation_rate,
            )
        else:
            mutation_rate = max(
                mutation_rate * config.mutation_rate_decay,
                config.mutation_rate_min,
            )

        if gen % 50 == 0 or gen == 1:
            logger.info(
                "Gen %4d | best=%.4f | mr=%.3f | stag=%d",
                gen, best_fitness, mutation_rate, stagnation_counter,
            )

    logger.info("GA finished | best_fitness=%.4f", best_fitness)
    return best_solution, best_fitness, history
