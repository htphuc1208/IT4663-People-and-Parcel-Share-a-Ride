"""
ga.py — Genetic Algorithm for the Share-a-Ride Problem (Min-Max)
================================================================

This module implements a Genetic Algorithm (GA) that operates EXCLUSIVELY
on the unified 1D chromosome representation defined in `Solution1D`.

The implementation is enhanced with ideas from standalone solvers, including:
  - A multi-stage repair function to fix gene sets and precedence.
  - A diverse set of mutation operators (swap, inversion, relocate).
  - Adaptive mutation rate to balance exploration and exploitation.

═══════════════════════════════════════════════════════════════════════════
CRITICAL: 1D ENCODING MANIPULATION RULES
═══════════════════════════════════════════════════════════════════════════
All genetic operators (crossover, mutation, repair) MUST obey these rules:

  1. DO NOT add, remove, or unnecessarily move the `0` separators.
     The chromosome must always contain exactly (K − 1) zeros. They are
     the "skeleton" that defines the K routes.

  2. Operate ONLY on non-zero genes whenever possible.
     Crossover and mutation should work on the sequence of non-zero nodes,
     then inject them back into the skeleton of zeros.

  3. Preserve the set of required non-zero node IDs.
     After any operator, the chromosome must contain the correct set of
     non-zero integers. The `repair` function is critical for this.

  4. Use penalty-driven fitness.
     Feasibility (capacity, time windows, precedence) is primarily enforced
     by the penalty functions in `fitness.py`. The GA's job is to explore
     the search space, and the fitness function guides it toward valid,
     high-quality solutions. The `repair` function provides a stronger
     nudge toward valid precedence.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

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
    population_size: int = 80
    max_generations: int = 800
    elite_count: int = 4
    tournament_size: int = 5

    # ── Adaptive mutation ────────────────────────────────────────────────
    mutation_rate_init: float = 0.20
    mutation_rate_min: float = 0.04
    mutation_rate_max: float = 0.70
    mutation_rate_increase: float = 0.05  # bump when stagnating
    mutation_rate_decay: float = 0.985    # decay when improving
    stagnation_limit: int = 25            # generations without improvement

    crossover_rate: float = 0.90
    seed: int | None = None
    time_limit_seconds: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       HELPER: EXTRACT / INJECT NON-ZEROS                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _non_zero_positions(chromosome: List[int]) -> List[int]:
    """Return the indices of all non-zero genes."""
    return [i for i, g in enumerate(chromosome) if g != 0]


def _extract_non_zeros(chromosome: List[int]) -> Tuple[List[int], List[int]]:
    """Extract non-zero values and their original positions."""
    positions: List[int] = _non_zero_positions(chromosome)
    values: List[int] = [chromosome[i] for i in positions]
    return values, positions


def _inject_non_zeros(
    chromosome_template: List[int],
    values: List[int],
    positions: List[int],
) -> List[int]:
    """Write `values` back into a chromosome at `positions`."""
    new_chrom: List[int] = list(chromosome_template)
    # Ensure the template has placeholders at the target positions
    for p in positions:
        if p < len(new_chrom):
            new_chrom[p] = -1 # Placeholder
        else:
            # This case should ideally not happen if positions are from a valid chrom
            while len(new_chrom) <= p:
                new_chrom.append(0)
            new_chrom[p] = -1

    # Inject values
    for pos, val in zip(positions, values):
        new_chrom[pos] = val

    # Clean up any remaining placeholders if lengths mismatch
    # This is a safeguard
    final_chrom = [g for g in new_chrom if g != -1]
    return final_chrom


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                             REPAIR OPERATORS                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _fix_gene_set(solution: Solution1D) -> Solution1D:
    """
    Ensures the chromosome contains exactly the required set of non-zero genes,
    preserving the positions of zeros.
    """
    problem = solution.problem
    values, positions = _extract_non_zeros(solution.chromosome)
    required_genes = set(problem.get_all_user_nodes())

    # Find missing and extra genes
    value_counts = {}
    for v in values:
        value_counts[v] = value_counts.get(v, 0) + 1

    missing = list(required_genes - set(values))
    extras = [v for v, count in value_counts.items() if count > 1 or v not in required_genes]

    # Replace extras/duplicates with missing genes
    final_values = []
    for v in values:
        is_extra = (v in value_counts and value_counts[v] > 1) or (v not in required_genes)
        if is_extra:
            if missing:
                new_gene = missing.pop()
                final_values.append(new_gene)
                if v in value_counts:
                    value_counts[v] -= 1
            # else, if no missing genes, we might have to remove it, but let's keep for now
            # This part of logic depends on how to handle length mismatches.
            # For now, we assume replacement is the primary goal.
        else:
            final_values.append(v)

    # If after replacement, the list is still not right, we might need a more robust fix
    # For now, we assume this handles the main cases.
    if len(final_values) != len(positions):
        # Fallback: rebuild from scratch, less ideal as it loses structure
        final_values = list(required_genes)
        random.shuffle(final_values)

    new_chrom = _inject_non_zeros(solution.chromosome, final_values, positions)
    return Solution1D(new_chrom, problem)


def _repair_precedence_for_route(problem: ProblemData, route: List[int]) -> List[int]:
    """For a single route, move any parcel pickup before its drop-off if ordered incorrectly."""
    pos = {node: i for i, node in enumerate(route)}
    for p_id in problem.get_parcel_ids():
        pickup_node = problem.get_node_id_from_parcel_id(p_id, is_pickup=True)
        dropoff_node = problem.get_node_id_from_parcel_id(p_id, is_pickup=False)

        if pickup_node in pos and dropoff_node in pos and pos[dropoff_node] < pos[pickup_node]:
            # Simple fix: swap them. More advanced: re-insert pickup before dropoff.
            idx_pu, idx_dr = pos[pickup_node], pos[dropoff_node]
            route[idx_pu], route[idx_dr] = route[idx_dr], route[idx_pu]
            # Update positions after swap for subsequent checks
            pos[pickup_node], pos[dropoff_node] = idx_dr, idx_pu
    return route


def repair(solution: Solution1D) -> Solution1D:
    """
    Applies a sequence of repairs to a solution to improve its feasibility.
    1. Fix the set of genes to ensure all required nodes are present exactly once.
    2. Fix parcel pickup/drop-off precedence within each route.
    """
    # 1. Fix gene set
    repaired_solution = _fix_gene_set(solution)

    # 2. Fix precedence per route
    routes = repaired_solution.get_routes()
    repaired_routes = [
        _repair_precedence_for_route(solution.problem, r) for r in routes
    ]

    # Reconstruct chromosome from repaired routes
    new_chrom = []
    for i, r in enumerate(repaired_routes):
        new_chrom.extend(r)
        if i < len(repaired_routes) - 1:
            new_chrom.append(0)

    return Solution1D(new_chrom, solution.problem)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     CROSSOVER: ORDER CROSSOVER (OX)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def order_crossover(
    parent_a: Solution1D,
    parent_b: Solution1D,
) -> Tuple[Solution1D, Solution1D]:
    """
    Order Crossover (OX) that operates ONLY on the non-zero genes.
    The zero-separator skeleton is inherited from the parents.
    """
    prob: ProblemData = parent_a.problem
    chrom_a: List[int] = parent_a.chromosome
    chrom_b: List[int] = parent_b.chromosome

    vals_a, positions_a = _extract_non_zeros(chrom_a)
    vals_b, positions_b = _extract_non_zeros(chrom_b) # Positions might differ if K is different
    n: int = len(vals_a)

    if n < 2:
        return parent_a.copy(), parent_b.copy()

    # Select two random cut points
    cx1, cx2 = sorted(random.sample(range(n), 2))

    def create_child_values(p1_vals, p2_vals):
        child_vals = [-1] * n
        # Copy segment from parent 1
        child_vals[cx1:cx2 + 1] = p1_vals[cx1:cx2 + 1]
        # Fill the rest from parent 2
        fill_genes = [g for g in p2_vals if g not in child_vals[cx1:cx2 + 1]]
        fill_idx = 0
        for i in range(n):
            if child_vals[i] == -1:
                child_vals[i] = fill_genes[fill_idx]
                fill_idx += 1
        return child_vals

    child_vals_1 = create_child_values(vals_a, vals_b)
    child_vals_2 = create_child_values(vals_b, vals_a)

    # Inject back into original chromosome structures
    child_chrom_1 = _inject_non_zeros(chrom_a, child_vals_1, positions_a)
    child_chrom_2 = _inject_non_zeros(chrom_b, child_vals_2, positions_b)

    # Repair is crucial after crossover
    child_a = repair(Solution1D(child_chrom_1, prob))
    child_b = repair(Solution1D(child_chrom_2, prob))

    return child_a, child_b


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          MUTATION OPERATORS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def swap_mutation(solution: Solution1D) -> Solution1D:
    """Swap two randomly chosen non-zero genes."""
    chrom: List[int] = list(solution.chromosome)
    positions: List[int] = _non_zero_positions(chrom)
    if len(positions) < 2:
        return solution.copy()

    i, j = random.sample(positions, 2)
    chrom[i], chrom[j] = chrom[j], chrom[i]
    return Solution1D(chrom, solution.problem)


def inversion_mutation(solution: Solution1D) -> Solution1D:
    """Reverse a sub-segment of non-zero genes within one route segment."""
    chrom: List[int] = list(solution.chromosome)
    routes: List[List[int]] = solution.get_routes()

    non_empty_indices: List[int] = [k for k, r in enumerate(routes) if len(r) >= 2]
    if not non_empty_indices:
        return solution.copy()

    route_idx: int = random.choice(non_empty_indices)
    route: List[int] = routes[route_idx]

    i, j = sorted(random.sample(range(len(route)), 2))
    route[i:j + 1] = reversed(route[i:j + 1])

    # Reconstruct chromosome
    new_chrom = []
    for k, r in enumerate(routes):
        new_chrom.extend(r)
        if k < len(routes) - 1:
            new_chrom.append(0)

    return Solution1D(new_chrom, solution.problem)


def relocate_mutation(solution: Solution1D) -> Solution1D:
    """Move a random non-zero gene to another random non-zero position."""
    vals, pos = _extract_non_zeros(solution.chromosome)
    if len(vals) < 2:
        return solution.copy()

    # Pick a gene to move
    gene_idx_to_move = random.randrange(len(vals))
    gene = vals.pop(gene_idx_to_move)

    # Pick a new position to insert it
    new_pos = random.randrange(len(vals) + 1)
    vals.insert(new_pos, gene)

    new_chrom = _inject_non_zeros(solution.chromosome, vals, pos)
    return Solution1D(new_chrom, solution.problem)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       SELECTION: TOURNAMENT                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def tournament_selection(
    population: List[Solution1D],
    fitnesses: List[float],
    tournament_size: int,
) -> Solution1D:
    """Select the best individual from a random tournament of given size."""
    size = min(tournament_size, len(population))
    indices: List[int] = random.sample(range(len(population)), size)
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
    population: List[Solution1D] = [greedy_init(problem, seed=config.seed)]
    for _ in range(config.population_size - 1):
        population.append(random_init(problem, seed=config.seed))

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

            # Crucial: Repair after mutation as well
            child_a = repair(child_a)
            child_b = repair(child_b)

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
            logger.debug("Gen %4d: New best fitness found: %.4f", gen, best_fitness)
        else:
            stagnation_counter += 1

        history.append(best_fitness)

        # ── Adaptive mutation rate adjustment ────────────────────────────
        if stagnation_counter >= config.stagnation_limit:
            mutation_rate = min(
                mutation_rate + config.mutation_rate_increase,
                config.mutation_rate_max,
            )
            stagnation_counter = 0  # reset after bump
            logger.debug(
                "Gen %d: Stagnation -> mutation_rate INCREASED to %.3f", gen, mutation_rate
            )
        else:
            mutation_rate = max(
                mutation_rate * config.mutation_rate_decay,
                config.mutation_rate_min,
            )

        if gen % 50 == 0 or gen == 1:
            logger.info(
                "Gen %4d | Best: %.4f | MR: %.3f | Stagnation: %d/%d",
                gen, best_fitness, mutation_rate, stagnation_counter, config.stagnation_limit
            )

    # Final polish on the best solution found
    final_solution = repair(best_solution)
    final_fitness = evaluate(final_solution)

    logger.info("GA finished | Final best fitness: %.4f", final_fitness)
    if final_fitness < best_fitness:
        return final_solution, final_fitness, history

    return best_solution, best_fitness, history
