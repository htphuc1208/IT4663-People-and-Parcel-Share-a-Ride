"""
alns.py — Adaptive Large Neighbourhood Search for the SARP (Min-Max)
=====================================================================

This module implements ALNS, which iteratively destroys and repairs the
current solution using a portfolio of operators.  Operator selection is
governed by an **adaptive roulette-wheel** mechanism that rewards operators
producing improvements.

═══════════════════════════════════════════════════════════════════════════
CRITICAL: 1D ENCODING MANIPULATION RULES
═══════════════════════════════════════════════════════════════════════════
All destroy and repair operators MUST obey these rules:

  1. NEVER duplicate or remove `0` separators.
     • The chromosome always contains exactly (K − 1) zeros.
     • A destroy operator REMOVES some non-zero genes but NEVER touches the
       zeros.  After destruction the chromosome is shorter, but every zero
       separator is still present at the correct logical boundary.
     • A repair operator REINSERTS the removed non-zero genes back into
       valid non-zero positions without altering zeros.

  2. Destroy: extract a subset of non-zero genes → store them in a "removal pool".
     • The chromosome temporarily shrinks by the number of removed genes.
     • Zeros remain pinned: if consecutive zeros result from removal, that
       simply means the corresponding vehicle route is now empty.

  3. Repair: reinsert every gene from the removal pool into the chromosome.
     • Insertion MUST happen at non-zero-adjacent positions (i.e., between
       two non-zeros or between a zero and a non-zero).  Inserting at a
       zero's index would shift a separator, which is FORBIDDEN.
     • After repair the chromosome length is restored to N + 2M + (K − 1).

  4. Preserve the set of non-zero node IDs.
     • After a full destroy + repair cycle, the chromosome MUST contain
       exactly the same set of non-zero integers as the original — no
       duplicates, no omissions.

  5. ADAPTIVE ROULETTE-WHEEL SELECTION:
     • Each destroy operator and each repair operator has a weight (score).
     • At each iteration, operators are selected with probability proportional
       to their weight (roulette wheel).
     • After applying a (destroy, repair) pair, the resulting solution quality
       determines a reward:
         — σ₁ if a new global best is found.
         — σ₂ if the solution improves on the current solution.
         — σ₃ if the solution is accepted (even if worse — via Simulated
           Annealing acceptance criterion).
         — 0  if the solution is rejected.
     • Weights are updated at the end of each segment (batch of iterations):
         w_new = (1 − r) · w_old  +  r · (π / θ)
       where π is the accumulated score, θ is the number of times the
       operator was used, and r is the reaction factor (learning rate).
     • This makes the algorithm SELF-TUNING: effective operators are used
       more often; poor operators are gradually phased out.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .encoding_and_read import ProblemData, Request, RequestType, Solution1D
from .fitness import evaluate
from .init import greedy_init

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         ALNS PARAMETERS                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ALNSConfig:
    """Configuration for ALNS."""
    max_iterations: int = 2000
    segment_length: int = 100       # iterations per weight-update segment

    # Destruction degree: fraction of non-zero genes to remove
    destroy_fraction_min: float = 0.10
    destroy_fraction_max: float = 0.40

    # ── Adaptive weight parameters ───────────────────────────────────────
    reaction_factor: float = 0.15   # r : learning rate for weight update
    sigma_1: float = 33.0           # reward: new global best
    sigma_2: float = 9.0            # reward: improves current
    sigma_3: float = 3.0            # reward: accepted (worse but accepted)

    # ── Simulated Annealing acceptance ───────────────────────────────────
    sa_start_temperature: float = 100.0
    sa_cooling_rate: float = 0.9995

    seed: int | None = None
    time_limit_seconds: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      HELPER: CHROMOSOME SURGERY                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _non_zero_positions(chromosome: List[int]) -> List[int]:
    """Return indices of all non-zero genes."""
    return [i for i, g in enumerate(chromosome) if g != 0]


def _remove_genes(
    chromosome: List[int],
    genes_to_remove: List[int],
) -> List[int]:
    """
    Remove specific non-zero gene values from the chromosome.

    Zeros are NEVER removed.  The chromosome shrinks by len(genes_to_remove).

    ── 1D ENCODING SAFETY ──
    • Only non-zero genes matching `genes_to_remove` are deleted.
    • All zeros remain in place.
    """
    removal_set: set = set(genes_to_remove)
    # Use a counter for duplicate values (if any) — shouldn't happen, but safe
    result: List[int] = []
    for g in chromosome:
        if g != 0 and g in removal_set:
            removal_set.discard(g)
            continue  # skip this gene (removed)
        result.append(g)
    return result


def _insert_gene(
    chromosome: List[int],
    gene: int,
    position: int,
) -> List[int]:
    """
    Insert a gene at the given position.

    ── 1D ENCODING SAFETY ──
    • The caller must ensure `position` is a valid non-zero-adjacent index.
    """
    chrom: List[int] = list(chromosome)
    chrom.insert(position, gene)
    return chrom


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        DESTROY OPERATORS                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def destroy_random(
    solution: Solution1D,
    num_remove: int,
) -> Tuple[List[int], List[int]]:
    """
    Random Removal: remove `num_remove` randomly chosen non-zero genes.

    Returns (partial_chromosome, removed_genes).

    ── 1D ENCODING SAFETY ──
    • Only non-zero positions are candidates.
    • Zeros remain pinned.
    """
    chrom: List[int] = list(solution.chromosome)
    positions: List[int] = _non_zero_positions(chrom)

    remove_count: int = min(num_remove, len(positions))
    remove_positions: List[int] = sorted(
        random.sample(positions, remove_count), reverse=True,
    )

    removed: List[int] = []
    for pos in remove_positions:
        removed.append(chrom.pop(pos))

    removed.reverse()  # restore original order
    return chrom, removed


def destroy_worst(
    solution: Solution1D,
    num_remove: int,
) -> Tuple[List[int], List[int]]:
    """
    Worst Removal: remove the non-zero genes whose removal most reduces cost.

    Heuristic approximation: estimate each gene's "cost contribution" as the
    distance from its predecessor to itself plus itself to its successor.
    Remove the genes with the highest contribution.

    ── 1D ENCODING SAFETY ──
    • Only non-zero genes are evaluated and removed.
    • Zeros (separators) are untouched.
    """
    chrom: List[int] = list(solution.chromosome)
    prob: ProblemData = solution.problem
    positions: List[int] = _non_zero_positions(chrom)
    remove_count: int = min(num_remove, len(positions))

    # Compute cost contribution of each non-zero gene
    contributions: List[Tuple[float, int]] = []
    for pos in positions:
        node: int = chrom[pos]

        # Find predecessor node (previous non-zero or depot 0)
        prev_node: int = 0  # default to depot
        for p in range(pos - 1, -1, -1):
            prev_node = chrom[p]  # could be 0 (depot) — that's fine
            break

        # Find successor node (next non-zero or depot 0)
        next_node: int = 0  # default to depot
        for p in range(pos + 1, len(chrom)):
            next_node = chrom[p]
            break

        # Cost contribution ≈ dist(prev, node) + dist(node, next) - dist(prev, next)
        cost: float = (
            prob.dist(prev_node, node)
            + prob.dist(node, next_node)
            - prob.dist(prev_node, next_node)
        )
        contributions.append((cost, pos))

    # Sort by cost descending → remove highest cost first
    contributions.sort(reverse=True)
    remove_positions: List[int] = sorted(
        [pos for _, pos in contributions[:remove_count]], reverse=True,
    )

    removed: List[int] = []
    for pos in remove_positions:
        removed.append(chrom.pop(pos))

    removed.reverse()
    return chrom, removed


def destroy_related(
    solution: Solution1D,
    num_remove: int,
) -> Tuple[List[int], List[int]]:
    """
    Related Removal (Shaw): remove nodes that are geographically close to a
    randomly chosen seed node.

    ── 1D ENCODING SAFETY ──
    • Selects a seed from non-zero genes, then picks its nearest neighbours.
    • Only non-zero genes are removed; zeros stay.
    """
    chrom: List[int] = list(solution.chromosome)
    prob: ProblemData = solution.problem
    positions: List[int] = _non_zero_positions(chrom)
    remove_count: int = min(num_remove, len(positions))

    if not positions:
        return chrom, []

    # Choose a random seed
    seed_pos: int = random.choice(positions)
    seed_node: int = chrom[seed_pos]

    # Compute distance from seed to every other non-zero gene
    distances: List[Tuple[float, int]] = []
    for pos in positions:
        if pos == seed_pos:
            continue
        node: int = chrom[pos]
        d: float = prob.dist(seed_node, node)
        distances.append((d, pos))

    distances.sort()  # nearest first
    remove_positions: List[int] = sorted(
        [seed_pos] + [pos for _, pos in distances[: remove_count - 1]],
        reverse=True,
    )

    removed: List[int] = []
    for pos in remove_positions:
        removed.append(chrom.pop(pos))

    removed.reverse()
    return chrom, removed


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         REPAIR OPERATORS                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def repair_greedy(
    partial_chrom: List[int],
    removed: List[int],
    problem: ProblemData,
) -> List[int]:
    """
    Greedy Repair: insert each removed gene at the position that causes the
    smallest increase in total route distance.

    ── 1D ENCODING SAFETY ──
    • Genes are inserted ONE AT A TIME at valid positions.
    • Valid positions: any index that is currently occupied by a non-zero gene,
      or immediately after a zero (i.e., at the start of a route segment),
      or at the end of the chromosome.
    • Zeros are never moved.  Each insertion increases chromosome length by 1.
    """
    chrom: List[int] = list(partial_chrom)

    for gene in removed:
        best_cost: float = float("inf")
        best_pos: int = 0

        # Try every possible insertion position
        for pos in range(len(chrom) + 1):
            candidate: List[int] = list(chrom)
            candidate.insert(pos, gene)

            # Quick cost estimate: dist(prev, gene) + dist(gene, next)
            prev_node: int = candidate[pos - 1] if pos > 0 else 0
            next_node: int = candidate[pos + 1] if pos < len(candidate) - 1 else 0

            insertion_cost: float = (
                problem.dist(prev_node, gene)
                + problem.dist(gene, next_node)
                - (problem.dist(prev_node, next_node) if pos > 0 else 0.0)
            )

            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_pos = pos

        chrom.insert(best_pos, gene)

    return chrom


def repair_random(
    partial_chrom: List[int],
    removed: List[int],
    problem: ProblemData,
) -> List[int]:
    """
    Random Repair: insert each removed gene at a random valid position.

    ── 1D ENCODING SAFETY ──
    • Each gene is inserted at a random index in [0, len(chrom)].
    • The chromosome grows by 1 per insertion.
    • Zeros are never moved.
    """
    chrom: List[int] = list(partial_chrom)
    random.shuffle(removed)

    for gene in removed:
        # Any position is valid; zeros will naturally act as route boundaries
        pos: int = random.randint(0, len(chrom))
        chrom.insert(pos, gene)

    return chrom


def repair_regret2(
    partial_chrom: List[int],
    removed: List[int],
    problem: ProblemData,
) -> List[int]:
    """
    Regret-2 Repair: prioritise inserting the gene whose cost difference
    between its best and second-best insertion position is largest.

    This prevents "easy" insertions from consuming good positions that are
    critical for harder-to-place genes.

    ── 1D ENCODING SAFETY ──
    • Same insertion rules as greedy_repair: one gene at a time into valid positions.
    """
    chrom: List[int] = list(partial_chrom)
    pool: List[int] = list(removed)

    while pool:
        best_regret: float = -float("inf")
        best_gene: int = pool[0]
        best_gene_pos: int = 0

        for gene in pool:
            # Compute insertion cost at every position
            costs: List[Tuple[float, int]] = []
            for pos in range(len(chrom) + 1):
                prev_node: int = chrom[pos - 1] if pos > 0 else 0
                next_node: int = chrom[pos] if pos < len(chrom) else 0
                cost: float = (
                    problem.dist(prev_node, gene)
                    + problem.dist(gene, next_node)
                    - problem.dist(prev_node, next_node)
                )
                costs.append((cost, pos))

            costs.sort()
            best_cost: float = costs[0][0]
            second_cost: float = costs[1][0] if len(costs) > 1 else best_cost
            regret: float = second_cost - best_cost

            if regret > best_regret:
                best_regret = regret
                best_gene = gene
                best_gene_pos = costs[0][1]

        chrom.insert(best_gene_pos, best_gene)
        pool.remove(best_gene)

    return chrom


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    OPERATOR WEIGHT MANAGEMENT                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class OperatorStats:
    """Track score and usage count for one operator within a segment."""
    score: float = 0.0
    uses: int = 0


def roulette_wheel_select(weights: List[float]) -> int:
    """
    Select an index via roulette-wheel (fitness-proportionate) selection.

    Parameters
    ----------
    weights : List[float]
        Non-negative weights for each candidate.

    Returns
    -------
    int
        Selected index.
    """
    total: float = sum(weights)
    if total <= 0:
        return random.randrange(len(weights))

    r: float = random.uniform(0, total)
    cumulative: float = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if cumulative >= r:
            return i
    return len(weights) - 1  # fallback


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          ALNS MAIN LOOP                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Type alias for destroy/repair callables
DestroyOp = Callable[[Solution1D, int], Tuple[List[int], List[int]]]
RepairOp = Callable[[List[int], List[int], ProblemData], List[int]]


def run_alns(
    problem: ProblemData,
    config: ALNSConfig | None = None,
    initial_solution: Solution1D | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Execute the ALNS algorithm.

    Parameters
    ----------
    problem : ProblemData
        Parsed SARP instance.
    config : ALNSConfig | None
        Hyper-parameters (uses defaults if None).
    initial_solution : Solution1D | None
        Starting point (greedy_init is used if None).

    Returns
    -------
    (best_solution, best_fitness, history)
        best_solution — the best Solution1D found.
        best_fitness  — its Min-Max objective value.
        history       — best fitness per iteration for plotting.
    """
    if config is None:
        config = ALNSConfig()
    if config.seed is not None:
        random.seed(config.seed)

    start_time: float = time.time()

    # ── Initial solution ─────────────────────────────────────────────────
    if initial_solution is None:
        current: Solution1D = greedy_init(problem)
    else:
        current = initial_solution.copy()

    current_fitness: float = evaluate(current)
    best_solution: Solution1D = current.copy()
    best_fitness: float = current_fitness
    history: List[float] = [best_fitness]

    # ── Register operators ───────────────────────────────────────────────
    destroy_ops: List[DestroyOp] = [
        destroy_random,
        destroy_worst,
        destroy_related,
    ]
    repair_ops: List[RepairOp] = [
        repair_greedy,
        repair_random,
        repair_regret2,
    ]
    destroy_names: List[str] = ["random", "worst", "related"]
    repair_names: List[str] = ["greedy", "random", "regret-2"]

    # ── Adaptive weights (initialised uniformly) ─────────────────────────
    num_destroy: int = len(destroy_ops)
    num_repair: int = len(repair_ops)
    destroy_weights: List[float] = [1.0] * num_destroy
    repair_weights: List[float] = [1.0] * num_repair

    # Per-segment statistics
    destroy_stats: List[OperatorStats] = [OperatorStats() for _ in range(num_destroy)]
    repair_stats: List[OperatorStats] = [OperatorStats() for _ in range(num_repair)]

    # ── Simulated Annealing temperature ──────────────────────────────────
    temperature: float = config.sa_start_temperature

    logger.info(
        "ALNS started | max_iter=%d | segment=%d | T0=%.2f",
        config.max_iterations, config.segment_length, temperature,
    )

    # ── Main loop ────────────────────────────────────────────────────────
    for iteration in range(1, config.max_iterations + 1):
        elapsed: float = time.time() - start_time
        if elapsed >= config.time_limit_seconds:
            logger.info("ALNS time limit reached at iteration %d", iteration)
            break

        # ── Determine destruction degree ─────────────────────────────────
        non_zeros: int = len(_non_zero_positions(current.chromosome))
        num_remove: int = random.randint(
            max(1, int(non_zeros * config.destroy_fraction_min)),
            max(1, int(non_zeros * config.destroy_fraction_max)),
        )

        # ── Select operators via roulette wheel ──────────────────────────
        d_idx: int = roulette_wheel_select(destroy_weights)
        r_idx: int = roulette_wheel_select(repair_weights)

        # ── Destroy ──────────────────────────────────────────────────────
        partial_chrom, removed = destroy_ops[d_idx](current, num_remove)

        # ── Repair ───────────────────────────────────────────────────────
        repaired_chrom: List[int] = repair_ops[r_idx](
            partial_chrom, removed, problem,
        )

        candidate = Solution1D(repaired_chrom, problem)
        candidate_fitness: float = evaluate(candidate)

        # ── Acceptance decision & scoring ────────────────────────────────
        reward: float = 0.0

        if candidate_fitness < best_fitness:
            # New global best
            best_fitness = candidate_fitness
            best_solution = candidate.copy()
            current = candidate
            current_fitness = candidate_fitness
            reward = config.sigma_1

        elif candidate_fitness < current_fitness:
            # Improves on current (but not global best)
            current = candidate
            current_fitness = candidate_fitness
            reward = config.sigma_2

        else:
            # Worse solution — accept with SA probability
            delta: float = candidate_fitness - current_fitness
            if temperature > 1e-12 and random.random() < math.exp(-delta / temperature):
                current = candidate
                current_fitness = candidate_fitness
                reward = config.sigma_3
            # else: reject, reward stays 0

        # ── Update operator statistics ───────────────────────────────────
        destroy_stats[d_idx].score += reward
        destroy_stats[d_idx].uses += 1
        repair_stats[r_idx].score += reward
        repair_stats[r_idx].uses += 1

        history.append(best_fitness)

        # ── Cool down ────────────────────────────────────────────────────
        temperature *= config.sa_cooling_rate

        # ── Segment boundary: update adaptive weights ────────────────────
        if iteration % config.segment_length == 0:
            r: float = config.reaction_factor

            for i in range(num_destroy):
                if destroy_stats[i].uses > 0:
                    avg_score: float = destroy_stats[i].score / destroy_stats[i].uses
                    destroy_weights[i] = (
                        (1 - r) * destroy_weights[i] + r * avg_score
                    )
                # Reset segment stats
                destroy_stats[i] = OperatorStats()

            for i in range(num_repair):
                if repair_stats[i].uses > 0:
                    avg_score = repair_stats[i].score / repair_stats[i].uses
                    repair_weights[i] = (
                        (1 - r) * repair_weights[i] + r * avg_score
                    )
                repair_stats[i] = OperatorStats()

            logger.debug(
                "Iter %d | weight update | destroy=%s | repair=%s",
                iteration,
                [f"{w:.2f}" for w in destroy_weights],
                [f"{w:.2f}" for w in repair_weights],
            )

        if iteration % 200 == 0 or iteration == 1:
            logger.info(
                "Iter %4d | best=%.4f | current=%.4f | T=%.4f | "
                "d_wt=%s | r_wt=%s",
                iteration, best_fitness, current_fitness, temperature,
                [f"{w:.2f}" for w in destroy_weights],
                [f"{w:.2f}" for w in repair_weights],
            )

    logger.info("ALNS finished | best_fitness=%.4f", best_fitness)
    return best_solution, best_fitness, history
