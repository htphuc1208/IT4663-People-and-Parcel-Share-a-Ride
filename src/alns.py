"""
alns.py — Adaptive Large Neighbourhood Search for the SARP (Min-Max)
=====================================================================

Iteratively destroys and repairs the current solution using a portfolio of
operators.  Operator selection uses an **adaptive roulette-wheel** mechanism
that rewards operators producing improvements.

═══════════════════════════════════════════════════════════════════════════
CRITICAL: 1D ENCODING MANIPULATION RULES
═══════════════════════════════════════════════════════════════════════════
All destroy and repair operators MUST obey:

  1. NEVER duplicate or remove `0` separators.
     • Destroy removes non-zero genes only; zeros stay pinned.
     • Repair reinserts the removed genes at non-zero-adjacent positions.

  2. Destroy: extract a subset of non-zero genes → "removal pool".
     • Chromosome shrinks by the number of removed genes.
     • Consecutive zeros are fine: they indicate an empty route.

  3. Repair: reinsert every gene from the removal pool.
     • Insertion is allowed at ANY index (genes shift zeros right/left but
       never remove them).  After full repair, chromosome length is restored.

  4. The set of non-zero node IDs must be identical before and after each
     destroy+repair cycle.

  5. ADAPTIVE ROULETTE-WHEEL:
     • Weights are updated every `segment_length` iterations:
         w_new = (1 − r) · w_old  +  r · (π / θ)
       (π = accumulated segment score, θ = segment usage count, r = reaction factor)
     • Rewards: σ₁ (new global best) > σ₂ (improves current) > σ₃ (SA-accepted)

  6. MIN-MAX AWARE REPAIR:
     • repair_greedy and repair_regret2 use route-balance heuristic:
       for each gene, choose the insertion position that minimises the
       resulting MAXIMUM route cost across all K taxis (not total cost).
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from .encoding_and_read import ProblemData, Solution1D
from .fitness import evaluate
from .init import greedy_init

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         ALNS PARAMETERS                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ALNSConfig:
    """Hyper-parameters for ALNS."""
    max_iterations: int = 2000
    segment_length: int = 100

    destroy_fraction_min: float = 0.10
    destroy_fraction_max: float = 0.40

    reaction_factor: float = 0.15
    sigma_1: float = 33.0
    sigma_2: float = 9.0
    sigma_3: float = 3.0

    sa_start_temperature: float = 100.0
    sa_cooling_rate: float = 0.9995

    seed: int | None = None
    time_limit_seconds: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      HELPER: CHROMOSOME SURGERY                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _non_zero_positions(chromosome: List[int]) -> List[int]:
    return [i for i, g in enumerate(chromosome) if g != 0]


def _remove_genes(chromosome: List[int], genes_to_remove: List[int]) -> List[int]:
    """Remove specific non-zero gene values from chromosome; zeros stay."""
    removal_set: set = set(genes_to_remove)
    result: List[int] = []
    for g in chromosome:
        if g != 0 and g in removal_set:
            removal_set.discard(g)
            continue
        result.append(g)
    return result


def _estimate_route_costs(chrom: List[int], prob: ProblemData) -> List[float]:
    """
    Fast route-cost estimate for the partial chromosome.

    Splits by zeros and sums edge distances including depot at start/end.
    Passenger dropoffs are NOT decoded here — this is an approximation
    used for Min-Max balance decisions during repair.
    """
    segments: List[List[int]] = []
    current: List[int] = []
    for g in chrom:
        if g == 0:
            segments.append(current)
            current = []
        else:
            current.append(g)
    segments.append(current)

    costs: List[float] = []
    for seg in segments:
        nodes = [0] + seg + [0]
        cost = sum(prob.dist(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1))
        costs.append(cost)
    return costs


def _route_idx_for_pos(pos: int, chrom: List[int]) -> int:
    """Return which route (0-indexed) an insertion at `pos` falls into."""
    return sum(1 for i, g in enumerate(chrom) if i < pos and g == 0)


def _delta_cost(prev_node: int, gene: int, next_node: int, prob: ProblemData) -> float:
    """Edge-delta when inserting `gene` between prev_node and next_node."""
    return (prob.dist(prev_node, gene)
            + prob.dist(gene, next_node)
            - prob.dist(prev_node, next_node))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        DESTROY OPERATORS                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def destroy_random(
    solution: Solution1D,
    num_remove: int,
) -> Tuple[List[int], List[int]]:
    """Random Removal: remove `num_remove` randomly chosen non-zero genes."""
    chrom: List[int] = list(solution.chromosome)
    positions: List[int] = _non_zero_positions(chrom)
    remove_count: int = min(num_remove, len(positions))
    remove_positions: List[int] = sorted(
        random.sample(positions, remove_count), reverse=True
    )
    removed: List[int] = []
    for pos in remove_positions:
        removed.append(chrom.pop(pos))
    removed.reverse()
    return chrom, removed


def destroy_worst(
    solution: Solution1D,
    num_remove: int,
) -> Tuple[List[int], List[int]]:
    """
    Worst Removal: remove non-zero genes with highest local cost contribution.

    Cost contribution ≈ dist(prev, node) + dist(node, next) − dist(prev, next).
    """
    chrom: List[int] = list(solution.chromosome)
    prob: ProblemData = solution.problem
    positions: List[int] = _non_zero_positions(chrom)
    remove_count: int = min(num_remove, len(positions))

    contributions: List[Tuple[float, int]] = []
    for pos in positions:
        node: int = chrom[pos]
        prev_node: int = chrom[pos - 1] if pos > 0 else 0
        next_node: int = chrom[pos + 1] if pos + 1 < len(chrom) else 0
        cost: float = _delta_cost(prev_node, node, next_node, prob)
        contributions.append((cost, pos))

    contributions.sort(reverse=True)
    remove_positions: List[int] = sorted(
        [pos for _, pos in contributions[:remove_count]], reverse=True
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
    Shaw (Related) Removal: remove geographically close nodes to a random seed.
    """
    chrom: List[int] = list(solution.chromosome)
    prob: ProblemData = solution.problem
    positions: List[int] = _non_zero_positions(chrom)
    remove_count: int = min(num_remove, len(positions))

    if not positions:
        return chrom, []

    seed_pos: int = random.choice(positions)
    seed_node: int = chrom[seed_pos]

    distances: List[Tuple[float, int]] = [
        (prob.dist(seed_node, chrom[pos]), pos)
        for pos in positions
        if pos != seed_pos
    ]
    distances.sort()

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
    Min-Max-aware Greedy Repair.

    For each removed gene, find the insertion position that minimises the
    resulting MAXIMUM estimated route cost across all K vehicles.

    This balances workload between taxis, which is critical for the Min-Max
    objective (minimise the longest route).
    """
    chrom: List[int] = list(partial_chrom)

    for gene in removed:
        best_max_cost: float = float("inf")
        best_pos: int = 0

        # Current estimated costs for all K routes
        route_costs: List[float] = _estimate_route_costs(chrom, problem)

        for pos in range(len(chrom) + 1):
            r_idx: int = _route_idx_for_pos(pos, chrom)
            prev_node: int = chrom[pos - 1] if pos > 0 else 0
            next_node: int = chrom[pos] if pos < len(chrom) else 0
            delta: float = _delta_cost(prev_node, gene, next_node, problem)

            costs: List[float] = list(route_costs)
            costs[r_idx] += delta
            new_max: float = max(costs)

            if new_max < best_max_cost:
                best_max_cost = new_max
                best_pos = pos

        chrom.insert(best_pos, gene)

    return chrom


def repair_random(
    partial_chrom: List[int],
    removed: List[int],
    problem: ProblemData,
) -> List[int]:
    """Random Repair: insert each removed gene at a random position."""
    chrom: List[int] = list(partial_chrom)
    random.shuffle(removed)
    for gene in removed:
        pos: int = random.randint(0, len(chrom))
        chrom.insert(pos, gene)
    return chrom


def repair_regret2(
    partial_chrom: List[int],
    removed: List[int],
    problem: ProblemData,
) -> List[int]:
    """
    Min-Max-aware Regret-2 Repair.

    Prioritises inserting the gene whose difference between its best and
    second-best MIN-MAX cost is largest.  This avoids wasting good balanced
    positions on genes that can be placed cheaply anywhere.
    """
    chrom: List[int] = list(partial_chrom)
    pool: List[int] = list(removed)

    while pool:
        best_regret: float = -float("inf")
        best_gene: int = pool[0]
        best_gene_pos: int = 0

        route_costs: List[float] = _estimate_route_costs(chrom, problem)

        for gene in pool:
            max_costs: List[Tuple[float, int]] = []

            for pos in range(len(chrom) + 1):
                r_idx: int = _route_idx_for_pos(pos, chrom)
                prev_node: int = chrom[pos - 1] if pos > 0 else 0
                next_node: int = chrom[pos] if pos < len(chrom) else 0
                delta: float = _delta_cost(prev_node, gene, next_node, problem)

                costs: List[float] = list(route_costs)
                costs[r_idx] += delta
                max_costs.append((max(costs), pos))

            max_costs.sort()
            best_max: float = max_costs[0][0]
            second_max: float = max_costs[1][0] if len(max_costs) > 1 else best_max
            regret: float = second_max - best_max

            if regret > best_regret:
                best_regret = regret
                best_gene = gene
                best_gene_pos = max_costs[0][1]

        chrom.insert(best_gene_pos, best_gene)
        pool.remove(best_gene)

    return chrom


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    OPERATOR WEIGHT MANAGEMENT                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class OperatorStats:
    score: float = 0.0
    uses: int = 0


def roulette_wheel_select(weights: List[float]) -> int:
    """Fitness-proportionate (roulette-wheel) selection."""
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

DestroyOp = Callable[[Solution1D, int], Tuple[List[int], List[int]]]
RepairOp = Callable[[List[int], List[int], ProblemData], List[int]]


def run_alns(
    problem: ProblemData,
    config: ALNSConfig | None = None,
    initial_solution: Solution1D | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Execute the ALNS algorithm.

    Returns
    -------
    (best_solution, best_fitness, history)
    """
    if config is None:
        config = ALNSConfig()
    if config.seed is not None:
        random.seed(config.seed)

    start_time: float = time.time()

    current: Solution1D = (
        greedy_init(problem) if initial_solution is None else initial_solution.copy()
    )
    current_fitness: float = evaluate(current)
    best_solution: Solution1D = current.copy()
    best_fitness: float = current_fitness
    history: List[float] = [best_fitness]

    # ── Register operators ───────────────────────────────────────────────
    destroy_ops: List[DestroyOp] = [destroy_random, destroy_worst, destroy_related]
    repair_ops: List[RepairOp] = [repair_greedy, repair_random, repair_regret2]
    destroy_names: List[str] = ["random", "worst", "related"]
    repair_names: List[str] = ["greedy", "random", "regret-2"]

    num_d: int = len(destroy_ops)
    num_r: int = len(repair_ops)
    destroy_weights: List[float] = [1.0] * num_d
    repair_weights: List[float] = [1.0] * num_r
    destroy_stats: List[OperatorStats] = [OperatorStats() for _ in range(num_d)]
    repair_stats: List[OperatorStats] = [OperatorStats() for _ in range(num_r)]

    temperature: float = config.sa_start_temperature

    logger.info(
        "ALNS started | max_iter=%d | segment=%d | T0=%.2f",
        config.max_iterations, config.segment_length, temperature,
    )

    # ── Main loop ────────────────────────────────────────────────────────
    for iteration in range(1, config.max_iterations + 1):
        if time.time() - start_time >= config.time_limit_seconds:
            logger.info("ALNS time limit reached at iteration %d", iteration)
            break

        # Destruction degree
        non_zeros: int = len(_non_zero_positions(current.chromosome))
        num_remove: int = random.randint(
            max(1, int(non_zeros * config.destroy_fraction_min)),
            max(1, int(non_zeros * config.destroy_fraction_max)),
        )

        # Select operators
        d_idx: int = roulette_wheel_select(destroy_weights)
        r_idx: int = roulette_wheel_select(repair_weights)

        # Destroy → Repair
        partial_chrom, removed = destroy_ops[d_idx](current, num_remove)
        repaired_chrom: List[int] = repair_ops[r_idx](partial_chrom, removed, problem)

        candidate = Solution1D(repaired_chrom, problem)
        candidate_fitness: float = evaluate(candidate)

        # Acceptance + reward
        reward: float = 0.0
        if candidate_fitness < best_fitness:
            best_fitness = candidate_fitness
            best_solution = candidate.copy()
            current = candidate
            current_fitness = candidate_fitness
            reward = config.sigma_1
        elif candidate_fitness < current_fitness:
            current = candidate
            current_fitness = candidate_fitness
            reward = config.sigma_2
        else:
            delta: float = candidate_fitness - current_fitness
            if temperature > 1e-12 and random.random() < math.exp(-delta / temperature):
                current = candidate
                current_fitness = candidate_fitness
                reward = config.sigma_3

        destroy_stats[d_idx].score += reward
        destroy_stats[d_idx].uses += 1
        repair_stats[r_idx].score += reward
        repair_stats[r_idx].uses += 1

        history.append(best_fitness)
        temperature *= config.sa_cooling_rate

        # ── Segment boundary: update adaptive weights ────────────────────
        if iteration % config.segment_length == 0:
            r: float = config.reaction_factor
            for i in range(num_d):
                if destroy_stats[i].uses > 0:
                    avg = destroy_stats[i].score / destroy_stats[i].uses
                    destroy_weights[i] = (1 - r) * destroy_weights[i] + r * avg
                destroy_stats[i] = OperatorStats()
            for i in range(num_r):
                if repair_stats[i].uses > 0:
                    avg = repair_stats[i].score / repair_stats[i].uses
                    repair_weights[i] = (1 - r) * repair_weights[i] + r * avg
                repair_stats[i] = OperatorStats()

            logger.debug(
                "Iter %d | d_wt=%s | r_wt=%s",
                iteration,
                [f"{w:.2f}" for w in destroy_weights],
                [f"{w:.2f}" for w in repair_weights],
            )

        if iteration % 200 == 0 or iteration == 1:
            logger.info(
                "Iter %4d | best=%.4f | cur=%.4f | T=%.4f | d=%s | r=%s",
                iteration, best_fitness, current_fitness, temperature,
                [f"{w:.2f}" for w in destroy_weights],
                [f"{w:.2f}" for w in repair_weights],
            )

    logger.info("ALNS finished | best_fitness=%.4f", best_fitness)
    return best_solution, best_fitness, history
