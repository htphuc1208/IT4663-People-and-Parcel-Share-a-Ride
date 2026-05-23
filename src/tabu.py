"""
tabu.py — Tabu Search for the Share-a-Ride Problem (Min-Max)
=============================================================

This module implements a Tabu Search (TS) meta-heuristic that operates
EXCLUSIVELY on the unified 1D chromosome representation (`Solution1D`).

═══════════════════════════════════════════════════════════════════════════
CRITICAL: 1D ENCODING MANIPULATION RULES
═══════════════════════════════════════════════════════════════════════════
All neighbourhood moves MUST obey these rules:

  1. NEVER duplicate or remove `0` separators.
     • The chromosome always contains exactly (K − 1) zeros.
     • Zeros are immovable partition walls.  A move that swaps a zero with
       a non-zero entry would merge or split vehicle routes, which changes
       the problem structure.  This is FORBIDDEN.

  2. Moves operate ONLY on non-zero genes.
     • Swap: exchange two non-zero genes at different positions.
     • Relocate: remove a non-zero gene and re-insert it at another
       non-zero-adjacent position.
     • 2-opt (intra-route): reverse a sub-sequence WITHIN one route
       segment (between two consecutive zeros / chromosome boundaries).
     All these moves preserve zero positions and the set of non-zero IDs.

  3. Tabu attribute:
     • A move is recorded as the tuple (node_id, from_position, to_position).
     • If a move's inverse is in the tabu list, the move is forbidden UNLESS
       it satisfies the aspiration criterion (produces a new global best).

  4. ADAPTIVE TABU TENURE:
     • The tabu tenure (the number of iterations a move stays forbidden)
       starts at an initial value and is adjusted dynamically:
       — If the best solution improves, the tenure is DECREASED (allowing
         more moves) to exploit the promising region.
       — If the search stagnates for `stagnation_limit` iterations, the
         tenure is INCREASED (blocking more moves) to diversify the search.
     • Tenure is clamped between `tenure_min` and `tenure_max`.
     • This self-regulating mechanism balances intensification and
       diversification without manual parameter tuning.

  5. Every neighbour solution is a valid `Solution1D` — same length,
     same separator count, same set of non-zero node IDs.
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Set, Tuple

from .encoding_and_read import ProblemData, Solution1D
from .fitness import evaluate
from .init import greedy_init

logger = logging.getLogger(__name__)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        TABU SEARCH PARAMETERS                           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class TabuConfig:
    """Configuration for Tabu Search."""
    max_iterations: int = 1000
    neighbourhood_sample_size: int = 30  # N(s) candidates per iteration

    # ── Adaptive tabu tenure ─────────────────────────────────────────────
    tenure_init: int = 10
    tenure_min: int = 5
    tenure_max: int = 40
    tenure_increase: int = 2       # bump when stagnating
    tenure_decrease: int = 1       # reduce when improving
    stagnation_limit: int = 20     # iterations without improvement

    seed: int | None = None
    time_limit_seconds: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           TABU LIST                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class TabuList:
    """
    Fixed-capacity tabu list storing move attributes as (node, from, to).

    The capacity (tenure) is mutable — the adaptive mechanism adjusts it
    at runtime.
    """

    def __init__(self, tenure: int) -> None:
        self._tenure: int = tenure
        self._moves: Deque[Tuple[int, int, int]] = deque(maxlen=tenure)

    @property
    def tenure(self) -> int:
        return self._tenure

    @tenure.setter
    def tenure(self, value: int) -> None:
        self._tenure = value
        # Rebuild deque with new maxlen (Python deques don't allow resize)
        old_items: List[Tuple[int, int, int]] = list(self._moves)
        self._moves = deque(old_items[-value:], maxlen=value)

    def add(self, move: Tuple[int, int, int]) -> None:
        """Record a move as tabu."""
        self._moves.append(move)

    def is_tabu(self, move: Tuple[int, int, int]) -> bool:
        """Check whether a move or its inverse is tabu."""
        node, frm, to = move
        inverse: Tuple[int, int, int] = (node, to, frm)
        return move in self._moves or inverse in self._moves


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     NEIGHBOURHOOD GENERATORS                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _non_zero_positions(chromosome: List[int]) -> List[int]:
    """Return indices of all non-zero genes."""
    return [i for i, g in enumerate(chromosome) if g != 0]


def generate_swap_neighbour(
    solution: Solution1D,
) -> Tuple[Solution1D, Tuple[int, int, int]]:
    """
    Generate a neighbour by swapping two random non-zero genes.

    Returns (neighbour, move_attribute) where move_attribute = (node, pos_from, pos_to).

    ── 1D ENCODING SAFETY ──
    • Only non-zero positions are selected.
    • Zeros are never touched.
    • The set of non-zero IDs is preserved (swap is bijective).
    """
    chrom: List[int] = list(solution.chromosome)
    positions: List[int] = _non_zero_positions(chrom)

    i, j = random.sample(positions, 2)
    node_id: int = chrom[i]
    chrom[i], chrom[j] = chrom[j], chrom[i]

    move: Tuple[int, int, int] = (node_id, i, j)
    return Solution1D(chrom, solution.problem), move


def generate_relocate_neighbour(
    solution: Solution1D,
) -> Tuple[Solution1D, Tuple[int, int, int]]:
    """
    Generate a neighbour by relocating a random non-zero gene to another
    non-zero-adjacent position.

    ── 1D ENCODING SAFETY ──
    • Only non-zero genes are moved.
    • The gene is removed and reinserted; no zeros are affected.
    • The set of non-zero IDs remains unchanged (same ID, new position).
    """
    chrom: List[int] = list(solution.chromosome)
    positions: List[int] = _non_zero_positions(chrom)

    if len(positions) < 2:
        move: Tuple[int, int, int] = (0, 0, 0)
        return solution.copy(), move

    src_pos: int = random.choice(positions)
    node_id: int = chrom.pop(src_pos)

    # Re-identify valid insertion points (between non-zeros / at segment edges)
    new_positions: List[int] = _non_zero_positions(chrom)
    if new_positions:
        dst_pos: int = random.choice(new_positions)
    else:
        dst_pos = 0

    chrom.insert(dst_pos, node_id)
    move = (node_id, src_pos, dst_pos)
    return Solution1D(chrom, solution.problem), move


def generate_intra_route_2opt(
    solution: Solution1D,
) -> Tuple[Solution1D, Tuple[int, int, int]]:
    """
    Reverse a sub-segment within a single route (intra-route 2-opt).

    ── 1D ENCODING SAFETY ──
    • Selects a random route segment (between two zeros / boundaries).
    • Reverses the non-zero portion within that segment.
    • Zeros are untouched.
    • No node IDs are added or removed.
    """
    chrom: List[int] = list(solution.chromosome)
    routes: List[List[int]] = solution.get_routes()

    non_empty: List[int] = [k for k, r in enumerate(routes) if len(r) >= 2]
    if not non_empty:
        return solution.copy(), (0, 0, 0)

    route_idx: int = random.choice(non_empty)
    route: List[int] = routes[route_idx]

    i, j = sorted(random.sample(range(len(route)), 2))
    # For move attribute, record first node and positions
    node_id: int = route[i]
    route[i:j + 1] = reversed(route[i:j + 1])

    # Write back into chromosome
    pos: int = 0
    for k in range(route_idx):
        pos += len(routes[k]) + 1  # +1 for separator
    for offset, node in enumerate(route):
        chrom[pos + offset] = node

    move: Tuple[int, int, int] = (node_id, pos + i, pos + j)
    return Solution1D(chrom, solution.problem), move


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      TABU SEARCH MAIN LOOP                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_tabu(
    problem: ProblemData,
    config: TabuConfig | None = None,
    initial_solution: Solution1D | None = None,
) -> Tuple[Solution1D, float, List[float]]:
    """
    Execute the Tabu Search algorithm.

    Parameters
    ----------
    problem : ProblemData
        Parsed SARP instance.
    config : TabuConfig | None
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
        config = TabuConfig()
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

    # ── Tabu list & adaptive tenure ──────────────────────────────────────
    tabu = TabuList(tenure=config.tenure_init)
    stagnation_counter: int = 0

    # Neighbourhood generators (sampled stochastically each iteration)
    move_generators = [
        generate_swap_neighbour,
        generate_relocate_neighbour,
        generate_intra_route_2opt,
    ]

    logger.info(
        "Tabu Search started | max_iter=%d | tenure_init=%d",
        config.max_iterations, config.tenure_init,
    )

    # ── Main loop ────────────────────────────────────────────────────────
    for iteration in range(1, config.max_iterations + 1):
        elapsed: float = time.time() - start_time
        if elapsed >= config.time_limit_seconds:
            logger.info("Tabu time limit reached at iteration %d", iteration)
            break

        # ── Generate neighbourhood ───────────────────────────────────────
        candidates: List[Tuple[Solution1D, float, Tuple[int, int, int]]] = []

        for _ in range(config.neighbourhood_sample_size):
            gen = random.choice(move_generators)
            neighbour, move = gen(current)
            fit: float = evaluate(neighbour)
            candidates.append((neighbour, fit, move))

        # ── Select best admissible move ──────────────────────────────────
        candidates.sort(key=lambda c: c[1])

        chosen: Optional[Tuple[Solution1D, float, Tuple[int, int, int]]] = None
        for neighbour, fit, move in candidates:
            if not tabu.is_tabu(move):
                chosen = (neighbour, fit, move)
                break
            # Aspiration criterion: accept if it's a new global best
            elif fit < best_fitness:
                chosen = (neighbour, fit, move)
                break

        if chosen is None:
            # All moves are tabu and none satisfy aspiration — take the best
            chosen = candidates[0]

        current, current_fitness, move = chosen
        tabu.add(move)

        # ── Update global best ───────────────────────────────────────────
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_solution = current.copy()
            stagnation_counter = 0

            # ADAPTIVE: decrease tenure → intensify around improving region
            tabu.tenure = max(tabu.tenure - config.tenure_decrease, config.tenure_min)
        else:
            stagnation_counter += 1

        history.append(best_fitness)

        # ── Adaptive tenure adjustment ───────────────────────────────────
        # If stagnation persists → INCREASE tenure to diversify
        if stagnation_counter >= config.stagnation_limit:
            tabu.tenure = min(tabu.tenure + config.tenure_increase, config.tenure_max)
            stagnation_counter = 0
            logger.debug(
                "Iter %d: stagnation → tenure ↑ %d", iteration, tabu.tenure,
            )

        if iteration % 100 == 0 or iteration == 1:
            logger.info(
                "Iter %4d | best=%.4f | current=%.4f | tenure=%d | stag=%d",
                iteration, best_fitness, current_fitness,
                tabu.tenure, stagnation_counter,
            )

    logger.info("Tabu Search finished | best_fitness=%.4f", best_fitness)
    return best_solution, best_fitness, history
