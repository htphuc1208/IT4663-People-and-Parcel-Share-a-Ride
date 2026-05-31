"""
main.py — CLI Entry Point for the SARP Min-Max Optimizer
=========================================================

This script provides an ``argparse``-based command-line interface that:
  1. Reads a SARP instance from one of the 3 data folds.
  2. Generates an initial solution via greedy heuristic.
  3. Runs the user-selected algorithm (GA, Tabu, ALNS, or all).
  4. Logs progress and saves results to the ``results/`` directory.

Usage Examples
--------------
  # Run GA on a single instance
  python -m src.main --instance data/fold1/instance_01.txt --algorithm ga

  # Run all algorithms with custom seeds and time limit
  python -m src.main --instance data/fold2/instance_05.txt --algorithm all \\
      --seed 42 --time-limit 300

  # Run Tabu Search on every instance in fold 3
  python -m src.main --fold data/fold3 --algorithm tabu --seed 0

  # Verbose debug logging
  python -m src.main --instance data/fold1/instance_01.txt --algorithm alns -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ── Local imports ────────────────────────────────────────────────────────────
from .encoding_and_read import ProblemData, Solution1D, read_instance, list_instances_in_fold
from .fitness import evaluate, evaluate_detailed, decode_routes
from .init import greedy_init
from .ga import GAConfig, run_ga
from .tabu import TabuConfig, run_tabu
from .alns import ALNSConfig, run_alns

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                           LOGGING SETUP                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _setup_logging(verbose: bool, log_file: str | None = None) -> None:
    """Configure root logger with console + optional file handlers."""
    level: int = logging.DEBUG if verbose else logging.INFO
    fmt: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt: str = "%Y-%m-%d %H:%M:%S"

    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        RESULT SERIALISATION                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _save_results(
    result_dir: str,
    instance_name: str,
    algorithm: str,
    best_solution: Solution1D,
    best_fitness: float,
    history: List[float],
    elapsed_seconds: float,
    config: Dict[str, Any],
) -> str:
    """
    Save results to a JSON file in the results directory.

    Returns the path to the saved file.
    """
    os.makedirs(result_dir, exist_ok=True)

    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename: str = f"{instance_name}_{algorithm}_{timestamp}.json"
    filepath: str = os.path.join(result_dir, filename)

    # Decode routes for human-readable output
    routes: List[List[int]] = decode_routes(best_solution)
    fitness_val, distances, penalties, totals = evaluate_detailed(best_solution)

    result: Dict[str, Any] = {
        "instance": instance_name,
        "algorithm": algorithm,
        "timestamp": timestamp,
        "best_fitness": best_fitness,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "chromosome": best_solution.chromosome,
        "decoded_routes": routes,
        "route_distances": [round(d, 4) for d in distances],
        "route_penalties": [round(p, 4) for p in penalties],
        "route_totals": [round(t, 4) for t in totals],
        "history_length": len(history),
        "history_first_10": history[:10],
        "history_last_10": history[-10:],
        "config": config,
    }

    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, default=str)

    return filepath


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       ALGORITHM DISPATCH                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _run_algorithm(
    algorithm: str,
    problem: ProblemData,
    seed: int | None,
    time_limit: float,
    result_dir: str,
) -> None:
    """
    Dispatch to the selected algorithm and save results.

    Parameters
    ----------
    algorithm : str
        One of 'ga', 'tabu', 'alns'.
    problem : ProblemData
        Parsed instance data.
    seed : int | None
        Random seed.
    time_limit : float
        Wall-clock time limit in seconds.
    result_dir : str
        Path to the results directory.
    """
    logger = logging.getLogger("main")
    logger.info(
        "Running %s on instance '%s' (N=%d, M=%d, K=%d)",
        algorithm.upper(), problem.name, problem.N, problem.M, problem.K,
    )

    t0: float = time.time()

    if algorithm == "ga":
        config = GAConfig(seed=seed, time_limit_seconds=time_limit)
        best_sol, best_fit, hist = run_ga(problem, config)
        cfg_dict = config.__dict__

    elif algorithm == "tabu":
        config_ts = TabuConfig(seed=seed, time_limit_seconds=time_limit)
        best_sol, best_fit, hist = run_tabu(problem, config_ts)
        cfg_dict = config_ts.__dict__

    elif algorithm == "alns":
        config_al = ALNSConfig(seed=seed, time_limit_seconds=time_limit)
        best_sol, best_fit, hist = run_alns(problem, config_al)
        cfg_dict = config_al.__dict__

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    elapsed: float = time.time() - t0

    # Validate the final solution
    is_valid, msg = best_sol.validate()
    if not is_valid:
        logger.warning("INVALID solution: %s", msg)
    else:
        logger.info("Solution validation: OK")

    # Save results
    filepath: str = _save_results(
        result_dir=result_dir,
        instance_name=problem.name,
        algorithm=algorithm,
        best_solution=best_sol,
        best_fitness=best_fit,
        history=hist,
        elapsed_seconds=elapsed,
        config=cfg_dict,
    )

    logger.info(
        "%s finished | fitness=%.4f | time=%.2fs | saved → %s",
        algorithm.upper(), best_fit, elapsed, filepath,
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       ARGUMENT PARSER                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="sarp_optimizer",
        description=(
            "Share-a-Ride Problem (SARP) Min-Max Optimizer — "
            "GA / Tabu Search / ALNS with Unified 1D Encoding"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input source (mutually exclusive) ────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--instance", "-i",
        type=str,
        help="Path to a single .txt instance file.",
    )
    input_group.add_argument(
        "--fold", "-f",
        type=str,
        help="Path to a data fold directory (runs all .txt files inside).",
    )

    # ── Algorithm selection ──────────────────────────────────────────────
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        choices=["ga", "tabu", "alns", "all"],
        default="all",
        help="Algorithm to run (default: all).",
    )

    # ── Execution parameters ─────────────────────────────────────────────
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--time-limit", "-t",
        type=float,
        default=float("inf"),
        help="Wall-clock time limit in seconds per algorithm run.",
    )
    parser.add_argument(
        "--results-dir", "-r",
        type=str,
        default="results",
        help="Directory to save result JSON files (default: results/).",
    )

    # ── Verbosity ────────────────────────────────────────────────────────
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    return parser


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """CLI entry point."""
    parser: argparse.ArgumentParser = build_parser()
    args = parser.parse_args()

    # ── Set up logging ───────────────────────────────────────────────────
    log_file: str = os.path.join(
        args.results_dir,
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    _setup_logging(verbose=args.verbose, log_file=log_file)
    logger = logging.getLogger("main")

    logger.info("=" * 70)
    logger.info("SARP Min-Max Optimizer — Session started")
    logger.info("=" * 70)

    # ── Resolve instance files ───────────────────────────────────────────
    instance_files: List[str] = []
    if args.instance:
        instance_files = [args.instance]
    elif args.fold:
        instance_files = list_instances_in_fold(args.fold)
        if not instance_files:
            logger.error("No .txt files found in fold: %s", args.fold)
            sys.exit(1)
        logger.info("Found %d instances in fold %s", len(instance_files), args.fold)

    # ── Determine algorithms to run ──────────────────────────────────────
    algorithms: List[str] = (
        ["ga", "tabu", "alns"] if args.algorithm == "all"
        else [args.algorithm]
    )

    # ── Main execution loop ──────────────────────────────────────────────
    for filepath in instance_files:
        logger.info("─" * 50)
        logger.info("Loading instance: %s", filepath)

        try:
            problem: ProblemData = read_instance(filepath)
        except (FileNotFoundError, ValueError) as e:
            logger.error("Failed to load instance: %s", e)
            continue

        logger.info(
            "Parsed: N=%d passengers, M=%d parcels, K=%d vehicles, Q=%s",
            problem.N, problem.M, problem.K, problem.vehicle_capacities,
        )
        logger.info("Chromosome length = %d", problem.chromosome_length)

        # ── Generate initial solution (shared starting point) ────────────
        init_sol: Solution1D = greedy_init(problem, seed=args.seed)
        init_fit: float = evaluate(init_sol)
        logger.info("Greedy initial fitness = %.4f", init_fit)

        for algo in algorithms:
            _run_algorithm(
                algorithm=algo,
                problem=problem,
                seed=args.seed,
                time_limit=args.time_limit,
                result_dir=args.results_dir,
            )

    logger.info("=" * 70)
    logger.info("All runs complete. Results saved in: %s", args.results_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
