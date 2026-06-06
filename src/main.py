"""
main.py — SARP Min-Max Optimizer: GA / Tabu Search / ALNS

Usage
-----
  # Chay 3 thuat toan tren 1 instance
  python -m src.main --instance data/official/small/small_01.in

  # Gioi han thoi gian 60 giay moi thuat toan, seed co dinh
  python -m src.main --instance data/official/small/small_01.in --time-limit 60 --seed 42

  # Chay toan bo instance trong 1 thu muc
  python -m src.main --fold data/official/small

  # In them log chi tiet cua tung thuat toan
  python -m src.main --instance data/official/small/small_01.in --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .encoding_and_read import ProblemData, Solution1D, read_instance, list_instances_in_fold
from .fitness import evaluate, evaluate_detailed, decode_routes
from .ga           import GAConfig,      run_ga
from .tabu         import TabuConfig,    run_tabu
from .alns         import ALNSConfig,    run_alns
from .ortools_solver import ORToolsConfig, run_ortools


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          KET QUA MOT THUAT TOAN                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class AlgoResult:
    name: str
    solution: Solution1D
    fitness: float
    distances: List[float]
    penalties: List[float]
    elapsed: float
    history: List[float]
    feasible: bool          # True khi tong penalty = 0


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          CHAY TUNG THUAT TOAN                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _run_one(
    name: str,
    problem: ProblemData,
    seed: Optional[int],
    time_limit: float,
    index: int,
    total: int,
) -> AlgoResult:
    """Chay mot thuat toan, in trang thai, tra ve AlgoResult."""

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  [{index}/{total}] {name}  —  {problem.name}")
    print(f"  N={problem.N} khach | M={problem.M} hang | K={problem.K} xe")
    print(sep)

    t0 = time.time()

    if name == "GA":
        cfg = GAConfig(seed=seed, time_limit_seconds=time_limit)
        sol, fit, hist = run_ga(problem, cfg)

    elif name == "Tabu":
        cfg = TabuConfig(seed=seed, time_limit_seconds=time_limit)
        sol, fit, hist = run_tabu(problem, cfg)

    elif name == "ALNS":
        cfg = ALNSConfig(seed=seed, time_limit_seconds=time_limit)
        sol, fit, hist = run_alns(problem, cfg)

    else:  # OR-Tools
        cfg = ORToolsConfig(time_limit_seconds=time_limit)
        sol, fit, hist = run_ortools(problem, cfg)

    elapsed = time.time() - t0

    _, distances, penalties, _ = evaluate_detailed(sol)
    feasible = (sum(penalties) == 0.0)

    # ── In ket qua tung xe ───────────────────────────────────────────────
    print(f"\n  Ket qua chi tiet:")
    print(f"  {'Xe':>4}  {'Khoang cach':>14}  {'Penalty':>10}  {'Tong':>14}")
    print(f"  {'-'*48}")
    for k, (d, p) in enumerate(zip(distances, penalties)):
        bottleneck = " <- bottleneck" if d == max(distances) else ""
        print(f"  {k:>4}  {d:>14.2f}  {p:>10.2f}  {d+p:>14.2f}{bottleneck}")
    print(f"  {'-'*48}")
    print(f"  {'max(dist)':>4}  {max(distances):>14.2f}  {'sum(pen)':>10}  {fit:>14.2f}")

    print(f"\n  Fitness   : {fit:.4f}")
    print(f"  Kha thi   : {'CO (penalty=0)' if feasible else 'KHONG (co vi pham)'}")
    print(f"  Thoi gian : {elapsed:.2f}s")
    print(f"  Lich su   : {len(hist)} diem | bat dau={hist[0]:.1f} | cuoi={hist[-1]:.1f}")

    return AlgoResult(
        name=name,
        solution=sol,
        fitness=fit,
        distances=distances,
        penalties=penalties,
        elapsed=elapsed,
        history=hist,
        feasible=feasible,
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          BANG SO SANH                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _print_summary(problem: ProblemData, results: List[AlgoResult]) -> None:
    """In bang so sanh cuoi cung giua 3 thuat toan."""
    sep  = "=" * 62
    dash = "-" * 62
    best = min(results, key=lambda r: r.fitness)

    print(f"\n{sep}")
    print(f"  BANG TONG KET  —  {problem.name}")
    print(f"  N={problem.N} | M={problem.M} | K={problem.K} | "
          f"Q={problem.vehicle_capacities}")
    print(dash)
    print(f"  {'Thuat toan':<10}  {'Fitness':>12}  {'T(s)':>7}  "
          f"{'Kha thi':>8}  {'Giam so voi init':>16}")
    print(dash)

    for r in results:
        star = " *" if r is best else "  "
        feasible_str = "CO" if r.feasible else "KHONG"
        print(f"  {r.name:<10}{star} {r.fitness:>12.2f}  {r.elapsed:>7.2f}  "
              f"{feasible_str:>8}")

    print(dash)
    print(f"  TOT NHAT: {best.name}  —  fitness = {best.fitness:.4f}")
    print(f"{sep}\n")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          LUU JSON                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _save_json(result_dir: str, problem: ProblemData, r: AlgoResult) -> str:
    os.makedirs(result_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(result_dir, f"{problem.name}_{r.name}_{ts}.json")

    payload = {
        "instance":       problem.name,
        "algorithm":      r.name,
        "timestamp":      ts,
        "fitness":        r.fitness,
        "feasible":       r.feasible,
        "elapsed_s":      round(r.elapsed, 3),
        "decoded_routes": decode_routes(r.solution),
        "distances":      [round(d, 4) for d in r.distances],
        "penalties":      [round(p, 4) for p in r.penalties],
        "chromosome":     r.solution.chromosome,
        "history":        {"length": len(r.history),
                           "first10": r.history[:10],
                           "last10":  r.history[-10:]},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          XU LY MOT INSTANCE                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def run_instance(
    problem: ProblemData,
    seed: Optional[int],
    time_limit: float,
    result_dir: str,
) -> List[AlgoResult]:
    """Chay GA, Tabu, ALNS doc lap tren cung mot instance."""

    algorithms = ["GA", "Tabu", "ALNS", "OR-Tools"]
    results: List[AlgoResult] = []

    for idx, name in enumerate(algorithms, start=1):
        try:
            result = _run_one(
                name=name,
                problem=problem,
                seed=seed,
                time_limit=time_limit,
                index=idx,
                total=len(algorithms),
            )
        except Exception as exc:
            print(f"\n  [{idx}/{len(algorithms)}] {name} — LOI: {exc}")
            continue

        results.append(result)

        # Luu JSON
        path = _save_json(result_dir, problem, result)
        print(f"  Luu: {path}")

    _print_summary(problem, results)
    return results


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          CLI                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sarp",
        description="SARP Min-Max Optimizer — GA / Tabu / ALNS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--instance", "-i", help="Duong dan file instance (.in / .txt)")
    src.add_argument("--fold",     "-f", help="Thu muc chua nhieu instance")

    p.add_argument("--seed",        "-s", type=int,   default=None,
                   help="Random seed (mac dinh: khong co)")
    p.add_argument("--time-limit",  "-t", type=float, default=float("inf"),
                   help="Gioi han thoi gian (giay) cho moi thuat toan")
    p.add_argument("--results-dir", "-r", default="results",
                   help="Thu muc luu file JSON (mac dinh: results/)")
    p.add_argument("--verbose",     "-v", action="store_true",
                   help="Hien thi log chi tiet cua tung thuat toan")
    return p


def main() -> None:
    args = build_parser().parse_args()

    # Logging: chi hien thi WARNING tro len theo mac dinh (giau log noi bo)
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # Thu thap danh sach file
    if args.instance:
        files = [args.instance]
    else:
        files = list_instances_in_fold(args.fold)
        if not files:
            print(f"Khong tim thay file .in/.txt trong: {args.fold}", file=sys.stderr)
            sys.exit(1)

    print(f"\nSARP Min-Max Optimizer")
    print(f"Instances  : {len(files)}")
    print(f"Time limit : {args.time_limit}s / thuat toan")
    print(f"Seed       : {args.seed}")
    print(f"Results    : {args.results_dir}/")

    for fp in files:
        try:
            problem = read_instance(fp)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n[LOI] Khong doc duoc {fp}: {e}", file=sys.stderr)
            continue

        run_instance(
            problem=problem,
            seed=args.seed,
            time_limit=args.time_limit,
            result_dir=args.results_dir,
        )


if __name__ == "__main__":
    main()
