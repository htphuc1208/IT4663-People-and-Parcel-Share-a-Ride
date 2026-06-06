"""
encoding_and_read.py — Data I/O, Request Dataclass & Unified 1D Solution Encoding
==================================================================================

This module is the foundation of the SARP optimizer. It defines:
  1. `Request`       — an immutable dataclass for each service request.
  2. `ProblemData`   — a container holding all parsed instance data.
  3. `Solution1D`    — the unified 1D-array solution representation shared by GA, Tabu & ALNS.
  4. `read_instance` — a parser that loads .txt instance files.

─────────────────────────────────────────────────────────────
INPUT FILE FORMAT
─────────────────────────────────────────────────────────────
  Line 1:        N  M  K
                   N = number of passenger requests  (1 ≤ N ≤ 500)
                   M = number of parcel requests     (1 ≤ M ≤ 500)
                   K = number of taxis               (1 ≤ K ≤ 100)
  Line 2:        q[1]  q[2]  ...  q[M]
                   Parcel demands  (1 ≤ q[i] ≤ 100)
  Line 3:        Q[1]  Q[2]  ...  Q[K]
                   Per-vehicle parcel capacities  (1 ≤ Q[k] ≤ 200)
  Line i+3
  (i=0..2N+2M):  Row i of the (2N+2M+1)×(2N+2M+1) distance matrix

─────────────────────────────────────────────────────────────
NODE NUMBERING
─────────────────────────────────────────────────────────────
  0              — depot
  1  ..  N       — passenger pickup nodes
  N+1 .. N+M     — parcel pickup nodes
  N+M+1 .. 2N+M  — passenger dropoff nodes
  2N+M+1 .. 2N+2M — parcel dropoff nodes

─────────────────────────────────────────────────────────────
UNIFIED 1D ENCODING RULES (shared by GA, Tabu & ALNS)
─────────────────────────────────────────────────────────────
  chromosome length = N + 2M + (K − 1)

  Passengers (N):   ONLY the pickup node ID appears in the chromosome.
                    The drop-off (pickup_id + N + M) is implicit — the decoder
                    inserts it immediately after the pickup (direct trip,
                    no interruption).
  Parcels   (M):    BOTH pickup and drop-off node IDs are present.
  Separators(K−1):  The integer 0 (depot) partitions the chromosome into
                    K route segments, one per taxi.
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                          ENUMERATIONS                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class RequestType(Enum):
    """Distinguishes passenger requests from parcel requests."""
    PASSENGER = auto()
    PARCEL = auto()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      REQUEST DATACLASS                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class Request:
    """
    Immutable record for a single service request.

    Attributes
    ----------
    request_id : int
        1-based index within its type (1..N for passengers, 1..M for parcels).
    request_type : RequestType
        PASSENGER or PARCEL.
    pickup_node : int
        Node ID for the pickup location.
    dropoff_node : int
        Node ID for the drop-off location.
    demand : int
        Capacity consumed by this request.
        Always 1 for passengers (passengers don't use parcel capacity Q[k]).
        Equals q[i] for parcel i.
    """
    request_id: int
    request_type: RequestType
    pickup_node: int
    dropoff_node: int
    demand: int = 1


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       PROBLEM DATA CONTAINER                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ProblemData:
    """
    Aggregated instance data parsed from a .txt file.

    Attributes
    ----------
    name : str
        Instance name (file stem).
    num_passengers : int  (N)
        Number of passenger requests.
    num_parcels : int  (M)
        Number of parcel requests.
    num_vehicles : int  (K)
        Number of available taxis.
    vehicle_capacities : List[int]
        Per-vehicle parcel capacity Q[k], length K.
        Each taxi k can carry at most Q[k] units of parcel demand simultaneously.
        Passengers do not consume this capacity.
    passengers : List[Request]
        Passenger requests in order (request_id 1..N).
    parcels : List[Request]
        Parcel requests in order (request_id 1..M).
    distance_matrix : List[List[float]]
        d[i][j] = travel distance from node i to node j,
        size (2N+2M+1) × (2N+2M+1).
    """
    name: str = ""
    num_passengers: int = 0
    num_parcels: int = 0
    num_vehicles: int = 0
    vehicle_capacities: List[int] = field(default_factory=list)
    passengers: List[Request] = field(default_factory=list)
    parcels: List[Request] = field(default_factory=list)
    distance_matrix: List[List[float]] = field(default_factory=list)

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def N(self) -> int:
        return self.num_passengers

    @property
    def M(self) -> int:
        return self.num_parcels

    @property
    def K(self) -> int:
        return self.num_vehicles

    @property
    def chromosome_length(self) -> int:
        """
        Required 1D chromosome length:
            N  (passenger pickups only)
          + 2M (parcel pickup + drop-off)
          + (K − 1) depot separators
        """
        return self.N + 2 * self.M + (self.K - 1)

    def all_requests(self) -> List[Request]:
        """Return passengers followed by parcels."""
        return self.passengers + self.parcels

    def dist(self, node_a: int, node_b: int) -> float:
        """Look up the travel distance between two node IDs."""
        return self.distance_matrix[node_a][node_b]

    # ── Cached lookup tables (built once, reused by fitness/ALNS) ─────────
    # These replace per-evaluate dict construction.  `cached_property`
    # memoises the result on first access; safe because instance data is
    # never mutated after parsing.

    @cached_property
    def demand_by_pickup(self) -> Dict[int, int]:
        """parcel pickup node → parcel demand q[i]."""
        return {r.pickup_node: r.demand for r in self.parcels}

    @cached_property
    def pickup_by_dropoff(self) -> Dict[int, int]:
        """parcel dropoff node → its pickup node."""
        return {r.dropoff_node: r.pickup_node for r in self.parcels}

    @cached_property
    def request_by_node(self) -> Dict[int, Request]:
        """Any pickup OR dropoff node → its owning Request (passenger/parcel)."""
        mapping: Dict[int, Request] = {}
        for req in self.all_requests():
            mapping[req.pickup_node] = req
            mapping[req.dropoff_node] = req
        return mapping


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      SOLUTION 1D CLASS                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Solution1D:
    """
    Unified 1D-array solution representation.

    The `chromosome` is a flat list[int] of length  N + 2M + (K − 1).
      • Non-zero entries are node IDs (passenger pickup / parcel pickup or drop-off).
      • Zeros are depot separators that partition the list into K routes.

    Example (N=2, M=1, K=2):
        chromosome = [1, 3, 6, 0, 2]
        Passenger 1: pickup=1, dropoff=auto(1+2+1=4)
        Passenger 2: pickup=2, dropoff=auto(2+2+1=5)
        Parcel 1:    pickup=3, dropoff=6 (both explicit)
        Route 0 after decode: [0, 1, 4, 3, 6, 0]
        Route 1 after decode: [0, 2, 5, 0]

    Invariants maintained by ALL algorithms:
      1. Exactly (K − 1) zeros in the chromosome.
      2. No duplicated non-zero node IDs.
      3. Parcel pickup must precede its drop-off within the same route segment
         (enforced via penalty; not structurally guaranteed by the encoding).
    """

    __slots__ = ("chromosome", "_problem")

    def __init__(self, chromosome: List[int], problem: ProblemData) -> None:
        self.chromosome: List[int] = chromosome
        self._problem: ProblemData = problem

    @classmethod
    def empty(cls, problem: ProblemData) -> "Solution1D":
        """Create a blank solution (K−1 separators only, no requests)."""
        return cls(chromosome=[0] * (problem.K - 1), problem=problem)

    @property
    def problem(self) -> ProblemData:
        return self._problem

    def get_routes(self) -> List[List[int]]:
        """
        Split the chromosome by `0` separators → K route segments.
        Each route is a list of non-zero node IDs for one taxi.
        """
        routes: List[List[int]] = []
        current_route: List[int] = []
        for gene in self.chromosome:
            if gene == 0:
                routes.append(current_route)
                current_route = []
            else:
                current_route.append(gene)
        routes.append(current_route)
        return routes

    def validate(self) -> Tuple[bool, str]:
        """Basic structural validation of the chromosome."""
        expected_len: int = self._problem.chromosome_length
        actual_len: int = len(self.chromosome)
        if actual_len != expected_len:
            return False, f"Length mismatch: expected {expected_len}, got {actual_len}"

        num_zeros: int = self.chromosome.count(0)
        expected_zeros: int = self._problem.K - 1
        if num_zeros != expected_zeros:
            return False, (
                f"Separator count mismatch: expected {expected_zeros} zeros, "
                f"got {num_zeros}"
            )

        non_zero: List[int] = [g for g in self.chromosome if g != 0]
        if len(non_zero) != len(set(non_zero)):
            return False, "Duplicate non-zero node IDs detected"

        return True, "OK"

    def copy(self) -> "Solution1D":
        return Solution1D(chromosome=list(self.chromosome), problem=self._problem)

    def __repr__(self) -> str:
        return f"Solution1D(len={len(self.chromosome)}, chromosome={self.chromosome})"

    def __len__(self) -> int:
        return len(self.chromosome)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     INSTANCE FILE PARSER                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def read_instance(filepath: str) -> ProblemData:
    """
    Parse a SARP instance from a plain-text (.txt) file.

    Expected format
    ───────────────
    Line 1:          N M K
    Line 2:          q[1] q[2] ... q[M]          (parcel demands)
    Line 3:          Q[1] Q[2] ... Q[K]          (vehicle capacities)
    Line i+3
    (i=0..2N+2M):    row i of the distance matrix

    Node numbering built by this parser:
      Passenger i (i=1..N) : pickup = i,     dropoff = i + N + M
      Parcel i   (i=1..M) : pickup = N + i,  dropoff = 2*N + M + i

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .txt instance file.

    Returns
    -------
    ProblemData

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be parsed correctly.
    """
    path: Path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        lines: List[str] = [ln.strip() for ln in fh if ln.strip()]

    if len(lines) < 3:
        raise ValueError(f"Instance file too short (got {len(lines)} non-empty lines)")

    # ── Line 1: N M K ────────────────────────────────────────────────────
    parts = lines[0].split()
    if len(parts) < 3:
        raise ValueError(f"Line 1 must contain N M K, got: '{lines[0]}'")
    N, M, K = int(parts[0]), int(parts[1]), int(parts[2])

    # ── Line 2: parcel demands q[1..M] ───────────────────────────────────
    parcel_demands: List[int] = list(map(int, lines[1].split()))
    if len(parcel_demands) != M:
        raise ValueError(
            f"Expected {M} parcel demands on line 2, got {len(parcel_demands)}"
        )

    # ── Line 3: vehicle capacities Q[1..K] ───────────────────────────────
    vehicle_caps: List[int] = list(map(int, lines[2].split()))
    if len(vehicle_caps) != K:
        raise ValueError(
            f"Expected {K} vehicle capacities on line 3, got {len(vehicle_caps)}"
        )

    # ── Lines 4..: distance matrix (2N+2M+1) × (2N+2M+1) ────────────────
    num_nodes: int = 2 * N + 2 * M + 1
    if len(lines) < 3 + num_nodes:
        raise ValueError(
            f"Expected at least {3 + num_nodes} non-empty lines "
            f"(3 header + {num_nodes} matrix rows), got {len(lines)}"
        )

    dist_matrix: List[List[float]] = []
    for i in range(num_nodes):
        row: List[float] = list(map(float, lines[3 + i].split()))
        if len(row) != num_nodes:
            raise ValueError(
                f"Distance matrix row {i} has {len(row)} values, "
                f"expected {num_nodes}"
            )
        dist_matrix.append(row)

    # ── Build passenger requests ──────────────────────────────────────────
    # Passenger i (i=1..N): pickup = i, dropoff = i + N + M
    passengers: List[Request] = [
        Request(
            request_id=i,
            request_type=RequestType.PASSENGER,
            pickup_node=i,
            dropoff_node=i + N + M,
            demand=1,
        )
        for i in range(1, N + 1)
    ]

    # ── Build parcel requests ─────────────────────────────────────────────
    # Parcel i (i=1..M): pickup = N+i, dropoff = 2*N+M+i
    parcels: List[Request] = [
        Request(
            request_id=i,
            request_type=RequestType.PARCEL,
            pickup_node=N + i,
            dropoff_node=2 * N + M + i,
            demand=parcel_demands[i - 1],
        )
        for i in range(1, M + 1)
    ]

    return ProblemData(
        name=path.stem,
        num_passengers=N,
        num_parcels=M,
        num_vehicles=K,
        vehicle_capacities=vehicle_caps,
        passengers=passengers,
        parcels=parcels,
        distance_matrix=dist_matrix,
    )


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FOLD / BATCH LOADING HELPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def list_instances_in_fold(fold_dir: str) -> List[str]:
    """Return sorted list of .in and .txt instance file paths inside a fold directory."""
    fold_path: Path = Path(fold_dir).resolve()
    if not fold_path.is_dir():
        raise FileNotFoundError(f"Fold directory not found: {fold_path}")
    files = list(fold_path.glob("*.in")) + list(fold_path.glob("*.txt"))
    return sorted(str(p) for p in files)


def load_fold(fold_dir: str) -> List[ProblemData]:
    """Parse every .txt instance in fold_dir and return them."""
    return [read_instance(fp) for fp in list_instances_in_fold(fold_dir)]
