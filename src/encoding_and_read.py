"""
encoding_and_read.py — Data I/O, Request Dataclass & Unified 1D Solution Encoding
==================================================================================

This module is the foundation of the SARP optimizer. It defines:
  1. `Request`       — an immutable dataclass for each service request (passenger / parcel).
  2. `ProblemData`   — a container holding all parsed instance data.
  3. `Solution1D`    — the unified 1D-array solution representation shared by GA, Tabu & ALNS.
  4. `read_instance` — a parser that loads .txt instance files from any of the 3 data folds.

─────────────────────────────────────────────────────────────
UNIFIED 1D ENCODING RULES (shared by ALL three algorithms):
─────────────────────────────────────────────────────────────
  • chromosome length = N + 2M + (K − 1)
  • Passengers (N):  ONLY the pickup node ID appears; the drop-off is implicit
                     at node  pickup_id + N + M  (direct trip, no detour).
  • Parcels   (M):   BOTH pickup and drop-off node IDs are present.
  • Separators(K−1): The integer `0` (depot) acts as a divider, splitting the
                     flat list into K route segments — one per taxi.
─────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Tuple


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
        1-based index of this request in the instance file.
    request_type : RequestType
        PASSENGER or PARCEL.
    pickup_node : int
        Node ID for the pickup location.
    dropoff_node : int
        Node ID for the drop-off location.
    pickup_x : float
        X-coordinate of the pickup location.
    pickup_y : float
        Y-coordinate of the pickup location.
    dropoff_x : float
        X-coordinate of the drop-off location.
    dropoff_y : float
        Y-coordinate of the drop-off location.
    demand : int
        Capacity consumed (1 for passengers, varies for parcels).
    earliest_pickup : float
        Earliest time the request can be picked up.
    latest_pickup : float
        Latest time the request can be picked up (time window).
    earliest_dropoff : float
        Earliest time the request can be dropped off.
    latest_dropoff : float
        Latest time the request can be dropped off (time window).
    service_time_pickup : float
        Service duration at the pickup node.
    service_time_dropoff : float
        Service duration at the drop-off node.
    max_ride_time : float
        Maximum allowed ride time for this request (quality constraint).
    """
    request_id: int
    request_type: RequestType
    pickup_node: int
    dropoff_node: int
    pickup_x: float
    pickup_y: float
    dropoff_x: float
    dropoff_y: float
    demand: int = 1
    earliest_pickup: float = 0.0
    latest_pickup: float = float("inf")
    earliest_dropoff: float = 0.0
    latest_dropoff: float = float("inf")
    service_time_pickup: float = 0.0
    service_time_dropoff: float = 0.0
    max_ride_time: float = float("inf")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                       PROBLEM DATA CONTAINER                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

@dataclass
class ProblemData:
    """
    Aggregated instance data parsed from a .txt file.

    Attributes
    ----------
    name : str
        Instance name / filename stem.
    num_passengers : int  (N)
        Number of passenger requests.
    num_parcels : int  (M)
        Number of parcel requests.
    num_vehicles : int  (K)
        Number of available taxis.
    vehicle_capacity : int
        Maximum taxi capacity; kept for backward compatibility with older
        uniform-capacity instances.
    vehicle_capacities : List[int]
        Per-taxi capacities Q[0]..Q[K-1].
    depot_x : float
        X-coordinate of the depot (node 0).
    depot_y : float
        Y-coordinate of the depot (node 0).
    passengers : List[Request]
        Ordered list of passenger requests (length N).
    parcels : List[Request]
        Ordered list of parcel requests (length M).
    distance_matrix : List[List[float]]
        Pre-computed Euclidean distance matrix between all nodes.
    coords : List[Tuple[float, float]]
        (x, y) coordinates indexed by node ID.
    planning_horizon : float
        Maximum allowed planning horizon T.
    """
    name: str = ""
    num_passengers: int = 0
    num_parcels: int = 0
    num_vehicles: int = 0
    vehicle_capacity: int = 4
    vehicle_capacities: List[int] = field(default_factory=list)
    depot_x: float = 0.0
    depot_y: float = 0.0
    passengers: List[Request] = field(default_factory=list)
    parcels: List[Request] = field(default_factory=list)
    distance_matrix: List[List[float]] = field(default_factory=list)
    coords: List[Tuple[float, float]] = field(default_factory=list)
    planning_horizon: float = float("inf")

    # ----- derived helpers -----
    @property
    def N(self) -> int:
        """Alias: number of passenger requests."""
        return self.num_passengers

    @property
    def M(self) -> int:
        """Alias: number of parcel requests."""
        return self.num_parcels

    @property
    def K(self) -> int:
        """Alias: number of vehicles."""
        return self.num_vehicles

    @property
    def chromosome_length(self) -> int:
        """
        Required length of the unified 1D chromosome:
            N  (passenger pickups only)
          + 2M (parcel pickup + drop-off)
          + (K − 1) separators
        """
        return self.N + 2 * self.M + (self.K - 1)

    def all_requests(self) -> List[Request]:
        """Return passengers followed by parcels."""
        return self.passengers + self.parcels

    def dist(self, node_a: int, node_b: int) -> float:
        """Look up the pre-computed distance between two node IDs."""
        return self.distance_matrix[node_a][node_b]

    def capacity_for_vehicle(self, vehicle_index: int) -> int:
        """Return the capacity for a zero-based vehicle index."""
        if self.vehicle_capacities:
            return self.vehicle_capacities[vehicle_index]
        return self.vehicle_capacity


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                      SOLUTION 1D CLASS                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Solution1D:
    """
    Unified 1D-array solution representation.

    The `chromosome` is a flat list[int] of length  N + 2M + (K − 1).
      • Non-zero entries are node IDs (pickup / drop-off).
      • Zeros are depot separators that partition the list into K routes.

    Example (N=2 passengers, M=1 parcel, K=3 taxis):
        [3, 1, 0, 2, 5, 8, 0]
         ── route 1 ──  ── route 2 ──  ── route 3 (empty) ──
        Length = 2 + 2*1 + (3-1) = 6  ✓  (add trailing empty → 7 entries)

    Invariants maintained by ALL algorithms:
      1. Exactly (K − 1) zeros in the chromosome.
      2. No duplicated non-zero node IDs.
      3. Parcel pickup must precede its drop-off within the same route segment.
    """

    __slots__ = ("chromosome", "_problem")

    def __init__(self, chromosome: List[int], problem: ProblemData) -> None:
        """
        Parameters
        ----------
        chromosome : List[int]
            The 1D encoded solution array.
        problem : ProblemData
            Reference to the parsed instance data.
        """
        self.chromosome: List[int] = chromosome
        self._problem: ProblemData = problem

    # ── Construction helpers ────────────────────────────────────────────

    @classmethod
    def empty(cls, problem: ProblemData) -> "Solution1D":
        """Create a blank solution (K−1 separators only, no requests)."""
        return cls(chromosome=[0] * (problem.K - 1), problem=problem)

    # ── Accessors ───────────────────────────────────────────────────────

    @property
    def problem(self) -> ProblemData:
        return self._problem

    def get_routes(self) -> List[List[int]]:
        """
        Split the chromosome by `0` separators and return a list of K routes.
        Each route is a list of non-zero node IDs for one taxi.

        Returns
        -------
        List[List[int]]
            Exactly K sub-lists (some may be empty).
        """
        routes: List[List[int]] = []
        current_route: List[int] = []
        for gene in self.chromosome:
            if gene == 0:
                routes.append(current_route)
                current_route = []
            else:
                current_route.append(gene)
        # The last segment after the final separator (or entire list if K=1)
        routes.append(current_route)
        return routes

    # ── Validation ──────────────────────────────────────────────────────

    def validate(self) -> Tuple[bool, str]:
        """
        Perform basic structural validation of the chromosome.

        Returns
        -------
        (is_valid, message)
        """
        expected_len: int = self._problem.chromosome_length
        actual_len: int = len(self.chromosome)
        if actual_len != expected_len:
            return False, (
                f"Length mismatch: expected {expected_len}, got {actual_len}"
            )

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

        expected_nodes = set(range(1, self._problem.N + 1))
        expected_nodes.update(
            range(self._problem.N + 1, self._problem.N + self._problem.M + 1)
        )
        expected_nodes.update(
            range(
                2 * self._problem.N + self._problem.M + 1,
                2 * self._problem.N + 2 * self._problem.M + 1,
            )
        )
        observed_nodes = set(non_zero)
        if observed_nodes != expected_nodes:
            missing = sorted(expected_nodes - observed_nodes)
            extra = sorted(observed_nodes - expected_nodes)
            return False, (
                f"Node set mismatch; missing={missing[:10]}, extra={extra[:10]}"
            )

        return True, "OK"

    # ── Copy ────────────────────────────────────────────────────────────

    def copy(self) -> "Solution1D":
        """Return a deep copy of this solution."""
        return Solution1D(
            chromosome=list(self.chromosome),
            problem=self._problem,
        )

    # ── Dunder helpers ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Solution1D(len={len(self.chromosome)}, chromosome={self.chromosome})"

    def __len__(self) -> int:
        return len(self.chromosome)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        DISTANCE UTILITIES                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def _euclidean(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute the Euclidean distance between two 2D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _build_distance_matrix(
    coords: List[Tuple[float, float]],
) -> List[List[float]]:
    """
    Build a symmetric distance matrix from a list of (x, y) coordinates.

    Parameters
    ----------
    coords : list of (x, y) tuples
        Index 0 is the depot.

    Returns
    -------
    List[List[float]]
        dist[i][j] = Euclidean distance between node i and node j.
    """
    n: int = len(coords)
    matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d: float = _euclidean(*coords[i], *coords[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     INSTANCE FILE PARSER                                 ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def read_instance(filepath: str) -> ProblemData:
    """
    Parse a SARP instance from a plain-text (.txt) file.

    Supported file formats:
    ─────────────────────────────────────────────────────────────────────────
    Official Project 11 matrix format:
    Line 1:  N  M  K
    Line 2:  q[1] ... q[M]
    Line 3:  Q[1] ... Q[K]
    Lines 4 .. 4+2N+2M:
             distance matrix rows d[0] .. d[2N+2M]

    Internal fold format:
    ─────────────────────────────────────────────────────────────────────────
    Line 1:  K  N  M  Q  T
          or K  N  M  Q1 Q2 ... QK  T
             K = number of vehicles
             N = number of passengers
             M = number of parcels
             Q = uniform vehicle capacity, or Q1..QK per-taxi capacities
             T = planning horizon
    Line 2:  depot_x  depot_y   (depot coordinates, node 0)
    Lines 3 .. 2+N:   Passenger request rows
        request_id  pickup_x  pickup_y  dropoff_x  dropoff_y  demand
        earliest_pickup  latest_pickup  earliest_dropoff  latest_dropoff
        service_time_pickup  service_time_dropoff  max_ride_time
    Lines 3+N .. 2+N+M:  Parcel request rows  (same format)
    ─────────────────────────────────────────────────────────────────────────

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .txt instance file.

    Returns
    -------
    ProblemData
        Fully populated problem data (including the distance matrix).

    Raises
    ------
    FileNotFoundError
        If the filepath does not exist.
    ValueError
        If the file cannot be parsed correctly.

    TODO
    ----
    Adapt the column indices below to match the EXACT format of your data folds.
    The current implementation assumes the format documented above.
    """
    path: Path = Path(filepath).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Instance file not found: {path}")

    data = ProblemData(name=path.stem)

    with open(path, "r", encoding="utf-8") as fh:
        lines: List[str] = [
            line.strip() for line in fh.readlines() if line.strip()
        ]

    if not lines:
        raise ValueError(f"Instance file is empty: {path}")

    header: List[str] = lines[0].split()
    if len(header) == 3:
        return _read_official_matrix_instance(path, lines)

    # ── Line 1: header ──────────────────────────────────────────────────
    data.num_vehicles = int(header[0])       # K
    data.num_passengers = int(header[1])     # N
    data.num_parcels = int(header[2])        # M
    if len(header) == 5:
        capacity = int(header[3])
        data.vehicle_capacities = [capacity for _ in range(data.num_vehicles)]
        data.vehicle_capacity = capacity
        data.planning_horizon = float(header[4])
    elif len(header) == data.num_vehicles + 4:
        capacities = [int(value) for value in header[3 : 3 + data.num_vehicles]]
        data.vehicle_capacities = capacities
        data.vehicle_capacity = max(capacities)
        data.planning_horizon = float(header[3 + data.num_vehicles])
    else:
        raise ValueError(
            "header must be either 'K N M Q T' or 'K N M Q1 ... QK T'"
        )

    # ── Line 2: depot ────────────────────────────────────────────────────
    depot_parts: List[str] = lines[1].split()
    data.depot_x = float(depot_parts[0])
    data.depot_y = float(depot_parts[1])

    # Coordinates list: index 0 = depot
    coords: List[Tuple[float, float]] = [(data.depot_x, data.depot_y)]

    # ── Node ID counters ─────────────────────────────────────────────────
    # Passenger pickup nodes:  1 .. N
    # Parcel   pickup nodes:  N+1 .. N+M
    # Passenger dropoff nodes: N+M+1 .. 2N+M   (implicit in chromosome)
    # Parcel   dropoff nodes:  2N+M+1 .. 2N+2M (explicit in chromosome)
    node_id: int = 1

    # ── Parse passenger requests ─────────────────────────────────────────
    for i in range(data.num_passengers):
        row: List[str] = lines[2 + i].split()
        # TODO: adjust column indices to match your actual data format
        pickup_x: float = float(row[1])
        pickup_y: float = float(row[2])
        dropoff_x: float = float(row[3])
        dropoff_y: float = float(row[4])
        demand: int = int(row[5]) if len(row) > 5 else 1
        earliest_pickup: float = float(row[6]) if len(row) > 6 else 0.0
        latest_pickup: float = float(row[7]) if len(row) > 7 else data.planning_horizon
        earliest_dropoff: float = float(row[8]) if len(row) > 8 else 0.0
        latest_dropoff: float = float(row[9]) if len(row) > 9 else data.planning_horizon
        svc_pickup: float = float(row[10]) if len(row) > 10 else 0.0
        svc_dropoff: float = float(row[11]) if len(row) > 11 else 0.0
        max_ride: float = float(row[12]) if len(row) > 12 else data.planning_horizon

        pickup_node_id: int = node_id
        dropoff_node_id: int = node_id + data.num_passengers + data.num_parcels

        req = Request(
            request_id=i + 1,
            request_type=RequestType.PASSENGER,
            pickup_node=pickup_node_id,
            dropoff_node=dropoff_node_id,
            pickup_x=pickup_x,
            pickup_y=pickup_y,
            dropoff_x=dropoff_x,
            dropoff_y=dropoff_y,
            demand=demand,
            earliest_pickup=earliest_pickup,
            latest_pickup=latest_pickup,
            earliest_dropoff=earliest_dropoff,
            latest_dropoff=latest_dropoff,
            service_time_pickup=svc_pickup,
            service_time_dropoff=svc_dropoff,
            max_ride_time=max_ride,
        )
        data.passengers.append(req)

        # Register both pickup and dropoff coords (in node-ID order)
        coords.append((pickup_x, pickup_y))
        node_id += 1

    # ── Parse parcel requests ────────────────────────────────────────────
    for i in range(data.num_parcels):
        row = lines[2 + data.num_passengers + i].split()
        pickup_x = float(row[1])
        pickup_y = float(row[2])
        dropoff_x = float(row[3])
        dropoff_y = float(row[4])
        demand = int(row[5]) if len(row) > 5 else 1
        earliest_pickup = float(row[6]) if len(row) > 6 else 0.0
        latest_pickup = float(row[7]) if len(row) > 7 else data.planning_horizon
        earliest_dropoff = float(row[8]) if len(row) > 8 else 0.0
        latest_dropoff = float(row[9]) if len(row) > 9 else data.planning_horizon
        svc_pickup = float(row[10]) if len(row) > 10 else 0.0
        svc_dropoff = float(row[11]) if len(row) > 11 else 0.0
        max_ride = float(row[12]) if len(row) > 12 else data.planning_horizon

        pickup_node_id = node_id
        dropoff_node_id = node_id + data.num_passengers + data.num_parcels

        req = Request(
            request_id=data.num_passengers + i + 1,
            request_type=RequestType.PARCEL,
            pickup_node=pickup_node_id,
            dropoff_node=dropoff_node_id,
            pickup_x=pickup_x,
            pickup_y=pickup_y,
            dropoff_x=dropoff_x,
            dropoff_y=dropoff_y,
            demand=demand,
            earliest_pickup=earliest_pickup,
            latest_pickup=latest_pickup,
            earliest_dropoff=earliest_dropoff,
            latest_dropoff=latest_dropoff,
            service_time_pickup=svc_pickup,
            service_time_dropoff=svc_dropoff,
            max_ride_time=max_ride,
        )
        data.parcels.append(req)
        coords.append((pickup_x, pickup_y))
        node_id += 1

    # ── Append drop-off coordinates (passenger dropoffs then parcel) ─────
    for req in data.passengers:
        coords.append((req.dropoff_x, req.dropoff_y))
    for req in data.parcels:
        coords.append((req.dropoff_x, req.dropoff_y))

    # ── Build distance matrix ────────────────────────────────────────────
    data.coords = coords
    data.distance_matrix = _build_distance_matrix(coords)

    return data


def _read_official_matrix_instance(path: Path, lines: List[str]) -> ProblemData:
    """Parse the Project 11 official matrix format: N M K, q, Q, d."""
    data = ProblemData(name=path.stem)
    data.num_passengers, data.num_parcels, data.num_vehicles = map(
        int,
        lines[0].split(),
    )
    q = [int(value) for value in lines[1].split()]
    capacities = [int(value) for value in lines[2].split()]
    if len(q) != data.num_parcels:
        raise ValueError(f"expected {data.num_parcels} parcel quantities, got {len(q)}")
    if len(capacities) != data.num_vehicles:
        raise ValueError(f"expected {data.num_vehicles} taxi capacities, got {len(capacities)}")

    node_count = 2 * data.N + 2 * data.M + 1
    matrix_lines = lines[3:]
    if len(matrix_lines) != node_count:
        raise ValueError(f"expected {node_count} distance rows, got {len(matrix_lines)}")

    matrix: List[List[float]] = []
    for row_index, line in enumerate(matrix_lines):
        row = [float(value) for value in line.split()]
        if len(row) != node_count:
            raise ValueError(
                f"distance row {row_index} expected {node_count} values, got {len(row)}"
            )
        matrix.append(row)

    data.vehicle_capacities = capacities
    data.vehicle_capacity = max(capacities)
    data.planning_horizon = float("inf")
    data.depot_x = 0.0
    data.depot_y = 0.0
    data.coords = [(0.0, 0.0) for _ in range(node_count)]
    data.distance_matrix = matrix

    for passenger_id in range(1, data.N + 1):
        data.passengers.append(
            Request(
                request_id=passenger_id,
                request_type=RequestType.PASSENGER,
                pickup_node=passenger_id,
                dropoff_node=passenger_id + data.N + data.M,
                pickup_x=0.0,
                pickup_y=0.0,
                dropoff_x=0.0,
                dropoff_y=0.0,
                demand=0,
            )
        )

    for parcel_id in range(1, data.M + 1):
        data.parcels.append(
            Request(
                request_id=data.N + parcel_id,
                request_type=RequestType.PARCEL,
                pickup_node=data.N + parcel_id,
                dropoff_node=2 * data.N + data.M + parcel_id,
                pickup_x=0.0,
                pickup_y=0.0,
                dropoff_x=0.0,
                dropoff_y=0.0,
                demand=q[parcel_id - 1],
            )
        )

    return data


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     FOLD / BATCH LOADING HELPERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def list_instances_in_fold(fold_dir: str) -> List[str]:
    """
    Return sorted list of .txt/.in instance file paths inside a fold directory.

    Parameters
    ----------
    fold_dir : str
        Path to one of the 3 data folds (e.g. ``data/fold1``).

    Returns
    -------
    List[str]
        Absolute paths to each instance file.
    """
    fold_path: Path = Path(fold_dir).resolve()
    if not fold_path.is_dir():
        raise FileNotFoundError(f"Fold directory not found: {fold_path}")
    paths = list(fold_path.glob("*.txt")) + list(fold_path.glob("*.in"))
    return sorted(str(p) for p in paths)


def load_fold(fold_dir: str) -> List[ProblemData]:
    """
    Convenience: parse every .txt/.in instance in *fold_dir* and return them.

    Parameters
    ----------
    fold_dir : str
        Path to a data fold directory.

    Returns
    -------
    List[ProblemData]
        One ProblemData per instance file, sorted by filename.
    """
    return [read_instance(fp) for fp in list_instances_in_fold(fold_dir)]
