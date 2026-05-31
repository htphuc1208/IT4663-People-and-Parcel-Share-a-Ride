from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))


@dataclass(frozen=True)
class DarpNode:
    node_id: int
    x: float
    y: float
    service_time: float
    demand: int
    earliest: float
    latest: float


@dataclass(frozen=True)
class DarpInstance:
    K: int
    R: int
    raw_capacity: int
    max_ride_time: float
    planning_horizon: float
    nodes: dict[int, DarpNode]
    notes: list[str]


def _parse_int(value: str) -> int:
    return int(float(value))


def parse_capacity_csv(text: str) -> list[int]:
    try:
        capacities = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--taxi-capacities must be a comma-separated integer list") from exc
    if not capacities:
        raise ValueError("--taxi-capacities must not be empty")
    return capacities


def _infer_request_count(header: list[str], nodes: dict[int, DarpNode]) -> int:
    def has_required_nodes(request_count: int) -> bool:
        required = {0}
        required.update(range(1, 2 * request_count + 1))
        return required.issubset(nodes)

    candidates: list[int] = []
    if len(header) > 1:
        declared_nodes = _parse_int(header[1])
        if declared_nodes > 0 and declared_nodes % 2 == 0:
            candidates.append(declared_nodes // 2)
    if len(header) > 2:
        candidates.append(_parse_int(header[2]))
    max_node_id = max(nodes) if nodes else 0
    if max_node_id > 0:
        candidates.append(max_node_id // 2 if max_node_id % 2 == 0 else (max_node_id - 1) // 2)

    for candidate in candidates:
        if candidate > 0 and has_required_nodes(candidate):
            return candidate
    raise ValueError("could not infer DARP request count from header and node ids")


def read_cordeau_darp(path: str | Path) -> DarpInstance:
    lines = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError("empty DARP file")

    header = lines[0].split()
    if len(header) < 5:
        raise ValueError("DARP header must contain K, node count, horizon, capacity, max ride time")

    K = _parse_int(header[0])
    raw_capacity = _parse_int(header[3])
    max_ride_time = float(header[4])
    nodes: dict[int, DarpNode] = {}

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 7:
            raise ValueError(f"invalid DARP node row: {line!r}")
        node_id = _parse_int(parts[0])
        nodes[node_id] = DarpNode(
            node_id=node_id,
            x=float(parts[1]),
            y=float(parts[2]),
            service_time=float(parts[3]),
            demand=_parse_int(parts[4]),
            earliest=float(parts[5]),
            latest=float(parts[6]),
        )

    R = _infer_request_count(header, nodes)
    notes: list[str] = []
    terminal_id = 2 * R + 1
    terminal = nodes.pop(terminal_id, None)
    if terminal is not None:
        depot = nodes.get(0)
        if depot and depot.x == terminal.x and depot.y == terminal.y:
            notes.append(f"ignored terminal depot node {terminal_id}; coordinates match node 0")
        else:
            notes.append(f"ignored terminal depot node {terminal_id}; coordinates differ from node 0")

    required_ids = {0}
    required_ids.update(range(1, 2 * R + 1))
    missing = sorted(required_ids - nodes.keys())
    if missing:
        raise ValueError(f"DARP file is missing required node ids: {missing[:10]}")

    planning_horizon = max(node.latest for node in nodes.values())
    return DarpInstance(
        K=K,
        R=R,
        raw_capacity=raw_capacity,
        max_ride_time=max_ride_time,
        planning_horizon=planning_horizon,
        nodes=nodes,
        notes=notes,
    )


def build_taxi_capacities(
    *,
    K: int,
    raw_capacity: int,
    max_quantity: int,
    rng: random.Random,
    capacity_mode: str,
    taxi_capacities: list[int] | None = None,
    capacity_min: int | None = None,
    capacity_max: int | None = None,
) -> tuple[list[int], list[str]]:
    if taxi_capacities is not None:
        if len(taxi_capacities) != K:
            raise ValueError(f"expected {K} explicit taxi capacities, got {len(taxi_capacities)}")
        if max(taxi_capacities) < max_quantity:
            raise ValueError("at least one taxi must fit the largest parcel")
        return taxi_capacities[:], ["using explicit per-taxi capacities"]

    cap_max = capacity_max if capacity_max is not None else raw_capacity
    if cap_max < max_quantity:
        cap_max = max_quantity
    if capacity_mode == "uniform":
        return [cap_max for _ in range(K)], []
    if capacity_mode != "heterogeneous":
        raise ValueError(f"unknown capacity mode {capacity_mode!r}")

    cap_min = capacity_min if capacity_min is not None else max(1, round(cap_max * 0.5))
    if cap_min <= 0 or cap_min > cap_max:
        raise ValueError("capacity min must be positive and <= capacity max")
    if K == 1:
        capacities = [cap_max]
    elif cap_min == cap_max:
        capacities = [cap_max for _ in range(K)]
    else:
        capacities = [
            int(round(cap_min + i * (cap_max - cap_min) / (K - 1)))
            for i in range(K)
        ]
        capacities[0] = cap_min
        capacities[-1] = cap_max
        rng.shuffle(capacities)
    return capacities, [f"generated heterogeneous taxi capacities in [{cap_min}, {cap_max}]"]


def _request_row(
    request_id: int,
    pickup: DarpNode,
    dropoff: DarpNode,
    *,
    demand: int,
    max_ride_time: float,
) -> str:
    values = [
        request_id,
        pickup.x,
        pickup.y,
        dropoff.x,
        dropoff.y,
        demand,
        pickup.earliest,
        pickup.latest,
        dropoff.earliest,
        dropoff.latest,
        pickup.service_time,
        dropoff.service_time,
        max_ride_time,
    ]
    return " ".join(str(value) for value in values)


def convert_to_sarp_text(
    raw: DarpInstance,
    *,
    seed: int = 42,
    capacity_mode: str = "uniform",
    taxi_capacities: list[int] | None = None,
    capacity_min: int | None = None,
    capacity_max: int | None = None,
) -> tuple[str, dict[str, Any]]:
    rng = random.Random(seed)
    request_ids = list(range(1, raw.R + 1))
    M = raw.R // 3
    parcel_old_ids = sorted(rng.sample(request_ids, M)) if M else []
    parcel_old_set = set(parcel_old_ids)
    passenger_old_ids = [request_id for request_id in request_ids if request_id not in parcel_old_set]
    N = len(passenger_old_ids)

    parcel_quantities = [max(1, abs(raw.nodes[old_id].demand)) for old_id in parcel_old_ids]
    capacities, capacity_notes = build_taxi_capacities(
        K=raw.K,
        raw_capacity=raw.raw_capacity,
        max_quantity=max(parcel_quantities, default=1),
        rng=rng,
        capacity_mode=capacity_mode,
        taxi_capacities=taxi_capacities,
        capacity_min=capacity_min,
        capacity_max=capacity_max,
    )

    depot = raw.nodes[0]
    lines = [
        " ".join(str(value) for value in [raw.K, N, M, *capacities, raw.planning_horizon]),
        f"{depot.x} {depot.y}",
    ]

    passenger_mapping: dict[int, dict[str, int]] = {}
    for new_id, old_id in enumerate(passenger_old_ids, start=1):
        pickup = raw.nodes[old_id]
        dropoff = raw.nodes[old_id + raw.R]
        lines.append(
            _request_row(
                new_id,
                pickup,
                dropoff,
                demand=1,
                max_ride_time=raw.max_ride_time,
            )
        )
        passenger_mapping[new_id] = {"old_request": old_id}

    parcel_mapping: dict[int, dict[str, int]] = {}
    for new_id, old_id in enumerate(parcel_old_ids, start=1):
        pickup = raw.nodes[old_id]
        dropoff = raw.nodes[old_id + raw.R]
        quantity = max(1, abs(pickup.demand))
        lines.append(
            _request_row(
                N + new_id,
                pickup,
                dropoff,
                demand=quantity,
                max_ride_time=raw.max_ride_time,
            )
        )
        parcel_mapping[new_id] = {"old_request": old_id, "quantity": quantity}

    metadata: dict[str, Any] = {
        "seed": seed,
        "raw_request_count": raw.R,
        "N": N,
        "M": M,
        "K": raw.K,
        "selected_parcels_original": parcel_old_ids,
        "passengers": passenger_mapping,
        "parcels": parcel_mapping,
        "raw_capacity": raw.raw_capacity,
        "capacity_mode": "explicit" if taxi_capacities is not None else capacity_mode,
        "taxi_capacities": capacities,
        "notes": [*raw.notes, *capacity_notes],
    }
    return "\n".join(lines) + "\n", metadata


def write_converted_instance(
    raw_path: Path,
    out_path: Path,
    *,
    seed: int,
    capacity_mode: str,
    taxi_capacities: list[int] | None,
    capacity_min: int | None,
    capacity_max: int | None,
) -> Path:
    raw = read_cordeau_darp(raw_path)
    text, metadata = convert_to_sarp_text(
        raw,
        seed=seed,
        capacity_mode=capacity_mode,
        taxi_capacities=taxi_capacities,
        capacity_min=capacity_min,
        capacity_max=capacity_max,
    )
    metadata["raw_file"] = str(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return meta_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Cordeau-Laporte DARP data to SARP fold format")
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity-mode", choices=["uniform", "heterogeneous"], default="uniform")
    parser.add_argument("--taxi-capacities", default=None)
    parser.add_argument("--capacity-min", type=int, default=None)
    parser.add_argument("--capacity-max", type=int, default=None)
    args = parser.parse_args()

    taxi_capacities = parse_capacity_csv(args.taxi_capacities) if args.taxi_capacities else None
    meta_path = write_converted_instance(
        args.raw,
        args.out,
        seed=args.seed,
        capacity_mode=args.capacity_mode,
        taxi_capacities=taxi_capacities,
        capacity_min=args.capacity_min,
        capacity_max=args.capacity_max,
    )
    print(f"Generated: {args.out}")
    print(f"Metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
