from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


Profile = Literal["random", "clustered", "line", "asymmetric"]
CapacityProfile = Literal["loose", "balanced", "tight", "heterogeneous"]


@dataclass(frozen=True)
class OfficialInstance:
    N: int
    M: int
    K: int
    q: list[int]
    Q: list[int]
    d: list[list[int]]
    coords: list[tuple[int, int]]
    profile: str
    capacity_profile: str
    seed: int


SAMPLE_INPUT = """\
3 3 2
8 4 5
16 16
0 8 7 9 6 5 11 6 11 12 12 12 13
8 0 4 1 2 8 5 13 19 12 4 8 9
7 4 0 3 3 8 4 12 15 8 5 6 7
9 1 3 0 3 9 4 14 19 11 3 7 8
6 2 3 3 0 6 6 11 17 11 6 9 10
5 8 8 9 6 0 12 5 16 15 12 15 15
11 5 4 4 6 12 0 16 18 7 4 3 4
6 13 12 14 11 5 16 0 15 18 17 18 19
11 19 15 19 17 16 18 15 0 13 21 17 17
12 12 8 11 11 15 7 18 13 0 11 5 4
12 4 5 3 6 12 4 17 21 11 0 7 8
12 8 6 7 9 15 3 18 17 5 7 0 1
13 9 7 8 10 15 4 19 17 4 8 1 0
"""


def _euclidean(a: tuple[int, int], b: tuple[int, int]) -> int:
    return int(round(math.hypot(a[0] - b[0], a[1] - b[1])))


def _random_coords(rng: random.Random, count: int, coord_max: int) -> list[tuple[int, int]]:
    return [(rng.randint(0, coord_max), rng.randint(0, coord_max)) for _ in range(count)]


def _clustered_coords(rng: random.Random, count: int, coord_max: int) -> list[tuple[int, int]]:
    center_count = max(3, min(12, count // 40))
    centers = _random_coords(rng, center_count, coord_max)
    radius = max(8, coord_max // 20)
    coords: list[tuple[int, int]] = []
    for _ in range(count):
        cx, cy = rng.choice(centers)
        x = min(coord_max, max(0, int(round(rng.gauss(cx, radius)))))
        y = min(coord_max, max(0, int(round(rng.gauss(cy, radius)))))
        coords.append((x, y))
    return coords


def _line_coords(rng: random.Random, count: int, coord_max: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for i in range(count):
        x = int(round(i * coord_max / max(1, count - 1)))
        y = min(coord_max, max(0, int(round(coord_max / 2 + rng.gauss(0, coord_max / 30)))))
        coords.append((x, y))
    return coords


def _coords_for_profile(
    rng: random.Random,
    count: int,
    coord_max: int,
    profile: Profile,
) -> list[tuple[int, int]]:
    if profile == "clustered":
        return _clustered_coords(rng, count, coord_max)
    if profile == "line":
        return _line_coords(rng, count, coord_max)
    return _random_coords(rng, count, coord_max)


def _distance_matrix(
    rng: random.Random,
    coords: list[tuple[int, int]],
    *,
    asymmetric: bool,
) -> list[list[int]]:
    n = len(coords)
    d = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            value = max(1, _euclidean(coords[i], coords[j]))
            if asymmetric:
                value += rng.randint(0, max(1, value // 10))
            d[i][j] = value
    return d


def _quantities(
    rng: random.Random,
    M: int,
    *,
    capacity_profile: CapacityProfile,
) -> list[int]:
    if capacity_profile == "tight":
        return [rng.randint(50, 100) for _ in range(M)]
    if capacity_profile == "loose":
        return [rng.randint(1, 30) for _ in range(M)]
    return [rng.randint(1, 100) for _ in range(M)]


def _capacities(
    rng: random.Random,
    K: int,
    q: list[int],
    *,
    capacity_profile: CapacityProfile,
) -> list[int]:
    q_max = max(q, default=1)
    if capacity_profile == "loose":
        low = max(q_max, 120)
        return [rng.randint(low, 200) for _ in range(K)]
    if capacity_profile == "tight":
        return [rng.randint(max(1, q_max - 15), min(200, q_max + 10)) for _ in range(K)]
    if capacity_profile == "heterogeneous":
        values = [
            int(round(max(1, q_max // 2) + i * (200 - max(1, q_max // 2)) / max(1, K - 1)))
            for i in range(K)
        ]
        rng.shuffle(values)
        return values
    return [rng.randint(q_max, min(200, max(q_max, 140))) for _ in range(K)]


def generate_instance(
    *,
    N: int,
    M: int,
    K: int,
    seed: int,
    profile: Profile = "random",
    capacity_profile: CapacityProfile = "balanced",
    coord_max: int = 10_000,
) -> OfficialInstance:
    if not (1 <= N <= 500 and 1 <= M <= 500 and 1 <= K <= 100):
        raise ValueError("expected 1 <= N,M <= 500 and 1 <= K <= 100")

    rng = random.Random(seed)
    node_count = 2 * N + 2 * M + 1
    coords = _coords_for_profile(rng, node_count, coord_max, profile)
    q = _quantities(rng, M, capacity_profile=capacity_profile)
    Q = _capacities(rng, K, q, capacity_profile=capacity_profile)
    d = _distance_matrix(rng, coords, asymmetric=profile == "asymmetric")
    return OfficialInstance(
        N=N,
        M=M,
        K=K,
        q=q,
        Q=Q,
        d=d,
        coords=coords,
        profile=profile,
        capacity_profile=capacity_profile,
        seed=seed,
    )


def format_official_instance(instance: OfficialInstance) -> str:
    lines = [
        f"{instance.N} {instance.M} {instance.K}",
        " ".join(str(value) for value in instance.q),
        " ".join(str(value) for value in instance.Q),
    ]
    lines.extend(" ".join(str(value) for value in row) for row in instance.d)
    return "\n".join(lines) + "\n"


def write_instance(path: Path, instance: OfficialInstance) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_official_instance(instance))
    meta = {
        "N": instance.N,
        "M": instance.M,
        "K": instance.K,
        "seed": instance.seed,
        "profile": instance.profile,
        "capacity_profile": instance.capacity_profile,
        "q_min": min(instance.q),
        "q_max": max(instance.q),
        "Q_min": min(instance.Q),
        "Q_max": max(instance.Q),
        "node_count": len(instance.d),
        "format": "official_matrix",
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def write_suite(root: Path, *, seed: int = 2026) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "sample").mkdir(parents=True, exist_ok=True)
    (root / "sample" / "sample_01.in").write_text(SAMPLE_INPUT)
    (root / "sample" / "sample_01.json").write_text(
        json.dumps(
            {
                "N": 3,
                "M": 3,
                "K": 2,
                "format": "official_matrix",
                "source": "problem_statement_sample",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    specs: list[tuple[str, str, int, int, int, Profile, CapacityProfile]] = [
        ("small", "small_01", 5, 5, 2, "random", "balanced"),
        ("small", "small_02_clustered", 10, 8, 3, "clustered", "loose"),
        ("small", "small_03_tight", 12, 12, 4, "random", "tight"),
        ("medium", "medium_01", 50, 50, 8, "random", "balanced"),
        ("medium", "medium_02_clustered", 80, 40, 10, "clustered", "heterogeneous"),
        ("medium", "medium_03_asymmetric", 60, 60, 12, "asymmetric", "balanced"),
        ("large", "large_01", 150, 150, 25, "random", "balanced"),
        ("large", "large_02_clustered", 250, 250, 40, "clustered", "heterogeneous"),
        ("large", "large_03_max", 500, 500, 80, "random", "balanced"),
        ("edge_cases", "edge_single_taxi", 20, 20, 1, "line", "loose"),
        ("edge_cases", "edge_many_taxis", 20, 20, 80, "clustered", "balanced"),
        ("edge_cases", "edge_parcel_heavy", 5, 80, 12, "random", "tight"),
        ("edge_cases", "edge_passenger_heavy", 80, 5, 12, "line", "heterogeneous"),
        ("edge_cases", "edge_asymmetric", 30, 30, 8, "asymmetric", "balanced"),
    ]

    for index, (group, name, N, M, K, profile, capacity_profile) in enumerate(specs, start=1):
        instance = generate_instance(
            N=N,
            M=M,
            K=K,
            seed=seed + index,
            profile=profile,
            capacity_profile=capacity_profile,
        )
        write_instance(root / group / f"{name}.in", instance)

    (root / "README.md").write_text(
        """# Official Matrix-Format Test Data

These files follow the Project 11 statement exactly:

```text
N M K
q[1] ... q[M]
Q[1] ... Q[K]
d[0][0] ... d[0][2N+2M]
...
d[2N+2M][0] ... d[2N+2M][2N+2M]
```

Groups:

- `sample`: problem statement sample.
- `small`: quick parser/debug cases.
- `medium`: mid-size benchmark cases.
- `large`: stress cases, including `large_03_max` with `N=M=500`.
- `edge_cases`: single taxi, many taxis, passenger-heavy, parcel-heavy, and asymmetric distances.

Each `.in` has a `.json` metadata file with generation parameters.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Project 11 official matrix-format datasets")
    parser.add_argument("--root", type=Path, default=Path("data/official"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    write_suite(args.root, seed=args.seed)
    print(f"Generated official datasets under {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
