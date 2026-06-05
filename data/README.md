# Data

## DARP-Derived Optimizer Folds

The current `fold*` files were converted from `data/raw_darp/` with seed `42`.

Capacity ranges:

- `fold1/a*.txt`: generated in `[1, 3]`
- `fold2/b*.txt`: generated in `[3, 6]`
- `fold3/pr*.txt`: generated in `[1, 6]`

The first line of each converted instance is:

```text
K N M Q1 Q2 ... QK T
```

This keeps the original `K N M Q T` format backward-compatible while allowing
per-taxi capacities for the project problem.

## Official Matrix-Format Cases

`data/official/` contains synthetic cases that follow the Project 11 statement
exactly:

```text
N M K
q[1] ... q[M]
Q[1] ... Q[K]
d[0][0] ... d[0][2N+2M]
...
d[2N+2M][0] ... d[2N+2M][2N+2M]
```

These are intended for parser, validator, and contest-format solver testing.
They are grouped into `sample`, `small`, `medium`, `large`, and `edge_cases`.
`src.encoding_and_read.read_instance()` auto-detects this format, so these
files can also be used with `python3 -m src.main --instance ...`.
