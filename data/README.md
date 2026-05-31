# Data Folds

The current fold files were converted from `data/raw_darp/` with seed `42`.

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
