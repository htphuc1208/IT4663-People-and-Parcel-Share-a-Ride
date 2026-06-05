# Official Matrix-Format Test Data

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
