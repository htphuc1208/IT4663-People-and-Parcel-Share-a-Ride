# Raw DARP Benchmarks

Downloaded from Maxime Chassaing's ELS DARP benchmark page:

- `cordeau_laporte_2003/all/pr01.txt` ... `pr20.txt`
- `ropke_cordeau_2007/part_a/a2-16.txt` ... `a8-96.txt`
- `ropke_cordeau_2007/part_b/b2-16.txt` ... `b8-96.txt`

The zip archives are kept next to the extracted files for reproducibility.

Convert one raw DARP file to this repository's fold `.txt` format:

```bash
python3 -m src.preprocess \
  --raw data/raw_darp/ropke_cordeau_2007/part_a/a2-16.txt \
  --out data/fold1/a2-16.txt \
  --seed 42 \
  --capacity-mode heterogeneous \
  --capacity-min 1 \
  --capacity-max 3
```

The original DARP files define one shared capacity for all vehicles. The
preprocessor can preserve that with `--capacity-mode uniform`, generate
heterogeneous capacities, or accept exact capacities through
`--taxi-capacities 1,3`.
