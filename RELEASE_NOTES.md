# DSCA mainline source archive

This directory contains the original Configuration-B source and existing prior
assets, provided with the DSCA preprint.

## Source preservation

- The Configuration-B source files are retained byte for byte. The original
  README is available as `REPRODUCTION.zh-CN.md`.
- Four existing prior-source files are copied without edits: the original teacher
  network and package initializer, student pruning/recovery script, and SDE training script.
- No algorithm, formula, parameter, optimizer, data-routing rule, training budget,
  inference function, or metric implementation has been changed.
- `FILE_MANIFEST.csv` records the relative source location, SHA-256, and size of
  every copied source file and checkpoint.

## Included assets

| Asset | Location | Count |
|---|---|---:|
| Recovered16 initializations | `assets/checkpoints/{WV3,QB}.pth` | 2 |
| Original FusionNet teacher checkpoints | `candidate/ZSVD-test-for-mix/teacher/weights/` | 2 |
| Original per-scene FR SDE weights | `data/FR/<sensor>/model_SDE/` | 80 |

The 8-band initialization/teacher uses WV3 weights for WV3 and WV2; the 4-band
initialization/teacher uses QB weights for QB and GF2. The FR SDE assets correspond
to the original 20-scene ordering of each sensor. They are not interchangeable
with mapper weights fitted for different observations or the independent RR task.

Original dataset H5 files and precomputed teacher prediction MAT files remain
external assets. Supply them at the paths specified in `REPRODUCTION.zh-CN.md`.
Teacher checkpoints are included as original assets; the online runner still
loads the original precomputed teacher MAT interface.

## Running this archive

Follow `REPRODUCTION.zh-CN.md` for the environment and FR/RR commands. From this directory,
the bundled paths can be selected with:

```bash
export DSCA_DATA_ROOT="$PWD/data"
export DSCA_CHECKPOINT_DIR="$PWD/assets/checkpoints"
export OMP_NUM_THREADS=1
```

Each sensor requires `observations.h5`, `internal_wald.h5`, and teacher MAT files
under `data/FR/<sensor>/`. The original FR SDE weight directory is already present.
Use a separate output directory for each sensor/seed training invocation.

`full.mat` and `adapted_rr.mat` contain the raw network predictions. The original
evaluation scripts calculate fixed-alpha MTF correction while scoring. Their
saved products are metrics; this archive does not introduce a new image-export
interface. `internal_wald.mat` is the internal anchor and is distinct from an
independently adapted official RR result.
