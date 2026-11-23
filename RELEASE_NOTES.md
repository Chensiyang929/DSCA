# DSCA assets

| Asset | Repository path | Count |
|---|---|---:|
| Recovered16 student initializations | `assets/checkpoints/{WV3,QB}.pth` | 2 |
| Frozen FusionNet teacher checkpoints | `candidate/teacher/weights/` | 2 |
| Per-scene FR gradient mapper weights | `data/FR/<sensor>/model_SDE/` | 80 |

Use the WV3 student and teacher checkpoints for WV-3 and WV-2, and the QB checkpoints for QB and GF-2. Match the gradient mapper weights to the sensor and scene IDs in the dataset. The official RR protocol uses its own target-scale assets. The [README](README.md) specifies the input layout and benchmark commands.

`FILE_MANIFEST.csv` indexes distributed files by path, size, and SHA-256. Run `sha256sum -c CHECKSUMS.sha256` from the repository root to verify the files.
