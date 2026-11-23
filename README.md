# DSCA: Dual-Scale Collaborative Adaptation for Single-Scene Full-Resolution Pansharpening

Qinghong Jiang\*, Siyang Chen\*, He Zou, Ding Yiheng, and Liang-Jian Deng<br>
University of Electronic Science and Technology of China<br>
\* These authors contributed equally.

DSCA adapts a compact pansharpening network to each observed panchromatic and multispectral image pair. It alternates reduced-resolution (RR) supervision with full-resolution (FR) consistency using scale-specific input adapters and independent Adam states. A fixed sensor gain scales the output MTF consistency correction.

## Setup

Use Linux, Python 3.10, PyTorch 2.8 with CUDA 12.8, torchvision 0.23, and GNU Octave with its `image` and `signal` packages. Install PyTorch and torchvision for your GPU, then install the remaining Python dependencies.

```bash
git clone https://github.com/Chensiyang929/DSCA.git
cd DSCA
python -m pip install -r requirements.txt
export DSCA_DATA_ROOT="$PWD/data"
export DSCA_CHECKPOINT_DIR="$PWD/assets/checkpoints"
export OMP_NUM_THREADS=1
```

## Data and weights

Obtain the FR and official RR benchmark data from [PanCollection](https://github.com/liangjiandeng/PanCollection). Keep the benchmark scene order (`0` through `19`) and sensor directory names (`WV3`, `WV2`, `QB`, `GF2`). Store HDF5 inputs in NCHW order and original digital numbers.

| FR input | Path under `DSCA_DATA_ROOT` |
|---|---|
| Observations (`ms`, `lms`, `pan`) | `FR/<sensor>/observations.h5` |
| Internal Wald pair (`ms`, `lms`, `pan`, `gt`) | `FR/<sensor>/internal_wald.h5` |
| Frozen FusionNet prediction (`proposed` or `sr`, HWC) | `FR/<sensor>/teacher/<scene>_student_SDE_0.15.mat` |
| Scene-specific gradient mapper, supplied in the repository | `FR/<sensor>/model_SDE/<scene>_ms2pan_convNet_dual.pth` |

Derive the internal Wald pair from the corresponding FR observations at scale ratio 4. Its `gt` is the observed `ms` from `observations.h5`. The [asset index](RELEASE_NOTES.md) lists the bundled student initializations, teacher checkpoints, and FR gradient mappers. Prepare the teacher prediction MAT files for the target scenes. The official RR experiment uses its own teacher predictions and gradient mappers for the corresponding RR scenes.

## Full-resolution benchmark

With the assets in place, this example adapts and scores WV-3 scene 0 with seed 10.

```bash
OUT="$PWD/repro_outputs"
python run_mainline4_full_variant.py WV3 --scenes 0 --seed 10 --output-root "$OUT/fr"
python evaluate.py fr WV3 --scenes 0 --result-root "$OUT/fr"
```

For the FR adapter study, set `--adapter shared` or `--adapter base` during adaptation. Use the same adapter with `evaluate.py fr-adapter` to score both zero gain and the fixed sensor gain. The default `head` selects the two scale-specific adapters.

## Official reduced-resolution benchmark

Prepare the target observations and their internal Wald pairs, then adapt each scene independently. `materialize_rr_evaluation_pairs.py` assembles the prediction and official reference for evaluation. Store the RR gradient mappers in `RR_SDE/<sensor>/`.

```bash
OFFICIAL=/absolute/path/to/official_rr.h5
PREPARED="$OUT/prepared/WV3"
RR_TEACHER=/absolute/path/to/rr_teacher/WV3
RR_SDE=/absolute/path/to/rr_model_SDE

python prepare_rr_target_adaptation.py --sensor WV3 --official-rr "$OFFICIAL" \
  --output-dir "$PREPARED"
python run_mainline4_rr_adaptation.py WV3 --prepared-root "$PREPARED" \
  --reuse-prepared --sde-root "$RR_SDE" --teacher-pattern "$RR_TEACHER/{scene}_student.mat" \
  --scenes 0 --seed 10 --output-root "$OUT/rr"
python materialize_rr_evaluation_pairs.py --sensor WV3 --official-rr "$OFFICIAL" \
  --result-root "$OUT/rr/WV3" --training-observations "$PREPARED/WV3_rr_observations_only.h5" \
  --output-dir "$OUT/rr_pairs" --scenes 0
python evaluate.py rr WV3 --result-root "$OUT/rr/WV3" --pairs "$OUT/rr_pairs" --scenes 0
```

## Settings and outputs

The benchmark uses 240 FP32 updates, the final checkpoint, and a fixed gain for each sensor. The exact schedules, learning rates, and gains are recorded in [`mainline4_config.py`](mainline4_config.py). Use `--scenes all` and seeds 10, 11, and 12 for the complete benchmark. Keep each sensor and seed in its own output directory. The FR benchmark reports the mean and standard deviation over the 60 scene and seed scores per sensor.

Adaptation saves raw predictions as `full.mat` and `adapted_rr.mat`. Scoring applies the fixed MTF gain and saves the metrics. The `fr` scorer also reports calibrated internal Wald diagnostics. The paper's internal RR anchor table uses the raw internal Wald predictions at zero gain. In official RR scoring, PSNR and SSIM crop 21 pixels from each edge. Q2n, SAM, and ERGAS use the evaluation toolbox's `dim_cut=21` setting (`21:end-21` in the two spatial dimensions).

The benchmark commands consume precomputed teacher predictions. The paper's per-scene timing covers fresh teacher inference, adaptation, final output correction, and file I/O.

[中文复现说明](REPRODUCTION.zh-CN.md) · [Evaluation toolbox license](eval_toolbox/LICENSE)
