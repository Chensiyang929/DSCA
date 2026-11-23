# DSCA: Dual-Scale Collaborative Adaptation for Single-Scene Full-Resolution Pansharpening

Qinghong Jiang\*, Siyang Chen\*, He Zou, Ding Yiheng, and Liang-Jian Deng  
University of Electronic Science and Technology of China  
\* Qinghong Jiang and Siyang Chen contributed equally and share first authorship.

This repository contains the DSCA implementation for single-scene pansharpening under the paper's Configuration B. DSCA alternates reduced-resolution supervision with full-resolution consistency on each target scene. Separate Adam optimizers retain scale-specific states, and a sensor-specific MTF correction is applied at inference evaluation.

## Reproduce the main experiments

Use Linux, Python 3.10, PyTorch 2.8 with CUDA 12.8, torchvision 0.23, and GNU Octave with its `image` and `signal` packages. Install PyTorch and torchvision for your GPU, then install the remaining Python dependencies.

```bash
git clone https://github.com/Chensiyang929/DSCA.git
cd DSCA
python -m pip install -r requirements.txt
export DSCA_DATA_ROOT="$PWD/data"
export DSCA_CHECKPOINT_DIR="$PWD/assets/checkpoints"
export OMP_NUM_THREADS=1
```

The repository includes two Recovered16 initializations, two FusionNet teacher checkpoints, and the full-resolution spatial mapper weights for 20 scenes per sensor. Obtain the dataset files from [PanCollection](https://github.com/liangjiandeng/PanCollection) and prepare the observation, internal Wald, and teacher prediction inputs listed in the [reproduction guide](REPRODUCTION.zh-CN.md). An independent official RR run uses its corresponding teacher predictions and spatial mapper weights.

With the assets in place, this example adapts and scores WV-3 scene 0 with seed 10.

```bash
OUT="$PWD/repro_outputs"
python run_mainline4_full_variant.py WV3 --scenes 0 --seed 10 --output-root "$OUT/fr"
python score_config_b.py fr WV3 --scenes 0 --result-root "$OUT/fr"
```

See the [reproduction guide](REPRODUCTION.zh-CN.md) for the full-resolution adapter ablation, official reduced-resolution evaluation, and sensor settings. The [evaluation toolbox license](eval_toolbox/LICENSE) accompanies its source.
