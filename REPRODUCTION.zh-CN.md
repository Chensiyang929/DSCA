# DSCA 复现指南

DSCA 对单景观测交替执行降分辨率（RR）监督和全分辨率（FR）一致性优化。论文主实验使用两个尺度专用适配器和独立的 Adam 状态，执行 240 次 FP32 更新，取最后一次更新的权重，并以固定的传感器增益校正输出。

| 卫星 | RR:FR | RR LR | FR LR | 控制参数 LR 倍率 | 固定 α |
|---|---:|---:|---:|---:|---:|
| WV3 | 2:8 | 0.0015 | 0.000483 | 1 | 0.421988727521689 |
| WV2 | 5:5 | 0.000625 | 0.0005 | 200 | 0.523164735197393 |
| QB | 2:8 | 0.0005 | 0.0004 | 150 | 0.7554278818849035 |
| GF2 | 4:6 | 0.00075 | 0.0008 | 1 | 1.348850350867117 |

`mainline4_config.py` 记录适配器类型、损失权重和完整精度的增益。每个传感器的固定 α 由官方 RR 校准预测汇总拟合，随后应用于该传感器的输出。官方 RR 参考真值用于评估。适配以当前场景的观测及其派生的内部 Wald 对为输入。

## 环境与资产

参考环境为 Linux、Python 3.10、CUDA 12.8、PyTorch 2.8、torchvision 0.23 和带 image、signal 包的 GNU Octave。按 GPU 环境安装 PyTorch／torchvision，再执行 `python -m pip install -r requirements.txt`。

仓库提供 Recovered16 初始化、FusionNet 教师权重，以及四颗卫星各 20 景的 FR 梯度映射权重。用 [PanCollection](https://github.com/liangjiandeng/PanCollection) 数据准备观测、内部 Wald 对和对应的教师预测 MAT 文件。保持数据集场景顺序及传感器目录名 `WV3`、`WV2`、`QB`、`GF2`。

| 文件 | 配置路径或格式 |
|---|---|
| 学生初始化 | `DSCA_CHECKPOINT_DIR/WV3.pth` 供 WV3／WV2 使用，`DSCA_CHECKPOINT_DIR/QB.pth` 供 QB／GF2 使用 |
| FR 观测 | `DSCA_DATA_ROOT/FR/<sensor>/observations.h5`，含原始 DN 下的 NCHW `ms`、`lms`、`pan` |
| FR 内部 Wald | `DSCA_DATA_ROOT/FR/<sensor>/internal_wald.h5`，含 `ms`、`lms`、`pan`、`gt`，`gt` 等于 FR 观测 MS |
| FR 教师 | `DSCA_DATA_ROOT/FR/<sensor>/teacher/<scene>_student_SDE_0.15.mat`；MAT 含 HWC 的 `proposed` 或 `sr` |
| FR 梯度映射权重（仓库提供） | `DSCA_DATA_ROOT/FR/<sensor>/model_SDE/<scene>_ms2pan_convNet_dual.pth` |
| 官方 RR 资产 | 官方 H5，以及与 RR 输出网格对应的教师预测和逐景梯度映射权重 |

## 运行示例

在仓库根目录运行以下示例。默认目录包含配套权重。示例运行 WV3 第 0 景、seed 10。完整评估使用 `--scenes all` 和 seed 10、11、12，各传感器和 seed 使用独立的输出目录。

```bash
export DSCA_DATA_ROOT="$PWD/data"
export DSCA_CHECKPOINT_DIR="$PWD/assets/checkpoints"
export OMP_NUM_THREADS=1
OUT="$PWD/repro_outputs"

python run_mainline4_full_variant.py WV3 --scenes 0 --seed 10 --output-root "$OUT/fr"
python evaluate.py fr WV3 --scenes 0 --result-root "$OUT/fr"
```

适配器消融使用 `--adapter base` 或 `--adapter shared`。`head` 是默认双适配器。训练与评分使用相同的适配器参数，评分分别报告 α = 0 和固定 α 下的 FR 指标。

```bash
python run_mainline4_full_variant.py WV3 --adapter shared --scenes 0 \
  --seed 10 --output-root "$OUT/fr_shared"
python evaluate.py fr-adapter WV3 --adapter shared --scenes 0 \
  --result-root "$OUT/fr_shared"
```

官方 RR 需要独立适配。RR 的 `--sde-root` 应包含 `WV3/<scene>_ms2pan_convNet_dual.pth`。

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
python evaluate.py rr WV3 --result-root "$OUT/rr/WV3" \
  --pairs "$OUT/rr_pairs" --scenes 0
```

`full.mat` 和 `adapted_rr.mat` 保存原始预测。评分时应用固定 MTF 增益并保存指标。`fr` 评分附带校正后的内部 Wald 指标，论文的内部 RR 锚点表用 α = 0 的原始内部 Wald 预测。官方 RR 的 PSNR、SSIM 在四边各裁 21 像素，Q2n、SAM、ERGAS 使用评估工具箱的 `dim_cut=21`，对应两个空间维度的 `21:end-21`。

## 源码索引

- `mainline4_config.py` 是配置和固定 α 的入口。
- `candidate/` 包含各星适配器、同一主干、优化与推理逻辑。
- `evaluate.py` 调用 Octave 评分。评估工具箱的许可证位于 `eval_toolbox/`。
