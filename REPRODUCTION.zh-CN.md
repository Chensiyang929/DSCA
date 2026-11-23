# DSCA Configuration B

本包提供论文配置 B 的主线 FR、官方 RR 和 FR 适配器消融。默认使用双适配器、两套独立 Adam、240 次更新、末次 checkpoint、关闭 AMP，以及表中固定的传感器级 MTF 增益。

| 卫星 | RR:FR | RR LR | FR LR | 控制参数 LR 倍率 | 固定 α |
|---|---:|---:|---:|---:|---:|
| WV3 | 2:8 | 0.0015 | 0.000483 | 1 | 0.421988727521689 |
| WV2 | 5:5 | 0.000625 | 0.0005 | 200 | 0.523164735197393 |
| QB | 2:8 | 0.0005 | 0.0004 | 150 | 0.7554278818849035 |
| GF2 | 4:6 | 0.00075 | 0.0008 | 1 | 1.348850350867117 |

`mainline4_config.py` 还记录适配器类型、教师和空间及光谱损失系数。历史固定 α 来自早期 Mix 的 20 景官方 RR 参考拟合，应用于配置 B 的输出。官方 RR 真值只在评估阶段装配与评分，训练使用 observations-only 文件及其内部 Wald 对。

## 环境与资产

参考环境为 Linux、Python 3.10、CUDA 12.8、PyTorch 2.8、torchvision 0.23 和带 image、signal 包的 GNU Octave。按 GPU 环境安装 PyTorch／torchvision，再执行 `python -m pip install -r requirements.txt`。

复现需要单独提供原始数据、Recovered16 初始化、冻结教师预测和逐景 SDE 权重。替换资产可能改变数值。

| 文件 | 配置路径或格式 |
|---|---|
| 学生初始化 | `DSCA_CHECKPOINT_DIR/WV3.pth` 供 WV3／WV2 使用，`DSCA_CHECKPOINT_DIR/QB.pth` 供 QB／GF2 使用 |
| FR 观测 | `DSCA_DATA_ROOT/FR/<sensor>/observations.h5`，含原始 DN 下的 NCHW `ms`、`lms`、`pan` |
| FR 内部 Wald | `DSCA_DATA_ROOT/FR/<sensor>/internal_wald.h5`，含 `ms`、`lms`、`pan`、`gt`，`gt` 等于 FR 观测 MS |
| FR 教师 | `DSCA_DATA_ROOT/FR/<sensor>/teacher/<scene>_student_SDE_0.15.mat`；MAT 含 HWC 的 `proposed` 或 `sr` |
| FR SDE | `DSCA_DATA_ROOT/FR/<sensor>/model_SDE/<scene>_ms2pan_convNet_dual.pth` |
| 官方 RR 资产 | 官方 H5，以及与 RR 输出网格对应的教师预测和逐景 SDE 权重 |

## 运行示例

在源码目录使用绝对资产路径。示例运行 WV3 第 0 景、seed 10。完整论文评估对四星分别运行场景 0–19、seed 10／11／12，给每次训练一个独立的输出目录。

```bash
export DSCA_DATA_ROOT=/absolute/path/to/data
export DSCA_CHECKPOINT_DIR=/absolute/path/to/checkpoints
export OMP_NUM_THREADS=1
OUT="$PWD/repro_outputs"

python run_mainline4_full_variant.py WV3 --scenes 0 --seed 10 --output-root "$OUT/fr"
python score_config_b.py fr WV3 --scenes 0 --result-root "$OUT/fr"
```

在 FR 命令中使用 `--adapter base` 或 `--adapter shared` 可复跑本次补充对照。`head` 是默认双适配器。消融评分同时输出 α = 0 和固定 α，适配器类型与训练命令保持一致。

```bash
python run_mainline4_full_variant.py WV3 --adapter shared --scenes 0 \
  --seed 10 --output-root "$OUT/fr_shared"
python score_config_b.py fr-adapter WV3 --adapter shared --scenes 0 \
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
python score_config_b.py rr WV3 --result-root "$OUT/rr/WV3" \
  --pairs "$OUT/rr_pairs" --scenes 0
```

FR 评分可输出内部 Wald 指标，`rr` 评分输出官方 RR 指标。二者对应不同的输入和适配过程。

## 源码索引

- `mainline4_config.py` 是配置和固定 α 的入口。
- `candidate/` 包含各星适配器、同一主干、优化与推理逻辑。
- `score_config_b.py` 调用 Octave 评分。现有评估工具箱的版权声明和许可证位于 `eval_toolbox/`。

## 验证范围

在 Linux 服务器用原始资产跑通了 WV3 第 0 景 seed 10 的主线 FR、共享适配器 FR、官方 RR 数据准备与评分。三个实例均完成 48 次 RR 和 192 次 FR 更新，AMP 跳步为零。与论文归档相比，单景复跑的 FR HQNR 差约 −0.000035，官方 RR PSNR 差约 +0.0047 dB。
