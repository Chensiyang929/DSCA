import argparse # type: ignore
import csv
import random
import time
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
# 导入混合精度训练所需的库
from torch.cuda.amp import autocast, GradScaler

from data import Dataset, ReducedDataset  # 数据加载器
from mymodel import FusionNet  # 学生模型
from SDE import Net_ms2pan_dual,sobel_filter,ms2pan_convNet_dual  # SDE模块
print("Updated sys.path:", sys.path)
from loss import LossCalculator  # 损失计算器
from teacher_output_loader import load_teacher_output  # 直接运行预训练FusionNet教师
from loss_weight_config import get_loss_weights
from training_ratio_config import get_training_cycle, get_training_epochs, epoch_schedule
from hqnr_selection import HQNRSelector, atomic_json, file_sha256
from wald_utilities import wald_protocol_v1, wald_protocol_v2  # 学生模型

# ================== 基础设置 =================== #
SEED = int(os.environ.get("EXPERIMENT_SEED", "10"))
if SEED < 0:
    raise ValueError("EXPERIMENT_SEED must be non-negative")
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.deterministic = True

# ============= 超参数设置 ========= #
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.005, help="学习率")
parser.add_argument("--epochs", type=int, default=None, help="训练上限；默认WV3=240/WV2=480/QB=480/GF2=120")
parser.add_argument("--dlpan_root", default=None, help="DLPan工具箱根目录")
parser.add_argument("--hqnr_engine", choices=['auto', 'octave', 'matlab'], default='auto')
parser.add_argument("--selection_mode", choices=['per_epoch_hqnr', 'last'], default='last')
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--device", type=str, default='cuda:0', help="训练设备")
parser.add_argument("--data_id", type=int, default=0, help="数据ID (0-19)")
parser.add_argument("--sensor", type=str, default='wv3', help="传感器类型")
parser.add_argument("--bands", type=int, default=8, choices=[4, 8], help="多光谱波段数")
parser.add_argument("--ratio", type=int, default=4, help="下采样比例")
parser.add_argument("--temperature", type=float, default=1.0, help="蒸馏温度参数")
# 添加混合精度训练的参数选项
parser.add_argument("--amp",default=True, action="store_true", help="启用混合精度训练")
parser.add_argument("--no-amp", dest="amp", action="store_false", help="Use strict FP32 without AMP skip steps")
parser.add_argument("--alfa", type=float, default=0.15, help="损失权重")
parser.add_argument("--data_path", type=str, default=r"HardDisk/HeZou/test_wv3_OrigScale_multiExm1.h5", help="数据文件路径")
parser.add_argument("--use_reduced", type=int, default=1, choices=[0, 1],
                    help="是否启用reduced数据训练 (1启用, 0禁用)")
parser.add_argument("--reduced_data_path", type=str, default=None,
                    help="预构造的reduced数据h5路径，默认: <data_path>去掉.h5加_reduced.h5")
parser.add_argument("--reduced_ratio", type=float, default=None,
                    help="省略时按卫星HQNR最佳周期训练；显式指定0~100则使用旧均匀比例调度")
parser.add_argument("--reduced_loss_weight", type=float, default=10000.0,
                    help="reduced数据gt损失的权重")
parser.add_argument("--reduced_patch_size", type=int, default=0,
                    help="当前主线固定为0：使用整幅Internal-Wald目标监督")
parser.add_argument("--reduced_steps_per_epoch", type=int, default=1,
                    help="每个GT epoch抽取的RR patch更新次数；默认1以保持epoch比例")
parser.add_argument("--model_dir", type=str, default="model_FUG",
                    help="学生模型保存目录")
parser.add_argument("--w_var", type=float, default=None,
                    help="教师损失系数；默认使用传感器配置")
parser.add_argument("--w_spa", type=float, default=None,
                    help="SDE空间损失系数；默认使用传感器配置")
parser.add_argument("--w_spec", type=float, default=None,
                    help="MTF光谱损失系数；默认使用传感器配置")
args = parser.parse_args()
if args.ratio != 4:
    parser.error('Per-epoch DLPan HQNR protocol currently requires --ratio 4.')
if args.reduced_patch_size != 0:
    parser.error('当前主线禁止GT patch；必须使用整幅Internal-Wald目标监督。')

lr = args.lr
epochs = get_training_epochs(args.sensor, args.epochs)
batch_size = args.batch_size
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
data_id = args.data_id
sensor = args.sensor.upper()
bands = args.bands
ratio = args.ratio
# Select FP32 or automatic mixed precision for FR training.
force_fp32 = os.environ.get("FORCE_FP32", "0") == "1"
use_amp = args.amp and (device.type == 'cuda') and not force_fp32
teacher_weight_mode = os.environ.get('TEACHER_WEIGHT_MODE', 'legacy')
if teacher_weight_mode not in {'legacy', 'static'}:
    parser.error('TEACHER_WEIGHT_MODE must be legacy or static')
teacher_enabled = os.environ.get('DISABLE_TEACHER', '0') != '1'


ri = 140*1
alfa = args.alfa 
# 损失权重设置（与train_or.py保持一致）
#w_var = 1800000.0*0.95   # 变分损失权重（对应train_or.py中的w_var，没有像素级权重模型中最优参数

#自适应亚像素级权重参数
sensor_weights = get_loss_weights(sensor)
w_var = args.w_var if args.w_var is not None else sensor_weights["teacher"]
if not teacher_enabled:
    w_var = 0.0


w_spa = args.w_spa if args.w_spa is not None else sensor_weights["spatial"]
w_spec = args.w_spec if args.w_spec is not None else sensor_weights["spectral"]

# 数据文件路径
data_path = args.data_path

# reduced 数据相关设置
use_reduced = bool(args.use_reduced)
selected_cycle = get_training_cycle(sensor)
reduced_ratio = (100.0 * selected_cycle[0] / sum(selected_cycle)
                 if args.reduced_ratio is None else
                 max(0.0, min(100.0, float(args.reduced_ratio))))
selected_schedule = epoch_schedule(sensor, epochs, args.reduced_ratio)
reduced_loss_weight = args.reduced_loss_weight
model_dir = args.model_dir
if args.reduced_data_path and args.reduced_data_path != 'None':
    reduced_data_path = args.reduced_data_path
else:
    reduced_data_path = os.path.splitext(data_path)[0] + "_reduced.h5"


# U2Net预训练模型路径

# =================== 模型初始化 =================== #
# 学生模型 (FusionNet)
model_student = FusionNet(bands).to(device)
# 一阶段训练：直接从随机初始化训练融合网络，不加载model_pretrain预热权重
print("学生模型初始化完成")

# 损失计算器初始化
loss_calculator = LossCalculator(sensor=sensor, ratio=ratio, N=41, device=device)

F_ms2pan = ms2pan_convNet_dual(bands).to(device)
F_ms2pan.load_state_dict(torch.load(f'model_SDE/{sensor}/{data_id}_ms2pan_convNet_dual.pth'))
F_ms2pan.eval()
F_ms2pan.requires_grad_(False)


# 优化器
optimizer = optim.Adam(model_student.parameters(), lr=lr, betas=(0.9, 0.999))


# 教师输出由预训练FusionNet直接生成



# Set the initial gradient scale when mixed precision is enabled.
amp_init_scale = float(os.environ.get("AMP_INIT_SCALE", "65536.0"))
if amp_init_scale <= 0:
    parser.error("AMP_INIT_SCALE must be positive")
scaler = GradScaler(init_scale=amp_init_scale) if use_amp else None



if use_amp:
    print("已启用混合精度训练 (AMP)")
elif args.amp and device.type != 'cuda':
    print("警告: 混合精度训练仅支持CUDA设备，已自动禁用")

# 模型保存函数
def save_checkpoint(model, identifier):
    os.makedirs(model_dir, exist_ok=True)
    model_out_path = os.path.join(model_dir, f"{identifier}.pth")
    torch.save(model.state_dict(), model_out_path)

# ================ 知识蒸馏训练过程 ================ #
def train(training_data_loader, reduced_data_loader, identifier, teacher_output):
    print("开始知识蒸馏训练...")
    actual_updates = {"gt": 0, "full": 0, "full_amp_skipped": 0}
    selector = (HQNRSelector(data_path, sensor, data_id, model_dir, identifier,
                             epochs, device, toolbox_root=args.dlpan_root,
                             engine=args.hqnr_engine)
                if args.selection_mode == 'per_epoch_hqnr' else None)
    
    # 教师输出在训练前由冻结的预训练FusionNet生成一次
    teacher_outputs = [teacher_output]
    
    min_full_loss = float("inf")
    min_red_loss = float("inf")
    full_log_interval = int(os.environ.get("TRAIN_LOG_INTERVAL", "1"))
    if full_log_interval < 1:
        raise ValueError("TRAIN_LOG_INTERVAL must be positive")
    target_count = sum(selected_schedule)
    if args.reduced_ratio is None:
        print(f"HQNR默认周期: {selected_cycle[0]} GT -> {selected_cycle[1]} Full; "
              f"合计{target_count} GT/{epochs-target_count} Full epochs")
    reduced_log_interval = min(full_log_interval, max(1, target_count))  # reduced 每隔 min(10, epochs*reduced比例) 个 epoch 打印一次
    last_full_log = None  # (epoch, var, spa, spec, total)：最近一个 full 轮的损失，供 reduced 轮打印 full 损失
    start_time = time.time()
    trace_path = os.path.join(model_dir, f"{identifier}_training_trace.csv")
    os.makedirs(model_dir, exist_ok=True)
    trace_fields = ["epoch", "domain", "seed", "total_loss", "loss_gt", "loss_var", "loss_spa", "loss_spec", "gt_updates", "full_updates", "amp_skipped", "backbone_delta", "head_gt_delta", "head_full_delta"]
    with open(trace_path, "w", newline="") as stream:
        csv.DictWriter(stream, fieldnames=trace_fields).writeheader()
    for epoch in range(1, epochs + 1):
        audit_updates = os.environ.get("AUDIT_PARAMETER_UPDATES", "0") == "1"
        before = {n: p.detach().clone() for n, p in model_student.named_parameters()} if audit_updates else {}
        model_student.train()
        epoch_total_loss = []
        epoch_loss_var, epoch_loss_spa, epoch_loss_spec, epoch_loss_red = [], [], [], []

        # reduced 混合比例(0~100): 0 → 纯full(无监督)，100 → 纯reduced(有监督)
        # 默认使用已验证的连续GT -> 连续Full周期；显式百分比保留旧调度。
        use_reduced_epoch = (reduced_data_loader is not None) and selected_schedule[epoch - 1]

        if use_reduced_epoch:
            # ---------- reduced 数据: 用 gt 构造损失 ----------
            for i, batch in enumerate(reduced_data_loader):
                lms, pan, gt = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer.zero_grad()

                # 确保pan维度正确
                if len(pan.shape) == 3:
                    pan = pan.unsqueeze(1)

                # 学生模型前向传播
                res_student = model_student(lms, pan, domain="gt")
                fusion_out = res_student + lms

                # reduced 数据由 full 构造得到，有 gt，直接用 gt 构造损失
                loss_red = torch.mean((fusion_out - gt) ** 2)
                total_loss = reduced_loss_weight * loss_red

                epoch_loss_red.append(loss_red.item())
                epoch_total_loss.append(total_loss.item())

                total_loss.backward()
                optimizer.step()
                actual_updates["gt"] += 1
                if (
                    args.reduced_steps_per_epoch > 0
                    and i + 1 >= args.reduced_steps_per_epoch
                ):
                    break
        else:
            # ---------- full 数据: 保持原有SDE蒸馏损失 ----------
            epoch_loss_var, epoch_loss_spa, epoch_loss_spec = [], [], []

            for i, batch in enumerate(training_data_loader):
                # 跳过教师模型预计算失败的批次
                if teacher_enabled and teacher_outputs[i] is None:
                    continue

                ms, lms, pan = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer.zero_grad()

                # 确保pan维度正确
                if len(pan.shape) == 3:
                    pan = pan.unsqueeze(1)

                # 使用混合精度进行前向传播和损失计算
                if use_amp:
                    with autocast():
                        # 学生模型前向传播
                        res_student = model_student(lms, pan, domain="full")
                        fusion_out_ori = res_student + lms  
                        fusion_out = fusion_out_ori.squeeze(0)  # [C, H, W]

                        # 使用预计算的教师模型输出
                        fusion_out_teacher = (teacher_outputs[i].to(device).squeeze(0)
                                              if teacher_enabled else fusion_out.detach())
                        fusion_out_teacher_bchw = fusion_out_teacher.unsqueeze(0)  # 确保维度正确b c h w

                        #计算patch-wise变分损失权重
                        # teacher_outputs_lr = F.max_pool2d(fusion_out_teacher_bchw, kernel_size=4, stride=4)  # 使用最大池化代替Wald协议

                        # delta = (teacher_outputs_lr - ms).abs()                     # 逐元素差值的绝对值
                        # # delta: [B, C, H, W]
                        # delta_max = delta.amax(dim=(2, 3), keepdim=True)   # 每个通道独立的最大值
                        # delta_inv = torch.exp(-delta / (delta_max + 1e-8))
                        # delta_up = F.interpolate(delta_inv, scale_factor=4, mode='nearest')

                        #计算pixel-wise变分损失权重
                        if teacher_weight_mode == 'static':
                            delta_up = torch.ones_like(fusion_out_teacher_bchw)
                        else:
                            ms_up = F.interpolate(ms, scale_factor=ratio, mode='nearest')
                            delta = (fusion_out_teacher_bchw - ms_up).abs()  #
                            delta_max = delta.amax(dim=(2, 3), keepdim=True)  # 每个通道独立的最大值
                            delta_up = torch.exp(-delta / (delta_max + 1e-8))

                        #print(delta_up.mean().item())
                        # 变分损失 (与train_or.py中的loss1对应)
                        loss_var = torch.mean(((fusion_out - fusion_out_teacher)*delta_up) ** 2)

                        # 转换为[H, W, C]格式用于计算其他损失
                        fusion_out_hw_c = fusion_out.permute(1, 2, 0)

                        # 获取低分辨率图像尺寸并计算block_size
                        _, H, _ = ms[0].shape
                        block_size = H // ratio

                        # # 空间保真度损失 (与train_or.py中的loss2对应)
                        # loss_spa = loss_calculator.compute_spatial_fidelity_loss(
                        #     fusion_out_hw_c,
                        #     ms[0].permute(1, 2, 0),
                        #     pan[0].squeeze(0),
                        #     block_size
                        # )
                        # 使用SDE网络将多光谱图像转换为全色图像
                        with autocast() if use_amp else torch.no_grad():
                            fusion_out_gra_x,fusion_out_gra_y = sobel_filter(fusion_out_ori)
                            sde_out_x,sde_out_y = F_ms2pan(fusion_out_gra_x,fusion_out_gra_y) # 假设SDE网络输出维度为 [batch_size, 1, H, W]，因此需要squeeze掉通道维度

                        # 计算MS再SDE网络输出和pan图的L2损失
                        pan_gra_x,pan_gra_y = sobel_filter(pan)
                        loss_x = loss_calculator.SDE_Loss(sde_out_x , pan_gra_x)
                        loss_y = loss_calculator.SDE_Loss(sde_out_y , pan_gra_y)
                        loss_spa = (loss_x + loss_y)/100
                        # loss_spa = loss_x/100

                        # 光谱损失 (与train_or.py中的loss3对应)
                        loss_spec = loss_calculator.compute_spectral_loss(
                            fusion_out_hw_c, 
                            ms[0].permute(1, 2, 0)
                        )
                        loss_spec_teacher = loss_calculator.compute_spectral_loss(
                            fusion_out_teacher.permute(1, 2, 0), 
                            ms[0].permute(1, 2, 0)
                        )

                        if teacher_weight_mode == 'static':
                            teacher_scale = loss_var.new_tensor(1.0)
                        else:
                            weight = torch.sigmoid(1*(torch.abs(loss_spec_teacher - 28.65)/10))*2.0
                            teacher_scale = 1.0 / weight

                        # 总损失 (使用train_or.py中的权重)

                        total_loss = (w_var * teacher_scale) * loss_var + (w_spa * loss_spa + w_spec * loss_spec)

                    # 使用梯度缩放器进行反向传播
                    scaler.scale(total_loss).backward()

                    # 更新梯度并执行优化步骤
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    #scaler.update()

                    scaler.update()
                    skipped = scaler.get_scale() < scale_before
                    actual_updates["full"] += int(not skipped)
                    actual_updates["full_amp_skipped"] += int(skipped)

                else:
                    # Compute the FR objective and update in FP32.
                    res_student = model_student(lms, pan, domain="full")
                    fusion_out_ori = res_student + lms
                    fusion_out = fusion_out_ori.squeeze(0)  # [C, H, W]

                    fusion_out_teacher = (teacher_outputs[i].to(device).squeeze(0)
                                          if teacher_enabled else fusion_out.detach())
                    fusion_out_teacher_bchw = fusion_out_teacher.unsqueeze(0)

                    if teacher_weight_mode == "static":
                        delta_up = torch.ones_like(fusion_out_teacher_bchw)
                    else:
                        ms_up = F.interpolate(ms, scale_factor=ratio, mode="nearest")
                        delta = (fusion_out_teacher_bchw - ms_up).abs()
                        delta_max = delta.amax(dim=(2, 3), keepdim=True)
                        delta_up = torch.exp(-delta / (delta_max + 1e-8))

                    loss_var = torch.mean(
                        ((fusion_out - fusion_out_teacher) * delta_up) ** 2
                    )

                    fusion_out_hw_c = fusion_out.permute(1, 2, 0)
                    _, H, _ = ms[0].shape
                    block_size = H // ratio

                    # Freeze SDE weights, but retain its derivative with respect
                    # to the student prediction (also in strict FP32).
                    with torch.enable_grad():
                        fusion_out_gra_x, fusion_out_gra_y = sobel_filter(fusion_out_ori)
                        sde_out_x, sde_out_y = F_ms2pan(
                            fusion_out_gra_x, fusion_out_gra_y
                        )
                    pan_gra_x, pan_gra_y = sobel_filter(pan)
                    loss_x = loss_calculator.SDE_Loss(sde_out_x, pan_gra_x)
                    loss_y = loss_calculator.SDE_Loss(sde_out_y, pan_gra_y)
                    with torch.no_grad():
                        lms_gra_x, lms_gra_y = sobel_filter(lms)
                        base_x, base_y = F_ms2pan(lms_gra_x, lms_gra_y)
                        base_spa = (
                            loss_calculator.SDE_Loss(base_x, pan_gra_x)
                            + loss_calculator.SDE_Loss(base_y, pan_gra_y)
                        ) / 2
                    loss_spa = torch.sqrt(
                        ((loss_x + loss_y) / 2) / (base_spa.detach() + 1e-8) + 1e-8
                    )

                    loss_spec = loss_calculator.compute_spectral_loss(
                        fusion_out_hw_c,
                        ms[0].permute(1, 2, 0),
                    )
                    loss_spec_teacher = loss_calculator.compute_spectral_loss(
                        fusion_out_teacher.permute(1, 2, 0),
                        ms[0].permute(1, 2, 0),
                    )

                    if teacher_weight_mode == "static":
                        teacher_scale = loss_var.new_tensor(1.0)
                    else:
                        weight = torch.sigmoid(
                            torch.abs(loss_spec_teacher - 28.65) / 10
                        ) * 2.0
                        teacher_scale = 1.0 / weight

                    total_loss = (
                        (w_var * teacher_scale) * loss_var
                        + w_spa * loss_spa
                        + w_spec * loss_spec
                    )

                    total_loss.backward()
                    optimizer.step()
                    actual_updates["full"] += 1

                # 记录各损失值
                epoch_loss_var.append(loss_var.item())
                epoch_loss_spa.append(loss_spa.item())
                epoch_loss_spec.append(loss_spec.item())
                epoch_total_loss.append(total_loss.item())


        # 计算平均损失
        if use_reduced_epoch:
            avg_loss_red = np.mean(epoch_loss_red)
            avg_total_loss = reduced_loss_weight * avg_loss_red
        else:
            avg_loss_var = np.mean(epoch_loss_var)
            avg_loss_spa = np.mean(epoch_loss_spa)
            avg_loss_spec = np.mean(epoch_loss_spec)
            # Use the loss actually differentiated, including teacher_scale.
            avg_total_loss = float(np.mean(epoch_total_loss))

        if not np.isfinite(avg_total_loss):
            raise RuntimeError(f"non-finite/empty training loss at epoch {epoch}")
        deltas = {"backbone_delta": 0.0, "head_gt_delta": 0.0, "head_full_delta": 0.0}
        if audit_updates:
            for name, parameter in model_student.named_parameters():
                delta = float((parameter.detach() - before[name]).abs().max())
                key = "backbone_delta" if name.startswith("backbone.") else "head_gt_delta" if name.startswith("head.0.") else "head_full_delta" if name.startswith("head.1.") else None
                if key:
                    deltas[key] = max(deltas[key], delta)
        row = dict(epoch=epoch, domain="gt" if use_reduced_epoch else "full", seed=SEED, total_loss=float(avg_total_loss), loss_gt=float(avg_loss_red) if use_reduced_epoch else "", loss_var="" if use_reduced_epoch else float(avg_loss_var), loss_spa="" if use_reduced_epoch else float(avg_loss_spa), loss_spec="" if use_reduced_epoch else float(avg_loss_spec), gt_updates=actual_updates["gt"], full_updates=actual_updates["full"], amp_skipped=actual_updates["full_amp_skipped"], **deltas)
        with open(trace_path, "a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=trace_fields).writerow(row)

        # 输出损失
        # full 损失每隔 full_log_interval(10) 个 epoch 打印一次；若该轮是 reduced 轮，
        # 则打印最近一个 full 轮的损失；reduced 损失每隔 reduced_log_interval 个 epoch 打印一次
        if use_reduced_epoch:
            if epoch % reduced_log_interval == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}] [REDUCED] - Loss_gt: {avg_loss_red:.6f}, "
                      f"Total Loss: {avg_total_loss:.6f}")
                if last_full_log is not None and (epoch % full_log_interval == 0 or epoch == epochs):
                    le, lv, ls, lsp, lt = last_full_log
                    print(f"Epoch [{epoch}/{epochs}] (full@epoch {le}) - Loss1(var): {lv:.6f}, "
                          f"Loss2(fspta): {ls:.6f}, Loss3(fspec): {lsp:.6f}, "
                          f"Total Loss: {lt:.6f}")
        else:
            last_full_log = (epoch, avg_loss_var, avg_loss_spa, avg_loss_spec, avg_total_loss)
            if epoch % full_log_interval == 0 or epoch == epochs:
                print(f"Epoch [{epoch}/{epochs}] - Loss1(var): {avg_loss_var:.6f}, "
                      f"Loss2(fspta): {avg_loss_spa:.6f}, Loss3(fspec): {avg_loss_spec:.6f}, "
                      f"Total Loss: {avg_total_loss:.6f}")
        
        # 保存最佳模型
        # full 与 reduced 的总损失量级相差约 7 个数量级，不能混在一起比较；
        # 否则 reduced 轮会永远刷新 best，存下"刚被 reduced 推离 full 最优方向"的模型。
        # best 只按 full 轮损失选择；纯 reduced 训练（无 full 轮）时按 reduced 损失兜底。
        if use_reduced_epoch:
            if avg_total_loss < min_red_loss:
                min_red_loss = avg_total_loss
                if min_full_loss == float("inf"):
                    save_checkpoint(model_student, f"{identifier}_best")
        else:
            if avg_total_loss < min_full_loss:
                min_full_loss = avg_total_loss
                save_checkpoint(model_student, f"{identifier}_best")

        # Evaluate every post-update epoch, including GT and AMP-skipped epochs.
        # This full FP32 forward is separate from training and consumes no RNG.
        if selector is not None:
            score = selector.update(model_student, epoch, 'GT' if use_reduced_epoch else 'FULL')
            print(f"Epoch [{epoch}/{epochs}] HQNR={score['HQNR']:.9f} "
                  f"Dl={score['Dl']:.9f} Ds={score['Ds']:.9f} "
                  f"best_hqnr_epoch={selector.best['epoch']}", flush=True)

    
    save_checkpoint(model_student, f"{identifier}_last")
    if actual_updates["full_amp_skipped"]:
        print(f"WARNING: AMP skipped {actual_updates['full_amp_skipped']} FR updates; epoch count is not effective update count", flush=True)
    last_path = os.path.join(model_dir, f"{identifier}_last.pth")
    if selector is not None:
        selector.finish()
    else:
        atomic_json(os.path.join(model_dir, f"{identifier}_last_selection.json"), {
            "sensor": sensor, "scene": data_id, "epoch_budget": epochs,
            "evaluated_epochs": epochs, "selection_metric": "final_epoch",
            "selection_rule": "last epoch only; no in-training HQNR",
            "patch_policy": "whole_image_internal_wald", "checkpoint": os.path.basename(last_path),
            "checkpoint_sha256": file_sha256(last_path), "completed": True,
            "actual_updates": actual_updates,
            "experiment_seed": SEED,
            "training_revision": "head_audit_20261028_v2",
            "training_trace": os.path.basename(trace_path),
            "amp_init_scale": amp_init_scale,
            "force_fp32": force_fp32,
            "scheduled_gt": int(sum(selected_schedule)),
            "scheduled_full": int(epochs - sum(selected_schedule)),
            "teacher_mat_root": os.environ.get("ZS_TEACHER_MAT_ROOT"),
            "precision": ("strict_full_fp32_gt_fp32" if not use_amp
                          else "frozen_mainline_full_amp_gt_fp32"),
            "teacher_weight_mode": teacher_weight_mode,
            "teacher_enabled": teacher_enabled})
    # 训练完成
    total_time = time.time() - start_time
    print(f"训练完成，总耗时 {total_time:.2f} 秒")
    print(f"最终最佳损失(full): {min_full_loss:.6f}")
    print(f"最终最佳损失(reduced): {min_red_loss:.6f}")
    if selector is not None:
        print(f"默认模型: {identifier}_best_hqnr.pth; 最佳轮次: {selector.best['epoch']}")
    else:
        print(f"默认模型: {identifier}_last.pth; 训练中未计算HQNR")
    print(f"损失对照: {identifier}_best.pth; 最终轮: {identifier}_last.pth")

# ================ 主函数 ================ #
def main():
    import h5py
    with h5py.File(data_path, 'r') as obs:
        if 'gt' in obs or not {'ms', 'lms', 'pan'}.issubset(obs.keys()):
            raise RuntimeError('外层目标输入必须只含ms/lms/pan，不得包含评估GT。')
        ms_hw = tuple(obs['ms'].shape[-2:])
        output_hw = tuple(obs['lms'].shape[-2:])
        if len(ms_hw) != 2 or ms_hw[0] != ms_hw[1] or any(v % ratio for v in ms_hw):
            raise RuntimeError(f'外层MS尺寸必须为可被{ratio}整除的正方形，实际为{ms_hw}。')
        if output_hw != (ratio * ms_hw[0], ratio * ms_hw[1]):
            raise RuntimeError(f'外层LMS必须为MS的{ratio}倍，实际MS={ms_hw}, LMS={output_hw}。')
        if tuple(obs['pan'].shape[-2:]) != output_hw:
            raise RuntimeError('外层PAN与LMS尺寸必须一致。')
        observation_ms = np.asarray(obs['ms'][:], dtype=np.float32)
    if use_reduced:
        with h5py.File(reduced_data_path, 'r') as wald:
            if not {'ms', 'lms', 'pan', 'gt'}.issubset(wald.keys()):
                raise RuntimeError('内部Wald文件缺少ms/lms/pan/gt。')
            inner_ms_hw = (ms_hw[0] // ratio, ms_hw[1] // ratio)
            if tuple(wald['ms'].shape[-2:]) != inner_ms_hw:
                raise RuntimeError(f'内部Wald MS必须为{inner_ms_hw}，实际为{tuple(wald["ms"].shape[-2:])}。')
            if any(tuple(wald[k].shape[-2:]) != ms_hw for k in ('lms', 'pan', 'gt')):
                raise RuntimeError(f'内部Wald LMS/PAN/GT必须为{ms_hw}。')
            if not np.array_equal(np.asarray(wald['gt'][:], dtype=np.float32), observation_ms):
                raise RuntimeError('内部Wald GT必须逐值等于同一目标场景的外层观测MS。')
    internal_protocol = (f'internal{ms_hw[0]}-gt-domain / no-patch'
                         if use_reduced else 'internal-wald-disabled')
    print(f'MAINLINE_PROTOCOL_PASS outer{output_hw[0]}-full-domain / '
          f'{internal_protocol} / no-evaluation-GT', flush=True)

    # 数据加载
    train_set = Dataset(data_path, data_id)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    # reduced 数据加载（预构造文件，避免每次训练重新降采样）
    reduced_loader = None
    if use_reduced:
        if os.path.exists(reduced_data_path):
            try:
                reduced_set = ReducedDataset(
                    reduced_data_path, data_id, patch_size=args.reduced_patch_size
                )
                reduced_loader = DataLoader(
                    dataset=reduced_set,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=True
                )
                print(f"已加载reduced数据: {reduced_data_path} (reduced比例: {reduced_ratio:.0f}%)")
            except Exception as e:
                raise RuntimeError('加载reduced数据失败；不允许静默改变已配置GT比例') from e
        else:
            raise FileNotFoundError(f'请先构造reduced数据，不会回退到纯无监督: {reduced_data_path}')
    
    # 模型标识（如果使用混合精度，则添加amp标记）
    identifier = f"{sensor}_{data_id}_FusionNet_SDE" 
    
    # 直接运行预训练FusionNet，生成当前场景的教师输出
    if teacher_enabled:
        teacher_output, teacher_source = load_teacher_output(
            data_id, device, data_path=data_path, sensor=sensor
        )
        print(f"成功导入教师模型输出: {teacher_source}")
    else:
        teacher_output, teacher_source = None, "disabled"
        print("教师损失已严格禁用：不加载教师输出，w_var=0", flush=True)

    # 开始训练
    train(train_loader, reduced_loader, identifier, teacher_output)

# 执行主函数
if __name__ == "__main__":
    main()
