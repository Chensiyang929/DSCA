"""Direct pretrained FusionNet teacher execution."""
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_CHECKPOINTS = {
    "WV3": PROJECT_ROOT / "teacher" / "weights" / "WV3.pth",
    "WV2": PROJECT_ROOT / "teacher" / "weights" / "WV3.pth",
    "QB": PROJECT_ROOT / "teacher" / "weights" / "QB.pth",
    "GF2": PROJECT_ROOT / "teacher" / "weights" / "QB.pth",
}


def load_teacher_output(data_id, device, data_path=None, checkpoint_path=None, sensor="WV3"):
    """Run the pretrained ERFT FusionNet teacher for one H5 sample.

    Returns ``(teacher_output, checkpoint_path)`` with output shape
    ``[1, C, H, W]`` on the same normalized scale as the training inputs.
    """
    if data_path is None:
        raise ValueError("data_path is required to run the FusionNet teacher")

    sensor = sensor.upper()
    if sensor not in DEFAULT_TEACHER_CHECKPOINTS:
        raise ValueError(f"unsupported teacher sensor mapping: {sensor}")
    checkpoint_path = checkpoint_path or os.environ.get(
        "FUSIONNET_TEACHER_CHECKPOINT", str(DEFAULT_TEACHER_CHECKPOINTS[sensor])
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"FusionNet teacher checkpoint not found: {checkpoint_path}"
        )
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from teacher.fusionnet import FusionNet

    norm = 1023.0 if sensor == "GF2" else 2047.0
    with h5py.File(data_path, "r") as data:
        lms = np.asarray(data["lms"][data_id], dtype=np.float32) / norm
        pan = np.asarray(data["pan"][data_id], dtype=np.float32) / norm

    lms_t = torch.from_numpy(lms).unsqueeze(0).to(device)
    pan_t = torch.from_numpy(pan).unsqueeze(0).to(device)
    if pan_t.ndim == 3:
        pan_t = pan_t.unsqueeze(1)

    teacher = FusionNet(spectral_num=lms_t.shape[1]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        checkpoint = {
            key.removeprefix("module."): value for key, value in checkpoint.items()
        }
    teacher.load_state_dict(checkpoint)
    teacher.eval()
    with torch.no_grad():
        output = teacher(lms_t, pan_t)
    return output, checkpoint_path
