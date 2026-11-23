"""Load a precomputed full-resolution teacher prediction."""
import os
from pathlib import Path

import numpy as np
import h5py
import torch
from scipy.io import loadmat


def direct_teacher(data_id, device, data_path=None, checkpoint_path=None, sensor="WV3"):
    source = Path(os.environ["SAMPLING_TEACHER_FILE"])
    content = loadmat(source, variable_names=["proposed", "sr"])
    key = "proposed" if "proposed" in content else "sr" if "sr" in content else None
    if key is None:
        raise KeyError(f"Teacher MAT must contain proposed or sr: {source}")
    value = np.asarray(content[key], dtype=np.float32)
    if value.ndim != 3 or not np.isfinite(value).all():
        raise ValueError(f"Invalid full-resolution teacher output: {source}")
    if data_path is not None:
        with h5py.File(data_path, "r") as observations:
            channels, height, width = observations["lms"].shape[1:]
        if value.shape != (height, width, channels):
            raise ValueError(f"Teacher/current-scene shape mismatch: {value.shape} vs {(height, width, channels)}: {source}")
    norm = 1023 if sensor.upper() == "GF2" else 2047
    scale = norm if float(np.nanmax(value)) > 2.0 else 1.0
    tensor = torch.from_numpy(value.transpose(2, 0, 1).copy() / scale).unsqueeze(0)
    return tensor.to(device), str(source)
