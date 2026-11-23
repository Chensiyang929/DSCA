"""Run full-resolution DSCA and its input adapter comparisons."""
import json
import os
import shutil
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from protocol_guard import TRAINING_REVISION, guard_output_root

ROOT = Path(os.environ.get("ZSVD_EXPERIMENT_ROOT", Path(__file__).resolve().parent)).resolve()
CODE = ROOT / "candidate"
OUT = Path(os.environ.get(
    "GATED_OUTPUT_ROOT",
    ROOT / "experiments" / os.environ.get("GATED_EXPERIMENT_NAME", "gated_domain_all_sensors"),
))
WORK = OUT / "work"
PYTHON = Path(sys.executable)
PRETRAIN = Path(os.environ["DSCA_CHECKPOINT_DIR"]).resolve()
SCENES = [0, 5, 10, 15, 19]
ARMS = ["head"]
WORKERS = int(os.environ.get("GATED_WORKERS", "1"))
TRAINER = os.environ.get("GATED_TRAINER")
INFER_SCRIPT = os.environ.get("GATED_INFER")
SENSORS = {
    "WV3": dict(obs="wv3", wald="WV3", teacher="WV3", bands=8, pre="WV3.pth"),
    "WV2": dict(obs="wv2", wald="WV2", teacher="WV2", bands=8, pre="WV3.pth"),
    "QB": dict(obs="qb", wald="QB", teacher="QB", bands=4, pre="QB.pth"),
    "GF2": dict(obs="gf2", wald="GF2", teacher="GF2", bands=4, pre="QB.pth"),
}

# Load sensor-specific training settings from the shared benchmark configuration.
from mainline4_config import SENSORS as FROZEN
for _sensor, _spec in FROZEN.items():
    SENSORS[_sensor].update({k: _spec[k] for k in ("epochs", "cycle", "gt_lr", "full_lr", "weights")})


def paths(sensor, config):
    data = Path(os.environ["DSCA_DATA_ROOT"]).resolve()
    obs = Path(os.environ.get("DSCA_OBSERVATION_PATH", data / "FR" / sensor / "observations.h5"))
    wald = data / "FR" / sensor / "internal_wald.h5"
    teacher = data / "FR" / sensor / "teacher"
    sde = data / "FR" / sensor / "model_SDE"
    return obs, wald, teacher, sde


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2))
    os.replace(temp, path)


def call(command, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        return subprocess.run([str(x) for x in command], cwd=WORK, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode


def gate_statistics(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    stats = {}
    for name, value in state.items():
        if name.endswith(".alpha") or name.endswith("gate_logit") or name.endswith("gate_raw"):
            if name.endswith("gate_logit"):
                gate = torch.sigmoid(value.float())
            elif name.endswith("gate_raw"):
                gate = value.float().clamp(0.0, 1.0)
            else:
                gate = torch.tanh(value.float())
            stats[name] = {"mean_abs": float(gate.abs().mean()), "max_abs": float(gate.abs().max()), "values": gate.tolist()}
    return stats


def task(sensor, arm, scene):
    config = SENSORS[sensor]
    spec = FROZEN[sensor]
    obs, wald, teacher_dir, sde_dir = paths(sensor, config)
    folder = OUT / sensor / arm / f"scene{scene:02d}"
    done = folder / "complete.json"
    if done.exists():
        return json.loads(done.read_text())
    sde_source = sde_dir / f"{scene}_ms2pan_convNet_dual.pth"
    sde_local = WORK / "model_SDE" / sensor / sde_source.name
    sde_local.parent.mkdir(parents=True, exist_ok=True)
    if not sde_local.exists():
        shutil.copy2(sde_source, sde_local)
    gt_count, full_count = config["cycle"]
    teacher_weight, spatial_weight, spectral_weight = config["weights"]
    teacher_enabled = os.environ.get("DISABLE_TEACHER", "0") != "1"
    teacher_weight_mode = os.environ.get("TEACHER_WEIGHT_MODE", "legacy")
    experiment_seed = int(os.environ.get("EXPERIMENT_SEED", "10"))
    teacher_id = os.environ.get("GATED_TEACHER_ID", "default")
    teacher_pattern = os.environ.get("GATED_TEACHER_PATTERN")
    teacher_file = (Path(teacher_pattern.format(
        sensor=sensor,
        sensor_lower=sensor.lower(),
        scene=scene,
        scene02=f"{scene:02d}",
    )) if teacher_pattern else teacher_dir / f"{scene}_student_SDE_0.15.mat")
    if teacher_enabled and not teacher_file.is_file():
        raise FileNotFoundError(f"teacher output not found: {teacher_file}")
    env = dict(
        os.environ,
        MKL_THREADING_LAYER="SEQUENTIAL",
        PYTHONUTF8="1",
        WALD_DEVICE="cuda:0",
        EXACT_GT=str(gt_count),
        EXACT_FULL=str(full_count),
        EXACT_SCHEDULE_MODE=os.environ.get("EXACT_SCHEDULE_MODE", "interleaved"),
        MAINLINE2_GT_LR=str(config["gt_lr"]),
        MAINLINE2_FULL_LR=str(config["full_lr"]),
        PRUNED_PRETRAIN=str(PRETRAIN / config["pre"]),
        SAMPLING_TEACHER_FILE=str(teacher_file),
        TEACHER_WEIGHT_MODE=teacher_weight_mode,
        EXPERIMENT_SEED=str(experiment_seed),
        DISABLE_TEACHER="0" if teacher_enabled else "1",
        EXACT_STUDENT_TRAINER=TRAINER or spec["trainer"],
        WRAP_VARIANT=arm,
        WRAP_BACKBONE_LR_SCALE="1.0",
        WRAP_ADAPTER_LR_SCALE="1.0",
        WRAP_CONTROL_LR_SCALE=os.environ.get("WRAP_CONTROL_LR_SCALE", str(spec["control_lr_scale"])),
        WRAP_MAX_GAIN="1.0",
    )
    env.pop("PYTHONNOUSERSITE", None)
    model_dir = folder / "models"
    checkpoint = model_dir / f"{sensor}_{scene}_FusionNet_SDE_last.pth"
    selection_path = checkpoint.with_name(checkpoint.stem + "_selection.json")
    selection = None
    if checkpoint.is_file() and selection_path.is_file():
        candidate_selection = json.loads(selection_path.read_text())
        if candidate_selection.get("completed") and candidate_selection.get("epoch_budget") == config["epochs"]:
            selection = candidate_selection
    if selection is None:
        command = [PYTHON, "-u", CODE / "train_exact_mix.py", "--sensor", sensor, "--bands", config["bands"], "--data_id", scene,
             "--data_path", obs, "--epochs", config["epochs"], "--lr", config["full_lr"],
             "--w_var", teacher_weight if teacher_enabled else 0,
             "--w_spa", spatial_weight, "--w_spec", spectral_weight,
             "--selection_mode", "last", "--model_dir", model_dir,
             "--use_reduced", "1" if gt_count > 0 else "0"]
        if gt_count > 0:
            command.extend(["--reduced_data_path", wald])
        rc = call(command, folder / "train.log", env)
        if not checkpoint.is_file() or not selection_path.is_file():
            raise RuntimeError((sensor, arm, scene, "train", rc))
        selection = json.loads(selection_path.read_text())
        if rc and not selection.get("completed"):
            raise RuntimeError((sensor, arm, scene, "train", rc))
        if rc:
            (folder / "postsave_nonzero_exit.txt").write_text(
                f"trainer exit code {rc}; accepted only after completed last-checkpoint audit\n"
            )
    expected_gt = sum(epoch % (gt_count + full_count) < gt_count for epoch in range(config["epochs"]))
    assert selection["actual_updates"]["gt"] == expected_gt
    if env.get("FORCE_FP32") == "1":
        expected = {"gt": expected_gt, "full": config["epochs"] - expected_gt, "full_amp_skipped": 0}
        if selection["actual_updates"] != expected:
            raise RuntimeError(f"Actual optimizer updates differ from the schedule: {selection['actual_updates']}")
    # Internal-Wald inference is evaluation-only when gt_count == 0.  The
    # strict Full-only training command above still receives no reduced path.
    outputs = [("full", obs, "full"), ("internal_wald", wald, "gt")]
    for name, data, domain in outputs:
        destination = folder / f"{name}.mat"
        rc = call(
            [PYTHON, CODE / (INFER_SCRIPT or spec["infer"]), "--sensor", sensor, "--scene", scene,
             "--data_path", data, "--checkpoint", checkpoint, "--output", destination, "--variant", arm,
             "--domain", domain],
            folder / f"infer_{name}.log", env,
        )
        if rc or not np.isfinite(loadmat(destination, variable_names=["proposed"])["proposed"]).all():
            raise RuntimeError((sensor, arm, scene, "infer", name, rc))
    record = {"sensor": sensor, "arm": arm, "scene": scene,
              "training_internal_wald": gt_count > 0,
              "evaluation_internal_wald": True,
              "teacher_enabled": teacher_enabled,
              "teacher_id": teacher_id if teacher_enabled else "disabled",
              "teacher_weight_mode": teacher_weight_mode,
              "experiment_seed": experiment_seed,
              "teacher_file": str(teacher_file) if teacher_enabled else None,
              "actual_updates": selection["actual_updates"],
              "gates": gate_statistics(checkpoint)}
    atomic(done, record)
    return record


def main():
    if WORKERS < 1:
        raise ValueError("GATED_WORKERS must be positive")
    for sensor, config in SENSORS.items():
        obs, wald, _, _ = paths(sensor, config)
        with h5py.File(obs) as observed:
            observation_ms = np.asarray(observed["ms"][:], np.float32)
        if config["cycle"][0] > 0:
            with h5py.File(wald) as reduced:
                assert np.array_equal(np.asarray(reduced["gt"][:], np.float32), observation_ms)
    jobs = [(sensor, arm, scene) for sensor in SENSORS for arm in ARMS for scene in SCENES]
    protocol = {"sensors": SENSORS, "arms": ARMS, "scenes": SCENES, "total": len(jobs), "workers": WORKERS,
                                          "selection": "last", "max_gain": 1.0, "identity_at_initialization": True,
                                          "schedule_mode": os.environ.get("EXACT_SCHEDULE_MODE", "interleaved"),
                                          "teacher_enabled": os.environ.get("DISABLE_TEACHER", "0") != "1",
                                          "teacher_id": os.environ.get("GATED_TEACHER_ID", "default"),
                                          "teacher_weight_mode": os.environ.get("TEACHER_WEIGHT_MODE", "legacy"),
                                          "experiment_seed": int(os.environ.get("EXPERIMENT_SEED", "10")),
                                          "teacher_pattern": os.environ.get("GATED_TEACHER_PATTERN"),
                                          "training_revision": TRAINING_REVISION,
                                          "force_fp32": os.environ.get("FORCE_FP32", "0") == "1",
                                          "amp_init_scale": float(os.environ.get("AMP_INIT_SCALE", "65536.0")),
                                          "inputs": {s: [str(p) for p in paths(s, cfg)] for s, cfg in SENSORS.items()},
                                          "control_lr_scales": {s: float(os.environ.get("WRAP_CONTROL_LR_SCALE", FROZEN[s]["control_lr_scale"])) for s in SENSORS},
                                          "trainers": {s: TRAINER or FROZEN[s]["trainer"] for s in SENSORS}}
    guard_output_root(OUT, protocol)
    atomic(OUT / "protocol.json", protocol)
    complete, errors = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(task, *job): job for job in jobs}
        for future in as_completed(futures):
            try:
                complete.append(future.result())
            except Exception as error:
                errors.append({"job": futures[future], "error": repr(error)})
            atomic(OUT / "status.json", {"pid": os.getpid(), "phase": "training", "completed": len(complete), "total": len(jobs), "errors": errors})
    atomic(OUT / "records.json", complete)
    atomic(OUT / "status.json", {"pid": os.getpid(), "phase": "complete" if not errors else "failed", "completed": len(complete),
                                      "total": len(jobs), "errors": errors, "evaluation_complete": False})
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
