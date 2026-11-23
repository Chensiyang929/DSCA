"""Adapt DSCA independently to each official RR target scene.

Training sees only the current scene's MS64/LMS256/PAN256 outer observation
and its derived MS16/LMS64/PAN64 -> MS64 Internal-Wald pair.  Official GT256
is never passed to this process.  The outer branch and final prediction use
domain="full"; the derived Internal-Wald branch uses domain="gt".
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat

from mainline4_config import SENSORS
from protocol_guard import TRAINING_REVISION, guard_output_root


default_root = os.environ.get("ZSVD_EXPERIMENT_ROOT", Path(__file__).resolve().parent)
ROOT = Path(os.environ.get("DSCA_EXPERIMENT_ROOT", default_root)).resolve()
CODE = ROOT / "candidate"
PRETRAIN = Path(os.environ.get("DSCA_CHECKPOINT_DIR", "checkpoints")).resolve()
PYTHON = Path(sys.executable)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def run_logged(command, log, env=None, cwd=ROOT):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            [str(item) for item in command],
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def parse_scenes(value, count):
    if value == "all":
        return list(range(count))
    scenes = sorted({int(item) for item in value.split(",") if item.strip()})
    if not scenes or scenes[0] < 0 or scenes[-1] >= count:
        raise ValueError(f"scene list must be within [0, {count - 1}]")
    return scenes


def render_pattern(pattern, sensor, scene, teacher):
    return Path(pattern.format(
        sensor=sensor,
        sensor_lower=sensor.lower(),
        scene=scene,
        scene02=f"{scene:02d}",
        teacher=teacher,
    ))


def check_teacher(path, bands, output_hw):
    if not path.is_file():
        raise FileNotFoundError(f"current-scene RR teacher output not found: {path}")
    content = loadmat(path, variable_names=["proposed", "sr"])
    key = "proposed" if "proposed" in content else "sr" if "sr" in content else None
    if key is None:
        raise KeyError(f"teacher MAT has neither proposed nor sr: {path}")
    value = np.asarray(content[key])
    if value.shape != (*output_hw, bands) or not np.isfinite(value).all():
        raise ValueError(
            f"teacher must match current RR target {(*output_hw, bands)}, got {value.shape}: {path}"
        )


def gate_statistics(checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    statistics = {}
    for name, value in state.items():
        if name.endswith((".alpha", "gate_logit", "gate_raw")):
            if name.endswith("gate_logit"):
                gate = torch.sigmoid(value.float())
            elif name.endswith("gate_raw"):
                gate = value.float().clamp(0.0, 1.0)
            else:
                gate = torch.tanh(value.float())
            statistics[name] = {
                "mean_abs": float(gate.abs().mean()),
                "max_abs": float(gate.abs().max()),
                "values": gate.tolist(),
            }
    return statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sensor", choices=sorted(SENSORS))
    parser.add_argument("--official-rr", default=None,
                        help="Required only when prepared observation/Wald files must be built")
    parser.add_argument(
        "--teacher-pattern",
        required=True,
        help=("Current-RR-scene teacher MAT pattern. Available fields: "
              "{teacher}, {sensor}, {sensor_lower}, {scene}, {scene02}."),
    )
    parser.add_argument("--teacher-id", default=None,
                        help="Audited teacher label stored in every record")
    parser.add_argument(
        "--sde-root",
        required=True,
        help="Root containing SENSOR/<scene>_ms2pan_convNet_dual.pth for the same RR scenes",
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--prepared-root", default=None)
    parser.add_argument("--scenes", default="all", help="all or comma-separated zero-based IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-prepared", action="store_true")
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be non-negative")

    sensor = args.sensor.upper()
    spec = dict(SENSORS[sensor])
    os.environ["FORCE_FP32"] = "1"
    os.environ["WRAP_INITIAL_GATE"] = "0.95"
    args.schedule_mode = "interleaved"
    args.teacher_weight_mode = "legacy"
    args.disable_teacher = False
    teacher_id = args.teacher_id or spec["teacher"]
    gt_count, full_count = spec["cycle"]
    uses_internal_wald = gt_count > 0
    official_rr = Path(args.official_rr).resolve() if args.official_rr else None
    output_root = Path(args.output_root or ROOT / "experiments" / "mainline4_rr_target_adaptation").resolve() / sensor
    prepared_root = Path(args.prepared_root or output_root / "prepared").resolve()
    observations = prepared_root / f"{sensor}_rr_observations_only.h5"
    internal_wald = prepared_root / f"{sensor}_rr_internal_wald.h5"

    prepared_ready = observations.is_file() and (internal_wald.is_file() or not uses_internal_wald)
    if not args.reuse_prepared or not prepared_ready:
        if official_rr is None:
            parser.error("--official-rr is required unless --reuse-prepared points to complete files")
        command = [
            PYTHON, ROOT / "prepare_rr_target_adaptation.py",
            "--sensor", sensor,
            "--official-rr", official_rr,
            "--output-dir", prepared_root,
            "--device", args.device,
        ]
        if not uses_internal_wald:
            command.append("--skip-internal-wald")
        rc = run_logged(command, output_root / "prepare.log", dict(os.environ, WALD_DEVICE=args.device))
        if rc:
            raise RuntimeError(f"RR target preparation failed with exit code {rc}")

    with h5py.File(observations, "r") as observed:
        if "gt" in observed:
            raise RuntimeError("leakage guard: observations-only H5 contains GT")
        count = observed["ms"].shape[0]
        ms_hw = tuple(observed["ms"].shape[-2:])
        output_hw = tuple(observed["lms"].shape[-2:])
        bands = observed["ms"].shape[1]
        if output_hw != (4 * ms_hw[0], 4 * ms_hw[1]):
            raise RuntimeError("outer RR observation does not have ratio 4")
        observation_ms = np.asarray(observed["ms"])
    if uses_internal_wald:
        with h5py.File(internal_wald, "r") as reduced:
            if not np.array_equal(np.asarray(reduced["gt"]), observation_ms):
                raise RuntimeError("Internal-Wald GT must exactly equal current target MS")

    scenes = parse_scenes(args.scenes, count)
    teacher_weight, spatial_weight, spectral_weight = spec["weights"]
    sde_root = Path(args.sde_root).resolve()
    work = output_root / "work"

    def task(scene):
        folder = output_root / f"scene{scene:02d}"
        done = folder / "complete.json"
        if done.is_file():
            return json.loads(done.read_text(encoding="utf-8"))

        teacher_file = render_pattern(args.teacher_pattern, sensor, scene, teacher_id)
        if not args.disable_teacher:
            check_teacher(teacher_file, bands, output_hw)
        sde_source = sde_root / sensor / f"{scene}_ms2pan_convNet_dual.pth"
        if not sde_source.is_file():
            raise FileNotFoundError(f"current-scene RR SDE checkpoint not found: {sde_source}")
        sde_local = work / "model_SDE" / sensor / sde_source.name
        sde_local.parent.mkdir(parents=True, exist_ok=True)
        if not sde_local.is_file():
            shutil.copy2(sde_source, sde_local)

        env = dict(
            os.environ,
            MKL_THREADING_LAYER="SEQUENTIAL",
            PYTHONUTF8="1",
            WALD_DEVICE=args.device,
            EXACT_GT=str(gt_count),
            EXACT_FULL=str(full_count),
            EXACT_SCHEDULE_MODE=args.schedule_mode,
            TEACHER_WEIGHT_MODE=args.teacher_weight_mode,
            DISABLE_TEACHER="1" if args.disable_teacher else "0",
            EXPERIMENT_SEED=str(args.seed),
            MAINLINE2_GT_LR=str(spec["gt_lr"]),
            MAINLINE2_FULL_LR=str(spec["full_lr"]),
            PRUNED_PRETRAIN=str(PRETRAIN / ("WV3.pth" if bands == 8 else "QB.pth")),
            SAMPLING_TEACHER_FILE=str(teacher_file),
            EXACT_STUDENT_TRAINER=spec["trainer"],
            WRAP_VARIANT=spec["adapter"],
            WRAP_BACKBONE_LR_SCALE="1.0",
            WRAP_ADAPTER_LR_SCALE="1.0",
            WRAP_CONTROL_LR_SCALE=str(spec["control_lr_scale"]),
            WRAP_MAX_GAIN="1.0",
        )
        env.pop("PYTHONNOUSERSITE", None)
        model_dir = folder / "models"
        checkpoint = model_dir / f"{sensor}_{scene}_FusionNet_SDE_last.pth"
        selection_path = checkpoint.with_name(checkpoint.stem + "_selection.json")
        train_command = [
                PYTHON, CODE / "train_exact_mix.py",
                "--sensor", sensor,
                "--bands", bands,
                "--data_id", scene,
                "--data_path", observations,
                "--epochs", spec["epochs"],
                "--lr", spec["full_lr"],
                "--w_var", 0 if args.disable_teacher else teacher_weight,
                "--w_spa", spatial_weight,
                "--w_spec", spectral_weight,
                "--selection_mode", "last",
                "--model_dir", model_dir,
                "--device", args.device,
                "--use_reduced", "1" if uses_internal_wald else "0",
            ]
        if uses_internal_wald:
            train_command.extend(["--reduced_data_path", internal_wald])
        selection = None
        if checkpoint.is_file() and selection_path.is_file():
            candidate = json.loads(selection_path.read_text(encoding="utf-8"))
            if candidate.get("completed") and candidate.get("epoch_budget") == spec["epochs"]:
                selection = candidate
        if selection is None:
            rc = run_logged(
                train_command,
                folder / "train.log",
                env,
                work,
            )
            if not checkpoint.is_file() or not selection_path.is_file():
                raise RuntimeError((sensor, scene, "train", rc, "missing last checkpoint"))
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if rc and not selection.get("completed"):
                raise RuntimeError((sensor, scene, "train", rc))
            if rc:
                (folder / "postsave_nonzero_exit.txt").write_text(
                    f"trainer exit code {rc}; accepted only after completed last-checkpoint audit\n",
                    encoding="utf-8",
                )
        expected_gt = sum(epoch % (gt_count + full_count) < gt_count
                          for epoch in range(spec["epochs"]))
        expected = {"gt": expected_gt, "full": spec["epochs"] - expected_gt, "full_amp_skipped": 0}
        if selection["actual_updates"] != expected:
            raise RuntimeError(f"Actual optimizer updates differ from the schedule: {selection['actual_updates']}")

        outputs = [("adapted_rr", observations, "full")]
        if uses_internal_wald:
            outputs.append(("internal_wald", internal_wald, "gt"))
        for name, data_path, domain in outputs:
            destination = folder / f"{name}.mat"
            rc = run_logged(
                [
                    PYTHON, CODE / spec["infer"],
                    "--sensor", sensor,
                    "--scene", scene,
                    "--data_path", data_path,
                    "--checkpoint", checkpoint,
                    "--output", destination,
                    "--variant", spec["adapter"],
                    "--domain", domain,
                    "--device", args.device,
                ],
                folder / f"infer_{name}.log",
                env,
            )
            if rc:
                raise RuntimeError((sensor, scene, "infer", name, rc))
            result = loadmat(destination, variable_names=["proposed", "gt"])
            if "proposed" not in result or not np.isfinite(result["proposed"]).all():
                raise RuntimeError((sensor, scene, "invalid prediction", name))
            if name == "adapted_rr" and "gt" in result:
                raise RuntimeError("leakage guard: final RR prediction MAT unexpectedly contains GT")

        record = {
            "sensor": sensor,
            "scene": scene,
            "protocol": "independent_current_scene_rr_target_adaptation",
            "outer_input": f"MS{ms_hw[0]}/LMS{output_hw[0]}/PAN{output_hw[0]}",
            "internal_wald": (f"MS{ms_hw[0] // 4}/LMS{ms_hw[0]}/PAN{ms_hw[0]}->MS{ms_hw[0]}"
                              if uses_internal_wald else None),
            "routing": ({"outer_and_final": "full", "internal_wald": "gt"}
                        if uses_internal_wald else {"outer_and_final": "full"}),
            "evaluation_gt_seen": False,
            "teacher_file": None if args.disable_teacher else str(teacher_file),
            "teacher_id": "disabled" if args.disable_teacher else teacher_id,
            "experiment_seed": args.seed,
            "sde_checkpoint": str(sde_source),
            "actual_updates": selection["actual_updates"],
            "gates": gate_statistics(checkpoint),
        }
        atomic_json(done, record)
        return record

    protocol = {
        "training_revision": TRAINING_REVISION,
        "spec": spec,
        "force_fp32": os.environ.get("FORCE_FP32", "0") == "1",
        "amp_init_scale": float(os.environ.get("AMP_INIT_SCALE", "65536.0")),
        "sensor": sensor,
        "scenes": scenes,
        "official_rr_source_used_only_by_preparation": str(official_rr) if official_rr else None,
        "training_observations": str(observations),
        "training_internal_wald": str(internal_wald) if uses_internal_wald else None,
        "teacher_pattern": args.teacher_pattern,
        "teacher_id": teacher_id,
        "experiment_seed": args.seed,
        "teacher_weight_mode": args.teacher_weight_mode,
        "teacher_enabled": not args.disable_teacher,
        "sde_root": str(sde_root),
        "routing": ({"outer_and_final": "full", "internal_wald": "gt"}
                    if uses_internal_wald else {"outer_and_final": "full"}),
        "use_reduced": uses_internal_wald,
        "selection": "last",
        "cycle": [gt_count, full_count],
        "schedule_mode": args.schedule_mode,
        "epochs": spec["epochs"],
        "evaluation_gt_used_during_training": False,
    }
    guard_output_root(output_root, protocol)
    atomic_json(output_root / "protocol.json", protocol)

    complete, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(task, scene): scene for scene in scenes}
        for future in as_completed(futures):
            scene = futures[future]
            try:
                complete.append(future.result())
            except Exception as error:
                errors.append({"scene": scene, "error": repr(error)})
            atomic_json(output_root / "status.json", {
                "phase": "training",
                "completed": len(complete),
                "total": len(scenes),
                "errors": errors,
            })

    atomic_json(output_root / "records.json", complete)
    atomic_json(output_root / "status.json", {
        "phase": "complete" if not errors else "failed",
        "completed": len(complete),
        "total": len(scenes),
        "errors": errors,
        "evaluation_gt_opened": False,
    })
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
