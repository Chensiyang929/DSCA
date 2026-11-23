"""Build leakage-safe per-scene target-adaptation inputs from official RR H5.

The output observation file contains only MS64/LMS256/PAN256.  The evaluation
GT256 is deliberately not copied.  A second H5 contains the one-level
Internal-Wald pair MS16/LMS64/PAN64 -> MS64.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
ZUP = ROOT / "candidate" / "ZSVD-test-for-mix" / "zup"
sys.path.insert(0, str(ZUP))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_official_rr(source, bands, ratio):
    # Deliberately do not require, inspect or read the evaluation-GT dataset.
    required = {"ms", "lms", "pan"}
    if not required.issubset(source.keys()):
        raise RuntimeError(f"official RR must contain {sorted(required)}")
    ms_shape = source["ms"].shape
    lms_shape = source["lms"].shape
    pan_shape = source["pan"].shape
    if ms_shape[1] != bands or lms_shape[1] != bands:
        raise RuntimeError("official RR band count does not match the sensor")
    if pan_shape[1] != 1:
        raise RuntimeError("PAN must contain exactly one band")
    if pan_shape[-2:] != lms_shape[-2:]:
        raise RuntimeError("official RR LMS and PAN must share output size")
    if lms_shape[-2:] != (ratio * ms_shape[-2], ratio * ms_shape[-1]):
        raise RuntimeError("official RR output size must be four times its MS size")
    if ms_shape[-2] % ratio or ms_shape[-1] % ratio:
        raise RuntimeError("official RR MS size must be divisible by four")
    return ms_shape[0]


def write_observations(source, destination):
    with h5py.File(destination, "w") as output:
        for key in ("ms", "lms", "pan"):
            output.create_dataset(key, data=np.asarray(source[key], dtype=np.float32))
        output.attrs["protocol"] = "official_rr_target_observation_without_evaluation_gt"
        output.attrs["contains_evaluation_gt"] = False


def write_internal_wald(source, destination, sensor, ratio, device):
    os.environ["WALD_DEVICE"] = device
    from wald_utilities import wald_protocol_v1, wald_protocol_v2

    norm = 1023.0 if sensor == "GF2" else 2047.0
    count = source["ms"].shape[0]
    inner_ms, inner_lms, inner_pan, inner_gt = [], [], [], []
    target_device = torch.device(device)
    with torch.inference_mode():
        for scene in range(count):
            # Evaluation GT is intentionally never read in this function.
            observed_ms = np.asarray(source["ms"][scene:scene + 1], np.float32)
            ms = torch.from_numpy(observed_ms / norm).to(target_device)
            lms = torch.from_numpy(np.asarray(source["lms"][scene:scene + 1], np.float32) / norm).to(target_device)
            pan = torch.from_numpy(np.asarray(source["pan"][scene:scene + 1], np.float32) / norm).to(target_device)
            ms16 = wald_protocol_v1(ms, pan, ratio, sensor, channels=ms.shape[1])
            lms64 = wald_protocol_v1(lms, pan, ratio, sensor, channels=lms.shape[1])
            pan64 = wald_protocol_v2(ms, pan, ratio, sensor, channels=ms.shape[1])
            inner_ms.append(ms16.detach().cpu().numpy() * norm)
            inner_lms.append(lms64.detach().cpu().numpy() * norm)
            inner_pan.append(pan64.detach().cpu().numpy() * norm)
            # Copy the raw observation so the pseudo-GT is bitwise identical;
            # avoid a divide/multiply round trip that changes float32 LSBs.
            inner_gt.append(observed_ms.copy())
            print(f"{sensor} scene{scene:02d}: built MS16/LMS64/PAN64 -> MS64", flush=True)

    with h5py.File(destination, "w") as output:
        output.create_dataset("ms", data=np.concatenate(inner_ms).astype(np.float32))
        output.create_dataset("lms", data=np.concatenate(inner_lms).astype(np.float32))
        output.create_dataset("pan", data=np.concatenate(inner_pan).astype(np.float32))
        output.create_dataset("gt", data=np.concatenate(inner_gt).astype(np.float32))
        output.attrs["protocol"] = "one_level_internal_wald_from_current_rr_observation"
        output.attrs["target_is_outer_observed_ms"] = True
        output.attrs["contains_official_evaluation_gt"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["WV3", "WV2", "QB", "GF2"])
    parser.add_argument("--official-rr", required=True, help="Official RR H5 containing ms/lms/pan/gt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--skip-internal-wald", action="store_true",
                        help="Create only ms/lms/pan for strict Full-only")
    args = parser.parse_args()
    if args.ratio != 4:
        parser.error("Mainline-4 currently freezes the sensor ratio at 4")

    sensor = args.sensor.upper()
    bands = 8 if sensor in {"WV3", "WV2"} else 4
    official = Path(args.official_rr).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observations = output_dir / f"{sensor}_rr_observations_only.h5"
    internal = output_dir / f"{sensor}_rr_internal_wald.h5"

    with h5py.File(official, "r") as source:
        count = validate_official_rr(source, bands, args.ratio)
        write_observations(source, observations)
        if not args.skip_internal_wald:
            write_internal_wald(source, internal, sensor, args.ratio, args.device)

    with h5py.File(observations, "r") as obs:
        if "gt" in obs:
            raise RuntimeError("leakage guard failed: observation file contains evaluation GT")
        if not args.skip_internal_wald:
            with h5py.File(internal, "r") as wald:
                if not np.array_equal(np.asarray(obs["ms"]), np.asarray(wald["gt"])):
                    raise RuntimeError("Internal-Wald target is not exactly the current target observation MS")

    manifest = {
        "sensor": sensor,
        "scenes": count,
        "ratio": args.ratio,
        "official_rr_source": str(official),
        "official_rr_sha256": sha256(official),
        "observations_only": str(observations),
        "observations_only_sha256": sha256(observations),
        "internal_wald": None if args.skip_internal_wald else str(internal),
        "internal_wald_sha256": None if args.skip_internal_wald else sha256(internal),
        "training_keys": ["ms", "lms", "pan"],
        "evaluation_gt_copied_to_training_files": False,
        "routing": ({"outer": "full"} if args.skip_internal_wald else
                    {"outer": "full", "internal_wald": "gt"}),
    }
    (output_dir / f"{sensor}_rr_adaptation_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
