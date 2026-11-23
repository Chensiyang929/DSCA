"""Adapt DSCA to full-resolution observations."""
import argparse
import os
from pathlib import Path

from mainline4_config import SENSORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sensor", choices=sorted(SENSORS))
    parser.add_argument("--adapter", choices=("head", "shared", "base"), default="head",
                        help="Dual adapters, shared adapter, or no adapter")
    parser.add_argument("--scenes", default="all", help="all or comma-separated zero-based scene IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--teacher-pattern", help="Override the Full teacher MAT path pattern")
    args = parser.parse_args()
    for key in ("DSCA_DATA_ROOT", "DSCA_CHECKPOINT_DIR"):
        if not os.environ.get(key):
            parser.error(f"Set {key} before running the experiment")
    if args.scenes == "all":
        scenes = list(range(20))
    else:
        try:
            scenes = [int(value) for value in args.scenes.split(",")]
        except ValueError:
            parser.error("--scenes must be 'all' or a comma-separated list of integers")
        if len(set(scenes)) != len(scenes) or any(not 0 <= scene < 20 for scene in scenes):
            parser.error("--scenes must contain unique IDs between 0 and 19")
    if args.seed < 0 or args.workers < 1:
        parser.error("Use a non-negative seed and at least one worker")
    spec = SENSORS[args.sensor]
    os.environ.update(
        ZSVD_EXPERIMENT_ROOT=str(Path(__file__).resolve().parent),
        GATED_OUTPUT_ROOT=str(args.output_root.resolve()),
        GATED_WORKERS=str(args.workers),
        GATED_TRAINER=spec["trainer"],
        GATED_INFER=spec["infer"],
        WRAP_CONTROL_LR_SCALE=str(spec["control_lr_scale"]),
        WRAP_INITIAL_GATE="0.95",
        EXACT_SCHEDULE_MODE="interleaved",
        TEACHER_WEIGHT_MODE="legacy",
        DISABLE_TEACHER="0",
        EXPERIMENT_SEED=str(args.seed),
        GATED_TEACHER_ID=spec["teacher"],
        FORCE_FP32="1",
        AMP_INIT_SCALE="65536.0",
    )
    if args.teacher_pattern:
        os.environ["GATED_TEACHER_PATTERN"] = args.teacher_pattern
    else:
        os.environ.pop("GATED_TEACHER_PATTERN", None)
    import run_gated_domain_all_sensors as runner

    runner.SENSORS = {args.sensor: runner.SENSORS[args.sensor]}
    runner.ARMS = [args.adapter]
    runner.SCENES = scenes
    runner.main()


if __name__ == "__main__":
    main()
