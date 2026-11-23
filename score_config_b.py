"""Score FR, official RR, or FR adapter outputs using the paper's fixed gain."""
import argparse
import subprocess
from pathlib import Path

from mainline4_config import FIXED_ALPHA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fr", "rr", "fr-adapter"))
    parser.add_argument("sensor", choices=sorted(FIXED_ALPHA))
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="Official RR reference-pair directory")
    parser.add_argument("--adapter", choices=("head", "shared", "base"), default="head")
    parser.add_argument("--scenes", default="all", help="all or comma-separated zero-based scene IDs")
    args = parser.parse_args()
    if args.mode == "rr" and args.pairs is None:
        parser.error("RR scoring requires --pairs")
    if args.mode != "fr-adapter" and args.adapter != "head":
        parser.error("Use fr-adapter mode to score the base and shared controls")
    scenes = ",".join(map(str, range(20))) if args.scenes == "all" else args.scenes
    try:
        ids = [int(value) for value in scenes.split(",")]
    except ValueError:
        parser.error("--scenes must be 'all' or a comma-separated list of integers")
    if len(set(ids)) != len(ids) or any(not 0 <= scene < 20 for scene in ids):
        parser.error("--scenes must contain unique IDs between 0 and 19")
    code = Path(__file__).resolve().parent
    result = str(args.result_root.resolve())
    alpha = repr(FIXED_ALPHA[args.sensor])
    script, prefix = {
        "fr": ("evaluate_mtf_fixed_scalar_v1.m", [result, args.sensor]),
        "rr": ("evaluate_official_rr_mtf_only_v1.m",
               [result, str(args.pairs.resolve()) if args.pairs else "", args.sensor]),
        "fr-adapter": ("evaluate_adapter_fr.m", [result, args.sensor, args.adapter]),
    }[args.mode]
    subprocess.run(["octave-cli", "-qf", str(code / script), *prefix, str(code), alpha, scenes],
                   check=True)


if __name__ == "__main__":
    main()
