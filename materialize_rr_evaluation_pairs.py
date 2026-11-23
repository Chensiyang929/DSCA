"""After adaptation is complete, pair predictions with official GT for metrics.

This is intentionally a separate process from training so evaluation GT cannot
enter adaptation, checkpoint selection or hyper-parameter selection.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat, savemat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor", required=True, choices=["WV3", "WV2", "QB", "GF2"])
    parser.add_argument("--official-rr", required=True)
    parser.add_argument("--result-root", required=True, help="SENSOR result directory")
    parser.add_argument("--training-observations", default=None,
                        help="Observations-only H5 used for adaptation; defaults to RESULT_ROOT/prepared")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenes", default="all")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = Path(args.training_observations or
                         result_root / "prepared" / f"{args.sensor}_rr_observations_only.h5")
    with h5py.File(args.official_rr, "r") as official, h5py.File(training_path, "r") as training:
        if "gt" in training:
            raise RuntimeError("training observation file unexpectedly contains evaluation GT")
        for key in ("ms", "lms", "pan"):
            if key not in official or key not in training:
                raise RuntimeError(f"missing pairing key: {key}")
            # Training canonicalizes observations to float32.  Compare the
            # exact tensor values seen by training, not float64 source bytes.
            official_training_value = np.asarray(official[key], dtype=np.float32)
            training_value = np.asarray(training[key], dtype=np.float32)
            if not np.array_equal(official_training_value, training_value):
                raise RuntimeError(
                    f"official RR and adaptation observations are not exactly paired for key {key}"
                )
        count = official["gt"].shape[0]
        scenes = range(count) if args.scenes == "all" else [int(x) for x in args.scenes.split(",")]
        for scene in scenes:
            folder = result_root / f"scene{scene:02d}"
            if not (folder / "complete.json").is_file():
                raise RuntimeError(f"adaptation not complete for scene {scene}")
            prediction = np.asarray(
                loadmat(folder / "adapted_rr.mat", variable_names=["proposed"])["proposed"],
                dtype=np.float32,
            )
            gt = np.asarray(official["gt"][scene], dtype=np.float32).transpose(1, 2, 0)
            if prediction.shape != gt.shape or not np.isfinite(prediction).all():
                raise RuntimeError(f"prediction/GT mismatch at scene {scene}: {prediction.shape} vs {gt.shape}")
            savemat(output_dir / f"scene{scene:02d}.mat", {"proposed": prediction, "gt": gt})
    print(f"Evaluation-only pairs written to {output_dir}")


if __name__ == "__main__":
    main()
