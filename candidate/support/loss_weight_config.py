"""Per-sensor coefficients from complete 20-scene HQNR tuning."""

SENSOR_LOSS_WEIGHTS = {
    "WV3": {"teacher": 4_230_000.0, "spatial": 21.0, "spectral": 119.0},
    "WV2": {"teacher": 6_345_000.0, "spatial": 22.05, "spectral": 89.25},
    "QB":  {"teacher": 3_172_500.0, "spatial": 24.0, "spectral": 476.0},
    "GF2": {"teacher": 1_057_500.0, "spatial": 21.0, "spectral": 238.0},
}


def get_loss_weights(sensor):
    key = str(sensor).upper()
    if key not in SENSOR_LOSS_WEIGHTS:
        raise ValueError(f"No validated loss weights for sensor: {sensor}")
    return SENSOR_LOSS_WEIGHTS[key].copy()
