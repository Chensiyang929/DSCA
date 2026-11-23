"""Per-sensor mainline defaults selected by complete 20-scene mean HQNR studies."""

SENSOR_CYCLES = {
    'WV3': (4, 6),
    'WV2': (8, 72),
    'QB': (5, 5),
    'GF2': (1, 9),
}

# Fixed budgets selected in the completed five-node epoch study.
# Within each budget, NEW training selects by per-epoch Full HQNR, not loss.
SENSOR_EPOCHS = {'WV3': 240, 'WV2': 480, 'QB': 480, 'GF2': 120}


def get_training_epochs(sensor, override=None):
    epochs = SENSOR_EPOCHS[sensor.upper()] if override is None else int(override)
    if epochs < 1:
        raise ValueError('epochs must be positive')
    return epochs


def get_training_cycle(sensor):
    """Return consecutive GT epochs then Full epochs, NOT interleaving."""
    return SENSOR_CYCLES[sensor.upper()]


def epoch_schedule(sensor, epochs, reduced_ratio=None):
    if epochs <= 0:
        raise ValueError('epochs must be positive')
    if reduced_ratio is None:
        gt, full = get_training_cycle(sensor)
        return [index % (gt + full) < gt for index in range(epochs)]
    # Preserve explicitly requested legacy percent schedules (including endpoints).
    percentage = max(0.0, min(100.0, float(reduced_ratio)))
    count = int(round(epochs * percentage / 100.0))
    return [((index + 1) * count) // epochs > (index * count) // epochs
            for index in range(epochs)]
