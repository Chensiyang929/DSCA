"""Configuration B: sensor-specific 240-update schedules and fixed MTF gains."""

FIXED_ALPHA = {
    "WV3": 0.421988727521689,
    "WV2": 0.523164735197393,
    "QB": 0.7554278818849035,
    "GF2": 1.348850350867117,
}

SENSORS = {
    "WV3": dict(
        trainer="train_linear_gated_domain_wrapped_normalized_spatial.py",
        infer="infer_linear_gated_domain_wrapped_all_sensors.py",
        adapter="head", family="linear", control_lr_scale=1,
        epochs=240, cycle=(2, 8), gt_lr=1.5e-3, full_lr=4.83e-4,
        weights=(4_230_000, 21, 89.25), teacher="original_fusionnet_WV3",
    ),
    "WV2": dict(
        trainer="train_response_gated_domain_wrapped_normalized_spatial.py",
        infer="infer_response_gated_domain_wrapped_all_sensors.py",
        adapter="head", family="response", control_lr_scale=200,
        epochs=240, cycle=(5, 5), gt_lr=6.25e-4, full_lr=5e-4,
        weights=(3_172_500, 13.23, 89.25), teacher="original_fusionnet_WV2",
    ),
    "QB": dict(
        trainer="train_response_gated_domain_wrapped_normalized_spatial.py",
        infer="infer_response_gated_domain_wrapped_all_sensors.py",
        adapter="head", family="response", control_lr_scale=150,
        epochs=240, cycle=(2, 8), gt_lr=5e-4, full_lr=4e-4,
        weights=(3_331_125, 24, 476), teacher="original_fusionnet_QB",
    ),
    "GF2": dict(
        trainer="train_open_gated_domain_wrapped_normalized_spatial.py",
        infer="infer_open_gated_domain_wrapped_all_sensors.py",
        adapter="head", family="open", control_lr_scale=1,
        epochs=240, cycle=(4, 6), gt_lr=7.5e-4, full_lr=8e-4,
        weights=(1_057_500, 21, 714), teacher="original_fusionnet_GF2",
    ),
}
