import json
from copy import deepcopy

# Base configuration for all ablations, with a differencing parameter
BASE_CONFIG = {
    "experiment": {
        "tags": ["lstm", "simclr", "feature-ablation"],
        "name": "unnamed",
        "group": "Feature Ablations with Differencing",
    },
    "data": {
        "features": ["open", "high", "low", "close", "volume"],
        "differencing": False  # Flag to indicate if differencing is applied
    },
}

# A list of parameters and their ablation values, including differencing options
ABLATION_FEATURES = [
    [],
    ["week_sin", "week_cos"],
    ["month_sin", "month_cos"],
    ["quarter_sin", "quarter_cos"],
    ["year_sin", "year_cos"],
    ["close_ema20"],
    ["close_ema50"],
    ["close_sma20"],
    ["close_sma50"],
    [
        "week_sin",
        "week_cos",
        "month_sin",
        "month_cos",
        "quarter_sin",
        "quarter_cos",
        "year_sin",
        "year_cos",
        "close_ema20",
        "close_ema50",
        "close_sma20",
        "close_sma50",
    ],
]

# Output file for configurations
OUTPUT_FILE = "configs/feature_ablations_with_differencing.json"

if __name__ == "__main__":
    ablation_configs = []
    for i, features in enumerate(ABLATION_FEATURES):
        # Base configuration for each ablation
        config = deepcopy(BASE_CONFIG)
        
        # Update experiment name and data features
        config["experiment"]["name"] = f"Ablation {i}: {features}"
        config["data"]["features"].extend(features)
        
        # Add differencing ablation configurations
        # Create two configurations: one with differencing and one without
        ablation_configs.append(deepcopy(config))  # Without differencing
        config_with_diff = deepcopy(config)
        config_with_diff["data"]["differencing"] = True
        ablation_configs.append(config_with_diff)  # With differencing

    # Save configurations to JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(ablation_configs, f, indent=4)
