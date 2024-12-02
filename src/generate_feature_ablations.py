import json
from copy import deepcopy

# This will provide the base configuration for all ablations
# i.e. the default configuration will be overriden with these values
# for all ablations
BASE_CONFIG = {
    "experiment": {
        "tags": ["lstm", "simclr", "feature-ablation"],
        "name": "unnamed",
        "group": "Feature Ablations 2",
    },
    "data": {"features": ["open", "high", "low", "close", "volume"]},
}

# A list of parameters and their ablation values.
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
        "year_sin",
        "year_cos",
        "close_ema20",
        "close_ema50",
        "close_sma20",
        "close_sma50",
    ],
]

# Output file
OUTPUT_FILE = "configs/feature_ablations.json"

if __name__ == "__main__":
    ablation_configs = []
    for i, features in enumerate(ABLATION_FEATURES):

        config = deepcopy(BASE_CONFIG)
        config["experiment"]["name"] = f"Ablation {i}: {features}"
        config["data"]["features"].extend(features)
        ablation_configs.append(config)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(ablation_configs, f, indent=4)
