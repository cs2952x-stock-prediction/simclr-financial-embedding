import json
from copy import deepcopy
from itertools import product

BASE_CONFIG = {
    "experiment": {"tags": ["lstm", "simclr", "grid-search"]},
    "models": {"encoder": {}},
}

GRID_VALUES = {
    "models.encoder.hidden_size": [64, 128, 256, 512],
    "models.encoder.num_layers": [1, 2, 3],
    "training.temperature": [0.5],
}

OUTPUT_FILE = "configs/grid_search.json"

if __name__ == "__main__":
    grid_values = []
    keys, values = zip(*GRID_VALUES.items())

    # 'product' takes in a set of iterables and returns
    # the cartesian product (all combinations) of the iterables.
    for i, values in enumerate(product(*values)):

        # Create a dictionary of the parameter values
        # For example, if values = (64, 1, 0.1), then param_set will be:
        # {
        #     "models.encoder.hidden_size": 64,
        #     "models.encoder.num_layers": 1,
        #     "training.temperature": 0.1
        # }
        param_set = dict(zip(keys, values))

        # Add the experiment name and encoder output size based
        # on the current param_set
        hidden_size = param_set["models.encoder.hidden_size"]
        num_layers = param_set["models.encoder.num_layers"]
        temperature = param_set["training.temperature"]
        param_set["experiment.name"] = f"Grid Point {i}: H={hidden_size} L={num_layers}"
        param_set["experiment.group"] = f"LSTM Grid Search 3"
        param_set["models.encoder.output_size"] = param_set[
            "models.encoder.hidden_size"
        ]

        # Turn the flattened keys back into a nested dictionary
        # and add the parameter values to the base configuration
        # Example: param = "models.encoder.hidden_size", value = 64

        # Step 1: Split the key into a list of subkeys
        # Example: param = ["models", "encoder", "hidden_size"], value = 64

        # Step 2) Use the list of keys to navigate the nested dictionary
        # and add the parameter value (create subdictionaries if necessary)
        # Example: config["models"]["encoder"]["hidden_size"] = 64

        # Final result: config = {
        #     "experiment": {"tags": ["lstm", "grid-search"], "name": "lstm_grid_0"},
        #     "models": {
        #         "encoder": {"hidden_size": 64, "output_size": 64}
        #     }
        #     "training": {"temperature": 0.1}
        # }
        config = deepcopy(BASE_CONFIG)
        for param, value in param_set.items():
            config["experiment"]["tags"].append(f"{param}={value}")
            param = param.split(".")
            subconfig = config
            for key in param[:-1]:
                if key not in subconfig:
                    subconfig[key] = {}
                subconfig = subconfig[key]
            subconfig[param[-1]] = value
        grid_values.append(config)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(grid_values, f, indent=4)
