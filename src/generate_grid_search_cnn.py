import json
from copy import deepcopy
from itertools import product

BASE_CONFIG = {
    "experiment": {"tags": ["cnn", "simclr", "grid-search"]},
    "models": {"encoder": {}},
}

GRID_VALUES = {
    "models.encoder.out_channels": [8, 16, 32, 64, 128],
    "models.encoder.kernel_size": [2, 4, 8, 16],
    "models.encoder.num_layers": [3],
    "training.temperature": [0.5],
    "optimizers.baseline_lr": [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
}

OUTPUT_FILE = "configs/grid_search_cnn.json"

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
        channels_out = param_set["models.encoder.out_channels"]
        num_layers = param_set["models.encoder.num_layers"]
        kernel_size = param_set["models.encoder.kernel_size"]
        temperature = param_set["training.temperature"]
        baseline_lr = param_set["optimizers.baseline_lr"]
        param_set["experiment.name"] = f"Grid Point {i}: C_out={channels_out} Kern_size={kernel_size} baseline_lr={baseline_lr}"
        param_set["experiment.group"] = f"Grid Search CNN 3"
      

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
