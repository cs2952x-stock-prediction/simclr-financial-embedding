import json
from copy import deepcopy
from itertools import product

BASE_CONFIG = {
    "experiment": {"tags": ["lstm", "grid-search"]},
    "models": {"encoder": {}},
}

# NOTE: The 'true' config is a nested set of dictionaries
# However, if we nest the keys in the PARAMS dictionary, then the 'product'
# function will not work as expected. Instead, we have to 'flatten' the keys.
#
# For example, instead of:
# PARAMS = {
#     "models": {
#         "encoder": {
#             "hidden_size": [64, 128, 256, 512],
#             "num_layers": [1, 2, 3],
#         }
#     },
#     "training": {"temperature": [0.1, 0.3, 0.5]},
# }
#
# We have:
# PARAMS = {
#     "models.encoder.hidden_size": [64, 128, 256, 512],
#     "models.encoder.num_layers": [1, 2, 3],
#     "training.temperature": [0.1, 0.3, 0.5],
# }
#
# This way, we can use the 'product' function to generate all possible combinations
# of the parameters. Then, we can 'unflatten' the keys to create the nested dictionary.

PARAMS = {
    "models.encoder.hidden_size": [64, 128, 256, 512],
    "models.encoder.num_layers": [1, 2, 3],
    "training.temperature": [0.1, 0.3, 0.5],
}


if __name__ == "__main__":
    grid_values = []
    keys, values = zip(*PARAMS.items())

    # Generate all possible combinations of the parameters
    #
    # In particular 'product' takes in a list of iterables and returns
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
        param_set["experiment.name"] = f"lstm_grid_{i}"
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

    with open("configs/lstm_grid_search.json", "w") as f:
        json.dump(grid_values, f, indent=4)
