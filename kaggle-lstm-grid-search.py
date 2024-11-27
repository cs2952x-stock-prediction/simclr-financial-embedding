import json
from itertools import product

params = {
    "models.encoder.hidden_size": [64, 128, 256, 512],
    "models.encoder.num_layers": [1, 2, 3],
    "training.temperature": [0.1, 0.3, 0.5],
}


def combinations():
    keys, values = zip(*params.items())
    for combination in product(*values):
        yield dict(zip(keys, combination))


if __name__ == "__main__":
    grid_values = []
    for i, param_set in enumerate(combinations()):
        config = {"experiment": {
                    "name": f"lstm_grid_{i}",
                    "tags": ["lstm", "grid-search"]
                    }
                }
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


    with open("configs/grid_search.json", "w") as f:
        json.dump(grid_values, f, indent=4)
