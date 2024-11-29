# simclr-financial-embedding

## TODO List

Today

- [x] add differencing to data pipeline
- [x] data processing should produce an 'undo' function/binary
- [x] upload polygon data to google drive
- [ ] create a download script for polygon data
- [x] create a large-scale dataloader for big CSV
- [ ] run ablation studies
- [x] log the real-value error (in percent and/or absolute dollar)
- [ ] add transformations for simcl training

Small/short-term pipline fixes:

- [ ] check that files are not overwritten and force is used correctly
- [ ] make sure destination folders are created if they down exist
- [ ] parsimony between config/arg parameter names
- [ ] every argparse should include a description

Longer term:

- [x] create dataset for large data (only loads some files in memory at a time)
      TODO: Still needs testing
- [ ] create polygon data cleaning script
- [ ] run ablation studies
- [x] run up grid search

## Getting Started

**Note:** Most of the python scripts described below should have the desired default settings/configurations (so they can be run without arguments).
However, many also have some configuration options.
To check the configuration options (or get a general description of the script) just run `python script.py --help`.

### Virtual Environment

To activate the virtual environment, run `source activate.sh` in the root directory.

If a virtual environment directory named `venv` is not already present, this script

1. creates a virtual-environment named `venv` in the root directory
2. installs the required packaged found in `requirements.txt`
3. activates the virtual environment

### Environment Variables

Many of the scripts require environment variables.
In particular, API keys for services like Weights and Biases, Kaggle, The Bureau of Labor Statistics, and Polygon should all be stored as environment variables.

To setup environment variables, create a file called `.env` in the root directory.
Each line of this file should set up a new environment variable.
Here are the variable names expected by the script:

- `BLS_API_KEY` for the Bureau of Labor Statistics
- `WANDB_API_KEY` for Weights and Biases
- `POLYGON_API_KEY` for Polygon

**Example:** Here is an example of how the contents of `.env` should be formatted:

```txt
BLS_API_KEY="27ff183f62b14238ab6acb60393ba4a4"
WANDB_API_KEY="b12a3f67aff566adf19f8578113d2c79b3c39bbb"
POLYGON_API_KEY="xHbp2Tu633_dkSBpLQe7gWGascRwQjcI"
```

### Data Processing

The data directory is set up with the following structure:

```txt
data
├── interim
│   ├── bls
│   ├── kaggle
│   └── polygon
├── processed
│   ├── bls
│   ├── kaggle
│   └── polygon
└── raw
    ├── bls
    ├── kaggle
    └── polygon
```

The idea here is to have a pipeline `raw -> interim -> processed`

1. Download the raw data from the API with `python src/data/<dataset>_download.py`.
   This create data in the `data/raw/<dataset>` directory.
   Download scripts do some light processing of the data, but not much:

   - standardize column names
   - convert timestamps to date-time format

2. Convert the raw data to 'interim/intermediate' data with `python src/data/<dataset>_clean.py`.
   This should take data from the `data/raw/<dataset>` directory and use it to produce data in the `data/interim/<dataset>` directory.
   This includes cleaning the data more thoroughly, making opinionated decisions to 'fix' the data, and large changes in structure.

   - drop or fill-in NaN values
   - potentially filling in missing entries
   - split single large raw data files into smaller ones (e.g. make a file for each symbol)
   - combine data from raw data (e.g. use raw inflation data to correct raw price data)

3. Convert the intermediate data to the final processed data that is ready for training with `python src/data/<dataset>_process.py`.
   This should take data from the `data/interim/<dataset>` directory and use it to produce data in the `data/processed/<dataset>` directory.
   This more-actively changes the values of the data and makes active changes just for training.

   - creating cyclic temporal features
   - normalizing the dataset
   - converting to log-space
   - embedding class labels

   **NOTE:** Besides the data, `data/processed/dataset/` might hold _models_ used in the processing pipeline.
   For example, we scale with sklearn's StandardScaler and then save the scaler for transforming/undoing data later.

### Training and Config Files

Training files are at the top-level of the `src` folder.
Since these scripts have a large number of parameters, we don't manually pass all the args with `argparse` like we might for other scripts.
Instead, we have config files in the `configs` directory that correspond to each of the scripts.
Each training script should load its corresponding config file and use those parameters during the experiment.

Arguments _can_ be provided --- they will overwrite the corresponding config values from the config file.

To run, just use run `python src/train_file.py` in the root directory.

### End-to-End Example

Running the following commands sets up the environment, downloads the Kaggle SP500 data, and begins a training run on it.
Replace the dummy API keys in the first two lines with your own.

```
$ echo WANDB_API_KEY=\"b12a3f67aff566adf19f8578113d2c79b3c39bbb\" >> .env
$ echo POLYGON_API_KEY=\"xHbp2Tu633_dkSBpLQe7gWGascRwQjcI\" >> .env
$ source activate.sh
(venv) $ python src/data/kaggle_download.py
(venv) $ python src/data/kaggle_clean.py
(venv) $ python src/data/kaggle_process.py
(venv) $ python src/train_simple_lstm.py
```

The script above includes setting up the environment and downloading the data.
However, after the initial setup, you can just run the following:

```
$ source activate.sh
(venv) $ python src/train_simple_lstm.py
```

**NOTE:** There is now a `kaggle_pipeline.py` file that calls `kaggle_download.py`, `kaggle_clean.py`, and `kaggle_process.py`.
It uses the `configs/kiggle-pipeline.yaml` config file to configure all the intermediate steps.
