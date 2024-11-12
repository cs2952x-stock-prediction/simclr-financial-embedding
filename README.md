# simclr-financial-embedding

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
│   ├── bls_inflation
│   │   └── from_raw.py
│   ├── kaggle_sp500
│   │   └── from_raw.py
│   └── polygon
│       └── from_raw.py
├── processed
│   ├── bls_inflation
│   │   └── from_interim.py
│   ├── kaggle_sp500
│   │   └── from_interim.py
│   └── polygon
│       └── from_interim.py
└── raw
    ├── bls_inflation
    │   └── download.py
    ├── kaggle_sp500
    │   └── download.py
    └── polygon
        └── download.py
```

The idea here is to have a pipeline `raw -> interim -> processed`

1. Download the raw data from the API with `python data/raw/<dataset>/download.py`.
   Download scripts do some light processing of the data, but not much:

   - standardize column names
   - convert timestamps to date-time format

2. Convert the raw data to 'interim/intermediate' data with `python data/interim/<dataset>/from_raw.py`.
   This includes cleaning the data more thoroughly, making opinionated decisions to 'fix' the data, and large changes in structure.

   - drop or fill-in NaN values
   - potentially filling in missing entries
   - split single large raw data files into smaller ones (e.g. make a file for each symbol)
   - combine data from raw data (e.g. use raw inflation data to correct raw price data)

3. Convert the intermediate data to the final processed data that is ready for training with `python data/processed/<dataset>/from_interim.py`.
   This more-actively changes the values of the data and makes active changes just for training.

   - creating cyclic temporal features
   - normalizing the dataset
   - converting to log-space
   - embedding class labels

All scripts populate/create the data in their own directories.

**Note:** Instead of dumping the data into the same directory as the scripts, I will probably update the scripts to put their data into a distinct output directory.
For example, rather than `data/raw/polygon/download.py` downloading its data into `data/raw/polygon/`, it would download data to `data/raw/polygon/out/`

### Training

There are several files in `src/`, but `src/training.py` is the only one meant to be run as a script.
Currently, the configuration for this file is set at the very top of the script (it does not expect command-line arguments) but this will be corrected later.

To run, just use run `python src/training.py` in the root directory.

### End-to-End Example

Running the following commands sets up the environment, downloads the Kaggle SP500 data, and begins a training run on it.
Replace the dummy API keys in the first two lines with your own.

```
$ echo WANDB_API_KEY=\"b12a3f67aff566adf19f8578113d2c79b3c39bbb\" >> .env
$ echo POLYGON_API_KEY=\"xHbp2Tu633_dkSBpLQe7gWGascRwQjcI\" >> .env
$ source activate.sh
(venv) $ python data/raw/kaggle_sp500/download.py
(venv) $ python data/interim/kaggle_sp500/from_raw.py
(venv) $ python data/processed/kaggle_sp500/from_interim.py
(venv) $ python src/training.py
```

The script above includes setting up the environment and downloading the data.
However, after the initial setup, you can just run the following:

```
$ source activate.sh
(venv) $ python src/training.py
```

## Notes:

- Consider working in log-space for the following reasons:

  - Captures multiplicative scaling of values
  - Avoids lopsided-weighting of large values/spikes
  - Model only needs to learn additive relations while implicitly capturing multiplicative ones
  - Can handle negatives implicitly (we don't need to handle that as a special case)

- Remember to only apply transformations to KPI values.

  - Should symbol embeddings be left alone? (invariant to the identity of the stock?)
  - Should temporal features have their own transformation functions? (masking would be a random _but valid_ cyclical embedding)

- Ablation studies on the contributions of cyclical temporal features (professor's suggestion)

- Possible baseline comparisons
  - compare to the results of Denoising Financial Data paper
  - compare to training on the downstream task directly (use probe on encoder and allow the gradients to back-propagate)
  - compare to classic autoregression methods (linear/polynomial/ARIMA)

## Possible Embedding Models:

### Simple LSTM

#### Pipeline

- Feed through a linear layer to get an embedding of the data.
- Feed the embedding into the lstm
- Apply a final linear layer to the lstm output

### CNN-only

#### Pipeline

1.

## To-Do
