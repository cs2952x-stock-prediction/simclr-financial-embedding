# simclr-financial-embedding

## Next Tasks

1. **Important** Do more hyperparameter tuning (this can be done on the kaggle dataset or polygon).
   The main issue right now is slow convergence of the SimCLR model --- try some increased values of the learning rate.
   Learning rate of finetuning less likely to need adjustment, but try that anyway.
   Baseline model might actually need to be lowered --- it already converges very quickly and might benefit from some stability.

   In particular, larger models probably need different rates from smaller models.
   See how different learning rates affects the the convergence of different model sizes.

3. **Important** Add more transformation options to the SimCLR pipeline (would be nice to have them as config parameters).
   These would be (1) smoothing/averaging (2) swapping adjacent entry values (3) smoothing by setting to zero (only if this makes sense).

4. **Important** Start training on the polygon data to see if 5-minutely frequency makes a big difference in outcomes.
   "BigSeriesDataset" is meant for training on ALL polygon data, but to start it could be more practical to create a folder with just a handful of stocks from the polgyon dataset and train on that the same way that we have been with kaggle.
   Perform a new grid search and ablations with this data.
   
6. Create a script that generates a graph of predicted price values against the true price values for a stock.
   General steps are:
   a. Set (1) model checkpoint directory (2) stock data file
   b. Load the model (encoder and probe are saved separately, so you will have to create code to combine)
   c. Load the data into a dataset (use step size 1 and adjustable sequence length --- no shuffling)
   d. Make predictions
   e. If the data is transformed, you will probably have to undo those transformations (unscale > undo diff with cumsum > undo log with exp)
   f. Plot and save
   
7. Check that differencing is not having a negative impact on results.
   Run smaller grid search without differencing and look for changes in the APE.
   **NOTE** The current function for calculating APE makes some assumptions about the transformations on the data (log-differencing).
   Make sure that the APE is corrected for differencing correctly.

8. Create script that compares the performance of a model to linear regression.
   Similar initial control flow to task (1).
   a. Set (1) model checkpoint directory (2) stock data file
   b. Load the model (encoder and probe are saved separately, so you will have to create code to combine)
   c. Load the data into a dataset (use step size 1 and adjustable sequence length --- no shuffling)
   d. Iterate through the dataset --- at each step, perform linear regression on the sequences.
      You will probably have to 
   e. Calculate the percent error of linear regression and model. Add them up and average at the end.
   **Note:** Linear regression _after differencing_ is actually a second-order (2-degree polynomial) regression on the original data.
   Not a bad thing --- just keep in mind for analysis and writing the paper.

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

## Dec 2 Checkpoint/Presentation Outline

### Data/Transformations

Our data:
- So far, we have been operating on the Kaggle dataset since it is faster to train/test on.
- Another good option would have been to use a small subset of the Polygon daset (not doing that is my bad -Ian)
- The main issue with running on all polygon data is that it doesn't fit into memory. I have a BigDataset object that should be able to load a single file at a time (and shuffle the contents of the file for training), but since we have been using Kaggle continued to do so for a consistent baseline of comparison.

Properties of time series data:
- We have to handle time series data very carefully to avoid data leaking (information/answers from future data being exposed to earlier data). For us, that means:
      - Splitting the data into train/test along dates (we pick Jan 1, 2023)
      - Scaler is applied to the test data, but the scaling values are only picked based on the training data.

Properties of Financial data:
- In particular, time series data has not-so-fun properties (1) high noise-signal ratio (2) experiences "distribution shift" (3) is non-negative and not zero-centered (4) follows a log-normal distribution (distribution of e^X where X is a normal RV) (5) experiences changes multiplicatively instead of linearly
- We attempt to solve (1) with SIMCL (extracting important information into the embedding)
- We can try solving (2) via differencing, logging
- We can try solving (3-5) via logging
- This should transform the data into a gaussian distribution and scaling at the end should make it normal

Here, it would be great to have some visualizations of (1) a non-transformed distribution (log-normal) and (2) a transformed distribution (normal).

Also note the ADF statistical test that shows that the price-based columns do have non-stationary distribution. 

^^^ It would be great to have some results that differencing also affects the empirical results i.e. the test loss of baseline or finetuning.

### Model

We've been focusing on the SimCLR model. Right now the framework only applied gaussian noise as a transformation. A next step would be to incorporate a composition of more transformations.

"Baseline" refers to a model with identical structure to that used in SimCLR, but trained end-to-end on the downstream task.
"Finetuned" refers to the encoder trained on SimCLR + the probe trained on the downstream task.

### Experiments + Results

- Results of our ablation
- Results of our grid search

Summary: results aren't great and baseline generally outperforms the simclr fine-tuned model.

