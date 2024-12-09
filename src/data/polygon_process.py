import argparse
import logging
import os
import pickle
import pprint
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from util.features import add_moving_avg_features, add_temporal_features

############################# GLOBAL VARIABLES #####################################################

DEFAULT_DST = f"data/processed/polygon"  # the folder containing the output files
DEFAULT_SRC = f"data/interim/polygon"  # the folder containing the source files

# Logger
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_LOG_FILE = f"logs/kaggle-process_{timestamp}.log"

logger = logging.getLogger(__name__)

# Default features to add to data
DEFAULT_TEMPORAL_FEATURES = [
    "seconds",
    "cyclic_week",
    "cyclic_month",
    "cyclic_quarter",
    "cyclic_year",
]
TEMPORAL_FEATURE_CHOICES = [
    "seconds",
    "cyclic_day",
    "cyclic_week",
    "cyclic_month",
    "cyclic_quarter",
    "cyclic_year",
]
DEFAULT_AVG_FEATURES = ["close_sma20", "close_ema20", "close_sma50", "close_ema50"]
DEFAULT_SHIFT_FEATURES = ["close"]

# Default columns to apply transformations
LOG_EPSILON = 1e-6  # to avoid log(0)
DEFAULT_LOG_TRANSFORM = [
    "open",
    "low",
    "high",
    "volume",
    "close",
    "adj_close",
    "close_sma20",
    "close_ema20",
    "close_sma50",
    "close_ema50",
    "close_next",
]
DEFAULT_DIFF_TRANSFORM = [
    "open",
    "low",
    "high",
    # "volume", # volue is more stationary than other features and often not differenced
    "close",
    "adj_close",
    # "close_sma20", # moving averages encode long-term trends. Differencing would remove this information.
    # "close_ema20",
    # "close_sma50",
    # "close_ema50",
    "close_next",
]
DEFAULT_SCALE_TRANSFORM = [
    "open",
    "low",
    "high",
    "volume",
    "close",
    "adj_close",
    "close_sma20",
    "close_ema20",
    "close_sma50",
    "close_ema50",
    "close_next",
]

############################## FUNCTIONS ###########################################################


def get_args():
    """
    Get the command line arguments

    Returns:
    - argparse.Namespace: The command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process the intermediate Polygon data into fully-processed training data."
        "Extract only the necessary features, split the data into training and testing sets."
        "Log-transform and scale the data if desired."
        "If the scaling occurs, we save the scaler model in the output directory for later use."
    )
    parser.add_argument(
        "--source",
        type=str,
        help="The directory to load the data from",
        default=DEFAULT_SRC,
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="The directory to export the data to",
        default=DEFAULT_DST,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and overwrite existing files",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="The log level to use",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="The log file to use",
    )
    parser.add_argument(
        "--del_features",
        type=str,
        nargs="*",
        default=[],
        help="The columns to delete before saving intermediate files",
    )
    parser.add_argument(
        "--training_cutoff",
        type=str,
        default="2023-01-01",
        help="The date to split the data into training and testing",
    )
    parser.add_argument(
        "--temporal_features",
        type=str,
        nargs="*",
        default=DEFAULT_TEMPORAL_FEATURES,
        choices=TEMPORAL_FEATURE_CHOICES,
        help="The temporal features to include",
    )
    parser.add_argument(
        "--avg_features",
        type=str,
        nargs="*",
        default=DEFAULT_AVG_FEATURES,
        help="The moving average features to include",
    )
    parser.add_argument(
        "--shift_features",
        type=str,
        nargs="*",
        default=DEFAULT_SHIFT_FEATURES,
        help="The features to shift for prediction",
    )
    parser.add_argument(
        "--log_transform",
        type=str,
        nargs="*",
        default=DEFAULT_LOG_TRANSFORM,
        help="The features to log-transform",
    )
    parser.add_argument(
        "--diff_transform",
        type=str,
        nargs="*",
        default=DEFAULT_DIFF_TRANSFORM,
        help="The features to difference-transform",
    )
    parser.add_argument(
        "--scale_transform",
        type=str,
        nargs="*",
        default=DEFAULT_SCALE_TRANSFORM,
        help="The features to scale-transform",
    )
    parser.add_argument(
        "--data_limit",
        type=float,
        default=5.0,
        help="The maximum amount of data in the dataset (in GB)",
    )
    # TODO: Add an argument for prioritizing certain files
    # e.g. prioritize larger files, files with fewer gaps, files with larger date range

    return parser.parse_args()


def configure_logger(log_level, log_file):
    """
    Setup the logger and ensure the log folder exists.

    Args:
    - log_level (str): The log level to use
    - log_file (str): The path to the log file
    """
    # make sure the log folder exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # configure the logger
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        encoding="utf-8",
    )
    logging.getLogger().setLevel(log_level)


def prepare_directories(config):
    """
    Prepare the output directories.

    Args:
    - config (dict): The configuration dictionary
    """
    # Raise an error if the source folder does not exist
    if not os.path.exists(config["source"]):
        logger.error(f"Source not found: {config['source']}")
        raise FileNotFoundError(f"Source not found: {config['source']}")

    # Create the destination folder if it does not exist
    if not os.path.exists(config["destination"]):
        os.makedirs(config["destination"])
        logger.info(f"Created destination folder: {config['destination']}")

    # Raise an error if the destination folder is not empty and force is not set
    if os.listdir(config["destination"]) and not config["force"]:
        logger.error(f"Destination folder is not empty: {config['destination']}")
        raise FileExistsError(
            f"Destination folder is not empty: {config['destination']}. Use --force to overwrite."
        )

    # Create the training data folder if it does not exist
    train_dir = f"{config['destination']}/train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        logger.info(f"Created training folder: {train_dir}")

    # Create the testing data folder if it does not exist
    test_dir = f"{config['destination']}/test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        logger.info(f"Created testing folder: {test_dir}")

    # Create the intermediate data folder if it does not exist
    inter_dir = f"{config['destination']}/intermediate"
    if not os.path.exists(inter_dir):
        os.makedirs(inter_dir)
        logger.info(f"Created intermediate folder: {inter_dir}")


def create_intermediate_files(config):
    """
    Add features to the data and save the intermediate files.

    Args:
    - config (dict): The configuration dictionary.
    """
    intermediate_dir = f"{config['destination']}/intermediate"

    # Sort files based on the size
    # Take all the files that fit within the data limit
    files = os.listdir(config["source"])
    files_with_size = [(os.path.getsize(f"{config['source']}/{f}"), f) for f in files]
    files_with_size = sorted(files_with_size, reverse=True)
    total_size = 0
    files = []
    for size, filename in files_with_size:
        total_size += size
        if total_size / 1e9 > config["data_limit"]:
            break
        files.append(filename)

    for filename in tqdm(files):
        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        logger.info(f"Adding features to file: {filename}")

        df = pd.read_csv(f"{config['source']}/{filename}")

        # Ensure the data is not empty
        if len(df) == 0:
            logger.error(f"Empty data file: {filename}")
            raise ValueError(f"Empty data file: {filename}")

        for col in config["del_features"]:
            if col in df.columns:
                df = df.drop(col, axis=1)

        # Add features
        add_temporal_features(df, config["temporal_features"])
        add_moving_avg_features(df, config["avg_features"])

        # Add shifted features
        for feature in config["shift_features"]:
            df[f"{feature}_next"] = df[feature].shift(-1)

        # Save intermediate data
        df = df.dropna()
        df.to_csv(f"{intermediate_dir}/{filename}", index=False)


def apply_logdiff(df, config):
    """
    Apply log and difference transformations to the data.

    Args:
    - df (pd.DataFrame): The data to transform.
    - config (dict): The configuration dictionary.

    Returns:
    - pd.DataFrame: The transformed data.
    """
    log_cols = config["log_transform"]
    df.loc[:, log_cols] = np.log((df[log_cols] + LOG_EPSILON).astype(float))

    diff_cols = config["diff_transform"]
    df.loc[:, diff_cols] = df[diff_cols].diff()
    df = df.iloc[1:, :]
    return df


def create_scaler(config):
    """
    Fit a scaler to the training data and save it to the output directory.

    Args:
    - config (dict): The configuration dictionary.

    Returns:
    - StandardScaler: The fitted scaler.
    """
    intermediate_dir = f"{config['destination']}/intermediate"
    training_cutoff = pd.to_datetime(config["training_cutoff"])
    scaler = StandardScaler()
    for filename in tqdm(os.listdir(config["source"])):
        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        logger.debug(f"Training scaler on file: {filename}")

        # Save intermediate data
        df = pd.read_csv(f"{intermediate_dir}/{filename}")

        # Split data into train and test
        timestamps = pd.to_datetime(df["timestamp"])
        train_data = df[timestamps < training_cutoff]

        if train_data.empty:
            logger.warning(f"Empty training data: {filename}")
            continue

        train_data = apply_logdiff(train_data, config)
        scaler.partial_fit(train_data[config["scale_transform"]])

    # Save the scaler
    scaler_file = f"{config['destination']}/scaler.pkl"
    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"Saved scaler to: {scaler_file}")

    return scaler


def create_train_test_files(config, scaler):
    """
    Transform the data and create training and testing files.

    Args:
    - config (dict): The configuration dictionary.
    """
    intermediate_dir = f"{config['destination']}/intermediate"
    train_dir = f"{config['destination']}/train"
    test_dir = f"{config['destination']}/test"
    training_cutoff = pd.to_datetime(config["training_cutoff"])
    pd.set_option("display.max_columns", None)
    for filename in tqdm(os.listdir(config["source"])):
        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        logger.debug(f"Creating train/test data for: {filename}")

        df = pd.read_csv(f"{intermediate_dir }/{filename}")

        # Split data into train and test
        timestamps = pd.to_datetime(df["timestamp"])
        train_data = df[timestamps < training_cutoff]
        test_data = df[timestamps >= training_cutoff]

        if train_data.empty:
            logger.warning(f"Empty training data: {filename}")
            continue

        if test_data.empty:
            logger.warning(f"Empty testing data: {filename}")
            continue

        # Transform the data and save
        scale_cols = config["scale_transform"]
        last_row = train_data.iloc[-1].copy()
        train_data = apply_logdiff(train_data, config)
        train_data.loc[:, scale_cols] = scaler.transform(train_data[scale_cols])
        test_data = pd.concat([last_row.to_frame().T, test_data], ignore_index=True)
        test_data = apply_logdiff(test_data, config)
        test_data.loc[:, scale_cols] = scaler.transform(test_data[scale_cols])

        # Save the training and testing data
        train_data.to_csv(f"{train_dir}/{filename}", index=False)
        test_data.to_csv(f"{test_dir}/{filename}", index=False)

        logger.debug(f"Saved training data to: {train_dir}/{filename}")
        logger.debug(f"Saved testing data to: {test_dir}/{filename}")


############################## MAIN FUNCTION ######################################################


def main(config):
    """
    Process the intermediate SP500 dataset from Kaggle into fully-processed training data.
    Extract only the necessary features, split the data into training and testing sets.
    Log-transform and scale the data if desired.
    If the scaling occurs, we save the scaler model in the output directory for later use.

    Args:
    - config (dict): The configuration dictionary
    """
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"Starting Kaggle data processing at {timestamp}")
    logger.info(f"Configuration:\n{pprint.pformat(config)}\n")

    prepare_directories(config)

    print("Adding features and generating intermediate files...")
    create_intermediate_files(config)

    print("Fitting the scaler...")
    scaler = create_scaler(config)

    print("Transforming the data and creating training and testing files...")
    create_train_test_files(config, scaler)


if __name__ == "__main__":
    # Get command line arguments
    args = get_args()

    # Start logging
    configure_logger(args.log_level, args.log_file)

    # Log the arguments
    formatted_args = "\n\t".join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments:\n\t{formatted_args}\n")

    # Run the main function
    config = {k: v for k, v in vars(args).items() if k not in ["log_level", "log_file"]}
    main(config)
