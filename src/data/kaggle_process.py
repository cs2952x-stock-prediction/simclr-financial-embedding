import argparse
import logging
import os
import pprint
from datetime import datetime

import pandas as pd
from tqdm import tqdm
from transform import *

############################# GLOBAL VARIABLES #####################################################

DEFAULT_DST = f"data/processed/kaggle/v2"  # the folder containing the output files
DEFAULT_SRC = f"data/interim/kaggle"  # the folder containing the source files

# Logger
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_LOG_FILE = f"logs/kaggle-process_{timestamp}.log"

logger = logging.getLogger(__name__)

# TODO: Move this to a configuration file
TRANSFORMER_CONFIG = {
    "log_features": ["open", "low", "high", "volume", "close", "adj_close"],
    "diff_features": ["open", "low", "high", "close", "adj_close"],
    "scale_features": ["open", "low", "high", "volume", "close", "adj_close"],
}

############################## FUNCTIONS ###########################################################


def get_args():
    """
    Get the command line arguments

    Returns:
    - argparse.Namespace: The command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process the intermediate SP500 dataset from Kaggle into fully-processed training data."
        "Extract only the necessary features, split the data into training and testing sets."
        "Log-transform and scale the data if desired."
        "If the scaling occurs, we save the scaler model in the output directory for later use."
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="The directory to download the data to",
        default=DEFAULT_DST,
    )
    parser.add_argument(
        "--source",
        type=str,
        help="The directory to load the data from",
        default=DEFAULT_SRC,
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
        "--use_columns",
        type=str,
        nargs="+",
        default=[
            "timestamp",
            "open",
            "low",
            "high",
            "volume",
            "close",
            "adj_close",
        ],
        help="The columns to use from the initially-loaded data",
    )
    parser.add_argument(
        "--del_columns",
        type=str,
        nargs="+",
        default=["timestamp"],
        help="The columns to delete after processing",
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
        default=[],
        choices=[
            "seconds",
            "cyclic_day",
            "cyclic_week",
            "cyclic_month",
            "cyclic_quarter",
            "cyclic_year",
        ],
        help="The temporal features to include",
    )
    parser.add_argument(
        "--transform_config",
        type=str,
        help="The configuration file for the transformer",
        default="config/transformer_config.yaml",
    )

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


def validate_paths(config):
    # If the source file does not exist, raise an error
    if not os.path.exists(config["source"]):
        logger.error(f"Source not found: {config['source']}")
        raise FileNotFoundError(f"Source not found: {config['source']}")

    # Ensure the destination folder exists
    if not os.path.exists(config["destination"]):
        os.makedirs(config["destination"])
        logger.info(f"Created destination folder: {config['destination']}")

    # If the destination folder is not empty, raise an error
    if os.listdir(config["destination"]) and not config["force"]:
        logger.error(f"Destination folder is not empty: {config['destination']}")
        raise FileExistsError(
            f"Destination folder is not empty: {config['destination']}. Use --force to overwrite."
        )

    # Ensure the training data folder exists
    train_dir = f"{config['destination']}/train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        logger.info(f"Created training folder: {train_dir}")

    # Ensure the testing data folder exists
    test_dir = f"{config['destination']}/test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        logger.info(f"Created testing folder: {test_dir}")


def add_temporal_features(config, df):
    if "cyclic_day" in config["temporal_features"]:
        add_cyclic_day_feature(df)

    if "cyclic_week" in config["temporal_features"]:
        add_cyclic_week_feature(df)

    if "cyclic_month" in config["temporal_features"]:
        add_cyclic_month_feature(df)

    if "cyclic_quarter" in config["temporal_features"]:
        add_cyclic_quarter_feature(df)

    if "cyclic_year" in config["temporal_features"]:
        add_cyclic_year_feature(df)

    if "seconds" in config["temporal_features"]:
        timestamps = pd.to_datetime(df["timestamp"])
        df["seconds"] = (timestamps - timestamps.min()).total_seconds()


def intersection(lst1, lst2):
    """
    Find the intersection of two lists.

    Args:
        lst1 (list): The first list.
        lst2 (list): The second list.

    Returns:
        list: The intersection of the two lists.
    """
    return [value for value in lst1 if value in lst2]


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

    # Validate the paths
    validate_paths(config)

    # apply basic transformations, add features, and split data
    print("Transforming data, adding features, and splitting data")
    training_cutoff = pd.to_datetime(config["training_cutoff"])

    train_dir = f"{config['destination']}/train"
    test_dir = f"{config['destination']}/test"
    for filename in tqdm(os.listdir(config["source"])):
        logger.info(f"Processing file: {filename}")

        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        df = pd.read_csv(f"{config['source']}/{filename}")
        df = df[config["use_columns"]]

        # Add features
        add_temporal_features(config, df)

        # Split data into train and test
        timestamps = pd.to_datetime(df["timestamp"])
        train_data = df[timestamps < training_cutoff]
        test_data = df[timestamps >= training_cutoff]

        # Post-split validation
        assert isinstance(train_data, pd.DataFrame), "train_data is not a DataFrame"
        assert isinstance(test_data, pd.DataFrame), "test_data is not a DataFrame"

        if train_data.empty:
            logger.warning(f"Empty training data: {filename}")
            continue

        if test_data.empty:
            logger.warning(f"Empty testing data: {filename}")
            continue

        # Transform the data and save
        transformer = DataTransformer(TRANSFORMER_CONFIG)
        last_row = train_data.iloc[-1]
        train_data = transformer.fit_transform(train_data)
        test_data = transformer.transform(test_data, last_row)
        logger.info(f"Transformed data using: {config['transform_config']}")
        file_stub = filename.split(".")[0]
        transformer.save(f"{config['destination']}/{file_stub}_transformer.pkl")

        # Remove unnecessary columns
        drop_cols = intersection(config["del_columns"], train_data.columns)
        train_data = train_data.drop(columns=drop_cols)
        test_data = test_data.drop(columns=drop_cols)

        # Save the training and testing data
        train_data.to_csv(f"{train_dir}/{filename}", index=False)
        test_data.to_csv(f"{test_dir}/{filename}", index=False)

        logger.debug(f"Saved training data to: {train_dir}/{filename}")
        logger.debug(f"Saved testing data to: {test_dir}/{filename}")


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
