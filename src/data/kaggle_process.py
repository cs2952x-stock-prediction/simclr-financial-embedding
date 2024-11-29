import argparse
import logging
import os
import pprint
from datetime import datetime

import pandas as pd
from tqdm import tqdm

from util.features import add_moving_avg_features, add_temporal_features
from util.transform import DataTransformer

############################# GLOBAL VARIABLES #####################################################

DEFAULT_DST = f"data/processed/kaggle/v2"  # the folder containing the output files
DEFAULT_SRC = f"data/interim/kaggle"  # the folder containing the source files

# Logger
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_LOG_FILE = f"logs/kaggle-process_{timestamp}.log"

logger = logging.getLogger(__name__)

# TODO: Move this to a configuration file
TRANSFORM_CONFIG = {
    "log_features": [
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
    ],
    "diff_features": [
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
    ],
    "scale_features": [
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
    ],
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
        nargs="*",
        default=[],
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
        default=[
            "seconds",
            "cyclic_day",
            "cyclic_week",
            "cyclic_month",
            "cyclic_quarter",
            "cyclic_year",
        ],
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
        "--avg_features",
        type=str,
        nargs="*",
        default=["sma_20", "ema_20", "sma_50", "ema_50"],
        help="The moving average features to include",
    )
    parser.add_argument(
        "--shift_features",
        type=str,
        nargs="*",
        default=["close"],
        help="The features to shift for prediction",
    )
    parser.add_argument(
        "--transform_config",
        type=str,
        default=None,
        help="The configuration file for the transformer",
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

    # Ensure the intermediate data folder exists
    inter_dir = f"{config['destination']}/intermediate"
    if not os.path.exists(inter_dir):
        os.makedirs(inter_dir)
        logger.info(f"Created intermediate folder: {inter_dir}")


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


def create_intermediate_files(config):
    inter_dir = f"{config['destination']}/intermediate"
    for filename in tqdm(os.listdir(config["source"])):
        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        logger.info(f"Adding features to file: {filename}")

        df = pd.read_csv(f"{config['source']}/{filename}")
        df = df[config["use_columns"]]

        # Add features
        add_temporal_features(df, config["temporal_features"])
        add_moving_avg_features(df, config["avg_features"])

        # Add shifted features
        assert isinstance(df, pd.DataFrame), "df is not a DataFrame"
        for feature in intersection(config["shift_features"], df.columns):
            df[f"{feature}_next"] = df[feature].shift(-1)

        # Save intermediate data
        df = df.dropna()
        df.to_csv(f"{inter_dir}/{filename}", index=False)


def train_transformer(config):
    inter_dir = f"{config['destination']}/intermediate"
    training_cutoff = pd.to_datetime(config["training_cutoff"])
    transformer = DataTransformer(config["transform_config"])
    for filename in tqdm(os.listdir(config["source"])):
        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        logger.info(f"Training transformer on file: {filename}")

        # Save intermediate data
        df = pd.read_csv(f"{inter_dir}/{filename}")

        # Split data into train and test
        timestamps = pd.to_datetime(df["timestamp"])
        train_data = df[timestamps < training_cutoff]

        # Post-split validation
        assert isinstance(train_data, pd.DataFrame), "train_data is not a DataFrame"

        if train_data.empty:
            logger.warning(f"Empty training data: {filename}")
            continue

        transformer.partial_fit(train_data)

    return transformer


def create_train_test_files(config, transformer):
    inter_dir = f"{config['destination']}/intermediate"
    train_dir = f"{config['destination']}/train"
    test_dir = f"{config['destination']}/test"
    training_cutoff = pd.to_datetime(config["training_cutoff"])
    for filename in tqdm(os.listdir(config["source"])):
        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        logger.info(f"Creating train/test data for: {filename}")

        df = pd.read_csv(f"{inter_dir}/{filename}")

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
        last_row = train_data.iloc[-1].copy()
        train_data = transformer.transform(train_data)
        test_data = transformer.transform(test_data, last_row)

        # Remove unnecessary columns
        drop_cols = intersection(config["del_columns"], train_data.columns)
        train_data = train_data.drop(columns=drop_cols)
        test_data = test_data.drop(columns=drop_cols)

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

    # Validate the paths
    validate_paths(config)

    print("Adding features and generating intermediate files...")
    create_intermediate_files(config)

    print("Training the transformer...")
    transformer = train_transformer(config)

    print("Transforming the data and creating training and testing files...")
    create_train_test_files(config, transformer)
    transformer.save(f"{config['destination']}/transformer.pkl")


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
    if "transform_config" not in args or args.transform_config is None:
        config["transform_config"] = TRANSFORM_CONFIG
    main(config)
