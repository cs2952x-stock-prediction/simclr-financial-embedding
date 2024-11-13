import argparse
import logging
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm

############################# GLOBAL VARIABLES #####################################################

THIS_FILE = os.path.abspath(__file__)  # the absolute path the current file
THIS_DIR = os.path.dirname(THIS_FILE)  # the folder containing the current file
DEFAULT_DST = f"{THIS_DIR}/out"  # the folder containing the output files
DEFAULT_SRC = f"{THIS_DIR}/../../interim/kaggle_sp500/out"  # the folder containing the source files
LOGS_DIR = f"{THIS_DIR}/logs"  # the folder containing the logs

# integer-valued columns that can be log-transformed
INT_LOG_COLUMNS = [
    "volume",
]
INT_EPS = 1  # quantity to add to integer columns before log-transforming

# float-valued columns that can be log-transformed
FLOAT_LOG_COLUMNS = [
    "close",
    "high",
    "low",
    "open",
]
FLOAT_EPS = 0.01  # quadrity to add to float columns before log-transforming


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
        help="The log level to use",
    )
    parser.add_argument(
        "--keep_columns",
        type=str,
        nargs="+",
        default=["close", "high", "low", "open", "volume"],
        help="The columns to keep",
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
        choices=["day", "week", "month", "quarter", "year"],
        help="The temporal features to include",
    )
    parser.add_argument(
        "--scale_features",
        type=str,
        nargs="*",
        default=["close", "high", "low", "open", "volume"],
        help="The features to scale",
    )
    parser.add_argument(
        "--log_transform", type=bool, default=True, help="Whether to log-transform"
    )

    return parser.parse_args()


def build_logger(log_level):
    """
    Setup the logget and ensure the log folder exists.

    Args:
    - log_level (str): The log level to use

    Returns:
    - logging.Logger: The logger
    """
    # start the logger --- make sure the log folder exists
    logger = logging.getLogger(__name__)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    logging.basicConfig(
        filename=f"{LOGS_DIR}/{timestamp}.log",
        filemode="w",
        encoding="utf-8",
    )
    logger.setLevel(log_level)
    return logger


############################## EXECUTION ##########################################################

if __name__ == "__main__":
    # Get command line arguments
    args = get_args()

    # Start logging
    logger = build_logger(args.log_level)

    logger.info(
        f"Arguments:\n\t{'\n\t'.join([f'{k}: {v}' for k, v in vars(args).items()])}"
    )

    # If the source file does not exist, raise an error
    if not os.path.exists(args.source):
        logger.error(f"Source not found: {args.source}")
        raise FileNotFoundError(f"Source not found: {args.source}")

    # Ensure the destination folder exists
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
        logger.info(f"Created destination folder: {args.destination}")

    # If the destination folder is not empty, raise an error
    if os.listdir(args.destination) and not args.force:
        logger.error(f"Destination folder is not empty: {args.destination}")
        raise FileExistsError(
            f"Destination folder is not empty: {args.destination}. Use --force to overwrite."
        )

    # Ensure the training data folder exists
    train_dir = f"{args.destination}/train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        logger.info(f"Created training folder: {train_dir}")

    # Ensure the testing data folder exists
    test_dir = f"{args.destination}/test"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        logger.info(f"Created testing folder: {test_dir}")

    # apply basic transformations, add features, and split data
    print("Transforming data, adding features, and splitting data")
    training_cutoff = pd.to_datetime(args.training_cutoff)
    for filename in tqdm(os.listdir(args.source)):
        logger.info(f"Processing file: {filename}")

        if not filename.endswith(".csv"):
            logger.warning(f"Skipping non-CSV file: {filename}")
            continue

        df = pd.read_csv(f"{args.source}/{filename}")

        # get datetimes and convert to milliseconds
        timestamps = pd.to_datetime(df["timestamp"])
        df["timestamp"] = (timestamps - timestamps.min()).astype("int64") // 10**6

        # Keep only desired columns
        df = df[args.keep_columns]

        # Apply log transformation to discrete/integer columns
        int_log_cols = [col for col in df.columns if col in INT_LOG_COLUMNS]
        if args.log_transform and len(int_log_cols) > 0:
            log_df = df[int_log_cols]
            assert isinstance(log_df, pd.DataFrame), "log_df is not a DataFrame"
            df[int_log_cols] = log_df.apply(lambda x: np.log(x + INT_EPS))

        # Apply log transformation to continuous/float columns
        float_log_cols = [col for col in df.columns if col in FLOAT_LOG_COLUMNS]
        if args.log_transform and len(float_log_cols) > 0:
            log_df = df[float_log_cols]
            assert isinstance(log_df, pd.DataFrame), "log_df is not a DataFrame"
            df[float_log_cols] = log_df.apply(lambda x: np.log(x + FLOAT_EPS))

        # Add cyclic temporal features

        # Add the cyclic day feature
        fraction_of_day = (
            timestamps.dt.hour + timestamps.dt.minute / 60 + timestamps.dt.second / 3600
        ) / 24
        if "day" in args.temporal_features:
            df["day_sin"] = np.sin(2 * np.pi * fraction_of_day)
            df["day_cos"] = np.cos(2 * np.pi * fraction_of_day)

        # Add the cyclic week feature
        if "week" in args.temporal_features:
            fraction_of_week = (timestamps.dt.dayofweek + fraction_of_day) / 7
            df["week_sin"] = np.sin(2 * np.pi * fraction_of_week)
            df["week_cos"] = np.cos(2 * np.pi * fraction_of_week)

        # Add the cyclic month feature
        if "month" in args.temporal_features:
            days_in_month = timestamps.dt.days_in_month
            fraction_of_month = (timestamps.dt.day + fraction_of_day) / days_in_month
            df["month_sin"] = np.sin(2 * np.pi * fraction_of_month)
            df["month_cos"] = np.cos(2 * np.pi * fraction_of_month)

        # Add the cyclic quarter feature
        if "quarter" in args.temporal_features:
            period = timestamps.dt.to_period("Q")
            days_in_quarter = period.apply(
                lambda x: x.end_time.dayofyear - x.start_time.dayofyear
            )
            day_of_quarter = timestamps.dt.dayofyear - period.apply(
                lambda x: x.start_time.dayofyear
            )
            fraction_of_quarter = (day_of_quarter + fraction_of_day) / days_in_quarter
            df["quarter_sin"] = np.sin(2 * np.pi * fraction_of_quarter)
            df["quarter_cos"] = np.cos(2 * np.pi * fraction_of_quarter)

        # Add the cyclic year feature
        if "year" in args.temporal_features:
            days_in_year = timestamps.dt.to_period("Y").apply(
                lambda x: x.end_time.dayofyear
            )
            day_of_year = timestamps.dt.dayofyear
            fraction_of_year = (day_of_year + fraction_of_day) / days_in_year
            df["year_sin"] = np.sin(2 * np.pi * fraction_of_year)
            df["year_cos"] = np.cos(2 * np.pi * fraction_of_year)

        # Split data into train and test
        train_data = df[timestamps < training_cutoff]
        test_data = df[timestamps >= training_cutoff]

        assert isinstance(train_data, pd.DataFrame), "train_data is not a DataFrame"
        assert isinstance(test_data, pd.DataFrame), "test_data is not a DataFrame"

        if train_data.empty:
            logger.warning(f"Empty training data: {filename}")

        if test_data.empty:
            logger.warning(f"Empty testing data: {filename}")

        # Save the training and testing data
        train_data.to_csv(f"{train_dir}/{filename}", index=False)
        test_data.to_csv(f"{test_dir}/{filename}", index=False)

        logger.info(f"Saved training data to: {train_dir}/{filename}")
        logger.info(f"Saved testing data to: {test_dir}/{filename}")

    if args.scale_features:
        scale_cols = args.scale_features
        logger.info("Normalizing data")

        # Do a pass through the training data to and use partial_fit to train the scaler
        # This allows us to train the scaler without having all files in memory at once
        scaler = preprocessing.StandardScaler()
        print("Training the scaler on the training data")
        for filename in tqdm(os.listdir(train_dir)):
            df = pd.read_csv(f"{train_dir}/{filename}")
            if df.empty:
                continue
            scaler.partial_fit(df[scale_cols])

        # Save the scaler
        scaler_filename = f"{args.destination}/scaler.pkl"
        with open(scaler_filename, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved scaler to: {scaler_filename}")

        # Scale the training data
        print("Scaling the training data")
        for filename in tqdm(os.listdir(train_dir)):
            df = pd.read_csv(f"{train_dir}/{filename}")
            if df.empty:
                continue
            scaled_data = scaler.transform(df[scale_cols])
            df[scale_cols] = pd.DataFrame(scaled_data, columns=scale_cols)
            df.to_csv(f"{train_dir}/{filename}", index=False)

        # Scale the testing data
        print("Scaling the testing data")
        for filename in tqdm(os.listdir(test_dir)):
            df = pd.read_csv(f"{test_dir}/{filename}")
            if df.empty:
                continue
            scaled_data = scaler.transform(df[scale_cols])
            df[scale_cols] = pd.DataFrame(scaled_data, columns=scale_cols)
            df.to_csv(f"{test_dir}/{filename}", index=False)
