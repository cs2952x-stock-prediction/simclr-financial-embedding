import argparse
import logging
import os
from datetime import datetime

import pandas as pd
from tqdm import tqdm

############################# GLOBAL VARIABLES #####################################################

THIS_FILE = os.path.abspath(__file__)  # the absolute path the current file
THIS_DIR = os.path.dirname(THIS_FILE)  # the folder containing the current file
DEFAULT_DST = f"{THIS_DIR}/out"  # the folder containing the output files
DEFAULT_SRC = os.path.abspath(
    "{THIS_DIR}/../../raw/kaggle_sp500/out/sp500_stocks.csv"  # the folder containing the source files
)
LOGS_DIR = f"{THIS_DIR}/logs"  # the folder containing the logs

COL_TO_INTERPOLATE = [
    "adj_close",
    "close",
    "high",
    "low",
    "open",
    "volume",
]

############################## FUNCTIONS ###########################################################


def get_args():
    """
    Get the command line arguments

    Returns:
    - argparse.Namespace: The command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Process the raw SP500 dataset from Kaggle into a more usable format."
        "This script will separate the data by symbol, drop or interpolate missing values, and remove any remaining NaN values."
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="The directory to download the dataset to",
        default=DEFAULT_DST,
    )
    parser.add_argument(
        "--source",
        type=str,
        help="The directory to load the dataset from",
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
        "--fill_method",
        type=str,
        default="drop",
        choices=["drop", "linear", "pad", "time"],
        help="The method to fill missing values",
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
        logger.error(f"Source file not found: {args.source}")
        raise FileNotFoundError(f"Source file not found: {args.source}")

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

    # Load data
    print(f"Loading data from {args.source}...")
    df = pd.read_csv(args.source)
    logger.info(f"Loaded data from {args.source}")
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Data columns: {df.columns}")
    logger.info(f"Data head:\n{df.head()}")

    # Separate series by symbol
    print(
        "Separating series by symbol, filling missing values, removing nan values, ensuring sorted..."
    )
    symbols = df["symbol"].unique()
    for symbol in tqdm(symbols):
        symbol_df = df[df["symbol"] == symbol]  # filter out only the current symbol
        assert isinstance(symbol_df, pd.DataFrame), "symbol_df is not a DataFrame"
        symbol_df = symbol_df.drop(columns=["symbol"])  # drop the symbol column

        # Handle missing values
        if args.fill_method != "drop":
            symbol_df[COL_TO_INTERPOLATE] = symbol_df[COL_TO_INTERPOLATE].interpolate(
                limit_direction="both",
                method=args.fill_method,
                axis=0,
            )

        # drop any remaining NaN values
        n = len(symbol_df)
        symbol_df.dropna(inplace=True)
        n_drop = n - len(symbol_df)
        if n_drop > 0:
            logger.warning(f"Removed {n - len(symbol_df)} NaN values from {symbol}")

        symbol_df.sort_values(by=["timestamp"], inplace=True)  # ensure sorted
        symbol_df.to_csv(f"{args.destination}/{symbol}.csv", index=False)
        logger.info(f"Saved {symbol}.csv to {args.destination}")
