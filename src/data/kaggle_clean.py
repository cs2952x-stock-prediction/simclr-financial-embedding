import argparse
import logging
import os
import pprint
from datetime import datetime

import pandas as pd
from tqdm import tqdm

############################# GLOBAL VARIABLES #####################################################

DEFAULT_DST = "data/interim/kaggle"  # the folder containing the output files
DEFAULT_SRC = "data/raw/kaggle/sp500_stocks.csv"  # the default source file

logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_LOG_FILE = f"logs/kaggle-clean_{timestamp}.log"

COLUMNS_TO_FILL = [
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
        "--fill_method",
        type=str,
        default="drop",
        choices=["drop", "linear", "pad", "time"],
        help="The method to fill missing values",
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
    logger.setLevel(log_level)


############################## MAIN FUNCTION ######################################################


def main(config):
    """
    Process the raw SP500 dataset from Kaggle into a more usable format.

    Args:
    - config (dict): The configuration dictionary
    """
    logger.info(f"Command config:\n{pprint.pformat(config)}")

    # If the source file does not exist, raise an error
    if not os.path.exists(config["source"]):
        logger.error(f"Source file not found: {config['source']}")
        raise FileNotFoundError(f"Source file not found: {config['source']}")

    # Ensure the destination folder exists
    if not os.path.exists(config["destination"]):
        os.makedirs(config["destination"])
        logger.info(f"Created destination folder: {config['destination']}")

    # If the destination folder is not empty and force is not set, raise an error
    if os.listdir(config["destination"]) and not config["force"]:
        logger.error(f"Destination folder is not empty: {config['destination']}")
        raise FileExistsError(
            f"Destination folder is not empty: {config['destination']}. Use --force to overwrite."
        )

    # Load data
    print(f"Loading data from {config['source']}...")
    df = pd.read_csv(config["source"])
    logger.info(f"Loaded data from {config['source']}")
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
        if config["fill_method"] != "drop":
            symbol_df[COLUMNS_TO_FILL] = symbol_df[COLUMNS_TO_FILL].interpolate(
                limit_direction="both",
                method=config["fill_method"],
                axis=0,
            )

        # drop any remaining NaN values
        n = len(symbol_df)
        symbol_df.dropna(inplace=True)
        n_drop = n - len(symbol_df)
        if n_drop > 0:
            logger.warning(f"Removed {n - len(symbol_df)} NaN values from {symbol}")

        symbol_df.sort_values(by=["timestamp"], inplace=True)  # ensure sorted
        symbol_df.to_csv(f"{config['destination']}/{symbol}.csv", index=False)
        logger.info(f"Saved {symbol}.csv to {config['destination']}")


if __name__ == "__main__":
    # Get command line arguments
    args = get_args()
    config = vars(args)

    # Start logging
    configure_logger(config["log_level"], config["log_file"])

    # Run the main function
    main(config)
