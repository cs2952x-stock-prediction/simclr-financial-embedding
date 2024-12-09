import argparse
import logging
import os
import pprint
from datetime import datetime

import pandas as pd
from tqdm import tqdm

############################# GLOBAL VARIABLES #####################################################

DEFAULT_DST = "data/interim/polygon"  # the folder containing the output files
DEFAULT_SRC = "data/raw/polygon"  # the folder containing the source files

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
        description="Process the raw 5-Minutely dataset from Kaggle into a more usable format."
        "This script will drop or interpolate missing values, and remove any remaining NaN values."
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
        help="Overwrite existing files",
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
        choices=["drop", "linear", "pad", "time", "ffill"],
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
    Process the raw 5-minutely dataset from Polygon into a more usable format.

    Args:
    - config (dict): The configuration dictionary
    """
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"Starting Kaggle data cleaning process at {timestamp}")
    logger.info(f"Configuration:\n{pprint.pformat(config)}\n")

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

    print("For each file, fill missing values, removing nan values, ensuring sorted...")
    files = os.listdir(config["source"])
    for file in tqdm(files):
        df = pd.read_csv(f"{config['source']}/{file}")
        symbol = file.split(".")[0]

        # Handle missing values
        if config["fill_method"] not in ["drop", "ffill"]:
            df[COLUMNS_TO_FILL] = df[COLUMNS_TO_FILL].interpolate(  # type: ignore
                limit_direction="both",
                method=config["fill_method"],
                axis=0,
            )

        if config["fill_method"] == "ffill":
            df[COLUMNS_TO_FILL] = df[COLUMNS_TO_FILL].ffill()

        # drop any remaining NaN values
        n = len(df)
        df.dropna(inplace=True)
        n_drop = n - len(df)
        if n_drop > 0:
            logger.warning(f"Removed {n - len(df)} NaN values from {file}")

        # Fail if the data is empty
        if len(df) == 0:
            logger.error(f"Empty data for {file}")
            continue

        df.sort_values(by=["timestamp"], inplace=True)  # type: ignore
        df.to_csv(f"{config['destination']}/{symbol}.csv", index=False)
        logger.debug(f"Saved {symbol}.csv to {config['destination']}")


if __name__ == "__main__":
    # Get command line arguments
    args = get_args()

    # Configure the logger
    configure_logger(args.log_level, args.log_file)

    # Log the arguments
    formatted_args = "\n\t".join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments:\n\t{formatted_args}\n")

    # Run the main function
    config = {k: v for k, v in vars(args).items() if k not in ["log_level", "log_file"]}
    main(config)
