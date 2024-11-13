import argparse
import logging
import os
from datetime import datetime

import kagglehub
import pandas as pd

############################# GLOBAL VARIABLES #####################################################

THIS_FILE = os.path.abspath(__file__)  # the absolute path the current file
THIS_DIR = os.path.dirname(THIS_FILE)  # the folder containing the current file
DEFAULT_DST = f"{THIS_DIR}/out"  # the folder containing the output files

STOCKS_FILENAME = "sp500_stocks.csv"  # the file containing the stock data (name set by Kaggle, do not change)
COMPANIES_FILENAME = "sp500_companies.csv"  # the file containing the company data (name set by Kaggle, do not change)
INDEX_FILENAME = "sp500_index.csv"  # the file containing the index data (name set by Kaggle, do not change)
LOGS_DIR = f"{THIS_DIR}/logs"  # the folder to store the logs

# The columns to rename in the stock data
STOCK_COLUMNS = {
    "date": "timestamp",
    "adj close": "adj_close",
}

# The columns to rename in the companies data
COMPANIES_COLUMNS = {
    "shortname": "short_name",
    "longname": "long_name",
    "currentprice": "current_price",
    "marketcap": "market_cap",
    "revenuegrowth": "revenue_growth",
    "fulltimeemployees": "full_time_employees",
    "longbusinesssummary": "long_business_summary",
}

# The columns to rename in the index data
INDEX_COLUMNS = {
    "date": "timestamp",
}

############################## FUNCTIONS ###########################################################


def get_args():
    """
    Get the command line arguments.

    Returns:
    - argparse.Namespace: The command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Download the daily SP500 dataset from Kaggle"
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="The directory to download the dataset to",
        default=DEFAULT_DST,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download and overwrite existing files",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=["sp500_stocks.csv", "sp500_companies.csv", "sp500_index.csv"],
        help="The files to download",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="The log level to use",
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


def process_stock_data(stock_file):
    """
    Process the stock data file in-place.

    Args:
    - stock_file (str): The path to the stock data file
    """
    print("Processing stock data file: ", stock_file)
    data = pd.read_csv(stock_file)
    data.columns = list(map(str.lower, data.columns))
    data.rename(columns=STOCK_COLUMNS, inplace=True)
    data = data[["timestamp"] + [col for col in data.columns if col != "timestamp"]]
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.to_csv(stock_file, index=False)
    logger.info(f"Processed stock data file: {stock_file}")


def process_company_data(companies_file):
    """
    Process the companies data file in-place.

    Args:
    - companies_file (str): The path to the companies data file
    """
    print("Processing companies data file: ", companies_file)
    data = pd.read_csv(companies_file)
    data.columns = list(map(str.lower, data.columns))
    data.rename(columns=COMPANIES_COLUMNS, inplace=True)
    data.to_csv(companies_file, index=False)
    logger.info(f"Processed companies data file: {companies_file}")


def process_index_data(index_file):
    """
    Process the index data file in-place.

    Args:
    - index_file (str): The path to the index data file
    """
    print("Processing index data file: ", index_file)
    data = pd.read_csv(index_file)
    data.columns = list(map(str.lower, data.columns))
    data.rename(columns=INDEX_COLUMNS, inplace=True)
    data = data[["timestamp"] + [col for col in data.columns if col != "timestamp"]]
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.to_csv(index_file, index=False)
    logger.info(f"Processed index data file: {index_file}")


############################## Execution ###########################################################

if __name__ == "__main__":
    # Get the command line arguments
    args = get_args()

    # build logger
    logger = build_logger(args.log_level)

    logger.info(
        f"Arguments:\n\t{'\n\t'.join([f'{k}: {v}' for k, v in vars(args).items()])}"
    )

    # Validating the provided files
    for file in args.files:

        # Check if the file is one of the three possible filenames
        if file not in [STOCKS_FILENAME, COMPANIES_FILENAME, INDEX_FILENAME]:
            logger.error(f"Invalid file: {file}")
            raise ValueError(f"Invalid file: {file}")

        # Check if we are about to overwrite a file (aborts if force is not set)
        new_file = f"{args.destination}/{file}"
        if os.path.exists(new_file) and not args.force:
            logger.error(f"File {new_file} already exists. Use --force to overwrite")
            raise FileExistsError(
                f"File {new_file} already exists. Use --force to overwrite"
            )

    # make sure that the destination directory exists
    if not os.path.exists(args.destination):
        os.makedirs(args.destination)
        logger.info(f"Created destination directory: {args.destination}")

    # Download latest kaggle data
    # Unfortunately, we can't control the initial destination directory of the files
    print("Downloading dataset from Kaggle...")
    download_dir = kagglehub.dataset_download(
        "andrewmvd/sp-500-stocks", force_download=True
    )
    downloaded_files = os.listdir(download_dir)
    logger.info(f"Dataset files downloaded to: {download_dir}")
    logger.info(f"Files:\n\t{'\n\t'.join(downloaded_files)}")

    # Move the dataset to this directory
    # move everything in the returned path to this directory
    print(f"Moving files to destination: {args.destination}")

    for file in args.files:
        old_file = f"{download_dir}/{file}"
        new_file = f"{args.destination}/{file}"

        # Check if the file exists
        if file not in downloaded_files:
            logger.warning(f"File {file} not found in downloaded files (skipping)")
            continue

        # Move the file
        os.rename(old_file, new_file)
        logger.info(f"Moved {old_file} to {new_file}")

        # Process the files
        if file == STOCKS_FILENAME:
            process_stock_data(new_file)

        if file == COMPANIES_FILENAME:
            process_company_data(new_file)

        if file == INDEX_FILENAME:
            process_index_data(new_file)
