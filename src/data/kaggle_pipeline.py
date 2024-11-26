import argparse
import logging
import os
import pprint
from datetime import datetime

import kaggle_clean
import kaggle_download
import kaggle_process
import yaml

############################# GLOBAL VARIABLES #####################################################

# Config
DEFAULT_CONFIG = "configs/kaggle-pipeline.yaml"

# Logger
logger = logging.getLogger(__name__)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_LOG_FILE = f"logs/kaggle-pipeline_{timestamp}.log"

############################## FUNCTIONS ###########################################################


def get_args():
    """
    Parse command line arguments.

    Returns:
    - args: the parsed arguments
    """
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to config file (contains config values specific to this run)",
    )
    arg_parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="The log level to use",
    )
    arg_parser.add_argument(
        "--log_file",
        type=str,
        default=DEFAULT_LOG_FILE,
        help="The log file to use",
    )

    return arg_parser.parse_args()


def load_config(args):
    """
    Load a configuration file.

    Args:
    - args: the parsed arguments

    Returns:
    - config: the configuration dictionary
    """

    # The config file contains the specific configuration for this run
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        args.config = None

    return config


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


############################## MAIN FUNCTION ######################################################


def main(config):
    """
    Download, clean, and process the Kaggle dataset.

    Args:
    - config (dict): The configuration dictionary
    """
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"Starting Kaggle pipeline at {timestamp}")
    logger.info(f"Configuration:\n{pprint.pformat(config)}\n")

    kaggle_download.main(config["download"])
    kaggle_clean.main(config["clean"])
    kaggle_process.main(config["process"])


if __name__ == "__main__":
    # Parse command line arguments
    args = get_args()

    # Configure the logger
    configure_logger(args.log_level, args.log_file)
    formatted_args = "\n\t".join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments:\n\t{formatted_args}\n")

    # Load configuration
    config = load_config(args)

    main(config)
