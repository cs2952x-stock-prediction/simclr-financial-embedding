import argparse
import logging
import os
import pprint
import time
import traceback
from datetime import datetime
from urllib.parse import parse_qs, parse_qsl, urlparse

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from kagglehub.gcs_upload import tqdm
from matplotlib import pyplot as plt
from pandas.io.formats.format import math

############################# GLOBAL VARIABLES #####################################################

DEFAULT_DST = f"data/raw/polygon"  # the folder containing the output files

# Logger
logger = logging.getLogger(__name__)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DEFAULT_LOG_FILE = f"logs/polygon-download_{timestamp}.log"

# load the environment variables
# (mainly just the polygon API key)
load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY")

# aggragate response parameters
AGGREGATE_RESPONSE_ADJUSTED = "true"
AGGREGATE_RESPONSE_ORDER = "asc"
AGGREGATE_RESPONSE_LIMIT = 5_000

# map the column names from the API to the desired column names
# (this is useful for standardizing the column names across different APIs)
COLUMN_MAP = {
    "t": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "n": "transactions",
    "vw": "volume_weighted_average_price",
}

############################## FUNCTIONS ###########################################################


def fetch_nyse_symbols():
    """
    Fetches all active NYSE symbols from the Polygon API.


    Returns:
    - list: List of NYSE symbols.
    """

    symbols = []

    url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "exchange": "XNYS",
        "active": "true",
        "limit": 1000,  # Higher limit to reduce the number of API calls
        "apiKey": API_KEY,
    }

    while True:
        logger.debug(f"Requesting URL: {url}")
        logger.debug(f"Requesting Params: {params}")
        response = requests.get(url, params=params)

        # Code 200: Success
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            for ticker in results:
                if ticker.get("active"):
                    symbols.append(ticker.get("ticker"))

            # Get the next_url from the response
            next_url = data.get("next_url")
            if not next_url:
                break

            logger.debug(f"Found another page with new_url: {next_url}")

            # Parse the next_url to extract parameters
            parsed_url = urlparse(next_url)
            query_params = parse_qs(parsed_url.query)

            # Update the 'cursor' parameter for the next request
            cursor = query_params.get("cursor", [None])[0]

            if cursor:
                params = {
                    "market": "stocks",
                    "exchange": "XNYS",
                    "active": "true",
                    "limit": 1000,
                    "cursor": cursor,
                    "apiKey": API_KEY,
                }
            else:
                raise Exception(
                    "While fetching NYSE symbols, the next_url did not contain a cursor"
                )

        # Code 429: Too many requests --- rate limit by waiting
        elif response.status_code == 429:
            wait_time = int(response.headers.get("Retry-After", 1))
            print(
                f"Warning: Rate limit exceeded when fetching NYSE symbols. Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

        # Other status codes: raise an exception
        else:
            raise Exception(
                f"Status code {response.status_code} while fetching NYSE symbols: {response.text}"
            )

    logger.info(f"{len(symbols)} NYSE Symbols Retrieved")
    return symbols


def get_args():
    """
    Parse command-line arguments for the script.

    Returns:
    - argparse.Namespace: Parsed arguments.
    """

    # Defaults
    default_symbols = None  # if None fetches all symbols
    default_end = datetime.today().strftime("%Y-%m-%d")
    default_start = (datetime.today() - relativedelta(years=10)).strftime("%Y-%m-%d")
    default_timespan = "minute"
    default_multiplier = 5

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        help="Space-separated list of symbols for which to download data",
        default=default_symbols,
    )
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date of the requested data in 'YYYY-MM-DD' format",
        default=default_start,
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date of the requested data in 'YYYY-MM-DD' format",
        default=default_end,
    )
    parser.add_argument(
        "--timespan",
        type=str,
        choices=["second", "minute", "hour", "day", "week"],
        help="Timespan for each data point (second, minute, hour, day, week)",
        default=default_timespan,
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        help="Multiplier for the timespan (e.g., 5 for 5-minute data)",
        default=default_multiplier,
    )
    parser.add_argument(
        "--destination",
        type=str,
        help="Destination folder to save the downloaded data",
        default=DEFAULT_DST,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
        default="INFO",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        help="Path to the log file",
        default=DEFAULT_LOG_FILE,
    )

    return parser.parse_args()


def ensure_api_key_in_url(url, api_key):
    """
    Ensures that the API key is included in the URL.
    """
    parsed_url = urlparse(url)
    query_params = dict(parse_qsl(parsed_url.query))
    query_params["apiKey"] = api_key
    new_query = "&".join([f"{k}={v}" for k, v in query_params.items()])
    new_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{new_query}"
    return new_url


def plot_weekly_distribution(df, destination):
    """
    Plots the distribution of data entries over a week using a bar plot.
    Useful for checking if there are gaps in the data distribution.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'timestamp' column with datetime types.
    - destination (str): Path to save the plot.
    """
    weekday = df["timestamp"].dt.weekday  # Returns 0 (Monday) to 6 (Sunday)
    hour = df["timestamp"].dt.hour
    overall_hour = weekday * 24 + hour  # Maps to [0, 167]

    # Create a Series with counts for each unique value in [0, 167], filling missing values with 0
    full_range = pd.Series(0, index=np.arange(168))  # Full range of hours in a week
    hour_counts = overall_hour.value_counts().sort_index()
    full_counts = full_range.add(hour_counts, fill_value=0)

    # Plot the distribution
    plt.figure(figsize=(12, 6))
    plt.bar(full_counts.index, full_counts.to_numpy(), color="skyblue", width=1.0)
    plt.title("Weekly Distribution of Data")
    plt.xlabel("Hour of the Week (0=Mon 00:00, 167=Sun 23:00)")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(
        ticks=np.arange(0, 168, 24),
        labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )

    # Save the plot to a file
    plt.savefig(destination)
    logger.info(f"Weekly distribution plot saved to {destination}")


def plot_overall_distribution(df, destination):
    """
    Plots the distribution of data entries over all time using a histogram.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'timestamp' column with datetime types.
    - destination (str): Path to save the plot and histogram data.
    """
    datetimes = df["timestamp"]

    # Plot a histogram of all timestamps
    plt.figure(figsize=(12, 6))
    plt.hist(
        datetimes, bins=200, color="skyblue", edgecolor="black"
    )  # NOTE: Adjust bins as needed for granularity
    plt.title("All-Time Distribution of Data")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    plt.savefig(destination)
    logger.info(f"All-time distribution plot saved to {destination}")


def fetch_symbol_bars(symbol, start_date, end_date, timespan, multiplier):
    """
    Fetches historical price data for a given stock symbol from the Polygon API.
    Specifically, this function fetches aggregate bars for a given symbol within a specified time range.

    Parameters:
    - symbol (str): Stock ticker symbol to fetch.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.
    - timespan (str): Size of the time window ('minute', 'hour', 'day', 'week').
    - multiplier (int): Size of the timespan multiplier.

    Returns:
    - df (pd.DataFrame): DataFrame containing the fetched data.
    """
    # parse the arguments as-needed:

    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Calculate the number of requests to make based on the total duration
    # and the maximum number of results per request
    total_duration = end_date - start_date
    interval_duration = pd.Timedelta(f"{multiplier} {timespan}")
    assert isinstance(interval_duration, pd.Timedelta)
    n_intervals = math.ceil(total_duration / interval_duration)
    n_requests = math.ceil(n_intervals / AGGREGATE_RESPONSE_LIMIT)

    logger.info(
        f"Fetching data for {symbol} from {start_date} to {end_date} with {multiplier} {timespan} intervals"
    )
    logger.info(
        f"Making {n_requests} requests and expecting (max) {n_intervals} results"
    )

    all_results = []
    base_url = (
        f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}"
    )
    for i in tqdm(range(n_requests)):
        # Calculate the start and end dates for the current request
        current_start = start_date + i * AGGREGATE_RESPONSE_LIMIT * interval_duration
        current_end = min(
            end_date, current_start + AGGREGATE_RESPONSE_LIMIT * interval_duration
        )

        # Convert to millisecond timestamp
        start_timestamp = int(current_start.timestamp() * 1000)
        end_timestamp = int(current_end.timestamp() * 1000)

        url = f"{base_url}/{start_timestamp}/{end_timestamp}"
        params = {
            "adjusted": AGGREGATE_RESPONSE_ADJUSTED,
            "limit": AGGREGATE_RESPONSE_LIMIT,  # Max limit
            "order": AGGREGATE_RESPONSE_ORDER,
            "apiKey": API_KEY,
        }

        # Continue fetching pages of results and appending them to the list
        while True:
            logger.debug(f"Requesting URL: {url}")
            logger.debug(f"Requesting Params: {params}")
            response = requests.get(url, params=params)

            # Code 200: Success
            if response.status_code == 200:
                data = response.json()  # python dictionary
                results = data.get("results", [])  # default value is an empty list
                all_results.extend(results)
                next_url = data.get("next_url")

                # Break if not results on the current page
                if len(results) == 0:
                    break

                # Break if no more pages
                if not next_url:
                    break

                # If there are more pages, request more pages
                # update the url and params
                logger.debug(f"Found another page with new_url: {next_url}")
                url = ensure_api_key_in_url(next_url, API_KEY)
                params = None  # No params after the first request

            # Code 429: Too many requests --- rate limit by waiting
            elif response.status_code == 429:
                wait_time = int(response.headers.get("Retry-After", 1))
                print(
                    f"Warning: Rate limit exceeded when fetching bars for {symbol}. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)

            # Other status codes: raise an exception
            else:
                raise Exception(
                    f"Status code {response.status_code} while fetching bars for {symbol}: {response.text}"
                )

            # Delay to avoid rate limits
            time.sleep(0.01)

    # Convert the list of results to a DataFrame
    logger.info(f"Fetched {len(all_results)} results for {symbol}")
    return pd.DataFrame(all_results)


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
    Download historical price data for a list of stock symbols from the Polygon API.

    Args:
    - config (dict): The configuration dictionary
    """
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    logger.info(f"Starting Polygon data download process at {timestamp}")
    logger.info(f"Configuration:\n{pprint.pformat(config)}")

    # Fetch all NYSE symbols if no symbols are provided
    if not config["symbols"]:
        logger.warning("Symbols not provided. Fetching all NYSE symbols...")
        config["symbols"] = fetch_nyse_symbols()

    # pd.Timedelta can't handle 'week' timespan, so convert to 'day'
    if config["timespan"] == "week":
        config["timespan"] = "day"
        config["multiplier"] *= 7

    # Get the time interval based on the timespan and multiplier
    interval = pd.Timedelta(f"{config['multiplier']} {config['timespan']}")

    # Ensure the destination folder path exists
    if not os.path.exists(config["destination"]):
        os.makedirs(config["destination"])

    # Download data for each symbol
    print(f"Downloading data for {len(config['symbols'])} symbols...")
    for i, symbol in enumerate(config["symbols"]):
        try:
            print(f"Fetching data for {symbol} ({i+1}/{len(config['symbols'])})...")
            df = fetch_symbol_bars(
                symbol,
                config["start_date"],
                config["end_date"],
                config["timespan"],
                config["multiplier"],
            )

            # standardize the column names
            df.rename(columns=COLUMN_MAP, inplace=True)

            # Sort the DataFrame by timestamp in ascending order
            df.sort_values("timestamp", inplace=True, ascending=True)

            # Drop duplicates
            n = len(df)
            df.drop_duplicates(subset=["timestamp"], inplace=True)
            n_duplicates = n - len(df)
            if n_duplicates > 0:
                logging.warning(f"Removed {n_duplicates} duplicate rows for {symbol}")

            # Drop NaN values
            n = len(df)
            df.dropna(inplace=True)
            n_nan = n - len(df)
            if n_nan > 0:
                logging.warning(f"Removed {n_nan} NaN rows for {symbol}")

            # Convert the timestamp to a datetime object
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            # Count the number of gaps in the data
            # The difference between adjacent timestamps should be equal to the timespan
            time_diffs = df["timestamp"].diff().dt.total_seconds()
            n_gaps = (time_diffs != interval.total_seconds()).sum()
            if n_gaps > 0:
                logging.warning(f"Found {n_gaps} gaps in the data for {symbol}")

            # Reorder the columns to [timestamp, ...rest]
            df = df[["timestamp"] + [col for col in df.columns if col != "timestamp"]]

            # Save the DataFrame to a CSV file
            logging.info(
                f"Saving data for {symbol} to {config['destination']}/{symbol}.csv with {len(df)} rows"
            )
            df.to_csv(f"{config['destination']}/{symbol}.csv", index=False)

            # NOTE: Uncomment to plot the distributions entries
            # useful for checking for gaps in the distribution are
            # plot_weekly_distribution(
            #     df, f"{config['destination}/{symbol}_weekly_distribution.png"
            # )
            # plot_overall_distribution(
            #     df, f"{config['destination}/{symbol}_overall_distribution.png"
            # )

            time.sleep(0.5)  # Adjust sleep time based on your rate limit

        except Exception as e:
            logger.error(f"Error: An error occurred while processing {symbol}")
            logger.error(e)
            logger.error(traceback.format_exc())


if __name__ == "__main__":
    # Parse command-line arguments
    args = get_args()

    # start logger
    configure_logger(args.log_level, args.log_file)

    # Log the arguments
    formatted_args = "\n\t".join([f"{k}: {v}" for k, v in vars(args).items()])
    logger.info(f"Arguments:\n\t{formatted_args}\n")

    config = vars(args)
    main(config)
