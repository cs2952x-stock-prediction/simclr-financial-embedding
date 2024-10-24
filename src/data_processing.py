import argparse
import os

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str, required=True, help="Path to raw data file")
    parser.add_argument(
        "-d",
        "--cyclic_day",
        type=bool,
        default=False,
        help="Whether to add a cyclic day feature",
    )
    parser.add_argument(
        "-w",
        "--cyclic_week",
        type=bool,
        default=False,
        help="Whether to add a cyclic week feature",
    )
    parser.add_argument(
        "-m",
        "--cyclic_month",
        type=bool,
        default=False,
        help="Whether to add a cyclic month feature",
    )
    parser.add_argument(
        "-q",
        "--cyclic_quarter",
        type=bool,
        default=False,
        help="Whether to add a cyclic quarter feature",
    )
    parser.add_argument(
        "-y",
        "--cyclic_year",
        type=bool,
        default=False,
        help="Whether to add a cyclic year feature",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        help="Path to save processed data",
    )
    return parser.parse_args()


def process_data(data: pd.DataFrame, **kwargs):
    if kwargs["cyclic_day"]:
        data["day_sin"] = data["datetime"]

    return data


# The data should have the form:


def main():
    args = parse_args()

    # Give a warning if the destination file already exists
    if os.path.exists(args.destination):
        print(
            f"Warning: The file {args.destination} already exists. It will be overwritten."
        )
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != "y":
            print("Exiting...")
            return

    raw_data = pd.read_csv(args.raw)
    processed_data = process_data(raw_data, **vars(args))
    processed_data.to_csv(args.destination, index=False)


if __name__ == "__main__":
    main()
