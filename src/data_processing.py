import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn import random_projection


def parse_args():
    parser = argparse.ArgumentParser()

    # the paths to retreive + save the data
    parser.add_argument("--raw", type=str, required=True, help="Path to raw data file")
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        help="The directory in which to save the processed data and other outputs",
    )

    # the temporal features to add the the data
    parser.add_argument(
        "-d",
        "--cyclic_day",
        action="store_true",
        help="Whether to add a cyclic day feature",
    )
    parser.add_argument(
        "-w",
        "--cyclic_week",
        action="store_true",
        help="Whether to add a cyclic week feature",
    )
    parser.add_argument(
        "-m",
        "--cyclic_month",
        action="store_true",
        help="Whether to add a cyclic month feature",
    )
    parser.add_argument(
        "-q",
        "--cyclic_quarter",
        action="store_true",
        help="Whether to add a cyclic quarter feature",
    )
    parser.add_argument(
        "-y",
        "--cyclic_year",
        action="store_true",
        help="Whether to add a cyclic year feature",
    )

    # the settings for encoding the ticker symbols
    parser.add_argument(
        "--symbol_encoding",
        type=int,
        default=None,
        help="The number of dimensions with which to encode the symbol column",
    )
    return parser.parse_args()


def process_data(data: pd.DataFrame, **kwargs):
    data = data.copy()

    # Generate the temporal features for the data
    datetime = pd.to_datetime(data["datetime"])
    data.drop(columns=["datetime"], inplace=True)

    # convert a time column with seconds since the minimum date
    data["time"] = (datetime - datetime.min()).dt.total_seconds()

    portion_of_day = (
        datetime.dt.hour
        + datetime.dt.minute / 60
        + datetime.dt.second / 3600
        + datetime.dt.microsecond / (3600 * 1e6)
    ) / 24

    if kwargs["cyclic_day"]:
        data["day_sin"] = np.sin(2 * np.pi * portion_of_day)
        data["day_cos"] = np.cos(2 * np.pi * portion_of_day)

    portion_of_week = (datetime.dt.day_of_week + portion_of_day) / 7

    if kwargs["cyclic_week"]:
        data["week_sin"] = np.sin(2 * np.pi * portion_of_week)
        data["week_cos"] = np.cos(2 * np.pi * portion_of_week)

    portion_of_month = (datetime.dt.day + portion_of_day) / datetime.dt.days_in_month

    if kwargs["cyclic_month"]:
        data["month_sin"] = np.sin(2 * np.pi * portion_of_month)
        data["month_cos"] = np.cos(2 * np.pi * portion_of_month)

    # NOTE: this is approximate, might need to be improved (probably good enough)
    quarter_length = 365.25 / 4
    portion_of_quarter = (
        datetime.dt.quarter * 3 + datetime.dt.day + portion_of_day
    ) / quarter_length

    if kwargs["cyclic_quarter"]:
        data["quarter_sin"] = np.sin(2 * np.pi * portion_of_quarter)
        data["quarter_cos"] = np.cos(2 * np.pi * portion_of_quarter)

    # NOTE: this is approximate (probably good enough)
    portion_of_year = (datetime.dt.day_of_year + portion_of_day) / 365.25

    if kwargs["cyclic_year"]:
        data["year_sin"] = np.sin(2 * np.pi * portion_of_year)
        data["year_cos"] = np.cos(2 * np.pi * portion_of_year)

    # Apply one-hot encding to the symbol column
    # if symbol_encoding is not None, then the sparse one-hot encodings will be reduced using random projection

    one_hot_encoding = pd.get_dummies(data["symbol"], sparse=True)
    data.drop(columns=["symbol"], inplace=True)

    # print the number of rows/columns in the on-hot encoding
    print(f"one_hot_encoding shape: {one_hot_encoding.shape}")

    if kwargs["symbol_encoding"] is not None:
        n_dims = kwargs["symbol_encoding"]
        encoder = random_projection.GaussianRandomProjection(n_components=n_dims)
        one_hot_encoding = encoder.fit_transform(one_hot_encoding)

        # conver the numpy array to a pandas DataFrame
        one_hot_encoding = pd.DataFrame(
            data=one_hot_encoding,
            columns=[f"symbol_dim{i}" for i in range(n_dims)],
        )

        # save the projection model to the same directory as the processed data
        with open(os.path.join(kwargs["destination"], "symbol_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)

    data = pd.concat([data, one_hot_encoding], axis=1)

    return data


# The data should have the form:


def main():
    args = parse_args()

    save_file = os.path.basename(args.raw)
    save_path = os.path.join(args.destination, save_file)

    # Give a warning if the destination file already exists
    if os.path.exists(save_path):
        print(f"Warning: {save_path} already exists. It will be overwritten.")
        response = input("Do you want to continue? (y/n): ")
        if response.lower() != "y":
            print("Exiting...")
            return

    raw_data = pd.read_csv(args.raw)
    processed_data = process_data(raw_data, **vars(args))
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(processed_data.head())
    processed_data.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
