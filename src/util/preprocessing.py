import argparse
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.random_projection import GaussianRandomProjection


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


def get_cyclic_day(datetimes: Series) -> Tuple[Series, Series]:
    portion_of_day = (
        datetimes.dt.hour
        + datetimes.dt.minute / 60
        + datetimes.dt.second / 3600
        + datetimes.dt.microsecond / (3600 * 1e6)
    ) / 24

    day_sin = np.sin(2 * np.pi * portion_of_day)
    day_cos = np.cos(2 * np.pi * portion_of_day)

    return day_sin, day_cos


def get_cyclic_week(datetimes: Series) -> Tuple[Series, Series]:
    portion_of_day = (
        datetimes.dt.hour
        + datetimes.dt.minute / 60
        + datetimes.dt.second / 3600
        + datetimes.dt.microsecond / (3600 * 1e6)
    ) / 24

    portion_of_week = (datetimes.dt.day_of_week + portion_of_day) / 7

    week_sin = np.sin(2 * np.pi * portion_of_week)
    week_cos = np.cos(2 * np.pi * portion_of_week)

    return week_sin, week_cos


def get_cyclic_month(datetimes: Series) -> Tuple[Series, Series]:
    portion_of_day = (
        datetimes.dt.hour
        + datetimes.dt.minute / 60
        + datetimes.dt.second / 3600
        + datetimes.dt.microsecond / (3600 * 1e6)
    ) / 24

    portion_of_month = (datetimes.dt.day + portion_of_day) / datetimes.dt.days_in_month

    month_sin = np.sin(2 * np.pi * portion_of_month)
    month_cos = np.cos(2 * np.pi * portion_of_month)

    return month_sin, month_cos


def get_cyclic_quarter(datetimes: Series) -> Tuple[Series, Series]:
    portion_of_day = (
        datetimes.dt.hour
        + datetimes.dt.minute / 60
        + datetimes.dt.second / 3600
        + datetimes.dt.microsecond / (3600 * 1e6)
    ) / 24

    quarter_length = 365.25 / 4
    portion_of_quarter = (
        datetimes.dt.quarter * 3 + datetimes.dt.day + portion_of_day
    ) / quarter_length

    quarter_sin = np.sin(2 * np.pi * portion_of_quarter)
    quarter_cos = np.cos(2 * np.pi * portion_of_quarter)

    return quarter_sin, quarter_cos


def get_cyclic_year(datetimes: Series) -> Tuple[Series, Series]:
    portion_of_day = (
        datetimes.dt.hour
        + datetimes.dt.minute / 60
        + datetimes.dt.second / 3600
        + datetimes.dt.microsecond / (3600 * 1e6)
    ) / 24

    portion_of_year = (datetimes.dt.day_of_year + portion_of_day) / 365.25

    year_sin = np.sin(2 * np.pi * portion_of_year)
    year_cos = np.cos(2 * np.pi * portion_of_year)

    return year_sin, year_cos


def get_nd_embedding(
    series: Series, n_dims: int
) -> Tuple[DataFrame, GaussianRandomProjection]:
    encoder = GaussianRandomProjection(n_components=n_dims)
    one_hot_encoding = pd.get_dummies(series, sparse=True)
    one_hot_encoding = encoder.fit_transform(one_hot_encoding)
    base_name = series.name
    one_hot_encoding = DataFrame(
        data=one_hot_encoding,
        columns=[f"{base_name}_dim{i}" for i in range(n_dims)],
    )

    return one_hot_encoding, encoder


def process_data(data: DataFrame, **kwargs):
    data = data.copy()

    # Generate the temporal features for the data
    datetime = pd.to_datetime(data["datetime"])

    # convert a time column with seconds since the minimum date
    data["time"] = (datetime - datetime.min()).dt.total_seconds()

    if kwargs["cyclic_day"]:
        print("Adding cyclic day features")
        data["day_sin"], data["day_cos"] = get_cyclic_day(datetime)

    if kwargs["cyclic_week"]:
        print("Adding cyclic week features")
        data["week_sin"], data["week_cos"] = get_cyclic_week(datetime)

    if kwargs["cyclic_month"]:
        print("Adding cyclic month features")
        data["month_sin"], data["month_cos"] = get_cyclic_month(datetime)

    if kwargs["cyclic_quarter"]:
        print("Adding cyclic quarter features")
        data["quarter_sin"], data["quarter_cos"] = get_cyclic_quarter(datetime)

    if kwargs["cyclic_year"]:
        print("Adding cyclic year features")
        data["year_sin"], data["year_cos"] = get_cyclic_year(datetime)

    # Generate the symbol encoding for the data
    if kwargs["symbol_encoding"] is not None:
        n_dims = kwargs["symbol_encoding"]
        print(f"Adding {n_dims} dimensional symbol encoding")
        embedding, encoder = get_nd_embedding(data["symbol"], n_dims)

        # save the projection model to the same directory as the processed data
        with open(os.path.join(kwargs["destination"], "symbol_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)

        data = pd.concat([data, embedding], axis=1)

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
