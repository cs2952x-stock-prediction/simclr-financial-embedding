import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

################################################# GLOBAL CONSTANTS #################################

# integer-valued columns that can be log-transformed
INT_LOG_COLUMNS = ["volume"]
INT_EPS = 1  # quantity to add to integer columns before log-transforming

# float-valued columns that can be log-transformed
FLOAT_LOG_COLUMNS = ["close", "high", "low", "open", "adj_close"]
FLOAT_EPS = 0.01  # quadrity to add to float columns before log-transforming

################################################ FUNCTIONS #########################################


def intersection(lst1, lst2):
    """
    Find the intersection of two lists.

    Args:
        lst1 (list): The first list.
        lst2 (list): The second list.

    Returns:
        list: The intersection of the two lists.
    """
    return [value for value in lst1 if value in lst2]


class DataTransformer:
    def __init__(self, config: dict):
        self.config = config  # Store the transformation configuration
        self.scaler = StandardScaler()

    def log_transform(self, df: pd.DataFrame):
        """
        Apply log transform to specified columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        for col in intersection(self.config.get("log_features", []), df.columns):
            if col not in df.columns:
                continue

            if (df[col] < 0).any():
                raise ValueError(
                    f"Column '{col}' contains negative values; log transform is invalid."
                )

            if col in INT_LOG_COLUMNS:
                df.loc[:, col] = np.log(df[col] + INT_EPS)
            elif col in FLOAT_LOG_COLUMNS:
                df.loc[:, col] = np.log(df[col] + FLOAT_EPS)
            else:
                raise ValueError(f"Unexpected column passed to log transform: '{col}'.")

        return df

    def invert_log(self, df: pd.DataFrame):
        """
        Invert log transform on specified columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        for col in intersection(self.config.get("log_features", []), df.columns):
            if col in INT_LOG_COLUMNS:
                df.loc[:, col] = np.exp(df[col]) - INT_EPS
            elif col in FLOAT_LOG_COLUMNS:
                df.loc[:, col] = np.exp(df[col]) - FLOAT_EPS
            else:
                raise ValueError(f"Unexpected column passed to log transform: '{col}'.")

        return df

    def diff_transform(self, df: pd.DataFrame, prev_row: Optional[pd.Series] = None):
        """
        Apply differencing to specified columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            prev_row (pd.Series): The previous row to difference against.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        if prev_row is not None:
            df = pd.concat([prev_row.to_frame().T, df], ignore_index=True)

        diff_cols = intersection(self.config.get("diff_features", []), df.columns)
        df.loc[:, diff_cols] = df[diff_cols].diff()
        df = df.iloc[1:, :]

        return df

    def inverse_diff(self, df: pd.DataFrame, prev_row: pd.Series):
        """
        Invert differencing on specified columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            prev_row (pd.Series): The previous row to difference against.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        diff_cols = intersection(self.config.get("diff_features", []), df.columns)
        for col in diff_cols:
            df.loc[:, col] = df[col].cumsum() + prev_row[col]

        df = pd.concat([prev_row.to_frame().T, df], ignore_index=True)

        return df

    def scale_transform(self, df: pd.DataFrame):
        """
        Apply standard scaling to specified columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        scale_cols = intersection(self.config.get("scale_features", []), df.columns)
        df.loc[:, scale_cols] = self.scaler.transform(df[scale_cols])

        return df

    def inverse_scale(self, df: pd.DataFrame):
        """
        Invert standard scaling on specified columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        scale_cols = intersection(self.config.get("scale_features", []), df.columns)
        df.loc[:, scale_cols] = self.scaler.inverse_transform(df[scale_cols])

        return df

    def transform(self, df: pd.DataFrame, prev_row: Optional[pd.Series] = None):
        """
        Apply all transformations to a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            prev_row (pd.Series): The previous row to difference against.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        df = self.log_transform(df)
        df = self.diff_transform(df, prev_row)
        df = self.scale_transform(df)

        return df

    def inverse_transform(self, df: pd.DataFrame, prev_row: pd.Series):
        """
        Invert all transformations on a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to transform.
            prev_row (pd.Series): The previous row to difference against.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        df = self.inverse_scale(df)
        df = self.inverse_diff(df, prev_row)
        df = self.invert_log(df)

        return df

    def fit(self, df: pd.DataFrame, prev_row: Optional[pd.Series] = None):
        """
        Fit the transformer to a DataFrame, learning transformation parameters.

        Args:
            df (pd.DataFrame): The DataFrame to fit.
            prev_row (pd.Series): The previous row to difference against.

        Returns:
            None
        """
        df = df.copy()
        df = self.log_transform(df)
        df = self.diff_transform(df, prev_row)

        scale_cols = intersection(self.config.get("scale_features", []), df.columns)
        self.scaler.fit(df[scale_cols])

    def fit_transform(self, df: pd.DataFrame, prev_row: Optional[pd.Series] = None):
        """
        Fit the transformer to a DataFrame and apply transformations.

        Args:
            df (pd.DataFrame): The DataFrame to fit and transform.
            prev_row (pd.Series): The previous row to difference against.

        Returns:
            pd.DataFrame: The transformed DataFrame.
        """
        self.fit(df, prev_row)
        return self.transform(df, prev_row)

    def save(self, filepath: str):
        """
        Save the transformer to a file.

        Args:
            filepath (str): The file to save the transformer to.
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        """
        Load a transformer from a file.

        Args:
            filepath (str): The file to load the transformer from.

        Returns:
            DataTransformer: The loaded transformer.
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


def add_cyclic_day_feature(df: pd.DataFrame, inplace=True):
    """
    Add cyclic day features to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add features to.
        inplace (bool): Whether to add the features in place.

    Returns:
        pd.DataFrame: The DataFrame with added features
    """
    if not inplace:
        df = df.copy()

    timestamps = df["timestamp"].dt
    fraction_of_day = (
        timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    ) / 24
    df["day_sin"] = np.sin(2 * np.pi * fraction_of_day)
    df["day_cos"] = np.cos(2 * np.pi * fraction_of_day)

    return df


def add_cyclic_week_feature(df: pd.DataFrame, inplace=True):
    """
    Add cyclic week features to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add features to.
        inplace (bool): Whether to add the features in place.

    Returns:
        pd.DataFrame: The DataFrame with added features
    """
    if not inplace:
        df = df.copy()

    timestamps = df["timestamp"].dt
    fraction_of_day = (
        timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    ) / 24
    fraction_of_week = (timestamps.dayofweek + fraction_of_day) / 7

    df["week_sin"] = np.sin(2 * np.pi * fraction_of_week)
    df["week_cos"] = np.cos(2 * np.pi * fraction_of_week)

    return df


def add_cyclic_month_feature(df: pd.DataFrame, inplace=True):
    """
    Add cyclic month of the year features to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add features to.
        inplace (bool): Whether to add the features in place.

    Returns:
        pd.DataFrame: The DataFrame with added features
    """
    if not inplace:
        df = df.copy()

    timestamps = df["timestamp"].dt
    days_in_month = timestamps.dt.days_in_month
    fraction_of_day = (
        timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    ) / 24
    fraction_of_month = (timestamps.dt.day + fraction_of_day) / days_in_month
    df["month_sin"] = np.sin(2 * np.pi * fraction_of_month)
    df["month_cos"] = np.cos(2 * np.pi * fraction_of_month)

    return df


def add_cyclic_quarter_feature(df: pd.DataFrame, inplace=True):
    """
    Add cyclic quarter of the year features to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add features to.
        inplace (bool): Whether to add the features in place.

    Returns:
        pd.DataFrame: The DataFrame with added features
    """
    if not inplace:
        df = df.copy()

    timestamps = df["timestamp"].dt
    period = timestamps.dt.to_period("Q")
    days_in_quarter = period.apply(
        lambda x: x.end_time.dayofyear - x.start_time.dayofyear
    )
    day_of_quarter = timestamps.dt.dayofyear - period.apply(
        lambda x: x.start_time.dayofyear
    )
    fraction_of_day = (
        timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    )
    fraction_of_quarter = (day_of_quarter + fraction_of_day) / days_in_quarter
    df["quarter_sin"] = np.sin(2 * np.pi * fraction_of_quarter)
    df["quarter_cos"] = np.cos(2 * np.pi * fraction_of_quarter)

    return df


def add_cyclic_year_feature(df: pd.DataFrame, inplace=True):
    """
    Add cyclic year features to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add features to.
        inplace (bool): Whether to add the features in place.

    Returns:
        pd.DataFrame: The DataFrame with added features
    """
    if not inplace:
        df = df.copy()

    timestamps = df["timestamp"].dt
    days_in_year = timestamps.dt.to_period("Y").apply(lambda x: x.end_time.dayofyear)
    day_of_year = timestamps.dt.dayofyear
    fraction_of_day = (
        timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    )
    fraction_of_year = (day_of_year + fraction_of_day) / days_in_year
    df["year_sin"] = np.sin(2 * np.pi * fraction_of_year)
    df["year_cos"] = np.cos(2 * np.pi * fraction_of_year)

    return df
