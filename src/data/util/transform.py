import pickle
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

################################################# GLOBAL CONSTANTS #################################

EPS = 0.01  # quantity to add before taking the log

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
        self.scale_cols = []

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

            df.loc[:, col] = np.log(df[col] + EPS)

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
            df.loc[:, col] = np.exp(df[col]) - EPS

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

        self.scale_cols = intersection(
            self.config.get("scale_features", []), df.columns
        )
        self.scaler.fit(df[self.scale_cols])

    def partial_fit(self, df: pd.DataFrame, prev_row: Optional[pd.Series] = None):
        """
        Update the transformer with new data, learning transformation parameters.

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
        self.scaler.partial_fit(df[scale_cols])

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
