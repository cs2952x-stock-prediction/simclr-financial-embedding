import numpy as np
import pandas as pd


def add_moving_avg_features(df, features):
    """
    Add moving average features to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add features to.
        features (list): A list of feature names in the format "col_typwnd_sz".

    Returns:
        pd.DataFrame: The DataFrame with added features
    """
    # NOTE: Moving averages are calculated over a number of rows rather than a time window.
    # This means the temporal window size can change depending on the frequency of the data.
    # Additionally, missing rows will affect the calculation of the moving average.
    for feature in features:
        # e.g. "close_sma20" -> col="close", avg_type="sma20"
        col, avg_type = feature.split("_")
        assert col in df.columns, f"Invalid column: {col}"

        # e.g. "sma20" -> typ="sma", wnd_sz="20"
        typ, wnd_sz = avg_type[:3], int(avg_type[3:])
        assert wnd_sz > 0, f"Invalid window size: {wnd_sz}"

        if typ == "sma":
            df[feature] = df[col].rolling(window=wnd_sz, min_periods=1).mean()
        elif typ == "ema":
            df[feature] = df[col].ewm(span=wnd_sz).mean()
            for i in range(min(wnd_sz, len(df))):
                df.loc[i, feature] = df.loc[:i, col].mean()
        else:
            raise ValueError(f"Invalid average type: {typ}")

    return df


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

    timestamps = pd.to_datetime(df["timestamp"]).dt
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

    timestamps = pd.to_datetime(df["timestamp"]).dt
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

    timestamps = pd.to_datetime(df["timestamp"]).dt
    days_in_month = timestamps.days_in_month
    fraction_of_day = (
        timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    ) / 24
    fraction_of_month = (timestamps.day + fraction_of_day) / days_in_month
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

    timestamps = pd.to_datetime(df["timestamp"]).dt
    period = timestamps.to_period("Q")
    days_in_quarter = period.apply(
        lambda x: x.end_time.dayofyear - x.start_time.dayofyear
    )
    day_of_quarter = timestamps.dayofyear - period.apply(
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

    timestamps = pd.to_datetime(df["timestamp"]).dt
    days_in_year = timestamps.to_period("Y").apply(lambda x: x.end_time.dayofyear)
    day_of_year = timestamps.dayofyear
    fraction_of_day = (
        timestamps.hour + timestamps.minute / 60 + timestamps.second / 3600
    )
    fraction_of_year = (day_of_year + fraction_of_day) / days_in_year
    df["year_sin"] = np.sin(2 * np.pi * fraction_of_year)
    df["year_cos"] = np.cos(2 * np.pi * fraction_of_year)

    return df


def add_temporal_features(df, features):
    if "seconds" in features:
        timestamps = pd.to_datetime(df["timestamp"])
        df["seconds"] = (timestamps - timestamps.min()).dt.total_seconds()

    if "cyclic_day" in features:
        add_cyclic_day_feature(df)

    if "cyclic_week" in features:
        add_cyclic_week_feature(df)

    if "cyclic_month" in features:
        add_cyclic_month_feature(df)

    if "cyclic_quarter" in features:
        add_cyclic_quarter_feature(df)

    if "cyclic_year" in features:
        add_cyclic_year_feature(df)
