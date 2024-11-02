import os

import pandas as pd
from tqdm import tqdm

src_filename = "sp500_stocks.csv"
src_dirpath = os.path.dirname(os.path.realpath(__file__))
src_dirname = src_dirpath.split("/")[-1]
src_path = f"{src_dirpath}/{src_filename}"

tgt_dirpath = f"{src_dirpath}/../../interim/{src_dirname}"

if __name__ == "__main__":
    # Load data
    print(f"Loading data from {src_filename}")
    df = pd.read_csv(src_path)
    print(df.head())

    # Separate series by symbol
    print("Separating series by symbol, interpolating, removing nan values...")
    symbols = df["symbol"].unique()
    for symbol in tqdm(symbols):
        symbol_series = df[df["symbol"] == symbol]
        symbol_series = symbol_series.drop(columns=["symbol"])

        # Interpolate missing values
        col_to_interpolate = [
            "adj_close",
            "close",
            "high",
            "low",
            "open",
            "volume",
        ]
        symbol_series[col_to_interpolate] = symbol_series[
            col_to_interpolate
        ].interpolate(method="linear", axis=0)
        symbol_series.dropna(inplace=True)
        symbol_series.to_csv(f"{tgt_dirpath}/{symbol}.csv", index=False)
