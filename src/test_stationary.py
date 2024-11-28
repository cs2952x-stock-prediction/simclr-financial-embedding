import os

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series, column_name):
    """Runs the ADF test on a pandas Series and returns the result."""
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "Column": column_name,
        "Test Statistic": result[0],
        "p-value": result[1],
        "Lags Used": result[2],
        "Observations Used": result[3],
        "Stationary": result[1] <= 0.05,
    }


def process_csv_files(directory, output_file):
    """Iterates through all CSV files in a directory and performs the ADF test."""
    results = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory, filename)
            try:
                print(f"Processing file: {filename}")
                # Load CSV
                data = pd.read_csv(file_path)

                # Perform ADF test on all numerical columns
                for column in data.select_dtypes(include=["float64", "int64"]).columns:
                    test_result = adf_test(data[column], column)
                    test_result["File"] = filename
                    results.append(test_result)

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Save results to a summary CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")


if __name__ == "__main__":
    # Specify the directory containing the CSV files and the output file
    input_directory = "data/processed/kaggle/train/"
    output_summary_file = "adf_test_summary.csv"

    # Run the process
    process_csv_files(input_directory, output_summary_file)
