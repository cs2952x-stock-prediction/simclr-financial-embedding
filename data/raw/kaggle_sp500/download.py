import os

import kagglehub
import pandas as pd

if __name__ == "__main__":
    # Download latest version
    download_dir = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
    print("Dataset files downloaded to: ", download_dir)

    # Move the dataset to this directory
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    print("Moving files to current directory: ", curr_dir)

    # move everything in the returned path to this directory
    new_files = []
    for file in os.listdir(download_dir):
        old_file = download_dir + "/" + file
        new_file = curr_dir + "/" + file
        new_files.append(new_file)
        os.rename(old_file, new_file)

    # remove the empty directory
    print("Removing empty directory: ", download_dir)
    os.rmdir(download_dir)

    # clean the data

    # clean the stock data file
    stock_file = curr_dir + "/sp500_stocks.csv"
    print("Cleaning stock data file: ", stock_file)

    data = pd.read_csv(stock_file)
    data.columns = list(map(str.lower, data.columns))
    rename_map = {
        "date": "datetime",
        "adj close": "adj_close",
    }
    data.rename(columns=rename_map, inplace=True)
    data.to_csv(stock_file, index=False)

    # clean the companies data file
    companies_file = curr_dir + "/sp500_companies.csv"
    print("Cleaning companies data file: ", companies_file)

    data = pd.read_csv(companies_file)
    data.columns = list(map(str.lower, data.columns))
    rename_map = {
        "shortname": "short_name",
        "longname": "long_name",
        "currentprice": "current_price",
        "marketcap": "market_cap",
        "revenuegrowth": "revenue_growth",
        "fulltimeemployees": "full_time_employees",
        "longbusinesssummary": "long_business_summary",
    }
    data.rename(columns=rename_map, inplace=True)
    data.to_csv(companies_file, index=False)

    # clean the index data file
    index_file = curr_dir + "/sp500_index.csv"
    print("Cleaning index data file: ", index_file)

    data = pd.read_csv(index_file)
    data.columns = list(map(str.lower, data.columns))
    rename_map = {
        "date": "datetime",
    }
    data.rename(columns=rename_map, inplace=True)
    data.to_csv(index_file, index=False)

    print("Files cleaned and moved to current directory: ", curr_dir)
