import os

import kagglehub
import pandas as pd

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
os.rmdir(download_dir)

# clean the data
stock_file = curr_dir + "/sp500_stocks.csv"

data = pd.read_csv(stock_file)
data.columns = list(map(str.lower, data.columns))
data.rename(columns={"date": "datetime"}, inplace=True)
data.to_csv(stock_file, index=False)
print(f"Standardized columns in {stock_file}")

companies_file = curr_dir + "/sp500_companies.csv"

data = pd.read_csv(companies_file)
data.columns = list(map(str.lower, data.columns))
replacements = {
    "shortname": "short_name",
    "longname": "long_name",
    "currentprice": "current_price",
    "marketcap": "market_cap",
    "revenuegrowth": "revenue_growth",
    "fulltimeemployees": "full_time_employees",
    "longbusinesssummary": "long_business_summary",
}
data.rename(columns=replacements, inplace=True)
data.to_csv(companies_file, index=False)
print(f"Standardized columns in {companies_file}")

index_file = curr_dir + "/sp500_index.csv"

data = pd.read_csv(index_file)
data.columns = list(map(str.lower, data.columns))
data.rename(columns={"date": "datetime"}, inplace=True)
data.to_csv(index_file, index=False)
print(f"Standardized columns in {index_file}")
