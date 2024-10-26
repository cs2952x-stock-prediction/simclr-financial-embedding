import json
import os

import pandas as pd
import requests

# Which data and how much

# check out popula data seris here: https://data.bls.gov/toppicks?survey=bls
SERIES_IDS = [
    "CUUR0000SA0",  # CPI for All Urban Consumers (CPI-U) 1982-84=100
    "SUUR0000SA0",
]
START_YEAR = 2000
END_YEAR = 2024
CHUNK_SIZE = 10  # Number of years per request


# Function to chunk the requests
def fetch_data_in_chunks(series_ids, start_year, end_year, chunk_size):
    headers = {"Content-type": "application/json"}
    data_by_series = {series_id: [] for series_id in series_ids}

    # Loop through years in chunks
    for year in range(start_year, end_year + 1, chunk_size):
        chunk_start = year
        chunk_end = min(year + chunk_size - 1, end_year)
        data = json.dumps(
            {
                "seriesid": series_ids,
                "startyear": str(chunk_start),
                "endyear": str(chunk_end),
            }
        )

        response = requests.post(
            "https://api.bls.gov/publicAPI/v2/timeseries/data/",
            data=data,
            headers=headers,
        )

        if response.status_code != 200:
            print(f"Error fetching data for {chunk_start}-{chunk_end}: {response.text}")
            continue

        json_data = response.json()

        if json_data["status"] != "REQUEST_SUCCEEDED":
            print(f"Failed to fetch data for {chunk_start}-{chunk_end}")
            continue

        # Collect data
        for series in json_data["Results"]["series"]:
            series_id = series["seriesID"]
            series_data = series["data"]
            data_by_series[series_id].extend(series_data)

    return data_by_series


# Fetch the data in chunks
json_data = fetch_data_in_chunks(SERIES_IDS, START_YEAR, END_YEAR, CHUNK_SIZE)

# Get the current directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

# Transfer the data to a DataFrame and then to CSV
for series_id, series_data in json_data.items():
    df = pd.DataFrame(
        columns=["series_id", "year", "period", "period_name", "value", "footnotes"]
    )

    for item in series_data:
        if "M01" <= item["period"] <= "M12":
            df = df._append(
                {
                    "series_id": series_id,
                    "year": item["year"],
                    "period": item["period"],
                    "period_name": item["periodName"],
                    "value": item["value"],
                    "footnotes": ",".join(
                        footnote["text"]
                        for footnote in item["footnotes"]
                        if "text" in footnote
                    ),
                },
                ignore_index=True,
            )
        else:
            print(f"Skipping {item['period']}")

    # Save each series to CSV
    df.sort_values(["year", "period"], inplace=True)
    df.to_csv(f"{curr_dir}/{series_id}.csv", index=False)
    print(f"Saved data for {series_id} to {series_id}.csv")
