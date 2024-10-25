import json
import os

import pandas as pd
import requests

# Which data and how much
series_ids = ["CUUR0000SA0", "SUUR0000SA0"]
start_year = "2000"
end_year = "2024"

# Request the data
headers = {"Content-type": "application/json"}
data = json.dumps(
    {"seriesid": series_ids, "startyear": start_year, "endyear": end_year}
)
p = requests.post(
    "https://api.bls.gov/publicAPI/v2/timeseries/data/", data=data, headers=headers
)
json_data = json.loads(p.text)

# Get the current directory
curr_dir = os.path.dirname(os.path.realpath(__file__))

# Transfer the data to a dataframe and then CSV
for series in json_data["Results"]["series"]:
    df = pd.DataFrame(
        columns=["series_id", "year", "period", "period_name", "value", "footnotes"]
    )
    series_id = series["seriesID"]
    for item in series["data"]:
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

    df.to_csv(f"{curr_dir}/{series_id}.csv", index=False)
