import requests
import pandas as pd
import os
import time
import random
import logging
from datetime import datetime
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Climate parameters
params = [
    "PRECTOTCORR", "PS", "QV2M", "RH2M", "T2M", "T2MWET", "T2M_MAX", "T2M_MIN", "T2M_RANGE",
    "TS", "WS10M", "WS10M_MAX", "WS10M_MIN", "WS10M_RANGE",
    "WS50M", "WS50M_MAX", "WS50M_MIN", "WS50M_RANGE"
]

# Districts lat long sanga
locations = [
    {"district": "Achham", "lat": 29.12, "lon": 81.3},
    {"district": "Bajura", "lat": 29.5, "lon": 81.5},
    {"district": "Bhojpur", "lat": 27.17, "lon": 87.05},
    {"district": "Jajarkot", "lat": 28.7, "lon": 82.17},
    {"district": "Kailali", "lat": 28.67, "lon": 80.65},
    {"district": "Kavrepalanchok", "lat": 27.63, "lon": 85.53},
    {"district": "Khotang", "lat": 27.2, "lon": 86.8},
    {"district": "Panchthar", "lat": 27.12, "lon": 87.8},
    {"district": "Parsa", "lat": 27.0, "lon": 84.88},
    {"district": "Pyuthan", "lat": 28.08, "lon": 82.87},
    {"district": "Ramechhap", "lat": 27.33, "lon": 86.07},
    {"district": "Sindhupalchok", "lat": 27.88, "lon": 85.7},
]

#File to save to
output_file = "nasa_climate.csv"

#Track if header has been written
write_header = not os.path.exists(output_file)

#Fetch data
for loc in tqdm(locations, desc="Fetching district data"):
    district = loc["district"]
    logging.info(f"🌍 Fetching: {district}")

    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    query = {
        "parameters": ",".join(params),
        "community": "AG",
        "longitude": loc["lon"],
        "latitude": loc["lat"],
        "start": "19810101",
        "end": "20191231",
        "format": "JSON"
    }

    try:
        response = requests.get(url, params=query)
        response.raise_for_status()
        json_data = response.json()

        #parameter validation
        if "parameter" not in json_data["properties"]:
            logging.warning(f"⚠️ No data for {district}")
            continue

        data = json_data["properties"]["parameter"]
        dates = list(data["T2M"].keys())  # reference key

        rows = []
        for date in dates:
            try:
                formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%m/%d/%Y")
                row = {
                    "Date": formatted_date,
                    "District": district,
                    "Latitude": loc["lat"],
                    "Longitude": loc["lon"],
                    "Precip": data["PRECTOTCORR"].get(date),
                    "Pressure": data["PS"].get(date),
                    "Humidity_2m": data["QV2M"].get(date),
                    "RH_2m": data["RH2M"].get(date),
                    "Temp_2m": data["T2M"].get(date),
                    "WetBulbTemp_2m": data["T2MWET"].get(date),
                    "MaxTemp_2m": data["T2M_MAX"].get(date),
                    "MinTemp_2m": data["T2M_MIN"].get(date),
                    "TempRange_2m": data["T2M_RANGE"].get(date),
                    "EarthSkinTemp": data["TS"].get(date),
                    "WindSpeed_10m": data["WS10M"].get(date),
                    "MaxWindSpeed_10m": data["WS10M_MAX"].get(date),
                    "MinWindSpeed_10m": data["WS10M_MIN"].get(date),
                    "WindSpeedRange_10m": data["WS10M_RANGE"].get(date),
                    "WindSpeed_50m": data["WS50M"].get(date),
                    "MaxWindSpeed_50m": data["WS50M_MAX"].get(date),
                    "MinWindSpeed_50m": data["WS50M_MIN"].get(date),
                    "WindSpeedRange_50m": data["WS50M_RANGE"].get(date)
                }
                rows.append(row)
            except Exception as e:
                logging.error(f"❌ Date error in {district} on {date}: {e}")

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_file, mode='a', index=False, header=write_header)
            write_header = False

        time.sleep(random.uniform(1.5, 2.5))

    except Exception as e:
        logging.error(f"❌ API error for {district}: {e}")

logging.info("✅ All done. Data saved to 'nasa_climate.csv'")
