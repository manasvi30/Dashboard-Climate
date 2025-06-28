import requests
import pandas as pd
from datetime import datetime
import time

# Climate parameters from NASA POWER API
params = [
    "PRECTOTCORR", "PS", "QV2M", "RH2M", "T2M", "T2MWET", "T2M_MAX", "T2M_MIN", "T2M_RANGE",
    "TS", "WS10M", "WS10M_MAX", "WS10M_MIN", "WS10M_RANGE",
    "WS50M", "WS50M_MAX", "WS50M_MIN", "WS50M_RANGE"
]

# Districts with latitude and longitude
locations = [
    {"district": "Achham", "lat": 29.12, "lon": 81.3},
    {"district": "Arghakhanchi", "lat": 27.98, "lon": 83.2},
    {"district": "Baglung", "lat": 28.27, "lon": 83.62},
    {"district": "Baitadi", "lat": 29.53, "lon": 80.58},
    {"district": "Bajhang", "lat": 29.83, "lon": 81.25},
    {"district": "Bajura", "lat": 29.5, "lon": 81.5},
    {"district": "Banke", "lat": 28.05, "lon": 81.67},
    {"district": "Bara", "lat": 27.03, "lon": 85.05},
    {"district": "Bardiya", "lat": 28.3, "lon": 81.5},
    {"district": "Bhaktapur", "lat": 27.67, "lon": 85.43},
    {"district": "Bhojpur", "lat": 27.17, "lon": 87.05},
    {"district": "Chitwan", "lat": 27.53, "lon": 84.5},
    {"district": "Dadeldhura", "lat": 29.3, "lon": 80.58},
    {"district": "Dailekh", "lat": 28.85, "lon": 81.73},
    {"district": "Dang", "lat": 28.75, "lon": 82.38},
    {"district": "Darchula", "lat": 30.15, "lon": 80.65},
    {"district": "Dhading", "lat": 27.9, "lon": 84.95},
    {"district": "Dhankuta", "lat": 26.98, "lon": 87.35},
    {"district": "Dhanusha", "lat": 26.85, "lon": 85.97},
    {"district": "Dolakha", "lat": 27.67, "lon": 86.02},
    {"district": "Dolpa", "lat": 29.05, "lon": 83.55},
    {"district": "Doti", "lat": 29.27, "lon": 80.98},
    {"district": "Eastern Rukum", "lat": 28.63, "lon": 82.55},
    {"district": "Gorkha", "lat": 28.0, "lon": 84.63},
    {"district": "Gulmi", "lat": 28.08, "lon": 83.23},
    {"district": "Humla", "lat": 29.97, "lon": 81.83},
    {"district": "Ilam", "lat": 26.9, "lon": 87.93},
    {"district": "Jajarkot", "lat": 28.7, "lon": 82.17},
    {"district": "Jhapa", "lat": 26.6, "lon": 88.08},
    {"district": "Jumla", "lat": 29.28, "lon": 82.18},
    {"district": "Kailali", "lat": 28.67, "lon": 80.65},
    {"district": "Kalikot", "lat": 29.13, "lon": 81.63},
    {"district": "Kanchanpur", "lat": 28.87, "lon": 80.33},
    {"district": "Kapilvastu", "lat": 27.55, "lon": 83.05},
    {"district": "Kaski", "lat": 28.27, "lon": 83.97},
    {"district": "Kathmandu", "lat": 27.72, "lon": 85.32},
    {"district": "Kavrepalanchok", "lat": 27.63, "lon": 85.53},
    {"district": "Khotang", "lat": 27.2, "lon": 86.8},
    {"district": "Lalitpur", "lat": 27.67, "lon": 85.32},
    {"district": "Lamjung", "lat": 28.15, "lon": 84.4},
    {"district": "Mahottari", "lat": 26.65, "lon": 85.83},
    {"district": "Makwanpur", "lat": 27.43, "lon": 85.1},
    {"district": "Manang", "lat": 28.65, "lon": 84.02},
    {"district": "Morang", "lat": 26.65, "lon": 87.45},
    {"district": "Mugu", "lat": 29.57, "lon": 82.37},
    {"district": "Mustang", "lat": 28.98, "lon": 83.93},
    {"district": "Myagdi", "lat": 28.38, "lon": 83.57},
    {"district": "Nawalparasi East", "lat": 27.63, "lon": 84.25},
    {"district": "Nawalparasi West", "lat": 27.6, "lon": 83.62},
    {"district": "Nuwakot", "lat": 27.9, "lon": 85.13},
    {"district": "Okhaldhunga", "lat": 27.33, "lon": 86.5},
    {"district": "Palpa", "lat": 27.88, "lon": 83.55},
    {"district": "Panchthar", "lat": 27.12, "lon": 87.8},
    {"district": "Parbat", "lat": 28.23, "lon": 83.6},
    {"district": "Parsa", "lat": 27.0, "lon": 84.88},
    {"district": "Pyuthan", "lat": 28.08, "lon": 82.87},
    {"district": "Ramechhap", "lat": 27.33, "lon": 86.07},
    {"district": "Rasuwa", "lat": 28.1, "lon": 85.37},
    {"district": "Rautahat", "lat": 26.97, "lon": 85.3},
    {"district": "Rolpa", "lat": 28.48, "lon": 82.38},
    {"district": "Rupandehi", "lat": 27.58, "lon": 83.5},
    {"district": "Salyan", "lat": 28.38, "lon": 82.18},
    {"district": "Sankhuwasabha", "lat": 27.55, "lon": 87.28},
    {"district": "Saptari", "lat": 26.65, "lon": 86.75},
    {"district": "Sarlahi", "lat": 26.98, "lon": 85.58},
    {"district": "Sindhuli", "lat": 27.25, "lon": 85.98},
    {"district": "Sindhupalchok", "lat": 27.88, "lon": 85.7},
    {"district": "Siraha", "lat": 26.65, "lon": 86.2},
    {"district": "Solukhumbu", "lat": 27.58, "lon": 86.6},
    {"district": "Sunsari", "lat": 26.63, "lon": 87.3},
    {"district": "Surkhet", "lat": 28.6, "lon": 81.63},
    {"district": "Syangja", "lat": 28.0, "lon": 83.87},
    {"district": "Tanahun", "lat": 28.05, "lon": 84.25},
    {"district": "Taplejung", "lat": 27.35, "lon": 87.67},
    {"district": "Tehrathum", "lat": 27.03, "lon": 87.58},
    {"district": "Udayapur", "lat": 26.85, "lon": 86.65},
    {"district": "Rukum", "lat": 28.63, "lon": 82.25}
]


# Collect all data here
all_data = []

# Fetch from API for each district
for loc in locations:
    print(f"📡 Fetching data for {loc['district']}...")
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    query = {
        "parameters": ",".join(params),
        "community": "AG",
        "longitude": loc["lon"],
        "latitude": loc["lat"],
        "start": "20200101",
        "end": "20241231",
        "format": "JSON"
    }

    try:
        response = requests.get(url, params=query)
        response.raise_for_status()
        json_data = response.json()

        # ✅ Check if 'parameter' block exists
        if "parameter" not in json_data["properties"]:
            print(f"⚠️ No climate data for {loc['district']} at ({loc['lat']}, {loc['lon']})")
            continue

        data = json_data["properties"]["parameter"]

        # Use T2M for date reference (since all keys share same structure)
        dates = list(data["T2M"].keys())

        for date in dates:
            try:
                formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%m/%d/%Y")
            except Exception as e:
                print(f"⛔️ Date formatting error: {date} — {e}")
                continue

            row = {
                "Date": formatted_date,
                "District": loc["district"],
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
            all_data.append(row)

        time.sleep(1)  

    except Exception as e:
        print(f"❌ Error for {loc['district']}: {e}")

# Convert to DataFrame and save
df = pd.DataFrame(all_data)
df.to_csv("nasa_climate.csv", index=False)
print("✅ Data saved to 'nasa_climate.csv'")
