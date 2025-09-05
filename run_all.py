import subprocess
import pandas as pd
import os
import shutil
import config

# Create 'Jobs' folder if it doesn't exist
jobs_folder = "Jobs"
os.makedirs(jobs_folder, exist_ok=True)

# Build from config.CITIES
scripts_and_outputs = []
for city in config.CITIES.keys():
    csv_file = f"jobs_{city.lower().replace(' ', '_')}.csv"
    scripts_and_outputs.append((city, csv_file, city))

# Run scrape_city.py per city
for city, csv_file, _ in scripts_and_outputs:
    print(f"Running scrape for {city}...")
    subprocess.run(["python", "scrape_city.py", "--city", city])

# Merge CSVs into one Excel file with multiple sheets in Jobs folder
excel_path = os.path.join(jobs_folder, "all_jobs.xlsx")

# Load available per-city CSVs
city_frames = []  # list of tuples (sheet_name, DataFrame)
for _, csv_file, sheet_name in scripts_and_outputs:
    csv_path = os.path.join(jobs_folder, csv_file)
    try:
        df = pd.read_csv(csv_path)
        city_frames.append((sheet_name, df))
    except Exception as e:
        print(f"Could not add {csv_file}: {e}")

# Write Excel with 'All' as the first sheet when any data exists
with pd.ExcelWriter(excel_path) as writer:
    if city_frames:
        all_df = pd.concat([df for _, df in city_frames], ignore_index=True)
        all_df.to_excel(writer, sheet_name="All", index=False)
        for sheet_name, df in city_frames:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        # create a placeholder sheet to avoid openpyxl IndexError
        pd.DataFrame().to_excel(writer, sheet_name="Empty", index=False)

print(f"Merged available CSVs into {excel_path}")

# Delete all CSVs in Jobs folder (do not keep any)
for fname in os.listdir(jobs_folder):
    if fname.lower().endswith(".csv"):
        try:
            os.remove(os.path.join(jobs_folder, fname))
        except Exception as e:
            print(f"Could not delete {fname}: {e}")
