import os
import math
import time
import requests
from PIL import Image
from io import BytesIO
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CSV_PATH = "train_cdc.csv"
# for downloading test images we will change the csv_path to test_cdc.csv
# and directory of image_dir to "sat1_images"
IMAGE_DIR = "sat_images"
ZOOM_LEVEL = 19
MAX_WORKERS = 5

os.makedirs(IMAGE_DIR, exist_ok=True)
df = pd.read_csv(CSV_PATH)


session = requests.Session()
retries = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)


def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
        / 2.0 * n
    )
    return x, y

def download_esri_image(lat, lon, zoom=ZOOM_LEVEL):
    x, y = latlon_to_tile(lat, lon, zoom)

    url = (
        "https://services.arcgisonline.com/ArcGIS/rest/services/"
        f"World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
    )

    resp = session.get(url, timeout=20)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")



def process_row(idx, row):
    lat = row["lat"]
    lon = row["long"]

    out_path = os.path.join(IMAGE_DIR, f"{idx}.jpg")
    if os.path.exists(out_path):
        return f"Skipped {idx}"

    try:
        img = download_esri_image(lat, lon)
        img.save(out_path, "JPEG", quality=90)

        return f"Saved {idx}"
    except Exception as e:
        return f"Failed {idx}: {type(e).__name__}"


def main():
    print(f"Downloading {len(df)} images with {MAX_WORKERS} workers\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_row, idx, row)
            for idx, row in df.iterrows()
        ]
        for future in as_completed(futures):
            print(future.result())

    print("\nDownload complete.")

if __name__ == "__main__":
    main()
