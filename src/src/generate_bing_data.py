import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
import random, os, sys
import tempfile
import time
import queue
from multiprocessing import Process
from time import sleep
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from global_land_mask import globe
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.service import Service as ChromeService
from pyvirtualdisplay import Display
import chromedriver_autoinstaller

from PIL import Image
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import csv
import requests

parent = os.path.dirname((os.path.dirname(os.path.realpath(__file__))))
sys.path.append(parent)

np.random.seed(42)
random.seed(42)

# paths
# change to your base path where you want the dataset to be stored
DATA_STORAGE_BASE = f"path/to/dataset/"
DATA_STORAGE_BASE = f"/home/jonas/mnt/sda1/test/CIL_dataset/"
DATA_STORAGE_SOURCE_PATH = f"{DATA_STORAGE_BASE}source/"
DATA_STORAGE_TARGET_PATH = f"{DATA_STORAGE_BASE}target/"

NR_PROCESSES = 6

# City coordinate sampling parameters
MIN_STD_LON_LAT_SAMPLING = 0.001
VARIABLE_STD_LON_LAT_SAMPLING = 0.01
NR_SAMPLES = (
    100  # chane to how many samples you want from the city coordinates csv file
)
NR_BATCHES = 1000
POPULATION_THRESHOLD = 5000

# https://learn.microsoft.com/en-us/bingmaps/articles/understanding-scale-and-resolution
# bing map level to meters/pixel
# 1:		78271.52	m/pixel		11:		76.44 m/pixel
# 2:		39135.76	m/pixel		12:		38.22 m/pixel
# 3:		19567.88	m/pixel		13:		19.11 m/pixel
# 4:		9783.94	m/pixel			14:		9.55 m/pixel
# 5:		4891.97	m/pixel			15:		4.78 m/pixel
# 6:		2445.98	m/pixel			16:		2.39 m/pixel
# 7:		1222.99	m/pixel			17:		1.19 m/pixel
# 8:		611.50	m/pixel			18:		0.60 m/pixel
# 9:		305.75	m/pixel			19:		0.30 m/pixel
# 10:		152.87	m/pixel
BING_LEVEL_DENSITIES = [
    78271.52,
    39135.76,
    19567.88,
    9783.94,
    4891.97,
    2445.98,
    1222.99,
    611.50,
    305.75,
    152.87,
    76.44,
    38.22,
    19.11,
    9.55,
    4.78,
    2.39,
    1.19,
    0.60,
    0.30,
]
BING_MAP_LEVEL = 18
BING_MAP_TILE_RESOLUTION = 500

# Openstreetmap variables
# m / pixel * pixel = m
BBOX_WIDTH = BING_LEVEL_DENSITIES[BING_MAP_LEVEL - 1] * BING_MAP_TILE_RESOLUTION

# empirically evaluated what gives the most detail with still low enough svg size (depends on BING_MAP_LEVEL)
# BING_MAP_LEVEL = 15:
# SCALE = 20000 # -> used for first round of target images
# SCALE = 1000 # works fast, no difference to scale 600 -> mor details than scale 20000
SCALE = 878  # works fast, no difference to scale 600 -> mor details than scale 20000

# Earth radius (average) = 6'378'100 m
EARTH_RADIUS = 6378100
LON_0 = 0

header = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36"
}

# Sample coordinates


def sample_city_coordinates():
    worldcities_url = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.76.zip"
    resp = urlopen(worldcities_url)

    zip_file = ZipFile(BytesIO(resp.read()))
    if "worldcities.csv" in zip_file.namelist():
        df = pd.read_csv(zip_file.open("worldcities.csv"))

    df = df.rename(columns={"lng": "lon"})
    df = df[df["population"] > POPULATION_THRESHOLD]

    std = (
        MIN_STD_LON_LAT_SAMPLING
        + VARIABLE_STD_LON_LAT_SAMPLING * df["population"] / df["population"].max()
    )
    mean = np.zeros(std.shape)

    result_df = pd.DataFrame(data=df[["lat", "lon"]].copy(), columns=["lat", "lon"])
    lon_lat = df[["lat", "lon"]].to_numpy()

    for i in range(NR_SAMPLES - 1):
        lon_deviation = np.random.normal(mean, std)
        lat_deviation = np.random.normal(mean, std)
        lon_lat_deviation = np.stack((lon_deviation, lat_deviation), axis=1)
        # latitude: -90 to 90
        # longitude: -180 to 180
        sample = lon_lat + lon_lat_deviation

        sample_clip = np.zeros(sample.shape)

        np.clip(sample[:, 0], a_min=-90, a_max=90, out=sample_clip[:, 0])
        np.clip(sample[:, 1], a_min=-180, a_max=180, out=sample_clip[:, 1])

        sample_df = pd.DataFrame(data=sample_clip.copy(), columns=["lat", "lon"])
        result_df = pd.concat([result_df, sample_df], ignore_index=True)

    count = 0

    indexes_to_drop = []
    # filter lat, lon pairs in ocean
    print("filter (lat, lon) pairs in ocean")
    for i, row in tqdm(result_df.iterrows()):
        lat = row["lat"]
        lon = row["lon"]
        if not globe.is_land(lat, lon):
            indexes_to_drop.append(i)
            count += 1

    result_df.drop(indexes_to_drop, inplace=True)

    df_size = result_df.shape[0]
    batch_size = np.ceil(df_size / NR_BATCHES)
    samples_per_batch = batch_size
    batch_index = np.repeat(np.arange(NR_BATCHES), samples_per_batch, axis=0)[:df_size]
    result_df["batch_nr"] = batch_index

    print(result_df.head())
    return result_df


# Downloading source images


def download_source_process_task(dataset_df):
    display = Display(visible=0, size=(500, 500))
    display.start()

    transform = transforms.CenterCrop(400)

    options = FirefoxOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")

    # if firefox doesnt work, change to chrome driver
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")
    # chrome_options.binary_location = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

    # driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=chrome_options)
    # driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver", chrome_options=chrome_options)

    driver = webdriver.Firefox(options=options)
    q = queue.Queue()

    ecount = 0
    nr_dl = 0
    for index, row in dataset_df.iterrows():
        q.put((index, row["lon"], row["lat"], row["batch_nr"]))

    while not q.empty():
        index, lon, lat, batch_nr = q.get()

        if not os.path.isdir(f"{DATA_STORAGE_SOURCE_PATH}BATCH_{int(batch_nr)}"):
            os.mkdir(f"{DATA_STORAGE_SOURCE_PATH}BATCH_{int(batch_nr)}")

        url = f"https://www.bing.com/maps/embed?h={BING_MAP_TILE_RESOLUTION}&w={BING_MAP_TILE_RESOLUTION}&cp={lat}~{lon}&lvl={18}&typ=s&sty=a&src=SHELL&FORM=MBEDV8"

        element_id = "embedMap"
        element_class = "atlas-control-container"
        try:
            # driver = webdriver.Firefox(options=options)
            # driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver', chrome_options=chrome_options)
            driver.get(url)
            # myElem = WebDriverWait(driver, 5).until(EC.presence_of_element_located((By.ID, element_id)))
            sleep(3)
            element = driver.find_element(By.ID, element_id)
            fname = f"{DATA_STORAGE_SOURCE_PATH}BATCH_{int(batch_nr)}/IMG_{index}_500x500.png"

        except Exception as e:
            print(f"failed at index: {index}, total nr of errors: {ecount}, error: {e}")
            ecount += 1
            # q.put((index, lon, lat, batch_nr))
            continue

        element.screenshot(fname)
        image = Image.open(fname)
        image = transform(image)
        image.save(f"{DATA_STORAGE_SOURCE_PATH}BATCH_{int(batch_nr)}/IMG_{index}.png")
        os.remove(fname)

        nr_dl += 1
        print(f"total dl: {nr_dl}, index: {index}, lon: {lon}, lat: {lat}")

    print(f"failed at {ecount} samples")

    driver.quit()


def download_bing_source_images(cities_df):
    filtered_cities_df = pd.DataFrame(columns=cities_df.columns)

    indexes_to_drop = []
    for index, row in cities_df.iterrows():
        batch_nr = row["batch_nr"]
        if os.path.exists(
            f"{DATA_STORAGE_SOURCE_PATH}BATCH_{int(batch_nr)}/IMG_{index}.png"
        ):
            indexes_to_drop.append(index)

    filtered_cities_df = cities_df.drop(indexes_to_drop)
    data_split = np.array_split(filtered_cities_df, NR_PROCESSES)

    processes = []
    for i in range(NR_PROCESSES):
        split = data_split[i].copy()
        p = Process(target=download_source_process_task, args=(split,))
        print(f"process {i}")
        processes.append(p)
        p.start()

    for i in range(NR_PROCESSES):
        processes[i].join()


# Downloading target images


# https://en.wikipedia.org/wiki/Mercator_projection
def mercatorProject(lon, lat):
    return (
        EARTH_RADIUS * np.radians(lon - LON_0),
        EARTH_RADIUS * np.log(np.tan((np.pi / 4.0) + (np.radians(lat) / 2.0))),
    )


def marcatorInverseProject(x, y):
    return (
        np.degrees(x / EARTH_RADIUS),
        np.degrees(2.0 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2.0),
    )


def convert_svg_to_png(svg_string, fname):
    svg = ET.fromstring(svg_string)
    p = svg.findall("{http://www.w3.org/2000/svg}g")[0]

    admissible_colors = [
        "rgb(100%,100%,100%)",
        "rgb(96.862745%,98.039216%,74.901961%)",  # yellow roads
        "rgb(90.980392%,57.254902%,63.529412%)",  # red roads
        "rgb(98.823529%,83.921569%,64.313725%)",  # orange roads
        "rgb(97.647059%,69.803922%,61.176471%)",  # 2nd type of orange road
        "rgb(86.666667%,86.666667%,90.980392%)",  # normal pedestrian zone
        "rgb(92.941176%,92.941176%,92.941176%)",  # Stadelhofen-type pedestrian zone
    ]

    # apply filters that make use of style attribute
    children = list(p.iter())
    for child in children:
        style = child.get("style")
        if style is None:
            continue
        # remove rail tracks
        if "stroke-dasharray" in style:
            p.remove(child)
            continue
        # remove all no abmissibly colored elements
        admissible = False
        for color in admissible_colors:
            if color in style:
                admissible = True
        if not admissible:
            p.remove(child)

    # apply filters that make use of d attribute
    children = list(p.iter())
    for child in children:
        d = child.get("d")
        if d is None:
            continue
        # remove complex geometry such as letters
        if "C " in d or "Z " in d:
            p.remove(child)

    # paint everything that survived black
    children = list(p.iter())
    for child in children:
        style = child.get("style")
        if style is None:
            continue
        for color in admissible_colors:
            if color in style:
                style = style.replace(color, "rgb(0%,0%,0%)")
        child.set("style", style)

    xmlstr = ElementTree.tostring(svg, encoding="unicode", method="xml")
    lines = []
    for line in xmlstr.splitlines():
        # line.encode()
        if "rgb(" in line and not "rgb(0%,0%,0%)" in line:
            continue
        else:
            lines.append(line)

    svg = "".join(lines).encode("ascii")

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(svg)
        tmp.flush()
        temp_filepath = f"/proc/{os.getpid()}/fd/{tmp.fileno()}"

        # write png file
        drawing = svg2rlg(temp_filepath)
        scale_width = BING_MAP_TILE_RESOLUTION / drawing.width
        scale_height = BING_MAP_TILE_RESOLUTION / drawing.height
        drawing.width = drawing.minWidth() * scale_width
        drawing.height = drawing.height * scale_width
        drawing.scale(scale_width, scale_height)
        drawing.height = BING_MAP_TILE_RESOLUTION
        renderPM.drawToFile(drawing, fname, fmt="PNG")


def download_target_process_task(dataset_df):
    with requests.Session() as session:
        ret = session.get("https://www.openstreetmap.org/", headers=header)
        cookie = ret.cookies
        cookie["_osm_location"] = "8.0751|47.2853|14|M"

    q = queue.Queue()
    # transform = transforms.CenterCrop(400)
    transform = transforms.Compose(
        [
            transforms.CenterCrop(400),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.invert(x)),
            transforms.ToPILImage(),
        ]
    )
    ecount = 0
    nr_dl = 0

    for index, row in dataset_df.iterrows():
        lon = row["lon"]
        lat = row["lat"]
        batch_nr = row["batch_nr"]

        if not os.path.isdir(f"{DATA_STORAGE_TARGET_PATH}BATCH_{int(batch_nr)}"):
            os.mkdir(f"{DATA_STORAGE_TARGET_PATH}BATCH_{int(batch_nr)}")

        x, y = mercatorProject(lon, lat)

        bbox_x1 = x - BBOX_WIDTH / 2.0
        bbox_x2 = x + BBOX_WIDTH / 2.0

        bbox_y1 = y - BBOX_WIDTH / 2.0
        bbox_y2 = y + BBOX_WIDTH / 2.0

        bbox_lat_1, bbox_lon_1 = marcatorInverseProject(bbox_x1, bbox_y1)
        bbox_lat_2, bbox_lon_2 = marcatorInverseProject(bbox_x2, bbox_y2)

        scale = SCALE

        q.put(
            ([bbox_lat_1, bbox_lat_2], [bbox_lon_1, bbox_lon_2], scale, index, batch_nr)
        )

    while not q.empty():
        bbox_lat, bbox_lon, scale, index, batch_nr = q.get()
        try:
            with requests.Session() as session:
                format_img = "svg"
                url = f"https://render.openstreetmap.org/cgi-bin/export?bbox={bbox_lat[0]},{bbox_lon[0]},{bbox_lat[1]},{bbox_lon[1]}&scale={scale}&format={format_img}"

                with session.get(url, cookies=cookie, headers=header) as res:
                    if res.status_code == 200:
                        data = res.text
                        fname = f"{DATA_STORAGE_TARGET_PATH}BATCH_{int(batch_nr)}/IMG_{index}_500x500.png"
                        convert_svg_to_png(data, fname)

                        image = Image.open(fname)
                        image = transform(image)
                        image.save(
                            f"{DATA_STORAGE_TARGET_PATH}BATCH_{int(batch_nr)}/IMG_{index}.png"
                        )
                        os.remove(fname)

                    else:
                        raise Exception(f"Http response: {res.status_code}")

        except Exception as e:
            print(f"failed at index: {index}, total nr of errors: {ecount}, error: {e}")
            ecount += 1
            # q.put((bbox_lat, bbox_lon, scale, index, batch_nr))
            continue

    print(f"failed at {ecount} samples")


def downlaod_openstreetmap_target_images(cities_df):
    N = cities_df.shape[0]

    indexes_to_drop = []
    for index, row in cities_df.iterrows():
        batch_nr = row["batch_nr"]
        if os.path.exists(
            f"{DATA_STORAGE_TARGET_PATH}/BATCH_{int(batch_nr)}/IMG_{index}.png"
        ):
            indexes_to_drop.append(index)

    filtered_cities_df = cities_df.drop(indexes_to_drop)
    data_split = np.array_split(filtered_cities_df, NR_PROCESSES)

    processes = []
    for i in range(NR_PROCESSES):
        split = data_split[i].copy()
        p = Process(target=download_target_process_task, args=(split,))
        print(f"process {i}")
        processes.append(p)
        p.start()

    for i in range(NR_PROCESSES):
        processes[i].join()


def main():
    DELETE_MISSING_SAMPLES = False

    print("Start downloading city coordinates:")
    city_coordinates = sample_city_coordinates()

    print("Start downloading bing map tiles:")
    download_bing_source_images(city_coordinates)

    print("Start downloading openstreetmap tiles:")
    downlaod_openstreetmap_target_images(city_coordinates)

    print("finished downloading")
    city_coordinates.to_parquet(f"{DATA_STORAGE_BASE}coordinates.parquet.gzip")

    src_missing = 0
    trg_missing = 0
    for index, row in city_coordinates.iterrows():
        batch_nr = row["batch_nr"]
        src_fname = f"{DATA_STORAGE_SOURCE_PATH}/BATCH_{int(batch_nr)}/IMG_{index}.png"
        trg_fname = f"{DATA_STORAGE_TARGET_PATH}/BATCH_{int(batch_nr)}/IMG_{index}.png"
        if os.path.exists(src_fname) ^ os.path.exists(trg_fname):
            if os.path.exists(src_fname):
                trg_missing += 1
                if DELETE_MISSING_SAMPLES:
                    os.remove(src_fname)

            elif os.path.exists(trg_fname):
                src_missing += 1
                if DELETE_MISSING_SAMPLES:
                    os.remove(trg_fname)

    print(f"Assymetrie in dataset: ")
    print(f"    {src_missing} source files missing")
    print(f"    {trg_missing} target files missing")
    print(
        "\nIf you want to delete the uncomplete samples change the variable DELETE_MISSING_SAMPLES to True and restart the script."
    )


if __name__ == "__main__":
    main()
