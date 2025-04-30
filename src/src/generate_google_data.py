import cv2
import io
import os
import numpy as np
from urllib import request
from PIL import Image
from tqdm import tqdm
from functools import lru_cache

GOOGLE_API_KEY = ""
N_IMAGES = 1000


@lru_cache(maxsize=None)
def google_request(url):
    return request.urlopen(url)


def get_google_maps_image(lattitude, longitude, maptype="satellite"):
    url = (
        "https://maps.googleapis.com/maps/api/staticmap?"
        "center=" + str(lattitude) + "," + str(longitude) + "&zoom=18"
        "&size=420x400"
        "&maptype="
        + maptype
        + "&key="
        + GOOGLE_API_KEY
        + "&style=feature:all|element:labels|visibility:off"
        "&scale=1"
    )

    response = google_request(url)
    image = Image.open(io.BytesIO(response.read()))
    image = image.convert("RGB")
    image = np.asarray(image)[:, :, ::-1]
    return image


def save_images(city_name, satellite_image, road_image, index):
    filename = city_name + "-" + str(index).zfill(6) + ".png"
    cv2.imwrite("google-maps-data/images/" + filename, satellite_image)
    cv2.imwrite("google-maps-data/groundtruth/" + filename, road_image)


if __name__ == "__main__":
    cities = {
        "los-angeles": {
            "lat-range": [33.790468, 34.075231],
            "lon-range": [-117.791966, -118.423706],
        },
        "portland": {
            "lat-range": [45.424466, 45.590689],
            "lon-range": [-122.464725, -122.788790],
        },
        "sacramento": {
            "lat-range": [38.475246, 38.672489],
            "lon-range": [-121.346170, -121.538441],
        },
        "saltlake": {
            "lat-range": [40.512446, 40.731811],
            "lon-range": [-111.840435, -112.037188],
        },
        "san-jose": {
            "lat-range": [37.253936, 37.399746],
            "lon-range": [-121.794937, -122.079367],
        },
        "seattle": {
            "lat-range": [47.655287, 47.804959],
            "lon-range": [-122.277406, -122.386397],
        },
    }
    for city_name in cities:
        for i in tqdm(range(0, N_IMAGES)):
            while True:
                lat = np.random.uniform(
                    low=cities[city_name]["lat-range"][0],
                    high=cities[city_name]["lat-range"][1],
                )
                lon = np.random.uniform(
                    low=cities[city_name]["lon-range"][0],
                    high=cities[city_name]["lon-range"][1],
                )
                satellite_image = get_google_maps_image(lat, lon, maptype="satellite")
                road_image = get_google_maps_image(lat, lon, maptype="terrain")
                satellite_image = satellite_image[:, :400]
                road_image = road_image[:, :400]
                road_image = (
                    cv2.threshold(road_image, 253, 255, type=cv2.THRESH_BINARY)[1]
                    == 255
                ) * 255
                road_ratio = np.sum(road_image) / (
                    road_image.shape[0] * road_image.shape[1] * 255
                )
                if road_ratio > 0.05:
                    break
            cv2.imwrite(
                os.path.join(
                    "google-maps-data/images",
                    city_name + "-" + str(i).zfill(6) + ".png",
                ),
                satellite_image,
            )
            cv2.imwrite(
                os.path.join(
                    "google-maps-data/groundtruth",
                    city_name + "-" + str(i).zfill(6) + ".png",
                ),
                road_image,
            )
