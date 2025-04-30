# simples baseline possible

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from statistics import mean


# a color is grey if its rgb components are close to each other and are not too large (white) or too small (black)
def acceptable_color(color):
    # PARAMS
    grey_tolerance = 10
    lower_bound = 0
    upper_bound = 255

    # Check if RGB components are close to each other
    color_range = max(color) - min(color)
    if color_range > grey_tolerance:
        return False

    color_avg = mean(color)
    if (color_avg < lower_bound) or (color_avg > upper_bound):
        return False

    return True


def compute_seg(sat_im):
    im = np.array(sat_im)
    # remove alpha channel of png
    data = im[:, :, 1:]

    out_im = np.zeros((len(data), len(data), 3))
    for i in range(len(im[0])):
        for j in range(len(data[1])):
            if acceptable_color(data[i][j]):
                out_im[i][j] = [255, 255, 255]

    return out_im


if __name__ == "__main__":
    sat_im = Image.open("../dataset/test/satimage_1.png")
    mask = Image.open("../dataset/test/groundtruth_1.png")

    segmentation = compute_seg(sat_im)

    f, axarr = plt.subplots(3)
    axarr[0].imshow(sat_im)
    axarr[1].imshow(mask)
    axarr[2].imshow(segmentation)
    plt.show(block=False)
    plt.pause(3)
    plt.close()
