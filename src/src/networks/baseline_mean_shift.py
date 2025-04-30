# this sctipt is veeery slow (because mean shift just is super slow)

from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import random as rand
from sklearn.cluster import mean_shift


def compute_seg(im):
    data = np.array(im)
    # remove alpha channel of png
    data = data[:, :, 1:]

    init_features = np.zeros((len(data), len(data), 5))
    for i in range(len(data)):
        for j in range(len(data)):
            init_features[i][j] = np.append(data[i][j], [i, j])

    init_features = init_features.reshape((-1, 5))

    final_features = mean_shift(init_features)

    return im


if __name__ == "__main__":
    sat_im = Image.open("../dataset/test/satimage_0.png")
    mask = Image.open("../dataset/test/groundtruth_0.png")

    segmentation = compute_seg(sat_im)

    f, axarr = plt.subplots(3)
    axarr[0].imshow(sat_im)
    axarr[1].imshow(mask)
    axarr[2].imshow(segmentation)
    plt.show()
