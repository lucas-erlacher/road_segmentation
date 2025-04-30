# compute mean and std dev of the sat ims in each channel (rgb).
# Computing over 40k dataset (but that should really be representative of the whole dataset).
# I don't think we ended up using the resulting statistics anywhere though ...

import os
import glob
import torch
import torchvision.transforms as transforms
from PIL import Image

if __name__ == "__main__":
    path = "../../dataset_40k/source_sample/BATCH_"
    in_ims = []
    convert_tensor = transforms.ToTensor()
    curr_mean = torch.zeros(3)
    curr_std = torch.zeros(3)
    for i in range(0, 10):
        for file_path in glob.glob(path + str(i) + "/*"):
            in_ims.append(convert_tensor(Image.open(file_path).convert("RGB")))
        in_ims = torch.stack(in_ims)
        mean = torch.mean(in_ims, dim=(0, 2, 3))
        std = torch.std(in_ims, dim=(0, 2, 3))
        curr_mean += mean
        curr_std += std
        in_ims = []

    mean = curr_mean / 10
    std = curr_std / 10

    print("mean: " + str(mean))
    print("std dev" + str(std))
