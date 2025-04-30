# script that loads a pretrained model and tries to enhance its performance by applying it to rotated/zoomed versions of the input image

# TAKEAWAYS:
# - rots help a lot to uncover narrow roads
# - zooming does not really help and introduces a good amount of noise

from variables import *
from dataset.dataloader import DataframeDataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
from torchvision import transforms
from networks.modded_unet import ModNet

num_batches_to_use = 4  # 4 was chosen to have an even mix between OSM data and data that was provided by the CIL team
DATA_FRACTION_2_USE = 0.2
TRAIN_EVAL_SPLIT = 0.99
NUM_WORKERS = 6
split_factor = 8
dtype = torch.float32

# processing params
num_rots = 36
threshold_val = 0.85


def rotate(X, proposer, enhancer, dtype=dtype):
    res = torch.zeros((X.shape[0], 1, X.shape[2], X.shape[3]), dtype=dtype)
    # idea: rotate the image 4 times, rotate the prediction back, sum all those preds and then in the end sigmoid the sum (bc some vals will very likely be > 1)
    # this could work bc maybe the net learns certain feature detectors (i.e. kernels) only for specifc orientations and this way we increase the chance of activating those kernels
    for anlge in range(0, 360, 360 // num_rots):
        X_rot = transforms.functional.rotate(X, angle=anlge)
        pred = proposer(X_rot)
        pred = torch.cat(
            (pred, X_rot), 1
        )  # not sure how much having 2 unets helps (vs just having one 2x the size) BUT what I think definitely helped is "injecting" the sat_im into the process a second time (which happens here)
        pred = enhancer(pred)
        pred = transforms.functional.rotate(pred, angle=-anlge)
        res += pred
    res = torch.sigmoid(res)
    return res


# idea for improving narrow roads: cut im into 4 patches, upsample each to 400x400, invoke model on all 4 patches, downsample back to 200x200, stitch together
# hope is that the model can then apply the kernels it has learnt for larger roads to the smaller roads as well
def zoom(X, proposer, enhancer, dtype=dtype):
    res = torch.zeros((X.shape[0], 1, X.shape[2], X.shape[3]), dtype=dtype)
    # process the patches
    for i in range(4):
        if i == 0:
            X_zoom = X[:, :, :200, :200]
        elif i == 1:
            X_zoom = X[:, :, :200, 200:]
        elif i == 2:
            X_zoom = X[:, :, 200:, :200]
        elif i == 3:
            X_zoom = X[:, :, 200:, 200:]

        X_zoom = transforms.Resize((400, 400))(X_zoom)
        pred = proposer(X_zoom)
        pred = torch.cat(
            (pred, X_zoom), 1
        )  # not sure how much having 2 unets helps (vs just having one 2x the size) BUT what I think definitely helped is "injecting" the sat_im into the process a second time (which happens here)
        pred = enhancer(pred)
        pred = transforms.Resize((200, 200))(pred)

        if i == 0:
            res[:, :, :200, :200] = pred
        elif i == 1:
            res[:, :, :200, 200:] = pred
        elif i == 2:
            res[:, :, 200:, :200] = pred
        elif i == 3:
            res[:, :, 200:, 200:] = pred

    # normal prediction
    pred = proposer(X)
    pred = torch.cat(
        (pred, X), 1
    )  # not sure how much having 2 unets helps (vs just having one 2x the size) BUT what I think definitely helped is "injecting" the sat_im into the process a second time (which happens here)
    pred = enhancer(pred)

    res = res + pred
    # normalize back down to an im with values between 0 and 1
    res = torch.sigmoid(res)
    return res


def rot_and_zoom(X, proposer, enhancer, dtype=dtype):
    res = torch.zeros((X.shape[0], 1, X.shape[2], X.shape[3]), dtype=dtype)
    for anlge in range(0, 360, 360 // num_rots):
        X_rot = transforms.functional.rotate(X, angle=anlge)
        pred = zoom(X_rot, proposer, enhancer)
        pred = transforms.functional.rotate(pred, angle=-anlge)
        res += pred
    res = torch.sigmoid(res)
    return res


def normal_pred(X, proposer, enhancer):
    pred = proposer(X)
    pred = torch.cat(
        (pred, X), 1
    )  # not sure how much having 2 unets helps (vs just having one 2x the size) BUT what I think definitely helped is "injecting" the sat_im into the process a second time (which happens here)
    pred = enhancer(pred)
    return pred


def threshold(im):
    im[im >= threshold_val] = 1
    im[im < threshold_val] = 0
    return im


if __name__ == "__main__":
    # Load models
    nets_path = (
        "/Users/lucaserlacher/Documents/Studium/Master/23FS/CIL/CIL23/runs/UNet/"
    )
    path = nets_path + "28_epochs"
    proposer = ModNet(3, 1, split_factor=split_factor, dilation_in=True, dtype=dtype)
    enhancer = ModNet(4, 1, split_factor=split_factor, dilation_in=True, dtype=dtype)
    proposer.load_state_dict(
        torch.load(path + "/proposer.pt", map_location=torch.device("cpu"))
    )
    enhancer.load_state_dict(
        torch.load(path + "/enhancer.pt", map_location=torch.device("cpu"))
    )
    proposer.eval()
    enhancer.eval()

    # Load eval data (model has not seen this data during training)
    in_ims = []
    eval_data_path = "/Users/lucaserlacher/Documents/Studium/Master/23FS/CIL/CIL23/src/dataset/data/test/images"
    # Load all images at eval_data_path
    for im_path in glob.glob(f"{eval_data_path}/*"):
        im = Image.open(im_path)
        im = transforms.ToTensor()(im)
        # Remove alpha channel
        im = im[:3, :, :]
        im = im.unsqueeze(0)
        in_ims.append(im)

    # Split in_ims into batches of size 8
    in_ims = torch.cat(in_ims, dim=0)
    in_ims = torch.split(in_ims, 8, dim=0)

    # Just use the first batch
    num_ims = 8  # if this is too high zsh will kill the process
    in_ims = [
        [in_ims[1]][0][:num_ims, :, :, :]
    ]  # my god this is ugly (but it is midnight and I am tired)

    rot_preds = []
    normal_preds = []
    rot_and_thresh_preds = []
    # For the first 10 images in eval_loader
    for i, im in enumerate(in_ims):
        normal_preds.append(normal_pred(im, proposer, enhancer))
        rot_preds.append(rotate(im, proposer, enhancer))
        rot_and_thresh_preds.append(threshold(torch.clone(rot_preds[i])))

    # Plot preds, non_rotated_preds, in_ims using matplotlib
    fig, axs = plt.subplots(4, num_ims)
    for i in range(num_ims):
        axs[0, i].imshow(in_ims[0][i].permute(1, 2, 0).detach().numpy())
        axs[1, i].imshow(normal_preds[0][i].permute(1, 2, 0).detach().numpy())
        axs[2, i].imshow(rot_preds[0][i].permute(1, 2, 0).detach().numpy())
        axs[3, i].imshow(rot_and_thresh_preds[0][i].permute(1, 2, 0).detach().numpy())
    # label y axis with discrete labels
    axs[0, 0].set_ylabel("Input")
    axs[1, 0].set_ylabel("Normal")
    axs[2, 0].set_ylabel("Rotated")
    axs[3, 0].set_ylabel("Rotated + thresholded")
    plt.show()
