# This is the training file that I have used to produce the best model so far.

# IDEAS
# - incorporate the rotating into the training and not just the inference
# - can images in rotate_model be improved with postprocessing? (e.g. thresholding)

from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
from networks.modded_unet import ModNet
from dataloader.segdataset import SegmentationDataset, SegmentationDatasetBuilder
import torch.utils.data
from networks.skip_connection import UnetSkipConnectionBlock
from utils import seed_everything
import torchmetrics
import torchvision.transforms.v2 as transforms
import argparse
from torchvision.datasets.folder import (
    default_loader,
)
import numpy as np
import ttach as tta
from train import DeepLab

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)
torchvision.disable_beta_transforms_warning()

seed_everything(42)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="cil-road-segmentation-2022/test"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--split_factor", type=int, default=8)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="runs/Jul14_15-56-56_scheschb-System-Product-Name/checkpoint/model_2700.pt",
    )
    return parser.parse_args()


def rotate_inference(model, img, num_rots):
    res = []
    # idea: rotate the image 4 times, rotate the prediction back, sum all those preds and then in the end sigmoid the sum (bc some vals will very likely be > 1)
    # this could work bc maybe the net learns certain feature detectors (i.e. kernels) only for specifc orientations and this way we increase the chance of activating those kernels
    for anlge in range(0, 360, 360 // num_rots):
        X_rot = transforms.functional.rotate(img, angle=anlge)
        pred = model(X_rot)
        pred = transforms.functional.rotate(pred, angle=-anlge)
        res.append(pred)
    res = torch.stack(res)
    res = torch.sum(res, dim=0)
    res = torch.sigmoid(res)
    res = res.squeeze(0)
    return res


if __name__ == "__main__":
    args = get_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", args.device)
    device = torch.device(args.device)

    os.makedirs(os.path.join(args.data_dir, "groundtruth"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "combined"), exist_ok=True)

    # Model
    model = DeepLab()
    if os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(
        model, tta.aliases.d4_transform(), merge_mode="mean"
    )
    with torch.no_grad():
        for file in tqdm(os.listdir(os.path.join(args.data_dir, "images"))):
            if not file.endswith(".png"):
                continue
            img = default_loader(os.path.join(args.data_dir, "images", file))
            img = transforms.ToTensor()(img)
            img = img.to(device)
            img = img.unsqueeze(0)
            pred = tta_model(img)
            pred = pred.squeeze(0)
            # pred = rotate_inference(model, img, 4)
            pred = pred > 0
            pred = pred.repeat(3, 1, 1)
            pred = pred * 255
            pred = pred.cpu()
            pred = pred.to(torch.uint8)
            torchvision.io.write_png(
                pred, os.path.join(args.data_dir, "groundtruth", file)
            )
            img = img.squeeze(0).cpu() * 255
            img = img.to(torch.uint8)
            combined_img = torch.cat((img, pred), dim=2)
            torchvision.io.write_png(
                combined_img, os.path.join(args.data_dir, "combined", file)
            )
