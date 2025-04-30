import os
from torch.utils import data
import torch
import numpy as np
from torchvision.datasets.folder import (
    IMG_EXTENSIONS,
    default_loader,
    has_file_allowed_extension,
)
from typing import Callable, Optional, Tuple, List, Any, Union
import torchvision.transforms.v2 as transforms
import torchvision.datapoints
import glob
from sklearn.model_selection import train_test_split

torchvision.disable_beta_transforms_warning()


class SegmentationDatasetBuilder:
    def __init__(
        self,
        dir_images: Union[str, None] = None,
        dir_masks: Union[str, None] = None,
        transform: Union[transforms.Transform, None] = None,
    ):
        if dir_images is not None and dir_masks is None:
            raise ValueError("dir_masks must be specified if dir_images is specified")
        if dir_images is None and dir_masks is not None:
            raise ValueError("dir_images must be specified if dir_masks is specified")
        if dir_images:
            self.add_path(dir_images, dir_masks, "Initialization")
        self.transform = transform
        self.dataset = {}

    def add_path(self, dir_images, dir_masks, name):
        self.dataset[name] = (dir_images, dir_masks)

    def set_test_dataset(self, name: Union[List[str], str], ratio: float = 0.2):
        if isinstance(name, str):
            name = [name]
        for n in name:
            if n not in self.dataset:
                raise ValueError(f"Dataset {n} not found in the added datasets")
        self.test_dataset = name
        self.test_ratio = ratio

    def get_datasets(self):
        if self.test_dataset is None:
            raise ValueError("Test dataset not specified")
        self.train_img_paths = []
        self.train_masks_paths = []
        self.test_img_paths = []
        self.test_masks_paths = []
        for n in self.dataset:
            img_path = self.dataset[n][0]
            mask_path = self.dataset[n][1]
            # List all paths in the dataset
            img_names = [
                os.path.basename(i)
                for i in sorted(glob.glob(os.path.join(img_path, "*")))
                if has_file_allowed_extension(i, IMG_EXTENSIONS)
            ]
            if n in self.test_dataset:
                # Split the dataset
                img_names_train, img_names_test = train_test_split(
                    img_names, test_size=self.test_ratio, random_state=42
                )
                self.test_img_paths += [
                    os.path.join(img_path, img_name) for img_name in img_names_test
                ]
                self.test_masks_paths += [
                    os.path.join(mask_path, img_name) for img_name in img_names_test
                ]
            else:
                img_names_train = img_names
            # Add to the dataset
            self.train_img_paths += [
                os.path.join(img_path, img_name) for img_name in img_names_train
            ]
            self.train_masks_paths += [
                os.path.join(mask_path, img_name) for img_name in img_names_train
            ]
        train_dataset = SegmentationDataset(
            self.train_img_paths, self.train_masks_paths, self.transform
        )
        test_dataset = SegmentationDataset(self.test_img_paths, self.test_masks_paths)
        print("Train dataset size:", len(train_dataset))
        print("Test dataset size:", len(test_dataset))
        return train_dataset, test_dataset


class SegmentationDataset(data.Dataset):
    def __init__(
        self,
        img_paths: List[str],
        masks_paths: List[str],
        transform: Union[transforms.Transform, None] = None,
    ):
        """A dataloader for segmentation datasets

        Args:
            img_paths (List[str]): List of paths to images
            masks_paths (List[str]): List of paths to masks
            transform (Union[transforms.Transform, None], optional): Transform to apply to the images.
                            Defaults to None.

        Note:
            Normalize, Lambda, Pad, ColorJitter and RandomErasing won't be applied to masks by default
        """
        super().__init__()
        self.transform = transform
        self.img_paths = img_paths
        self.mask_paths = masks_paths
        self.to_tensor = transforms.ToTensor()
        self.n = len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        img = default_loader(img_path)
        mask = default_loader(mask_path)

        assert img.size == mask.size

        # Convert img to tensor
        img = self.to_tensor(img)
        mask = torch.tensor(np.array(mask)[:, :, 0] == 255, dtype=torch.int)
        mask = torchvision.datapoints.Mask(mask)

        if not self.transform:
            return img, mask
        img, mask = self.transform(img, mask)

        return img, mask

    def __len__(self):
        return self.n
