import os

import h5py
import numpy as np
import torch
from torch.utils import data
from PIL import Image


class SLDataset(data.Dataset):
    def __init__(self, data_params, phase, transforms=None):
        self.transforms = transforms

        if phase == "train":
            data_files = h5py.File(
                os.path.join(data_params["data_dir"], data_params["train_data_file"]),
                "r",
            )
            labels = h5py.File(
                os.path.join(data_params["data_dir"], data_params["train_label_file"]),
                "r",
            )
            class_weights = h5py.File(
                os.path.join(
                    data_params["data_dir"], data_params["train_class_weights_file"]
                ),
                "r",
            )
            weights = h5py.File(
                os.path.join(
                    data_params["data_dir"], data_params["train_weights_file"]
                ),
                "r",
            )

        if phase == "val":
            data_files = h5py.File(
                os.path.join(data_params["data_dir"], data_params["test_data_file"]),
                "r",
            )
            labels = h5py.File(
                os.path.join(data_params["data_dir"], data_params["test_label_file"]),
                "r",
            )
            class_weights = h5py.File(
                os.path.join(
                    data_params["data_dir"], data_params["test_class_weights_file"]
                ),
                "r",
            )
            weights = h5py.File(
                os.path.join(data_params["data_dir"], data_params["test_weights_file"]),
                "r",
            )

        self.data_files = data_files
        self.labels = labels
        self.class_weights = class_weights
        self.weights = weights

    def __getitem__(self, index):

        self.X = self.data_files["data"][index]
        self.y = self.labels["label"][index]
        self.w = self.class_weights["class_weights"][index]

        img = self.X
        label = self.y
        weight = self.w

        label_3d = np.expand_dims(label, axis=0)
        weight_3d = np.expand_dims(weight, axis=0)

        sample = {"image": img, "label": label_3d, "weight": weight_3d}

        if self.transforms is not None:
            sample = self.transforms(sample)

        img = sample["image"].unsqueeze(dim=0)
        label = sample["label"].squeeze()
        weight = sample["weight"].squeeze()

        return img, label, weight

    def __len__(self):
        return len(self.labels["label"])
