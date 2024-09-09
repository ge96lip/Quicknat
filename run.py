import argparse
import os

import torch
import utils.evaluator as eu

# NEW MONAI CODE
# from quicknat import QuickNat
# from monai.networks.nets import UNet
from quicknat import QuickNat
from quicknetMonai import QuickNAT

from config_files.settings import Settings
from utils.data_utils import get_imdb_dataset
from utils.log_utils import LogWriter
import shutil
from utils.data_loader import SLDataset

import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate as ipol
from unet import UNet
from solver import Solver

torch.set_default_tensor_type("torch.FloatTensor")


# ====================================================================================================================
# Transforms
# ====================================================================================================================


class elastic_deform(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data["image"]
        lab = data["label"]
        wgt = data["weight"]

        x_coo = np.random.randint(100, 300)
        y_coo = np.random.randint(100, 300)
        dx = np.random.randint(10, 40)
        dy = np.random.randint(10, 40)
        if random.random() < self.p:
            img = elastic_deformation(img, x_coo, y_coo, dx, dy)
            lab = elastic_deformation(lab, x_coo, y_coo, dx, dy)
            wgt = elastic_deformation(wgt, x_coo, y_coo, dx, dy)

            lab = np.where(lab <= 20, 0, lab)
            lab = np.where(lab > 20, 255, lab)

        return {"image": img, "label": lab, "weight": wgt}


def elastic_deformation(image, x_coord, y_coord, dx, dy):
    """Applies random elastic deformation to the input image
        with given coordinates and displacement values of deformation points.
        Keeps the edge of the image steady by adding a few frame points that get displacement value zero.
    Input: image: array of shape (N.M,C) (Haven't tried it out for N != M), C number of channels
           x_coord: array of shape (L,) contains the x coordinates for the deformation points
           y_coord: array of shape (L,) contains the y coordinates for the deformation points
           dx: array of shape (L,) contains the displacement values in x direction
           dy: array of shape (L,) contains the displacement values in x direction
    Output: the deformed image (shape (N,M,C))
    """

    image = image.transpose((2, 1, 0))
    ## Preliminaries
    # dimensions of the input image
    shape = image.shape

    # centers of x and y axis
    x_center = shape[1] / 2
    y_center = shape[0] / 2

    ## Construction of the coarse grid
    # deformation points: coordinates

    # anker points: coordinates
    x_coord_anker_points = np.array(
        [0, x_center, shape[1] - 1, 0, shape[1] - 1, 0, x_center, shape[1] - 1]
    )
    y_coord_anker_points = np.array(
        [0, 0, 0, y_center, y_center, shape[0] - 1, shape[0] - 1, shape[0] - 1]
    )
    # anker points: values
    dx_anker_points = np.zeros(8)
    dy_anker_points = np.zeros(8)

    # combine deformation and anker points to coarse grid
    x_coord_coarse = np.append(x_coord, x_coord_anker_points)
    y_coord_coarse = np.append(y_coord, y_coord_anker_points)
    coord_coarse = np.array(list(zip(x_coord_coarse, y_coord_coarse)))

    dx_coarse = np.append(dx, dx_anker_points)
    dy_coarse = np.append(dy, dy_anker_points)

    ## Interpolation onto fine grid
    # coordinates of fine grid
    coord_fine = [[x, y] for x in range(shape[1]) for y in range(shape[0])]
    # interpolate displacement in both x and y direction
    dx_fine = ipol.griddata(
        coord_coarse, dx_coarse, coord_fine, method="cubic"
    )  # cubic works better but takes longer
    dy_fine = ipol.griddata(
        coord_coarse, dy_coarse, coord_fine, method="cubic"
    )  # other options: 'linear'
    # get the displacements into shape of the input image (the same values in each channel)

    dx_fine = dx_fine.reshape(shape[0:2])
    dx_fine = np.stack([dx_fine] * shape[2], axis=-1)
    dy_fine = dy_fine.reshape(shape[0:2])
    dy_fine = np.stack([dy_fine] * shape[2], axis=-1)

    ## Deforming the image: apply the displacement grid
    # base grid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    # add displacement to base grid (-> new coordinates)
    indices = (
        np.reshape(y + dy_fine, (-1, 1)),
        np.reshape(x + dx_fine, (-1, 1)),
        np.reshape(z, (-1, 1)),
    )
    # evaluate the image at the new coordinates
    # print("input ndim ", image.ndim)
    deformed_image = map_coordinates(image, indices, order=2, mode="nearest")
    deformed_image = deformed_image.reshape(image.shape)

    return deformed_image.transpose((2, 0, 1))


class ToTensor(object):
    def __call__(self, sample):
        image, labels, weight = sample["image"], sample["label"], sample["weight"]

        return {
            "image": torch.from_numpy(image.copy()),
            "label": torch.from_numpy(labels.copy()),
            "weight": torch.from_numpy(weight.copy()),
        }


def norm(ar):
    ar = ar - np.min(ar)
    ar = ar / np.ptp(ar)
    return ar


class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data["image"]
        lab = data["label"]
        wgt = data["weight"]

        if random.random() < self.p:
            img = np.flip(img, axis=0)
            lab = np.flip(lab, axis=0)
            wgt = np.flip(wgt, axis=0)

        return {"image": img, "label": lab, "weight": wgt}


class Resize(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, data):
        img = data["image"]
        lab = data["label"]
        wgt = data["weight"]

        img = np.squeeze(img)
        lab = np.squeeze(lab)
        wgt = np.squeeze(wgt)

        img = Image.fromarray(img.astype(np.float32))
        lab = Image.fromarray(lab.astype(np.float32))
        wgt = Image.fromarray(wgt.astype(np.float32))

        img = F.resize(img, (self.w, self.h))
        lab = F.resize(lab, (self.w, self.h))
        wgt = F.resize(wgt, (self.w, self.h))

        img = np.asarray(img)
        lab = np.asarray(lab)
        wgt = np.asarray(wgt)

        return {"image": img, "label": lab, "weight": wgt}


class RandomCrop(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, data):
        img = data["image"]
        lab = data["label"]
        wgt = data["weight"]

        img = np.squeeze(img)
        lab = np.squeeze(lab)
        wgt = np.squeeze(wgt)

        a = img.shape[0] - self.w
        b = img.shape[1] - self.h
        x = random.randint(0, a)
        y = random.randint(0, b)

        img = img[x : self.w + x, y : self.h + y]
        lab = lab[x : self.w + x, y : self.h + y]
        wgt = wgt[x : self.w + x, y : self.h + y]
        return {"image": img, "label": lab, "weight": wgt}


transform_train = transforms.Compose(
    [
        # Resize(256, 256),
        # RandomCrop(256, 256),
        # RandomVerticalFlip(p=0.25),
        ToTensor(),
    ]
)

transform_val = transforms.Compose(
    [
        # Resize(256, 256),
        # RandomCrop(256, 256),
        ToTensor(),
    ]
)


def train(train_params, common_params, data_params, net_params):
    train_data = SLDataset(data_params, "train", transform_train)
    val_data = SLDataset(data_params, "val", transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=train_params["train_batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=train_params["val_batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if train_params["use_pre_trained"]:
        quicknat_model = torch.load(train_params["pre_trained_path"])
    else:
        # NEW MONAI CODE
        #quicknat_model = QuickNat(net_params)
        #quicknat_model = UNet(spatial_dims = 2, in_channels = 1, out_channels=33, channels=(4,8,16,32), strides=(2,2,2), num_res_units=3)
        quicknat_model = QuickNAT(
            net_params["num_class"], 
            net_params["num_channels"],
            net_params["num_filters"], 
            net_params["kernel_h"],
            net_params["kernel_w"],
            net_params["kernel_c"], 
            net_params["stride_conv"],
            net_params["pool"],
            net_params["stride_pool"],
            net_params["se_block"],
            net_params["drop_out"]
        )

    solver = Solver(
        quicknat_model,
        device=common_params["device"],
        num_class=net_params["num_class"],
        optim_args={
            "lr": train_params["learning_rate"],
            "betas": train_params["optim_betas"],
            "eps": train_params["optim_eps"],
            "weight_decay": train_params["optim_weight_decay"],
        },
        model_name=common_params["model_name"],
        exp_name=train_params["exp_name"],
        labels=data_params["labels"],
        log_nth=train_params["log_nth"],
        num_epochs=train_params["num_epochs"],
        lr_scheduler_step_size=train_params["lr_scheduler_step_size"],
        lr_scheduler_gamma=train_params["lr_scheduler_gamma"],
        use_last_checkpoint=train_params["use_last_checkpoint"],
        log_dir=common_params["log_dir"],
        exp_dir=common_params["exp_dir"],
    )

    solver.train(train_loader, val_loader)
    final_model_path = os.path.join(
        common_params["save_model_dir"], train_params["final_model_file"]
    )
    quicknat_model.save(final_model_path)
    print("final model saved @ " + str(final_model_path))


def evaluate(eval_params, net_params, data_params, common_params, train_params):
    eval_model_path = eval_params["eval_model_path"]
    num_classes = net_params["num_class"]
    labels = data_params["labels"]
    data_dir = eval_params["data_dir"]
    label_dir = eval_params["label_dir"]
    volumes_txt_file = eval_params["volumes_txt_file"]
    remap_config = eval_params["remap_config"]
    device = common_params["device"]
    log_dir = common_params["log_dir"]
    exp_dir = common_params["exp_dir"]
    exp_name = train_params["exp_name"]
    save_predictions_dir = eval_params["save_predictions_dir"]
    prediction_path = os.path.join(exp_dir, exp_name, save_predictions_dir)
    orientation = eval_params["orientation"]

    logWriter = LogWriter(num_classes, log_dir, exp_name, labels=labels)

    #quicknat_model = QuickNat(net_params)
    #quicknat_model = UNet(spatial_dims = 2, in_channels = 1, out_channels=33, channels=(4,8,16,32), striddes=(2,2,2), num_res_units=3)
    quicknat_model = QuickNAT(
            net_params["num_class"], 
            net_params["num_channels"],
            net_params["num_filters"], 
            net_params["kernel_h"],
            net_params["kernel_w"],
            net_params["kernel_c"], 
            net_params["stride_conv"],
            net_params["pool"],
            net_params["stride_pool"],
            net_params["se_block"],
            net_params["drop_out"]
    )

    avg_dice_score, class_dist = eu.evaluate_dice_score(
        quicknat_model,
        eval_model_path,
        num_classes,
        data_dir,
        label_dir,
        volumes_txt_file,
        remap_config,
        orientation,
        prediction_path,
        device,
        logWriter,
    )
    logWriter.close()


def delete_contents(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", "-m", required=True, help="run mode, valid values are train and eval"
    )
    parser.add_argument(
        "--settings",
        "-s",
        required=True,
        help="which settings file to use, valid values are local and cluster",
    )

    args = parser.parse_args()

    if args.settings == "local":
        settings_file = "./Quicknat_codes/config_files/settings_local.ini"
    elif args.settings == "cluster":
        settings_file = "./Quicknat_codes/config_files/settings.ini"

    settings = Settings(settings_file)
    common_params, data_params, net_params, train_params, eval_params = (
        settings["COMMON"],
        settings["DATA"],
        settings["NETWORK"],
        settings["TRAINING"],
        settings["EVAL"],
    )

    if common_params["polyaxon_flag"] == True:
        # Change import polyaxon_helper for
        from polyaxon_client.tracking import get_data_paths, get_outputs_path

        common_params["log_dir"] = get_outputs_path()
        common_params["save_model_dir"] = get_outputs_path()
        common_params["exp_dir"] = get_outputs_path()
        data_params["data_dir"] = get_data_paths()["data1"] + data_params["data_dir"]

        print(data_params["data_dir"])

    if args.mode == "train":
        train(train_params, common_params, data_params, net_params)
    # elif args.mode == 'retrain':
    # train(train_params, common_params, data_params, net_params) all params for network frozen on top layers, training on bottom layers
    elif args.mode == "eval":
        evaluate(eval_params, net_params, data_params, common_params, train_params)
    elif args.mode == "clear":
        shutil.rmtree(os.path.join(common_params["exp_dir"], train_params["exp_name"]))
        print("Cleared current experiment directory successfully!!")
        shutil.rmtree(os.path.join(common_params["log_dir"], train_params["exp_name"]))
        print("Cleared current log directory successfully!!")

    elif args.mode == "clear-all":
        delete_contents(common_params["exp_dir"])
        print("Cleared experiments directory successfully!!")
        delete_contents(common_params["log_dir"])
        print("Cleared logs directory successfully!!")
    else:
        raise ValueError(
            "Invalid value for mode. only support values are train, eval and clear"
        )
