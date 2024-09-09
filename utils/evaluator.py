import os

import nibabel as nib
import numpy as np
import torch

import utils.common_utils as common_utils
import utils.data_utils as du

from utils.data_utils import get_imdb_dataset, get_test_dataset

import matplotlib.pyplot as plt


def dice_confusion_matrix(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_cm = torch.zeros(num_classes, num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        for j in range(num_classes):
            Pred = (vol_output == j).float()
            inter = torch.sum(torch.mul(GT, Pred))
            union = torch.sum(GT) + torch.sum(Pred) + 0.0001
            dice_cm[i, j] = 2 * torch.div(inter, union)
    avg_dice = torch.mean(torch.diagflat(dice_cm))
    return avg_dice, dice_cm


def dice_score_perclass(vol_output, ground_truth, num_classes, no_samples=10, mode='train'):
    dice_perclass = torch.zeros(num_classes)
    if mode == 'train':
        samples = np.random.choice(len(vol_output), no_samples)
        vol_output, ground_truth = vol_output[samples], ground_truth[samples]
    for i in range(num_classes):
        GT = (ground_truth == i).float()
        Pred = (vol_output == i).float()
        inter = torch.sum(torch.mul(GT, Pred))
        union = torch.sum(GT) + torch.sum(Pred) + 0.0001
        dice_perclass[i] = (2 * torch.div(inter, union))
    return dice_perclass


def image_per_epoch(prediction):
    print("Sample Images...", end='', flush=True)
    ncols = 1
    nrows = len(prediction)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 20))
    for i in range(nrows):
        ax[i].imshow(prediction[i], cmap='CMRmap', vmin=0, vmax=1)
        ax[i].set_title("Predicted", fontsize=10, color="blue")
        ax[i].axis('off')
    fig.set_tight_layout(True)
    plt.show(fig)
    print('printed', flush=True)


def evaluate_dice_score(model, model_path, num_classes, data_dir, label_dir, volumes_txt_file, remap_config, orientation,
                        prediction_path, device=0, logWriter=None, mode='eval'):
    print("**Starting evaluation. Please check tensorboard for plots if a logWriter is provided in arguments**")

    batch_size = 4

    with open(volumes_txt_file) as file_handle:
        volumes_to_use = file_handle.read().splitlines()

    print("Running on: ", device)
    print("Using model from: ", model_path)

    if not device:
        model.load_state_dict(torch.load(model_path), strict=False)
        model.to(device)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        model.cuda(device)

    model.eval()

    common_utils.create_if_not(prediction_path)
    volume_dice_score_list = []
    file_paths = du.load_file_paths(data_dir, volumes_txt_file)
    file_paths = [[x, y] for x, y in file_paths if x.split('/')[-1][0] != '@']
    with torch.no_grad():
        for vol_idx, file_path in enumerate(file_paths):
            volume, labelmap, class_weights, weights, header = du.load_and_preprocess(file_path,
                                                                                      orientation=orientation,
                                                                                      remap_config=remap_config,
                                                                                      resize_var=True, shuffle_var=False, label_available=False)

            volume = volume if len(volume.shape) == 4 else volume[:, np.newaxis, :, :]
            volume = torch.tensor(volume).type(torch.FloatTensor)

            volume_prediction = []
            for i in range(0, len(volume), batch_size):
                batch_x = volume[i: i + batch_size]
                if cuda_available:
                    batch_x = batch_x.cuda(device)
                out = model(batch_x)

                _, batch_output = torch.max(out, dim=1)
                volume_prediction.append(batch_output)

            volume_prediction = torch.cat(volume_prediction)

            volume_prediction = (volume_prediction.cpu().numpy())
            nifti_img = nib.MGHImage(np.squeeze(volume_prediction).astype('uint8'), np.eye(4), header=header)
            
            # ------------------------------------------------------------------------------------------------------------------
            vol_str = file_path[0].replace('.nii.nii', '.nii').split('/')[-1]
            print('Saving {0} at: {1}'.format(vol_str, prediction_path))
            print('------------------------------------------------------------------------------------------------------------')
            nib.save(nifti_img, os.path.join(prediction_path, vol_str))
            # ------------------------------------------------------------------------------------------------------------------
            
    print("DONE")

    return None, None
