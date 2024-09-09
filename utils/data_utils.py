import os

import h5py
import nibabel as nb
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
try:
    import utils.preprocessor as preprocessor                     
except ImportError:
    import preprocessor
from PIL import Image
import random
from scipy.ndimage.interpolation import map_coordinates
from scipy import interpolate as ipol
from sklearn.utils import shuffle
import math
import torchvision.transforms.functional as F

# ====================================================================================================================
# Transforms
# ====================================================================================================================

class elastic_deform(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['label']
        wgt = data['weight']

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

        return {'image': img, 'label': lab, 'weight': wgt}


def elastic_deformation(image, x_coord, y_coord, dx, dy):
    """ Applies random elastic deformation to the input image
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
    x_coord_anker_points = np.array([0, x_center, shape[1] - 1, 0, shape[1] - 1, 0, x_center, shape[1] - 1])
    y_coord_anker_points = np.array([0, 0, 0, y_center, y_center, shape[0] - 1, shape[0] - 1, shape[0] - 1])
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
    dx_fine = ipol.griddata(coord_coarse, dx_coarse, coord_fine, method='cubic')  # cubic works better but takes longer
    dy_fine = ipol.griddata(coord_coarse, dy_coarse, coord_fine, method='cubic')  # other options: 'linear'
    # get the displacements into shape of the input image (the same values in each channel)


    dx_fine = dx_fine.reshape(shape[0:2])
    dx_fine = np.stack([dx_fine] * shape[2], axis=-1)
    dy_fine = dy_fine.reshape(shape[0:2])
    dy_fine = np.stack([dy_fine] * shape[2], axis=-1)

    ## Deforming the image: apply the displacement grid
    # base grid
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    # add displacement to base grid (-> new coordinates)
    indices = np.reshape(y + dy_fine, (-1, 1)), np.reshape(x + dx_fine, (-1, 1)), np.reshape(z, (-1, 1))
    # evaluate the image at the new coordinates
    #print("input ndim ", image.ndim)
    deformed_image = map_coordinates(image, indices, order=2, mode='nearest')
    deformed_image = deformed_image.reshape(image.shape)

    return deformed_image.transpose((2, 0, 1))


class ToTensor(object):
    def __call__(self, sample):
        image, labels, weight = sample['image'], sample['label'], sample['weight']

        return {'image': torch.from_numpy(image.copy()),
                'label': torch.from_numpy(labels.copy()),
                'weight': torch.from_numpy(weight.copy())}

def norm(ar):
    ar = ar - np.min(ar)
    ar = ar / np.ptp(ar)
    return ar


class RandomVerticalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, data):
        img = data['image']
        lab = data['label']
        wgt = data['weight']

        if random.random() < self.p:
            img = np.flip(img, axis=0)
            lab = np.flip(lab, axis=0)
            wgt = np.flip(wgt, axis=0)

        return {'image': img, 'label': lab, 'weight': wgt}


transform_train = transforms.Compose([
    #transforms.Resize((288, 288)),
    #RandomVerticalFlip(p=0.5),
    ToTensor(),
])

transform_val = transforms.Compose([
    #transforms.Resize((288, 288)),
    #transforms.RandomCrop(256, pad_if_needed=True),
    ToTensor(),
])

# ====================================================================================================================
# Data Loader
# ====================================================================================================================



class ImdbData(data.Dataset):
    def __init__(self, X, y, w, w_mfb, transforms=None):
        self.X = X if len(X.shape) == 4 else X[:, np.newaxis, :, :]
        self.y = y
        self.w = w
        self.w_mfb = w_mfb
        self.transforms = transforms

    def __getitem__(self, index):

        img = self.X[index]
        label = self.y[index]
        weight = self.w[index]

        label_3d = np.expand_dims(label, axis=0)
        weight_3d = np.expand_dims(weight, axis=0)

        sample = {'image': img, 'label': label_3d, 'weight': weight_3d}

        if self.transforms is not None:
            sample = self.transforms(sample)

        weight_mfb = torch.from_numpy(self.w_mfb)

        img = sample['image']
        label = sample['label'].squeeze()
        weight = sample['weight'].squeeze()


        return img, label, weight, weight_mfb

    def __len__(self):
        return len(self.y)


def get_imdb_dataset(data_params):
    data_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_data_file']), 'r')
    label_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_label_file']), 'r')
    class_weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_class_weights_file']), 'r')
    weight_train = h5py.File(os.path.join(data_params['data_dir'], data_params['train_weights_file']), 'r')

    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
    class_weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
    weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')


    return (ImdbData(data_train['data'][()], label_train['label'][()], class_weight_train['class_weights'][()],
                     weight_train['weights'][()], transforms=transform_train),
            ImdbData(data_test['data'][()], label_test['label'][()], class_weight_test['class_weights'][()],
                     weight_test['weights'][()], transforms=transform_val))

def get_test_dataset(data_params):
    data_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_data_file']), 'r')
    label_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_label_file']), 'r')
    class_weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_class_weights_file']), 'r')
    weight_test = h5py.File(os.path.join(data_params['data_dir'], data_params['test_weights_file']), 'r')

    return(ImdbData(data_test['data'][()], label_test['label'][()], class_weight_test['class_weights'][()],
                     weight_test['weights'][()], transforms=None))


def load_dataset(file_paths,
                 orientation,
                 remap_config,
                 return_weights=False,
                 reduce_slices=False,
                 remove_black=False):
    print("Loading and preprocessing data...")
    volume_list, labelmap_list, headers, class_weights_list, weights_list = [], [], [], [], []

    for file_path in file_paths:
        volume, labelmap, class_weights, weights, header = load_and_preprocess(file_path, orientation,
                                                                               remap_config=remap_config,
                                                                               reduce_slices=reduce_slices,
                                                                               remove_black=remove_black,
                                                                               return_weights=return_weights)

        volume_list.append(volume)
        labelmap_list.append(labelmap)

        if return_weights:
            class_weights_list.append(class_weights)
            weights_list.append(weights)

        headers.append(header)

        print("#", end='', flush=True)
    print("100%", flush=True)
    if return_weights:
        return volume_list, labelmap_list, class_weights_list, weights_list, headers
    else:
        return volume_list, labelmap_list, headers


def load_and_preprocess(file_path, orientation, remap_config, reduce_slices=False,
                        remove_black=False,
                        return_weights=False,
                        resize_var=True, shuffle_var=True, label_available=True):
    volume, labelmap, header = load_data(file_path, orientation, resize_var=resize_var, shuffle_var=shuffle_var, label_available=label_available)

    volume, labelmap, class_weights, weights = preprocess(volume, labelmap, remap_config=remap_config,
                                                          reduce_slices=reduce_slices,
                                                          remove_black=remove_black,
                                                          return_weights=return_weights)
    return volume, labelmap, class_weights, weights, header


def load_data(file_path, orientation, resize_var=True, shuffle_var=True, label_available=True):
    # Used for Convert_h5
    #vol_str = file_path[0].split('.')[0] + '_2d_vol_r.' + file_path[0].split('.')[-1]
    #label_str = file_path[1].split('.')[0] + '_2d_label_r.' + file_path[1].split('.')[-1]
    # Used for evaluation
    vol_str = file_path[0]
    label_str = file_path[1]
    
    if label_available:
        volume_nifty_4d, labelmap_nifty_4d = nb.load(vol_str), nb.load(label_str)
    else:
        # Delete the duplicated .nii
        vol_str = file_path[0].replace('.nii.nii', '.nii')
        volume_nifty_4d = nb.load(vol_str)

    # take only three dimensions from volume_nifty_4d
    volume = np.squeeze(volume_nifty_4d.get_fdata())
    print("Loading '{}'".format(vol_str))
    print("volume shape: ", volume.shape)
    
    w, h, s = volume.shape
    
    if label_available:
        labelmap = np.squeeze(labelmap_nifty_4d.get_fdata())
        print("label shape: ", labelmap.shape)
    else:
        labelmap =  np.zeros((w, h, s))

    # normalize pixel values
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    
    if resize_var:

        # Used for eval only
        #num_s = math.ceil(s/4)*4
        #num_h = math.ceil(h/16)*16
        #num_w = math.ceil(w/16)*16
        
        # Used for Convert_h5
        num_s = math.ceil(s/4)*4
        #num_s = 552
        num_h = 384
        num_w = 448
        
        
        if s < num_s:
            array = np.zeros((w, h, num_s-s))
            volume, labelmap = np.concatenate((volume, array), axis=2), np.concatenate((labelmap, array), axis=2)
        else:
            span = int((s-num_s)/2)
            volume, labelmap = volume[:, :, span:span+num_s], labelmap[:, :, span:span+num_s]

        if h < num_h:
            array = np.zeros((w, num_h-h, num_s))
            volume, labelmap = np.concatenate((array, volume), axis=1), np.concatenate((array, labelmap), axis=1)
        else:
            span = int((h-num_h)/2)
            volume, labelmap = volume[:, span:span+num_h, :], labelmap[:, span:span+num_h, :]
     
        if w < num_w:
            array = np.zeros((num_w-w, num_h, num_s))
            volume, labelmap = np.concatenate((volume, array), axis=0), np.concatenate((labelmap, array), axis=0)
        else:
            span = int((w-num_w)/2)
            volume, labelmap = volume[span:span+num_w, :, :], labelmap[span:span+num_w, :, :]

        print("new volume shape ", volume.shape)
        if label_available:
            print("new label shape ", labelmap.shape)
        volume, labelmap = preprocessor.rotate_orientation(volume, labelmap, orientation)

    return volume, labelmap, volume_nifty_4d.header

def preprocess(volume, labelmap, remap_config, reduce_slices=False, remove_black=False, return_weights=False):
    if reduce_slices:
        volume, labelmap = preprocessor.reduce_slices(volume, labelmap)

    if remap_config:
        labelmap = preprocessor.remap_labels(labelmap, remap_config)

    if remove_black:
        volume, labelmap = preprocessor.remove_black(volume, labelmap)

    if return_weights:
        class_weights, weights = preprocessor.estimate_weights_mfb(labelmap)
        return volume, labelmap, class_weights, weights
    else:
        return volume, labelmap, None, None


def load_file_paths(data_dir, label_dir=None, volumes_txt_file=None):
    """
    This function returns the file paths combined as a list where each element is a 2 element tuple, 0th being data and 1st being label.
    It should be modified to suit the need of the project
    :param data_dir: Directory which contains the data files
    :param label_dir: Directory which contains the label files
    :param volumes_txt_file: (Optional) Path to the a csv file, when provided only these data points will be read
    :return: list of file paths as string
    """

    volume_exclude_list = ['IXI290', 'IXI423']
    if volumes_txt_file:
        with open(volumes_txt_file) as file_handle:
            volumes_to_use = file_handle.read().splitlines()
    else:
        volumes_to_use = [name for name in os.listdir(data_dir)]

    file_paths = [
        [os.path.join(data_dir, vol+'.nii'), os.path.join(label_dir, vol+'.nii')]
       for
        vol in volumes_to_use]

    return file_paths
