import os
import time
import glob
import sys
import argparse
import json
from pathlib import Path
sys.path.append("../..")
#import opensimplex

#from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import set_determinism
from monai.data.utils import pad_list_data_collate
from torch.amp import autocast
from tqdm import tqdm
import random

import nibabel as nib

from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler


from monai.utils import StrEnum
from typing import Union

import pandas as pd

import utils.custom_transforms as custom_transforms
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm
import utils.thor_ddpm as thor_ddpm
from utils.utils import define_instance

from monai.metrics import compute_iou, DiceMetric

from scipy import stats
import copy

def scale_intensity_from_histogram_peak(input_image, target_value=1.0):
    # to be used only on mri images with intensities between 0 and 1
    input_np = input_image.cpu().numpy()

    hist, bin_edges = np.histogram(input_np.flatten(), bins=100, range=(np.max(input_np)/15.0, 0.8))

    peak_value = bin_edges[np.argmax(hist)]

    normalized_image = input_image / peak_value * target_value

    return normalized_image

def launch_compute_metrics_thor_anomaly_detection(args):
    DEVICE_TYPE = "cuda:0"
    device = torch.device(DEVICE_TYPE)

    set_determinism(0)

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/"
    if args.dataset["save_anomaly_maps"]:
        os.makedirs(ANOMALY_MAPS_DIR, exist_ok=True)

    IMAGE_SIZE = args.image_size

    model_path = f"{args.root_dir}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/{SUB_EXPERIMENT_NAME}_best_model.pth"

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)

    NOISE_MIN = int(args.compute_metrics_reconstruction["noise_rate_min"]*args.noise["num_timesteps_full_noise"])
    NOISE_MAX = int(args.compute_metrics_reconstruction["noise_rate_max"]*args.noise["num_timesteps_full_noise"])+1
    NOISE_INTERVAL = int(args.compute_metrics_reconstruction["noise_timesteps_interval"])
    NOISE_RANGE = range(NOISE_MIN,NOISE_MAX,args.compute_metrics_reconstruction["noise_timesteps_interval"])

    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'

    TEXTCOLOR = 'black'
    plt.rcParams['text.color'] = TEXTCOLOR
    plt.rcParams['axes.labelcolor'] = TEXTCOLOR
    plt.rcParams['xtick.color'] = TEXTCOLOR
    plt.rcParams['ytick.color'] = TEXTCOLOR


    # ----------- MODEL SETTINGS -----------

    if args.dataset["test"] == "brats":
        test_anomaly_images = sorted(glob.glob(ROOT_DIR+"datasets/final_flair_dataset_small/brats_registered/*.nii.gz"))[:300] #otherwise there are too many images (1200)
        test_masks = sorted(glob.glob(ROOT_DIR+"datasets/final_flair_dataset_small/brats_masks_registered/*.nii.gz"))[:300] # TODO

        # Read the CSV file and put every line in a list
        masks_to_exclude = []
        
        with open(ROOT_DIR+"AnoDiffExperiments/data_splits_lists/final_flair_dataset_small/exclude_brats_middle_slice.csv", 'r') as f:
            for line in f:
                masks_to_exclude.append(line.strip())
        images_to_exclude = [name.replace("seg", "t2f") for name in masks_to_exclude]

        test_anomaly_images = [path for path in test_anomaly_images if os.path.basename(path) not in images_to_exclude]
        test_masks = [path for path in test_masks if os.path.basename(path) not in masks_to_exclude]
        #print(test_anomaly_images)

        num_workers = 4
        ano_batch_size = 32

        test_anomaly_transforms = define_instance(args, "val_transforms")
        test_anomaly_ds = CacheDataset(data=test_anomaly_images, transform=test_anomaly_transforms)

        test_anomaly_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_anomaly_ds[:len(test_anomaly_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_anomaly_ds[len(test_anomaly_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        if args.spatial_dims_val_test == 2:
            test_masks_transforms = transforms.Compose(
                [
                    transforms.LoadImage(),
                    transforms.EnsureChannelFirst(),
                    custom_transforms.Get2DSlice(axis=2, offset=+2),
                    transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size)),
                    custom_transforms.SetBackgroundToZero()
                ]
            )
        elif args.spatial_dims_val_test == 3:
            test_masks_transforms = transforms.Compose(
                [
                    transforms.LoadImage(),
                    transforms.EnsureChannelFirst(),
                    transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                    custom_transforms.SetBackgroundToZero()
                ]
            )
        test_masks_ds = CacheDataset(data=test_masks, transform=test_masks_transforms)
        
        test_masks_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_masks_ds[:len(test_masks_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_masks_ds[len(test_masks_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    elif args.dataset["test"] == "isles":
        large_group = ['sub-strokecase0023_ses-0001_msk.nii.gz', 'sub-strokecase0031_ses-0001_msk.nii.gz', 'sub-strokecase0047_ses-0001_msk.nii.gz', 'sub-strokecase0048_ses-0001_msk.nii.gz', 'sub-strokecase0062_ses-0001_msk.nii.gz', 'sub-strokecase0066_ses-0001_msk.nii.gz', 'sub-strokecase0081_ses-0001_msk.nii.gz', 'sub-strokecase0083_ses-0001_msk.nii.gz', 'sub-strokecase0087_ses-0001_msk.nii.gz', 'sub-strokecase0091_ses-0001_msk.nii.gz', 'sub-strokecase0123_ses-0001_msk.nii.gz', 'sub-strokecase0161_ses-0001_msk.nii.gz', 'sub-strokecase0162_ses-0001_msk.nii.gz', 'sub-strokecase0171_ses-0001_msk.nii.gz', 'sub-strokecase0176_ses-0001_msk.nii.gz', 'sub-strokecase0201_ses-0001_msk.nii.gz', 'sub-strokecase0211_ses-0001_msk.nii.gz', 'sub-strokecase0222_ses-0001_msk.nii.gz', 'sub-strokecase0223_ses-0001_msk.nii.gz', 'sub-strokecase0023_ses-0001_msk.nii.gz', 'sub-strokecase0031_ses-0001_msk.nii.gz', 'sub-strokecase0047_ses-0001_msk.nii.gz', 'sub-strokecase0048_ses-0001_msk.nii.gz', 'sub-strokecase0062_ses-0001_msk.nii.gz', 'sub-strokecase0066_ses-0001_msk.nii.gz', 'sub-strokecase0081_ses-0001_msk.nii.gz', 'sub-strokecase0083_ses-0001_msk.nii.gz', 'sub-strokecase0087_ses-0001_msk.nii.gz', 'sub-strokecase0091_ses-0001_msk.nii.gz', 'sub-strokecase0123_ses-0001_msk.nii.gz', 'sub-strokecase0161_ses-0001_msk.nii.gz', 'sub-strokecase0162_ses-0001_msk.nii.gz', 'sub-strokecase0171_ses-0001_msk.nii.gz', 'sub-strokecase0176_ses-0001_msk.nii.gz', 'sub-strokecase0201_ses-0001_msk.nii.gz', 'sub-strokecase0211_ses-0001_msk.nii.gz', 'sub-strokecase0222_ses-0001_msk.nii.gz', 'sub-strokecase0223_ses-0001_msk.nii.gz', 'sub-strokecase0230_ses-0001_msk.nii.gz', 'sub-strokecase0237_ses-0001_msk.nii.gz', 'sub-strokecase0240_ses-0001_msk.nii.gz', 'sub-strokecase0246_ses-0001_msk.nii.gz']
        large_group_adc_images = [ROOT_DIR+"datasets/final_adc_dataset_small/ISLES_registered/"+filename.replace("msk", "adc") for filename in large_group]
        large_group_flair_images = [ROOT_DIR+"datasets/final_flair_dataset_small/isles_registered/"+filename.replace("msk", "FLAIR") for filename in large_group]
        large_group_flair_images = [path for path in large_group_flair_images if "0222_ses-0001" not in path]
        large_group_masks = [ROOT_DIR+"datasets/final_adc_dataset_small/ISLES_masks_registered/"+filename for filename in large_group]

        medium_group = ['sub-strokecase0001_ses-0001_msk.nii.gz', 'sub-strokecase0003_ses-0001_msk.nii.gz', 'sub-strokecase0011_ses-0001_msk.nii.gz', 'sub-strokecase0013_ses-0001_msk.nii.gz', 'sub-strokecase0015_ses-0001_msk.nii.gz', 'sub-strokecase0021_ses-0001_msk.nii.gz', 'sub-strokecase0027_ses-0001_msk.nii.gz', 'sub-strokecase0033_ses-0001_msk.nii.gz', 'sub-strokecase0039_ses-0001_msk.nii.gz', 'sub-strokecase0043_ses-0001_msk.nii.gz', 'sub-strokecase0052_ses-0001_msk.nii.gz', 'sub-strokecase0057_ses-0001_msk.nii.gz', 'sub-strokecase0065_ses-0001_msk.nii.gz', 'sub-strokecase0085_ses-0001_msk.nii.gz', 'sub-strokecase0092_ses-0001_msk.nii.gz', 'sub-strokecase0101_ses-0001_msk.nii.gz', 'sub-strokecase0102_ses-0001_msk.nii.gz', 'sub-strokecase0114_ses-0001_msk.nii.gz', 'sub-strokecase0116_ses-0001_msk.nii.gz', 'sub-strokecase0120_ses-0001_msk.nii.gz', 'sub-strokecase0122_ses-0001_msk.nii.gz', 'sub-strokecase0124_ses-0001_msk.nii.gz', 'sub-strokecase0127_ses-0001_msk.nii.gz', 'sub-strokecase0140_ses-0001_msk.nii.gz', 'sub-strokecase0146_ses-0001_msk.nii.gz', 'sub-strokecase0153_ses-0001_msk.nii.gz', 'sub-strokecase0154_ses-0001_msk.nii.gz', 'sub-strokecase0155_ses-0001_msk.nii.gz', 'sub-strokecase0164_ses-0001_msk.nii.gz', 'sub-strokecase0165_ses-0001_msk.nii.gz', 'sub-strokecase0166_ses-0001_msk.nii.gz', 'sub-strokecase0168_ses-0001_msk.nii.gz', 'sub-strokecase0178_ses-0001_msk.nii.gz', 'sub-strokecase0179_ses-0001_msk.nii.gz', 'sub-strokecase0180_ses-0001_msk.nii.gz', 'sub-strokecase0186_ses-0001_msk.nii.gz', 'sub-strokecase0188_ses-0001_msk.nii.gz', 'sub-strokecase0189_ses-0001_msk.nii.gz', 'sub-strokecase0190_ses-0001_msk.nii.gz', 'sub-strokecase0191_ses-0001_msk.nii.gz', 'sub-strokecase0192_ses-0001_msk.nii.gz', 'sub-strokecase0194_ses-0001_msk.nii.gz', 'sub-strokecase0195_ses-0001_msk.nii.gz', 'sub-strokecase0199_ses-0001_msk.nii.gz', 'sub-strokecase0204_ses-0001_msk.nii.gz', 'sub-strokecase0206_ses-0001_msk.nii.gz', 'sub-strokecase0207_ses-0001_msk.nii.gz', 'sub-strokecase0208_ses-0001_msk.nii.gz', 'sub-strokecase0209_ses-0001_msk.nii.gz', 'sub-strokecase0215_ses-0001_msk.nii.gz', 'sub-strokecase0219_ses-0001_msk.nii.gz', 'sub-strokecase0220_ses-0001_msk.nii.gz', 'sub-strokecase0001_ses-0001_msk.nii.gz', 'sub-strokecase0003_ses-0001_msk.nii.gz', 'sub-strokecase0011_ses-0001_msk.nii.gz', 'sub-strokecase0013_ses-0001_msk.nii.gz', 'sub-strokecase0015_ses-0001_msk.nii.gz', 'sub-strokecase0021_ses-0001_msk.nii.gz', 'sub-strokecase0027_ses-0001_msk.nii.gz', 'sub-strokecase0033_ses-0001_msk.nii.gz', 'sub-strokecase0039_ses-0001_msk.nii.gz', 'sub-strokecase0043_ses-0001_msk.nii.gz', 'sub-strokecase0052_ses-0001_msk.nii.gz', 'sub-strokecase0057_ses-0001_msk.nii.gz', 'sub-strokecase0065_ses-0001_msk.nii.gz', 'sub-strokecase0085_ses-0001_msk.nii.gz', 'sub-strokecase0092_ses-0001_msk.nii.gz', 'sub-strokecase0101_ses-0001_msk.nii.gz', 'sub-strokecase0102_ses-0001_msk.nii.gz', 'sub-strokecase0114_ses-0001_msk.nii.gz', 'sub-strokecase0116_ses-0001_msk.nii.gz', 'sub-strokecase0120_ses-0001_msk.nii.gz', 'sub-strokecase0122_ses-0001_msk.nii.gz', 'sub-strokecase0124_ses-0001_msk.nii.gz', 'sub-strokecase0127_ses-0001_msk.nii.gz', 'sub-strokecase0140_ses-0001_msk.nii.gz', 'sub-strokecase0146_ses-0001_msk.nii.gz', 'sub-strokecase0153_ses-0001_msk.nii.gz', 'sub-strokecase0154_ses-0001_msk.nii.gz', 'sub-strokecase0155_ses-0001_msk.nii.gz', 'sub-strokecase0164_ses-0001_msk.nii.gz', 'sub-strokecase0165_ses-0001_msk.nii.gz', 'sub-strokecase0166_ses-0001_msk.nii.gz', 'sub-strokecase0168_ses-0001_msk.nii.gz', 'sub-strokecase0178_ses-0001_msk.nii.gz', 'sub-strokecase0179_ses-0001_msk.nii.gz', 'sub-strokecase0180_ses-0001_msk.nii.gz', 'sub-strokecase0186_ses-0001_msk.nii.gz', 'sub-strokecase0188_ses-0001_msk.nii.gz', 'sub-strokecase0189_ses-0001_msk.nii.gz', 'sub-strokecase0190_ses-0001_msk.nii.gz', 'sub-strokecase0191_ses-0001_msk.nii.gz', 'sub-strokecase0192_ses-0001_msk.nii.gz', 'sub-strokecase0194_ses-0001_msk.nii.gz', 'sub-strokecase0195_ses-0001_msk.nii.gz', 'sub-strokecase0199_ses-0001_msk.nii.gz', 'sub-strokecase0204_ses-0001_msk.nii.gz', 'sub-strokecase0206_ses-0001_msk.nii.gz', 'sub-strokecase0207_ses-0001_msk.nii.gz', 'sub-strokecase0208_ses-0001_msk.nii.gz', 'sub-strokecase0209_ses-0001_msk.nii.gz', 'sub-strokecase0215_ses-0001_msk.nii.gz', 'sub-strokecase0219_ses-0001_msk.nii.gz', 'sub-strokecase0220_ses-0001_msk.nii.gz', 'sub-strokecase0226_ses-0001_msk.nii.gz', 'sub-strokecase0227_ses-0001_msk.nii.gz', 'sub-strokecase0236_ses-0001_msk.nii.gz', 'sub-strokecase0238_ses-0001_msk.nii.gz', 'sub-strokecase0243_ses-0001_msk.nii.gz', 'sub-strokecase0245_ses-0001_msk.nii.gz', 'sub-strokecase0248_ses-0001_msk.nii.gz']
        medium_group_adc_images = [ROOT_DIR+"datasets/final_adc_dataset_small/ISLES_registered/"+filename.replace("msk", "adc") for filename in medium_group]
        medium_group_flair_images = [ROOT_DIR+"datasets/final_flair_dataset_small/isles_registered/"+filename.replace("msk", "FLAIR") for filename in medium_group]
        medium_group_masks = [ROOT_DIR+"datasets/final_adc_dataset_small/ISLES_masks_registered/"+filename for filename in medium_group]

        small_group = ['sub-strokecase0004_ses-0001_msk.nii.gz', 'sub-strokecase0009_ses-0001_msk.nii.gz', 'sub-strokecase0010_ses-0001_msk.nii.gz', 'sub-strokecase0017_ses-0001_msk.nii.gz', 'sub-strokecase0022_ses-0001_msk.nii.gz', 'sub-strokecase0024_ses-0001_msk.nii.gz', 'sub-strokecase0026_ses-0001_msk.nii.gz', 'sub-strokecase0036_ses-0001_msk.nii.gz', 'sub-strokecase0038_ses-0001_msk.nii.gz', 'sub-strokecase0040_ses-0001_msk.nii.gz', 'sub-strokecase0041_ses-0001_msk.nii.gz', 'sub-strokecase0049_ses-0001_msk.nii.gz', 'sub-strokecase0053_ses-0001_msk.nii.gz', 'sub-strokecase0054_ses-0001_msk.nii.gz', 'sub-strokecase0056_ses-0001_msk.nii.gz', 'sub-strokecase0064_ses-0001_msk.nii.gz', 'sub-strokecase0067_ses-0001_msk.nii.gz', 'sub-strokecase0074_ses-0001_msk.nii.gz', 'sub-strokecase0076_ses-0001_msk.nii.gz', 'sub-strokecase0080_ses-0001_msk.nii.gz', 'sub-strokecase0082_ses-0001_msk.nii.gz', 'sub-strokecase0084_ses-0001_msk.nii.gz', 'sub-strokecase0090_ses-0001_msk.nii.gz', 'sub-strokecase0095_ses-0001_msk.nii.gz', 'sub-strokecase0097_ses-0001_msk.nii.gz', 'sub-strokecase0108_ses-0001_msk.nii.gz', 'sub-strokecase0110_ses-0001_msk.nii.gz', 'sub-strokecase0129_ses-0001_msk.nii.gz', 'sub-strokecase0137_ses-0001_msk.nii.gz', 'sub-strokecase0145_ses-0001_msk.nii.gz', 'sub-strokecase0152_ses-0001_msk.nii.gz', 'sub-strokecase0158_ses-0001_msk.nii.gz', 'sub-strokecase0159_ses-0001_msk.nii.gz', 'sub-strokecase0163_ses-0001_msk.nii.gz', 'sub-strokecase0167_ses-0001_msk.nii.gz', 'sub-strokecase0169_ses-0001_msk.nii.gz', 'sub-strokecase0182_ses-0001_msk.nii.gz', 'sub-strokecase0183_ses-0001_msk.nii.gz', 'sub-strokecase0185_ses-0001_msk.nii.gz', 'sub-strokecase0187_ses-0001_msk.nii.gz', 'sub-strokecase0193_ses-0001_msk.nii.gz', 'sub-strokecase0196_ses-0001_msk.nii.gz', 'sub-strokecase0197_ses-0001_msk.nii.gz', 'sub-strokecase0200_ses-0001_msk.nii.gz', 'sub-strokecase0210_ses-0001_msk.nii.gz', 'sub-strokecase0214_ses-0001_msk.nii.gz', 'sub-strokecase0218_ses-0001_msk.nii.gz', 'sub-strokecase0004_ses-0001_msk.nii.gz', 'sub-strokecase0009_ses-0001_msk.nii.gz', 'sub-strokecase0010_ses-0001_msk.nii.gz', 'sub-strokecase0017_ses-0001_msk.nii.gz', 'sub-strokecase0022_ses-0001_msk.nii.gz', 'sub-strokecase0024_ses-0001_msk.nii.gz', 'sub-strokecase0026_ses-0001_msk.nii.gz', 'sub-strokecase0036_ses-0001_msk.nii.gz', 'sub-strokecase0038_ses-0001_msk.nii.gz', 'sub-strokecase0040_ses-0001_msk.nii.gz', 'sub-strokecase0041_ses-0001_msk.nii.gz', 'sub-strokecase0049_ses-0001_msk.nii.gz', 'sub-strokecase0053_ses-0001_msk.nii.gz', 'sub-strokecase0054_ses-0001_msk.nii.gz', 'sub-strokecase0056_ses-0001_msk.nii.gz', 'sub-strokecase0064_ses-0001_msk.nii.gz', 'sub-strokecase0067_ses-0001_msk.nii.gz', 'sub-strokecase0074_ses-0001_msk.nii.gz', 'sub-strokecase0076_ses-0001_msk.nii.gz', 'sub-strokecase0080_ses-0001_msk.nii.gz', 'sub-strokecase0082_ses-0001_msk.nii.gz', 'sub-strokecase0084_ses-0001_msk.nii.gz', 'sub-strokecase0090_ses-0001_msk.nii.gz', 'sub-strokecase0095_ses-0001_msk.nii.gz', 'sub-strokecase0097_ses-0001_msk.nii.gz', 'sub-strokecase0108_ses-0001_msk.nii.gz', 'sub-strokecase0110_ses-0001_msk.nii.gz', 'sub-strokecase0129_ses-0001_msk.nii.gz', 'sub-strokecase0137_ses-0001_msk.nii.gz', 'sub-strokecase0145_ses-0001_msk.nii.gz', 'sub-strokecase0152_ses-0001_msk.nii.gz', 'sub-strokecase0158_ses-0001_msk.nii.gz', 'sub-strokecase0159_ses-0001_msk.nii.gz', 'sub-strokecase0163_ses-0001_msk.nii.gz', 'sub-strokecase0167_ses-0001_msk.nii.gz', 'sub-strokecase0169_ses-0001_msk.nii.gz', 'sub-strokecase0182_ses-0001_msk.nii.gz', 'sub-strokecase0183_ses-0001_msk.nii.gz', 'sub-strokecase0185_ses-0001_msk.nii.gz', 'sub-strokecase0187_ses-0001_msk.nii.gz', 'sub-strokecase0193_ses-0001_msk.nii.gz', 'sub-strokecase0196_ses-0001_msk.nii.gz', 'sub-strokecase0197_ses-0001_msk.nii.gz', 'sub-strokecase0200_ses-0001_msk.nii.gz', 'sub-strokecase0210_ses-0001_msk.nii.gz', 'sub-strokecase0214_ses-0001_msk.nii.gz', 'sub-strokecase0218_ses-0001_msk.nii.gz', 'sub-strokecase0225_ses-0001_msk.nii.gz', 'sub-strokecase0229_ses-0001_msk.nii.gz', 'sub-strokecase0232_ses-0001_msk.nii.gz', 'sub-strokecase0235_ses-0001_msk.nii.gz', 'sub-strokecase0244_ses-0001_msk.nii.gz', 'sub-strokecase0247_ses-0001_msk.nii.gz', 'sub-strokecase0249_ses-0001_msk.nii.gz']
        small_group_adc_images = [ROOT_DIR+"datasets/final_adc_dataset_small/ISLES_registered/"+filename.replace("msk", "adc") for filename in small_group]
        small_group_flair_images = [ROOT_DIR+"datasets/final_flair_dataset_small/isles_registered/"+filename.replace("msk", "FLAIR") for filename in small_group]
        small_group_masks = [ROOT_DIR+"datasets/final_adc_dataset_small/ISLES_masks_registered/"+filename for filename in small_group]

        num_workers = 4
        ano_batch_size = 48

        test_anomaly_transforms = define_instance(args, "val_transforms")
        
        if "flair" in args.dataset["name"].lower():
            large_group_masks = [path for path in large_group_masks if "0222_ses-0001" not in path]
            test_anomaly_large_ds = CacheDataset(data=large_group_flair_images, transform=test_anomaly_transforms)
            test_anomaly_medium_ds = CacheDataset(data=medium_group_flair_images, transform=test_anomaly_transforms)
            test_anomaly_small_ds = CacheDataset(data=small_group_flair_images, transform=test_anomaly_transforms)
        elif "adc" in args.dataset["name"].lower():
            test_anomaly_large_ds = CacheDataset(data=large_group_adc_images, transform=test_anomaly_transforms)
            test_anomaly_medium_ds = CacheDataset(data=medium_group_adc_images, transform=test_anomaly_transforms)
            test_anomaly_small_ds = CacheDataset(data=small_group_adc_images, transform=test_anomaly_transforms)


        test_anomaly_large_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_anomaly_large_ds[:len(test_anomaly_large_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_large_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_anomaly_large_ds[len(test_anomaly_large_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        test_anomaly_medium_loader_select_params = DataLoader(
            test_anomaly_medium_ds[:len(test_anomaly_medium_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_medium_loader_metrics = DataLoader(
            test_anomaly_medium_ds[len(test_anomaly_medium_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        test_anomaly_small_loader_select_params = DataLoader(
            test_anomaly_small_ds[:len(test_anomaly_small_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_small_loader_metrics = DataLoader(
            test_anomaly_small_ds[len(test_anomaly_small_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )



        if args.spatial_dims_val_test == 2:
            test_masks_transforms = transforms.Compose(
                [
                    transforms.LoadImage(),
                    transforms.EnsureChannelFirst(),
                    custom_transforms.Get2DSlice(axis=2, offset=+2),
                    transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size)),
                    custom_transforms.SetBackgroundToZero()
                ]
            )
        elif args.spatial_dims_val_test == 3:
            test_masks_transforms = transforms.Compose(
                [
                    transforms.LoadImage(),
                    transforms.EnsureChannelFirst(),
                    transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                    custom_transforms.SetBackgroundToZero()
                ]
            )
        test_masks_large_ds = CacheDataset(data=large_group_masks, transform=test_masks_transforms)
        test_masks_medium_ds = CacheDataset(data=medium_group_masks, transform=test_masks_transforms)
        test_masks_small_ds = CacheDataset(data=small_group_masks, transform=test_masks_transforms)


        test_masks_large_loader_select_params = DataLoader(
            test_masks_large_ds[:len(test_masks_large_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_large_loader_metrics = DataLoader(
            test_masks_large_ds[len(test_masks_large_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        test_masks_medium_loader_select_params = DataLoader(
            test_masks_medium_ds[:len(test_masks_medium_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_medium_loader_metrics = DataLoader(
            test_masks_medium_ds[len(test_masks_medium_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        test_masks_small_loader_select_params = DataLoader(
            test_masks_small_ds[:len(test_masks_small_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_small_loader_metrics = DataLoader(
            test_masks_small_ds[len(test_masks_small_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    elif args.dataset["test"] == "soop":
        if "flair" in args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        elif "adc" in args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        basic_affine = nib.load(test_anomaly_images[0]).affine

        images_to_exclude = []
        with open(ROOT_DIR+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(ROOT_DIR+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        

        test_anomaly_images = [path for path in test_anomaly_images if os.path.basename(path).split('_')[0] not in images_to_exclude]
        
        
        num_workers = 4
        ano_batch_size = 64

        test_anomaly_transforms = define_instance(args, "val_transforms")
        test_anomaly_ds = CacheDataset(data=test_anomaly_images, transform=test_anomaly_transforms)

        test_anomaly_loader_metrics = DataLoader(
            test_anomaly_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    model = define_instance(args, "network_def").to(device)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE_TYPE))
    model.eval()


    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])


    timesteps_harmonization = np.linspace(10, NOISE_MAX-1, num=7, dtype=int).tolist()

    @torch.no_grad()
    def sample_thor(image, infer_scheduler, timesteps=100, return_intermediates=False):
        
        if timesteps >= infer_scheduler.num_train_timesteps:
            print(timesteps, "is too high. Setting to", infer_scheduler.num_train_timesteps-1)
        
        timesteps_list = torch.Tensor([timesteps for a in range(image.shape[0])]).to(image.device).long()

        simplexObj = simplex.Simplex_CLASS()

        original_image = copy.deepcopy(image)

        if args.noise["type"] == "simplex":
            noise = simplex_ddpm.generate_simplex_noise(simplexObj, image.shape, normalize=args.noise["normalize"]).to(device)
        if args.noise["type"] == "gaussian":
            noise = torch.randn(image.shape).to(device)
        

        image = infer_scheduler.add_noise(image, noise, timesteps_list).to(device)


        intermediates_mixed_images_visualize = []
        intermediates_pseudo_anomaly_masks = []
        intermediates_pseudo_anomaly_masks_processed = []

                
        for t in range(timesteps, 0, -1): # goes from timesteps to 0
            
            # compute previous image
            model_output = model(image, timesteps=torch.Tensor((t,)).to(device), context=None)
            image, image_before_step = infer_scheduler.step(model_output, t, image) # here image_before_step is just the image at the timestep+1
                
            
            if t in timesteps_harmonization:
                
                
                pseudo_anomaly_mask, _, _ = thor_ddpm.get_anomaly_mask(copy.deepcopy(image_before_step), copy.deepcopy(original_image), device=device, hist_eq=False)
                
                intermediates_pseudo_anomaly_masks.append(pseudo_anomaly_mask)
                pseudo_anomaly_mask = pseudo_anomaly_mask.cpu().detach().numpy()
                

                pseudo_anomaly_mask_processed = torch.Tensor(thor_ddpm.get_region_anomaly_mask(pseudo_anomaly_mask, kernel_size=4)).to(device).clip(0,1) # simple erosion dilation 
                

                pseudo_anomaly_mask_processed = pseudo_anomaly_mask_processed.clip(0,1) 

                intermediates_pseudo_anomaly_masks_processed.append(pseudo_anomaly_mask_processed)

                image_0 = pseudo_anomaly_mask_processed * image_before_step + (1-pseudo_anomaly_mask_processed) * original_image

                image_0 = torch.clamp(image_0, 0, 1)
                
                image_0 = scale_intensity_from_histogram_peak(image_0, 2.0/7.0) #TODO

                image_0 = torch.clamp(image_0, 0, 1)

                image = infer_scheduler.add_noise(image_0, noise, torch.Tensor((t,)).to(device).long())
                
                intermediates_mixed_images_visualize.append(image)

        if return_intermediates:
            return image, intermediates_mixed_images_visualize, intermediates_pseudo_anomaly_masks, intermediates_pseudo_anomaly_masks_processed
        else:
            return image, intermediates_pseudo_anomaly_masks_processed


    dm = DiceMetric(reduction="sum")

    def compute_select_params(image_loader, mask_loader):


        num_timesteps_to_try = np.arange(NOISE_MIN, NOISE_MAX, NOISE_INTERVAL)
        thresholds_to_try = np.arange(0.0, 0.4, 0.01) # from 0.0 to 0.4 with step 0.01

        iou_scores_df = pd.DataFrame(index=num_timesteps_to_try, columns=thresholds_to_try)
        iou_scores_df.fillna(0.0, inplace=True)

        dice_scores_df = pd.DataFrame(index=num_timesteps_to_try, columns=thresholds_to_try)
        dice_scores_df.fillna(0.0, inplace=True)


        for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice

            test_images = image_batch.to(device)
            test_masks = mask_batch.to(device)
            test_masks[test_masks>0.5] = 1.0
            test_masks[test_masks<=0.5] = 0.0

            for infer_timesteps in num_timesteps_to_try:
                with autocast(device_type=DEVICE_TYPE, enabled=True):
                    if args.spatial_dims_val_test == 2: # single 2D slice segmentation
                        _, pseudo_anomaly_masks_processed = sample_thor(test_images, infer_scheduler=infer_scheduler, timesteps=infer_timesteps, return_intermediates=False)
                        final_anomaly_map = stats.hmean(np.stack([p.cpu() for p in pseudo_anomaly_masks_processed]), axis=0)
                    
                    elif args.spatial_dims_val_test == 3: # full slice by slice 3D volume segmentation
                        infered_slices = []
                        for slice_idx in range(args.slice_indexes_start, args.slice_indexes_end):
                            _, pseudo_anomaly_masks_processed = sample_thor(test_images[...,slice_idx], infer_scheduler=infer_scheduler, timesteps=infer_timesteps, return_intermediates=False)
                            infered_slice = stats.hmean(np.stack([p.cpu() for p in pseudo_anomaly_masks_processed]), axis=0)
                            infered_slices.append(torch.Tensor(infered_slice).unsqueeze(-1))
                        final_anomaly_map = torch.cat(infered_slices, dim=-1).to(device)

                for threshold in thresholds_to_try:
                    if args.spatial_dims_val_test==2:
                        ano_segmentation = np.abs(final_anomaly_map) > threshold
                        
                        # IOU score
                        iou_score = compute_iou(torch.Tensor(ano_segmentation).to(device), test_masks) #TODO here 19 sept 2025 16:30
                        flattened_iou_score = iou_score.cpu().numpy().flatten()
                        flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values

                        if np.isnan(iou_scores_df.loc[infer_timesteps, threshold]): # if the cell is empty
                            iou_scores_df.loc[infer_timesteps, threshold] = np.sum(flattened_iou_score)
                        else:
                            iou_scores_df.loc[infer_timesteps, threshold] += np.sum(flattened_iou_score) # this average is false
                        
                        # DICE score
                        dice_score = dm(torch.Tensor(ano_segmentation).to(device), test_masks).cpu().numpy().flatten()
                        dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values

                        if np.isnan(dice_scores_df.loc[infer_timesteps, threshold]): # if the cell is empty
                            dice_scores_df.loc[infer_timesteps, threshold] = np.sum(dice_score)
                        else:
                            dice_scores_df.loc[infer_timesteps, threshold] += np.sum(dice_score)

                    elif args.spatial_dims_val_test == 3:
                        ano_segmentation = torch.abs(final_anomaly_map) > threshold

                        # IOU score
                        iou_score = compute_iou(ano_segmentation, test_masks[...,args.slice_indexes_start:args.slice_indexes_end])
                        flattened_iou_score = iou_score.cpu().numpy().flatten()
                        flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values
                        
                        if np.isnan(iou_scores_df.loc[infer_timesteps, threshold]): # if the cell is empty
                            iou_scores_df.loc[infer_timesteps, threshold] = np.sum(flattened_iou_score)
                        else:
                            iou_scores_df.loc[infer_timesteps, threshold] += np.sum(flattened_iou_score) 

                        # DICE score
                        dice_score = dm(ano_segmentation, test_masks[...,args.slice_indexes_start:args.slice_indexes_end]).cpu().numpy().flatten()
                        dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values

                        if np.isnan(dice_scores_df.loc[infer_timesteps, threshold]): # if the cell is empty
                            dice_scores_df.loc[infer_timesteps, threshold] = np.sum(dice_score)
                        else:
                            dice_scores_df.loc[infer_timesteps, threshold] += np.sum(dice_score)

        #divide everything by the number of images
        iou_scores_df = iou_scores_df / len(image_loader.dataset)
        dice_scores_df = dice_scores_df / len(image_loader.dataset)

        return iou_scores_df, dice_scores_df

    def compute_metrics(image_loader, mask_loader, timesteps, threshold):

        iou_scores = []
        dice_scores = []

        no_masks = False
        if mask_loader is None:
            mask_loader = image_loader # hack so the for loop works
            no_masks = True

        for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice

            test_images = image_batch.to(device)
            test_masks = mask_batch.to(device)
            test_masks[test_masks>0.5] = 1.0
            test_masks[test_masks<=0.5] = 0.0

            with autocast(device_type=DEVICE_TYPE, enabled=True):
                if args.spatial_dims_val_test == 2: # single 2D slice segmentation
                    _, pseudo_anomaly_masks_processed = sample_thor(test_images, infer_scheduler=infer_scheduler, timesteps=timesteps, return_intermediates=False)
                    final_anomaly_map = stats.hmean(np.stack([p.cpu() for p in pseudo_anomaly_masks_processed]), axis=0)
                    if args.dataset["save_anomaly_maps"]:
                        for idx_in_batch in range(final_anomaly_map.shape[0]):
                            #save as png
                            image_id= i*image_batch.shape[0] + idx_in_batch
                            image_name = os.path.basename(test_anomaly_images[image_id])
                            plt.imsave(ANOMALY_MAPS_DIR+f"ano_map_thor_{image_name.split('.')[0]}.png", final_anomaly_map[idx_in_batch].cpu().numpy(), vmin=0, vmax=1, cmap='jet')


                elif args.spatial_dims_val_test == 3: # full slice by slice 3D volume segmentation
                    infered_slices = []
                    for slice_idx in range(args.slice_indexes_start, args.slice_indexes_end):
                        _, pseudo_anomaly_masks_processed = sample_thor(test_images[...,slice_idx], infer_scheduler=infer_scheduler, timesteps=timesteps, return_intermediates=False)
                        infered_slice = stats.hmean(np.stack([p.cpu() for p in pseudo_anomaly_masks_processed]), axis=0)
                        infered_slices.append(torch.Tensor(infered_slice).unsqueeze(-1))

                    stacked_anomaly_maps = torch.cat(infered_slices, dim=-1).to(device)
                    final_anomaly_map = torch.zeros_like(test_images)
                    final_anomaly_map[...,args.slice_indexes_start:args.slice_indexes_end] = stacked_anomaly_maps
                    
                    if args.dataset["save_anomaly_maps"]:
                        for idx_in_batch in range(final_anomaly_map.shape[0]):
                            image_id= i*image_batch.shape[0] + idx_in_batch
                            image_name = os.path.basename(test_anomaly_images[image_id])
                            nib.save(nib.Nifti1Image(final_anomaly_map[idx_in_batch].squeeze().cpu().numpy(), basic_affine), ANOMALY_MAPS_DIR+f"ano_map_thor_{image_name}")


            if args.spatial_dims_val_test == 2 and not no_masks:
                ano_segmentation = torch.abs(final_anomaly_map) > threshold

                iou_score = compute_iou(torch.Tensor(ano_segmentation).to(device), test_masks)
                flattened_iou_score = iou_score.cpu().numpy().flatten()
                flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values

                iou_scores.append(flattened_iou_score)

                dice_score = dm(torch.Tensor(ano_segmentation).to(device), test_masks).cpu().numpy().flatten()
                dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values
                dice_scores.append(dice_score)
            elif args.spatial_dims_val_test == 3 and not no_masks:
                ano_segmentation = torch.abs(final_anomaly_map) > threshold

                iou_score = compute_iou(torch.Tensor(ano_segmentation).to(device), test_masks)
                flattened_iou_score = iou_score.cpu().numpy().flatten()
                flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values

                iou_scores.append(flattened_iou_score)

                dice_score = dm(torch.Tensor(ano_segmentation).to(device), test_masks).cpu().numpy().flatten()
                dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values
                dice_scores.append(dice_score)

        if no_masks:
            return

        mean_iou = np.mean(np.concatenate(iou_scores))
        std_iou = np.std(np.concatenate(iou_scores))

        mean_dice = np.mean(np.concatenate(dice_scores))
        std_dice = np.std(np.concatenate(dice_scores))

        return mean_iou, std_iou, mean_dice, std_dice
    
    # ----------- COMPUTING METRICS -----------

    if args.spatial_dims_val_test == 2:
        metrics_result_text = "Thor Anomaly Detection Segmentation scores (single middle 2D slice)\n"
    elif args.spatial_dims_val_test == 3:
        metrics_result_text = "Thor Anomaly Detection Segmentation scores (full 3D volume slice by slice)\n"

    if args.dataset["test"] == "brats":
        iou_scores_df, dice_scores_df = compute_select_params(test_anomaly_loader_select_params, test_masks_loader_select_params)

        best_threshold = iou_scores_df.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df.max(axis=1).idxmax()

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_loader_metrics, test_masks_loader_metrics, timesteps=best_num_timesteps, threshold=best_threshold)

        
        metrics_result_text += f"mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Best Threshold: {best_threshold:.4f}\n"

        
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps}"
        print(metrics_result_text)
    
    elif args.dataset["test"] == "isles":
        # large group
        iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params(test_anomaly_large_loader_select_params, test_masks_large_loader_select_params)

        best_threshold = iou_scores_df_large_group.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df_large_group.max(axis=1).idxmax()

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_large_loader_metrics, test_masks_large_loader_metrics, timesteps=best_num_timesteps, threshold=best_threshold)

        
        metrics_result_text += f"THOR Large group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Large group: best threshold: {best_threshold:.4f}\n"

        
        metrics_result_text += f"Large group: best number of Timesteps: {best_num_timesteps}\n"
        metrics_result_text += "\n" 
        print(metrics_result_text)
        

        # medium group
        iou_scores_df_medium_group, dice_scores_df_medium_group = compute_select_params(test_anomaly_medium_loader_select_params, test_masks_medium_loader_select_params)

        best_threshold = iou_scores_df_medium_group.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df_medium_group.max(axis=1).idxmax()

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_medium_loader_metrics, test_masks_medium_loader_metrics, timesteps=best_num_timesteps, threshold=best_threshold)

        
        metrics_result_text += f"THOR Medium group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Medium group: best threshold: {best_threshold:.4f}\n"

        
        metrics_result_text += f"Medium group: best number of Timesteps: {best_num_timesteps}\n"
        metrics_result_text += "\n"
        print(metrics_result_text)


        # small group
        iou_scores_df_small_group, dice_scores_df_small_group = compute_select_params(test_anomaly_small_loader_select_params, test_masks_small_loader_select_params)

        best_threshold = iou_scores_df_small_group.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df_small_group.max(axis=1).idxmax()

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_small_loader_metrics, test_masks_small_loader_metrics, timesteps=best_num_timesteps, threshold=best_threshold)

        
        metrics_result_text += f"THOR Small group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Small group: best threshold: {best_threshold:.4f}\n"

        
        metrics_result_text += f"Small group: best number of Timesteps: {best_num_timesteps}"
        print(metrics_result_text)

    elif args.dataset["test"] == "soop":
        if "flair" in args.dataset["name"].lower():
            num_timesteps = 350 
            compute_metrics(test_anomaly_loader_metrics, None, timesteps=num_timesteps, threshold=None)
        elif "adc" in args.dataset["name"].lower():
            num_timesteps = 100
            compute_metrics(test_anomaly_loader_metrics, None, timesteps=num_timesteps, threshold=None)




    # ----------- SUMMARY FIGURE -----------
    
    if args.show_summary_figure:
        #infer_timesteps_visualize = int(args.compute_metrics_reconstruction["noise_rate_visualize"]*args.noise["num_timesteps_full_noise"])
        infer_timesteps_visualize = best_num_timesteps


        if args.dataset["test"] == "brats":
            image_loader = test_anomaly_loader_metrics
            mask_loader = test_masks_loader_metrics
        elif args.dataset["test"] == "isles":
            image_loader = test_anomaly_large_loader_metrics
            mask_loader = test_masks_large_loader_metrics

        for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice
            if i>0:break

            if args.spatial_dims_val_test == 2:
                test_anomaly_images = image_batch.to(device)
                test_anomaly_masks = mask_batch.to(device)
            elif args.spatial_dims_val_test == 3:
                test_anomaly_images = image_batch[..., image_batch.shape[-1]//2].to(device)
                test_anomaly_masks = mask_batch[..., mask_batch.shape[-1]//2].to(device)
            
            test_anomaly_masks[test_anomaly_masks>0.5] = 1.0
            test_anomaly_masks[test_anomaly_masks<=0.5] = 0.0

            with autocast(device_type=DEVICE_TYPE, enabled=True):

                _, pseudo_anomaly_masks_processed = sample_thor(test_anomaly_images, infer_scheduler=infer_scheduler, timesteps=infer_timesteps_visualize, return_intermediates=False)
                final_anomaly_map = stats.hmean(np.stack([p.cpu() for p in pseudo_anomaly_masks_processed]), axis=0)

        # ----------- PLOT -----------
        
        fig, axes = plt.subplots(5, 8, figsize=(20, 17), constrained_layout=True)
        plt.tight_layout()

        for idx in range(min(4, test_anomaly_images.shape[0])):

            # Original test_anomaly images
            original_image = test_anomaly_images[idx, 0].cpu().numpy()
            axes[0, idx*2].imshow(original_image, cmap='gray', vmin=0, vmax=1)
            axes[0, idx*2].set_title(f'Original {idx+1}')
            axes[0, idx*2].axis('off')

            axes[0, idx*2+1].hist(original_image[original_image>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
            axes[0, idx*2+1].set_ylim(0, 2000)
            axes[0, idx*2+1].set_aspect('auto')  # Set the aspect ratio to auto to match the imshow plot
            
            
            # inferred ano map
            #print(average_infered_image.shape)
            axes[1, idx*2].imshow(final_anomaly_map[idx][0], cmap='jet', vmin=0, vmax=1)
            axes[1, idx*2].set_title(f'Inferred {idx+1}')
            axes[1, idx*2].axis('off')

            axes[1, idx*2+1].hist(final_anomaly_map[idx][0][final_anomaly_map[idx][0]>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
            axes[1, idx*2+1].set_ylim(0, 2000)
            axes[1, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

            # Thresholded difference images
            thresholded_difference_image = (final_anomaly_map[idx][0] > best_threshold).astype(np.float32)
            axes[2, idx*2].imshow(thresholded_difference_image, cmap='gray', vmin=0, vmax=1)
            axes[2, idx*2].set_title(f'Thresholded anomaly map {idx+1}')
            axes[2, idx*2].axis('off')

            # ground truth masks
            ground_truth_mask = test_anomaly_masks[idx, 0].cpu().numpy()
            axes[3, idx*2].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=1)
            axes[3, idx*2].set_title(f'Ground Truth {idx+1}')
            axes[3, idx*2].axis('off')

            axes[3, idx*2+1].hist(ground_truth_mask[ground_truth_mask>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
            axes[3, idx*2+1].set_ylim(0, 2000)
            axes[3, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

            axes[0, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
            axes[1, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
            axes[2, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
            axes[3, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot

        
        # Add an empty row to create more whitespace for the figtext
        for idx in range(8):
            axes[4, idx].axis('off')

        # Add overall title with metric results
        if args.spatial_dims_val_test == 2:
            plt.suptitle(f"THOR Anomaly detection for {EXPERIMENT_NAME}, single 2D slice", fontsize=16)
        elif args.spatial_dims_val_test == 3:
            plt.suptitle(f"THOR Anomaly detection for {EXPERIMENT_NAME}, full slice by slice volume inference, large group", fontsize=16)

        plt.figtext(0.0, 0.0, metrics_result_text, fontsize=16)


        if args.spatial_dims_val_test == 2:
            plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_{args.dataset['test']}_metrics_thor_anomaly_detection_single_slice.png", transparent=False, dpi=150)
        if args.spatial_dims_val_test == 3:
            plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_{args.dataset['test']}_metrics_thor_anomaly_detection_full_volume_slice_by_slice.png", transparent=False, dpi=150)

