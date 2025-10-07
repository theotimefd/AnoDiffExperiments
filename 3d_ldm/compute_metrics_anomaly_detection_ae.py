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
from utils.utils import define_instance

from monai.metrics import compute_iou, DiceMetric

import lpips



def launch_compute_metrics_anomaly_detection_ae(args):
    # Two parts : the first 50% of the test data is used to select the best noise timestep value and best threshold.
    # The second 50% is used to compute the final IOU and DICE metrics with these best values.



    DEVICE_TYPE = "cuda:0"
    device = torch.device(DEVICE_TYPE)

    set_determinism(0)

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"

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

    # ----------- DATA -----------

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

        ano_batch_size = args.autoencoder_train["batch_size"]
        num_workers = args.autoencoder_train["num_workers"]

        test_anomaly_transforms = define_instance(args, "val_transforms")
        test_anomaly_ds = CacheDataset(data=test_anomaly_images, transform=test_anomaly_transforms)

        test_anomaly_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_anomaly_ds[:len(test_anomaly_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_anomaly_ds[len(test_anomaly_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )


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

        ano_batch_size = args.autoencoder_train["batch_size"]
        num_workers = args.autoencoder_train["num_workers"]

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

        test_masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
                custom_transforms.SetBackgroundToZero()
            ]
        )
        
        test_masks_large_ds = CacheDataset(data=large_group_masks, transform=test_masks_transforms)
        test_masks_medium_ds = CacheDataset(data=medium_group_masks, transform=test_anomaly_transforms)
        test_masks_small_ds = CacheDataset(data=small_group_masks, transform=test_anomaly_transforms)


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
    
    # Define Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_autoencoder.pt")

    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=device, weights_only=True))
    autoencoder.eval()



    dm = DiceMetric(reduction="sum")

    def compute_select_params(image_loader, mask_loader):
        print("Computing best threshold... (compute_select_params)")

        thresholds_to_try = np.arange(0.0, 0.6, 0.01) # from 0.0 to 0.6 with step 0.05

        iou_scores_dict = {thresh: [] for thresh in thresholds_to_try}

        dice_scores_dict = {thresh: [] for thresh in thresholds_to_try}


        for i,(image_batch, mask_batch) in tqdm(enumerate(zip(image_loader, mask_loader))): 

            test_images = image_batch.to(device)
            test_masks = mask_batch.to(device)
            test_masks[test_masks>0.5] = 1.0
            test_masks[test_masks<=0.5] = 0.0

            with autocast(device_type=DEVICE_TYPE, enabled=True):
                with torch.no_grad():
                    infered, _, _ = autoencoder(test_images)
            
                for threshold in thresholds_to_try:
                
                    ano_segmentation = torch.abs(infered - test_images) > threshold

                    iou_score = compute_iou(ano_segmentation, test_masks)
                    flattened_iou_score = iou_score.cpu().numpy().flatten()
                    flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values

                    iou_scores_dict[threshold].append(np.sum(flattened_iou_score))

                    dice_score = dm(ano_segmentation, test_masks).cpu().numpy().flatten()
                    dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values

                    dice_scores_dict[threshold].append(np.sum(dice_score))

                    

        iou_scores_dict = {thresh: np.mean(iou_scores_dict[thresh]) for thresh in thresholds_to_try}
        dice_scores_dict = {thresh: np.mean(dice_scores_dict[thresh]) for thresh in thresholds_to_try}

        return iou_scores_dict, dice_scores_dict

    def compute_metrics(image_loader, mask_loader, threshold):
        print("Computing final metrics... (compute_metrics)")

        iou_scores = []
        dice_scores = []
       
        for i,(image_batch, mask_batch) in tqdm(enumerate(zip(image_loader, mask_loader))): 

            test_images = image_batch.to(device)
            test_masks = mask_batch.to(device)
            test_masks[test_masks>0.5] = 1.0
            test_masks[test_masks<=0.5] = 0.0


            with autocast(device_type=DEVICE_TYPE, enabled=True):

                    with torch.no_grad():
                        infered, _, _ = autoencoder(test_images)

            
            ano_segmentation = torch.abs(infered - test_images) > threshold

            iou_score = compute_iou(ano_segmentation, test_masks)
            flattened_iou_score = iou_score.cpu().numpy().flatten()
            flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values

            iou_scores.append(flattened_iou_score)

            dice_score = dm(ano_segmentation, test_masks).cpu().numpy().flatten()
            dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values
            dice_scores.append(dice_score)

        mean_iou = np.mean(np.concatenate(iou_scores))
        std_iou = np.std(np.concatenate(iou_scores))

        mean_dice = np.mean(np.concatenate(dice_scores))
        std_dice = np.std(np.concatenate(dice_scores))

        return mean_iou, std_iou, mean_dice, std_dice

    # ----------- COMPUTING METRICS -----------

    metrics_result_text = "Autoencoder 3D segmentation scores\n"

    if args.dataset["test"] == "brats":
        iou_scores_dict, dice_scores_dict = compute_select_params(test_anomaly_loader_select_params, test_masks_loader_select_params)

        best_threshold = max(iou_scores_dict, key=iou_scores_dict.get)
        

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_loader_metrics, test_masks_loader_metrics, threshold=best_threshold)

        
        metrics_result_text += f"mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Best Threshold: {best_threshold:.4f}\n"


        

    elif args.dataset["test"] == "isles":
        # large group
        iou_scores_dict_large_group, dice_scores_dict_large_group = compute_select_params(test_anomaly_large_loader_select_params, test_masks_large_loader_select_params)

        best_threshold = max(iou_scores_dict_large_group, key=iou_scores_dict_large_group.get)

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_large_loader_metrics, test_masks_large_loader_metrics, threshold=best_threshold)

        metrics_result_text += f"Large group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Large group: best threshold: {best_threshold:.4f}\n"

        metrics_result_text += "\n"

        

        # medium group
        iou_scores_dict_medium_group, dice_scores_dict_medium_group = compute_select_params(test_anomaly_medium_loader_select_params, test_masks_medium_loader_select_params)

        best_threshold = max(iou_scores_dict_medium_group, key=iou_scores_dict_medium_group.get)

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_medium_loader_metrics, test_masks_medium_loader_metrics, threshold=best_threshold)

        metrics_result_text += f"Medium group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Medium group: best threshold: {best_threshold:.4f}\n"

        metrics_result_text += "\n"


        # small group
        iou_scores_dict_small_group, dice_scores_dict_small_group = compute_select_params(test_anomaly_small_loader_select_params, test_masks_small_loader_select_params)

        best_threshold = max(iou_scores_dict_small_group, key=iou_scores_dict_small_group.get)

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(test_anomaly_small_loader_metrics, test_masks_small_loader_metrics, threshold=best_threshold)

        metrics_result_text += f"Small group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Small group: best threshold: {best_threshold:.4f}\n"




    # ----------- VISUALIZATION OF A BATCH -----------
    #infer_timesteps_visualize = int(args.compute_metrics_reconstruction["noise_rate_visualize"]*args.noise["num_timesteps_full_noise"])

    if args.dataset["test"] == "brats":
        image_loader = test_anomaly_loader_metrics
        mask_loader = test_masks_loader_metrics
    elif args.dataset["test"] == "isles":
        image_loader = test_anomaly_large_loader_metrics
        mask_loader = test_masks_large_loader_metrics

    for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice
        if i>0:break

        test_anomaly_images = image_batch.to(device)
        test_anomaly_masks = mask_batch.to(device)

        test_anomaly_masks[test_anomaly_masks>0.5] = 1.0
        test_anomaly_masks[test_anomaly_masks<=0.5] = 0.0

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            with torch.no_grad():
                infered, _, _ = autoencoder(test_anomaly_images)

    # ----------- PLOT -----------

    fig, axes = plt.subplots(6, 8, figsize=(25, 17), constrained_layout=True)
    plt.tight_layout()

    for idx in range(min(4, test_anomaly_images.shape[0])):

        # Original test_anomaly images
        original_image = test_anomaly_images[idx, 0,:,:,image_batch.shape[-1]//2].cpu().numpy()
        axes[0, idx*2].imshow(original_image, cmap='gray', vmin=0, vmax=1)
        axes[0, idx*2].set_title(f'Original {idx+1}')
        axes[0, idx*2].axis('off')

        axes[0, idx*2+1].hist(original_image[original_image>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[0, idx*2+1].set_ylim(0, 2000)
        axes[0, idx*2+1].set_aspect('auto')  # Set the aspect ratio to auto to match the imshow plot

        # 3x average inferred images
        print(infered.shape)
        infered_cpu = infered[idx, 0,:,:,image_batch.shape[-1]//2].cpu().numpy()
        axes[1, idx*2].imshow(infered_cpu, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].set_title(f'Inferred {idx+1}')
        axes[1, idx*2].axis('off')

        axes[1, idx*2+1].hist(infered_cpu[infered_cpu>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[1, idx*2+1].set_ylim(0, 2000)
        axes[1, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Difference images
        difference_image = np.abs(original_image - infered_cpu)
        axes[2, idx*2].imshow(difference_image, cmap='jet', vmin=0, vmax=1)
        axes[2, idx*2].set_title(f'Difference {idx+1}')
        axes[2, idx*2].axis('off')

        axes[2, idx*2+1].hist(difference_image[difference_image>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[2, idx*2+1].set_ylim(0, 2000)
        axes[2, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Thresholded difference images
        thresholded_difference_image = (difference_image > best_threshold).astype(np.float32)
        axes[3, idx*2].imshow(thresholded_difference_image, cmap='gray', vmin=0, vmax=1)
        axes[3, idx*2].set_title(f'Thresholded Difference {idx+1}')
        axes[3, idx*2].axis('off')

        # ground truth masks
        ground_truth_mask = test_anomaly_masks[idx, 0,:,:,image_batch.shape[-1]//2].cpu().numpy()
        axes[4, idx*2].imshow(ground_truth_mask, cmap='gray', vmin=0, vmax=1)
        axes[4, idx*2].set_title(f'Ground Truth {idx+1}')
        axes[4, idx*2].axis('off')

        axes[4, idx*2+1].hist(ground_truth_mask[ground_truth_mask>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[4, idx*2+1].set_ylim(0, 2000)
        axes[4, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        axes[0, idx*2+1].set_box_aspect(1) # Set the aspect ratio of the histogram subplot 
        axes[1, idx*2+1].set_box_aspect(1)  
        axes[2, idx*2+1].set_box_aspect(1)  
        axes[3, idx*2+1].set_box_aspect(1) 
        axes[4, idx*2+1].set_box_aspect(1)  

    
    # Add an empty row to create more whitespace for the figtext
    for idx in range(8):
        axes[5, idx].axis('off')
    # Add overall title with metric results
    plt.suptitle(f"Anomaly detection for {EXPERIMENT_NAME}, 3D autoencoder volume inference, large group", fontsize=16)

    plt.figtext(0.0, 0.0, metrics_result_text, fontsize=16)

    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_{args.dataset['test']}_metrics_anomaly_detection_3D_autoencoder.png", transparent=False, dpi=150)
   