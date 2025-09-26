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



def launch_compute_metrics_anomaly_detection(args):
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

        num_workers = 4
        ano_batch_size = 32

        test_anomaly_transforms = define_instance(args, "val_transforms")
        test_anomaly_ds = CacheDataset(data=test_anomaly_images, transform=test_anomaly_transforms)
        test_anomaly_loader = DataLoader(
            test_anomaly_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        test_masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                custom_transforms.Get2DSlice(axis=2, offset=+2),
                transforms.ResizeWithPadOrCrop(spatial_size=(128, 128)),
                custom_transforms.SetBackgroundToZero()
            ]
        )
        test_masks_ds = CacheDataset(data=test_masks, transform=test_masks_transforms)
        test_masks_loader = DataLoader(
            test_masks_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
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
        ano_batch_size = 32

        test_anomaly_transforms = define_instance(args, "val_transforms")

        if "flair" in args.dataset["name"].lower():
            test_anomaly_large_ds = CacheDataset(data=large_group_flair_images, transform=test_anomaly_transforms)
            test_anomaly_medium_ds = CacheDataset(data=medium_group_flair_images, transform=test_anomaly_transforms)
            test_anomaly_small_ds = CacheDataset(data=small_group_flair_images, transform=test_anomaly_transforms)
        elif "adc" in args.dataset["name"].lower():
            test_anomaly_large_ds = CacheDataset(data=large_group_adc_images, transform=test_anomaly_transforms)
            test_anomaly_medium_ds = CacheDataset(data=medium_group_adc_images, transform=test_anomaly_transforms)
            test_anomaly_small_ds = CacheDataset(data=small_group_adc_images, transform=test_anomaly_transforms)


        test_anomaly_large_loader = DataLoader(
            test_anomaly_large_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_medium_loader = DataLoader(
            test_anomaly_medium_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_small_loader = DataLoader(
            test_anomaly_small_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        test_masks_transforms = transforms.Compose(
            [
                transforms.LoadImage(),
                transforms.EnsureChannelFirst(),
                custom_transforms.Get2DSlice(axis=2, offset=+2),
                transforms.ResizeWithPadOrCrop(spatial_size=(128, 128)),
                custom_transforms.SetBackgroundToZero()
            ]
        )
        test_masks_large_ds = CacheDataset(data=large_group_masks, transform=test_masks_transforms)
        test_masks_medium_ds = CacheDataset(data=medium_group_masks, transform=test_anomaly_transforms)
        test_masks_small_ds = CacheDataset(data=small_group_masks, transform=test_anomaly_transforms)


        test_masks_large_loader = DataLoader(
            test_masks_large_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_medium_loader = DataLoader(
            test_masks_medium_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_small_loader = DataLoader(
            test_masks_small_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    
    model = define_instance(args, "network_def").to(device)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE_TYPE))
    model.eval()


    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])


    @torch.no_grad()
    def my_sample(image, infer_scheduler, timesteps, return_intermediates=False):
        
        simplexObj = simplex.Simplex_CLASS()

        if args.noise["type"] == "simplex":
            noise = simplex_ddpm.generate_simplex_noise(simplexObj, image.shape, normalize=args.noise["normalize"]).to(device)
        if args.noise["type"] == "gaussian":
            noise = torch.randn(image.shape).to(device)
        

        if timesteps >= infer_scheduler.num_train_timesteps:
            print(timesteps, "is too high. Setting to", infer_scheduler.num_train_timesteps-1)

        timesteps_list = torch.Tensor([timesteps for a in range(image.shape[0])]).to(image.device).long()

        image = infer_scheduler.add_noise(image, noise, timesteps_list).to(device) #TODO


        intermediates = []
        intermediates_step = 20

                
        for t in range(timesteps, 0, -1): # va de timesteps à 0
            
            model_output = model(
                image, timesteps=torch.Tensor((t,)).to(device), context=None
            )
            #print(model_output.shape)
            
            image, _ = infer_scheduler.step(model_output, t, image)
        
            if (t== timesteps-1 or t%intermediates_step == 0) and return_intermediates:
                intermediates.append(image)

        if return_intermediates:
            return image, intermediates
        else:
            return image


    dm = DiceMetric(reduction="sum")

    def compute(image_loader, mask_loader):


        num_timesteps_to_try = np.arange(NOISE_MIN, NOISE_MAX, NOISE_INTERVAL)
        thresholds_to_try = np.arange(0.0, 0.6, 0.01) # from 0.0 to 0.6 with step 0.05

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
                    # Perform 3 inferences and average the results
                    infered_images = []
                    for _ in range(3):
                        infered_images.append(my_sample(test_images, infer_scheduler, timesteps=infer_timesteps, return_intermediates=False))
                    average_infered_image = torch.stack(infered_images, dim=0).mean(dim=0)
            
                for threshold in thresholds_to_try:
                    ano_segmentation = torch.abs(average_infered_image - test_images) > threshold

                    iou_score = compute_iou(ano_segmentation, test_masks)
                    flattened_iou_score = iou_score.cpu().numpy().flatten()
                    flattened_iou_score[np.isnan(flattened_iou_score)] = 0.0

                    
                    if np.isnan(iou_scores_df.loc[infer_timesteps, threshold]): # if the cell is empty
                        iou_scores_df.loc[infer_timesteps, threshold] = np.sum(flattened_iou_score)
                    else:
                        iou_scores_df.loc[infer_timesteps, threshold] += np.sum(flattened_iou_score) 

                    dice_score = dm(ano_segmentation, test_masks).cpu().numpy().flatten()
                    dice_score[np.isnan(dice_score)] = 0.0

                    if np.isnan(dice_scores_df.loc[infer_timesteps, threshold]): # if the cell is empty
                        dice_scores_df.loc[infer_timesteps, threshold] = np.sum(dice_score)
                    else:
                        dice_scores_df.loc[infer_timesteps, threshold] += np.sum(dice_score)

        #divide everything by the number of images
        iou_scores_df = iou_scores_df / len(image_loader.dataset)
        dice_scores_df = dice_scores_df / len(image_loader.dataset)

        return iou_scores_df, dice_scores_df

    # ----------- COMPUTING METRICS -----------

    metrics_result_text = ""

    if args.dataset["test"] == "brats":
        iou_scores_df, dice_scores_df = compute(test_anomaly_loader, test_masks_loader)

        best_iou = iou_scores_df.max().max()
        best_threshold = iou_scores_df.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df.max(axis=1).idxmax()

        best_dice = dice_scores_df.loc[best_num_timesteps, best_threshold]

        print(f"Best IOU: {best_iou}")
        metrics_result_text += f"Best IOU: {best_iou:.4f} - corresponding DICE {best_dice:.4f}\n"

        print(f"Best Threshold: {best_threshold}")
        metrics_result_text += f"Best Threshold: {best_threshold:.4f}\n"

        print(f"Best Number of Timesteps: {best_num_timesteps}")
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps}\n"

        iou_scores_df.to_csv(f"{SUB_EXPERIMENT_NAME}_{args.dataset["test"]}_scores_iou.csv")

    elif args.dataset["test"] == "isles":
        # large group
        iou_scores_df_large_group, dice_scores_df_large_group = compute(test_anomaly_large_loader, test_masks_large_loader)

        best_iou = iou_scores_df_large_group.max().max()
        best_threshold = iou_scores_df_large_group.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df_large_group.max(axis=1).idxmax()

        best_dice = dice_scores_df_large_group.loc[best_num_timesteps, best_threshold]

        print(f"Best IOU (large group): {best_iou}")
        metrics_result_text += f"Best IOU (large group): {best_iou:.4f} - corresponding DICE {best_dice:.4f}\n"

        print(f"Best Threshold (large group): {best_threshold}")
        metrics_result_text += f"Best Threshold (large group): {best_threshold:.4f}\n"

        print(f"Best Number of Timesteps (large group): {best_num_timesteps}")
        metrics_result_text += f"Best Number of Timesteps (large group): {best_num_timesteps}\n"
        metrics_result_text += "\n"

        iou_scores_df_large_group.to_csv(f"{SUB_EXPERIMENT_NAME}_{args.dataset["test"]}_scores_iou_large_group.csv")

        # medium group
        iou_scores_df_medium_group, dice_scores_df_medium_group = compute(test_anomaly_medium_loader, test_masks_medium_loader)

        best_iou = iou_scores_df_medium_group.max().max()
        best_threshold = iou_scores_df_medium_group.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df_medium_group.max(axis=1).idxmax()

        best_dice = dice_scores_df_medium_group.loc[best_num_timesteps, best_threshold]

        print(f"Best IOU (medium group): {best_iou}")
        metrics_result_text += f"Best IOU (medium group): {best_iou:.4f} - corresponding DICE {best_dice:.4f}\n"

        print(f"Best Threshold (medium group): {best_threshold}")
        metrics_result_text += f"Best Threshold (medium group): {best_threshold:.4f}\n"

        print(f"Best Number of Timesteps (medium group): {best_num_timesteps}")
        metrics_result_text += f"Best Number of Timesteps (medium group): {best_num_timesteps}\n"
        metrics_result_text += "\n"

        iou_scores_df_medium_group.to_csv(f"{SUB_EXPERIMENT_NAME}_{args.dataset["test"]}_scores_iou_medium_group.csv")

        # small group
        iou_scores_df_small_group, dice_scores_df_small_group = compute(test_anomaly_small_loader, test_masks_small_loader)

        best_iou = iou_scores_df_small_group.max().max()
        best_threshold = iou_scores_df_small_group.max(axis=0).idxmax()
        best_num_timesteps = iou_scores_df_small_group.max(axis=1).idxmax()

        best_dice = dice_scores_df_small_group.loc[best_num_timesteps, best_threshold]

        print(f"Best IOU (small group): {best_iou}")
        metrics_result_text += f"Best IOU (small group): {best_iou:.4f} - corresponding DICE {best_dice:.4f}\n"

        print(f"Best Threshold (small group): {best_threshold}")
        metrics_result_text += f"Best Threshold (small group): {best_threshold:.4f}\n"

        print(f"Best Number of Timesteps (small group): {best_num_timesteps}")
        metrics_result_text += f"Best Number of Timesteps (small group): {best_num_timesteps}\n"

        iou_scores_df_small_group.to_csv(f"{SUB_EXPERIMENT_NAME}_{args.dataset["test"]}_scores_iou_small_group.csv")



    # ----------- VISUALIZATION OF A BATCH -----------
    infer_timesteps_visualize = int(args.compute_metrics_reconstruction["noise_rate_visualize"]*args.noise["num_timesteps_full_noise"])

    if args.dataset["test"] == "brats":
        image_loader = test_anomaly_loader
        mask_loader = test_masks_loader
    elif args.dataset["test"] == "isles":
        image_loader = test_anomaly_large_loader
        mask_loader = test_masks_large_loader

    for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice
        if i>0:break

        test_anomaly_images = image_batch.to(device)
        test_anomaly_masks = mask_batch.to(device)
        test_anomaly_masks[test_anomaly_masks>0.5] = 1.0
        test_anomaly_masks[test_anomaly_masks<=0.5] = 0.0

        with autocast(device_type=DEVICE_TYPE, enabled=True):

            # Perform 3 inferences and average the results
            infered_images = []
            for _ in range(3):
                infered_images.append(my_sample(test_anomaly_images, infer_scheduler, timesteps=infer_timesteps_visualize, return_intermediates=False))
            average_infered_image = torch.stack(infered_images, dim=0).mean(dim=0)

    # ----------- PLOT -----------

    fig, axes = plt.subplots(6, 8, figsize=(25, 17), constrained_layout=True)
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
        
        

        # 3x average inferred images
        print(average_infered_image.shape)
        average_infered_image_cpu = average_infered_image[idx, 0].cpu().numpy()
        axes[1, idx*2].imshow(average_infered_image_cpu, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].set_title(f'Inferred {idx+1}')
        axes[1, idx*2].axis('off')

        axes[1, idx*2+1].hist(average_infered_image_cpu[average_infered_image_cpu>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[1, idx*2+1].set_ylim(0, 2000)
        axes[1, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Difference images
        difference_image = np.abs(original_image - average_infered_image_cpu)
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
        ground_truth_mask = test_anomaly_masks[idx, 0].cpu().numpy()
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
    plt.suptitle(f"Healthy reconstruction for {EXPERIMENT_NAME}, large group", fontsize=16)

    plt.figtext(0.0, 0.0, metrics_result_text, fontsize=16)


    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_{args.dataset["test"]}_metrics_anomaly_detection.png", transparent=False, dpi=150)

