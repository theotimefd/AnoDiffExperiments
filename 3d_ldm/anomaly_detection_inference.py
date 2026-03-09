"""
This file only works for 3D slice by slice inference.
"""

import os
import glob
import sys
from pathlib import Path

sys.path.append("../..")
#import opensimplex

from datasets import anomaly_datasets

#from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism, StrEnum
from torch.amp import autocast
from tqdm import tqdm
from monai.inferers import LatentDiffusionInferer
import nibabel as nib

from monai.networks.schedulers import DDPMScheduler

from make_anomaly_maps import make_anomaly_maps

import utils.custom_transforms as custom_transforms

import utils.simplex_ddpm as simplex_ddpm


from utils.utils import *

#from compute_metrics_anomaly_detection import compute_metrics



DEVICE_TYPE = "cuda:0"



def launch_anomaly_detection_inference(args, no_abs_value=False):
    # Two parts : the first 50% of the test data is used to select the best noise timestep value and best threshold.
    # The second 50% is used to compute the final IOU and DICE metrics with these best values.
    DEVICE_TYPE = "cuda:0"
    device = torch.device(DEVICE_TYPE)

    set_determinism(0)

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    SUB_EXPERIMENT_DIR = f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/"
    
    ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params/"
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/for_combine_experiment/" # final anomaly maps with best params
    dtprint(f"Anomaly maps best params will be saved in {ANOMALY_MAPS_DIR_SELECT_PARAMS}")
    os.makedirs(ANOMALY_MAPS_DIR_SELECT_PARAMS, exist_ok=True)
    os.makedirs(ANOMALY_MAPS_DIR, exist_ok=True)

    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)

    NOISE_MIN = int(args.noise["noise_rate_min"]*args.noise["num_timesteps_full_noise"])
    NOISE_MAX = int(args.noise["noise_rate_max"]*args.noise["num_timesteps_full_noise"])+1
    NOISE_INTERVAL = int(args.noise["noise_timesteps_interval"])

    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'

    TEXTCOLOR = 'black'
    plt.rcParams['text.color'] = TEXTCOLOR
    plt.rcParams['axes.labelcolor'] = TEXTCOLOR
    plt.rcParams['xtick.color'] = TEXTCOLOR
    plt.rcParams['ytick.color'] = TEXTCOLOR


    test_masks_transforms = transforms.Compose(
        [
            transforms.LoadImage(),
            transforms.EnsureChannelFirst(),
            transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
            custom_transforms.SetBackgroundToZero()
        ]
    )
    


    # -------------------- define the data --------------------
    if args.dataset["test"] == "brats":
        ano_dataset = anomaly_datasets.BRATS(args)

    if args.dataset["test"] == "isles":
        ano_dataset = anomaly_datasets.ISLES(args)
    
    if args.dataset["test"] == "soop":
        ano_dataset = anomaly_datasets.SOOP_large_only(args, batch_size=4)
    
    if args.dataset["test"] == "soop_fast":
        ano_dataset = anomaly_datasets.SOOP_Fast(args)
    
    # Define Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_autoencoder.pt")

    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=device, weights_only=True))
    autoencoder.eval()

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_network_def").to(device)

    trained_diffusion_path = os.path.join(MODELS_DIR, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(MODELS_DIR, "diffusion_unet_last.pt")

    unet.load_state_dict(torch.load(trained_diffusion_path, map_location="cuda:0"))
    unet.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=args.noise["num_timesteps_full_noise"],
        schedule="scaled_linear_beta",
        beta_start=args.noise["beta_start"],
        beta_end=args.noise["beta_end"],
    )


    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with autocast("cuda", enabled=True):
            check_data = ano_dataset.first()
            z = autoencoder.encode_stage_2_inputs(check_data.to(device))


    scale_factor = 1 / torch.std(z)
    tprint(f"Scale_factor: {scale_factor}")


    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-5 * 1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[100, 1000], gamma=0.1)




    

    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])


    # ------------------------ SOOP dataset ------------------------ #
    if args.dataset["test"] == "soop":
        
        # --------------------------------- large group
        if "adc" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 170
        elif "flair" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 250
                

        if no_abs_value:
            os.makedirs(ANOMALY_MAPS_DIR+"large_no_abs_value/", exist_ok=True)
            dtprint("No abs value save anomaly maps: not yet implemented")
        else:
            os.makedirs(ANOMALY_MAPS_DIR+"large/", exist_ok=True)
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, ano_dataset.test_anomaly_large_loader_metrics, ano_dataset.test_anomaly_large_images_metrics, best_num_timesteps_large_group, ANOMALY_MAPS_DIR+"large/")
        
    