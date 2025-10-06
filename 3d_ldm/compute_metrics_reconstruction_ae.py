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

from monai.metrics import compute_iou

from monai.metrics import PSNRMetric, SSIMMetric, MultiScaleSSIMMetric

import lpips

def launch_compute_metrics_reconstruction_ae(args):
    """
    Computes reconstruction metrics on the test_reconstruction set and visualize some results
    Works for models trained on 2D slices, either with single 2D slice validation or full volume validation
    """


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
    NOISE_RANGE = range(NOISE_MIN,NOISE_MAX,args.compute_metrics_reconstruction["noise_timesteps_interval"])

    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'

    TEXTCOLOR = 'black'
    plt.rcParams['text.color'] = TEXTCOLOR
    plt.rcParams['axes.labelcolor'] = TEXTCOLOR
    plt.rcParams['xtick.color'] = TEXTCOLOR
    plt.rcParams['ytick.color'] = TEXTCOLOR

    # ----------- MODEL SETTINGS -----------

    test_reconstruction_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset['name']}/test.csv")
    test_reconstruction_images_path = []

    with open(test_reconstruction_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):
            #print(line)
            test_reconstruction_images_path.append(ROOT_DIR+line[0])

    #test_reconstruction_datalist = sorted(test_reconstruction_images_path)
    test_reconstruction_datalist = test_reconstruction_images_path

    #test_unhealthy_datalist = test_unhealthy_images_path

    batch_size = args.autoencoder_train["batch_size"]
    num_workers = args.autoencoder_train["num_workers"]


    # transforms
    test_reconstruction_transforms = define_instance(args, "val_transforms")
    test_reconstruction_ds = CacheDataset(data=test_reconstruction_datalist, transform=test_reconstruction_transforms)


    test_reconstruction_loader = DataLoader(
        test_reconstruction_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

        # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    trained_g_path = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_autoencoder.pt")

    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=device, weights_only=True))
    autoencoder.eval()


    # ----------- COMPUTING METRICS -----------


    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)

    mse = []
    psnr = []
    ssim = []
    lpips_list = []

    loss_fn_lpips = lpips.LPIPS(net='alex').to(device) # Higher means further/more different. Lower means more similar.

    full_volume_test = False

    for image_batch in tqdm(test_reconstruction_loader):

        test_reconstruction_images = image_batch.to(device)

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            with torch.no_grad():
                infered, _, _ = autoencoder(test_reconstruction_images)

                mse.append(F.mse_loss(infered, test_reconstruction_images).detach().cpu().numpy().flatten())
                ssim.append(np.mean(ssim_metric(test_reconstruction_images, infered).detach().cpu().numpy().flatten()))
                psnr.append(np.mean(psnr_metric(infered, test_reconstruction_images).detach().cpu().numpy().flatten()))

                lpips_metric_volume = []
                for i in range(infered.shape[-1]):
                    lpips_metric_volume.append(loss_fn_lpips.forward(infered[..., i], test_reconstruction_images[..., i]).detach().cpu().numpy().flatten())
                lpips_list.append(np.mean(lpips_metric_volume))


    # ----------- VISUALIZATION OF A BATCH -----------

    for i,(image_batch) in enumerate(test_reconstruction_loader):
        if i>0:break


        test_reconstruction_images = image_batch.to(device)

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            with torch.no_grad():
                infered, _, _ = autoencoder(test_reconstruction_images)

    # ----------- PLOT -----------
    if not full_volume_test:
        metric_result_text = f"Autoencoder reconstruction performance on the whole test_reconstruction_dataset (n={batch_size}), \n"
    
    metric_result_text += f"Mean MSE ↓: {np.mean(mse):.3f}\n"
    metric_result_text += f"Mean PSNR ↑: {np.mean(psnr):.3f}\n"
    metric_result_text += f"Mean SSIM ↑: {np.mean(ssim):.3f}\n"
    metric_result_text += f"Mean LPIPS ↓: {np.mean(lpips_list):.3f}\n"

    fig, axes = plt.subplots(4, 8, figsize=(25, 25), constrained_layout=True)
    plt.tight_layout()

    for idx in range(min(4, test_reconstruction_images.shape[0])):

        # Original test_reconstruction images
        original_image = test_reconstruction_images[idx, 0,:,:,test_reconstruction_images.shape[-1]//2].cpu().numpy()
        axes[0, idx*2].imshow(original_image, cmap='gray', vmin=0, vmax=1)
        axes[0, idx*2].set_title(f'Original {idx+1}')
        axes[0, idx*2].axis('off')

        axes[0, idx*2+1].hist(original_image[original_image>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[0, idx*2+1].set_ylim(0, 2000)
        axes[0, idx*2+1].set_aspect('auto')  # Set the aspect ratio to auto to match the imshow plot
        

        # Inferred images
        infered_image = infered[idx, 0,:,:,infered.shape[-1]//2].cpu().numpy()
        axes[1, idx*2].imshow(infered_image, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].set_title(f'Inferred {idx+1}')
        axes[1, idx*2].axis('off')

        axes[1, idx*2+1].hist(infered_image[infered_image>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[1, idx*2+1].set_ylim(0, 2000)
        axes[1, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Difference images
        axes[2, idx*2].imshow(np.abs(infered_image-original_image), cmap='jet', vmin=0, vmax=1)
        axes[2, idx*2].set_title(f'Difference {idx+1}')
        axes[2, idx*2].axis('off')

        axes[2, idx*2+1].hist(np.abs(infered_image-original_image).flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[2, idx*2+1].set_ylim(0, 2000)
        axes[2, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        axes[0, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
        axes[1, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
        axes[2, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot

        


        # Arrow from original image to infered image
        axes[0, idx*2].annotate(
            '', xy=(-5, 64), xycoords=axes[0, idx*2].transData, #'axes fraction',
            xytext=(-5, 64), textcoords=axes[1, idx*2].transData,
            arrowprops=dict(arrowstyle="<->", color='grey', lw=2, connectionstyle="arc3, rad=-0.07")
        )

    # Add overall title with metric results
    plt.suptitle(f"Healthy reconstruction for {EXPERIMENT_NAME}", fontsize=16)



    # Add an empty row to create more whitespace for the figtext
    for idx in range(8):
        axes[3, idx].axis('off')


    plt.figtext(0.04, 0.04, f"Autoencoder reconstruction metrics for the whole test_reconstruction dataset\nPSNR: {np.mean(psnr):.2f} ± {np.std(psnr):.2f}\nSSIM: {np.mean(ssim):.4f} ± {np.std(ssim):.4f}\nMSE: {np.mean(mse):.4f} ± {np.std(mse):.4f}\nLPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}", fontsize=16)


    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_metrics_reconstruction_autoencoder.png", transparent=False, dpi=150)

