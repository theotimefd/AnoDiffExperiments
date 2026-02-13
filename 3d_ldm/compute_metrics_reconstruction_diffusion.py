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
from monai.utils import set_determinism, first
from monai.data.utils import pad_list_data_collate
from torch.amp import autocast
from tqdm import tqdm
import random
from monai.inferers import LatentDiffusionInferer

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
from utils.utils import *

from monai.metrics import compute_iou

from monai.metrics import PSNRMetric, SSIMMetric, MultiScaleSSIMMetric

import lpips

def launch_compute_metrics_reconstruction_diffusion(args):
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

    NOISE_MIN = int(args.noise["noise_rate_min"]*args.noise["num_timesteps_full_noise"])
    NOISE_MAX = int(args.noise["noise_rate_max"]*args.noise["num_timesteps_full_noise"])+1
    NOISE_RANGE = range(NOISE_MIN,NOISE_MAX,args.noise["noise_timesteps_interval"])

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
            check_data = first(test_reconstruction_loader)
            z = autoencoder.encode_stage_2_inputs(check_data.to(device))


    scale_factor = 1 / torch.std(z)
    tprint(f"Scale_factor: {scale_factor}")


    # We define the inferer using the scale factor:
    #inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-5 * 1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[100, 1000], gamma=0.1)

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads()) 
    torch.autograd.set_detect_anomaly(False)

    # ----------- COMPUTING METRICS -----------

    ssim_metric = SSIMMetric(spatial_dims=3, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)

    mse = {noise: [] for noise in NOISE_RANGE} # for each noise level there is a list of mse values
    psnr = {noise: [] for noise in NOISE_RANGE}
    ssim = {noise: [] for noise in NOISE_RANGE}
    #lpips_dict = {noise: [] for noise in NOISE_RANGE}

    #loss_fn_lpips = lpips.LPIPS(net='alex').to(device) # Higher means further/more different. Lower means more similar.

    full_volume_test = False

    for image_batch in tqdm(test_reconstruction_loader):

        optimizer_diff.zero_grad(set_to_none=True)
        test_reconstruction_images = image_batch.to(device)

        with torch.no_grad():
            with autocast("cuda", enabled=True):

                latents = autoencoder.encode_stage_2_inputs(test_reconstruction_images)    

                for noise_timesteps in NOISE_RANGE:      
                    # Add noise to latents
                    noise = torch.randn_like(latents).to(device)
                    timesteps = torch.randint(0, noise_timesteps, (latents.shape[0],), device=device).long()
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    
                    # Denoise completely using the UNet
                    scheduler.set_timesteps(scheduler.num_train_timesteps)
                    current_latents = noisy_latents * scale_factor
                    
                    for t in range(noise_timesteps-1, -1, -1):
                        noise_pred = unet(current_latents, timesteps=torch.tensor([t], device=device).expand(latents.shape[0]))
                        current_latents, _ = scheduler.step(noise_pred, t, current_latents)
                    
                    # Decode the denoised latents
                    current_latents = current_latents / scale_factor

                    reconstructed_images = autoencoder.decode(current_latents)
                    normalized_reconstructed_images = scale_intensity_from_histogram_peak(reconstructed_images, target_value=2.0/7.0)

                    mse[noise_timesteps].append(F.mse_loss(normalized_reconstructed_images, test_reconstruction_images).detach().cpu().numpy().flatten())
                    ssim[noise_timesteps].append(ssim_metric(test_reconstruction_images, normalized_reconstructed_images).detach().cpu().numpy().flatten())
                    psnr[noise_timesteps].append(psnr_metric(normalized_reconstructed_images, test_reconstruction_images).detach().cpu().numpy().flatten())

                    """lpips_metric_volume = []
                    for i in range(infered.shape[-1]):
                        lpips_metric_volume.append(loss_fn_lpips.forward(infered[..., i], test_reconstruction_images[..., i]).detach().cpu().numpy().flatten())
                    lpips_list.append(np.mean(lpips_metric_volume))"""


    #clean up the metric lists
    for noise_timesteps in NOISE_RANGE:
        #flatten the list of lists
        mse[noise_timesteps] = [item for sublist in mse[noise_timesteps] for item in sublist]
        psnr[noise_timesteps] = [item for sublist in psnr[noise_timesteps] for item in sublist]
        ssim[noise_timesteps] = [item for sublist in ssim[noise_timesteps] for item in sublist]
        #lpips_dict[noise_timesteps] = [item for sublist in lpips_dict[noise_timesteps] for item in sublist]

    # ----------- VISUALIZATION OF A BATCH -----------

    infer_timesteps_visualize = int(args.noise["noise_rate_visualize"]*args.noise["num_timesteps_full_noise"])

    for i,(image_batch) in enumerate(test_reconstruction_loader):
        if i>0:break


        test_reconstruction_images = image_batch.to(device)

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            with torch.no_grad():
                
                latents = autoencoder.encode_stage_2_inputs(test_reconstruction_images)  

                noise = torch.randn_like(latents).to(device)
                timesteps = torch.randint(0, infer_timesteps_visualize, (latents.shape[0],), device=device).long()
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                # Denoise completely using the UNet
                scheduler.set_timesteps(scheduler.num_train_timesteps)
                current_latents = noisy_latents * scale_factor
                
                for t in tqdm(range(infer_timesteps_visualize-1, -1, -1)):
                    noise_pred = unet(current_latents, timesteps=torch.tensor([t], device=device).expand(latents.shape[0]))
                    current_latents, _ = scheduler.step(noise_pred, t, current_latents)
                
                # Decode the denoised latents
                current_latents = current_latents / scale_factor

                reconstructed_images = autoencoder.decode(current_latents)
                normalized_reconstructed_images = scale_intensity_from_histogram_peak(reconstructed_images, target_value=2.0/7.0)

    # ----------- PLOT -----------
    if not full_volume_test:
        metric_result_text = f"With ({NOISE_MIN},{NOISE_MAX}) timesteps noise range, on the whole test_reconstruction_dataset (n={batch_size}), \n"
    elif full_volume_test:
        metric_result_text = f"With ({NOISE_MIN},{NOISE_MAX}) timesteps noise range, on the whole test_reconstruction_dataset (n={batch_size}), full volume test\n"
    metric_result_text += f"Mean MSE ↓: {np.mean([item for sublist in mse.values() for item in sublist]):.3f}\n"
    metric_result_text += f"Mean PSNR ↑: {np.mean([item for sublist in psnr.values() for item in sublist]):.3f}\n"
    metric_result_text += f"Mean SSIM ↑: {np.mean([item for sublist in ssim.values() for item in sublist]):.3f}\n"
    

    fig, axes = plt.subplots(6, 8, figsize=(25, 25), constrained_layout=True)
    plt.tight_layout()

    tprint(psnr)

    for idx in range(min(4, test_reconstruction_images.shape[0])):

        # Original test_reconstruction images
        original_image = test_reconstruction_images[idx, 0].cpu().numpy()
        axes[0, idx*2].imshow(original_image[...,original_image.shape[-1]//2], cmap='gray', vmin=0, vmax=1)
        axes[0, idx*2].set_title(f'Original {idx+1}')
        axes[0, idx*2].axis('off')

        axes[0, idx*2+1].hist(original_image[original_image>0.01][...,original_image.shape[-1]//2].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[0, idx*2+1].set_ylim(0, 2000)
        axes[0, idx*2+1].set_aspect('auto')  # Set the aspect ratio to auto to match the imshow plot
        
        

        # Inferred images
        infered_image = normalized_reconstructed_images[idx, 0].cpu().numpy()
        axes[1, idx*2].imshow(infered_image[...,infered_image.shape[-1]//2], cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].set_title(f'Inferred {idx+1}')
        axes[1, idx*2].axis('off')

        axes[1, idx*2+1].hist(infered_image[infered_image>0.01][...,infered_image.shape[-1]//2].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[1, idx*2+1].set_ylim(0, 2000)
        axes[1, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Difference images
        axes[2, idx*2].imshow(np.abs(infered_image-original_image)[...,infered_image.shape[-1]//2], cmap='jet', vmin=0, vmax=1)
        axes[2, idx*2].set_title(f'Difference {idx+1}')
        axes[2, idx*2].axis('off')

        axes[2, idx*2+1].hist(np.abs(infered_image-original_image)[...,infered_image.shape[-1]//2].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
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


    for ax in axes[4, 0:2]: # two merge two subplots
        ax.remove()
    gs = axes[4, 0].get_gridspec()
    axbig1 = fig.add_subplot(gs[4, 0:2])

    # PSNR plot with error bars
    axbig1.errorbar(
        [noise_it/args.noise["num_timesteps_full_noise"] for noise_it in NOISE_RANGE],
        [np.mean(psnr[noise_it]) for noise_it in NOISE_RANGE],
        yerr=[np.std(psnr[noise_it]) for noise_it in NOISE_RANGE],
        marker='o', label='PSNR', color='blue', capsize=4
    )
    axbig1.set_title('Peak Signal-to-Noise Ratio (PSNR) ↑')
    axbig1.set_xlabel('Noise Timesteps')
    axbig1.set_ylabel('PSNR')
    axbig1.grid(True)
    axbig1.legend()

    for ax in axes[4, 2:4]:
        ax.remove()
    gs = axes[4, 4].get_gridspec()
    axbig2 = fig.add_subplot(gs[4, 2:4])


    # SSIM plot
    axbig2.errorbar(
        [noise_it/args.noise["num_timesteps_full_noise"] for noise_it in NOISE_RANGE],
        [np.mean(ssim[noise_it]) for noise_it in NOISE_RANGE],
        yerr=[np.std(ssim[noise_it]) for noise_it in NOISE_RANGE],
        marker='o', label='SSIM', color='blue', capsize=4
    )
    axbig2.set_title('Structural Similarity Index Metric (SSIM) ↑')
    axbig2.set_xlabel('Noise rate')
    axbig2.set_ylabel('SSIM')
    axbig2.grid(True)
    axbig2.legend()

    for ax in axes[4, 4:6]:
        ax.remove()
    gs = axes[4, 2].get_gridspec()
    axbig3 = fig.add_subplot(gs[4, 4:6])

    # MSE plot
    axbig3.errorbar(
        [noise_it/args.noise["num_timesteps_full_noise"] for noise_it in NOISE_RANGE],
        [np.mean(mse[noise_it]) for noise_it in NOISE_RANGE],
        yerr=[np.std(mse[noise_it]) for noise_it in NOISE_RANGE],
        marker='o', label='MSE', color='blue', capsize=4
    )
    axbig3.set_title('Mean Squared Error (MSE) ↓')
    axbig3.set_xlabel('Noise rate')
    axbig3.set_ylabel('MSE')
    axbig3.grid(True)
    axbig3.legend()

    for ax in axes[4, 6:8]:
        ax.remove()
    gs = axes[4, 2].get_gridspec()
    axbig4 = fig.add_subplot(gs[4, 6:8])

    # LPIPS plot
    """
    axbig4.errorbar(
        [noise/args.noise["num_timesteps_full_noise"] for noise in NOISE_RANGE],
        [np.mean(lpips_dict[noise]) for noise in NOISE_RANGE],
        yerr=[np.std(lpips_dict[noise]) for noise in NOISE_RANGE],
        marker='o', label='LPIPS', color='blue', capsize=4
    )
    axbig4.set_title('Learned Perceptual Image Patch Similarity (LPIPS) ↓')
    axbig4.set_xlabel('Noise rate')
    axbig4.set_ylabel('LPIPS')
    axbig4.grid(True)
    axbig4.legend()
    """
    # Add an empty row to create more whitespace for the figtext
    for idx in range(8):
        axes[5, idx].axis('off')


    plt.figtext(0.04, 0.04, f"Reconstruction metrics for the whole test_reconstruction dataset (std error bars)\nFor {args.noise['noise_rate_max']*100}% noise:\nPSNR: {np.mean(psnr[NOISE_RANGE[-1]]):.2f} ± {np.std(psnr[NOISE_RANGE[-1]]):.2f}\nSSIM: {np.mean(ssim[NOISE_RANGE[-1]]):.4f} ± {np.std(ssim[NOISE_RANGE[-1]]):.4f}\nMSE: {np.mean(mse[NOISE_RANGE[-1]]):.4f} ± {np.std(mse[NOISE_RANGE[-1]]):.4f}", fontsize=16)


    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_metrics_reconstruction_diffusion.png", transparent=False, dpi=150)

