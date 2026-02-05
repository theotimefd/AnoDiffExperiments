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

from monai.networks.schedulers import DDPMScheduler


from monai.utils import StrEnum
from typing import Union

import pandas as pd

import utils.custom_transforms as custom_transforms
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm
from utils.utils import *

from monai.metrics import PSNRMetric, SSIMMetric, MultiScaleSSIMMetric

import lpips

def scale_intensity_from_histogram_peak(input_image, target_value=1.0):
    # to be used only on mri images with intensities between 0 and 1
    input_np = input_image.cpu().numpy()

    hist, bin_edges = np.histogram(input_np.flatten(), bins=100, range=(np.max(input_np)/15.0, 0.8))

    peak_value = bin_edges[np.argmax(hist)]

    normalized_image = input_image / peak_value * target_value

    return normalized_image

def launch_compute_metrics_reconstruction(args):
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

    batch_size = args.dataset["batch_size"]
    num_workers = args.dataset["num_workers"]


    # transforms
    test_reconstruction_transforms = define_instance(args, "val_transforms")
    test_reconstruction_ds = CacheDataset(data=test_reconstruction_datalist, transform=test_reconstruction_transforms)


    test_reconstruction_loader = DataLoader(
        test_reconstruction_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = define_instance(args, "network_def").to(device)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE_TYPE))
    model.eval()


    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], 
                                                            schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], 
                                                            persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

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
            tprint(f"{timesteps} is too high. Setting to {infer_scheduler.num_train_timesteps-1}")

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

    # ----------- COMPUTING METRICS -----------

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    ssim_metric_3d = SSIMMetric(spatial_dims=3, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)

    mse = {noise: [] for noise in NOISE_RANGE} # for each noise level there is a list of mse values
    psnr = {noise: [] for noise in NOISE_RANGE}
    ssim = {noise: [] for noise in NOISE_RANGE}
    lpips_dict = {noise: [] for noise in NOISE_RANGE}

    loss_fn_lpips = lpips.LPIPS(net='alex').to(device) # Higher means further/more different. Lower means more similar.

    full_volume_test = False

    for image_batch in tqdm(test_reconstruction_loader):

        test_reconstruction_images = image_batch.to(device)

        with autocast(device_type=DEVICE_TYPE, enabled=True):

            for i, noise_timesteps in enumerate(NOISE_RANGE):

                #print(f"inference for {noise_timesteps} noise timesteps")
                
                if len(image_batch.shape)==4: # 2D slices
                    
                    infered = my_sample(test_reconstruction_images, infer_scheduler, timesteps=noise_timesteps, return_intermediates=False)
                    infered = torch.clamp(scale_intensity_from_histogram_peak(infered, 2.0/7.0), 0.0, 1.0)

                    mse[noise_timesteps].append(F.mse_loss(infered, test_reconstruction_images).detach().cpu().numpy().flatten())
                    ssim[noise_timesteps].append(np.mean(ssim_metric(test_reconstruction_images, infered).detach().cpu().numpy().flatten()))
                    psnr[noise_timesteps].append(np.mean(psnr_metric(infered, test_reconstruction_images).detach().cpu().numpy().flatten()))
                    lpips_dict[noise_timesteps].append(np.mean(loss_fn_lpips.forward(infered.to(device), test_reconstruction_images).detach().cpu().numpy().flatten()))

                elif len(image_batch.shape)==5: # full volumes by slices
                    full_volume_test = True
                    infered_slices = []
                    lpips_volume = [] 

                    for slice_idx in range(args.slice_indexes_start, args.slice_indexes_end):
                        infered_slice = my_sample(test_reconstruction_images[...,slice_idx], infer_scheduler, timesteps=noise_timesteps, return_intermediates=False)
                        lpips_volume.append(np.mean(loss_fn_lpips.forward(infered_slice.to(device), test_reconstruction_images[...,slice_idx]).detach().cpu().numpy().flatten())) # since lpips only works for 2D images
                        infered_slices.append(infered_slice.unsqueeze(-1))

                    infered = torch.cat(infered_slices, dim=-1)
                    infered = torch.clamp(scale_intensity_from_histogram_peak(infered, 2.0/7.0), 0.0, 1.0)
                    
                    #TODO here we shouldn't use mean, should instead put all the values inside the list and only call average once at the end
                    mse[noise_timesteps].append(F.mse_loss(infered, test_reconstruction_images[...,args.slice_indexes_start:args.slice_indexes_end]).detach().cpu().numpy().flatten())
                    ssim[noise_timesteps].append(np.mean(ssim_metric_3d(test_reconstruction_images[...,args.slice_indexes_start:args.slice_indexes_end], infered).detach().cpu().numpy().flatten()))
                    psnr[noise_timesteps].append(np.mean(psnr_metric(infered, test_reconstruction_images[...,args.slice_indexes_start:args.slice_indexes_end]).detach().cpu().numpy().flatten()))
                    lpips_dict[noise_timesteps].append(np.mean(lpips_volume))


    # ----------- VISUALIZATION OF A BATCH -----------
    infer_timesteps_visualize = int(args.noise["noise_rate_visualize"]*args.noise["num_timesteps_full_noise"])


    for i,(image_batch) in enumerate(test_reconstruction_loader):
        if i>0:break

        if len(image_batch.shape)==4: # 2D slices
            test_reconstruction_images = image_batch.to(device)
        elif len(image_batch.shape)==5: # full volumes
            test_reconstruction_images = image_batch[...,image_batch.shape[-1]//2].to(device) # visualize the slice in the middle of the volume

        with autocast(device_type=DEVICE_TYPE, enabled=True):

            infered, intermediates = my_sample(test_reconstruction_images, infer_scheduler, timesteps=infer_timesteps_visualize, return_intermediates=True)
            first_noisy_images = intermediates[0]

    # ----------- PLOT -----------
    if not full_volume_test:
        metric_result_text = f"With ({NOISE_MIN},{NOISE_MAX}) timesteps noise range, on the whole test_reconstruction_dataset (n={batch_size}), \n"
    elif full_volume_test:
        metric_result_text = f"With ({NOISE_MIN},{NOISE_MAX}) timesteps noise range, on the whole test_reconstruction_dataset (n={batch_size}), full volume test\n"
    metric_result_text += f"Mean MSE ↓: {np.mean([np.mean(item) for sublist in mse.values() for item in sublist]):.3f}\n"
    metric_result_text += f"Mean PSNR ↑: {np.mean([np.mean(item) for sublist in psnr.values() for item in sublist]):.3f}\n"
    metric_result_text += f"Mean SSIM ↑: {np.mean([np.mean(item) for sublist in ssim.values() for item in sublist]):.3f}\n"
    metric_result_text += f"Mean SSIM ↓: {np.mean([np.mean(item) for sublist in ssim.values() for item in sublist]):.3f}\n"

    fig, axes = plt.subplots(6, 8, figsize=(25, 25), constrained_layout=True)
    plt.tight_layout()

    for idx in range(min(4, test_reconstruction_images.shape[0])):

        # Original test_reconstruction images
        original_image = test_reconstruction_images[idx, 0].cpu().numpy()
        axes[0, idx*2].imshow(original_image, cmap='gray', vmin=0, vmax=1)
        axes[0, idx*2].set_title(f'Original {idx+1}')
        axes[0, idx*2].axis('off')

        axes[0, idx*2+1].hist(original_image[original_image>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[0, idx*2+1].set_ylim(0, 2000)
        axes[0, idx*2+1].set_aspect('auto')  # Set the aspect ratio to auto to match the imshow plot
        
        
        # First noisy images

        first_noisy_image_no_background = first_noisy_images[idx, 0].cpu().numpy().copy()
        first_noisy_image_no_background[original_image < 0.01] = 0.0
        

        

        if args.noise["normalize"]:
            #axes[1, idx*2].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
            axes[1, idx*2].imshow(first_noisy_image_no_background, cmap='gray', vmin=0, vmax=1)
            axes[1, idx*2].set_title(f'Noisy {idx+1}, {args.noise["noise_rate_visualize"]*100}% noise (timesteps={infer_timesteps_visualize})')
            axes[1, idx*2].axis('off')

            axes[1, idx*2+1].hist(first_noisy_image_no_background.flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
            axes[1, idx*2+1].set_ylim(0, 2000)
            axes[1, idx*2+1].set_aspect('auto')  # Set the aspect ratio to auto to match the imshow plot
        else:
            #axes[1, idx*2].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
            axes[1, idx*2].imshow(first_noisy_image_no_background, cmap='gray', vmin=-1, vmax=1)
            axes[1, idx*2].set_title(f'Noisy {idx+1}, {args.noise["noise_rate_visualize"]*100}% noise (timesteps={infer_timesteps_visualize})')
            axes[1, idx*2].axis('off')

            axes[1, idx*2+1].hist(first_noisy_image_no_background.flatten(), bins=50, color='blue', alpha=0.7, range=(-0.3, 1.0))
            axes[1, idx*2+1].set_ylim(0, 2000)
            axes[1, idx*2+1].set_aspect('auto')  # Set the aspect ratio to auto to match the imshow plot


        # Inferred images
        infered_image = infered[idx, 0].cpu().numpy()
        axes[2, idx*2].imshow(infered_image, cmap='gray', vmin=0, vmax=1)
        axes[2, idx*2].set_title(f'Inferred {idx+1}')
        axes[2, idx*2].axis('off')

        axes[2, idx*2+1].hist(infered_image[infered_image>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[2, idx*2+1].set_ylim(0, 2000)
        axes[2, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Difference images
        axes[3, idx*2].imshow(np.abs(infered_image-original_image), cmap='jet', vmin=0, vmax=1)
        axes[3, idx*2].set_title(f'Difference {idx+1}')
        axes[3, idx*2].axis('off')

        axes[3, idx*2+1].hist(np.abs(infered_image-original_image).flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[3, idx*2+1].set_ylim(0, 2000)
        axes[3, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        axes[0, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
        axes[1, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
        axes[2, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
        axes[3, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot

        # Arrow from original image to noisy image
        axes[0, idx*2].annotate( 
            '', xy=(0.0, 128), xycoords=axes[0, idx*2].transData,
            xytext=(0.0, 0), textcoords=axes[1, idx*2].transData,
            arrowprops=dict(arrowstyle="<->", color='grey', lw=2, connectionstyle="arc3, rad=-0.2")
        )
        true = test_reconstruction_images[idx, 0].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        noisy = torch.from_numpy(first_noisy_image_no_background).to(device).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions   
        text_metrics_orig_and_noisy = f"MSE: {F.mse_loss(true, noisy).detach().cpu().numpy().mean():.4f}\n"
        text_metrics_orig_and_noisy += f"SSIM: {np.mean(ssim_metric(true, noisy).detach().cpu().numpy().mean()):.4f}\n"
        text_metrics_orig_and_noisy += f"PSNR: {psnr_metric(true, noisy).detach().cpu().numpy().mean():.2f}"

        axes[0, idx*2].text(
            -3, 165, text_metrics_orig_and_noisy, transform=axes[0, idx*2].transData, #TODO first_noisy_images[idx, 0])
            color='grey', fontsize=12, verticalalignment='center'
        )


        # Arrow from original image to infered image
        axes[0, idx*2].annotate(
            '', xy=(-5, 64), xycoords=axes[0, idx*2].transData, #'axes fraction',
            xytext=(-5, 64), textcoords=axes[2, idx*2].transData,
            arrowprops=dict(arrowstyle="<->", color='grey', lw=2, connectionstyle="arc3, rad=-0.07")
        )

        #pred = infered[idx, 0].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions   
        """text_metrics_orig_and_pred = f"MSE: {F.mse_loss(true, pred).detach().cpu().numpy().mean():.4f}\n"
        text_metrics_orig_and_pred += f"SSIM: {np.mean(ssim_metric(true, pred).detach().cpu().numpy().mean()):.4f}\n"
        text_metrics_orig_and_pred += f"PSNR: {psnr_metric(true, pred).detach().cpu().numpy().mean():.2f}"


        axes[2, idx*2].text(
            -10, 160, text_metrics_orig_and_pred, transform=axes[1, idx*2].transData,
            color='grey', fontsize=12, verticalalignment='center'
        ) """

    # Add overall title with metric results
    plt.suptitle(f"Healthy reconstruction for {EXPERIMENT_NAME}", fontsize=16)


    for ax in axes[4, 0:2]: # two merge two subplots
        ax.remove()
    gs = axes[4, 0].get_gridspec()
    axbig1 = fig.add_subplot(gs[4, 0:2])

    # PSNR plot with error bars
    axbig1.errorbar(
        [noise/args.noise["num_timesteps_full_noise"] for noise in NOISE_RANGE],
        [np.mean(psnr[noise]) for noise in NOISE_RANGE],
        yerr=[np.std(psnr[noise]) for noise in NOISE_RANGE],
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
        [noise/args.noise["num_timesteps_full_noise"] for noise in NOISE_RANGE],
        [np.mean(ssim[noise]) for noise in NOISE_RANGE],
        yerr=[np.std(ssim[noise]) for noise in NOISE_RANGE],
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
        [noise/args.noise["num_timesteps_full_noise"] for noise in NOISE_RANGE],
        [np.mean(mse[noise]) for noise in NOISE_RANGE],
        yerr=[np.std(mse[noise]) for noise in NOISE_RANGE],
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

    # Add an empty row to create more whitespace for the figtext
    for idx in range(8):
        axes[5, idx].axis('off')


    plt.figtext(0.04, 0.04, f"Reconstruction metrics for the whole test_reconstruction dataset (std error bars)\nFor {args.noise['noise_rate_max']*100}% noise:\nPSNR: {np.mean(psnr[NOISE_RANGE[-1]]):.2f} ± {np.std(psnr[NOISE_RANGE[-1]]):.2f}\nSSIM: {np.mean(ssim[NOISE_RANGE[-1]]):.4f} ± {np.std(ssim[NOISE_RANGE[-1]]):.4f}\nMSE: {np.mean(mse[NOISE_RANGE[-1]]):.4f} ± {np.std(mse[NOISE_RANGE[-1]]):.4f}\nLPIPS: {np.mean(lpips_dict[NOISE_RANGE[-1]]):.4f} ± {np.std(lpips_dict[NOISE_RANGE[-1]]):.4f}", fontsize=16)


    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_metrics_reconstruction.png", transparent=False, dpi=150)

