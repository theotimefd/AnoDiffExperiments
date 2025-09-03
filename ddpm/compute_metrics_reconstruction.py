import os
import time
import glob
import sys
import argparse
import json
from pathlib import Path
sys.path.append("../..")
sys.path.append("..")
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

def launch_compute_metrics_reconstruction(args):
    DEVICE_TYPE = "cuda:0"

    set_determinism(0)

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"

    IMAGE_SIZE = args.image_size

    model_path = f"{args.root_dir}/AnoDiffExperiments/{args['experiment_name']}/{args['sub_experiment_name']}/models/{SUB_EXPERIMENT_NAME}_best_model.pth"

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)

    NOISE_MIN = args.compute_metrics_reconstruction["noise_timesteps_min"]
    NOISE_MAX = args.compute_metrics_reconstruction["noise_timesteps_max"]+1
    NOISE_RANGE = range(NOISE_MIN,NOISE_MAX,args.compute_metrics_reconstruction["noise_interval"])

    plt.rcParams['axes.facecolor']='white'
    plt.rcParams['savefig.facecolor']='white'

    TEXTCOLOR = 'black'
    plt.rcParams['text.color'] = TEXTCOLOR
    plt.rcParams['axes.labelcolor'] = TEXTCOLOR
    plt.rcParams['xtick.color'] = TEXTCOLOR
    plt.rcParams['ytick.color'] = TEXTCOLOR

    # ----------- MODEL SETTINGS -----------

    test_reconstruction_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset["name"]}/test.csv")
    test_reconstruction_images_path = []

    with open(test_reconstruction_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):
            #print(line)
            test_reconstruction_images_path.append(ROOT_DIR+line[0])

    #test_reconstruction_datalist = sorted(test_reconstruction_images_path)
    test_reconstruction_datalist = test_reconstruction_images_path

    #val_datalist = sorted(val_images_path)
    val_datalist = val_images_path

    #test_unhealthy_datalist = test_unhealthy_images_path

    batch_size = args.dataset["batch_size"]
    num_workers = args.dataset["num_workers"]


    # transforms
    test_reconstruction_transforms = define_instance(args, "test_reconstruction_transforms")
    test_reconstruction_ds = CacheDataset(data=test_reconstruction_datalist, transform=test_reconstruction_transforms)

    if ddp_bool:
        test_reconstruction_sampler = torch.utils.data.distributed.DistributedSampler(test_reconstruction_ds, num_replicas=world_size, rank=rank)
    else:
        test_reconstruction_sampler = None
    
    test_reconstruction_loader = DataLoader(
        test_reconstruction_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=num_workers, pin_memory=True, sampler=test_reconstruction_sampler
    )

    model = define_instance(args, "network_def").to(device)


    if args.noise["type"] == "simplex":
        simplexObj = simplex.Simplex_CLASS()
        num_train_timesteps = args.noise["num_train_timesteps"]

        scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=num_train_timesteps)
    elif args.noise["type"] == "gaussian":
        num_train_timesteps = args.noise["num_train_timesteps"]

        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
            beta_start=args.noise["beta_start"],
            beta_end=args.noise["beta_end"],)

    if args.diffusion_train["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.diffusion_train["optimizer"]["lr"] * world_size)


    inferer = DiffusionInferer(scheduler)


    model.load_state_dict(torch.load(model_path, map_location=DEVICE_TYPE))
    model.eval()

    @torch.no_grad()
    def my_sample(image, timesteps=100, progress_bar=True, return_first_noisy_image=False):
        
        
        num_infer_timesteps = timesteps #100 # higher number = more noise at first timestep, more denoising steps
        
        infer_scheduler = SimplexDDPMScheduler(num_train_timesteps=num_infer_timesteps)

        all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=infer_scheduler.timesteps.dtype)))

        first_noisy_image = torch.zeros_like(image)

        if progress_bar:
            progress_bar = tqdm(
                zip(infer_scheduler.timesteps, all_next_timesteps),
                total=min(len(infer_scheduler.timesteps), len(all_next_timesteps)),
            )
        else:
            progress_bar = zip(infer_scheduler.timesteps, all_next_timesteps)
                
                
        for t, next_t in progress_bar: # va de num_infer_timesteps à 0
            # 1. predict noise model_output
            diffusion_model = model
            
            model_output = diffusion_model(
                image, timesteps=torch.Tensor((t,)).to(device), context=None
            )
            #inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
            # 2. compute previous image: x_t -> x_t-1
            
            image, _ = infer_scheduler.step(model_output, t, image)

            if t == num_infer_timesteps-1:
                first_noisy_image = image

        if return_first_noisy_image:
            return image, first_noisy_image
        else:
            return image

    # ----------- COMPUTING METRICS -----------

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)

    mse = {noise: [] for noise in NOISE_RANGE} # for each noise level there is a list of mse values
    psnr = {noise: [] for noise in NOISE_RANGE}
    ssim = {noise: [] for noise in NOISE_RANGE}

    for image_batch in tqdm(test_reconstruction_loader):

        test_reconstruction_images = image_batch.to(device)

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            # Perform 5 inferences and average the results
            infered_images = []

            for i, noise_timesteps in enumerate(NOISE_RANGE):

                print(f"inference for {noise_timesteps} noise timesteps")

                infered, first_noisy_images = my_sample(test_reconstruction_images, timesteps=noise_timesteps, progress_bar=False, return_first_noisy_image=True)


                mse[noise_timesteps].append(F.mse_loss(infered, test_reconstruction_images).detach().cpu().numpy().flatten())
                ssim[noise_timesteps].append(np.mean(ssim_metric(test_reconstruction_images, infered).detach().cpu().numpy().flatten()))
                psnr[noise_timesteps].append(psnr_metric(infered, test_reconstruction_images).detach().cpu().numpy().flatten())

    infer_timesteps = NOISE_MIN+NOISE_MAX//2


    for i,(image_batch) in enumerate(test_reconstruction_loader):
        if i>0:break

        test_reconstruction_images = image_batch.to(device)

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            infered_images = []
            infered, first_noisy_images = my_sample(test_reconstruction_images, timesteps=infer_timesteps, progress_bar=False, return_first_noisy_image=True)

    # ----------- PLOT -----------

    metric_result_text = f"With ({NOISE_MIN},{NOISE_MAX}) timesteps noise range, on the whole test_reconstruction_dataset (n={batch_size})\n"
    metric_result_text += f"Mean MSE: {np.mean([np.mean(item) for sublist in mse.values() for item in sublist]):.3f}\n"
    metric_result_text += f"Mean PSNR: {np.mean([np.mean(item) for sublist in psnr.values() for item in sublist]):.3f}\n"
    metric_result_text += f"Mean SSIM: {np.mean([np.mean(item) for sublist in ssim.values() for item in sublist]):.3f}\n"

    fig, axes = plt.subplots(4, 8, figsize=(25, 17), constrained_layout=True)
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
        

        #axes[1, idx*2].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].imshow(first_noisy_image_no_background, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].set_title(f'Noisy {idx+1}, timesteps={infer_timesteps}')
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
        axes[0, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
        axes[1, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot
        axes[2, idx*2+1].set_box_aspect(1)  # Set the aspect ratio of the histogram subplot

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

        pred = infered[idx, 0].unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions   
        text_metrics_orig_and_pred = f"MSE: {F.mse_loss(true, pred).detach().cpu().numpy().mean():.4f}\n"
        text_metrics_orig_and_pred += f"SSIM: {np.mean(ssim_metric(true, pred).detach().cpu().numpy().mean()):.4f}\n"
        text_metrics_orig_and_pred += f"PSNR: {psnr_metric(true, pred).detach().cpu().numpy().mean():.2f}"


        axes[2, idx*2].text(
            -10, 160, text_metrics_orig_and_pred, transform=axes[1, idx*2].transData,
            color='grey', fontsize=12, verticalalignment='center'
        )

    # Add overall title with metric results
    plt.suptitle(f"Healthy reconstruction for {EXPERIMENT_NAME}", fontsize=16)

    plt.figtext(0.0, -0.1, metric_result_text, fontsize=16)

    plt.figtext(0.0, 0.27, "Reconstruction metrics for the whole test_reconstruction dataset", fontsize=16)

    for ax in axes[3, 0:2]: # two merge two subplots
        ax.remove()
    gs = axes[3, 0].get_gridspec()
    axbig1 = fig.add_subplot(gs[3, 0:2])

    # MSE plot
    axbig1.plot(NOISE_RANGE, [np.mean(mse[noise]) for noise in NOISE_RANGE], marker='o', label='MSE')
    axbig1.set_title('Mean Squared Error (MSE)')
    axbig1.set_xlabel('Noise Timesteps')
    axbig1.set_ylabel('MSE')
    axbig1.grid(True)
    axbig1.legend()

    for ax in axes[3, 2:4]:
        ax.remove()
    gs = axes[3, 4].get_gridspec()
    axbig2 = fig.add_subplot(gs[3, 2:4])


    # PSNR plot
    axbig2.plot(NOISE_RANGE, [np.mean(psnr[noise]) for noise in NOISE_RANGE], marker='o', label='PSNR', color='red')
    axbig2.set_title('Peak Signal-to-Noise Ratio (PSNR)')
    axbig2.set_xlabel('Noise Timesteps')
    axbig2.set_ylabel('PSNR')
    axbig2.grid(True)
    axbig2.legend()

    for ax in axes[3, 4:6]:
        ax.remove()
    gs = axes[3, 2].get_gridspec()
    axbig3 = fig.add_subplot(gs[3, 4:6])

    # SSIM plot
    axbig3.plot(NOISE_RANGE, [np.mean(ssim[noise]) for noise in NOISE_RANGE], marker='o', label='SSIM', color='green')
    axbig3.set_title('Structural Similarity Index (SSIM)')
    axbig3.set_xlabel('Noise Timesteps')
    axbig3.set_ylabel('SSIM')
    axbig3.grid(True)
    axbig3.legend()

    fig.delaxes(axes[3,6])
    fig.delaxes(axes[3,7])

    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_metrics_reconstruction.png", transparent=False, dpi=150)

