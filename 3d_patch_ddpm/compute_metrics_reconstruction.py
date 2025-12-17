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
from typing import Dict, List, Optional, Sequence, Tuple
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

def scale_intensity_from_histogram_peak(input_image, target_value=1.0):
    # to be used only on mri images with intensities between 0 and 1
    input_np = input_image.cpu().numpy()

    hist, bin_edges = np.histogram(input_np.flatten(), bins=100, range=(np.max(input_np)/15.0, 0.8))

    peak_value = bin_edges[np.argmax(hist)]

    normalized_image = input_image / peak_value * target_value

    return normalized_image

def _generate_patch_slices(spatial_shape: Sequence[int], patch_size: Sequence[int], overlap: Sequence[int]):


    ranges: List[List[int]] = []

    for dim, size, ov in zip(spatial_shape, patch_size, overlap):
        step = max(size - ov, 1)

        if dim <= size:
            coords = [0]
        else:
            coords = list(range(0, max(dim - size, 0) + 1, step))
            if coords[-1] != dim - size:
                coords.append(dim - size)
        ranges.append(coords)
        
    for h in ranges[0]:
        for w in ranges[1]:
            for d in ranges[2]:
                yield (slice(h, h + patch_size[0]), slice(w, w + patch_size[1]), slice(d, d + patch_size[2]))



@torch.no_grad()
def my_sample(model, device, noise_type, simplexObj, image, infer_scheduler, timesteps, return_intermediates=False):

    if noise_type == "simplex":
        noise = simplex_ddpm.generate_simplex_noise(simplexObj, image.shape, normalize=False).to(device)
    if noise_type == "gaussian":
        noise = torch.randn(image.shape).to(device)


    timesteps_list = torch.Tensor([timesteps for a in range(image.shape[0])]).to(image.device).long()

    image = infer_scheduler.add_noise(image, noise, timesteps_list).to(device) #TODO


    intermediates = []
    intermediates_step = 20

            
    for t in tqdm(range(timesteps, 0, -1)): # va de timesteps à 0
        
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


def _create_patch_weight(patch_size: Sequence[int], sigma_scale: float = 0.125) -> torch.Tensor:
    """
    Create a 3D Gaussian weight map that gives more importance to the center of the patch.
    This helps blend overlapping patches smoothly and eliminates seam artifacts.
    """
    weight = torch.ones(patch_size)
    
    for dim in range(3):
        size = patch_size[dim]

        # Create 1D Gaussian-like weight using cosine tapering
        # This gives weight 1 at center and smoothly decreases to ~0.5 at edges
        coords = torch.linspace(0, 1, size)

        # Cosine window (Hann-like): smooth transition from edges to center
        window = 0.5 * (1 - torch.cos(2 * np.pi * coords))

        #window = window * 0.5 + 0.5  # Range: 0.5 to 1.0
        window = window * 0.9 + 0.1  # Range: 0.9 to 1.0

        # Reshape for broadcasting
        shape = [1, 1, 1]
        shape[dim] = size
        window = window.view(shape)
        
        weight = weight * window
    
    return weight


def _run_patchwise_test(
    volume: torch.Tensor,
    patch_size: Sequence[int],
    overlap: Sequence[int],
    patch_batch_size: int,
    noise_type: str,
    simplexObj,
    model,
    infer_scheduler,
    num_timesteps: int,
    device,
):
    aggregator_pred = torch.zeros_like(volume, dtype=torch.float32)
    weight_sum = torch.zeros_like(volume, dtype=torch.float32)
    patch_queue: List[torch.Tensor] = []
    slice_queue: List[Tuple[slice, slice, slice]] = []
    total_patches = 0
    
    # Create weight map for smooth blending
    patch_weight = _create_patch_weight(patch_size).to(device)
    """
    Same as _run_patchwise_inference but goes all the way to the fully denoised image
    Uses weighted averaging to eliminate seam artifacts at patch boundaries.
    """

    def _flush_queue():
        nonlocal total_patches

        if not patch_queue:
            return
        
        batch_tensor = torch.cat(patch_queue, dim=0) # transforms all the patches into a single batch tensor

        preds = my_sample(model, device, noise_type, simplexObj, batch_tensor, infer_scheduler, num_timesteps)

        patch_count = batch_tensor.shape[0]
        total_patches += patch_count

        for idx, patch_slices in enumerate(slice_queue):
            target_slice = (slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])
            
            # Apply weighted contribution instead of simple addition
            aggregator_pred[target_slice] += preds[idx].unsqueeze(0).float() * patch_weight
            weight_sum[target_slice] += patch_weight  # Accumulate weights instead of counts


        patch_queue.clear()
        slice_queue.clear()

    for patch_slices in _generate_patch_slices(volume.shape[-3:], patch_size, overlap): # goes through the slices that define each patch

        patch = volume[(slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])] # extracts the patch using the slices

        patch_queue.append(patch) # patch_queue stores all the patches for the current volume batch
        slice_queue.append(patch_slices)

        if len(patch_queue) >= patch_batch_size: # makes sure there aren't too many patches at one time (memory issues)
            _flush_queue() # does the inference and computes loss

    _flush_queue()

    weight_sum = torch.clamp(weight_sum, min=1e-8)
    stitched_pred = aggregator_pred / weight_sum  # Weighted average for smooth blending

    return stitched_pred


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
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_healthy_reconstruction/" # final anomaly maps with best params
    os.makedirs(ANOMALY_MAPS_DIR, exist_ok=True)

    IMAGE_SIZE = args.image_size

    model_path = f"{args.root_dir}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/{SUB_EXPERIMENT_NAME}_best_model.pth"

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)

    NOISE_MIN = int(args.compute_metrics_reconstruction["noise_rate_min"]*args.noise["num_timesteps_full_noise"])
    NOISE_MAX = int(args.compute_metrics_reconstruction["noise_rate_max"]*args.noise["num_timesteps_full_noise"])+1
    #NOISE_RANGE = range(NOISE_MIN,NOISE_MAX,args.compute_metrics_reconstruction["noise_timesteps_interval"])
    NOISE_RANGE = range(NOISE_MAX,NOISE_MIN,-args.compute_metrics_reconstruction["noise_timesteps_interval"]) # reverse to see more noisy images first

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

    infer_patch_size = args.patch_size
    patch_overlap = args.dataset["patch_overlap"]

    patch_infer_batch_size = args.dataset["batch_size"]

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

    simplexObj = None

    if args.noise["type"] == "simplex":
        simplexObj = simplex.Simplex_CLASS()
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], 
                                                            schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], 
                                                            persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])




    # ----------- COMPUTING METRICS -----------

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    ssim_metric_3d = SSIMMetric(spatial_dims=3, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)

    mse = {noise: [] for noise in NOISE_RANGE} # for each noise level there is a list of mse values
    psnr = {noise: [] for noise in NOISE_RANGE}
    ssim = {noise: [] for noise in NOISE_RANGE}
    #lpips_dict = {noise: [] for noise in NOISE_RANGE}

    #loss_fn_lpips = lpips.LPIPS(net='alex').to(device) # Higher means further/more different. Lower means more similar.

    full_volume_test = False

    for i, image_batch in tqdm(enumerate(test_reconstruction_loader)):

        test_reconstruction_images = image_batch.to(device)

        test_reconstruction_images = test_reconstruction_images[..., args.slice_indexes_start:args.slice_indexes_end]
                    
        volumes = test_reconstruction_images.shape[0]

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            with torch.no_grad():

                for n, noise_timesteps in enumerate(NOISE_RANGE):                

                    infered_batch = torch.zeros_like(test_reconstruction_images)

                    for idx in range(volumes): 

                        volume = test_reconstruction_images[idx : idx + 1] #TODO Here it does it volume per volume, any way to run the inference by batch?
                        volume = volume.to(device)
                        
                        stitched_pred = _run_patchwise_test(
                            volume,
                            infer_patch_size,
                            patch_overlap,
                            patch_infer_batch_size,
                            args.noise["type"],
                            simplexObj,
                            model,
                            infer_scheduler,
                            noise_timesteps,
                            device,
                        )
                        infered = torch.clamp(scale_intensity_from_histogram_peak(stitched_pred, 2.0/7.0), 0.0, 1.0)
                        infered_batch[idx : idx + 1] = infered

                        # save the anomaly map (raw, no absolute value) if noise_timesteps==100 for adc or 150 for flair
                        if ("adc" in args.dataset["name"] and noise_timesteps == 100) or ("flair" in args.dataset["name"] and noise_timesteps == 150):
                            # Convert to numpy and save as NIfTI
                            anomaly_map_np = (infered - volume).cpu().numpy().squeeze()
                            anomaly_map_nifti = nib.Nifti1Image(anomaly_map_np, affine=np.eye(4))
                            patient_id = os.path.basename(test_reconstruction_datalist[i*batch_size+idx]).split('.')[0]
                            anomaly_map_path = os.path.join(ANOMALY_MAPS_DIR, f"anomaly_map_noise_{noise_timesteps}_{patient_id}.nii.gz")
                            nib.save(anomaly_map_nifti, anomaly_map_path)
                    
                    mse[noise_timesteps].append(F.mse_loss(infered_batch, test_reconstruction_images).detach().cpu().numpy().flatten())
                    #tprint(f"ssim for noise timesteps {noise_timesteps} on batch {i} of test reconstruction data: {ssim_metric_3d(test_reconstruction_images, infered_batch).detach().cpu().numpy().flatten()}")
                    #tprint(f"batch size: {test_reconstruction_images.shape[0]}")
                    ssim[noise_timesteps].append(ssim_metric_3d(test_reconstruction_images, infered_batch).detach().cpu().numpy().flatten())
                    psnr[noise_timesteps].append(psnr_metric(infered_batch, test_reconstruction_images).detach().cpu().numpy().flatten())
                    tprint(f"Computed metrics for noise timesteps {noise_timesteps} on batch {i} of test reconstruction data.")


                    #lpips_dict[noise_timesteps].append(np.mean(lpips_volume))
        
        tprint(f"Processed batch {i} of test reconstruction data.")
        tprint(f"Total processed volumes: {i * test_reconstruction_images.shape[0]}.")
        for noise_timesteps in NOISE_RANGE:
            tprint(f" Noise timesteps: {noise_timesteps}: MSE: {np.mean(mse[noise_timesteps]):.4f}, PSNR: {np.mean(psnr[noise_timesteps]):.2f}, SSIM: {np.mean(ssim[noise_timesteps]):.4f}")
        

    # ----------- VISUALIZATION OF A BATCH -----------
    infer_timesteps_visualize = int(args.compute_metrics_reconstruction["noise_rate_visualize"]*args.noise["num_timesteps_full_noise"])


    for i,(image_batch) in enumerate(test_reconstruction_loader):
        if i>0:break

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
        

        #axes[1, idx*2].imshow(noisy_image, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].imshow(first_noisy_image_no_background, cmap='gray', vmin=-1, vmax=1)
        axes[1, idx*2].set_title(f'Noisy {idx+1}, {args.compute_metrics_reconstruction["noise_rate_visualize"]*100}% noise (timesteps={infer_timesteps_visualize})')
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


    plt.figtext(0.04, 0.04, f"Reconstruction metrics for the whole test_reconstruction dataset (std error bars)\nFor {args.compute_metrics_reconstruction['noise_rate_max']*100}% noise:\nPSNR: {np.mean(psnr[NOISE_RANGE[-1]]):.2f} ± {np.std(psnr[NOISE_RANGE[-1]]):.2f}\nSSIM: {np.mean(ssim[NOISE_RANGE[-1]]):.4f} ± {np.std(ssim[NOISE_RANGE[-1]]):.4f}\nMSE: {np.mean(mse[NOISE_RANGE[-1]]):.4f} ± {np.std(mse[NOISE_RANGE[-1]]):.4f}\nLPIPS: {np.mean(lpips_dict[NOISE_RANGE[-1]]):.4f} ± {np.std(lpips_dict[NOISE_RANGE[-1]]):.4f}", fontsize=16)


    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_metrics_reconstruction.png", transparent=False, dpi=150)

