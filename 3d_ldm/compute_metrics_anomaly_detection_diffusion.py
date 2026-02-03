"""
This file only works for 3D slice by slice inference.
"""

import os
import glob
import sys
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
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism, StrEnum, first
from monai.inferers import LatentDiffusionInferer
from torch.amp import autocast
from tqdm import tqdm

import nibabel as nib

from monai.networks.schedulers import DDPMScheduler

from typing import Union

import pandas as pd

import AnoDDPM.simplex as simplex

import utils.custom_transforms as custom_transforms

import utils.simplex_ddpm as simplex_ddpm
from utils.utils import *
from make_anomaly_maps import make_anomaly_maps

from monai.metrics import compute_iou, DiceMetric, compute_hausdorff_distance


from scipy.ndimage import median_filter, binary_erosion, binary_dilation
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

DEVICE_TYPE = "cuda:0"

def process_anomaly_file(anomaly_file, anomaly_maps_folder, masks_folder, 
                             thresholds_to_try, median_filter_sizes_to_try, 
                             erosion_dilation_iterations_to_try):
        """Process a single anomaly file and return scores for all parameter combinations."""
        
        dm = DiceMetric(reduction="sum")

        local_iou_scores = {}
        local_dice_scores = {}
        
        
        #tprint(f"starting processing {anomaly_file}")

        anomaly_map_nib = nib.load(os.path.join(anomaly_maps_folder, anomaly_file))
        anomaly_map = anomaly_map_nib.get_fdata()
        timesteps = int(anomaly_file.split('.')[0].split('_')[-1])

        mask_nib = nib.load(os.path.join(masks_folder, f"{anomaly_file.split('_')[0]}.nii.gz"))
        mask = torch.from_numpy(mask_nib.get_fdata())
        mask = mask.unsqueeze(0).unsqueeze(0)  # B1HWD
        
        # Pre-compute filtered versions
        filtered_maps = {-1: torch.from_numpy(anomaly_map)}
        for median_filter_size in median_filter_sizes_to_try:
            if median_filter_size > 0:
                filtered_np = median_filter(anomaly_map, size=median_filter_size)
                filtered_maps[median_filter_size] = torch.from_numpy(filtered_np)
        
        
        #tprint(f"computed median filtered for {anomaly_file}")

        # Iterate through all combinations efficiently
        for median_filter_size in median_filter_sizes_to_try:
            final_anomaly_map = filtered_maps[median_filter_size]
            
            
            #tprint(f"median filter size {median_filter_size} for {anomaly_file}")
            #tprint(f"next: thresholds to try: {thresholds_to_try}")

            for threshold in thresholds_to_try:
                ano_segmentation_base = (final_anomaly_map > threshold)

                
                #tprint(f"threshold {threshold} for {anomaly_file}")
                
                for erosion_dilation_iterations in erosion_dilation_iterations_to_try:

                    
                    #tprint(f"erosion_dilation {erosion_dilation_iterations} for {anomaly_file}")

                    if erosion_dilation_iterations > 0:
                        ano_segmentation_np = ano_segmentation_base.cpu().numpy()
                        
                        ano_segmentation_np = binary_erosion(ano_segmentation_np, iterations=erosion_dilation_iterations)
                        ano_segmentation_np = binary_dilation(ano_segmentation_np, iterations=erosion_dilation_iterations)

                        ano_segmentation = torch.from_numpy(ano_segmentation_np)
                    else:
                        ano_segmentation = ano_segmentation_base

                    #tprint(f"computing iou score for {anomaly_file} ..")
                    # ano_segmentation and masks must be in format : B1HWD
                    ano_segmentation = ano_segmentation.unsqueeze(0).unsqueeze(0)

                    # Compute metrics
                    iou_score = compute_iou(ano_segmentation, mask)
                    flattened_iou_score = iou_score.cpu().numpy().flatten()
                    flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)]

                    #tprint(f"computing dice score for {anomaly_file} ..")

                    dice_score = dm(ano_segmentation, mask).cpu().numpy().flatten()
                    dice_score = dice_score[~np.isnan(dice_score)]

                    # Store results
                    idx = (timesteps, threshold, median_filter_size, erosion_dilation_iterations)
                    local_iou_scores[idx] = np.sum(flattened_iou_score)
                    local_dice_scores[idx] = np.sum(dice_score)
        
        #tprint(f"finished for {anomaly_file} ..")

        return local_iou_scores, local_dice_scores

def compute_select_params_multithreaded(args, anomaly_maps_folder, masks_folder, total_nb_images, num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try):
    # takes an input folder with all the saved infered 3D anomaly maps 
    # and tests different combinations of post-processing and returns and saves a table with all the scores for each set of parameters

    tprint("launching compute_select_parasms_multithreaded")
    

    # Create the MultiIndex from timesteps, thresholds, median filter sizes, erosion and dilation iterations
    iou_scores_midx = pd.MultiIndex.from_product([num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try])
    iou_scores_df = pd.DataFrame(index=iou_scores_midx, columns=["IOU"])
    iou_scores_df.fillna(0.0, inplace=True)
    iou_scores_df.index.names = ['timesteps', 'threshold', 'median_filter_size', 'erosion_dilation_iterations']

    dice_scores_midx = pd.MultiIndex.from_product([num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try])
    dice_scores_df = pd.DataFrame(index=dice_scores_midx, columns=["DICE"])
    dice_scores_df.fillna(0.0, inplace=True)
    dice_scores_df.index.names = ['timesteps', 'threshold', 'median_filter_size', 'erosion_dilation_iterations']


    anomaly_files = [entry.name for entry in os.scandir(anomaly_maps_folder) if entry.is_file() and entry.name.endswith(".nii.gz")]
    if not anomaly_files:
        raise RuntimeError(f"No anomaly map files found in '{anomaly_maps_folder}'.")

    process_func = partial(
        process_anomaly_file,
        anomaly_maps_folder=anomaly_maps_folder,
        masks_folder=masks_folder,
        thresholds_to_try=thresholds_to_try,
        median_filter_sizes_to_try=median_filter_sizes_to_try,
        erosion_dilation_iterations_to_try=erosion_dilation_iterations_to_try,
    )

    max_workers = min(32, mp.cpu_count())
    ctx = mp.get_context("spawn")

    if len(anomaly_files) == 1 or max_workers == 1:
        results = [process_func(file_name) for file_name in anomaly_files]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(process_func, file_name): file_name for file_name in anomaly_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing anomaly maps"):
                results.append(future.result())

    """# Prepare partial function with fixed parameters (This one gets stuck with no errors)
    process_func = partial(process_anomaly_file, 
                          anomaly_maps_folder=anomaly_maps_folder,
                          masks_folder=masks_folder,
                          thresholds_to_try=thresholds_to_try,
                          median_filter_sizes_to_try=median_filter_sizes_to_try,
                          erosion_dilation_iterations_to_try=erosion_dilation_iterations_to_try)
    
    # Use multiprocessing to process files in parallel
    anomaly_files = [entry.name for entry in os.scandir(anomaly_maps_folder) if entry.is_file() and entry.name.endswith(".nii.gz")]
    if not anomaly_files:
        raise RuntimeError(f"No anomaly map files found in '{anomaly_maps_folder}'.")
    
    max_workers = min(4, cpu_count())
    num_processes = min(max_workers, len(anomaly_files))
    
    if num_processes <= 1:
        results = [process_func(file_name) for file_name in anomaly_files]
    else:
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_func, anomaly_files)"""
    
    # try without multiprocessing
    """results = []

    for anomaly_file in os.listdir(anomaly_maps_folder):

        tprint(f"processing first anomaly file {anomaly_file}")

        local_iou_scores,local_dice_scores = process_anomaly_file(anomaly_file, anomaly_maps_folder, masks_folder, 
                             thresholds_to_try, median_filter_sizes_to_try, 
                             erosion_dilation_iterations_to_try)
        results.append((local_iou_scores, local_dice_scores))"""
    
    tprint("multiprocesses all finished")

    # Aggregate results from all processes
    for local_iou_scores, local_dice_scores in results:
        for idx, iou_val in local_iou_scores.items():
            if np.isnan(iou_scores_df.loc[idx, "IOU"]):
                iou_scores_df.loc[idx, "IOU"] = iou_val
                dice_scores_df.loc[idx, "DICE"] = local_dice_scores[idx]
            else:
                iou_scores_df.loc[idx, "IOU"] += iou_val
                dice_scores_df.loc[idx, "DICE"] += local_dice_scores[idx]
    
    # Divide everything by the number of images (moved outside the loop)
    iou_scores_df = iou_scores_df / total_nb_images
    dice_scores_df = dice_scores_df / total_nb_images

    return iou_scores_df, dice_scores_df

def compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR, infer_scheduler, image_loader, image_paths, mask_loader, infer_timesteps, threshold, median_filter_size, erosion_dilation_iterations):
        """
        input:
            image_loader: DataLoader for the anomaly images
            mask_loader: DataLoader for the anomaly masks
            timesteps: number of noise timesteps to use for inference
            threshold: threshold to use for anomaly segmentation
            median_filter_size: size of the median filter to apply to the anomaly map, use -1 or None to not apply any filtering
            erosion_iterations: number of erosion iterations to apply to the anomaly segmentation, use 0 to not apply any erosion
            dilation_iterations: number of dilation iterations to apply to the anomaly segmentation, use 0 to not apply any dilation
        output:
            mean_iou: mean IOU score
            std_iou: std IOU score
            mean_dice: mean DICE score
            std_dice: std DICE score
        """
        dm = DiceMetric(reduction="sum")
        
        iou_scores = []
        dice_scores = []

        basic_affine = nib.load(image_paths[0]).affine

        no_masks = False
        if mask_loader is None:
            mask_loader = image_loader # hack so the for loop works
            no_masks = True
        #tprint(f"launching comute metrics with timesteps={timesteps}, threshold={threshold}, median_filter_size={median_filter_size}, erosion_dilation_iterations={erosion_dilation_iterations}")
        #tprint(f"len(image_loader.dataset)={len(image_loader.dataset)}")
        #tprint(f"len(mask_loader.dataset)={len(mask_loader.dataset)}")
       
       # Compute Scaling factor
        # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
        # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
        # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
        # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
        # and the results will not differ from those obtained when it is not used._

        with torch.no_grad():
            with autocast("cuda", enabled=True):
                check_data = first(image_loader)
                z = autoencoder.encode_stage_2_inputs(check_data.to(device))

        scale_factor = 1 / torch.std(z)
       
       # for every batch
        for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice

            test_images = image_batch.to(device)
            test_masks = mask_batch.to(device)
            test_masks[test_masks>0.5] = 1.0
            test_masks[test_masks<=0.5] = 0.0

            with torch.no_grad():
                with autocast(device_type=DEVICE_TYPE, enabled=True):
                    
                    latents = autoencoder.encode_stage_2_inputs(test_images)    

                    # Add noise to latents
                    noise = torch.randn_like(latents).to(device)
                    timesteps = torch.randint(0, infer_timesteps, (latents.shape[0],), device=device).long()
                    noisy_latents = infer_scheduler.add_noise(latents, noise, timesteps)
                    
                    # Denoise completely using the UNet
                    infer_scheduler.set_timesteps(infer_scheduler.num_train_timesteps)
                    current_latents = noisy_latents * scale_factor
                    
                    for t in tqdm(range(infer_timesteps-1, -1, -1)):
                        noise_pred = unet(current_latents, timesteps=torch.tensor([t], device=device).expand(latents.shape[0]))
                        current_latents, _ = infer_scheduler.step(noise_pred, t, current_latents)
                    
                    # Decode the denoised latents
                    current_latents = current_latents / scale_factor

                    reconstructed_images = autoencoder.decode(current_latents)
                    normalized_reconstructed_images = torch.zeros_like(reconstructed_images)
                    for volume in range(reconstructed_images.shape[0]):
                        normalized_reconstructed_images[volume] = scale_intensity_from_histogram_peak(reconstructed_images[volume], target_value=2.0/7.0)

                    # make the anomaly map (difference between infered and original)
                    final_anomaly_map = torch.abs(normalized_reconstructed_images - test_images)

                    # save the anomaly maps if specified
                    if args.dataset["save_anomaly_maps"]:
                        for idx_in_batch in range(final_anomaly_map.shape[0]):
                            image_id = i*test_images.shape[0] + idx_in_batch
                            image_name = os.path.basename(image_paths[image_id])
                            nib.save(nib.Nifti1Image(final_anomaly_map[idx_in_batch].squeeze().cpu().numpy(), basic_affine), ANOMALY_MAPS_DIR+f"{image_name}")

                    # apply median filter if specified
                    if median_filter_size is not None and median_filter_size > 0:
                        final_anomaly_map_np = final_anomaly_map.cpu().numpy()
                        for b in range(final_anomaly_map_np.shape[0]):
                            final_anomaly_map_np[b] = median_filter(final_anomaly_map_np[b], size=median_filter_size)
                        final_anomaly_map = torch.from_numpy(final_anomaly_map_np).to(device)
                    
                    

            #tprint(f"unprocessed anomaly map shape: {final_anomaly_map.shape}")

            if not no_masks:

                # make the segmentation map with threshold
                ano_segmentation = final_anomaly_map > threshold

                # perform erosion if specified
                if erosion_dilation_iterations > 0:
                    ano_segmentation_np = ano_segmentation.cpu().numpy()
                    for b in range(ano_segmentation_np.shape[0]):
                        ano_segmentation_np[b,0] = binary_erosion(ano_segmentation_np[b,0], iterations=erosion_dilation_iterations)
                        ano_segmentation_np[b,0] = binary_dilation(ano_segmentation_np[b,0], iterations=erosion_dilation_iterations)
                    ano_segmentation = torch.from_numpy(ano_segmentation_np).to(device)
                

                iou_score = compute_iou(ano_segmentation, test_masks)
                flattened_iou_score = iou_score.cpu().numpy().flatten()
                #tprint(f"flattened_iou_score (before removing nan values): {flattened_iou_score}")
                flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values

                iou_scores.append(flattened_iou_score)

                dice_score = dm(ano_segmentation, test_masks).cpu().numpy().flatten()
                #tprint(f"flattened_dice_score (before removing nan values): {dice_score}")
                dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values
                dice_scores.append(dice_score)

        if no_masks:
            return

        mean_iou = np.mean(np.concatenate(iou_scores))
        std_iou = np.std(np.concatenate(iou_scores))

        mean_dice = np.mean(np.concatenate(dice_scores))
        std_dice = np.std(np.concatenate(dice_scores))

        return mean_iou, std_iou, mean_dice, std_dice


def show_summary_figure(args, device, autoencoder, unet, infer_scheduler, image_loader, mask_loader, infer_timesteps, median_filter_size, threshold, erosion_dilation_iterations, metrics_result_text, ROOT_DIR, EXPERIMENT_NAME, SUB_EXPERIMENT_NAME):

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with autocast("cuda", enabled=True):
            check_data = first(image_loader)
            z = autoencoder.encode_stage_2_inputs(check_data.to(device))

    scale_factor = 1 / torch.std(z)
    
    for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice
        if i>0:break

        test_anomaly_images = image_batch[..., image_batch.shape[-1]//2].to(device)
        test_anomaly_masks = mask_batch[..., mask_batch.shape[-1]//2].to(device)
        
        test_anomaly_masks[test_anomaly_masks>0.5] = 1.0
        test_anomaly_masks[test_anomaly_masks<=0.5] = 0.0

        with torch.no_grad():
            with autocast(device_type=DEVICE_TYPE, enabled=True):

                latents = autoencoder.encode_stage_2_inputs(image_batch.to(device))    

                # Add noise to latents
                noise = torch.randn_like(latents).to(device)
                timesteps = torch.randint(0, infer_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = infer_scheduler.add_noise(latents, noise, timesteps)
                
                # Denoise completely using the UNet
                infer_scheduler.set_timesteps(infer_scheduler.num_train_timesteps)
                current_latents = noisy_latents * scale_factor
                
                for t in tqdm(range(infer_timesteps-1, -1, -1)):
                    noise_pred = unet(current_latents, timesteps=torch.tensor([t], device=device).expand(latents.shape[0]))
                    current_latents, _ = infer_scheduler.step(noise_pred, t, current_latents)
                
                # Decode the denoised latents
                current_latents = current_latents / scale_factor

                infered_image = autoencoder.decode(current_latents)

                for volume in range(infered_image.shape[0]):
                    infered_image[volume] = torch.clamp(scale_intensity_from_histogram_peak(infered_image[volume:volume+1], 2.0/7.0), 0.0, 1.0)
                #infered_image = torch.clamp(scale_intensity_from_histogram_peak(infered_image, 2.0/7.0), 0.0, 1.0)
        
        

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
        #print(infered_image.shape)
        infered_image_slice = infered_image[idx, 0].cpu().numpy()
        infered_image_slice = infered_image_slice[..., infered_image_slice.shape[-1]//2]
        
        axes[1, idx*2].imshow(infered_image_slice, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].set_title(f'Inferred {idx+1}')
        axes[1, idx*2].axis('off')

        axes[1, idx*2+1].hist(infered_image_slice[infered_image_slice>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[1, idx*2+1].set_ylim(0, 2000)
        axes[1, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Difference images
        difference_image = np.abs(original_image - infered_image_slice)
        # apply median filter if specified
        if median_filter_size is not None and median_filter_size > 0:
            final_anomaly_map_np = difference_image
            for b in range(final_anomaly_map_np.shape[0]):
                final_anomaly_map_np[b] = median_filter(final_anomaly_map_np[b], size=median_filter_size)
            final_anomaly_map = final_anomaly_map_np
        else:
            final_anomaly_map = difference_image
        
        axes[2, idx*2].imshow(final_anomaly_map, cmap='jet', vmin=0, vmax=1)
        axes[2, idx*2].set_title(f'Difference {idx+1}, median filter size: {median_filter_size}')
        axes[2, idx*2].axis('off')

        axes[2, idx*2+1].hist(final_anomaly_map[final_anomaly_map>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[2, idx*2+1].set_ylim(0, 2000)
        axes[2, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Thresholded difference images
        thresholded_difference_image = (final_anomaly_map > threshold)#.astype(np.float32)
        ano_segmentation_np = thresholded_difference_image
        """if erosion_dilation_iterations_visualize > 0: #TODO
            ano_segmentation_np = thresholded_difference_image
            ano_segmentation_np = binary_erosion(ano_segmentation_np, iterations=erosion_dilation_iterations_visualize).astype(ano_segmentation_np.dtype)
            ano_segmentation_np = binary_dilation(ano_segmentation_np, iterations=erosion_dilation_iterations_visualize).astype(ano_segmentation_np.dtype)"""

        axes[3, idx*2].imshow(ano_segmentation_np, cmap='gray', vmin=0, vmax=1)
        axes[3, idx*2].set_title(f'Thresholded Difference {idx+1}, erosion-dilation steps: {erosion_dilation_iterations}')
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

    plt.suptitle(f"Anomaly detection for {EXPERIMENT_NAME}, LDM 3D volumes", fontsize=16)

    plt.figtext(0.0, 0.0, metrics_result_text, fontsize=14)


    plt.savefig(f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}_{args.dataset['test']}_metrics_anomaly_detection_ldm_3d_volumes.png", transparent=False, dpi=150)


def launch_compute_metrics_anomaly_detection_diffusion(args):
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
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    
    ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params/"
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/" # final anomaly maps with best params
    os.makedirs(ANOMALY_MAPS_DIR_SELECT_PARAMS, exist_ok=True)
    os.makedirs(ANOMALY_MAPS_DIR, exist_ok=True)


    torch.backends.cudnn.benchmark = False #True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)

    NOISE_MIN = int(args.compute_metrics_reconstruction["noise_rate_min"]*args.noise["num_timesteps_full_noise"])
    NOISE_MAX = int(args.compute_metrics_reconstruction["noise_rate_max"]*args.noise["num_timesteps_full_noise"])+1
    NOISE_INTERVAL = int(args.compute_metrics_reconstruction["noise_timesteps_interval"])

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
    
    num_workers = args.autoencoder_train["num_workers"]
    ano_batch_size = args.autoencoder_train["batch_size"]

    # -------------------- define the data --------------------

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


        test_anomaly_transforms = define_instance(args, "val_transforms")
        test_anomaly_ds = CacheDataset(data=test_anomaly_images, transform=test_anomaly_transforms)

        test_anomaly_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_anomaly_ds[:len(test_anomaly_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_images_select_params = test_anomaly_images[:len(test_anomaly_ds)//2]

        test_anomaly_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_anomaly_ds[len(test_anomaly_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_images_metrics = test_anomaly_images[len(test_anomaly_ds)//2:]

        test_masks_ds = CacheDataset(data=test_masks, transform=test_masks_transforms)
        
        test_masks_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_masks_ds[:len(test_masks_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_masks_ds[len(test_masks_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

    if args.dataset["test"] == "isles": #TODO renommer les massks et les images pour qu'ils aient exactement le meme nom
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
    elif args.dataset["test"] == "soop":
        
        large_group = ['sub-1010', 'sub-1013', 'sub-1015', 'sub-1032', 'sub-1035', 'sub-1039', 'sub-1041', 'sub-1045', 'sub-1046', 'sub-1071', 'sub-1073', 'sub-1086', 'sub-1102', 'sub-1107', 'sub-1115', 'sub-113', 'sub-114', 'sub-1149', 'sub-1150', 'sub-116', 'sub-1164', 'sub-1165', 'sub-118', 'sub-1200', 'sub-1204', 'sub-1209', 'sub-1213', 'sub-1215', 'sub-1223', 'sub-1227', 'sub-1232', 'sub-1246', 'sub-1258', 'sub-127', 'sub-1280', 'sub-1282', 'sub-1283', 'sub-1285', 'sub-1292', 'sub-1305', 'sub-1306', 'sub-1309', 'sub-1312', 'sub-1314', 'sub-1320', 'sub-1323', 'sub-135', 'sub-1354', 'sub-1355', 'sub-1358', 'sub-1364', 'sub-1366', 'sub-1369', 'sub-1373', 'sub-1379', 'sub-1382', 'sub-1386', 'sub-1395', 'sub-1409', 'sub-1410', 'sub-1413', 'sub-1422', 'sub-1432', 'sub-1445', 'sub-1447', 'sub-1475', 'sub-1478', 'sub-1480', 'sub-1483', 'sub-1485', 'sub-1488', 'sub-1507', 'sub-1508', 'sub-1511', 'sub-1517', 'sub-1552', 'sub-1554', 'sub-1555', 'sub-1569', 'sub-1598', 'sub-1612', 'sub-1634', 'sub-1637', 'sub-1656', 'sub-1670', 'sub-1677', 'sub-1719', 'sub-1725', 'sub-1727', 'sub-1736', 'sub-174', 'sub-177', 'sub-185', 'sub-190', 'sub-196', 'sub-198', 'sub-2', 'sub-221', 'sub-235', 'sub-241', 'sub-247', 'sub-249', 'sub-260', 'sub-262', 'sub-264', 'sub-278', 'sub-284', 'sub-294', 'sub-3', 'sub-303', 'sub-314', 'sub-321', 'sub-326', 'sub-335', 'sub-338', 'sub-339', 'sub-341', 'sub-343', 'sub-345', 'sub-359', 'sub-360', 'sub-366', 'sub-370', 'sub-374', 'sub-386', 'sub-398', 'sub-400', 'sub-401', 'sub-412', 'sub-42', 'sub-422', 'sub-432', 'sub-433', 'sub-443', 'sub-446', 'sub-447', 'sub-457', 'sub-463', 'sub-464', 'sub-466', 'sub-47', 'sub-494', 'sub-498', 'sub-501', 'sub-505', 'sub-512', 'sub-517', 'sub-521', 'sub-523', 'sub-525', 'sub-529', 'sub-53', 'sub-530', 'sub-539', 'sub-543', 'sub-56', 'sub-563', 'sub-572', 'sub-613', 'sub-620', 'sub-631', 'sub-634', 'sub-638', 'sub-651', 'sub-652', 'sub-661', 'sub-682', 'sub-692', 'sub-694', 'sub-698', 'sub-699', 'sub-707', 'sub-719', 'sub-723', 'sub-724', 'sub-751', 'sub-754', 'sub-760', 'sub-761', 'sub-768', 'sub-776', 'sub-789', 'sub-79', 'sub-791', 'sub-8', 'sub-803', 'sub-806', 'sub-82', 'sub-823', 'sub-826', 'sub-843', 'sub-844', 'sub-845', 'sub-858', 'sub-861', 'sub-865', 'sub-866', 'sub-873', 'sub-877', 'sub-881', 'sub-896', 'sub-917', 'sub-937', 'sub-939', 'sub-942', 'sub-946', 'sub-95', 'sub-952', 'sub-959', 'sub-960', 'sub-968', 'sub-990']
        medium_group = ['sub-100', 'sub-1011', 'sub-1014', 'sub-1016', 'sub-1018', 'sub-102', 'sub-1024', 'sub-103', 'sub-1052', 'sub-1054', 'sub-1055', 'sub-1056', 'sub-1057', 'sub-106', 'sub-1064', 'sub-1075', 'sub-1076', 'sub-1096', 'sub-110', 'sub-1101', 'sub-1105', 'sub-1106', 'sub-1113', 'sub-1118', 'sub-1119', 'sub-112', 'sub-1120', 'sub-1127', 'sub-1128', 'sub-1130', 'sub-1136', 'sub-1140', 'sub-1144', 'sub-1147', 'sub-1148', 'sub-1154', 'sub-1157', 'sub-1163', 'sub-1182', 'sub-1183', 'sub-1186', 'sub-1189', 'sub-1193', 'sub-1198', 'sub-1202', 'sub-1211', 'sub-1212', 'sub-1217', 'sub-122', 'sub-1229', 'sub-123', 'sub-1234', 'sub-1237', 'sub-1239', 'sub-124', 'sub-1242', 'sub-1244', 'sub-1248', 'sub-1260', 'sub-1266', 'sub-128', 'sub-1281', 'sub-129', 'sub-1291', 'sub-1296', 'sub-1297', 'sub-1301', 'sub-131', 'sub-1310', 'sub-1319', 'sub-1324', 'sub-1326', 'sub-1330', 'sub-1331', 'sub-1332', 'sub-1338', 'sub-1346', 'sub-1347', 'sub-1348', 'sub-1349', 'sub-1352', 'sub-1363', 'sub-1370', 'sub-1374', 'sub-138', 'sub-1380', 'sub-1388', 'sub-1396', 'sub-1404', 'sub-1408', 'sub-1415', 'sub-1417', 'sub-1423', 'sub-1427', 'sub-1429', 'sub-1438', 'sub-1440', 'sub-1443', 'sub-1446', 'sub-1449', 'sub-145', 'sub-1450', 'sub-146', 'sub-1463', 'sub-1466', 'sub-147', 'sub-148', 'sub-1489', 'sub-1490', 'sub-1494', 'sub-1496', 'sub-1501', 'sub-1503', 'sub-1506', 'sub-1509', 'sub-1514', 'sub-1518', 'sub-1519', 'sub-1521', 'sub-1522', 'sub-1523', 'sub-1525', 'sub-1541', 'sub-1545', 'sub-1548', 'sub-155', 'sub-1550', 'sub-1556', 'sub-1557', 'sub-156', 'sub-1562', 'sub-1567', 'sub-1568', 'sub-1571', 'sub-1578', 'sub-1583', 'sub-1595', 'sub-16', 'sub-1603', 'sub-1605', 'sub-1608', 'sub-161', 'sub-1629', 'sub-1638', 'sub-1646', 'sub-165', 'sub-1652', 'sub-1660', 'sub-1672', 'sub-1673', 'sub-1674', 'sub-1678', 'sub-1682', 'sub-1683', 'sub-1688', 'sub-1695', 'sub-1697', 'sub-1701', 'sub-1707', 'sub-1715', 'sub-191', 'sub-203', 'sub-204', 'sub-206', 'sub-219', 'sub-243', 'sub-245', 'sub-25', 'sub-27', 'sub-273', 'sub-274', 'sub-277', 'sub-289', 'sub-295', 'sub-296', 'sub-297', 'sub-305', 'sub-320', 'sub-322', 'sub-328', 'sub-329', 'sub-33', 'sub-330', 'sub-331', 'sub-332', 'sub-333', 'sub-344', 'sub-348', 'sub-35', 'sub-352', 'sub-355', 'sub-36', 'sub-364', 'sub-379', 'sub-382', 'sub-384', 'sub-397', 'sub-403', 'sub-408', 'sub-409', 'sub-415', 'sub-416', 'sub-420', 'sub-426', 'sub-435', 'sub-444', 'sub-449', 'sub-462', 'sub-467', 'sub-473', 'sub-478', 'sub-485', 'sub-487', 'sub-49', 'sub-490', 'sub-50', 'sub-507', 'sub-515', 'sub-518', 'sub-522', 'sub-538', 'sub-541', 'sub-542', 'sub-544', 'sub-546', 'sub-551', 'sub-552', 'sub-557', 'sub-560', 'sub-580', 'sub-587', 'sub-589', 'sub-594', 'sub-595', 'sub-596', 'sub-616', 'sub-62', 'sub-622', 'sub-626', 'sub-654', 'sub-657', 'sub-663', 'sub-67', 'sub-674', 'sub-68', 'sub-680', 'sub-681', 'sub-685', 'sub-69', 'sub-703', 'sub-717', 'sub-721', 'sub-728', 'sub-75', 'sub-752', 'sub-759', 'sub-794', 'sub-801', 'sub-807', 'sub-813', 'sub-821', 'sub-822', 'sub-830', 'sub-834', 'sub-839', 'sub-848', 'sub-853', 'sub-860', 'sub-869', 'sub-870', 'sub-878', 'sub-888', 'sub-889', 'sub-894', 'sub-9', 'sub-908', 'sub-910', 'sub-911', 'sub-918', 'sub-924', 'sub-927', 'sub-931', 'sub-933', 'sub-943', 'sub-944', 'sub-947', 'sub-96', 'sub-965', 'sub-969', 'sub-972', 'sub-976', 'sub-982', 'sub-991', 'sub-996', 'sub-998']
        small_group = ['sub-1', 'sub-10', 'sub-1000', 'sub-1001', 'sub-1003', 'sub-1004', 'sub-1005', 'sub-1006', 'sub-1007', 'sub-1009', 'sub-101', 'sub-1012', 'sub-1017', 'sub-1019', 'sub-1020', 'sub-1021', 'sub-1022', 'sub-1025', 'sub-1026', 'sub-1027', 'sub-1028', 'sub-1029', 'sub-1033', 'sub-1034', 'sub-1036', 'sub-1037', 'sub-1038', 'sub-104', 'sub-1040', 'sub-1042', 'sub-1043', 'sub-1047', 'sub-1049', 'sub-105', 'sub-1050', 'sub-1053', 'sub-1058', 'sub-1061', 'sub-1062', 'sub-1063', 'sub-1065', 'sub-1066', 'sub-1067', 'sub-1068', 'sub-1069', 'sub-1070', 'sub-1072', 'sub-1077', 'sub-1078', 'sub-108', 'sub-1081', 'sub-1082', 'sub-1083', 'sub-1084', 'sub-1085', 'sub-1087', 'sub-1088', 'sub-1089', 'sub-1090', 'sub-1091', 'sub-1092', 'sub-1093', 'sub-1095', 'sub-1098', 'sub-1099', 'sub-1100', 'sub-1104', 'sub-1108', 'sub-1109', 'sub-111', 'sub-1110', 'sub-1111', 'sub-1112', 'sub-1117', 'sub-1121', 'sub-1122', 'sub-1124', 'sub-1126', 'sub-1129', 'sub-1131', 'sub-1132', 'sub-1133', 'sub-1134', 'sub-1137', 'sub-1143', 'sub-1145', 'sub-1146', 'sub-115', 'sub-1151', 'sub-1152', 'sub-1153', 'sub-1155', 'sub-1156', 'sub-1158', 'sub-1159', 'sub-1160', 'sub-1162', 'sub-1166', 'sub-1168', 'sub-1169', 'sub-117', 'sub-1171', 'sub-1172', 'sub-1175', 'sub-1177', 'sub-1178', 'sub-1179', 'sub-1180', 'sub-1181', 'sub-1184', 'sub-1185', 'sub-1187', 'sub-1188', 'sub-119', 'sub-1190', 'sub-1191', 'sub-1192', 'sub-1195', 'sub-1197', 'sub-1199', 'sub-120', 'sub-1201', 'sub-1203', 'sub-1205', 'sub-1206', 'sub-1207', 'sub-1208', 'sub-121', 'sub-1210', 'sub-1214', 'sub-1216', 'sub-1218', 'sub-1219', 'sub-1220', 'sub-1221', 'sub-1222', 'sub-1224', 'sub-1225', 'sub-1226', 'sub-1228', 'sub-1230', 'sub-1233', 'sub-1236', 'sub-1238', 'sub-1240', 'sub-1243', 'sub-1245', 'sub-1247', 'sub-1249', 'sub-1250', 'sub-1252', 'sub-1253', 'sub-1255', 'sub-1257', 'sub-1259', 'sub-126', 'sub-1261', 'sub-1262', 'sub-1263', 'sub-1264', 'sub-1265', 'sub-1267', 'sub-1268', 'sub-1269', 'sub-1270', 'sub-1271', 'sub-1272', 'sub-1273', 'sub-1274', 'sub-1275', 'sub-1277', 'sub-1278', 'sub-1279', 'sub-1286', 'sub-1287', 'sub-1288', 'sub-1289', 'sub-1290', 'sub-1294', 'sub-1295', 'sub-1298', 'sub-1299', 'sub-13', 'sub-130', 'sub-1300', 'sub-1302', 'sub-1303', 'sub-1304', 'sub-1307', 'sub-1313', 'sub-1315', 'sub-1316', 'sub-1317', 'sub-1318', 'sub-132', 'sub-1321', 'sub-1325', 'sub-1327', 'sub-1328', 'sub-1333', 'sub-1334', 'sub-1336', 'sub-1341', 'sub-1342', 'sub-1343', 'sub-1344', 'sub-1345', 'sub-1350', 'sub-1351', 'sub-1353', 'sub-1356', 'sub-1359', 'sub-136', 'sub-1360', 'sub-1361', 'sub-1362', 'sub-1365', 'sub-1367', 'sub-1368', 'sub-137', 'sub-1371', 'sub-1372', 'sub-1375', 'sub-1376', 'sub-1377', 'sub-1378', 'sub-1383', 'sub-1385', 'sub-1387', 'sub-139', 'sub-1391', 'sub-1392', 'sub-1393', 'sub-1394', 'sub-1397', 'sub-1398', 'sub-1399', 'sub-14', 'sub-140', 'sub-1400', 'sub-1401', 'sub-1402', 'sub-1403', 'sub-1405', 'sub-1407', 'sub-141', 'sub-1411', 'sub-1412', 'sub-1414', 'sub-1416', 'sub-1418', 'sub-142', 'sub-1420', 'sub-1424', 'sub-1425', 'sub-1426', 'sub-1430', 'sub-1431', 'sub-1433', 'sub-1435', 'sub-1436', 'sub-1439', 'sub-144', 'sub-1441', 'sub-1451', 'sub-1452', 'sub-1453', 'sub-1455', 'sub-1456', 'sub-1457', 'sub-1458', 'sub-1460', 'sub-1461', 'sub-1464', 'sub-1465', 'sub-1467', 'sub-1468', 'sub-1469', 'sub-1470', 'sub-1471', 'sub-1472', 'sub-1473', 'sub-1474', 'sub-1476', 'sub-1477', 'sub-1479', 'sub-1481', 'sub-1482', 'sub-1484', 'sub-1486', 'sub-1487', 'sub-149', 'sub-1491', 'sub-1493', 'sub-1495', 'sub-1497', 'sub-1498', 'sub-1499', 'sub-15', 'sub-150', 'sub-1502', 'sub-1505', 'sub-151', 'sub-1510', 'sub-1512', 'sub-1515', 'sub-1516', 'sub-152', 'sub-1524', 'sub-1526', 'sub-1527', 'sub-1528', 'sub-1529', 'sub-153', 'sub-1530', 'sub-1531', 'sub-1532', 'sub-1535', 'sub-1536', 'sub-1537', 'sub-1538', 'sub-154', 'sub-1540', 'sub-1542', 'sub-1543', 'sub-1544', 'sub-1546', 'sub-1547', 'sub-1549', 'sub-1559', 'sub-1560', 'sub-1561', 'sub-1563', 'sub-1565', 'sub-1566', 'sub-157', 'sub-1570', 'sub-1572', 'sub-1573', 'sub-1574', 'sub-1576', 'sub-1577', 'sub-1579', 'sub-158', 'sub-1580', 'sub-1581', 'sub-1582', 'sub-1584', 'sub-1585', 'sub-1586', 'sub-1587', 'sub-1590', 'sub-1591', 'sub-1592', 'sub-1593', 'sub-1594', 'sub-1596', 'sub-1597', 'sub-1599', 'sub-160', 'sub-1600', 'sub-1601', 'sub-1604', 'sub-1606', 'sub-1607', 'sub-1609', 'sub-1610', 'sub-1614', 'sub-1615', 'sub-1616', 'sub-1617', 'sub-1618', 'sub-1619', 'sub-162', 'sub-1620', 'sub-1621', 'sub-1622', 'sub-1623', 'sub-1624', 'sub-1625', 'sub-1627', 'sub-1628', 'sub-163', 'sub-1630', 'sub-1631', 'sub-1632', 'sub-1633', 'sub-1635', 'sub-1636', 'sub-1639', 'sub-164', 'sub-1642', 'sub-1643', 'sub-1645', 'sub-1648', 'sub-1649', 'sub-1650', 'sub-1651', 'sub-1653', 'sub-1654', 'sub-1655', 'sub-1657', 'sub-1658', 'sub-166', 'sub-1661', 'sub-1662', 'sub-1663', 'sub-1664', 'sub-1665', 'sub-1666', 'sub-1667', 'sub-1668', 'sub-1669', 'sub-1671', 'sub-1675', 'sub-1676', 'sub-1679', 'sub-168', 'sub-1680', 'sub-1684', 'sub-1685', 'sub-1686', 'sub-1687', 'sub-169', 'sub-1690', 'sub-1691', 'sub-1692', 'sub-1693', 'sub-1694', 'sub-1698', 'sub-1699', 'sub-17', 'sub-170', 'sub-1700', 'sub-1703', 'sub-1704', 'sub-1705', 'sub-1706', 'sub-1709', 'sub-171', 'sub-1710', 'sub-1711', 'sub-1712', 'sub-1714', 'sub-1716', 'sub-1718', 'sub-172', 'sub-1720', 'sub-1721', 'sub-1722', 'sub-1723', 'sub-1726', 'sub-1728', 'sub-1729', 'sub-173', 'sub-1731', 'sub-1732', 'sub-1734', 'sub-1735', 'sub-1737', 'sub-175', 'sub-176', 'sub-178', 'sub-179', 'sub-180', 'sub-181', 'sub-182', 'sub-183', 'sub-184', 'sub-186', 'sub-187', 'sub-188', 'sub-189', 'sub-19', 'sub-192', 'sub-194', 'sub-195', 'sub-197', 'sub-199', 'sub-20', 'sub-200', 'sub-201', 'sub-202', 'sub-207', 'sub-208', 'sub-21', 'sub-210', 'sub-211', 'sub-212', 'sub-213', 'sub-214', 'sub-215', 'sub-216', 'sub-217', 'sub-218', 'sub-220', 'sub-222', 'sub-223', 'sub-225', 'sub-226', 'sub-229', 'sub-23', 'sub-230', 'sub-239', 'sub-24', 'sub-240', 'sub-242', 'sub-244', 'sub-246', 'sub-248', 'sub-250', 'sub-252', 'sub-253', 'sub-254', 'sub-255', 'sub-256', 'sub-257', 'sub-258', 'sub-259', 'sub-26', 'sub-261', 'sub-263', 'sub-265', 'sub-266', 'sub-267', 'sub-268', 'sub-269', 'sub-270', 'sub-272', 'sub-275', 'sub-276', 'sub-28', 'sub-280', 'sub-281', 'sub-283', 'sub-285', 'sub-286', 'sub-287', 'sub-288', 'sub-290', 'sub-291', 'sub-292', 'sub-293', 'sub-298', 'sub-299', 'sub-30', 'sub-300', 'sub-302', 'sub-304', 'sub-306', 'sub-308', 'sub-309', 'sub-310', 'sub-311', 'sub-312', 'sub-313', 'sub-315', 'sub-316', 'sub-318', 'sub-319', 'sub-324', 'sub-325', 'sub-327', 'sub-334', 'sub-336', 'sub-337', 'sub-34', 'sub-340', 'sub-342', 'sub-346', 'sub-347', 'sub-349', 'sub-350', 'sub-351', 'sub-354', 'sub-356', 'sub-357', 'sub-358', 'sub-361', 'sub-362', 'sub-365', 'sub-367', 'sub-368', 'sub-371', 'sub-373', 'sub-375', 'sub-376', 'sub-377', 'sub-38', 'sub-381', 'sub-385', 'sub-388', 'sub-389', 'sub-39', 'sub-390', 'sub-391', 'sub-392', 'sub-393', 'sub-394', 'sub-395', 'sub-396', 'sub-399', 'sub-4', 'sub-405', 'sub-406', 'sub-407', 'sub-41', 'sub-410', 'sub-411', 'sub-413', 'sub-418', 'sub-419', 'sub-423', 'sub-424', 'sub-425', 'sub-427', 'sub-429', 'sub-43', 'sub-430', 'sub-431', 'sub-437', 'sub-438', 'sub-439', 'sub-44', 'sub-440', 'sub-441', 'sub-442', 'sub-445', 'sub-45', 'sub-450', 'sub-452', 'sub-454', 'sub-456', 'sub-458', 'sub-46', 'sub-461', 'sub-468', 'sub-469', 'sub-470', 'sub-472', 'sub-474', 'sub-475', 'sub-476', 'sub-477', 'sub-479', 'sub-480', 'sub-481', 'sub-482', 'sub-484', 'sub-486', 'sub-488', 'sub-489', 'sub-492', 'sub-493', 'sub-496', 'sub-499', 'sub-5', 'sub-502', 'sub-503', 'sub-504', 'sub-506', 'sub-508', 'sub-509', 'sub-51', 'sub-510', 'sub-513', 'sub-514', 'sub-516', 'sub-519', 'sub-52', 'sub-520', 'sub-526', 'sub-527', 'sub-528', 'sub-531', 'sub-532', 'sub-533', 'sub-534', 'sub-535', 'sub-536', 'sub-540', 'sub-549', 'sub-554', 'sub-555', 'sub-556', 'sub-558', 'sub-559', 'sub-562', 'sub-564', 'sub-565', 'sub-566', 'sub-567', 'sub-569', 'sub-571', 'sub-573', 'sub-575', 'sub-576', 'sub-577', 'sub-579', 'sub-58', 'sub-581', 'sub-583', 'sub-584', 'sub-585', 'sub-59', 'sub-590', 'sub-591', 'sub-593', 'sub-597', 'sub-598', 'sub-599', 'sub-60', 'sub-603', 'sub-604', 'sub-605', 'sub-606', 'sub-607', 'sub-608', 'sub-609', 'sub-614', 'sub-615', 'sub-617', 'sub-618', 'sub-619', 'sub-623', 'sub-624', 'sub-628', 'sub-63', 'sub-630', 'sub-632', 'sub-635', 'sub-636', 'sub-637', 'sub-639', 'sub-64', 'sub-640', 'sub-642', 'sub-643', 'sub-645', 'sub-649', 'sub-650', 'sub-653', 'sub-655', 'sub-656', 'sub-658', 'sub-659', 'sub-66', 'sub-660', 'sub-662', 'sub-664', 'sub-665', 'sub-675', 'sub-676', 'sub-677', 'sub-679', 'sub-683', 'sub-687', 'sub-688', 'sub-689', 'sub-690', 'sub-693', 'sub-696', 'sub-697', 'sub-7', 'sub-70', 'sub-700', 'sub-701', 'sub-702', 'sub-704', 'sub-705', 'sub-708', 'sub-709', 'sub-71', 'sub-712', 'sub-713', 'sub-714', 'sub-715', 'sub-716', 'sub-718', 'sub-722', 'sub-725', 'sub-726', 'sub-727', 'sub-73', 'sub-730', 'sub-731', 'sub-732', 'sub-733', 'sub-734', 'sub-743', 'sub-744', 'sub-745', 'sub-747', 'sub-748', 'sub-749', 'sub-750', 'sub-753', 'sub-755', 'sub-758', 'sub-762', 'sub-763', 'sub-769', 'sub-77', 'sub-772', 'sub-773', 'sub-774', 'sub-775', 'sub-777', 'sub-778', 'sub-779', 'sub-780', 'sub-783', 'sub-784', 'sub-785', 'sub-786', 'sub-787', 'sub-790', 'sub-792', 'sub-795', 'sub-796', 'sub-798', 'sub-799', 'sub-80', 'sub-800', 'sub-804', 'sub-805', 'sub-808', 'sub-809', 'sub-81', 'sub-810', 'sub-812', 'sub-814', 'sub-815', 'sub-816', 'sub-818', 'sub-819', 'sub-820', 'sub-824', 'sub-825', 'sub-827', 'sub-828', 'sub-829', 'sub-83', 'sub-832', 'sub-833', 'sub-835', 'sub-838', 'sub-84', 'sub-840', 'sub-841', 'sub-846', 'sub-849', 'sub-850', 'sub-852', 'sub-854', 'sub-855', 'sub-856', 'sub-857', 'sub-86', 'sub-862', 'sub-863', 'sub-868', 'sub-87', 'sub-872', 'sub-875', 'sub-876', 'sub-880', 'sub-882', 'sub-884', 'sub-885', 'sub-886', 'sub-887', 'sub-89', 'sub-891', 'sub-892', 'sub-893', 'sub-895', 'sub-897', 'sub-898', 'sub-899', 'sub-90', 'sub-900', 'sub-901', 'sub-903', 'sub-904', 'sub-905', 'sub-906', 'sub-907', 'sub-91', 'sub-913', 'sub-914', 'sub-915', 'sub-919', 'sub-92', 'sub-920', 'sub-921', 'sub-922', 'sub-923', 'sub-925', 'sub-926', 'sub-928', 'sub-929', 'sub-930', 'sub-932', 'sub-934', 'sub-935', 'sub-938', 'sub-94', 'sub-941', 'sub-945', 'sub-948', 'sub-950', 'sub-951', 'sub-953', 'sub-955', 'sub-956', 'sub-957', 'sub-958', 'sub-961', 'sub-962', 'sub-963', 'sub-966', 'sub-967', 'sub-97', 'sub-970', 'sub-971', 'sub-973', 'sub-974', 'sub-975', 'sub-978', 'sub-979', 'sub-98', 'sub-983', 'sub-985', 'sub-986', 'sub-987', 'sub-988', 'sub-989', 'sub-99', 'sub-992', 'sub-993', 'sub-994', 'sub-995', 'sub-999']

        if "flair" in args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        elif "adc" in args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        tests_anomaly_masks = glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz")

        basic_affine = nib.load(test_anomaly_images[0]).affine

        images_to_exclude = []
        with open(ROOT_DIR+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(ROOT_DIR+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        
        test_anomaly_transforms = define_instance(args, "val_transforms")

        
        test_anomaly_large_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        large_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]

        test_anomaly_medium_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in medium_group]        
        medium_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in medium_group]

        test_anomaly_small_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in small_group]        
        small_group_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in small_group]
        
        test_anomaly_large_images = sorted(test_anomaly_large_images, key=lambda x: os.path.basename(x).split('.')[0])
        large_group_masks = sorted(large_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        test_anomaly_medium_images = sorted(test_anomaly_medium_images, key=lambda x: os.path.basename(x).split('.')[0])
        medium_group_masks = sorted(medium_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        test_anomaly_small_images = sorted(test_anomaly_small_images, key=lambda x: os.path.basename(x).split('.')[0])
        small_group_masks = sorted(small_group_masks, key=lambda x: os.path.basename(x).split('.')[0])

        test_anomaly_small_images = test_anomaly_small_images[:200] # Test set : SOOP: we only kept the first 200 small group images otherwis takes too much time
        small_group_masks = small_group_masks[:200] # Test set : SOOP: we only kept the first 200 small group images otherwis takes too much time


        test_anomaly_large_ds = CacheDataset(data=test_anomaly_large_images, transform=test_anomaly_transforms)
        test_anomaly_medium_ds = CacheDataset(data=test_anomaly_medium_images, transform=test_anomaly_transforms)
        test_anomaly_small_ds = CacheDataset(data=test_anomaly_small_images, transform=test_anomaly_transforms)

    if args.dataset["test"] == "isles" or args.dataset["test"] == "soop":

        test_anomaly_large_loader_select_params = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_anomaly_large_ds[:len(test_anomaly_large_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_large_images_select_params = test_anomaly_large_images[:len(test_anomaly_large_images)//2]

        test_anomaly_large_loader_metrics = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_anomaly_large_ds[len(test_anomaly_large_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_large_images_metrics = test_anomaly_large_images[len(test_anomaly_large_images)//2:]


        test_anomaly_medium_loader_select_params = DataLoader(
            test_anomaly_medium_ds[:len(test_anomaly_medium_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_medium_images_select_params = test_anomaly_medium_images[:len(test_anomaly_medium_images)//2]

        test_anomaly_medium_loader_metrics = DataLoader(
            test_anomaly_medium_ds[len(test_anomaly_medium_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_medium_images_metrics = test_anomaly_medium_images[len(test_anomaly_medium_images)//2:]

        test_anomaly_small_loader_select_params = DataLoader(
            test_anomaly_small_ds[:len(test_anomaly_small_ds)//2], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_small_images_select_params = test_anomaly_small_images[:len(test_anomaly_small_images)//2]

        test_anomaly_small_loader_metrics = DataLoader(
            test_anomaly_small_ds[len(test_anomaly_small_ds)//2:], batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_small_images_metrics = test_anomaly_small_images[len(test_anomaly_small_images)//2:]
        
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
    if args.dataset["test"] == "soop_fast": 
        large_group = ['sub-1010', 'sub-1013', 'sub-1015', 'sub-1032', 'sub-1035', 'sub-1039', 'sub-1041', 'sub-1045', 'sub-1046', 'sub-1071', 'sub-1073', 'sub-1086', 'sub-1102', 'sub-1107', 'sub-1115', 'sub-113', 'sub-114', 'sub-1149', 'sub-1150', 'sub-116', 'sub-1164', 'sub-1165', 'sub-118', 'sub-1200', 'sub-1204', 'sub-1209', 'sub-1213', 'sub-1215', 'sub-1223', 'sub-1227', 'sub-1232', 'sub-1246', 'sub-1258', 'sub-127', 'sub-1280', 'sub-1282', 'sub-1283', 'sub-1285', 'sub-1292', 'sub-1305', 'sub-1306', 'sub-1309', 'sub-1312', 'sub-1314', 'sub-1320', 'sub-1323', 'sub-135', 'sub-1354', 'sub-1355', 'sub-1358', 'sub-1364', 'sub-1366', 'sub-1369', 'sub-1373', 'sub-1379', 'sub-1382', 'sub-1386', 'sub-1395', 'sub-1409', 'sub-1410', 'sub-1413', 'sub-1422', 'sub-1432', 'sub-1445', 'sub-1447', 'sub-1475', 'sub-1478', 'sub-1480', 'sub-1483', 'sub-1485', 'sub-1488', 'sub-1507', 'sub-1508', 'sub-1511', 'sub-1517', 'sub-1552', 'sub-1554', 'sub-1555', 'sub-1569', 'sub-1598', 'sub-1612', 'sub-1634', 'sub-1637', 'sub-1656', 'sub-1670', 'sub-1677', 'sub-1719', 'sub-1725', 'sub-1727', 'sub-1736', 'sub-174', 'sub-177', 'sub-185', 'sub-190', 'sub-196', 'sub-198', 'sub-2', 'sub-221', 'sub-235', 'sub-241', 'sub-247', 'sub-249', 'sub-260', 'sub-262', 'sub-264', 'sub-278', 'sub-284', 'sub-294', 'sub-3', 'sub-303', 'sub-314', 'sub-321', 'sub-326', 'sub-335', 'sub-338', 'sub-339', 'sub-341', 'sub-343', 'sub-345', 'sub-359', 'sub-360', 'sub-366', 'sub-370', 'sub-374', 'sub-386', 'sub-398', 'sub-400', 'sub-401', 'sub-412', 'sub-42', 'sub-422', 'sub-432', 'sub-433', 'sub-443', 'sub-446', 'sub-447', 'sub-457', 'sub-463', 'sub-464', 'sub-466', 'sub-47', 'sub-494', 'sub-498', 'sub-501', 'sub-505', 'sub-512', 'sub-517', 'sub-521', 'sub-523', 'sub-525', 'sub-529', 'sub-53', 'sub-530', 'sub-539', 'sub-543', 'sub-56', 'sub-563', 'sub-572', 'sub-613', 'sub-620', 'sub-631', 'sub-634', 'sub-638', 'sub-651', 'sub-652', 'sub-661', 'sub-682', 'sub-692', 'sub-694', 'sub-698', 'sub-699', 'sub-707', 'sub-719', 'sub-723', 'sub-724', 'sub-751', 'sub-754', 'sub-760', 'sub-761', 'sub-768', 'sub-776', 'sub-789', 'sub-79', 'sub-791', 'sub-8', 'sub-803', 'sub-806', 'sub-82', 'sub-823', 'sub-826', 'sub-843', 'sub-844', 'sub-845', 'sub-858', 'sub-861', 'sub-865', 'sub-866', 'sub-873', 'sub-877', 'sub-881', 'sub-896', 'sub-917', 'sub-937', 'sub-939', 'sub-942', 'sub-946', 'sub-95', 'sub-952', 'sub-959', 'sub-960', 'sub-968', 'sub-990']
        # super small fast test dataset

        if "flair" in args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/flair_registered/*.nii.gz"))
        elif "adc" in args.dataset["name"].lower():
            test_anomaly_images = sorted(glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/adc_registered/*.nii.gz"))

        tests_anomaly_masks = glob.glob(ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/*.nii.gz")

        basic_affine = nib.load(test_anomaly_images[0]).affine

        images_to_exclude = []
        with open(ROOT_DIR+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())

        with open(ROOT_DIR+"AnoDiffExperiments/data_splits_lists/final_soop_dataset_small/exclude_non_axial_thick_slices.csv", 'r') as f:
            for line in f:
                images_to_exclude.append(line.strip())
        
        test_anomaly_transforms = define_instance(args, "val_transforms")
        
        test_anomaly_large_images = [path for path in test_anomaly_images if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]        
        test_anomaly_large_masks = [path for path in tests_anomaly_masks if os.path.basename(path).split('.')[0] not in images_to_exclude and os.path.basename(path).split('.')[0] in large_group]

        test_anomaly_large_images = sorted(test_anomaly_large_images, key=lambda x: os.path.basename(x).split('.')[0])
        test_anomaly_large_masks = sorted(test_anomaly_large_masks, key=lambda x: os.path.basename(x).split('.')[0])


        num_workers = 4
        small_ano_batch_size = args.autoencoder_train["batch_size"]


        test_anomaly_large_ds = CacheDataset(data=test_anomaly_large_images, transform=test_anomaly_transforms)
        test_masks_large_ds = CacheDataset(data=test_anomaly_large_masks, transform=test_masks_transforms)
        
        test_anomaly_large_loader_select_params_small = DataLoader( # the first 50% of the test data is used to select the best noise timestep value and best threshold.
            test_anomaly_large_ds[:small_ano_batch_size], batch_size=small_ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_large_images_select_params = test_anomaly_large_images[:small_ano_batch_size]

        test_anomaly_large_loader_metrics_small = DataLoader(       # The second 50% is used to compute the final IOU and DICE metrics with these best values.
            test_anomaly_large_ds[small_ano_batch_size:2*small_ano_batch_size], batch_size=small_ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_anomaly_large_images_metrics = test_anomaly_large_images[small_ano_batch_size:2*small_ano_batch_size]
        
        test_masks_large_loader_select_params_small = DataLoader(
            test_masks_large_ds[:small_ano_batch_size], batch_size=small_ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        test_masks_large_loader_metrics_small = DataLoader(
            test_masks_large_ds[small_ano_batch_size:2*small_ano_batch_size], batch_size=small_ano_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    

    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])

    num_timesteps_to_try = np.arange(NOISE_MIN, NOISE_MAX, NOISE_INTERVAL)
    thresholds_to_try = np.arange(0.02, 0.2, 0.02) # from 0.0 to 0.2 with step 0.02
    median_filter_sizes_to_try = [-1, 3, 5, 7, 10] # -1 means no median filter
    erosion_dilation_iterations_to_try = [0, 1, 2]

    # ------------------------ Compute the raw anomaly maps and save them as nifti files ------------------------ #
    # So that they can be used to compute metrics later with different postprocessing steps without having to recompute the anomaly maps each time.

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
            check_data = first(test_anomaly_large_loader_select_params)
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


    if args.dataset["test"] == "brats":
        for timesteps in num_timesteps_to_try:
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_loader_select_params, test_anomaly_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS)
            
        iou_scores_df, dice_scores_df = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS, ROOT_DIR+"datasets/final_flair_dataset_small/brats_masks_registered/", len(test_anomaly_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
            
        iou_scores_df.to_csv(SUB_EXPERIMENT_DIR+"iou_scores_param_search_brats.csv")
        dice_scores_df.to_csv(SUB_EXPERIMENT_DIR+"dice_scores_param_search_brats.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations = best_params

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR, infer_scheduler, test_anomaly_loader_metrics, test_anomaly_images_metrics, test_masks_loader_metrics, timesteps=best_num_timesteps, threshold=best_threshold, median_filter_size=best_median_filter_size, erosion_dilation_iterations=best_erosion_dilation_iterations)

        
        metrics_result_text = f"mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps}\n"
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size}\n"
        metrics_result_text += f"Best Threshold: {best_threshold:.4f}\n"
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}\n"
        
        tprint(metrics_result_text)

        if args.show_summary_figures:
            show_summary_figure(args, 
                            device, 
                            autoencoder, unet, 
                            infer_scheduler, 
                            test_anomaly_large_loader_metrics, 
                            test_masks_large_loader_metrics, 
                            timesteps=best_num_timesteps, 
                            median_filter_size=best_median_filter_size, 
                            threshold=best_threshold, 
                            erosion_dilation_iterations=best_erosion_dilation_iterations,
                            metrics_result_text=metrics_result_text,
                            ROOT_DIR=ROOT_DIR,
                            EXPERIMENT_NAME=EXPERIMENT_NAME,
                            SUB_EXPERIMENT_NAME=SUB_EXPERIMENT_NAME
                            )

    if args.dataset["test"] == "isles": # TODO: finir pour isles et changer les noms de tous les fichiers isles pour qu'ils aient tous le même nom
        tprint("WARNING ISLES test is still not completely implemented")
        for timesteps in num_timesteps_to_try:

            # --------------------------------- large group
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_large_loader_select_params, test_anomaly_large_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS+"large/")
        
        os.makedirs(ANOMALY_MAPS_DIR+"large/", exist_ok=True)
        os.makedirs(ANOMALY_MAPS_DIR+"medium/", exist_ok=True)
        os.makedirs(ANOMALY_MAPS_DIR+"small/", exist_ok=True)

        iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS, ROOT_DIR+"datasets/TODO/TODO/", len(test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
        
        iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_isles_large_group.csv")
        dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_isles_large_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_large_group.idxmax()['IOU']
        best_num_timesteps_large_group, best_threshold_large_group, best_median_filter_size_large_group, best_erosion_dilation_iterations_large_group = best_params

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR_SELECT_PARAMS+"large/", infer_scheduler, test_anomaly_large_loader_metrics, test_anomaly_large_images_metrics, test_masks_large_loader_metrics, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group)

        
        metrics_result_text = f"Large group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)
        
        if args.show_summary_figures:
            show_summary_figure(args, 
                            device, 
                            autoencoder, unet, 
                            infer_scheduler, 
                            test_anomaly_large_loader_metrics, 
                            test_masks_large_loader_metrics, 
                            timesteps=best_num_timesteps_large_group, 
                            median_filter_size=best_median_filter_size_large_group, 
                            threshold=best_threshold_large_group, 
                            erosion_dilation_iterations=best_erosion_dilation_iterations_large_group,
                            metrics_result_text=metrics_result_text,
                            ROOT_DIR=ROOT_DIR,
                            EXPERIMENT_NAME=EXPERIMENT_NAME,
                            SUB_EXPERIMENT_NAME=SUB_EXPERIMENT_NAME
                            )
        
        for timesteps in num_timesteps_to_try:
            # --------------------------------- medium group
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_medium_loader_select_params, test_anomaly_medium_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/")

        iou_scores_df_medium_group, dice_scores_df_medium_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/", ROOT_DIR+"datasets/final_flair_dataset_small/brats_masks_registered/", len(test_anomaly_medium_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
        
        iou_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_isles_medium_group.csv")
        dice_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_isles_medium_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_medium_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations = best_params

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR+"medium/", infer_scheduler, test_anomaly_medium_loader_metrics, test_anomaly_medium_images_metrics, test_masks_medium_loader_metrics, timesteps=best_num_timesteps, threshold=best_threshold, median_filter_size=best_median_filter_size, erosion_dilation_iterations=best_erosion_dilation_iterations)

        
        metrics_result_text += f"Medium group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

            # --------------------------------- small group
        for timesteps in num_timesteps_to_try:
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_small_loader_select_params, test_anomaly_small_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/")

        iou_scores_df_small_group, dice_scores_df_small_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/", ROOT_DIR+"datasets/final_flair_dataset_small/brats_masks_registered/", len(test_anomaly_small_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
        
        iou_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_isles_small_group.csv")
        dice_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_isles_small_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_small_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations = best_params

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR+"small/", infer_scheduler, test_anomaly_small_loader_metrics, test_anomaly_small_images_metrics, test_masks_small_loader_metrics, timesteps=best_num_timesteps, threshold=best_threshold, median_filter_size=best_median_filter_size, erosion_dilation_iterations=best_erosion_dilation_iterations)

        
        metrics_result_text += f"Small group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}\n"
        tprint(metrics_result_text)
    
            

    if args.dataset["test"] == "soop":
        
        # --------------------------------- large group
        for timesteps in num_timesteps_to_try:       
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_large_loader_select_params, test_anomaly_large_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS+"large/")

        os.makedirs(ANOMALY_MAPS_DIR+"large/", exist_ok=True)
        os.makedirs(ANOMALY_MAPS_DIR+"medium/", exist_ok=True)
        os.makedirs(ANOMALY_MAPS_DIR+"small/", exist_ok=True)

        iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"large/", ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
        
        iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop_large_group.csv")
        dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop_large_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_large_group.idxmax()['IOU']
        best_num_timesteps_large_group, best_threshold_large_group, best_median_filter_size_large_group, best_erosion_dilation_iterations_large_group = best_params

        tprint(f"Best params large group: {best_params}")

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR+"large/", infer_scheduler, test_anomaly_large_loader_metrics, test_anomaly_large_images_metrics, test_masks_large_loader_metrics, best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group)

        
        metrics_result_text = f"Large group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)
        
        if args.show_summary_figures:
            show_summary_figure(args, 
                                device, 
                                autoencoder,
                                unet, 
                                infer_scheduler, 
                                test_anomaly_large_loader_metrics, 
                                test_masks_large_loader_metrics, 
                                infer_timesteps=best_num_timesteps_large_group, 
                                median_filter_size=best_median_filter_size_large_group, 
                                threshold=best_threshold_large_group, 
                                erosion_dilation_iterations=best_erosion_dilation_iterations_large_group,
                                metrics_result_text=metrics_result_text,
                                ROOT_DIR=ROOT_DIR,
                                EXPERIMENT_NAME=EXPERIMENT_NAME,
                                SUB_EXPERIMENT_NAME=SUB_EXPERIMENT_NAME
                                )

        # --------------------------------- medium group
        for timesteps in num_timesteps_to_try:       
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_medium_loader_select_params, test_anomaly_medium_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/")
        
        iou_scores_df_medium_group, dice_scores_df_medium_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/", ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(test_anomaly_medium_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
        
        iou_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop_medium_group.csv")
        dice_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop_medium_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_medium_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations = best_params

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR+"medium/", infer_scheduler, test_anomaly_medium_loader_metrics, test_anomaly_medium_images_metrics, test_masks_medium_loader_metrics, best_num_timesteps, threshold=best_threshold, median_filter_size=best_median_filter_size, erosion_dilation_iterations=best_erosion_dilation_iterations)

        
        metrics_result_text += f"Medium group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

        # --------------------------------- small group
        for timesteps in num_timesteps_to_try:       
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_small_loader_select_params, test_anomaly_small_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/")
        
        iou_scores_df_small_group, dice_scores_df_small_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/", ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(test_anomaly_small_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
        

        iou_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop_small_group.csv")
        dice_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop_small_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_small_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations = best_params

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR+"small/", infer_scheduler, test_anomaly_small_loader_metrics, test_anomaly_small_images_metrics, test_masks_small_loader_metrics, best_num_timesteps, threshold=best_threshold, median_filter_size=best_median_filter_size, erosion_dilation_iterations=best_erosion_dilation_iterations)

        
        metrics_result_text += f"Small group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}\n"
        tprint(metrics_result_text)
    
    if args.dataset["test"] == "soop_fast":
        
        # --------------------------------- large group
        for timesteps in num_timesteps_to_try:
            make_anomaly_maps(args, autoencoder, unet, device, infer_scheduler, test_anomaly_large_loader_select_params_small, test_anomaly_large_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS)

        iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS, ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
        
        iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop.csv")
        dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_large_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations = best_params

        mean_iou, std_iou, mean_dice, std_dice = compute_metrics(args, autoencoder, unet, device, ANOMALY_MAPS_DIR, infer_scheduler, test_anomaly_large_loader_metrics_small, test_anomaly_large_images_metrics, test_masks_large_loader_metrics_small, best_num_timesteps, threshold=best_threshold, median_filter_size=best_median_filter_size, erosion_dilation_iterations=best_erosion_dilation_iterations)

        
        metrics_result_text = f"soop_fast: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}\n"

        tprint(metrics_result_text)


        if args.show_summary_figures:
            show_summary_figure(args, 
                                device, 
                                autoencoder, 
                                unet, 
                                infer_scheduler, 
                                test_anomaly_large_loader_metrics_small, 
                                test_masks_large_loader_metrics_small, 
                                timesteps=best_num_timesteps, 
                                median_filter_size=best_median_filter_size, 
                                threshold=best_threshold, 
                                erosion_dilation_iterations=best_erosion_dilation_iterations,
                                metrics_result_text=metrics_result_text,
                                ROOT_DIR=ROOT_DIR,
                                EXPERIMENT_NAME=EXPERIMENT_NAME,
                                SUB_EXPERIMENT_NAME=SUB_EXPERIMENT_NAME
                                )