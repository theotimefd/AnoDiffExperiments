"""
This file only works for 3D slice by slice inference.
"""

import os
import sys

sys.path.append("../..")


import numpy as np
import json
import argparse
import csv
import torch
from tqdm import tqdm

import nibabel as nib

from utils.utils import *
import utils.scores as scores


from scipy.ndimage import median_filter, binary_erosion, binary_dilation, binary_fill_holes
import time


DEVICE_TYPE = "cuda:0"

def process_anomaly_file(anomaly_file, anomaly_maps_folder, masks_folder,
                             thresholds_to_try, median_filter_sizes_to_try, 
                             erosion_dilation_iterations_to_try, binary_fill_holes_to_try):
    """
    Process a single raw anomaly file and return a dict with scores 
    for all parameter combinations (median filter size, threshold, erosion/dilation).
    """
    

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
    filter_start_time = time.time()
    for median_filter_size in median_filter_sizes_to_try:
        if median_filter_size > 0:
            filtered_np = median_filter(anomaly_map, size=median_filter_size)
            filtered_maps[median_filter_size] = torch.from_numpy(filtered_np)
    filter_elapsed_time = time.time() - filter_start_time
    tprint(f"Filtered anomaly maps computed in {filter_elapsed_time:.4f} seconds for {anomaly_file}")

    
    other_processing_start_time = time.time()
    # Iterate through all combinations efficiently
    for median_filter_size in median_filter_sizes_to_try:
        final_anomaly_map = filtered_maps[median_filter_size]
        
        for threshold in thresholds_to_try:
            ano_segmentation_base = (final_anomaly_map > threshold)
     
            for erosion_dilation_iterations in erosion_dilation_iterations_to_try:
                
                for binary_fill_holes_param in binary_fill_holes_to_try:

                    if erosion_dilation_iterations > 0:
                        ano_segmentation_np = ano_segmentation_base.cpu().numpy()
                        
                        ano_segmentation_np = binary_erosion(ano_segmentation_np, iterations=erosion_dilation_iterations)
                        ano_segmentation_np = binary_dilation(ano_segmentation_np, iterations=erosion_dilation_iterations)

                        if binary_fill_holes_param==1:
                            
                            ano_segmentation_np = binary_fill_holes(ano_segmentation_np)
                            

                        ano_segmentation = torch.from_numpy(ano_segmentation_np)
                    else:
                        ano_segmentation = ano_segmentation_base

                    
                    # ano_segmentation and masks must be in format : B1HWD
                    ano_segmentation = ano_segmentation.unsqueeze(0).unsqueeze(0)
                    
                    iou_scores, dice_scores, _, _, _, _ = scores.compute_scores(ano_segmentation, mask, only_dice_iou=True)

                    # Store results
                    idx = (timesteps, threshold, median_filter_size, erosion_dilation_iterations, binary_fill_holes_param)
                    local_iou_scores[idx] = np.sum(iou_scores)
                    local_dice_scores[idx] = np.sum(dice_scores)
    other_processing_elapsed_time = time.time() - other_processing_start_time
    tprint(f"Other processing (thresholding, erosion/dilation, scoring) completed in {other_processing_elapsed_time:.4f} seconds for {anomaly_file}")
    
    return local_iou_scores, local_dice_scores