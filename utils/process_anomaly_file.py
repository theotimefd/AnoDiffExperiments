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
import json
import argparse
import csv
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism, StrEnum
from torch.amp import autocast
from tqdm import tqdm

import nibabel as nib

from monai.networks.schedulers import DDPMScheduler

from typing import Union

import pandas as pd

import AnoDDPM.simplex as simplex

import utils.custom_transforms as custom_transforms

import utils.simplex_ddpm as simplex_ddpm
import utils.thor_ddpm as thor_ddpm
from utils.utils import *
import utils.scores as scores


from scipy.ndimage import median_filter, binary_erosion, binary_dilation
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

DEVICE_TYPE = "cuda:0"

def process_anomaly_file(anomaly_file, anomaly_maps_folder, masks_folder, 
                             thresholds_to_try, median_filter_sizes_to_try, 
                             erosion_dilation_iterations_to_try):
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
    for median_filter_size in median_filter_sizes_to_try:
        if median_filter_size > 0:
            filtered_np = median_filter(anomaly_map, size=median_filter_size)
            filtered_maps[median_filter_size] = torch.from_numpy(filtered_np)
    
    
    # Iterate through all combinations efficiently
    for median_filter_size in median_filter_sizes_to_try:
        final_anomaly_map = filtered_maps[median_filter_size]
        
        
        for threshold in thresholds_to_try:
            ano_segmentation_base = (final_anomaly_map > threshold)

                        
            for erosion_dilation_iterations in erosion_dilation_iterations_to_try:


                if erosion_dilation_iterations > 0:
                    ano_segmentation_np = ano_segmentation_base.cpu().numpy()
                    
                    ano_segmentation_np = binary_erosion(ano_segmentation_np, iterations=erosion_dilation_iterations)
                    ano_segmentation_np = binary_dilation(ano_segmentation_np, iterations=erosion_dilation_iterations)

                    ano_segmentation = torch.from_numpy(ano_segmentation_np)
                else:
                    ano_segmentation = ano_segmentation_base

                
                # ano_segmentation and masks must be in format : B1HWD
                ano_segmentation = ano_segmentation.unsqueeze(0).unsqueeze(0)
                
                iou_scores, dice_scores, hausdorff_distances, precision_scores, recall_scores, f1_scores = scores.compute_scores(ano_segmentation, mask)

                # Store results
                idx = (timesteps, threshold, median_filter_size, erosion_dilation_iterations)
                local_iou_scores[idx] = np.sum(iou_scores)
                local_dice_scores[idx] = np.sum(dice_scores)
    
    
    return local_iou_scores, local_dice_scores