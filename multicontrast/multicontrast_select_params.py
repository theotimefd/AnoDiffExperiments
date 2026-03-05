import nibabel as nib
import os
import sys
sys.path.append("..")
import torch
import numpy as np
import json
import argparse
import pandas as pd

from scipy.ndimage import median_filter, binary_erosion, binary_dilation, binary_fill_holes

import utils.scores as scores

from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from utils.utils import dtprint

from tqdm import tqdm

def process_file(anomaly_file,
        combined_ano_maps_folder,
                    combined_masks_folder,
                    thresholds_to_try, median_filter_sizes_to_try,
                    erosion_dilation_iterations_to_try, binary_fill_holes_to_try,
                 ):


    ano_map = nib.load(os.path.join(combined_ano_maps_folder, anomaly_file))
    ano_map_data = ano_map.get_fdata()

    mask_file = os.path.join(combined_masks_folder, anomaly_file.replace('_t_110', ''))
    mask_img = nib.load(mask_file)
    mask_data = mask_img.get_fdata()


    local_iou_scores = {}
    local_dice_scores = {}


    # make sure ano_map and mask are in 11hwd format
    if len(ano_map_data.shape) == 3:
        ano_map_data = np.expand_dims(ano_map_data, axis=0)  # add channel dimension at the beginning
        ano_map_data = np.expand_dims(ano_map_data, axis=0)  # add batch dimension at the beginning
    if len(mask_data.shape) == 3:
        mask_data = np.expand_dims(mask_data, axis=0)  # add channel dimension at the beginning
        mask_data = np.expand_dims(mask_data, axis=0)  # add batch dimension at the beginning
    
    mask_data = torch.from_numpy(mask_data)

    # Pre-compute median filtered versions
    filtered_maps = {-1: torch.from_numpy(ano_map_data)}
    
    for median_filter_size in median_filter_sizes_to_try:

        if median_filter_size > 0:
            filtered_np = median_filter(ano_map_data, size=median_filter_size)
            filtered_maps[median_filter_size] = torch.from_numpy(filtered_np)


    for median_filter_size in median_filter_sizes_to_try:

        filtered_map = filtered_maps[median_filter_size]

        for threshold in thresholds_to_try:

            for erosion_dilation_iterations in erosion_dilation_iterations_to_try:
                
                for binary_fill_holes_param in binary_fill_holes_to_try:

 
                    # thresholding
                    segmentation = (filtered_map > threshold)

                    # erosion and dilation
                    if erosion_dilation_iterations > 0:
                        if isinstance(segmentation, torch.Tensor):
                            segmentation = segmentation.cpu().numpy()
                        segmentation = binary_erosion(segmentation, iterations=erosion_dilation_iterations)
                        segmentation = binary_dilation(segmentation, iterations=erosion_dilation_iterations)

                    # fill holes
                    if binary_fill_holes_param==1:
                        if isinstance(segmentation, torch.Tensor):
                            segmentation = segmentation.cpu().numpy()
                        segmentation = binary_fill_holes(segmentation)
                        
                    if isinstance(segmentation, np.ndarray):
                        segmentation = torch.from_numpy(segmentation)
                    
                    # compute the IOU and DICE scores
                    iou_scores, dice_scores, _, _, _, _ = scores.compute_scores(segmentation, mask_data, only_dice_iou=True)

                    idx = (median_filter_size, threshold, erosion_dilation_iterations, binary_fill_holes_param)
                    local_iou_scores[idx] = np.sum(iou_scores)
                    local_dice_scores[idx] = np.sum(dice_scores)

    return local_iou_scores, local_dice_scores

def launch():

    ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

    GROUP = "large"

    adc_anomaly_maps_select_params_folder = f"{ROOT_DIR}datasets/anomaly_maps/exp_2_4_select_params/{GROUP}/"
    adc_anomaly_maps_folder = f"{ROOT_DIR}datasets/anomaly_maps/exp_2_4/{GROUP}/"

    flair_anomaly_maps_select_params_folder = f"{ROOT_DIR}datasets/anomaly_maps/exp_3_4_select_params/{GROUP}/"
    flair_anomaly_maps_folder = f"{ROOT_DIR}datasets/anomaly_maps/exp_3_4/{GROUP}/"

    combined_anomaly_maps_select_params_folder = f"{ROOT_DIR}datasets/anomaly_maps/combined_ano_maps_select_params/3d_patch_ddpm/{GROUP}/"
    combined_anomaly_maps_folder = f"{ROOT_DIR}datasets/anomaly_maps/combined_ano_maps/3d_patch_ddpm/{GROUP}/"

    combined_masks_folder = f"{ROOT_DIR}datasets/final_soop_dataset_small/masks_combined_registered/"

    combined_ano_maps_select_params = os.listdir(combined_anomaly_maps_select_params_folder)
    combined_ano_maps_select_params = sorted([f for f in combined_ano_maps_select_params if "seg" not in f])

    combined_ano_maps = os.listdir(combined_anomaly_maps_folder)
    combined_ano_maps = sorted([f for f in combined_ano_maps if "seg" not in f])


    combined_masks_select_params = sorted([path for path in combined_ano_maps_select_params])

    combined_masks = sorted([path.replace("ano_map_", "") for path in combined_ano_maps])


    # get config json from the adc exp_2_2

    config_dict = json.load(open(ROOT_DIR+"AnoDiffExperiments/multicontrast/config.json", "r"))
    args = argparse.Namespace(**config_dict)

    median_filter_sizes_to_try = args.anomaly_detection_param_search["median_filter_sizes"] # -1 means no median filter
    thresholds_to_try = args.anomaly_detection_param_search["thresholds"]
    erosion_dilation_iterations_to_try = args.anomaly_detection_param_search["erosion_dilation_iterations"]
    binary_fill_holes_to_try = args.anomaly_detection_param_search["binary_fill_holes"]

    # Create the MultiIndex from timesteps, median filter sizes, thresholds, erosion and dilation iterations
    iou_scores_midx = pd.MultiIndex.from_product([median_filter_sizes_to_try, thresholds_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try])
    iou_scores_df = pd.DataFrame(index=iou_scores_midx, columns=["IOU"])
    iou_scores_df.fillna(0.0, inplace=True)
    iou_scores_df.index.names = ['median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes']

    dice_scores_midx = pd.MultiIndex.from_product([median_filter_sizes_to_try, thresholds_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try])
    dice_scores_df = pd.DataFrame(index=dice_scores_midx, columns=["DICE"])
    dice_scores_df.fillna(0.0, inplace=True)
    dice_scores_df.index.names = ['median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes']


    process_func = partial(
        process_file,
        combined_ano_maps_folder = combined_anomaly_maps_folder,
        combined_masks_folder = combined_masks_folder,
        thresholds_to_try = thresholds_to_try,
        median_filter_sizes_to_try = median_filter_sizes_to_try,
        erosion_dilation_iterations_to_try = erosion_dilation_iterations_to_try,
        binary_fill_holes_to_try = binary_fill_holes_to_try,
    )

    max_workers = min(64, mp.cpu_count()) # 48 cores per gpu https://gricad-doc.univ-grenoble-alpes.fr/hpc/kraken/kraken/#the-kraken-platform
    
    dtprint(f"Using max_workers={max_workers} for multiprocessing")
    
    ctx = mp.get_context("spawn")


    results = []
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(process_func, file_name): file_name for file_name in combined_ano_maps}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing anomaly maps"):
            results.append(future.result())


    
    dtprint("multiprocesses all finished")

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
    iou_scores_df = iou_scores_df / len(combined_ano_maps)
    dice_scores_df = dice_scores_df / len(combined_ano_maps)


    iou_scores_df.to_csv(os.path.join(ROOT_DIR, f"AnoDiffExperiments/multicontrast/anomaly_detection_param_search_iou_scores_{GROUP}_group.csv"))
    dice_scores_df.to_csv(os.path.join(ROOT_DIR, f"AnoDiffExperiments/multicontrast/anomaly_detection_param_search_dice_scores_{GROUP}_group.csv"))

    best_params = iou_scores_df.idxmax()['IOU']
    best_median_filter_size, best_threshold, best_erosion_dilation_iterations, best_binary_fill_holes = best_params

    # Save best parameters to CSV
    best_params_df = pd.DataFrame({
        'parameter': ['median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
        'value': [best_median_filter_size, best_threshold, best_erosion_dilation_iterations, best_binary_fill_holes]
    })
    best_params_df.to_csv(os.path.join(ROOT_DIR, f"AnoDiffExperiments/multicontrast/best_params_{GROUP}.csv"), index=False)


    
    metrics_result_text = f"Best Median Filter Size: {best_median_filter_size}\n"
    metrics_result_text += f"Best Threshold: {best_threshold:.4f}\n"
    metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}\n"
    metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}\n"
    dtprint(metrics_result_text)

if __name__ == "__main__":
    
    launch()