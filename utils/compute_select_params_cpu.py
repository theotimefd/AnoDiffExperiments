"""
This file only works for 3D slice by slice inference.
"""

import os
import glob
import sys

sys.path.append("../..")
#import opensimplex

#from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
from monai.utils import set_determinism
from tqdm import tqdm

import nibabel as nib

from monai.networks.schedulers import DDPMScheduler


import pandas as pd


import utils.simplex_ddpm as simplex_ddpm

import utils.process_anomaly_file as process_anomaly_file

from datasets import anomaly_datasets

from utils.utils import *


from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp



def compute_select_params_multithreaded(args, 
                                        anomaly_maps_folder, 
                                        masks_folder, 
                                        total_nb_images, 
                                        num_timesteps_to_try, 
                                        thresholds_to_try, 
                                        median_filter_sizes_to_try, 
                                        erosion_dilation_iterations_to_try, 
                                        binary_fill_holes_to_try):
    
    # takes an input folder with all the saved infered 3D anomaly maps 
    # and tests different combinations of post-processing and returns and saves a table with all the scores for each set of parameters

    tprint("launching compute_select_params_multithreaded")
    

    # Create the MultiIndex from timesteps, thresholds, median filter sizes, erosion and dilation iterations
    iou_scores_midx = pd.MultiIndex.from_product([num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try])
    iou_scores_df = pd.DataFrame(index=iou_scores_midx, columns=["IOU"])
    iou_scores_df.fillna(0.0, inplace=True)
    iou_scores_df.index.names = ['timesteps', 'threshold', 'median_filter_size', 'erosion_dilation_iterations', 'binary_fill_holes']

    dice_scores_midx = pd.MultiIndex.from_product([num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try])
    dice_scores_df = pd.DataFrame(index=dice_scores_midx, columns=["DICE"])
    dice_scores_df.fillna(0.0, inplace=True)
    dice_scores_df.index.names = ['timesteps', 'threshold', 'median_filter_size', 'erosion_dilation_iterations', 'binary_fill_holes']
    
    tprint(f"num timesteps to try: {num_timesteps_to_try}")

    # list all anomaly map files in the input folder
    anomaly_files = [entry.name for entry in os.scandir(anomaly_maps_folder) if entry.is_file() and entry.name.endswith(".nii.gz")]
    # only keep files that have a timestep in the filename that is in num_timesteps_to_try
    anomaly_files = [f for f in anomaly_files if int(f.split('.')[0].split('_')[-1]) in num_timesteps_to_try]

    if not anomaly_files:
        raise RuntimeError(f"No anomaly map files found in '{anomaly_maps_folder}'.")

    process_func = partial(
        process_anomaly_file.process_anomaly_file,
        anomaly_maps_folder=anomaly_maps_folder,
        masks_folder=masks_folder,
        thresholds_to_try=thresholds_to_try,
        median_filter_sizes_to_try=median_filter_sizes_to_try,
        erosion_dilation_iterations_to_try=erosion_dilation_iterations_to_try,
        binary_fill_holes_to_try=binary_fill_holes_to_try
    )

    max_workers = min(192, mp.cpu_count()) # 48 cores per gpu https://gricad-doc.univ-grenoble-alpes.fr/hpc/kraken/kraken/#the-kraken-platform
    tprint(f"Using max_workers={max_workers} for multiprocessing")
    ctx = mp.get_context("spawn")

    if len(anomaly_files) == 1 or max_workers == 1:
        results = [process_func(file_name) for file_name in anomaly_files]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(process_func, file_name): file_name for file_name in anomaly_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing anomaly maps"):
                results.append(future.result())


    
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


def launch_compute_select_params_cpu(args):

    set_determinism(0)

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    SUB_EXPERIMENT_DIR = f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/"
    
    ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params/"
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/" # final anomaly maps with best params
    os.makedirs(ANOMALY_MAPS_DIR_SELECT_PARAMS, exist_ok=True)
    os.makedirs(ANOMALY_MAPS_DIR, exist_ok=True)


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


    # -------------------- define the data --------------------

    if args.dataset["test"] == "brats":
        ano_dataset = anomaly_datasets.BRATS(args)

    if args.dataset["test"] == "isles":
        ano_dataset = anomaly_datasets.ISLES(args)
    
    if args.dataset["test"] == "soop":
        ano_dataset = anomaly_datasets.SOOP(args)
        


    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])

    num_timesteps_to_try = np.arange(NOISE_MIN, NOISE_MAX, NOISE_INTERVAL)

    median_filter_sizes_to_try = args.anomaly_detection_param_search["median_filter_sizes"] # -1 means no median filter
    thresholds_to_try = args.anomaly_detection_param_search["thresholds"]
    erosion_dilation_iterations_to_try = args.anomaly_detection_param_search["erosion_dilation_iterations"]
    binary_fill_holes_to_try = args.anomaly_detection_param_search["binary_fill_holes"]


    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])


    if args.dataset["test"] == "brats":
 
        iou_scores_df, dice_scores_df = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS, ROOT_DIR+"datasets/final_flair_dataset_small/brats_masks_registered/", len(ano_dataset.test_anomaly_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try)
            
        iou_scores_df.to_csv(SUB_EXPERIMENT_DIR+"iou_scores_param_search_brats.csv")
        dice_scores_df.to_csv(SUB_EXPERIMENT_DIR+"dice_scores_param_search_brats.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations, best_binary_fill_holes = best_params

        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps, best_median_filter_size, best_threshold, best_erosion_dilation_iterations, best_binary_fill_holes]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_brats.csv", index=False)

        metrics_result_text = "Brats\n"
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps}\n"
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size}\n"
        metrics_result_text += f"Best Threshold: {best_threshold:.4f}\n"
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations}\n"
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}\n"
        tprint(metrics_result_text)


    if args.dataset["test"] == "isles": # TODO: finir pour isles et changer les noms de tous les fichiers isles pour qu'ils aient tous le même nom
        print("WARNING ISLES test is still not completely implemented")
        

        iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS, ROOT_DIR+"datasets/TODO/TODO/", len(ano_dataset.test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
        
        iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_isles_large_group.csv")
        dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_isles_large_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_large_group.idxmax()['IOU']
        best_num_timesteps_large_group, best_threshold_large_group, best_median_filter_size_large_group, best_erosion_dilation_iterations_large_group, best_binary_fill_holes_large_group = best_params

        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps_large_group, best_median_filter_size_large_group, best_threshold_large_group, best_erosion_dilation_iterations_large_group, best_binary_fill_holes_large_group]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_isles_large_group.csv", index=False)
        
        metrics_result_text = "Large group\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group} "
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes_large_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

        # --------------------------------- medium group
        os.makedirs(ANOMALY_MAPS_DIR+"medium/", exist_ok=True)

        iou_scores_df_medium_group, dice_scores_df_medium_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/", ROOT_DIR+"datasets/final_flair_dataset_small/brats_masks_registered/", len(ano_dataset.test_anomaly_medium_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
        
        iou_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_isles_medium_group.csv")
        dice_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_isles_medium_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_medium_group.idxmax()['IOU']
        best_num_timesteps_medium_group, best_threshold_medium_group, best_median_filter_size_medium_group, best_erosion_dilation_iterations_medium_group, best_binary_fill_holes_medium_group = best_params

        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps_medium_group, best_median_filter_size_medium_group, best_threshold_medium_group, best_erosion_dilation_iterations_medium_group, best_binary_fill_holes_medium_group]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_isles_medium_group.csv", index=False)

        metrics_result_text = "Medium group\n"
        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_medium_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_medium_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_medium_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_medium_group} "
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes_medium_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

        os.makedirs(ANOMALY_MAPS_DIR+"small/", exist_ok=True)
        # --------------------------------- small group

        iou_scores_df_small_group, dice_scores_df_small_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/", ROOT_DIR+"datasets/final_flair_dataset_small/brats_masks_registered/", len(ano_dataset.test_anomaly_small_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
        
        iou_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_isles_small_group.csv")
        dice_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_isles_small_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_small_group.idxmax()['IOU']
        best_num_timesteps_small_group, best_threshold_small_group, best_median_filter_size_small_group, best_erosion_dilation_iterations_small_group, best_binary_fill_holes_small_group = best_params

        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps_small_group, best_median_filter_size_small_group, best_threshold_small_group, best_erosion_dilation_iterations_small_group, best_binary_fill_holes_small_group]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_isles_small_group.csv", index=False)
        
        metrics_result_text = "Small group\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_small_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_small_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_small_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_small_group} "
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes_small_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)
    
            

    if args.dataset["test"] == "soop":
        
        # --------------------------------- large group
        os.makedirs(ANOMALY_MAPS_DIR+"large/", exist_ok=True)

        iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"large/", ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
        
        iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop_large_group.csv")
        dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop_large_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_large_group.idxmax()['IOU']
        best_num_timesteps_large_group, best_threshold_large_group, best_median_filter_size_large_group, best_erosion_dilation_iterations_large_group, best_binary_fill_holes_large_group = best_params
        
        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps_large_group, best_median_filter_size_large_group, best_threshold_large_group, best_erosion_dilation_iterations_large_group, best_binary_fill_holes_large_group]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_soop_large_group.csv", index=False)

        metrics_result_text = "Large group\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group} "
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes_large_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

        # --------------------------------- medium group
        os.makedirs(ANOMALY_MAPS_DIR+"medium/", exist_ok=True)

        iou_scores_df_medium_group, dice_scores_df_medium_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/", ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_medium_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
        
        iou_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop_medium_group.csv")
        dice_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop_medium_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_medium_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations, best_binary_fill_holes = best_params

        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps_medium_group, best_median_filter_size_medium_group, best_threshold_medium_group, best_erosion_dilation_iterations_medium_group, best_binary_fill_holes_medium_group]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_soop_medium_group.csv", index=False)
        
        metrics_result_text = "Medium group\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations} "
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

        # --------------------------------- small group
        os.makedirs(ANOMALY_MAPS_DIR+"small/", exist_ok=True)

        iou_scores_df_small_group, dice_scores_df_small_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/", ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_small_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
        

        iou_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop_small_group.csv")
        dice_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop_small_group.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_small_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations, best_binary_fill_holes = best_params

        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps_small_group, best_median_filter_size_small_group, best_threshold_small_group, best_erosion_dilation_iterations_small_group, best_binary_fill_holes_small_group]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_soop_small_group.csv", index=False)

        metrics_result_text = "Small group\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations} "
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)
    
    if args.dataset["test"] == "soop_fast":
        
        # --------------------------------- large group

        iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS, ROOT_DIR+"datasets/final_soop_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
        
        iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_soop_fast.csv")
        dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_soop_fast.csv")

        # Find the best parameters based on IOU score
        best_params = iou_scores_df_large_group.idxmax()['IOU']
        best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations, best_binary_fill_holes = best_params

        # Save best parameters to CSV
        best_params_df = pd.DataFrame({
            'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
            'value': [best_num_timesteps_large_group, best_median_filter_size_large_group, best_threshold_large_group, best_erosion_dilation_iterations_large_group, best_binary_fill_holes_large_group]
        })
        best_params_df.to_csv(SUB_EXPERIMENT_DIR+"best_params_soop_fast_large_group.csv", index=False)

        metrics_result_text = "Large group (SOOP Fast)\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
        metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations} "
        metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

