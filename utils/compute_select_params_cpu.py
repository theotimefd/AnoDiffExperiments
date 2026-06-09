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
from itertools import product



def compute_select_params_multithreaded(args, 
                                        anomaly_maps_folder, 
                                        masks_folder, 
                                        total_nb_images, 
                                        num_timesteps_to_try, 
                                        thresholds_to_try, 
                                        median_filter_sizes_to_try, 
                                        erosion_dilation_iterations_to_try, 
                                        binary_fill_holes_to_try):
    """Compute parameter-search scores for a folder of anomaly maps.

    This variant keeps all score tables in memory. It evaluates every
    combination of timestep, threshold, median filter size, erosion/dilation
    iterations, and binary fill-holes choice for each anomaly map file, then
    accumulates the IOU and DICE scores across all processed files.

    Args:
        args: Experiment configuration namespace used by downstream scoring
            helpers.
        anomaly_maps_folder: Directory containing saved anomaly map NIfTI
            files.
        masks_folder: Directory containing the corresponding ground-truth mask
            files.
        total_nb_images: Number of images used to normalize the accumulated
            scores.
        num_timesteps_to_try: Iterable of diffusion timesteps to evaluate.
        thresholds_to_try: Iterable of anomaly-score thresholds to evaluate.
        median_filter_sizes_to_try: Iterable of median filter sizes to
            evaluate. Use ``-1`` to disable median filtering.
        erosion_dilation_iterations_to_try: Iterable of morphological iteration
            counts to evaluate.
        binary_fill_holes_to_try: Iterable of binary fill-holes flags to
            evaluate.

    Returns:
        A tuple ``(iou_scores_df, dice_scores_df)`` of pandas DataFrames whose
        MultiIndex spans the full parameter grid and whose values are the mean
        IOU and DICE scores over all processed images.
    """
    
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
    
    dtprint(f"num timesteps to try: {num_timesteps_to_try}")

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

    max_workers = min(64, mp.cpu_count()) # 48 cores per gpu https://gricad-doc.univ-grenoble-alpes.fr/hpc/kraken/kraken/#the-kraken-platform
    dtprint(f"Using max_workers={max_workers} for multiprocessing")
    ctx = mp.get_context("spawn")

    if len(anomaly_files) == 1 or max_workers == 1:
        results = [process_func(file_name) for file_name in anomaly_files]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            futures = {executor.submit(process_func, file_name): file_name for file_name in anomaly_files}
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
    iou_scores_df = iou_scores_df / total_nb_images
    dice_scores_df = dice_scores_df / total_nb_images

    return iou_scores_df, dice_scores_df


def _initialize_param_search_checkpoint_csv(checkpoint_csv_path,
                                            num_timesteps_to_try,
                                            thresholds_to_try,
                                            median_filter_sizes_to_try,
                                            erosion_dilation_iterations_to_try,
                                            binary_fill_holes_to_try):
    """Create or reset the checkpoint CSV used by the disk-backed search.

    The generated file contains one row per parameter combination and stores
    running sums for IOU and DICE. Those values are updated incrementally after
    each processed anomaly map so the full score table does not need to stay in
    memory.

    Args:
        checkpoint_csv_path: Path to the checkpoint CSV to create.
        num_timesteps_to_try: Iterable of diffusion timesteps to enumerate.
        thresholds_to_try: Iterable of thresholds to enumerate.
        median_filter_sizes_to_try: Iterable of median filter sizes to
            enumerate.
        erosion_dilation_iterations_to_try: Iterable of morphology iteration
            counts to enumerate.
        binary_fill_holes_to_try: Iterable of binary fill-holes flags to
            enumerate.
    """
    with open(checkpoint_csv_path, "w", newline="", encoding="utf-8") as checkpoint_file:
        writer = csv.writer(checkpoint_file)
        writer.writerow([
            "timesteps",
            "threshold",
            "median_filter_size",
            "erosion_dilation_iterations",
            "binary_fill_holes",
            "iou_sum",
            "dice_sum",
        ])

        for timesteps, threshold, median_filter_size, erosion_dilation_iterations, binary_fill_holes in product(
            num_timesteps_to_try,
            thresholds_to_try,
            median_filter_sizes_to_try,
            erosion_dilation_iterations_to_try,
            binary_fill_holes_to_try,
        ):
            writer.writerow([
                int(timesteps),
                format(float(threshold), ".17g"),
                int(median_filter_size),
                int(erosion_dilation_iterations),
                int(binary_fill_holes),
                0.0,
                0.0,
            ])


def _checkpoint_param_search_csv(checkpoint_csv_path, local_iou_scores, local_dice_scores):
    """Merge one image's local scores into the persistent checkpoint CSV.

    The checkpoint file is rewritten atomically via a temporary file. Only the
    rows present in the local result dictionaries are updated, which is
    important because each anomaly map only contributes scores for one timestep
    value.

    Args:
        checkpoint_csv_path: Path to the persistent checkpoint CSV.
        local_iou_scores: Mapping from parameter tuple to the IOU score sum for
            one processed anomaly map.
        local_dice_scores: Mapping from parameter tuple to the DICE score sum
            for one processed anomaly map.
    """
    checkpoint_tmp_path = f"{checkpoint_csv_path}.tmp"

    with open(checkpoint_csv_path, "r", newline="", encoding="utf-8") as checkpoint_file, \
         open(checkpoint_tmp_path, "w", newline="", encoding="utf-8") as checkpoint_tmp_file:
        reader = csv.DictReader(checkpoint_file)
        writer = csv.DictWriter(checkpoint_tmp_file, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            idx = (
                int(row["timesteps"]),
                float(row["threshold"]),
                int(row["median_filter_size"]),
                int(row["erosion_dilation_iterations"]),
                int(row["binary_fill_holes"]),
            )

            # Each processed anomaly file only contributes scores for one timestep value.
            # Keep all other rows unchanged in this checkpoint file.
            if idx in local_iou_scores:
                row["iou_sum"] = format(float(row["iou_sum"]) + float(local_iou_scores[idx]), ".17g")
                row["dice_sum"] = format(float(row["dice_sum"]) + float(local_dice_scores[idx]), ".17g")
            writer.writerow(row)

    os.replace(checkpoint_tmp_path, checkpoint_csv_path)


def _finalize_param_search_checkpoint_csv(checkpoint_csv_path,
                                           total_nb_images,
                                           iou_scores_csv_path,
                                           dice_scores_csv_path,
                                           best_params_csv_path):
    """Convert checkpointed sums into final score tables and best parameters.

    This reads the accumulated per-parameter sums, divides them by the total
    number of images, writes final IOU and DICE CSV files, and stores the best
    parameter combination in a dedicated CSV.

    Args:
        checkpoint_csv_path: Path to the checkpoint CSV containing running
            sums.
        total_nb_images: Number of images used to normalize the sums.
        iou_scores_csv_path: Output path for the final IOU score table.
        dice_scores_csv_path: Output path for the final DICE score table.
        best_params_csv_path: Output path for the CSV containing the best
            parameter set.

    Returns:
        A tuple ``(best_params, best_iou_score)`` where ``best_params`` is the
        parameter tuple with the highest IOU score and ``best_iou_score`` is
        that normalized score.
    """
    best_params = None
    best_iou_score = -np.inf

    with open(checkpoint_csv_path, "r", newline="", encoding="utf-8") as checkpoint_file, \
         open(iou_scores_csv_path, "w", newline="", encoding="utf-8") as iou_scores_file, \
         open(dice_scores_csv_path, "w", newline="", encoding="utf-8") as dice_scores_file:
        reader = csv.DictReader(checkpoint_file)

        iou_writer = csv.writer(iou_scores_file)
        dice_writer = csv.writer(dice_scores_file)

        iou_writer.writerow([
            "timesteps",
            "threshold",
            "median_filter_size",
            "erosion_dilation_iterations",
            "binary_fill_holes",
            "IOU",
        ])
        dice_writer.writerow([
            "timesteps",
            "threshold",
            "median_filter_size",
            "erosion_dilation_iterations",
            "binary_fill_holes",
            "DICE",
        ])

        for row in reader:
            idx = (
                int(row["timesteps"]),
                float(row["threshold"]),
                int(row["median_filter_size"]),
                int(row["erosion_dilation_iterations"]),
                int(row["binary_fill_holes"]),
            )

            iou_score = float(row["iou_sum"]) / total_nb_images
            dice_score = float(row["dice_sum"]) / total_nb_images

            iou_writer.writerow([
                idx[0],
                format(idx[1], ".17g"),
                idx[2],
                idx[3],
                idx[4],
                format(iou_score, ".17g"),
            ])
            dice_writer.writerow([
                idx[0],
                format(idx[1], ".17g"),
                idx[2],
                idx[3],
                idx[4],
                format(dice_score, ".17g"),
            ])

            if iou_score > best_iou_score:
                best_iou_score = iou_score
                best_params = idx

    best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations, best_binary_fill_holes = best_params
    best_params_df = pd.DataFrame({
        "parameter": ["num_timesteps", "median_filter_size", "threshold", "erosion_dilation_iterations", "binary_fill_holes"],
        "value": [best_num_timesteps, best_median_filter_size, best_threshold, best_erosion_dilation_iterations, best_binary_fill_holes],
    })
    best_params_df.to_csv(best_params_csv_path, index=False)

    return best_params, best_iou_score


def compute_select_params_multithreaded_checkpointed(args,
                                                     anomaly_maps_folder,
                                                     masks_folder,
                                                     total_nb_images,
                                                     num_timesteps_to_try,
                                                     thresholds_to_try,
                                                     median_filter_sizes_to_try,
                                                     erosion_dilation_iterations_to_try,
                                                     binary_fill_holes_to_try,
                                                     output_dir,
                                                     output_prefix="param_search"):
    """Compute parameter-search scores while checkpointing partial results to disk.

    This is the disk-backed counterpart to
    :func:`compute_select_params_multithreaded`. It evaluates the same parameter
    grid, but after each processed anomaly map it updates a checkpoint CSV
    instead of keeping all intermediate sums in memory.

    Args:
        args: Experiment configuration namespace used by downstream scoring
            helpers.
        anomaly_maps_folder: Directory containing saved anomaly map NIfTI
            files.
        masks_folder: Directory containing the corresponding ground-truth mask
            files.
        total_nb_images: Number of images used to normalize the accumulated
            scores.
        num_timesteps_to_try: Iterable of diffusion timesteps to evaluate.
        thresholds_to_try: Iterable of anomaly-score thresholds to evaluate.
        median_filter_sizes_to_try: Iterable of median filter sizes to
            evaluate.
        erosion_dilation_iterations_to_try: Iterable of morphology iteration
            counts to evaluate.
        binary_fill_holes_to_try: Iterable of binary fill-holes flags to
            evaluate.
        output_dir: Directory where the checkpoint and final CSV files will be
            written.
        output_prefix: Prefix used for all generated CSV filenames.

    Returns:
        A dictionary containing the generated CSV paths, the best parameter
        tuple, and the corresponding best IOU score.
    """

    tprint("launching compute_select_params_multithreaded_checkpointed")

    os.makedirs(output_dir, exist_ok=True)

    checkpoint_csv_path = os.path.join(output_dir, f"{output_prefix}_checkpoint.csv")
    iou_scores_csv_path = os.path.join(output_dir, f"{output_prefix}_iou_scores.csv")
    dice_scores_csv_path = os.path.join(output_dir, f"{output_prefix}_dice_scores.csv")
    best_params_csv_path = os.path.join(output_dir, f"{output_prefix}_best_params.csv")

    _initialize_param_search_checkpoint_csv(
        checkpoint_csv_path,
        num_timesteps_to_try,
        thresholds_to_try,
        median_filter_sizes_to_try,
        erosion_dilation_iterations_to_try,
        binary_fill_holes_to_try,
    )

    dtprint(f"num timesteps to try: {num_timesteps_to_try}")

    anomaly_files = [entry.name for entry in os.scandir(anomaly_maps_folder) if entry.is_file() and entry.name.endswith(".nii.gz")]
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
        binary_fill_holes_to_try=binary_fill_holes_to_try,
    )

    max_workers = min(64, mp.cpu_count())
    dtprint(f"Using max_workers={max_workers} for multiprocessing")
    ctx = mp.get_context("spawn")

    if len(anomaly_files) == 1 or max_workers == 1:
        results_iterator = (process_func(file_name) for file_name in anomaly_files)
        for local_iou_scores, local_dice_scores in tqdm(results_iterator, total=len(anomaly_files), desc="Processing anomaly maps"):
            _checkpoint_param_search_csv(checkpoint_csv_path, local_iou_scores, local_dice_scores)
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            for local_iou_scores, local_dice_scores in tqdm(executor.map(process_func, anomaly_files), total=len(anomaly_files), desc="Processing anomaly maps"):
                _checkpoint_param_search_csv(checkpoint_csv_path, local_iou_scores, local_dice_scores)

    dtprint("multiprocesses all finished")

    best_params, best_iou_score = _finalize_param_search_checkpoint_csv(
        checkpoint_csv_path,
        total_nb_images,
        iou_scores_csv_path,
        dice_scores_csv_path,
        best_params_csv_path,
    )

    dtprint(f"Best IOU score: {best_iou_score:.6f}")

    return {
        "checkpoint_csv_path": checkpoint_csv_path,
        "iou_scores_csv_path": iou_scores_csv_path,
        "dice_scores_csv_path": dice_scores_csv_path,
        "best_params_csv_path": best_params_csv_path,
        "best_params": best_params,
        "best_iou_score": best_iou_score,
    }


def launch_compute_select_params_cpu(args):
    """Run the CPU-based parameter search used by the inference pipeline.

    This function prepares the experiment-specific directories, loads the test
    dataset and diffusion scheduler, and dispatches the parameter search for
    the configured dataset split. For ISLES and SOOP-like datasets it handles
    the large, medium, and small anomaly groups separately; for BRATS it runs a
    single global search.

    Args:
        args: Experiment configuration object with nested fields such as
            ``root_dir``, ``experiment_name``, ``sub_experiment_name``,
            ``dataset``, ``noise``, and ``anomaly_detection_param_search``.

    Returns:
        None. The function writes CSV outputs and diagnostic text files to the
        experiment directory structure.
    """

    set_determinism(0)

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    SUB_EXPERIMENT_DIR = f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/"
    
    nb_inferences = args.nb_inferences

    try:
        is_thor = args.thor["enable"]
    except:
        is_thor = False
    
    dtprint(f"THOR enabled: {is_thor}")

    if is_thor == False:
        ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params/"
        ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/" # final anomaly maps with best params
    else:
        ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params_thor/"
        ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_thor/" # final anomaly maps with best params
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
        dtprint(metrics_result_text)


    if args.dataset["test"] == "isles" or args.dataset["test"] == "soop":
        
        # --------------------------------- large group
        os.makedirs(ANOMALY_MAPS_DIR+"large/", exist_ok=True)

        best_params_csv_path = SUB_EXPERIMENT_DIR+f"best_params_{args.dataset['test']}_large_group.csv"

        if not os.path.exists(best_params_csv_path):
            dtprint("Large group")
            dtprint(f"Computing best parameters with CPU...")
            #iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"large/", ROOT_DIR+f"datasets/final_{args.dataset['test']}_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
            iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded_checkpointed(args, 
                                                                                                                     ANOMALY_MAPS_DIR_SELECT_PARAMS+"large/", 
                                                                                                                     ROOT_DIR+f"datasets/final_{args.dataset['test']}_dataset_small/masks_combined_registered/", 
                                                                                                                     len(ano_dataset.test_anomaly_large_images_select_params), 
                                                                                                                     num_timesteps_to_try, 
                                                                                                                     thresholds_to_try, 
                                                                                                                     median_filter_sizes_to_try, 
                                                                                                                     erosion_dilation_iterations_to_try, 
                                                                                                                     binary_fill_holes_to_try,
                                                                                                                     output_dir=SUB_EXPERIMENT_DIR)
        
            iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_{args.dataset['test']}_large_group.csv")
            dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_{args.dataset['test']}_large_group.csv")

            # Find the best parameters based on IOU score
            best_params = iou_scores_df_large_group.idxmax()['IOU']
            best_num_timesteps_large_group, best_threshold_large_group, best_median_filter_size_large_group, best_erosion_dilation_iterations_large_group, best_binary_fill_holes_large_group = best_params
            
            # Save best parameters to CSV
            best_params_df = pd.DataFrame({
                'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
                'value': [best_num_timesteps_large_group, best_median_filter_size_large_group, best_threshold_large_group, best_erosion_dilation_iterations_large_group, best_binary_fill_holes_large_group]
            })
            best_params_df.to_csv(best_params_csv_path, index=False)

            metrics_result_text = "Large group\n"

            metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
            metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
            metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
            metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group} "
            metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes_large_group}"
            metrics_result_text += "\n"
            dtprint(metrics_result_text)

        # --------------------------------- medium group
        os.makedirs(ANOMALY_MAPS_DIR+"medium/", exist_ok=True)
        best_params_csv_path = SUB_EXPERIMENT_DIR+f"best_params_{args.dataset['test']}_medium_group.csv"

        if not os.path.exists(best_params_csv_path):
            dtprint("Medium group")
            dtprint(f"Computing best parameters with CPU...")
            #iou_scores_df_medium_group, dice_scores_df_medium_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/", ROOT_DIR+f"datasets/final_{args.dataset['test']}_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_medium_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
            iou_scores_df_medium_group, dice_scores_df_medium_group = compute_select_params_multithreaded_checkpointed(args, 
                                                                                                                       ANOMALY_MAPS_DIR_SELECT_PARAMS+"medium/", 
                                                                                                                       ROOT_DIR+f"datasets/final_{args.dataset['test']}_dataset_small/masks_combined_registered/", 
                                                                                                                       len(ano_dataset.test_anomaly_medium_images_select_params), 
                                                                                                                       num_timesteps_to_try, 
                                                                                                                       thresholds_to_try, 
                                                                                                                       median_filter_sizes_to_try, 
                                                                                                                       erosion_dilation_iterations_to_try, 
                                                                                                                       binary_fill_holes_to_try,
                                                                                                                       output_dir=SUB_EXPERIMENT_DIR)

            iou_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_{args.dataset['test']}_medium_group.csv")
            dice_scores_df_medium_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_{args.dataset['test']}_medium_group.csv")

            # Find the best parameters based on IOU score
            best_params = iou_scores_df_medium_group.idxmax()['IOU']
            best_num_timesteps_medium_group, best_threshold_medium_group, best_median_filter_size_medium_group, best_erosion_dilation_iterations_medium_group, best_binary_fill_holes_medium_group = best_params

            # Save best parameters to CSV
            best_params_df = pd.DataFrame({
                'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
                'value': [best_num_timesteps_medium_group, best_median_filter_size_medium_group, best_threshold_medium_group, best_erosion_dilation_iterations_medium_group, best_binary_fill_holes_medium_group]
            })
            best_params_df.to_csv(best_params_csv_path, index=False)
            
            metrics_result_text = "Medium group\n"

            metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_medium_group} "
            metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_medium_group} "
            metrics_result_text += f"Best Threshold: {best_threshold_medium_group:.4f} "
            metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_medium_group} "
            metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes_medium_group}"
            metrics_result_text += "\n"
            tprint(metrics_result_text)

        # --------------------------------- small group
        os.makedirs(ANOMALY_MAPS_DIR+"small/", exist_ok=True)
        best_params_csv_path = SUB_EXPERIMENT_DIR+f"best_params_{args.dataset['test']}_small_group.csv"

        if not os.path.exists(best_params_csv_path):
            dtprint("Small group")
            dtprint(f"Computing best parameters with CPU...")
            #iou_scores_df_small_group, dice_scores_df_small_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/", ROOT_DIR+f"datasets/final_{args.dataset['test']}_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_small_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)
            iou_scores_df_small_group, dice_scores_df_small_group = compute_select_params_multithreaded_checkpointed(args, 
                                                                                                                     ANOMALY_MAPS_DIR_SELECT_PARAMS+"small/", 
                                                                                                                     ROOT_DIR+f"datasets/final_{args.dataset['test']}_dataset_small/masks_combined_registered/", 
                                                                                                                     len(ano_dataset.test_anomaly_small_images_select_params), 
                                                                                                                     num_timesteps_to_try, 
                                                                                                                     thresholds_to_try, 
                                                                                                                     median_filter_sizes_to_try, 
                                                                                                                     erosion_dilation_iterations_to_try, 
                                                                                                                     binary_fill_holes_to_try,
                                                                                                                     output_dir=SUB_EXPERIMENT_DIR)
            

            iou_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_{args.dataset['test']}_small_group.csv")
            dice_scores_df_small_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_{args.dataset['test']}_small_group.csv")

            # Find the best parameters based on IOU score
            best_params = iou_scores_df_small_group.idxmax()['IOU']
            best_num_timesteps_small_group, best_threshold_small_group, best_median_filter_size_small_group, best_erosion_dilation_iterations_small_group, best_binary_fill_holes_small_group = best_params

            # Save best parameters to CSV
            best_params_df = pd.DataFrame({
                'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
                'value': [best_num_timesteps_small_group, best_median_filter_size_small_group, best_threshold_small_group, best_erosion_dilation_iterations_small_group, best_binary_fill_holes_small_group]
            })
            best_params_df.to_csv(best_params_csv_path, index=False)

            metrics_result_text = "Small group\n"

            metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_small_group} "
            metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_small_group} "
            metrics_result_text += f"Best Threshold: {best_threshold_small_group:.4f} "
            metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_small_group} "
            metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes_small_group}"
            metrics_result_text += "\n"
            tprint(metrics_result_text)
    
    if args.dataset["test"] == "soop_fast":
        
        # --------------------------------- large group
        
        best_params_csv_path = SUB_EXPERIMENT_DIR+f"best_params_{args.dataset['test']}.csv"

        if not os.path.exists(best_params_csv_path):
            dtprint(f"Computing best parameters with CPU...")
            iou_scores_df_large_group, dice_scores_df_large_group = compute_select_params_multithreaded(args, ANOMALY_MAPS_DIR_SELECT_PARAMS, ROOT_DIR+f"datasets/final_{args.dataset['test']}_dataset_small/masks_combined_registered/", len(ano_dataset.test_anomaly_large_images_select_params), num_timesteps_to_try, thresholds_to_try, median_filter_sizes_to_try, erosion_dilation_iterations_to_try, binary_fill_holes_to_try)

            iou_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"iou_scores_param_search_{args.dataset['test']}.csv")
            dice_scores_df_large_group.to_csv(SUB_EXPERIMENT_DIR+f"dice_scores_param_search_{args.dataset['test']}.csv")

            # Find the best parameters based on IOU score
            best_params = iou_scores_df_large_group.idxmax()['IOU']
            best_num_timesteps, best_threshold, best_median_filter_size, best_erosion_dilation_iterations, best_binary_fill_holes = best_params

            # Save best parameters to CSV
            best_params_df = pd.DataFrame({
                'parameter': ['num_timesteps', 'median_filter_size', 'threshold', 'erosion_dilation_iterations', 'binary_fill_holes'],
                'value': [best_num_timesteps, best_median_filter_size, best_threshold, best_erosion_dilation_iterations, best_binary_fill_holes]
            })
            best_params_df.to_csv(best_params_csv_path, index=False)

            metrics_result_text = "Large group (SOOP Fast)\n"

            metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
            metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
            metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
            metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations} "
            metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}"
            metrics_result_text += "\n"
            tprint(metrics_result_text)

