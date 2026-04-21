"""
This file only works for 3D slice by slice inference.
"""

import os
import glob
import sys
from pathlib import Path

sys.path.append("../..")
#import opensimplex

from datasets import anomaly_datasets

#from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
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
from make_anomaly_maps import make_anomaly_maps

import utils.custom_transforms as custom_transforms

import utils.simplex_ddpm as simplex_ddpm
import utils.thor_ddpm as thor_ddpm
import utils.scores as scores

from utils.utils import *

from compute_metrics_anomaly_detection import compute_metrics


from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

DEVICE_TYPE = "cuda:0"



def launch_anomaly_detection_inference(args, no_abs_value=False, nb_inferences=1):
    # Two parts : the first 50% of the test data is used to select the best noise timestep value and best threshold.
    # The second 50% is used to compute the final IOU and DICE metrics with these best values.
    DEVICE_TYPE = "cuda:0"
    device = torch.device(DEVICE_TYPE)

    set_determinism(0)
    
    dtprint("launching anomaly detection inference with the following settings:")
    dtprint(f"no_abs_value: {no_abs_value}")
    dtprint(f"nb_inferences: {nb_inferences}")

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    SUB_EXPERIMENT_DIR = f"{ROOT_DIR}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/"
    
    ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params/"
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/for_combine_experiment/" # final anomaly maps with best params
    dtprint(f"Anomaly maps best params will be saved in {ANOMALY_MAPS_DIR_SELECT_PARAMS}")
    os.makedirs(ANOMALY_MAPS_DIR_SELECT_PARAMS, exist_ok=True)
    os.makedirs(ANOMALY_MAPS_DIR, exist_ok=True)


    model_path = f"{args.root_dir}/AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/{SUB_EXPERIMENT_NAME}_best_model.pth"

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


    test_masks_transforms = transforms.Compose(
        [
            transforms.LoadImage(),
            transforms.EnsureChannelFirst(),
            transforms.ResizeWithPadOrCrop(spatial_size=(args.image_size, args.image_size, args.image_size)),
            custom_transforms.SetBackgroundToZero()
        ]
    )
       

    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])

    # ------------------------ Compute the raw anomaly maps and save them as nifti files ------------------------ #
    # So that they can be used to compute metrics later with different postprocessing steps without having to recompute the anomaly maps each time.

    model = define_instance(args, "network_def").to(device)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE_TYPE))
    model.eval()


    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])

    # ------------------------ SOOP dataset ------------------------ #
    if args.dataset["test"] == "soop":
        
        group = "large"

        ano_dataset = anomaly_datasets.SOOP(args, batch_size=16, num_workers=4, groups_to_load=[group])

        if "flair" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 150
            best_num_timesteps_medium_group = 130
        elif "adc" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 90
            best_num_timesteps_medium_group = 90
        
        

        if no_abs_value:
            output_folder = f"{ANOMALY_MAPS_DIR}{group}_no_abs_value/"
            os.makedirs(output_folder, exist_ok=True)
            dtprint("No abs value save anomaly maps: not yet implemented")
        else:
            if nb_inferences == 1:
                output_folder = f"{ANOMALY_MAPS_DIR}{group}/"
            else:
                output_folder = f"{ANOMALY_MAPS_DIR}{group}_{nb_inferences}x_inference/"
            
            os.makedirs(output_folder, exist_ok=True)
            dtprint(f"Computing anomaly maps for the {group} group, {nb_inferences}x_inference with best num timesteps: "+str(best_num_timesteps_large_group))
            dtprint(f"Output folder: {output_folder}")
            if group == "large":
                make_anomaly_maps(args, model, device, 
                                  infer_scheduler, 
                                  ano_dataset.test_anomaly_large_loader_metrics, 
                                  ano_dataset.test_anomaly_large_images_metrics, 
                                  best_num_timesteps_large_group, 
                                  output_folder, 
                                  replace_existing_files=False)
            elif group == "medium":
                make_anomaly_maps(args, model, device, 
                                  infer_scheduler, 
                                  ano_dataset.test_anomaly_medium_loader_metrics, 
                                  ano_dataset.test_anomaly_medium_images_metrics, 
                                  best_num_timesteps_medium_group, 
                                  output_folder, 
                                  replace_existing_files=False)
        
    
    if args.dataset["test"] == "healthy_test_set":

        if "flair" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 150
            best_median_filter_size_large_group=5
            best_threshold_large_group=0.04
            best_erosion_dilation_iterations_large_group=2

            test_reconstruction_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/final_flair_dataset_small_added_oasis/test.csv")
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


        elif "adc" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 100
            best_median_filter_size_large_group=5
            best_threshold_large_group=0.06
            best_erosion_dilation_iterations_large_group=2
        
            test_reconstruction_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/final_adc_dataset_small_added_ixi/test.csv")
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
        

        if no_abs_value:
            dir_name = ANOMALY_MAPS_DIR+"healthy_test_set_no_abs_value/"
            os.makedirs(dir_name, exist_ok=True)
            final_scores = compute_metrics(args, model, device, dir_name, infer_scheduler, test_reconstruction_loader, test_reconstruction_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
        else:
            dir_name = ANOMALY_MAPS_DIR+"healthy_test_set/"
            os.makedirs(dir_name, exist_ok=True)
            final_scores = compute_metrics(args, model, device, dir_name, infer_scheduler, test_reconstruction_loader, test_reconstruction_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
        
        metrics_result_text = "".join([f"{key}: mean {final_scores[key][0]} 95% CI [{final_scores[key][1]} - {final_scores[key][2]}]\n" for key in final_scores])

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)

    if args.dataset["test"] == "aini-stroke_ait":

        # images with failed registration
        bad_images_flair = ["aini-stroke_15092", "aini-stroke_17043", "aini-stroke_18254"] # registration problems
        bad_images_adc = ["aini-stroke_13607", "aini-stroke_21053"]


        if "flair" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 150
            best_median_filter_size_large_group=5
            best_threshold_large_group=0.06
            best_erosion_dilation_iterations_large_group=2

            
            #test_reconstruction_datalist = sorted(test_reconstruction_images_path)
            test_ait_datalist = os.listdir(ROOT_DIR+"datasets/aini-stroke_ait/flair_registered/")
            test_ait_datalist = [os.path.join(ROOT_DIR+"datasets/aini-stroke_ait/flair_registered/", img) for img in test_ait_datalist if img.split('.')[0] not in bad_images_flair]

            #test_unhealthy_datalist = test_unhealthy_images_path

            batch_size = args.dataset["batch_size"]
            num_workers = args.dataset["num_workers"]


            # transforms
            test_ait_transforms = define_instance(args, "val_transforms")
            test_ait_ds = CacheDataset(data=test_ait_datalist, transform=test_ait_transforms)


            test_ait_loader = DataLoader(
                test_ait_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )


        elif "adc" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 100
            best_median_filter_size_large_group=5
            best_threshold_large_group=0.06
            best_erosion_dilation_iterations_large_group=2
        
            #test_reconstruction_datalist = sorted(test_reconstruction_images_path)
            test_ait_datalist = os.listdir(ROOT_DIR+"datasets/aini-stroke_ait/adc_registered/")
            test_ait_datalist = [os.path.join(ROOT_DIR+"datasets/aini-stroke_ait/adc_registered/", img) for img in test_ait_datalist if img.split('.')[0] not in bad_images_adc]


            #test_unhealthy_datalist = test_unhealthy_images_path

            batch_size = args.dataset["batch_size"]
            num_workers = args.dataset["num_workers"]


            # transforms
            test_ait_transforms = define_instance(args, "val_transforms")
            test_ait_ds = CacheDataset(data=test_ait_datalist, transform=test_ait_transforms)


            test_ait_loader = DataLoader(
                test_ait_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
            )
        

        if no_abs_value:
            dir_name = ANOMALY_MAPS_DIR+"ait_no_abs/"
            os.makedirs(dir_name, exist_ok=True)
            final_scores = compute_metrics(args, model, device, dir_name, infer_scheduler, test_ait_loader, test_ait_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
        else:
            dir_name = ANOMALY_MAPS_DIR+"ait/"
            os.makedirs(dir_name, exist_ok=True)
            final_scores = compute_metrics(args, model, device, dir_name, infer_scheduler, test_ait_loader, test_ait_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
        
        metrics_result_text = "".join([f"{key}: mean {final_scores[key][0]} 95% CI [{final_scores[key][1]} - {final_scores[key][2]}]\n" for key in final_scores])

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group}"
        metrics_result_text += "\n"
        tprint(metrics_result_text)