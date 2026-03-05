"""
This file only works for 3D slice by slice inference.
"""

import os
import glob
import sys
from pathlib import Path

from datasets import anomaly_datasets

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
from utils.utils import *
from make_anomaly_maps_optim import make_anomaly_maps_optim


DEVICE_TYPE = "cuda:0"



def launch_anomaly_detection_inference(args, no_abs_value=False):
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
    
    ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params/"
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/for_combine_experiment/" # final anomaly maps with best params
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
    

    # -------------------- define the data --------------------

    if args.dataset["test"] == "brats":
        ano_dataset = anomaly_datasets.BRATS(args)

    if args.dataset["test"] == "isles":
        ano_dataset = anomaly_datasets.ISLES(args)
    
    if args.dataset["test"] == "soop":
        ano_dataset = anomaly_datasets.SOOP(args)
    
    if args.dataset["test"] == "soop_fast":
        ano_dataset = anomaly_datasets.SOOP_Fast(args)
    

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
        
        # --------------------------------- large group
        if "flair" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 150

        elif "adc" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 90
        

        if no_abs_value:
            os.makedirs(ANOMALY_MAPS_DIR+"large_no_abs_value/", exist_ok=True)
            make_anomaly_maps_optim(args, model, device, infer_scheduler=infer_scheduler, image_loader=ano_dataset.test_anomaly_large_loader_metrics, image_paths=ano_dataset.test_anomaly_large_images_select_params, infer_timesteps=best_num_timesteps_large_group, output_folder=ANOMALY_MAPS_DIR+"large_no_abs_value/", replace_existing_files=False, no_abs_value=no_abs_value)
        else:
            os.makedirs(ANOMALY_MAPS_DIR+"large/", exist_ok=True)
            make_anomaly_maps_optim(args, model, device, infer_scheduler=infer_scheduler, image_loader=ano_dataset.test_anomaly_large_loader_metrics, image_paths=ano_dataset.test_anomaly_large_images_select_params, infer_timesteps=best_num_timesteps_large_group, output_folder=ANOMALY_MAPS_DIR+"large/", replace_existing_files=False, no_abs_value=no_abs_value)
        
        """metrics_result_text = f"Large group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group}"
        metrics_result_text += "\n"
        """
        #tprint(metrics_result_text)
    
    if args.dataset["test"] == "healthy_test_set":

        if "flair" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 150

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
            #make_anomaly_maps_optim(args, model, device, dir_name, infer_scheduler, test_reconstruction_loader, test_reconstruction_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
            
        else:
            dir_name = ANOMALY_MAPS_DIR+"healthy_test_set/"
            os.makedirs(dir_name, exist_ok=True)
            #mean_iou, std_iou, mean_dice, std_dice = inference(args, model, device, dir_name, infer_scheduler, test_reconstruction_loader, test_reconstruction_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
        
        make_anomaly_maps_optim(args, model, device, infer_scheduler=infer_scheduler, image_loader=test_reconstruction_loader, image_paths=test_reconstruction_datalist, infer_timesteps=best_num_timesteps_large_group, output_folder=dir_name, replace_existing_files=False, no_abs_value=no_abs_value)
        """
        metrics_result_text = f"Large group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group}"
        metrics_result_text += "\n"
        """
        #tprint(metrics_result_text)

    if args.dataset["test"] == "aini-stroke_ait":

        # images with failed registration
        bad_images_flair = ["aini-stroke_15092", "aini-stroke_17043", "aini-stroke_18254"] # registration problems
        bad_images_adc = ["aini-stroke_13607", "aini-stroke_21053"]


        if "flair" in args.dataset["name"].lower():
            best_num_timesteps_large_group = 150

            
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
            #mean_iou, std_iou, mean_dice, std_dice = inference(args, model, device, dir_name, infer_scheduler, test_ait_loader, test_ait_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
            
        else:
            dir_name = ANOMALY_MAPS_DIR+"ait/"
            os.makedirs(dir_name, exist_ok=True)
            #mean_iou, std_iou, mean_dice, std_dice = inference(args, model, device, dir_name, infer_scheduler, test_ait_loader, test_ait_datalist, None, timesteps=best_num_timesteps_large_group, threshold=best_threshold_large_group, median_filter_size=best_median_filter_size_large_group, erosion_dilation_iterations=best_erosion_dilation_iterations_large_group, no_abs_value=no_abs_value)
        
        make_anomaly_maps_optim(args, model, device, infer_scheduler=infer_scheduler, image_loader=test_ait_loader, image_paths=test_ait_datalist, infer_timesteps=best_num_timesteps_large_group, output_folder=dir_name, replace_existing_files=False, no_abs_value=no_abs_value)
        
        """
        metrics_result_text = f"Large group: mean IOU: {mean_iou:.4f} std: {std_iou:.4f} - mean DICE {mean_dice:.4f} std: {std_dice:.4f}\n"

        metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps_large_group} "
        metrics_result_text += f"Best Median Filter Size: {best_median_filter_size_large_group} "
        metrics_result_text += f"Best Threshold: {best_threshold_large_group:.4f} "
        metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations_large_group}"
        metrics_result_text += "\n"
        """
        #tprint(metrics_result_text)