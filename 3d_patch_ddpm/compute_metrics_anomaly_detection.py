import os

import sys

from pathlib import Path

from datasets import anomaly_datasets
from utils.compute_select_params_cpu import launch_compute_select_params_cpu
sys.path.append("../..")
#import opensimplex

#from torchvision.utils import save_image

import matplotlib.pyplot as plt
import numpy as np
import csv
import torch

from monai.utils import set_determinism
from torch.amp import autocast
from tqdm import tqdm


import nibabel as nib

from monai.networks.schedulers import DDPMScheduler



import pandas as pd

import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm
import utils.scores as scores

from utils.utils import *
from make_anomaly_maps_optim import make_anomaly_maps_optim, _run_patchwise_test_optim



from scipy.ndimage import median_filter, binary_erosion, binary_dilation, binary_fill_holes


def scale_intensity_from_histogram_peak(input_image, target_value=1.0):
    # to be used only on mri images with intensities between 0 and 1
    input_np = input_image.cpu().numpy()

    hist, bin_edges = np.histogram(input_np.flatten(), bins=100, range=(np.max(input_np)/15.0, 0.8))

    peak_value = bin_edges[np.argmax(hist)]

    normalized_image = input_image / peak_value * target_value

    return normalized_image


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



def show_summary_figure(args, device, model, infer_scheduler, 
                        image_loader, 
                        mask_loader, 
                        infer_timesteps, 
                        median_filter_size, 
                        threshold, 
                        erosion_dilation_iterations, 
                        binary_fill_holes_param,
                        metrics_result_text, 
                        ROOT_DIR, 
                        EXPERIMENT_NAME, 
                        SUB_EXPERIMENT_NAME):

    
    if args.noise["type"] == "simplex":
        simplexObj = simplex.Simplex_CLASS()


    for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))): # i=6 batch is nice
        if i>0:break

        test_anomaly_images = image_batch[..., image_batch.shape[-1]//2].to(device) # list of 2d images
        test_anomaly_masks = mask_batch[..., mask_batch.shape[-1]//2].to(device) # list of 2d images
        
        test_anomaly_masks[test_anomaly_masks>0.5] = 1.0
        test_anomaly_masks[test_anomaly_masks<=0.5] = 0.0

        final_anomaly_maps = torch.zeros_like(test_anomaly_images) # list of 2d images
        infered_maps = torch.zeros_like(test_anomaly_images) # list of 2d images

        with torch.no_grad():
            with autocast(device_type="cuda", enabled=True):

                stitched_pred = _run_patchwise_test_optim(
                    image_batch.to(device),
                    args.patch_size,
                    args.dataset["patch_overlap"],
                    args.dataset["patch_batch_size"],
                    args.noise["type"],
                    simplexObj,
                    model,
                    infer_scheduler,
                    infer_timesteps,
                    device,
                )

                for idx, infered_volume in enumerate(stitched_pred):
                    
                    normalized_infered_volume = torch.clamp(scale_intensity_from_histogram_peak(infered_volume, 2.0/7.0), 0.0, 1.0)
                    infered_maps[idx] = normalized_infered_volume[..., normalized_infered_volume.shape[-1]//2]
                    # make the anomaly map (difference between infered and original)
                    final_anomaly_map = torch.abs(normalized_infered_volume.to(device) - image_batch[idx].to(device))
                    final_anomaly_maps[idx] = final_anomaly_map[..., final_anomaly_map.shape[-1]//2]
                    
        

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
        infered_image_slice = infered_maps[idx, 0].cpu().numpy()
        
        axes[1, idx*2].imshow(infered_image_slice, cmap='gray', vmin=0, vmax=1)
        axes[1, idx*2].set_title(f'Inferred {idx+1}')
        axes[1, idx*2].axis('off')

        axes[1, idx*2+1].hist(infered_image_slice[infered_image_slice>0.01].flatten(), bins=50, color='blue', alpha=0.7, range=(0.0, 1.0))
        axes[1, idx*2+1].set_ylim(0, 2000)
        axes[1, idx*2+1].set_aspect('auto') # Set the aspect ratio to auto to match the imshow plot

        # Difference images
        difference_image = final_anomaly_maps[idx, 0].cpu().numpy()
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

        if erosion_dilation_iterations > 0:
            ano_segmentation_np = binary_erosion(ano_segmentation_np, iterations=erosion_dilation_iterations).astype(ano_segmentation_np.dtype)
            ano_segmentation_np = binary_dilation(ano_segmentation_np, iterations=erosion_dilation_iterations).astype(ano_segmentation_np.dtype)

        if binary_fill_holes_param==1:
            ano_segmentation_np = binary_fill_holes(ano_segmentation_np).astype(ano_segmentation_np.dtype)

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


def compute_metrics(args, model, device, ANOMALY_MAPS_DIR, 
                    infer_scheduler, 
                    image_loader, 
                    image_paths, 
                    mask_loader, 
                    timesteps, 
                    threshold, 
                    median_filter_size, 
                    erosion_dilation_iterations, 
                    binary_fill_holes, 
                    no_abs_value=False):
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
            final_scores: a dictionary containing the mean and confidence intervals for each metric, with the following format:
            {
                "iou": [mean_iou, lower_iou, upper_iou],
                "dice": [mean_dice, lower_dice, upper_dice],
                "hausdorff": [mean_hausdorff, lower_hausdorff, upper_hausdorff],
                "precision": [mean_precision, lower_precision, upper_precision],
                "recall": [mean_recall, lower_recall, upper_recall],
                "f1": [mean_f1, lower_f1, upper_f1]
            }
    """
    
    iou_scores = []
    dice_scores = []
    hausdorff_distances = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    no_masks = False
    if mask_loader is None:
        mask_loader = image_loader # hack so the for loop works
        no_masks = True

    
    simplexObj = None

    if args.noise["type"] == "simplex":
        simplexObj = simplex.Simplex_CLASS()
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], 
                                                            schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], 
                                                            persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])

    # first we save all the anomaly maps
    # then we compute the metrics on them

    # for every batch

    make_anomaly_maps_optim(args, model, device, infer_scheduler, image_loader, image_paths, timesteps, ANOMALY_MAPS_DIR, no_abs_value=no_abs_value)

    if no_abs_value:
        return {}

    if not no_masks:

        # make the segmentation map with threshold

        for i,(image_batch, mask_batch) in enumerate(tqdm(zip(image_loader, mask_loader))):

            test_images = image_batch.to(device)
            test_masks = mask_batch.to(device)
            test_masks[test_masks>0.5] = 1.0
            test_masks[test_masks<=0.5] = 0.0
            
            infered_batch = torch.zeros_like(image_batch)

            volumes = test_images.shape[0]

            # load infered images from saved files
            for idx in range(volumes): 

                image_id = i*test_images.shape[0] + idx
                patient_id = os.path.basename(image_paths[image_id]).replace(".nii.gz", "")
                
                reconstructed_map_path = ANOMALY_MAPS_DIR+f"{patient_id}_t_{timesteps}.nii.gz"
                
                # Load the reconstructed NIfTI file
                reconstructed_nifti = nib.load(reconstructed_map_path)
                reconstructed_data = reconstructed_nifti.get_fdata()

                # Convert to torch tensor and assign to infered_batch
                infered_batch[idx : idx + 1] = torch.from_numpy(reconstructed_data).unsqueeze(0).unsqueeze(0).to(device).float()

            # apply median filter if specified
            if median_filter_size is not None and median_filter_size > 0:
                final_anomaly_map_np = infered_batch.cpu().numpy()
                for b in range(final_anomaly_map_np.shape[0]):
                    final_anomaly_map_np[b] = median_filter(final_anomaly_map_np[b], size=median_filter_size)
                final_anomaly_map = torch.from_numpy(final_anomaly_map_np).to(device)

            ano_segmentation = final_anomaly_map > threshold

            # perform erosion if specified
            if erosion_dilation_iterations > 0:
                ano_segmentation_np = ano_segmentation.cpu().numpy()
                for b in range(ano_segmentation_np.shape[0]):
                    ano_segmentation_np[b,0] = binary_erosion(ano_segmentation_np[b,0], iterations=erosion_dilation_iterations)
                    ano_segmentation_np[b,0] = binary_dilation(ano_segmentation_np[b,0], iterations=erosion_dilation_iterations)
                ano_segmentation = torch.from_numpy(ano_segmentation_np).to(device)
            
            if binary_fill_holes == 1:
                ano_segmentation_np = ano_segmentation.cpu().numpy()
                for b in range(ano_segmentation_np.shape[0]):
                    ano_segmentation_np[b,0] = binary_fill_holes(ano_segmentation_np[b,0])
                ano_segmentation = torch.from_numpy(ano_segmentation_np).to(device)

            iou_scores_batch, dice_scores_batch, hausdorff_distances_batch, precision_scores_batch, recall_scores_batch, f1_scores_batch = scores.compute_scores(ano_segmentation, test_masks)

            # put all the batch score lists into one big list
            iou_scores.append(iou_scores_batch)
            dice_scores.append(dice_scores_batch)
            hausdorff_distances.append(hausdorff_distances_batch)
            precision_scores.append(precision_scores_batch)
            recall_scores.append(recall_scores_batch)
            f1_scores.append(f1_scores_batch)

    if no_masks:
        return {}

    
    mean_iou, lower_iou, upper_iou = scores.make_confidence_intervals(np.array(iou_scores).flatten())
    
    mean_dice, lower_dice, upper_dice = scores.make_confidence_intervals(np.array(dice_scores).flatten())

    mean_hausdorff, lower_hausdorff, upper_hausdorff = scores.make_confidence_intervals(np.array(hausdorff_distances).flatten())

    mean_precision, lower_precision, upper_precision = scores.make_confidence_intervals(np.array(precision_scores).flatten())

    mean_recall, lower_recall, upper_recall = scores.make_confidence_intervals(np.array(recall_scores).flatten())
    
    mean_f1, lower_f1, upper_f1 = scores.make_confidence_intervals(np.array(f1_scores).flatten())


    final_scores = {
        "iou": [np.round(mean_iou, 4), np.round(lower_iou, 4), np.round(upper_iou, 4)],
        "dice": [np.round(mean_dice, 4), np.round(lower_dice, 4), np.round(upper_dice, 4)],
        "hausdorff": [np.round(mean_hausdorff, 4), np.round(lower_hausdorff, 4), np.round(upper_hausdorff, 4)],
        "precision": [np.round(mean_precision, 4), np.round(lower_precision, 4), np.round(upper_precision, 4)],
        "recall": [np.round(mean_recall, 4), np.round(lower_recall, 4), np.round(upper_recall, 4)],
        "f1": [np.round(mean_f1, 4), np.round(lower_f1, 4), np.round(upper_f1, 4)]
    }

    return final_scores


def launch_compute_metrics_anomaly_detection(args):
    """
    Computes reconstruction metrics on the test_reconstruction set and visualize some results
    """


    DEVICE_TYPE = "cuda:0"
    device = torch.device(DEVICE_TYPE)

    set_determinism(0)

    # ----------- SETTINGS -----------

    ROOT_DIR = args.root_dir

    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    SUB_EXPERIMENT_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/"
    ANOMALY_MAPS_DIR_SELECT_PARAMS = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}_select_params/"
    ANOMALY_MAPS_DIR = ROOT_DIR+f"datasets/anomaly_maps/{SUB_EXPERIMENT_NAME}/" # final anomaly maps with best params
    os.makedirs(ANOMALY_MAPS_DIR_SELECT_PARAMS, exist_ok=True)
    os.makedirs(ANOMALY_MAPS_DIR, exist_ok=True)

    IMAGE_SIZE = args.image_size

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

    # -------------------- define the data --------------------

    if args.dataset["test"] == "brats":
        ano_dataset = anomaly_datasets.BRATS(args)

    if args.dataset["test"] == "isles":
        ano_dataset = anomaly_datasets.ISLES(args)
    
    if args.dataset["test"] == "soop":
        ano_dataset = anomaly_datasets.SOOP(args)
    
    if args.dataset["test"] == "soop_fast":
        ano_dataset = anomaly_datasets.SOOP_Fast(args)

    
    


    model = define_instance(args, "network_def").to(device)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE_TYPE))
    model.eval()


    if args.noise["type"] == "simplex":
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], 
                                                            schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], 
                                                            persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])


    num_timesteps_to_try = np.arange(NOISE_MIN, NOISE_MAX, NOISE_INTERVAL)

    if args.dataset["test"] == "brats":
        # Compute the raw anomaly maps and save them as nifti files
        # So they can be used to compute metrics later with different postprocessing steps without having to recompute the anomaly maps each time.
        
        # Check if select best params has already been done
        best_params_csv_path = SUB_EXPERIMENT_DIR+"best_params_brats.csv"

        if not os.path.exists(best_params_csv_path):
            dtprint("Generating raw anomaly maps for different timesteps...")
            for timesteps in num_timesteps_to_try:
                make_anomaly_maps_optim(args, model, device, infer_scheduler, ano_dataset.test_anomaly_loader_select_params, ano_dataset.test_anomaly_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS)

            dtprint(f"Computing best parameters with CPU...")
            launch_compute_select_params_cpu(args)
        else:
            dtprint(f"Best parameters already computed: {best_params_csv_path}")        

        # Check if the final mterics have already been computed
        metrics_csv_path = SUB_EXPERIMENT_DIR + f"metrics_{args.dataset['test']}.csv"

        if not os.path.exists(metrics_csv_path):

            # Read best parameters from CSV file
            best_params_df = pd.read_csv(best_params_csv_path)
            best_params_dict = dict(zip(best_params_df['parameter'], best_params_df['value']))

            best_num_timesteps = int(best_params_dict['num_timesteps'])
            best_median_filter_size = int(best_params_dict['median_filter_size'])
            best_threshold = float(best_params_dict['threshold'])
            best_erosion_dilation_iterations = int(best_params_dict['erosion_dilation_iterations'])
            best_binary_fill_holes = int(best_params_dict['binary_fill_holes'])
            
            dtprint(f"Computing final metrics with best parameters...")
            final_scores = compute_metrics(args, model, device, ANOMALY_MAPS_DIR, infer_scheduler, 
                                        ano_dataset.test_anomaly_loader_metrics, 
                                        ano_dataset.test_anomaly_images_metrics, 
                                        ano_dataset.test_masks_loader_metrics, 
                                        timesteps=best_num_timesteps, 
                                        threshold=best_threshold, 
                                        median_filter_size=best_median_filter_size, 
                                        erosion_dilation_iterations=best_erosion_dilation_iterations,
                                        binary_fill_holes_param=best_binary_fill_holes)

            
            metrics_result_text = "BRATS\n"
            metrics_result_text += "".join([f"{key}: mean {final_scores[key][0]} 95% CI [{final_scores[key][1]} - {final_scores[key][2]}]\n" for key in final_scores])
            
            metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
            metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
            metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
            metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations} "
            metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}"
            metrics_result_text += "\n"
            dtprint(metrics_result_text)

            # Save metrics results to CSV file
            
            metrics_df = pd.DataFrame({
                'metric': list(final_scores.keys()),
                'mean': [final_scores[key][0] for key in final_scores],
                'lower_ci': [final_scores[key][1] for key in final_scores],
                'upper_ci': [final_scores[key][2] for key in final_scores]
            })
            metrics_df.to_csv(metrics_csv_path, index=False)
            

            if args.show_summary_figures:
                dtprint("Making summary figure...")
                show_summary_figure(args, 
                                device, 
                                model, 
                                infer_scheduler, 
                                ano_dataset.test_anomaly_large_loader_metrics, 
                                ano_dataset.test_masks_large_loader_metrics, 
                                timesteps=best_num_timesteps, 
                                median_filter_size=best_median_filter_size, 
                                threshold=best_threshold, 
                                erosion_dilation_iterations=best_erosion_dilation_iterations,
                                binary_fill_holes_param=best_binary_fill_holes,
                                metrics_result_text=metrics_result_text,
                                ROOT_DIR=ROOT_DIR,
                                EXPERIMENT_NAME=EXPERIMENT_NAME,
                                SUB_EXPERIMENT_NAME=SUB_EXPERIMENT_NAME
                                )
        else:
            dtprint(f"Final metrics already computed: {metrics_csv_path}")

    if args.dataset["test"] == "isles" or args.dataset["test"] == "soop": # TODO: finir pour isles et changer les noms de tous les fichiers isles pour qu'ils aient tous le même nom
        
        groups = ["large", "medium", "small"] # the groups of anomalies based on their size, large is the easiest and small is the hardest

        for group in groups:
            
            dtprint(f"{group} group")

            # Check if select best params has already been done
            best_params_csv_path = SUB_EXPERIMENT_DIR+f"best_params_{args.dataset['test']}_{group}_group.csv"

            if not os.path.exists(best_params_csv_path):

                dtprint("Generating raw anomaly maps for different timesteps...")
                os.makedirs(ANOMALY_MAPS_DIR+f"{group}/", exist_ok=True)
                
                for timesteps in num_timesteps_to_try:
                    make_anomaly_maps_optim(args, model, device, infer_scheduler, ano_dataset.get_anomaly_loader_select_params(group), ano_dataset.get_anomaly_images_select_params(group), timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS+group+"/")
                
                dtprint(f"Computing best parameters with CPU...")
                launch_compute_select_params_cpu(args)
            else:
                dtprint(f"Best parameters already computed: {best_params_csv_path}")
            

            # Check if the final mterics have already been computed
            metrics_csv_path = SUB_EXPERIMENT_DIR + f"metrics_{group}_group_{args.dataset['test']}.csv"

            if not os.path.exists(metrics_csv_path):

                # Read best parameters from CSV file
                best_params_df = pd.read_csv(best_params_csv_path)
                best_params_dict = dict(zip(best_params_df['parameter'], best_params_df['value']))

                best_num_timesteps = int(best_params_dict['num_timesteps'])
                best_median_filter_size = int(best_params_dict['median_filter_size'])
                best_threshold = float(best_params_dict['threshold'])
                best_erosion_dilation_iterations = int(best_params_dict['erosion_dilation_iterations'])
                best_binary_fill_holes = int(best_params_dict['binary_fill_holes'])

                dtprint(f"Computing final metrics with best parameters...")
                final_scores = compute_metrics(args, model, device, ANOMALY_MAPS_DIR+f"{group}/", infer_scheduler, 
                                            ano_dataset.get_anomaly_loader_metrics(group), 
                                            ano_dataset.get_anomaly_images_metrics(group), 
                                            ano_dataset.get_masks_loader_metrics(group), 
                                            timesteps=best_num_timesteps, 
                                            threshold=best_threshold, 
                                            median_filter_size=best_median_filter_size, 
                                            erosion_dilation_iterations=best_erosion_dilation_iterations,
                                            binary_fill_holes_param=best_binary_fill_holes)
                
                metrics_result_text = f"{args.dataset['test']} {group} group\n"
                metrics_result_text += "".join([f"{key}: mean {final_scores[key][0]} 95% CI [{final_scores[key][1]} - {final_scores[key][2]}]\n" for key in final_scores])

                metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
                metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
                metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
                metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations} "
                metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}"
                metrics_result_text += "\n"
                dtprint(metrics_result_text)
                
                # Save metrics results to CSV file
                metrics_df = pd.DataFrame({
                    'metric': list(final_scores.keys()),
                    'mean': [final_scores[key][0] for key in final_scores],
                    'lower_ci': [final_scores[key][1] for key in final_scores],
                    'upper_ci': [final_scores[key][2] for key in final_scores]
                })
                metrics_df.to_csv(metrics_csv_path, index=False)
                
                if args.show_summary_figures and group == "large":
                    dtprint("Making summary figure...")
                    show_summary_figure(args, 
                                    device, 
                                    model, 
                                    infer_scheduler, 
                                    ano_dataset.get_anomaly_loader_metrics(group), 
                                    ano_dataset.get_masks_loader_metrics(group), 
                                    timesteps=best_num_timesteps, 
                                    median_filter_size=best_median_filter_size, 
                                    threshold=best_threshold, 
                                    erosion_dilation_iterations=best_erosion_dilation_iterations,
                                    binary_fill_holes_param=best_binary_fill_holes,
                                    metrics_result_text=metrics_result_text,
                                    ROOT_DIR=ROOT_DIR,
                                    EXPERIMENT_NAME=EXPERIMENT_NAME,
                                    SUB_EXPERIMENT_NAME=SUB_EXPERIMENT_NAME
                                    )
            else:
                dtprint(f"Final metrics already computed: {metrics_csv_path}")    
        
        
    
    if args.dataset["test"] == "soop_fast":
        
        # --------------------------------- large group
        dtprint("Large group")
        dtprint("Generating raw anomaly maps for different timesteps...")

        best_params_csv_path = SUB_EXPERIMENT_DIR+f"best_params_{args.dataset['test']}.csv"
        
        if not os.path.exists(best_params_csv_path):
        
            for timesteps in num_timesteps_to_try:
                make_anomaly_maps_optim(args, model, device, infer_scheduler, ano_dataset.test_anomaly_large_loader_select_params_small, ano_dataset.test_anomaly_large_images_select_params, timesteps, ANOMALY_MAPS_DIR_SELECT_PARAMS)


            dtprint(f"Computing best parameters with CPU...")
            launch_compute_select_params_cpu(args)
 

        metrics_csv_path = SUB_EXPERIMENT_DIR + f"metrics_{args.dataset['test']}.csv"
        if not os.path.exists(metrics_csv_path):

            # Read best parameters from CSV file
            best_params_df = pd.read_csv(best_params_csv_path)
            best_params_dict = dict(zip(best_params_df['parameter'], best_params_df['value']))

            best_num_timesteps = int(best_params_dict['num_timesteps'])
            best_median_filter_size = int(best_params_dict['median_filter_size'])
            best_threshold = float(best_params_dict['threshold'])
            best_erosion_dilation_iterations = int(best_params_dict['erosion_dilation_iterations'])
            best_binary_fill_holes = int(best_params_dict['binary_fill_holes'])

            dtprint(f"Computing final metrics with best parameters...")

            final_scores = compute_metrics(args, model, device, ANOMALY_MAPS_DIR, infer_scheduler, 
                                        ano_dataset.test_anomaly_large_loader_metrics_small, 
                                        ano_dataset.test_anomaly_large_images_metrics, 
                                        ano_dataset.test_masks_large_loader_metrics_small, 
                                        timesteps=best_num_timesteps, 
                                        threshold=best_threshold, 
                                        median_filter_size=best_median_filter_size, 
                                        erosion_dilation_iterations=best_erosion_dilation_iterations, 
                                        binary_fill_holes_param=best_binary_fill_holes)

            
            metrics_result_text = "SOOP fast\n"
            metrics_result_text += "".join([f"{key}: mean {final_scores[key][0]} 95% CI [{final_scores[key][1]} - {final_scores[key][2]}]\n" for key in final_scores])
            
            metrics_result_text += f"Best Number of Timesteps: {best_num_timesteps} "
            metrics_result_text += f"Best Median Filter Size: {best_median_filter_size} "
            metrics_result_text += f"Best Threshold: {best_threshold:.4f} "
            metrics_result_text += f"Best Erosion Dilation Iterations: {best_erosion_dilation_iterations} "
            metrics_result_text += f"Best Binary Fill Holes: {best_binary_fill_holes}\n"

            dtprint(metrics_result_text)

            # Save metrics results to CSV file
            metrics_df = pd.DataFrame({
                'metric': list(final_scores.keys()),
                'mean': [final_scores[key][0] for key in final_scores],
                'lower_ci': [final_scores[key][1] for key in final_scores],
                'upper_ci': [final_scores[key][2] for key in final_scores]
            })
            metrics_df.to_csv(metrics_csv_path, index=False)

            
            if args.show_summary_figures:
                dtprint("Making summary figure...")
                show_summary_figure(args, 
                                    device, 
                                    model, 
                                    infer_scheduler, 
                                    ano_dataset.test_anomaly_large_loader_metrics_small, 
                                    ano_dataset.test_masks_large_loader_metrics_small, 
                                    timesteps=best_num_timesteps, 
                                    median_filter_size=best_median_filter_size, 
                                    threshold=best_threshold, 
                                    erosion_dilation_iterations=best_erosion_dilation_iterations,
                                    binary_fill_holes_param=best_binary_fill_holes,
                                    metrics_result_text=metrics_result_text,
                                    ROOT_DIR=ROOT_DIR,
                                    EXPERIMENT_NAME=EXPERIMENT_NAME,
                                    SUB_EXPERIMENT_NAME=SUB_EXPERIMENT_NAME
                                    )
            