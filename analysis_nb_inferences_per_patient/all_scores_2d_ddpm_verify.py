import os
import sys
sys.path.append("..")
from utils.scores import compute_scores
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import median_filter, binary_erosion, binary_dilation, binary_fill_holes
import json


DEVICE_TYPE = "cuda:0"
device = torch.device(DEVICE_TYPE)

# ADC large group 1x
adc_large_1x_timesteps = 190
adc_large_1x_median_filter_size = 7
adc_large_1x_threshold = 0.06
adc_large_1x_erosion_dilation = 4
adc_large_1x_fill_holes = True

# ADC large group 10x
adc_large_10x_timesteps = 130
adc_large_10x_median_filter_size = 7
adc_large_10x_threshold = 0.04
adc_large_10x_erosion_dilation = 4
adc_large_10x_fill_holes = True

ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

anomaly_maps_1x_dir = ROOT_DIR + "datasets/anomaly_maps/exp_3_2_fixed/large/" 
anomaly_maps_10x_dir = ROOT_DIR + "datasets/anomaly_maps/exp_3_2_10x_inference/large/" 

masks_dir = ROOT_DIR + "datasets/final_soop_dataset_small/masks_combined_registered/" 


segmentation_1x_no_post_proc_folder = ROOT_DIR + "datasets/segmentations/exp_3_2_fixed/segmentation_1x_no_post_proc/" #TODO finish path
os.makedirs(segmentation_1x_no_post_proc_folder, exist_ok=True)
segmentation_1x_with_post_proc_folder = ROOT_DIR + "datasets/segmentations/exp_3_2_fixed/segmentation_1x_with_post_proc/" #TODO finish path
os.makedirs(segmentation_1x_with_post_proc_folder, exist_ok=True)

segmentation_10x_no_post_proc_folder = ROOT_DIR + "datasets/segmentations/exp_3_2_10x_inference/segmentation_10x_no_post_proc/" #TODO finish path
os.makedirs(segmentation_10x_no_post_proc_folder, exist_ok=True)
segmentation_10x_with_post_proc_folder = ROOT_DIR + "datasets/segmentations/exp_3_2_10x_inference/segmentation_10x_with_post_proc/" #TODO finish path
os.makedirs(segmentation_10x_with_post_proc_folder, exist_ok=True)





class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

def launch_compute():

    scores_patients = {}

    # 1x inference

    for ano_file in tqdm(os.listdir(anomaly_maps_1x_dir)):
        
        patient_nb = ano_file.split(".")[0]

        scores_patients[patient_nb] = {}

        ano_map = nib.load(os.path.join(anomaly_maps_1x_dir, ano_file)).get_fdata()
        mask = nib.load(os.path.join(masks_dir, ano_file)).get_fdata()

        # make sure the images are in shape (1, 1, H, W, D) for the scores function
        if len(ano_map.shape) == 3:
            ano_map = torch.from_numpy(ano_map).unsqueeze(0).unsqueeze(0).to(device)
        if len(mask.shape) == 3:
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

        # no post processing
        segmentation_no_post_proc = ano_map > adc_large_1x_threshold

        iou_scores, dice_scores, hausdorff_distances, precision_scores, recall_scores, f1_scores = compute_scores(segmentation_no_post_proc, mask)

        

        scores_patients[patient_nb]["1x_no_post_proc"] = {
            "iou": iou_scores,
            "dice": dice_scores,
            "hausdorff_distance": hausdorff_distances,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1_score": f1_scores
        }

        # save the segmentation map as a nifti file for visualization
        segmentation_no_post_proc_np = segmentation_no_post_proc.cpu().numpy().astype(np.uint8)
        segmentation_no_post_proc_nifti = nib.Nifti1Image(segmentation_no_post_proc_np[0,0], affine=np.eye(4))
        nib.save(segmentation_no_post_proc_nifti, os.path.join(segmentation_1x_no_post_proc_folder, patient_nb + ".nii.gz"))
        

        # apply median filter
        anomaly_map_np = ano_map.cpu().numpy()
        for b in range(anomaly_map_np.shape[0]):
            anomaly_map_np[b] = median_filter(anomaly_map_np[b], size=adc_large_1x_median_filter_size)
        anomaly_map = torch.from_numpy(anomaly_map_np).to(device)


        # make the segmentation map with threshold
        ano_segmentation = anomaly_map > adc_large_1x_threshold

        # perform erosion if specified
        if adc_large_1x_erosion_dilation > 0:
            ano_segmentation_np = ano_segmentation.cpu().numpy()
            for b in range(ano_segmentation_np.shape[0]):
                ano_segmentation_np[b,0] = binary_erosion(ano_segmentation_np[b,0], iterations=adc_large_1x_erosion_dilation)
                ano_segmentation_np[b,0] = binary_dilation(ano_segmentation_np[b,0], iterations=adc_large_1x_erosion_dilation)
            ano_segmentation = torch.from_numpy(ano_segmentation_np).to(device)
        
        if adc_large_1x_fill_holes:
            ano_segmentation_np = ano_segmentation.cpu().numpy()
            for b in range(ano_segmentation_np.shape[0]):
                ano_segmentation_np[b,0] = binary_fill_holes(ano_segmentation_np[b,0])
            ano_segmentation = torch.from_numpy(ano_segmentation_np).to(device)

        # save the segmentation map as a nifti file for visualization
        segmentation_with_post_proc_np = segmentation_no_post_proc.cpu().numpy().astype(np.uint8)
        segmentation_with_post_proc_nifti = nib.Nifti1Image(segmentation_with_post_proc_np[0,0], affine=np.eye(4))
        nib.save(segmentation_with_post_proc_nifti, os.path.join(segmentation_1x_with_post_proc_folder, patient_nb + ".nii.gz"))

        iou_scores, dice_scores, hausdorff_distances, precision_scores, recall_scores, f1_scores = compute_scores(ano_segmentation, mask)

        scores_patients[patient_nb]["1x_with_post_proc"] = {
            "iou": iou_scores,
            "dice": dice_scores,
            "hausdorff_distance": hausdorff_distances,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1_score": f1_scores
        }

    # 10x inference

    for ano_file in tqdm(os.listdir(anomaly_maps_10x_dir)):
        
        patient_nb = ano_file.split(".")[0]

        ano_map = nib.load(os.path.join(anomaly_maps_10x_dir, ano_file)).get_fdata()
        mask = nib.load(os.path.join(masks_dir, ano_file)).get_fdata()

        # make sure the images are in shape (1, 1, H, W, D) for the scores function
        if len(ano_map.shape) == 3:
            ano_map = torch.from_numpy(ano_map).unsqueeze(0).unsqueeze(0).to(device)
        if len(mask.shape) == 3:
            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(device)

        # no post processing
        segmentation_no_post_proc = ano_map > adc_large_10x_threshold

        iou_scores, dice_scores, hausdorff_distances, precision_scores, recall_scores, f1_scores = compute_scores(segmentation_no_post_proc, mask)

        

        scores_patients[patient_nb]["10x_no_post_proc"] = {
            "iou": iou_scores,
            "dice": dice_scores,
            "hausdorff_distance": hausdorff_distances,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1_score": f1_scores
        }

        # save the segmentation map as a nifti file for visualization
        segmentation_no_post_proc_np = segmentation_no_post_proc.cpu().numpy().astype(np.uint8)
        segmentation_no_post_proc_nifti = nib.Nifti1Image(segmentation_no_post_proc_np[0,0], affine=np.eye(4))
        nib.save(segmentation_no_post_proc_nifti, os.path.join(segmentation_10x_no_post_proc_folder, patient_nb + ".nii.gz"))
        

        # apply median filter
        anomaly_map_np = ano_map.cpu().numpy()
        for b in range(anomaly_map_np.shape[0]):
            anomaly_map_np[b] = median_filter(anomaly_map_np[b], size=adc_large_10x_median_filter_size)
        anomaly_map = torch.from_numpy(anomaly_map_np).to(device)


        # make the segmentation map with threshold
        ano_segmentation = anomaly_map > adc_large_10x_threshold

        # perform erosion if specified
        if adc_large_10x_erosion_dilation > 0:
            ano_segmentation_np = ano_segmentation.cpu().numpy()
            for b in range(ano_segmentation_np.shape[0]):
                ano_segmentation_np[b,0] = binary_erosion(ano_segmentation_np[b,0], iterations=adc_large_10x_erosion_dilation)
                ano_segmentation_np[b,0] = binary_dilation(ano_segmentation_np[b,0], iterations=adc_large_10x_erosion_dilation)
            ano_segmentation = torch.from_numpy(ano_segmentation_np).to(device)
        
        if adc_large_10x_fill_holes:
            ano_segmentation_np = ano_segmentation.cpu().numpy()
            for b in range(ano_segmentation_np.shape[0]):
                ano_segmentation_np[b,0] = binary_fill_holes(ano_segmentation_np[b,0])
            ano_segmentation = torch.from_numpy(ano_segmentation_np).to(device)

        # save the segmentation map as a nifti file for visualization
        segmentation_with_post_proc_np = segmentation_no_post_proc.cpu().numpy().astype(np.uint8)
        segmentation_with_post_proc_nifti = nib.Nifti1Image(segmentation_with_post_proc_np[0,0], affine=np.eye(4))
        nib.save(segmentation_with_post_proc_nifti, os.path.join(segmentation_10x_with_post_proc_folder, patient_nb + ".nii.gz"))

        iou_scores, dice_scores, hausdorff_distances, precision_scores, recall_scores, f1_scores = compute_scores(ano_segmentation, mask)

        scores_patients[patient_nb]["10x_with_post_proc"] = {
            "iou": iou_scores,
            "dice": dice_scores,
            "hausdorff_distance": hausdorff_distances,
            "precision": precision_scores,
            "recall": recall_scores,
            "f1_score": f1_scores
        }
    

    with open(ROOT_DIR + "AnoDiffExperiments/analysis_nb_inferences_per_patient/scores_patients_2d_ddpm_flair_verify.json", "w") as f:
        json.dump(scores_patients, f, cls=NumpyEncoder)


if __name__ == "__main__":
    launch_compute()