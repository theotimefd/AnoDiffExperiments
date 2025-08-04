import os
import glob
#import opensimplex

#from torchvision.utils import save_image

import numpy as np
import csv
import torch
import torch.nn.functional as F
import monai.data
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import first, set_determinism
from monai.data.utils import pad_list_data_collate
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random

import nibabel as nib

from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

import pandas as pd
import AnoDDPM.simplex as simplex

from monai.metrics import compute_iou

#print_config()

DEVICE_TYPE = "cuda:0"

#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

set_determinism(0)

IMAGE_SIZE = 128

class Get2DSlice(transforms.Transform):
    """
    Fetch the middle slice of a 3D volume.
    Args:
        axis: The axis along which to slice the volume. 0 for axial, 1 for coronal, 2 for sagittal.
        offset : Offset the index by a specified amount (default=0)
    """

    def __init__(
        self,
        axis: int = 0,
        offset: int=0
    ):
        super().__init__()
        self.axis = axis
        self.offset = offset

    def __call__(self, data):
        #print(data.shape)
        if self.axis==0:
            return data[:, data.shape[1]//2+self.offset,:,:]
        elif self.axis==1:
            return data[:, :,data.shape[2]//2+self.offset,:]
        elif self.axis==2:
            return data[:, :, :,data.shape[3]//2+self.offset]

class SetBackgroundToZero(transforms.Transform):
    """
    Custom MONAI transform that zeros out voxels with the most frequent intensity value.
    
    Args:
        keys (str or list): Keys of the dictionary to apply the transform to.
        tolerance (int): Optional range around the mode value to also zero.
    """
    def __init__(self, tolerance: int = 0):
        super().__init__()
        self.tolerance = tolerance

    def __call__(self, data):
        
            
        is_tensor = isinstance(data, torch.Tensor)
        data_np = data.cpu().numpy() if is_tensor else data

        # Flatten and compute histogram
        flat = data_np.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        mode_val = unique[np.argmax(counts)]

        # Apply tolerance if specified
        if self.tolerance > 0:
            mask = np.isin(data_np, range(mode_val - self.tolerance, mode_val + self.tolerance + 1))
        else:
            mask = data_np == mode_val

        # Zero out the background
        data_np[mask] = 0

        # Put back in original type
        data = torch.from_numpy(data_np) if is_tensor else data_np

        return data

device = torch.device(DEVICE_TYPE)



model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 128),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=(0, 128, 128),
)
model.to(device)

num_train_timesteps = 1000 #TODO vu que je fais de la détec d'anomalies, pas besoin d'aller jusqu'à 1000 (full bruit) je peux faire 200 par ex
scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

inferer = DiffusionInferer(scheduler)



model.load_state_dict(torch.load(os.path.join(ROOT_DIR+"AnoDiffExperiments/Best_models/experiment_1", "exp_1_2_best_model.pth"), map_location=DEVICE_TYPE))
model.eval()

ISLES_ADC_datalist = sorted(glob.glob(ROOT_DIR+"datasets/ISLES_ADC_registered_resampled/*.nii.gz"))

problematic_images_indexes = [59,100,101,136,152,153,161,163,164,165,166,181,183,184,185,187,188,190,192, #TODO: va falloir régler les pb sur ces images, ça a l'air d'être la partie ou je fais rotate90 qui foire (peut être avec le remplacement de la matrice affine)
                              193,194,196,198,199,201,205,206,207,208,214,218,221,225,228,229,235,237,238,
                              240,242,244,247,248,249]
problematic_images = [os.path.basename(ISLES_ADC_datalist[i]) for i in problematic_images_indexes]
ISLES_ADC_datalist = [image for image in ISLES_ADC_datalist if os.path.basename(image) not in problematic_images]
#print(ISLES_ADC_datalist[:10])
ISLES_masks_datalist = sorted(glob.glob(ROOT_DIR+"datasets/ISLES_masks_registered_resampled/*.nii.gz"))
ISLES_masks_datalist = [image for image in ISLES_masks_datalist]
ISLES_masks_datalist = [image for image in ISLES_masks_datalist if os.path.basename(image).replace('msk', 'adc') not in problematic_images]
#print(ISLES_masks_datalist[:10])
#print(problematic_images)

img_isles_to_keep = [0,2,3,9,10,12,14,16,20,21,22,23,25,26,
                     30,32,37,38,39,40,42,46,47,48,51,52,53,
                     55,56,60,62,63,64,65,72,74,75,79, 80,81,
                     82,83,85,90,93,103,104,110,112,114,116,118,
                     120,123,125,135,140,141,147,148,151,154,155,
                     156,157,159164,166,167,168,170,171,172,173,175,176,178,
                     180,181,84,187,188,190,192,194,199,200,201,202,203,204,205]

ISLES_ADC_datalist_final = [] # only the images where there is a lesion on the middle slice
ISLES_masks_datalist_final = []
 
for i, element in enumerate(ISLES_ADC_datalist):
    if i not in img_isles_to_keep:
        pass
    else:
        ISLES_ADC_datalist_final.append(element)
        ISLES_masks_datalist_final.append(ISLES_masks_datalist[i])
        

num_workers = 4
ano_batch_size = 10

test_anomaly_transforms = transforms.Compose(
    [
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        Get2DSlice(axis=2, offset=+2),
        transforms.ScaleIntensity(minv=0.0, maxv=3000.0),
        transforms.ScaleIntensityRange(a_min=0.0, a_max=3200.0, b_min=0.0, b_max=1.0, clip=True), # change a_max to increase or decrease intensity of the images after transformation
        transforms.ResizeWithPadOrCrop(spatial_size=(128, 128)),
        SetBackgroundToZero(tolerance=0),  # Set background to zero
    ]
)
test_anomaly_ds = CacheDataset(data=ISLES_ADC_datalist_final, transform=test_anomaly_transforms) # TODO ISLES_ADC_datalist[:10]


test_anomaly_loader = DataLoader(
    test_anomaly_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True
)



test_masks_transforms = transforms.Compose(
    [
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        Get2DSlice(axis=2, offset=+2),
        transforms.ResizeWithPadOrCrop(spatial_size=(128, 128)),
    ]
)
test_masks_ds = CacheDataset(data=ISLES_masks_datalist_final, transform=test_masks_transforms) # TODO ISLES_ADC_datalist[:10]


test_masks_loader = DataLoader(
    test_masks_ds, batch_size=ano_batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True
)


@torch.no_grad()
def my_sample(image, timesteps=100, progress_bar=True):
    
    
    num_infer_timesteps = timesteps #100 # higher number = more noise at first timestep, more denoising steps
    intermediate_steps = num_infer_timesteps//10
    infer_scheduler = DDPMScheduler(num_train_timesteps=num_infer_timesteps)

    all_next_timesteps = torch.cat((scheduler.timesteps[1:], torch.tensor([0], dtype=infer_scheduler.timesteps.dtype)))

    if progress_bar:
        progress_bar = tqdm(
            zip(infer_scheduler.timesteps, all_next_timesteps),
            total=min(len(infer_scheduler.timesteps), len(all_next_timesteps)),
        )
    else:
        progress_bar = zip(infer_scheduler.timesteps, all_next_timesteps)
            
    intermediates = []
            
    for t, next_t in progress_bar:          # va de num_infer_timesteps à 0
        # 1. predict noise model_output
        diffusion_model = model
        
        model_output = diffusion_model(
            image, timesteps=torch.Tensor((t,)).to(device), context=None
        )
        #inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
        # 2. compute previous image: x_t -> x_t-1
        
        image, _ = infer_scheduler.step(model_output, t, image)
        
        if t % intermediate_steps == 0:
            intermediates.append(image)
    
    return image, intermediates



# 8h sur bigfoot celui la
num_timesteps_to_try = np.arange(10, 500, 10)
thresholds_to_try = np.arange(0.0, 1.0, 0.01) # from 0.0 to 1.0 with step 0.05

iou_scores_df = pd.DataFrame(index=num_timesteps_to_try, columns=thresholds_to_try)
iou_scores_df.fillna(0.0, inplace=True)

best_iou = 0.0
best_threshold = 0.0
best_num_timesteps = 0


for i,(image_batch, mask_batch) in enumerate(tqdm(zip(test_anomaly_loader, test_masks_loader))): # i=6 batch is nice

    test_images = image_batch.to(device)
    test_masks = mask_batch.to(device)
    test_masks[test_masks>0.5] = 1.0
    test_masks[test_masks<=0.5] = 0.0

    for infer_timesteps in num_timesteps_to_try:
        with autocast(device_type=DEVICE_TYPE, enabled=True):
            infered_image, intermediates = my_sample(test_images, timesteps=infer_timesteps, progress_bar=False)
    
        for threshold in thresholds_to_try:
            ano_segmentation = torch.abs(infered_image - test_images) > threshold
            iou_score = compute_iou(ano_segmentation, test_masks)
            flattened_iou_score = iou_score.cpu().numpy().flatten()
            flattened_iou_score[np.isnan(flattened_iou_score)] = 0.0

            if np.isnan(iou_scores_df.loc[infer_timesteps, threshold]): # if the cell is empty
                iou_scores_df.loc[infer_timesteps, threshold] = np.sum(flattened_iou_score)
            else:
                iou_scores_df.loc[infer_timesteps, threshold] += np.sum(flattened_iou_score) # this average is false

#divide everything by the number of images since compute_iou returns a list of iou of size batch_size
iou_scores_df = iou_scores_df / len(test_anomaly_loader.dataset)

best_iou = iou_scores_df.max().max()
best_threshold = iou_scores_df.max(axis=0).idxmax()
best_num_timesteps = iou_scores_df.max(axis=1).idxmax()

print(f"Best IOU: {best_iou}")
print(f"Best Threshold: {best_threshold}")
print(f"Best Number of Timesteps: {best_num_timesteps}")

iou_scores_df.to_csv("exp_1-2_scores_iou.csv")