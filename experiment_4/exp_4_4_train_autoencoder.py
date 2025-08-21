# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
from pathlib import Path

import numpy as np

import torch
from monai import transforms
from monai.losses import PatchAdversarialLoss, PerceptualLoss
from monai.networks.nets import PatchDiscriminator
from monai.data import CacheDataset, DataLoader, SmartCacheDataset, Dataset
from monai.utils import set_determinism
from torch.nn import L1Loss, MSELoss
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from monai.data.utils import pad_list_data_collate
from monai.inferers import SlidingWindowInferer

from visualize_image import visualize_one_slice_in_3d_image
import csv
from tqdm import tqdm

device="cuda:0"

torch.cuda.set_device(device)
print(f"Using {device}")

torch.backends.cudnn.benchmark = True
torch.set_num_threads(4)
torch.autograd.set_detect_anomaly(True)

set_determinism(0)

EXPERIMENT_NAME = "exp_4_4_autoencoder"
#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
MODELS_DIR = ROOT_DIR+"AnoDiffExperiments/best_models/experiment_4/"
RESUME_TRAINING = False

train_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/Final_ADC_dataset/train.csv")
train_images_path = []

with open(train_csv, mode='r') as file:
    reader = csv.reader(file)
    for line in tqdm(reader):
        #print(line)
        train_images_path.append(ROOT_DIR+line[0])

val_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/Final_ADC_dataset/val.csv")
val_images_path = []

with open(val_csv, mode='r') as file:
    reader = csv.reader(file)
    for line in tqdm(reader):

        val_images_path.append(ROOT_DIR+line[0])

test_reconstruction_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/Final_ADC_dataset/test.csv")
test_reconstruction_images_path = []

with open(test_reconstruction_csv, mode='r') as file:
    reader = csv.reader(file)
    for line in tqdm(reader):

        test_reconstruction_images_path.append(ROOT_DIR+line[0])

#train_datalist = sorted(train_images_path)
train_datalist = train_images_path

#val_datalist = sorted(val_images_path)
val_datalist = val_images_path

#val_datalist = sorted(val_images_path)
test_reconstruction_datalist = test_reconstruction_images_path


batch_size = 32 # batch_size = 48 -> 91 Gb vram
num_workers = 4

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

IMAGE_SIZE = 225
PATCH_SIZE = 64


train_transforms = transforms.Compose(
    [
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        transforms.RandAffine(prob=0.2, rotate_range=(0.10, 0.10, 0.10), lazy=True),#+- 0.10 radians for each axis
        transforms.RandScaleCrop(roi_scale=0.9, max_roi_scale=1.1, random_size=True, lazy=True),
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE), lazy=True),
        transforms.ScaleIntensityRange(a_min=0.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandScaleIntensity(factors=0.15),
        SetBackgroundToZero(),
        transforms.RandSpatialCrop(roi_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)), # <-----
    ]
)

#train_ds = CacheDataset(data=train_datalist, transform=train_transforms)

train_ds = SmartCacheDataset(
    data=train_datalist,
    transform=train_transforms,
    cache_num=600, # number of images to load in RAM
    replace_rate=0.2, # how much of the cache is replaced every epoch
)

train_loader = DataLoader(
    #collate_fn=pad_list_data_collate: any tensors are centrally padded to match the shape of the biggest tensor in each dimension
    train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=pad_list_data_collate
)


val_transforms = transforms.Compose( # validation needs to be done on the whole image
    [
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        transforms.ScaleIntensityRange(a_min=0.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.Resize(spatial_size=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE), lazy=True),
        SetBackgroundToZero(),
        #transforms.RandSpatialCrop(roi_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)), # <-----
    ]
)
val_ds = CacheDataset(data=val_datalist[:100], transform=val_transforms) # using only 100 otherwise its way too long because full image 3D with sliding window has to be done on cpu
"""
val_ds = SmartCacheDataset(
    data=val_datalist,
    transform=val_transforms,
    cache_num=100, # number of images to load in RAM
    replace_rate=0.2, # how much of the cache is replaced every epoch
)"""
val_loader = DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=num_workers, persistent_workers=True
)

test_reconstruction_transforms = transforms.Compose(
    [
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        transforms.ScaleIntensityRange(a_min=0.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.Resize(spatial_size=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE,IMAGE_SIZE,IMAGE_SIZE), lazy=True),
        SetBackgroundToZero(),
        #transforms.RandSpatialCrop(roi_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE)), # <-----
    ]
)
#val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
test_reconstruction_ds = Dataset(data=test_reconstruction_datalist[:2], transform=test_reconstruction_transforms)
test_reconstruction_loader = DataLoader(
    test_reconstruction_ds, batch_size=1, shuffle=False, num_workers=num_workers, persistent_workers=True #TODO batch_size
)

# Step 2: Define Autoencoder KL network and discriminator
autoencoder = AutoencoderKL(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(32, 64, 64, 128),
    latent_channels=3,
    num_res_blocks=1,
    norm_num_groups=16,
    attention_levels=(False, False, True, True),
)
autoencoder.to(device)

discriminator_norm = "INSTANCE"
discriminator = PatchDiscriminator(
    spatial_dims=3,
    num_layers_d=3,
    channels=32,
    in_channels=1,
    out_channels=1,
    norm=discriminator_norm,
).to(device)

trained_g_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_autoencoder_best.pt")
trained_d_path = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_discriminator_best.pt")
trained_g_path_last = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_autoencoder_last.pt")
trained_d_path_last = os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_discriminator_last.pt")

Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

if RESUME_TRAINING:
    map_location = "cuda"
    try:
        autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location, weights_only=True))
        print(f"Load trained autoencoder from {trained_g_path}")
    except:
        print(f"Train autoencoder from scratch.")

    try:
        discriminator.load_state_dict(torch.load(trained_d_path, map_location=map_location, weights_only=True))
        print(f"Load trained discriminator from {trained_d_path}")
    except:
        print(f"Train discriminator from scratch.")


intensity_loss = L1Loss()

adv_loss = PatchAdversarialLoss(criterion="least_squares")
loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
loss_perceptual.to(device)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]

adv_weight = 0.01
perceptual_weight = 0.001
kl_weight = 1e-8
# kl_weight: important hyper-parameter.
#     If too large, decoder cannot recon good results from latent space.
#     If too small, latent space will not be regularized enough for the diffusion model

optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)


tensorboard_writer = SummaryWriter(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{EXPERIMENT_NAME}")

# Step 4: training
autoencoder_warm_up_n_epochs = 5
max_epochs = 10000
val_interval = 10
intermediary_images = []
n_example_images = 4
best_val_recon_epoch_loss = 100.0
total_step = 0

inferer = SlidingWindowInferer(
    roi_size=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE),
    sw_batch_size=1,
    sw_device=torch.device("cpu"),#"cpu"
    device=torch.device("cpu"),
    overlap=0.25,
    mode="gaussian",
    padding_mode="replicate",
)

def perform_validation(images): #TODO
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            reconstruction = inferer(images.to(torch.device("cpu")), autoencoder.to(torch.device("cpu")))[0]
        recons_loss = intensity_loss(
            reconstruction.to(torch.device("cuda")).float(), images.to(torch.device("cuda")).float()
        ) + perceptual_weight * loss_perceptual(reconstruction.to(torch.device("cuda")).float(), images.to(torch.device("cuda")).float())

    return recons_loss.to(torch.device("cuda"))

for epoch in range(max_epochs):
    # train
    autoencoder.train()
    discriminator.train()

    for step, batch in enumerate(train_loader):
        images = batch.to(device)

        # train Generator part
        optimizer_g.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(): #TODO AMP
            reconstruction, z_mu, z_sigma = autoencoder(images)

        recons_loss = intensity_loss(reconstruction, images)
        kl_loss = KL_loss(z_mu, z_sigma)
        p_loss = loss_perceptual(reconstruction.float(), images.float())
        loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

        if epoch > autoencoder_warm_up_n_epochs:
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = loss_g + adv_weight * generator_loss

        loss_g.backward()
        optimizer_g.step()

        if epoch > autoencoder_warm_up_n_epochs:
            # train Discriminator part
            optimizer_d.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(): #TODO AMP
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

        # write train loss for each batch into tensorboard

        total_step += 1
        tensorboard_writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)
        tensorboard_writer.add_scalar("train_kl_loss_iter", kl_loss, total_step)
        tensorboard_writer.add_scalar("train_perceptual_loss_iter", p_loss, total_step)
        if epoch > autoencoder_warm_up_n_epochs:
            tensorboard_writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
            tensorboard_writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
            tensorboard_writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)

    # validation
    if epoch % val_interval == 0:
        autoencoder.eval()
        val_recon_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            print(f"Validation step {step} of epoch {epoch}")
            images = batch.to(device)  
            recons_loss = perform_validation(images) # validation must be done on the whole image with slidingwindow

            val_recon_epoch_loss += recons_loss.item()

        val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

        autoencoder.to(torch.device("cuda"))

        print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")

        torch.save(autoencoder.state_dict(), trained_g_path_last)
        torch.save(discriminator.state_dict(), trained_d_path_last)
        
        # save best model
        if val_recon_epoch_loss < best_val_recon_epoch_loss:
            best_val_recon_epoch_loss = val_recon_epoch_loss

            torch.save(autoencoder.state_dict(), trained_g_path)
            torch.save(discriminator.state_dict(), trained_d_path)
            print("Got best val recon loss.")
            print("Save trained autoencoder to", trained_g_path)
            print("Save trained discriminator to", trained_d_path)

            tensorboard_writer.add_scalar("best_val_recon_loss", best_val_recon_epoch_loss, epoch)
            """
            for axis in range(3):
                tensorboard_writer.add_image(
                    "val_img_" + str(axis),
                    visualize_one_slice_in_3d_image(images[0, 0, ...], axis).transpose([2, 1, 0]),
                    epoch,
                )
                tensorboard_writer.add_image(
                    "val_recon_" + str(axis),
                    visualize_one_slice_in_3d_image(reconstruction[0, 0, ...], axis).transpose([2, 1, 0]),
                    epoch,
                ) """
        #
        # write val loss for each epoch into tensorboard
        tensorboard_writer.add_scalar("val_recon_loss", val_recon_epoch_loss, epoch)
        

