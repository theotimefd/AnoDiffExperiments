import os
import time
import sys


import numpy as np
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
from monai.data.utils import pad_list_data_collate
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random

from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler


DEVICE_TYPE = "cuda:0"


#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

sys.path.append("..")
import AnoDDPM.simplex as simplex

set_determinism(0)


IMAGE_SIZE = 128



train_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/final_adc_dataset_small_with_augmentation_by_registration/train.csv")
train_images_path = []

with open(train_csv, mode='r') as file:
    reader = csv.reader(file)
    for line in tqdm(reader):
        #print(line)
        train_images_path.append(ROOT_DIR+line[0])

val_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/final_adc_dataset_small_with_augmentation_by_registration/val.csv")
val_images_path = []

with open(val_csv, mode='r') as file:
    reader = csv.reader(file)
    for line in tqdm(reader):

        val_images_path.append(ROOT_DIR+line[0])

test_reconstruction_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/final_adc_dataset_small_with_augmentation_by_registration/test.csv")
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

#test_unhealthy_datalist = test_unhealthy_images_path

batch_size = 16
num_workers = 8



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


class Get2DSliceWithRandomOffset(transforms.RandomizableTransform):
    """
    Will return the middle slice with a random offset in addition to the specified fixed offset.
    Args:
        axis: The axis along which to slice the volume. 0 for axial, 1 for coronal, 2 for sagittal.
        offset : Offset the index by a specified amount (default=0)
    """

    def __init__(
        self,
        axis: int = 0,
        fixed_offset: int=0
    ):
        super().__init__()
        self.axis = axis
        self.fixed_offset = fixed_offset
        self.rand_offset = 0


    def randomize(self):
        super().randomize(None)
        self.rand_offset = random.randint(-10, 10)

    def __call__(self, data):
        #print(data.shape)
        self.randomize()

        #print(self.rand_offset)
        if self.axis==0:
            return data[:, data.shape[1]//2+self.fixed_offset+self.rand_offset,:,:]
        elif self.axis==1:
            return data[:, :,data.shape[2]//2+self.fixed_offset+self.rand_offset,:]
        elif self.axis==2:
            return data[:, :, :,data.shape[3]//2+self.fixed_offset+self.rand_offset]


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



img = transforms.LoadImage(image_only=True)(train_datalist[0])


train_transforms = transforms.Compose(
    [
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.RandAffine(prob=0.3, rotate_range=(0.15, 0.15, 0.15)),  # Increased augmentation
        Get2DSliceWithRandomOffset(axis=2, fixed_offset=0),
        transforms.ScaleIntensityRange(a_min=0.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.RandScaleCrop(roi_scale=0.85, max_roi_scale=1.15, random_size=True),  # More aggressive cropping
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandScaleIntensity(factors=0.2),  # Increased intensity variation
        transforms.RandFlip(prob=0.5, spatial_axis=0),
        transforms.RandGaussianNoise(prob=0.1, mean=0.0, std=0.01),  # Add noise augmentation
        transforms.RandGaussianSmooth(prob=0.1, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0)),  # Add smoothing
        SetBackgroundToZero()
    ]
)
train_ds = CacheDataset(data=train_datalist, transform=train_transforms) #TODO datalist[:32]
train_loader = DataLoader(
    #collate_fn=pad_list_data_collate: any tensors are centrally padded to match the shape of the biggest tensor in each dimension
    train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, persistent_workers=True, collate_fn=pad_list_data_collate
)



val_transforms = transforms.Compose(
    [
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        Get2DSlice(axis=2),
        transforms.ScaleIntensityRange(a_min=0.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        SetBackgroundToZero()
    ]
)
val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True
)



test_reconstruction_transforms = transforms.Compose(
    [
        transforms.LoadImage(),
        transforms.EnsureChannelFirst(),
        Get2DSlice(axis=2),
        transforms.ScaleIntensityRange(a_min=0.0, a_max=4000.0, b_min=0.0, b_max=1.0, clip=True),
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        SetBackgroundToZero()
    ]
)
test_reconstruction_ds = CacheDataset(data=test_reconstruction_datalist, transform=test_reconstruction_transforms)
test_reconstruction_loader = DataLoader(
    test_reconstruction_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True
)



# ### Opensimplex noise 
# 
# https://code.larus.se/lmas/opensimplex 
# 
# https://pypi.org/project/opensimplex/
# 
# https://github.com/Julian-Wyatt/AnoDDPM/blob/master/simplex.py#L212
# 
# https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Wyatt_AnoDDPM_Anomaly_Detection_With_Denoising_Diffusion_Probabilistic_Models_Using_Simplex_CVPRW_2022_paper.pdf

# AnoDDPM:
# 
# Instead of using the default simplex noise function, we
# can apply a number of octaves of noise (also known as frac-
# tal noise). This involves combining N frequencies of noise
# together, where the next frequency’s amplitude reduces by
# some decay rate γ. Figure 2b shows that low frequency
# noise cannot be well approximated with a Gaussian distri-
# bution; however, by applying an increasing number of oc-
# taves of noise, the distribution becomes closer to a Gaussian
# distribution. This is paramount for our DDPM model as we
# assume our noising function is sampling from a Gaussian
# distribution. Therefore, unless stated otherwise, we use a
# starting frequency of ν = 2^6 (et pas 2^(-6) comme écrit dans l'article), octave of N = 6 and a de-
# cay of γ = 0.8. Furthermore, when generating the simplex
# noise, we shuffle the seed before every noise calculation,
# and take a slice t from the 3-dimensional noise function,
# as we found that artefacts were introduced when sampling
# from the 2-dimensional noise function
# 
# """
#             Returns a layered fractal noise in 2D\
#         :param shape: Shape of 2D tensor output\
#         :param octaves: Number of levels of fractal noise\
#         :param persistence: float between (0-1) -> Rate at which amplitude of each level decreases\
#         :param frequency: Frequency of initial octave of noise\
#         :return: Fractal noise sample with n lots of 2D images
#         """
# 


simplexObj = simplex.Simplex_CLASS()
simplexObj.newSeed()
# take a slice t from the 3-dimensional noise function as we found that artefacts 
# were introduced when sampling from the 2-dimensional noise function
simplex_noise = simplexObj.rand_3d_octaves(shape=(128,128, 128), octaves=6, persistence=0.8, frequency=64)[12,...]
#plt.figure(figsize=(2, 2))
#plt.imshow(simplex_noise, cmap="gray")
#plt.axis("off")
#plt.show()


# !!! **shuffle the seed before every noise calculation (if timestep = 100: 100 shuffles)**,
# 
# 
# and take a slice t from the 3-dimensional noise function,
# as we found that artefacts were introduced when sampling
# from the 2-dimensional noise function



def generate_simplex_noise(simplexObj, shape):
    """Generate spatially correlated simplex noise."""

    simplexObj.newSeed()

    if len(shape) == 2:
        # take a slice t from the 3-dimensional noise function as we found that artefacts 
        # were introduced when sampling from the 2-dimensional noise function
        simplex = simplexObj.rand_3d_octaves(shape=(shape[0], shape[0], shape[1]), octaves=6, persistence=0.8, frequency=64)[12,...]
    elif len(shape) == 3:
        simplex = simplexObj.rand_3d_octaves(shape=shape, octaves=6, persistence=0.8, frequency=64)
    elif len(shape) == 4 and shape[1] == 1: # to make it work with shapes of type (batch_size, 1, height, width)
        simplex = simplexObj.rand_3d_octaves(shape=(shape[0], shape[2], shape[3]), octaves=6, persistence=0.8, frequency=64)
        simplex = np.expand_dims(simplex, axis=1)

    return torch.tensor(simplex, dtype=torch.float32)



from monai.utils import StrEnum
from typing import Union

class DDPMPredictionType(StrEnum):
    """
    Set of valid prediction type names for the DDPM scheduler's `prediction_type` argument.

    epsilon: predicting the noise of the diffusion process
    sample: directly predicting the noisy sample
    v_prediction: velocity prediction, see section 2.4 https://imagen.research.google/video/paper.pdf
    """

    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"

class SimplexDDPMScheduler(DDPMScheduler):
    def __init__(self, *args, noise_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scale = noise_scale
        self.simplex_obj = simplex.Simplex_CLASS()
        self.simplex_obj.newSeed()

    #def step(
    #    self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator: torch.Generator | None = None
    #) -> tuple[torch.Tensor, torch.Tensor]:
    def step(
        self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, generator: Union[torch.Generator, None] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output: direct output from learned diffusion model.
            timestep: current discrete timestep in the diffusion chain.
            sample: current instance of sample being created by diffusion process.
            generator: random number generator.

        Returns:
            pred_prev_sample: Predicted previous sample
        """
        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[timestep - 1] if timestep > 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.prediction_type == DDPMPredictionType.EPSILON:
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.prediction_type == DDPMPredictionType.SAMPLE:
            pred_original_sample = model_output
        elif self.prediction_type == DDPMPredictionType.V_PREDICTION:
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output

        # 3. Clip "predicted x_0"
        if self.clip_sample:
            pred_original_sample = torch.clamp(
                pred_original_sample, self.clip_sample_values[0], self.clip_sample_values[1]
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep] ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise
        variance: torch.Tensor = torch.tensor(0)
        if timestep > 0:
            self.simplex_obj.newSeed()
            noise = generate_simplex_noise(self.simplex_obj, shape=model_output.size()).to(model_output.device)
            
            """ #TODO
            noise = torch.randn(
                model_output.size(),
                dtype=model_output.dtype,
                layout=model_output.layout,
                generator=generator,
                device=model_output.device,
            )"""
            variance = (self._get_variance(timestep, predicted_variance=predicted_variance) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample



device = torch.device(DEVICE_TYPE)
simplexObj = simplex.Simplex_CLASS()


model = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    channels=(64, 128, 256, 512, 512),  # Increased capacity
    attention_levels=(False, False, True, True, True),  # More attention
    num_head_channels=(0, 0, 64, 128, 128),  # Adjusted head channels
    num_res_blocks=2,  # Add residual blocks
    #dropout_cattn=0.1,  # Add crossattention layers dropout for regularization
)
model.to(device)

num_train_timesteps = 400
scheduler = SimplexDDPMScheduler(num_train_timesteps=num_train_timesteps)

optimizer = torch.optim.Adam(params=model.parameters(), lr=10e-5, betas=(0.9, 0.999), weight_decay=1e-6)

# Add learning rate scheduler
lr_scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-7)
# Alternative: lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)

inferer = DiffusionInferer(scheduler)

# Add EMA for better model stability
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

ema = EMA(model, decay=0.9999)



def train_epoch(epoch, best_val_epoch_loss, best_val_epoch):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            # Generate random noise
            #noise = torch.randn_like(images).to(device)
            noise = generate_simplex_noise(simplexObj, images.shape).to(device)

            # Create timesteps
            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

            # Improved loss computation with L1 regularization
            mse_loss = F.mse_loss(noise_pred.float(), noise.float())
            l1_loss = F.l1_loss(noise_pred.float(), noise.float())
            loss = mse_loss + 0.1 * l1_loss  # Combined loss

        scaler.scale(loss).backward()
        
        # Add gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update EMA
        ema.update()
        
        

        epoch_loss += loss.item()
        

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1), "lr": optimizer.param_groups[0]['lr']})

    # Update learning rate
    lr_scheduler.step()

    epoch_loss_list.append(epoch_loss / (step + 1))
    writer.add_scalar("train_loss", epoch_loss / (step + 1), epoch)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

    if (epoch + 1) % val_interval == 0:
        # Use EMA model for validation
        ema.apply_shadow()
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch.to(device)
            with torch.no_grad(), autocast(device_type=DEVICE_TYPE, enabled=True):
                noise = generate_simplex_noise(simplexObj, shape=images.shape).to(device)

                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()

            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

        val_epoch_loss_list.append(val_epoch_loss / (step + 1))
        writer.add_scalar("val_loss", val_epoch_loss / (step + 1), epoch)   # moi

        if val_epoch_loss < best_val_epoch_loss:
            best_val_epoch_loss = val_epoch_loss
            best_val_epoch = epoch + 1
            # Save EMA model
            torch.save(
                ema.shadow,
                os.path.join(ROOT_DIR+"AnoDiffExperiments/best_models/experiment_3", "exp_3_7_best_model_ema.pth"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(ROOT_DIR+"AnoDiffExperiments/best_models/experiment_3", "exp_3_7_best_model.pth"),
            )
            print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current val loss: {val_epoch_loss/(step + 1):.4f}"
                f"\nbest val loss: {best_val_epoch_loss/(step + 1):.4f}"
                f" at epoch: {best_val_epoch}"
            )
            writer.add_scalar("best_val_loss", best_val_epoch_loss/(step + 1), best_val_epoch)
        
        # Restore original model weights
        ema.restore()

            # can't visualize an inference image since we don't train from pure noise here
            #noise = generate_simplex_noise(simplexObj, shape=(1,1,IMAGE_SIZE, IMAGE_SIZE)).to(device)
            #noise = noise.to(device)
            #scheduler.set_timesteps(num_inference_steps=1000)
            #with autocast(device_type=DEVICE_TYPE, enabled=True):
            #    image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)
            #writer.add_image("sampled_image", image[0, 0].cpu().numpy(), global_step=epoch, dataformats="HW")
            #plt.figure(figsize=(2, 2))
            #plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
            #plt.tight_layout()
            #plt.axis("off")
            #plt.show()
    
    return best_val_epoch_loss, best_val_epoch

RESUME_TRAINING = False

if RESUME_TRAINING == False:
    print("STARTING NEW TRAINING")
    os.makedirs(ROOT_DIR+"AnoDiffExperiments/tensorboard/exp_3_7", exist_ok=True)
    writer = SummaryWriter(ROOT_DIR+"AnoDiffExperiments/tensorboard/exp_3_7")

    max_epochs = 20000
    val_interval = 4
    epoch_loss_list = []
    val_epoch_loss_list = []

    best_val_epoch_loss = np.inf
    best_val_epoch = 0

    scaler = GradScaler(DEVICE_TYPE)
    total_start = time.time()


    for epoch in range(max_epochs):
        best_val_epoch_loss, best_val_epoch = train_epoch(epoch, best_val_epoch_loss, best_val_epoch)

else:
    
    last_epoch = 1316
    print("RESUMING TRAINING from epoch {last_epoch}")
    os.makedirs(ROOT_DIR+"AnoDiffExperiments/tensorboard/exp_3_7", exist_ok=True)
    writer = SummaryWriter(ROOT_DIR+"AnoDiffExperiments/tensorboard/exp_3_7")

    max_epochs = 20000
    val_interval = 4
    epoch_loss_list = []
    val_epoch_loss_list = []

    best_val_epoch_loss = 0.001376*62
    best_val_epoch = 1316

    scaler = GradScaler(DEVICE_TYPE)
    total_start = time.time()

    model.load_state_dict(torch.load(os.path.join(ROOT_DIR+"AnoDiffExperiments/best_models/experiment_3", "exp_3_7_best_model.pth"), map_location=DEVICE_TYPE))

    for epoch in range(last_epoch,max_epochs):
        best_val_epoch_loss, best_val_epoch = train_epoch(epoch, best_val_epoch_loss, best_val_epoch)



total_time = time.time() - total_start
print(f"train completed, total time: {total_time}.")


