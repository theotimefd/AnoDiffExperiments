import sys
sys.path.append("../..")
import nibabel as nib
import os
import torch
from torch.amp import autocast
from tqdm import tqdm


from monai.inferers import LatentDiffusionInferer
from monai.utils import first

from utils.utils import *


def make_anomaly_maps(args, autoencoder, unet, device, scheduler, image_loader, image_paths, infer_timesteps, output_folder, replace_existing_files=False):
    # multiple 2D inference stacked to make a 3D anomaly maps for a given nb timesteps
    # saves all the anomaly maps in the output_folder
    # reaplce_existing_files=False by default

    os.makedirs(output_folder, exist_ok=True)    

    basic_affine = nib.load(image_paths[0]).affine


    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads()) 
    torch.autograd.set_detect_anomaly(False)

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    with torch.no_grad():
        with autocast("cuda", enabled=True):
            check_data = first(image_loader)
            z = autoencoder.encode_stage_2_inputs(check_data.to(device))

    scale_factor = 1 / torch.std(z)

    #inferer = LatentDiffusionInferer(infer_scheduler, scale_factor=scale_factor)

    for i, image_batch in enumerate(image_loader):

        test_images = image_batch.to(device)

        def process_batch():

            with torch.no_grad():
                with autocast("cuda", enabled=True):

                    latents = autoencoder.encode_stage_2_inputs(test_images)    

                    # Add noise to latents
                    noise = torch.randn_like(latents).to(device)
                    timesteps = torch.randint(0, infer_timesteps, (latents.shape[0],), device=device).long()
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                    
                    # Denoise completely using the UNet
                    scheduler.set_timesteps(scheduler.num_train_timesteps)
                    current_latents = noisy_latents * scale_factor
                    
                    for t in tqdm(range(infer_timesteps-1, -1, -1)):
                        noise_pred = unet(current_latents, timesteps=torch.tensor([t], device=device).expand(latents.shape[0]))
                        current_latents, _ = scheduler.step(noise_pred, t, current_latents)
                    
                    # Decode the denoised latents
                    current_latents = current_latents / scale_factor

                    reconstructed_images = autoencoder.decode(current_latents)
                    normalized_reconstructed_images = torch.zeros_like(reconstructed_images)
                    for volume in range(reconstructed_images.shape[0]):
                        normalized_reconstructed_images[volume] = scale_intensity_from_histogram_peak(reconstructed_images[volume], target_value=2.0/7.0)

                    # make the anomaly map (difference between infered and original)
                    final_anomaly_map = torch.abs(normalized_reconstructed_images - test_images)


                    # save the images
                    for idx_in_batch in range(final_anomaly_map.shape[0]):
                        image_id = i*test_images.shape[0] + idx_in_batch
                        image_name = os.path.basename(image_paths[image_id])
                        output_path = output_folder+f"{image_name.split('.')[0]}_t_{infer_timesteps}.nii.gz"
                        #if the output file doesn't exist already
                        if not os.path.exists(output_path):
                            nib.save(nib.Nifti1Image(final_anomaly_map[idx_in_batch].squeeze().cpu().numpy(), basic_affine), output_path)
                        elif replace_existing_files:
                            nib.save(nib.Nifti1Image(final_anomaly_map[idx_in_batch].squeeze().cpu().numpy(), basic_affine), output_path)
            
        if replace_existing_files:
            process_batch()
        else:
            
            for idx_in_batch in range(test_images.shape[0]):
                image_id = i*test_images.shape[0] + idx_in_batch
                image_name = os.path.basename(image_paths[image_id])
                output_path = output_folder+f"{image_name.split('.')[0]}_t_{infer_timesteps}.nii.gz"
                #if the output file doesn't exist already
                if not os.path.exists(output_path): # if just one file is missing, we need to process the batch
                    process_batch()
                    break