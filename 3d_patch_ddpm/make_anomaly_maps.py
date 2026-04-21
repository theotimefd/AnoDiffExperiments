import sys
sys.path.append("../..")
import nibabel as nib
import os
import torch
from torch.amp import autocast
from tqdm import tqdm
from typing import List, Sequence, Tuple

from monai.inferers import LatentDiffusionInferer
from monai.utils import first

import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm

from monai.networks.schedulers import DDPMScheduler

from utils.utils import *

from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def scale_intensity_from_histogram_peak(input_image, target_value=1.0):
    # to be used only on mri images with intensities between 0 and 1
    input_np = input_image.cpu().numpy()

    hist, bin_edges = np.histogram(input_np.flatten(), bins=100, range=(np.max(input_np)/15.0, 0.8))

    peak_value = bin_edges[np.argmax(hist)]

    normalized_image = input_image / peak_value * target_value

    return normalized_image

def _generate_patch_slices(spatial_shape: Sequence[int], patch_size: Sequence[int], overlap: Sequence[int]):


    ranges: List[List[int]] = []

    for dim, size, ov in zip(spatial_shape, patch_size, overlap):
        step = max(size - ov, 1)

        if dim <= size:
            coords = [0]
        else:
            coords = list(range(0, max(dim - size, 0) + 1, step))
            if coords[-1] != dim - size:
                coords.append(dim - size)
        ranges.append(coords)
        
    for h in ranges[0]:
        for w in ranges[1]:
            for d in ranges[2]:
                yield (slice(h, h + patch_size[0]), slice(w, w + patch_size[1]), slice(d, d + patch_size[2]))



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


def _create_patch_weight(patch_size: Sequence[int], sigma_scale: float = 0.125) -> torch.Tensor:
    """
    Create a 3D Gaussian weight map that gives more importance to the center of the patch.
    This helps blend overlapping patches smoothly and eliminates seam artifacts.
    """
    weight = torch.ones(patch_size)
    
    for dim in range(3):
        size = patch_size[dim]

        # Create 1D Gaussian-like weight using cosine tapering
        # This gives weight 1 at center and smoothly decreases to ~0.5 at edges
        coords = torch.linspace(0, 1, size)

        # Cosine window (Hann-like): smooth transition from edges to center
        window = 0.5 * (1 - torch.cos(2 * np.pi * coords))

        #window = window * 0.5 + 0.5  # Range: 0.5 to 1.0
        window = window * 0.9 + 0.1  # Range: 0.9 to 1.0

        # Reshape for broadcasting
        shape = [1, 1, 1]
        shape[dim] = size
        window = window.view(shape)
        
        weight = weight * window
    
    return weight


def _run_patchwise_test(
    volume: torch.Tensor,
    patch_size: Sequence[int],
    overlap: Sequence[int],
    patch_batch_size: int,
    noise_type: str,
    simplexObj,
    model,
    infer_scheduler,
    num_timesteps: int,
    device,
):
    aggregator_pred = torch.zeros_like(volume, dtype=torch.float32)
    weight_sum = torch.zeros_like(volume, dtype=torch.float32)
    patch_queue: List[torch.Tensor] = []
    slice_queue: List[Tuple[slice, slice, slice]] = []
    total_patches = 0
    
    # Create weight map for smooth blending
    patch_weight = _create_patch_weight(patch_size).to(device)
    """
    Same as _run_patchwise_inference but goes all the way to the fully denoised image
    Uses weighted averaging to eliminate seam artifacts at patch boundaries.
    """

    def _flush_queue():
        nonlocal total_patches

        if not patch_queue:
            return
        
        batch_tensor = torch.cat(patch_queue, dim=0) # transforms all the patches into a single batch tensor

        preds = my_sample(model, device, noise_type, simplexObj, batch_tensor, infer_scheduler, num_timesteps)

        patch_count = batch_tensor.shape[0]
        total_patches += patch_count

        for idx, patch_slices in enumerate(slice_queue):
            target_slice = (slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])
            
            # Apply weighted contribution instead of simple addition
            aggregator_pred[target_slice] += preds[idx].unsqueeze(0).float() * patch_weight
            weight_sum[target_slice] += patch_weight  # Accumulate weights instead of counts


        patch_queue.clear()
        slice_queue.clear()

    for patch_slices in _generate_patch_slices(volume.shape[-3:], patch_size, overlap): # goes through the slices that define each patch

        patch = volume[(slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])] # extracts the patch using the slices

        patch_queue.append(patch) # patch_queue stores all the patches for the current volume batch
        slice_queue.append(patch_slices)

        if len(patch_queue) >= patch_batch_size: # makes sure there aren't too many patches at one time (memory issues)
            _flush_queue() # does the inference and computes loss

    _flush_queue()

    weight_sum = torch.clamp(weight_sum, min=1e-8)
    stitched_pred = aggregator_pred / weight_sum  # Weighted average for smooth blending

    return stitched_pred

def process_volume(idx, 
                   test_images, 
                   image_paths, 
                   i, 
                   infer_patch_size, 
                   patch_overlap, 
                   patch_infer_batch_size, 
                   args, 
                   simplexObj, 
                   model, 
                   infer_scheduler, 
                   infer_timesteps, 
                   device, 
                   output_folder, 
                   basic_affine, 
                   replace_existing_files,
                   nb_inferences=1):

    image_id = i*test_images.shape[0] + idx
    image_name = os.path.basename(image_paths[image_id])
    output_path = output_folder+f"{image_name.split('.')[0]}_t_{infer_timesteps}.nii.gz"
    
    if os.path.exists(output_path) and not replace_existing_files:
        tprint(f"Reconstructed image for patient {image_name.split('.')[0]} at noise timesteps {infer_timesteps} already exists, skipping inference.")
        return

    volume = test_images[idx : idx + 1] #TODO Here it does it volume per volume, any way to run the inference by batch?
    volume = volume.to(device)
    
    stitched_preds = []

    for inference_idx in range(nb_inferences):

        stitched_pred = _run_patchwise_test(
            volume,
            infer_patch_size,
            patch_overlap,
            patch_infer_batch_size,
            args.noise["type"],
            simplexObj,
            model,
            infer_scheduler,
            infer_timesteps,
            device,
        )
        stitched_preds.append(stitched_pred)

    average_infered_image = torch.mean(torch.stack(stitched_preds, dim=0), dim=0)
    average_infered_image = torch.clamp(scale_intensity_from_histogram_peak(average_infered_image, 2.0/7.0), 0.0, 1.0)

    # make the anomaly map (difference between infered and original)
    final_anomaly_map = torch.abs(average_infered_image - volume)
    
    #if the output file doesn't exist already
    if not os.path.exists(output_path):
        nib.save(nib.Nifti1Image(final_anomaly_map.squeeze().cpu().numpy(), basic_affine), output_path)
    elif replace_existing_files:
        nib.save(nib.Nifti1Image(final_anomaly_map.squeeze().cpu().numpy(), basic_affine), output_path)

def make_anomaly_maps(args, model, device, 
                      infer_scheduler, 
                      image_loader, 
                      image_paths, 
                      infer_timesteps, 
                      output_folder, 
                      replace_existing_files=False,
                      nb_inferences=1):
    # multiple 2D inference stacked to make a 3D anomaly maps for a given nb timesteps
    # saves all the anomaly maps in the output_folder
    # reaplce_existing_files=False by default


    os.makedirs(output_folder, exist_ok=True)    

    basic_affine = nib.load(image_paths[0]).affine

    infer_patch_size = args.patch_size
    patch_overlap = args.dataset["patch_overlap"]

    patch_infer_batch_size = args.dataset["batch_size"]

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads()) 
    torch.autograd.set_detect_anomaly(False)

    simplexObj = None

    if args.noise["type"] == "simplex":
        simplexObj = simplex.Simplex_CLASS()
        infer_scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], 
                                                            schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], 
                                                            persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        infer_scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])

    #inferer = LatentDiffusionInferer(infer_scheduler, scale_factor=scale_factor)

    for i, image_batch in enumerate(image_loader):

        test_images = image_batch.to(device)

        with torch.no_grad():
            with autocast("cuda", enabled=True):

                volumes = test_images.shape[0]

                tprint(f"making anomaly maps for batch {i+1}/{len(image_loader)} with {volumes} volumes at noise timesteps {infer_timesteps}")


                # use multiprocessing to process two volumes in parallel
                # Set spawn method for CUDA compatibility
                mp.set_start_method('spawn', force=True)
                num_processes = 4
                with ProcessPoolExecutor(max_workers=num_processes, mp_context=mp.get_context('spawn')) as executor:
                    futures = [executor.submit(process_volume, idx, test_images, image_paths, i, infer_patch_size, patch_overlap, patch_infer_batch_size, args, simplexObj, model, infer_scheduler, infer_timesteps, device, output_folder, basic_affine, replace_existing_files, nb_inferences) for idx in range(volumes)]
                    for future in as_completed(futures):
                        future.result()  # To raise exceptions if any
                    