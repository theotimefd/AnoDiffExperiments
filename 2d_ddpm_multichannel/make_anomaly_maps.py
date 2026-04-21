import os
import sys
sys.path.append("../..")

from scipy import stats
import nibabel as nib
import torch

from sample import my_sample, sample_thor
from utils.utils import *


def make_anomaly_maps(args, model, device, infer_scheduler, image_loader, image_paths, timesteps, output_folder, replace_existing_files=False):
    # multiple 2D inference stacked to make a 3D anomaly maps for a given nb timesteps
    # saves all the anomaly maps in the output_folder
    # reaplce_existing_files=False by default

    os.makedirs(output_folder, exist_ok=True)    

    basic_affine = nib.load(image_paths[0]).affine

    for i, image_batch in enumerate(image_loader):

        test_images = image_batch.to(device)

        def process_batch():
            infered_slices = []

            # infer slice by slice
            for slice_idx in range(args.slice_indexes_start, args.slice_indexes_end):
                if args.thor["enable"]:
                    _, pseudo_anomaly_masks_processed = sample_thor(args, model, device, test_images[...,slice_idx], infer_scheduler, timesteps=timesteps, return_intermediates=False)
                    infered_slice = torch.Tensor(stats.hmean(np.stack([p.cpu() for p in pseudo_anomaly_masks_processed]), axis=0)).to(device)
                else:
                    infered_slice = my_sample(args, model, device, test_images[...,slice_idx], infer_scheduler, timesteps=timesteps, return_intermediates=False)
                infered_slices.append(infered_slice.unsqueeze(-1))

            # stack the slices back to a 3D volume
            average_infered_image = torch.cat(infered_slices, dim=-1)

            for batch_idx in range(average_infered_image.shape[0]):
                for channel_idx in range(average_infered_image.shape[1]):
                    average_infered_image[batch_idx, channel_idx] = torch.clamp(scale_intensity_from_histogram_peak(average_infered_image[batch_idx, channel_idx], 2.0/7.0), 0.0, 1.0)

            # make the anomaly map (difference between infered and original)
            final_anomaly_map = torch.zeros_like(test_images)
            final_anomaly_map[...,args.slice_indexes_start:args.slice_indexes_end] = torch.abs(average_infered_image - test_images[...,args.slice_indexes_start:args.slice_indexes_end])


            # save the images
            for idx_in_batch in range(final_anomaly_map.shape[0]):
                image_id = i*test_images.shape[0] + idx_in_batch
                image_name = os.path.basename(image_paths[image_id])
                output_path = output_folder+f"{image_name.split('.')[0]}_t_{timesteps}.nii.gz"
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
                output_path = output_folder+f"{image_name.split('.')[0]}_t_{timesteps}.nii.gz"
                #if the output file doesn't exist already
                if not os.path.exists(output_path): # if just one file is missing, we need to process the batch
                    process_batch()
                    break