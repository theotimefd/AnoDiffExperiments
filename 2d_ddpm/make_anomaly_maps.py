import sys
sys.path.append("../..")
from sample import my_sample
import nibabel as nib
import os
import torch
from utils.utils import *



def make_anomaly_maps(args, device, infer_scheduler, image_loader, image_paths, timesteps, output_folder, replace_existing_files=False):
    # multiple 2D inference stacked to make a 3D anomaly maps for a given nb timesteps
    # saves all the anomaly maps in the output_folder
    # reaplce_existing_files=False by default

    basic_affine = nib.load(image_paths[0]).affine

    for i, image_batch in enumerate(image_loader):

        test_images = image_batch.to(device)

        def process_batch():
            infered_slices = []

            # infer slice by slice
            for slice_idx in range(args.slice_indexes_start, args.slice_indexes_end):
                infered_slice = my_sample(test_images[...,slice_idx], infer_scheduler, timesteps=timesteps, return_intermediates=False)
                infered_slices.append(infered_slice.unsqueeze(-1))

            # stack the slices back to a 3D volume
            average_infered_image = torch.cat(infered_slices, dim=-1)
            average_infered_image = torch.clamp(scale_intensity_from_histogram_peak(average_infered_image, 2.0/7.0), 0.0, 1.0)

            # make the anomaly map (difference between infered and original)
            final_anomaly_map = torch.zeros_like(test_images)
            final_anomaly_map[...,args.slice_indexes_start:args.slice_indexes_end] = torch.abs(average_infered_image - test_images[...,args.slice_indexes_start:args.slice_indexes_end])


            # save the images
            for idx_in_batch in range(final_anomaly_map.shape[0]):
                image_id = i*test_images.shape[0] + idx_in_batch
                image_name = os.path.basename(image_paths[image_id])
                output_path = output_folder+f"ano_map_{image_name}_f{timesteps}.nii.gz"
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
                output_path = output_folder+f"ano_map_{image_name}_ts_{timesteps}.nii.gz"
                #if the output file doesn't exist already
                if not os.path.exists(output_path): # if just one file is missing, we need to process the batch
                    process_batch()
                    break