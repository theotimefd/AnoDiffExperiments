import os
import sys
sys.path.append("../..")

from scipy import stats
import nibabel as nib
import torch

from sample import my_sample, sample_thor
from utils.utils import *


def make_anomaly_maps(args, model, device, 
                      infer_scheduler, 
                      image_loader, 
                      image_paths, 
                      timesteps, 
                      output_folder, 
                      replace_existing_files=False, 
                      ):
    # multiple 2D inference stacked to make a 3D anomaly maps for a given nb timesteps
    # saves all the anomaly maps in the output_folder
    # reaplce_existing_files=False by default

    nb_inferences = args.nb_inferences

    dtprint(f"launching def make_anomaly_maps with timesteps {timesteps} and nb_inferences {nb_inferences}")

    os.makedirs(output_folder, exist_ok=True)    

    dtprint(f"output_folder: {output_folder}")

    basic_affine = nib.load(image_paths[0]).affine

    dtprint(f"number of batches: {len(image_loader)}")

    dtprint(f"image loader")
    dtprint(image_loader)

    for i, image_batch in enumerate(image_loader):
        dtprint(f"Processing batch {i+1}/{len(image_loader)}")
        test_images = image_batch.to(device)

        def process_batch():

            infered_images = []

            for inference_idx in range(nb_inferences):
                infered_slices = []
                dtprint(f"  Inference {inference_idx+1}/{nb_inferences} for batch {i+1}/{len(image_loader)}")

                # infer slice by slice
                for slice_idx in range(args.slice_indexes_start, args.slice_indexes_end):
                    if args.thor["enable"]:
                        _, pseudo_anomaly_masks_processed = sample_thor(args, model, device, test_images[...,slice_idx], infer_scheduler, timesteps=timesteps, return_intermediates=False)
                        infered_slice = torch.Tensor(stats.hmean(np.stack([p.cpu() for p in pseudo_anomaly_masks_processed]), axis=0)).to(device)
                    else:
                        infered_slice = my_sample(args, model, device, test_images[...,slice_idx], infer_scheduler, timesteps=timesteps, return_intermediates=False)
                    infered_slices.append(infered_slice.unsqueeze(-1))

                # stack the slices back to a 3D volume
                infered_images.append(torch.cat(infered_slices, dim=-1))
            
            average_infered_image = torch.mean(torch.stack(infered_images, dim=0), dim=0)
            

            #go through infered images in the batch
            for b in range(average_infered_image.shape[0]):
                average_infered_image[b] = torch.clamp(scale_intensity_from_histogram_peak(average_infered_image[b], 2.0/7.0), 0.0, 1.0)

            # make the anomaly map (difference between infered and original)
            final_anomaly_map = torch.zeros_like(test_images)
            final_anomaly_map[...,args.slice_indexes_start:args.slice_indexes_end] = torch.abs(average_infered_image - test_images[...,args.slice_indexes_start:args.slice_indexes_end])


            # save the images
            for idx_in_batch in range(final_anomaly_map.shape[0]):
                image_id = i*test_images.shape[0] + idx_in_batch
                image_name = os.path.basename(image_paths[image_id])

                output_path = output_folder+f"{image_name.split('.')[0]}_t_{timesteps}.nii.gz"

                #if the output file doesn't exist already
                if not os.path.exists(output_path) or replace_existing_files: # if just one file is missing, we need to process the batch
                    dtprint(f"  Saving anomaly map for image {image_name} at {output_path}")
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