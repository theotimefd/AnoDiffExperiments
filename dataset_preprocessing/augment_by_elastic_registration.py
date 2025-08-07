import os
import glob

import ants

from tqdm import tqdm

#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

HCP_YA_input_folder_registered = ROOT_DIR+"datasets/final_adc_dataset_small/HCP-YA_registered/"
AINI_stroke_input_folder_registered = ROOT_DIR+"datasets/final_adc_dataset_small/AIT_final_registered/"
Dallas_input_folder_registered = ROOT_DIR+"datasets/final_adc_dataset_small/Dallas_registered/"

image_paths_HCP_YA = sorted(glob.glob(os.path.join(HCP_YA_input_folder_registered, "*.nii.gz")))
image_paths_AINI_stroke = sorted(glob.glob(os.path.join(AINI_stroke_input_folder_registered, "*.nii.gz")))
image_paths_Dallas = sorted(glob.glob(os.path.join(Dallas_input_folder_registered, "*.nii.gz")))

combined_image_paths = image_paths_HCP_YA + image_paths_AINI_stroke + image_paths_Dallas

exclude = [ROOT_DIR+"Dallas_registered/sub-3242_ses-wave2_ADC.nii.gz",
ROOT_DIR+"AIT_final_registered/aini-stroke-17579_425560_ADC_HR_DIFF_RESOLVE_3MM_FP_ADC.nii.gz",
ROOT_DIR+"AIT_final_registered/aini-stroke-13607_424097_ADC_HR_DIFF_RESOLVE_3MM_FP_ADC.nii.gz"]

import random

# for every image in combined_image_paths, choose two random images and register them elastically
# save the result in a new folder

output_folder = ROOT_DIR + "datasets/final_adc_dataset_small/Elastic_augmentations_1/"
os.makedirs(output_folder, exist_ok=True)

max_num_augmentations = 500

for i in tqdm(range(max_num_augmentations)):

        fixed_path = combined_image_paths[random.randint(0, len(combined_image_paths) - 1)] 
        moving_path = combined_image_paths[random.randint(0, len(combined_image_paths) - 1)]
        
        moving_path_basename = os.path.splitext(os.path.splitext(os.path.basename(moving_path))[0])[0]
        fixed_path_basename = os.path.splitext(os.path.splitext(os.path.basename(fixed_path))[0])[0]

        output_name = f"{fixed_path_basename}_{moving_path_basename}_elastic_augmented.nii.gz"

        if fixed_path_basename == moving_path_basename:
            print(f"Skipping {output_name}, fixed and moving images are the same.")
        elif os.path.exists(os.path.join(output_folder, output_name)):
            print(f"Skipping {output_name}, already exists.")
        elif fixed_path in exclude or moving_path in exclude:
            print(f"Skipping {output_name}, is in exclude list.")
        else:

            fixed = ants.image_read(fixed_path)
            moving = ants.image_read(moving_path)

            reg = ants.registration(fixed, moving, type_of_transform='SyN')
            registered_img = reg['warpedmovout']
            
            output_path = os.path.join(output_folder, output_name)
            ants.image_write(registered_img, output_path)

