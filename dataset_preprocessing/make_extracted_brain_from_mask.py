import os
import nibabel as nib

#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

masks_directory = ROOT_DIR+"/datasets/FTRACT_brain_mask/"
full_head_directory = ROOT_DIR+"datasets/FTRACT_compressed_nifti/"
out_extracted_brain_directory = ROOT_DIR+"/datasets/FTRACT_extracted_brain/"

empty_header = nib.Nifti1Header()

for file in os.listdir(masks_directory):
    if file.endswith(".nii.gz") and not file.startswith("00115-Guys-0738"):

        mask_img = nib.load(masks_directory+file)
        full_head_img = nib.load(full_head_directory+file)

        extracted_brain_data = full_head_img.get_fdata()*mask_img.get_fdata()
        extracted_brain_img = nib.Nifti1Image(extracted_brain_data, full_head_img.affine, empty_header)
        nib.save(extracted_brain_img, out_extracted_brain_directory+file)
