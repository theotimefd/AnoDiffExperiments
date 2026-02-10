"""
Docstring for utils.dataset_preprocessing
This module contains functions for preprocessing NIFTI files, including: 
- padding (singlefile, folder and folder_multithreaded)
- n4 bias field correction (singlefile, folder and folder_multithreaded)
- resampling (singlefile, folder and folder_multithreaded)
- registration (singlefile, folder and folder_multithreaded)
"""
import os
import ants
import numpy as np
from tqdm import tqdm
import glob
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
from concurrent.futures import ThreadPoolExecutor, as_completed


# ----------------------------- Padding -----------------------------
# Single file
def pad_img(nifti_path, target_size):
    # Pad images to target size with SimpleITK
    
    img = sitk.ReadImage(nifti_path)

    original_size = img.GetSize()

    lower_pad = [(target_size[i] - original_size[i]) // 2 for i in range(3)]
    upper_pad = [target_size[i] - original_size[i] - lower_pad[i] for i in range(3)]


    # Calculate the required padding for each dimension
    padded_img = sitk.ConstantPad(img, lower_pad, upper_pad, 0)

    return padded_img

# Folder
def pad_nifti_files_folder(input_folder, output_folder, target_size):
    """
    Pads all NIfTI files in the input_folder folder to the target size using SimpleITK.

    Parameters:
    - input_folder: Path to the folder containing NIfTI files.
    - output_folder: Path to save the padded NIfTI files.
    - target_size: Desired output size (tuple of 3 integers).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all NIfTI files in the input_folder
    nifti_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
    
    if not nifti_files:
        print("No NIfTI files found in the input folder.")
        return

    for nifti_file in tqdm(nifti_files):
        output_filepath = os.path.join(output_folder, os.path.basename(nifti_file))
        if not os.path.isfile(output_filepath):
            padded_img = pad_img(nifti_file, target_size)
            sitk.WriteImage(padded_img, output_filepath)
        else:
            print("skipped padding already done")

# Folder multithreaded
def pad_nifti_files_folder_multithreaded(input_folder, output_folder, target_size, max_workers=4):
    """
    Pads all NIfTI files in the input_folder folder to the target size using SimpleITK with multithreading.

    Parameters:
    - input_folder: Path to the folder containing NIfTI files.
    - output_folder: Path to save the padded NIfTI files.
    - target_size: Desired output size (tuple of 3 integers).
    - max_workers: Maximum number of threads to use.
    """
    os.makedirs(output_folder, exist_ok=True)

    nifti_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
    
    if not nifti_files:
        print("No NIfTI files found in the input folder.")
        return

    def process_file(nifti_file):
        output_filepath = os.path.join(output_folder, os.path.basename(nifti_file))
        if not os.path.isfile(output_filepath):
            padded_img = pad_img(nifti_file, target_size)
            sitk.WriteImage(padded_img, output_filepath)
            return f"Processed {nifti_file}"
        else:
            return f"Skipped {nifti_file} (already done)"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in nifti_files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
# ----------------------------- N4 Bias Field Correction -----------------------------
# Single file
def n4_bias_field_correction(nifti_path):
    # Apply N4 bias field correction to a single image using SimpleITK

    img = sitk.ReadImage(nifti_path, sitk.sitkFloat32)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(img)

    return corrected_img

# Folder
def n4_bias_field_correction_folder(input_folder, output_folder):
    """
    Applies N4 bias field correction to all NIfTI files in the input_folder folder using SimpleITK.

    Parameters:
    - input_folder: Path to the folder containing NIfTI files.
    - output_folder: Path to save the corrected NIfTI files.
    """
    os.makedirs(output_folder, exist_ok=True)

    nifti_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))

    for nifti_file in tqdm(nifti_files):
        output_filepath = os.path.join(output_folder, os.path.basename(nifti_file))
        if not os.path.isfile(output_filepath):
            corrected_img = n4_bias_field_correction(nifti_file)
            sitk.WriteImage(corrected_img, output_filepath)
        else:
            print("skipped n4 bias field correction already done")

# Folder multithreaded
def n4_bias_field_correction_folder_multithreaded(input_folder, output_folder, max_workers=4):
    """
    Applies N4 bias field correction to all NIfTI files in the input_folder folder using SimpleITK with multithreading.

    Parameters:
    - input_folder: Path to the folder containing NIfTI files.
    - output_folder: Path to save the corrected NIfTI files.
    - max_workers: Maximum number of threads to use.
    """
    os.makedirs(output_folder, exist_ok=True)

    nifti_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))

    def process_file(nifti_file):
        output_filepath = os.path.join(output_folder, os.path.basename(nifti_file))
        if not os.path.isfile(output_filepath):
            corrected_img = n4_bias_field_correction(nifti_file)
            sitk.WriteImage(corrected_img, output_filepath)
            return f"Processed {nifti_file}"
        else:
            return f"Skipped {nifti_file} (already done)"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in nifti_files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()

# ----------------------------- Resampling -----------------------------
# Single file
def resample_img(nifti_path, out_spacing=[1.0, 1.0, 1.0]):
    # Resample images to 2mspecified spacing with SimpleITK

    img = sitk.ReadImage(nifti_path)

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(img.GetPixelIDValue())


    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(img)

# Folder
def resample_nifti_files_folder(input_folder, output_folder, out_spacing=[1.0, 1.0, 1.0]):
    """
    Resamples all NIfTI files in the input_folder folder to the specified spacing using SimpleITK.

    Parameters:
    - input_folder: Path to the folder containing NIfTI files.
    - output_folder: Path to save the resampled NIfTI files.
    - out_spacing: Desired output spacing (tuple of 3 floats).
    """
    os.makedirs(output_folder, exist_ok=True)

    nifti_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
    
    if not nifti_files:
        print("No NIfTI files found in the input folder.")
        return

    for nifti_file in tqdm(nifti_files):
        output_filepath = os.path.join(output_folder, os.path.basename(nifti_file))
        if not os.path.isfile(output_filepath):
            resampled_img = resample_img(nifti_file, out_spacing)
            sitk.WriteImage(resampled_img, output_filepath)
        else:
            print("skipped resampling already done")

# Folder multithreaded
def resample_nifti_files_folder_multithreaded(input_folder, output_folder, out_spacing=[1.0, 1.0, 1.0], max_workers=4):
    """
    Resamples all NIfTI files in the input_folder folder to the specified spacing using SimpleITK with multithreading.

    Parameters:
    - input_folder: Path to the folder containing NIfTI files.
    - output_folder: Path to save the resampled NIfTI files.
    - out_spacing: Desired output spacing (tuple of 3 floats).
    - max_workers: Maximum number of threads to use.
    """
    os.makedirs(output_folder, exist_ok=True)

    nifti_files = sorted(glob.glob(os.path.join(input_folder, "*.nii*")))
    
    if not nifti_files:
        print("No NIfTI files found in the input folder.")
        return

    def process_file(nifti_file):
        output_filepath = os.path.join(output_folder, os.path.basename(nifti_file))
        if not os.path.isfile(output_filepath):
            resampled_img = resample_img(nifti_file, out_spacing)
            sitk.WriteImage(resampled_img, output_filepath)
            return f"Processed {nifti_file}"
        else:
            return f"Skipped {nifti_file} (already done)"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, f): f for f in nifti_files}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()

# ----------------------------- Register -----------------------------
# Folder
def register_nifti_files_folder(fixed_reference, dataset_path, output_path, register_masks=False, dataset_masks_path=None, output_masks_path=None):
    """
    Registers all NIfTI files in the dataset_path folder to the same space using ANTs.

    Parameters:
    - dataset_path: Path to the folder containing NIfTI files.
    - output_path: Path to save the registered NIfTI files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Get all NIfTI files in the dataset_path
    nifti_files = sorted(glob.glob(os.path.join(dataset_path, "*.nii*")))
    if register_masks:
        nifti_masks_files = sorted(glob.glob(os.path.join(dataset_masks_path, "*.nii*")))
    print("nifti_files", nifti_files)
    if register_masks: print("nifti_masks_files", nifti_masks_files)
    if not nifti_files:
        print("No NIfTI files found in the dataset path.")
        return

    # Use the first file as the fixed image (reference)
    fixed_image = ants.image_read(fixed_reference)

    for i, moving_file in enumerate(tqdm(nifti_files)):
        
        output_filepath = os.path.join(output_path, os.path.basename(moving_file))

        if register_masks:
            mask_file = nifti_masks_files[i]
            output_mask_filepath = os.path.join(output_masks_path, os.path.basename(mask_file))
        
        
        if not os.path.isfile(output_filepath):
            moving_image = ants.image_read(moving_file)
            
            # Perform registration
            registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Similarity') # https://antspy.readthedocs.io/en/latest/registration.html types of transforms

            # for the mask
            if register_masks:
                mask_image = ants.image_read(mask_file)
                transform = registration['fwdtransforms']
                mask_moved = ants.apply_transforms(fixed_image, mask_image, transformlist=transform, interpolation='nearestNeighbor')
                mask_moved.to_file(output_mask_filepath)

            # Save the registered image
            ants.image_write(registration['warpedmovout'], output_filepath)
    
            #print(f"Registered {moving_file} and saved to {output_filepath}")
        else:
            print("skipped registration already done")

# Folder multithreaded
def register_nifti_files_folder_multithreaded(fixed_reference, dataset_path, output_path, register_masks=False, dataset_masks_path=None, output_masks_path=None, max_workers=4):
    """
    Registers all NIfTI files in the dataset_path folder to the same space using ANTs with multithreading.

    Parameters:
    - fixed_reference: Path to the fixed reference NIfTI file.
    - dataset_path: Path to the folder containing NIfTI files.
    - output_path: Path to save the registered NIfTI files.
    - register_masks: Whether to also register mask files.
    - dataset_masks_path: Path to the folder containing mask NIfTI files.
    - output_masks_path: Path to save the registered mask NIfTI files.
    - max_workers: Maximum number of threads to use.
    """
    os.makedirs(output_path, exist_ok=True)
    if register_masks and output_masks_path:
        os.makedirs(output_masks_path, exist_ok=True)

    nifti_files = sorted(glob.glob(os.path.join(dataset_path, "*.nii*")))
    if register_masks:
        nifti_masks_files = sorted(glob.glob(os.path.join(dataset_masks_path, "*.nii*")))
    
    if not nifti_files:
        print("No NIfTI files found in the dataset path.")
        return

    fixed_image = ants.image_read(fixed_reference)

    def process_file(args):
        if register_masks:
            idx, moving_file = args
            mask_file = nifti_masks_files[idx]
            output_mask_filepath = os.path.join(output_masks_path, os.path.basename(mask_file))
        else:
            _, moving_file = args
        
        output_filepath = os.path.join(output_path, os.path.basename(moving_file))
        
        if not os.path.isfile(output_filepath):
            moving_image = ants.image_read(moving_file)
            
            registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='Similarity')
            
            if register_masks:
                mask_image = ants.image_read(mask_file)
                transform = registration['fwdtransforms']
                mask_moved = ants.apply_transforms(fixed_image, mask_image, transformlist=transform, interpolation='nearestNeighbor')
                mask_moved.to_file(output_mask_filepath)
            
            ants.image_write(registration['warpedmovout'], output_filepath)
            return f"Processed {moving_file}"
        else:
            return f"Skipped {moving_file} (already done)"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, (i, f)): f for i, f in enumerate(nifti_files)}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()