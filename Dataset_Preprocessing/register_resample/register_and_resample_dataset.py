"""
Module: register_and_resample_dataset.py

This module provides a function to perform rigid registration followed by resampling to a specified voxel spacing
It uses SimpleITK for image I/O, registration, and resampling.

Usage:
    from nifti_align_resample import align_and_resample_dataset

    align_and_resample_dataset(
        reference_image_path="path/to/reference.nii.gz",
        dataset_folder="path/to/folder/with/niftis",
        output_folder="path/to/save/aligned/images"
    )
"""

import os
import SimpleITK as sitk
from glob import glob
import numpy as np
from tqdm import tqdm 

def resample_img(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    # Resample images to 2mspecified spacing with SimpleITK

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())


    resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)



def register_and_resample_dataset(reference_image_path, dataset_folder, output_folder, spacing):
    """
    perform rigid registration followed by resampling to a specified voxel spacing
    It uses SimpleITK for image I/O, registration, and resampling.


    Parameters:
    - reference_image_path (str): Path to the reference NIfTI image.
    - dataset_folder (str): Folder containing the input NIfTI images.
    - output_folder (str): Folder where the aligned and resampled images will be saved.
    - spacing (tuple): Desired voxel spacing for the output images (x, y, z).
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load and resample reference image to specified spacing
    reference = sitk.ReadImage(reference_image_path)
    reference = sitk.Cast(reference, sitk.sitkFloat32)

    reference = resample_img(reference, out_spacing=spacing)

    # Find all .nii or .nii.gz files in the dataset folder
    nifti_paths = sorted(glob(os.path.join(dataset_folder, '*.nii*')))

    for moving_path in tqdm(nifti_paths):
        moving = sitk.ReadImage(moving_path)
        moving = sitk.Cast(moving, sitk.sitkFloat32)

        # Initialize transform
        initial_transform = sitk.CenteredTransformInitializer(
            reference,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # Set up registration method
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(
            learningRate=1.0,
            numberOfIterations=100,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetInitialTransform(initial_transform, inPlace=False)
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Execute registration
        final_transform = registration.Execute(reference, moving)

        # Resample to reference space
        resampled = sitk.Resample(
            moving,
            reference,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID()
        )

        resampled = resample_img(resampled, out_spacing=spacing)

        # Save output
        filename = os.path.basename(moving_path)
        output_path = os.path.join(output_folder, filename.replace('.nii', '_registered_resampled.nii').replace('.gz', ''))
        sitk.WriteImage(resampled, output_path)
        #print(f"Saved: {output_path}")
