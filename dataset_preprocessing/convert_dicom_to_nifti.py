import pydicom
import nibabel as nib
import dicom2nifti
import os

#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"

dicom_directory = ROOT_DIR+'datasets/fastMRI_brain_DICOM/'
output_file = 'output/path/for/nifti_file.nii'

dicom2nifti.convert_directory(dicom_directory, os.path.dirname(output_file))

# The NIfTI file will be saved in the specified output directory