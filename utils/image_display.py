import nibabel as nib
import matplotlib.pyplot as plt
import glob
import os

def show_single_nifti(path):
    
    
    img = nib.load(path)
    data = img.get_fdata()
    print(f"Image shape: {data.shape}")
    
    # Get middle slices
    mid_axial = data.shape[2] // 2
    mid_coronal = data.shape[1] // 2
    mid_sagittal = data.shape[0] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(data[:, :, mid_axial], cmap='gray', origin='lower')
    axes[0].set_title('Axial')
    axes[0].axis('off')
    
    axes[1].imshow(data[:, mid_coronal, :], cmap='gray', origin='lower')
    axes[1].set_title('Coronal')
    axes[1].axis('off')
    
    axes[2].imshow(data[mid_sagittal, :, :], cmap='gray', origin='lower')
    axes[2].set_title('Sagittal')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_first_4_nifti(folder_path):

    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.nii.gz")))[:4]

    # Plot the images
    plt.suptitle("First 4 images")
    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
    for i, image_path in enumerate(image_paths):
        image = nib.load(image_path).get_fdata()
        print(f"Image {i+1} shape: {image.shape}")
        # Extract slices
        axial_slice = image[image.shape[0] // 2, :, :]  # Middle axial slice
        sagittal_slice = image[:, image.shape[1] // 2, :]  # Middle sagittal slice
        coronal_slice = image[:, :, image.shape[2] // 2]  # Middle coronal slice

        # Plot slices
        axes[i, 0].imshow(axial_slice, cmap="gray")
        axes[i, 0].set_title(f"Image {i+1} - Sagittal")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(sagittal_slice, cmap="gray")
        axes[i, 1].set_title(f"Image {i+1} - Coronal")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(coronal_slice, cmap="gray")
        axes[i, 2].set_title(f"Image {i+1} - Axial")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()