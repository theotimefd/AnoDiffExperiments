import numpy as np
import random

import torch
from monai import transforms



class Get2DSlice(transforms.Transform):
    """
    Fetch the middle slice of a 3D volume.
    Args:
        axis: The axis along which to slice the volume. 0 for axial, 1 for coronal, 2 for sagittal.
        offset : Offset the index by a specified amount (default=0)
    """

    def __init__(
        self,
        axis: int = 0,
        offset: int=0
    ):
        super().__init__()
        self.axis = axis
        self.offset = offset

    def __call__(self, data):
        #print(data.shape)
        if self.axis==0:
            return data[:, data.shape[1]//2+self.offset,:,:]
        elif self.axis==1:
            return data[:, :,data.shape[2]//2+self.offset,:]
        elif self.axis==2:
            return data[:, :, :,data.shape[3]//2+self.offset]


class Get2DSliceWithRandomOffset(transforms.RandomizableTransform):
    """
    Will return the middle slice with a random offset in addition to the specified fixed offset.
    Args:
        axis: The axis along which to slice the volume. 0 for axial, 1 for coronal, 2 for sagittal.
        offset : Offset the index by a specified amount (default=0)
    """

    def __init__(
        self,
        axis: int = 0,
        fixed_offset: int=0,
        range_offset: int=5
    ):
        super().__init__()
        self.axis = axis
        self.fixed_offset = fixed_offset
        self.rand_offset = 0
        self.range_offset = range_offset


    def randomize(self):
        super().randomize(None)
        self.rand_offset = random.randint(-self.range_offset, self.range_offset)

    def __call__(self, data):
        #print(data.shape)
        self.randomize()

        #print(self.rand_offset)
        if self.axis==0:
            return data[:, data.shape[1]//2+self.fixed_offset+self.rand_offset,:,:]
        elif self.axis==1:
            return data[:, :,data.shape[2]//2+self.fixed_offset+self.rand_offset,:]
        elif self.axis==2:
            return data[:, :, :,data.shape[3]//2+self.fixed_offset+self.rand_offset]

class Get2DSliceFromIndexes(transforms.RandomizableTransform):
    """
    Will return a random slice from the specified indexes.
    Args:
        indexes_start: The starting index for the random slice.
        indexes_end: The ending index for the random slice.
    """

    def __init__(
        self,
        axis: int = 0,
        indexes_start: int = 0,
        indexes_end: int = 15
    ):
        super().__init__()
        self.axis = axis
        self.indexes_start = indexes_start
        self.indexes_end = indexes_end

    def randomize(self):
        super().randomize(None)
        self.rand_offset = random.randint(self.indexes_start, self.indexes_end)

    def __call__(self, data):
        #print(data.shape)
        self.randomize()

        #print(self.rand_offset)
        if self.axis==0:
            return data[:, self.rand_offset,:,:]
        elif self.axis==1:
            return data[:, :,self.rand_offset,:]
        elif self.axis==2:
            return data[:, :, :,self.rand_offset]


class SetBackgroundToZero(transforms.Transform):
    """
    Custom MONAI transform that zeros out voxels with the most frequent intensity value.
    
    Args:
        keys (str or list): Keys of the dictionary to apply the transform to.
        tolerance (int): Optional range around the mode value to also zero.
    """
    def __init__(self, tolerance: int = 0):
        super().__init__()
        self.tolerance = tolerance

    def __call__(self, data):
        
            
        is_tensor = isinstance(data, torch.Tensor)
        data_np = data.cpu().numpy() if is_tensor else data

        # Flatten and compute histogram
        flat = data_np.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        mode_val = unique[np.argmax(counts)]

        # Apply tolerance if specified
        if self.tolerance > 0:
            mask = np.isin(data_np, range(mode_val - self.tolerance, mode_val + self.tolerance + 1))
        else:
            mask = data_np == mode_val

        # Zero out the background
        data_np[mask] = 0

        # Put back in original type
        data = torch.from_numpy(data_np) if is_tensor else data_np

        return data

class ScaleIntensityFromHistogramPeak(transforms.Transform):
    """
    Custom MONAI transform that scales the intensity values so that the most frequent intensity value (peak of histogram) maps to a target value.
    
    Args:
        keys (str or list): Keys of the dictionary to apply the transform to.
        target_value (float): target value for the peak of the histogram.
    """
    def __init__(self, target_value: int = 0):
        super().__init__()
        self.target_value = target_value

    def __call__(self, data):
        
            
        is_tensor = isinstance(data, torch.Tensor)
        data_np = data.cpu().numpy() if is_tensor else data

        # Compute the histogram of the image slice
        hist, bins = np.histogram(data_np.flatten(), bins=100, range=(np.max(data_np)/15.0, np.max(data_np)))

        # Find the value corresponding to the maximum of the histogram
        most_occurred_pixel_value = bins[np.argmax(hist)]

        data_np = data_np/most_occurred_pixel_value*self.target_value # scale it so the peak is always at target_value

        # Put back in original type
        data = torch.from_numpy(data_np) if is_tensor else data_np

        return data
