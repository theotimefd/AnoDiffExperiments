import argparse
import json
from pathlib import Path
import os
import time
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Tuple
import sys
sys.path.append("../..")

import numpy as np
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader, ThreadDataLoader
from monai.data.utils import pad_list_data_collate
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random

from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import utils.custom_transforms as custom_transforms
from utils.utils import *
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm




def compute_loss_simplex(images, simplexObj, model, inferer, num_timesteps, device, return_pred=False):
    with autocast("cuda", enabled=True):
        # Generate random noise
 
        noise = simplex_ddpm.generate_simplex_noise(simplexObj, images.shape, normalize=False).to(device, non_blocking=True) #TODO: check non_blocking (21/09/2025)
 
        # Create timesteps
        timesteps = torch.randint(0, num_timesteps, (images.shape[0],), device=images.device).long()

        # Get model prediction
        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())
        if return_pred:
            return loss, noise_pred, noise
        return loss

def compute_loss_gaussian(images, model, inferer, num_timesteps, device, return_pred=False):
    with autocast("cuda", enabled=True):
        # Generate random noise
        noise = torch.randn_like(images).to(device, non_blocking=True) #TODO: check non_blocking (21/09/2025)

        # Create timesteps
        timesteps = torch.randint(0, num_timesteps, (images.shape[0],), device=images.device).long()

        # Get model prediction
        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())
        if return_pred:
            return loss, noise_pred, noise
        return loss



def _diffusion_step(images, noise_type, simplexObj, model, inferer, num_timesteps, device, return_pred):
    if noise_type == "simplex":
        return compute_loss_simplex(images, simplexObj, model, inferer, num_timesteps, device, return_pred)
    return compute_loss_gaussian(images, model, inferer, num_timesteps, device, return_pred)


def _generate_patch_slices(spatial_shape: Sequence[int], patch_size: Sequence[int], overlap: Sequence[int]):


    ranges: List[List[int]] = []

    for dim, size, ov in zip(spatial_shape, patch_size, overlap):
        step = max(size - ov, 1)

        if dim <= size:
            coords = [0]
        else:
            coords = list(range(0, max(dim - size, 0) + 1, step))
            if coords[-1] != dim - size:
                coords.append(dim - size)
        ranges.append(coords)
        
    for h in ranges[0]:
        for w in ranges[1]:
            for d in ranges[2]:
                yield (slice(h, h + patch_size[0]), slice(w, w + patch_size[1]), slice(d, d + patch_size[2]))


def _run_patchwise_inference(
    volume: torch.Tensor,
    patch_size: Sequence[int],
    overlap: Sequence[int],
    patch_batch_size: int,
    noise_type: str,
    simplexObj,
    model,
    inferer,
    num_timesteps: int,
    device,
    collect_output: bool = False,
):
    aggregator_pred = torch.zeros_like(volume, dtype=torch.float32)
    aggregator_target = torch.zeros_like(volume, dtype=torch.float32)
    counts = torch.zeros_like(volume, dtype=torch.float32)
    patch_queue: List[torch.Tensor] = []
    slice_queue: List[Tuple[slice, slice, slice]] = []
    total_patch_loss = 0.0
    total_patches = 0

    def _flush_queue():
        nonlocal total_patch_loss, total_patches

        if not patch_queue:
            return
        
        batch_tensor = torch.cat(patch_queue, dim=0) # transforms all the patches into a single batch tensor

        loss, preds, targets = _diffusion_step(
            batch_tensor, noise_type, simplexObj, model, inferer, num_timesteps, device, return_pred=True
        )

        patch_count = batch_tensor.shape[0]
        total_patch_loss += loss.item() * patch_count
        total_patches += patch_count

        for idx, patch_slices in enumerate(slice_queue):
            target_slice = (slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])
            
            aggregator_pred[target_slice] += preds[idx].unsqueeze(0).float() # puts the predicted patch back to its original location in the volume using the slices
            aggregator_target[target_slice] += targets[idx].unsqueeze(0).float() # puts the target patch back to its original location in the volume using the slices
            counts[target_slice] += 1.0 # counts how many times a voxel has been predicted (for overlapping patches)


        patch_queue.clear()
        slice_queue.clear()

    for patch_slices in _generate_patch_slices(volume.shape[-3:], patch_size, overlap): # goes through the slices that define each patch

        patch = volume[(slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])] # extracts the patch using the slices

        patch_queue.append(patch) # patch_queue stores all the patches for the current volume batch
        slice_queue.append(patch_slices)

        if len(patch_queue) >= patch_batch_size: # makes sure there aren't too many patches at one time (memory issues)
            _flush_queue() # does the inference and computes loss

    _flush_queue()

    counts = torch.clamp(counts, min=1.0)
    stitched_pred = aggregator_pred / counts # counts is a tensor that stores how  many times there is an overlap, per voxel -> divide by count 
    stitched_target = aggregator_target / counts
    volume_loss = F.mse_loss(stitched_pred.float(), stitched_target.float()).item()

    if collect_output:
        return volume_loss, stitched_pred, stitched_target
    return volume_loss, None, None

def _run_patchwise_test(
    volume: torch.Tensor,
    patch_size: Sequence[int],
    overlap: Sequence[int],
    patch_batch_size: int,
    noise_type: str,
    simplexObj,
    model,
    inferer,
    num_timesteps: int,
    device,
    collect_output: bool = False,
):
    aggregator_pred = torch.zeros_like(volume, dtype=torch.float32)
    aggregator_target = torch.zeros_like(volume, dtype=torch.float32)
    counts = torch.zeros_like(volume, dtype=torch.float32)
    patch_queue: List[torch.Tensor] = []
    slice_queue: List[Tuple[slice, slice, slice]] = []
    total_patch_loss = 0.0
    total_patches = 0
    """
    Same as _run_patchwise_inference but goes all the way to the fully denoised image
    """

    def _flush_queue():
        nonlocal total_patch_loss, total_patches

        if not patch_queue:
            return
        
        batch_tensor = torch.cat(patch_queue, dim=0) # transforms all the patches into a single batch tensor

        for t in range(num_timesteps, 0, -1): # TODO goes from timesteps to 0
            loss, batch_tensor, targets = _diffusion_step(
                batch_tensor, noise_type, simplexObj, model, inferer, t, device, return_pred=True
            )


        preds = batch_tensor
        
        patch_count = batch_tensor.shape[0]
        total_patch_loss += loss.item() * patch_count
        total_patches += patch_count

        for idx, patch_slices in enumerate(slice_queue):
            target_slice = (slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])
            
            aggregator_pred[target_slice] += preds[idx].unsqueeze(0).float() # puts the predicted patch back to its original location in the volume using the slices
            aggregator_target[target_slice] += targets[idx].unsqueeze(0).float() # puts the target patch back to its original location in the volume using the slices
            counts[target_slice] += 1.0 # counts how many times a voxel has been predicted (for overlapping patches)


        patch_queue.clear()
        slice_queue.clear()

    for patch_slices in _generate_patch_slices(volume.shape[-3:], patch_size, overlap): # goes through the slices that define each patch

        patch = volume[(slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])] # extracts the patch using the slices

        patch_queue.append(patch) # patch_queue stores all the patches for the current volume batch
        slice_queue.append(patch_slices)

        if len(patch_queue) >= patch_batch_size: # makes sure there aren't too many patches at one time (memory issues)
            _flush_queue() # does the inference and computes loss

    _flush_queue()

    counts = torch.clamp(counts, min=1.0)
    stitched_pred = aggregator_pred / counts # counts is a tensor that stores how  many times there is an overlap, per voxel -> divide by count 
    stitched_target = aggregator_target / counts
    volume_loss = F.mse_loss(stitched_pred.float(), stitched_target.float()).item()

    if collect_output:
        return volume_loss, stitched_pred, stitched_target
    return volume_loss, None, None


def _validate_batch_with_patches(
    images: torch.Tensor,
    patch_size: Sequence[int],
    overlap: Sequence[int],
    patch_batch_size: int,
    noise_type: str,
    simplexObj,
    model,
    inferer,
    num_timesteps: int,
    device,
    collect_output: bool,
):
    batch_loss = 0.0
    volumes = images.shape[0]
    for idx in range(volumes):
        volume = images[idx : idx + 1]
        vol_loss, stitched_pred, stitched_target = _run_patchwise_inference(
            volume, patch_size, overlap, patch_batch_size, noise_type, simplexObj, model, inferer, num_timesteps, device, collect_output=collect_output
        )
        batch_loss += vol_loss
    if collect_output:
        return batch_loss, volumes, stitched_pred
    else:
        return batch_loss, volumes

@torch.no_grad()
def my_sample_3d(device, model, image, noise_type, infer_scheduler, timesteps, return_intermediates=False):
    
    simplexObj = simplex.Simplex_CLASS()

    if noise_type == "simplex":
        noise = simplex_ddpm.generate_simplex_noise(simplexObj, image.shape, normalize=False).to(device)
    if noise_type == "gaussian":
        noise = torch.randn(image.shape).to(device)
    

    if timesteps >= infer_scheduler.num_train_timesteps:
        tprint(f"{timesteps} is too high. Setting to {infer_scheduler.num_train_timesteps-1}")

    timesteps_list = torch.Tensor([timesteps for a in range(image.shape[0])]).to(image.device).long()

    image = infer_scheduler.add_noise(image, noise, timesteps_list).to(device) #TODO


    intermediates = []
    intermediates_step = 20

            
    for t in range(timesteps, 0, -1): # va de timesteps à 0
        
        model_output = model(
            image, timesteps=torch.Tensor((t,)).to(device), context=None
        )
        
        image, _ = infer_scheduler.step(model_output, t, image)
    
        if (t== timesteps-1 or t%intermediates_step == 0) and return_intermediates:
            intermediates.append(image)

    if return_intermediates:
        return image, intermediates
    else:
        return image