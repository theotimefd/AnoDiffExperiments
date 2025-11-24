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
from utils.utils import define_instance
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion training on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device

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

def _wrap_with_patch_crop(transform, patch_size: Optional[Tuple[int, int, int]]):
    if patch_size is None:
        return transform
    patch_crop = transforms.RandSpatialCrop(roi_size=patch_size, random_size=False)
    if transform is None:
        return patch_crop
    return transforms.Compose([transform, patch_crop])


def _diffusion_step(images, noise_type, simplexObj, model, inferer, num_timesteps, normalize, device, return_pred=False):
    if noise_type == "simplex":
        return compute_loss_simplex(images, simplexObj, model, inferer, num_timesteps, normalize, device, return_pred=return_pred)
    return compute_loss_gaussian(images, model, inferer, num_timesteps, device, return_pred=return_pred)


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
    normalize: bool,
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
        batch_tensor = torch.cat(patch_queue, dim=0)
        loss, preds, targets = _diffusion_step(
            batch_tensor, noise_type, simplexObj, model, inferer, num_timesteps, normalize, device, return_pred=True
        )
        patch_count = batch_tensor.shape[0]
        total_patch_loss += loss.item() * patch_count
        total_patches += patch_count
        for idx, patch_slices in enumerate(slice_queue):
            target_slice = (slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])
            aggregator_pred[target_slice] += preds[idx].unsqueeze(0).float()
            aggregator_target[target_slice] += targets[idx].unsqueeze(0).float()
            counts[target_slice] += 1.0
        patch_queue.clear()
        slice_queue.clear()

    for patch_slices in _generate_patch_slices(volume.shape[-3:], patch_size, overlap):
        patch = volume[(slice(None), slice(None), patch_slices[0], patch_slices[1], patch_slices[2])]
        patch_queue.append(patch)
        slice_queue.append(patch_slices)
        if len(patch_queue) >= patch_batch_size:
            _flush_queue()

    _flush_queue()

    counts = torch.clamp(counts, min=1.0)
    stitched_pred = aggregator_pred / counts
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
    normalize: bool,
    device,
):
    batch_loss = 0.0
    volumes = images.shape[0]
    for idx in range(volumes):
        volume = images[idx : idx + 1]
        vol_loss, _, _ = _run_patchwise_inference(
            volume, patch_size, overlap, patch_batch_size, noise_type, simplexObj, model, inferer, num_timesteps, normalize, device
        )
        batch_loss += vol_loss
    return batch_loss, volumes


def launch_train_patch(args):

    ROOT_DIR = args.root_dir
    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    os.makedirs(MODELS_DIR, exist_ok=True)

    
    train_patch_size = args.dataset["patch_size"]
    infer_patch_size = args.dataset["patch_size"]
    patch_overlap = args.dataset["patch_overlap"]

    patch_infer_batch_size = args.dataset["batch_size"]

    image_size_value = args.image_size
    base_edge = image_size_value[0]

    fallback_patch_size = None
    if base_edge is not None and base_edge > 0:
        tentative_edge = max(base_edge // 2, 16)
        tentative_edge = min(tentative_edge, base_edge)
        fallback_patch_size = (tentative_edge, tentative_edge, tentative_edge)


    ddp_bool = args.gpus > 1  # whether to use distributed data parallel

    if ddp_bool:
        rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist, device = setup_ddp(rank, world_size)
    else:
        rank = 0
        world_size = 1
        device = 0

    torch.cuda.set_device(device)
    print(f"Using {device}")

    
    if rank == 0:
        print(
            "Patch-based validation enabled with patch size "
            f"{infer_patch_size} and overlap {patch_overlap} (patch batch size={patch_infer_batch_size})."
        )

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)


    train_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset['name']}/train.csv")
    train_images_path = []

    with open(train_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):
            #print(line)
            train_images_path.append(ROOT_DIR+line[0])

    val_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset['name']}/val.csv")
    val_images_path = []

    with open(val_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):

            val_images_path.append(ROOT_DIR+line[0])

    #train_datalist = sorted(train_images_path)
    train_datalist = train_images_path

    #val_datalist = sorted(val_images_path)
    val_datalist = val_images_path

    #test_unhealthy_datalist = test_unhealthy_images_path

    batch_size = args.dataset["batch_size"]
    num_workers = args.dataset["num_workers"]



    train_transforms = define_instance(args, "train_transforms")
    train_transforms = _wrap_with_patch_crop(train_transforms, train_patch_size)
    train_ds = CacheDataset(data=train_datalist[:batch_size], transform=train_transforms) #TODO: train_datalist[:batch_size]


    val_transforms = define_instance(args, "val_transforms")
    val_ds = CacheDataset(data=val_datalist[:batch_size], transform=val_transforms) #TODO: val_datalist[:batch_size]
    

    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=num_workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader( # smaller batch size for validation since we are validating on full volumes
        val_ds, batch_size=5, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=val_sampler
    )

    model = define_instance(args, "network_def").to(device)

    simplexObj = None
    if args.noise["type"] == "simplex":
        simplexObj = simplex.Simplex_CLASS()
        scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"], octaves=args.noise["simplex_octaves"], persistence=args.noise["simplex_persistence"], frequency=args.noise["simplex_frequency"], normalize=args.noise["normalize"])

    elif args.noise["type"] == "gaussian":
        scheduler = DDPMScheduler(num_train_timesteps=args.noise["num_timesteps_full_noise"], schedule=args.noise["schedule"])

    num_diffusion_steps = int(args.noise["noise_rate_train_and_infer"] * args.noise["num_timesteps_full_noise"])

    if args.diffusion_train["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.diffusion_train["optimizer"]["lr"] * world_size)
    
    if args.diffusion_train["lr_scheduler"]!= "none":
        
        if args.diffusion_train["lr_scheduler"] == "MultiStepLR":
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=args.diffusion_train["lr_scheduler_milestones"],
            gamma=0.1)



    inferer = DiffusionInferer(scheduler)

    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device], output_device=rank, find_unused_parameters=False)
    
    if rank==0:
        os.makedirs(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{SUB_EXPERIMENT_NAME}", exist_ok=True)
        writer = SummaryWriter(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{SUB_EXPERIMENT_NAME}")

    max_epochs = args.diffusion_train["max_epochs"]
    val_interval = args.diffusion_train["val_interval"]

    best_val_epoch_loss = np.inf
    best_val_epoch = 0

    scaler = GradScaler("cuda")


    for epoch in range(max_epochs):
        model.train()
        if rank==0 and args.diffusion_train["lr_scheduler"] != "none":
            lr_scheduler.step()
        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        epoch_loss = 0
        #progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        #progress_bar.set_description(f"Epoch {epoch}")

        #for step, batch in progress_bar:
        for step, batch in enumerate(train_loader):
            images = batch.to(device, non_blocking=True) #TODO: check non_blocking (21/09/2025)
            optimizer.zero_grad(set_to_none=True)

            if args.noise["type"] == "simplex":
                loss = compute_loss_simplex(images, simplexObj, model, inferer, num_diffusion_steps, device=device)
            elif args.noise["type"] == "gaussian":
                loss = compute_loss_gaussian(images, model, inferer, num_diffusion_steps, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            

            #progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        if rank==0:
            print(f"epoch_loss {epoch_loss}")
            print(f"step {step}")
            print(f"epoch {epoch}")
            writer.add_scalar("train_loss", epoch_loss / (step + 1), epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()

            val_epoch_loss = 0.0
            total_val_volumes = 0
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    images = batch.to(device)
                    images = images[..., args.slice_indexes_start:args.slice_indexes_end]
                    batch_loss, processed = _validate_batch_with_patches(
                        images,
                        infer_patch_size,
                        patch_overlap,
                        patch_infer_batch_size,
                        args.noise["type"],
                        simplexObj,
                        model,
                        inferer,
                        num_diffusion_steps,
                        args.noise.get("normalize", False),
                        device,
                    )
                    val_epoch_loss += batch_loss
                    total_val_volumes += processed
            avg_val_loss = val_epoch_loss / max(total_val_volumes, 1)

            if rank==0:
                writer.add_scalar("val_loss", avg_val_loss, epoch)

                if avg_val_loss < best_val_epoch_loss:
                    best_val_epoch_loss = avg_val_loss
                    best_val_epoch = epoch + 1

                    if ddp_bool:
                        torch.save(model.module.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))

                    print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current val loss: {avg_val_loss:.4f}"
                        f"\nbest val loss: {best_val_epoch_loss:.4f}"
                        f" at epoch: {best_val_epoch}"
                    )
                    writer.add_scalar("best_val_loss", best_val_epoch_loss, best_val_epoch)

                    # can't visualize an inference image since we don't train from pure noise here
                    #noise = generate_simplex_noise(simplexObj, shape=(1,1,IMAGE_SIZE, IMAGE_SIZE)).to(device)
                    #noise = noise.to(device)
                    #scheduler.set_timesteps(num_inference_steps=1000)
                    #with autocast(device_type=DEVICE_TYPE, enabled=True):
                    #    image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)
                    #writer.add_image("sampled_image", image[0, 0].cpu().numpy(), global_step=epoch, dataformats="HW")
                    #plt.figure(figsize=(2, 2))
                    #plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
                    #plt.tight_layout()
                    #plt.axis("off")
                    #plt.show()
        
    print(f"Training complete, best val loss: {best_val_epoch_loss:.6f} at epoch {best_val_epoch}")
    
