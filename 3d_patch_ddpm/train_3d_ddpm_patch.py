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

from 3d_patch_ddpm.utils_3d_ddpm import *

import utils.custom_transforms as custom_transforms
from utils.utils import *
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm


def setup_ddp(rank, world_size):
    tprint(f"Running DDP diffusion training on rank {rank}/world_size {world_size}.")
    tprint(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device

def launch_train_patch(args):

    ROOT_DIR = args.root_dir
    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    os.makedirs(MODELS_DIR, exist_ok=True)

    
    train_patch_size = args.patch_size
    infer_patch_size = args.patch_size
    patch_overlap = args.dataset["patch_overlap"]

    patch_infer_batch_size = args.dataset["batch_size"]

    ddp_bool = False

    rank = 0
    world_size = 1
    device = 0

    torch.cuda.set_device(device)
    tprint(f"Using {device}")

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads())
    torch.autograd.set_detect_anomaly(False)


    exclude_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset['name']}/exclude.csv")
    exclude_list = []
    if os.path.exists(exclude_csv):
        with open(exclude_csv, mode="r") as file:
            reader = csv.reader(file)
            for line in reader:
                    exclude_list.append(line[0])

    train_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset['name']}/train.csv")
    train_images_path = []

    with open(train_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in reader:
            if line not in exclude_list:
                train_images_path.append(ROOT_DIR+line[0])

    val_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset['name']}/val.csv")
    val_images_path = []

    with open(val_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in reader:
            if line not in exclude_list:
                val_images_path.append(ROOT_DIR+line[0])

    #train_datalist = sorted(train_images_path)
    train_datalist = train_images_path

    #val_datalist = sorted(val_images_path)
    val_datalist = val_images_path

    #test_unhealthy_datalist = test_unhealthy_images_path

    batch_size = args.dataset["batch_size"]
    num_workers = args.dataset["num_workers"]



    train_transforms = define_instance(args, "train_transforms")
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms) #TODO: train_datalist[:batch_size]


    val_transforms = define_instance(args, "val_transforms")
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms) #TODO: val_datalist[:batch_size]
    

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
            tprint(f"epoch_loss {epoch_loss}, epoch {epoch}")
            writer.add_scalar("train_loss", epoch_loss / (step + 1), epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()

            val_epoch_loss = 0.0
            total_val_volumes = 0
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    images = batch.to(device)
                    images = images[..., args.slice_indexes_start:args.slice_indexes_end]
                    batch_loss, processed, stitched_pred = _validate_batch_with_patches(
                        images,
                        infer_patch_size,
                        patch_overlap,
                        patch_infer_batch_size,
                        args.noise["type"],
                        simplexObj,
                        model,
                        inferer,
                        num_diffusion_steps,
                        device,
                        collect_output=True
                    )
                    val_epoch_loss += batch_loss
                    total_val_volumes += processed
            avg_val_loss = val_epoch_loss / max(total_val_volumes, 1)

            print(stitched_pred.shape)
            if rank == 0 and stitched_pred is not None and len(stitched_pred) > 0:
                # Take the first volume from stitched_pred and a middle slice
                sample_volume = stitched_pred[0, 0]*2+0.5 # Shape: [D, H, W]

                writer.add_image(f"stitched_pred_sample D", sample_volume[sample_volume.shape[0]//2,...].cpu().numpy(), epoch, dataformats="HW")
                writer.add_image(f"stitched_pred_sample H", sample_volume[:,sample_volume.shape[1]//2,:].cpu().numpy(), epoch, dataformats="HW")
                writer.add_image(f"stitched_pred_sample W", sample_volume[...,sample_volume.shape[2]//2].cpu().numpy(), epoch, dataformats="HW")

            if rank==0:
                writer.add_scalar("val_loss", avg_val_loss, epoch)

                if avg_val_loss < best_val_epoch_loss:
                    best_val_epoch_loss = avg_val_loss
                    best_val_epoch = epoch + 1

                    if ddp_bool:
                        torch.save(model.module.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))

                    tprint("saved new best metric model")
                    tprint(
                        f"current epoch: {epoch + 1} current val loss: {avg_val_loss:.4f}"
                        f"\nbest val loss: {best_val_epoch_loss:.4f}"
                        f" at epoch: {best_val_epoch}"
                    )
                    writer.add_scalar("best_val_loss", best_val_epoch_loss, best_val_epoch)


        
    tprint(f"Training complete, best val loss: {best_val_epoch_loss:.6f} at epoch {best_val_epoch}")
    
    
