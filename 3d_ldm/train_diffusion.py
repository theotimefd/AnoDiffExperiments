import argparse
import json
from pathlib import Path
import os
import time
from datetime import timedelta
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
from monai.utils import first, set_determinism

from monai.inferers import LatentDiffusionInferer
from monai.networks.nets import DiffusionModelUNet, AutoencoderKL
from monai.networks.schedulers import DDPMScheduler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import utils.custom_transforms as custom_transforms
from utils.utils import *
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandAffined,
    RandScaleCropd,
    ResizeWithPadOrCropd,
    ScaleIntensityRangeD,
    RandFlipd,
    Lambdad,
)


def setup_ddp(rank, world_size):
    tprint(f"Running DDP LDM training on rank {rank}/world_size {world_size}.")
    tprint(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


def launch_train_diffusion(args):

    # ----------------- SETUP ----------------- #
    ROOT_DIR = args.root_dir
    EXPERIMENT_NAME = args.experiment_name
    SUB_EXPERIMENT_NAME = args.sub_experiment_name
    MODELS_DIR = ROOT_DIR+f"AnoDiffExperiments/{EXPERIMENT_NAME}/{SUB_EXPERIMENT_NAME}/models/"
    os.makedirs(MODELS_DIR, exist_ok=True)

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
    tprint(f"Using {device}")

    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(torch.get_num_threads()) 
    torch.autograd.set_detect_anomaly(False)


    # ----------------- DATASET AND DATALOADER ----------------- #
    train_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset["name"]}/train.csv")
    train_images_path = []

    with open(train_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):
            #tprint(line)
            train_images_path.append(ROOT_DIR+line[0])

    val_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset["name"]}/val.csv")
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

    batch_size = args.autoencoder_train["batch_size"]
    num_workers = args.autoencoder_train["num_workers"]



    train_transforms = define_instance(args, "train_transforms")
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms)


    val_transforms = define_instance(args, "val_transforms")
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
    

    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=num_workers, pin_memory=True, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, sampler=val_sampler
    )

    # ----------------- MODEL, OPTIMIZER, LOSS, LR SCHEDULER ----------------- #
    # Define Autoencoder KL network and diffusion model
    # Load Autoencoder KL network
    autoencoder = define_instance(args, "autoencoder_def").to(device)

    trained_g_path = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_autoencoder.pt")

    map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
    autoencoder.load_state_dict(torch.load(trained_g_path, map_location=map_location, weights_only=True))
    tprint(f"Rank {rank}: Load trained autoencoder from {trained_g_path}")

    # Compute Scaling factor
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM,
    # if the standard deviation of the latent space distribution drifts too much from that of a Gaussian.
    # For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one,
    # and the results will not differ from those obtained when it is not used._

    if rank==0:
        os.makedirs(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{SUB_EXPERIMENT_NAME}", exist_ok=True)
        writer = SummaryWriter(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{SUB_EXPERIMENT_NAME}")


    with torch.no_grad():
        with autocast("cuda", enabled=True):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data.to(device))
            if rank == 0:
                tprint(f"Latent feature shape {z.shape}")
                for axis in range(3):
                    writer.add_image(
                        "train_img_" + str(axis),
                        visualize_one_slice_in_3d_image(check_data[0, 0, ...], axis).transpose([2, 1, 0]),
                        1,
                    )
                tprint(f"Scaling factor set to {1/torch.std(z)}")
    scale_factor = 1 / torch.std(z)
    tprint(f"Rank {rank}: local scale_factor: {scale_factor}")
    if ddp_bool:
        dist.barrier()
        dist.all_reduce(scale_factor, op=torch.distributed.ReduceOp.AVG)
    tprint(f"Rank {rank}: final scale_factor -> {scale_factor}")

    # Define Diffusion Model
    unet = define_instance(args, "diffusion_network_def").to(device)

    trained_diffusion_path = os.path.join(MODELS_DIR, "diffusion_unet.pt")
    trained_diffusion_path_last = os.path.join(MODELS_DIR, "diffusion_unet_last.pt")

    if args.resume_ckpt:
        map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
        try:
            unet.load_state_dict(torch.load(trained_diffusion_path, map_location=map_location, weights_only=True))
            tprint(f"Rank {rank}: Load trained diffusion model from", trained_diffusion_path)
        except:
            tprint(f"Rank {rank}: Train diffusion model from scratch.")

    scheduler = DDPMScheduler(
        num_train_timesteps=args.noise["num_timesteps_full_noise"],
        schedule="scaled_linear_beta",
        beta_start=args.noise["beta_start"],
        beta_end=args.noise["beta_end"],
    )

    if ddp_bool:
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=True)
        unet = DDP(unet, device_ids=[device], output_device=rank, find_unused_parameters=True)

    # We define the inferer using the scale factor:
    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

    # Step 3: training config
    optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=args.diffusion_train["optimizer"]["lr"] * world_size)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[100, 1000], gamma=0.1)

    # Step 4: training
    max_epochs = args.diffusion_train["max_epochs"]
    val_interval = args.diffusion_train["val_interval"]
    autoencoder.eval()
    scaler = GradScaler("cuda")
    total_step = 0
    best_val_recon_epoch_loss = 100.0

    for epoch in range(max_epochs):
        unet.train()
        epoch_loss = 0
        lr_scheduler.step()
        if ddp_bool:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_loader):
            images = batch.to(device)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=True):
                # Generate random noise
                noise_shape = [images.shape[0]] + list(z.shape[1:])
                noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                #TODO: implement SimplexDDPMScheduler for the ldm
                # Create timesteps
                timesteps = torch.randint(
                    0, int(args.noise["noise_rate_train_and_infer"]*args.noise["num_timesteps_full_noise"]), (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                if ddp_bool:
                    inferer_autoencoder = autoencoder.module
                else:
                    inferer_autoencoder = autoencoder
                noise_pred = inferer(
                    inputs=images,
                    autoencoder_model=inferer_autoencoder,
                    diffusion_model=unet,
                    noise=noise,
                    timesteps=timesteps,
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                writer.add_scalar("train_diffusion_loss_iter", loss, total_step)

        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            unet.eval()
            val_recon_epoch_loss = 0
            with torch.no_grad():
                with autocast("cuda", enabled=True):
                    # compute val loss
                    for step, batch in enumerate(val_loader):
                        images = batch.to(device)
                        noise_shape = [images.shape[0]] + list(z.shape[1:])
                        noise = torch.randn(noise_shape, dtype=images.dtype).to(device)

                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        # Get model prediction
                        if ddp_bool:
                            inferer_autoencoder = autoencoder.module
                        else:
                            inferer_autoencoder = autoencoder
                        noise_pred = inferer(
                            inputs=images,
                            autoencoder_model=inferer_autoencoder,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                        )
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())
                        val_recon_epoch_loss += val_loss
                    val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)

                    if ddp_bool:
                        dist.barrier()
                        dist.all_reduce(val_recon_epoch_loss, op=torch.distributed.ReduceOp.AVG)

                    val_recon_epoch_loss = val_recon_epoch_loss.item()

                    # write val loss and save best model
                    if rank == 0:
                        writer.add_scalar("val_diffusion_loss", val_recon_epoch_loss, epoch)
                        tprint(f"Epoch {epoch} val_diffusion_loss: {val_recon_epoch_loss}")
                        # save last model
                        if ddp_bool:
                            torch.save(unet.module.state_dict(), trained_diffusion_path_last)
                        else:
                            torch.save(unet.state_dict(), trained_diffusion_path_last)

                        # save best model
                        if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                            best_val_recon_epoch_loss = val_recon_epoch_loss
                            if ddp_bool:
                                torch.save(unet.module.state_dict(), trained_diffusion_path)
                            else:
                                torch.save(unet.state_dict(), trained_diffusion_path)
                            tprint("Got best val noise pred loss.")
                            tprint(f"Save trained latent diffusion model to {trained_diffusion_path}")

                        # visualize synthesized image
                        if (epoch) % (50 * val_interval) == 0:  # time cost of synthesizing images is large
                            synthetic_images = inferer.sample(
                                input_noise=noise[0:1, ...],
                                autoencoder_model=inferer_autoencoder,
                                diffusion_model=unet,
                                scheduler=scheduler,
                            )
                            for axis in range(3):
                                writer.add_image(
                                    "val_diff_synimg_" + str(axis),
                                    visualize_one_slice_in_3d_image(synthetic_images[0, 0, ...], axis).transpose(
                                        [2, 1, 0]
                                    ),
                                    epoch,
                                )
