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

from monai.inferers import DiffusionInferer
from monai.networks.nets import PatchDiscriminator
from monai.networks.schedulers import DDPMScheduler
from monai.losses import PatchAdversarialLoss, PerceptualLoss

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import L1Loss, MSELoss
import torch.distributed as dist

import utils.custom_transforms as custom_transforms
from utils.utils import define_instance, visualize_one_slice_in_3d_image
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm


def setup_ddp(rank, world_size):
    print(f"Running DDP LDM training on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device



def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(
        z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1,
        dim=list(range(1, len(z_sigma.shape))),
    )
    return torch.sum(kl_loss) / kl_loss.shape[0]


def launch_train_autoencoder(args):

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
    print(f"Using {device}")

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

    # Step 2: Define Autoencoder KL network and discriminator
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    discriminator_norm = "INSTANCE"
    discriminator = PatchDiscriminator(
        spatial_dims=args.spatial_dims,
        num_layers_d=3,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm=discriminator_norm,
    ).to(device)
    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator)

    trained_g_path = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_autoencoder.pt")
    trained_d_path = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_discriminator.pt")
    trained_g_path_last = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_autoencoder_last.pt")
    trained_d_path_last = os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_discriminator_last.pt")


    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        autoencoder = DDP(autoencoder, device_ids=[device], output_device=rank, find_unused_parameters=False)
        discriminator = DDP(discriminator, device_ids=[device], output_device=rank, find_unused_parameters=False)
    

    # Step 3: training config
    if "recon_loss" in args.autoencoder_train and args.autoencoder_train["recon_loss"] == "l2":
        intensity_loss = MSELoss()
        if rank == 0:
            print("Use l2 loss")
    else:
        intensity_loss = L1Loss()
        if rank == 0:
            print("Use l1 loss")
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)

    adv_weight = 0.01
    perceptual_weight = args.autoencoder_train["perceptual_weight"]
    # kl_weight: important hyper-parameter. TODO
    #     If too large, decoder cannot recon good results from latent space.
    #     If too small, latent space will not be regularized enough for the diffusion model
    kl_weight = args.autoencoder_train["kl_weight"]

    if args.autoencoder_train["optimizer_autoencoder"]["type"] == "Adam":
        optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=args.autoencoder_train["optimizer_autoencoder"]["lr"] * world_size)
    
    if args.autoencoder_train["optimizer_discriminator"]["type"] == "Adam":
        optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=args.autoencoder_train["optimizer_discriminator"]["lr"] * world_size)
    

    if args.autoencoder_train["optimizer_autoencoder"]["lr_scheduler"] == "MultiStepLR":
        lr_scheduler_autoencoder = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_g,
        milestones=args.autoencoder_train["optimizer_autoencoder"]["lr_scheduler_milestones"],
        gamma=0.1)
    
    if args.autoencoder_train["optimizer_discriminator"]["lr_scheduler"] == "MultiStepLR":
        lr_scheduler_discriminator = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_d,
        milestones=args.autoencoder_train["optimizer_discriminator"]["lr_scheduler_milestones"],
        gamma=0.1)

    if rank==0:
        os.makedirs(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{SUB_EXPERIMENT_NAME}", exist_ok=True)
        writer = SummaryWriter(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{SUB_EXPERIMENT_NAME}")

    max_epochs = args.autoencoder_train["max_epochs"]
    val_interval = args.autoencoder_train["val_interval"]
    autoencoder_warm_up_n_epochs = 5
    total_step = 0
    best_val_recon_epoch_loss = np.inf
    best_val_epoch_loss = np.inf
    best_val_epoch = 0

    scaler = GradScaler("cuda")


    for epoch in range(max_epochs):
        autoencoder.train()
        discriminator.train()

        if rank==0 and args.autoencoder_train["optimizer_autoencoder"]["lr_scheduler"] != "none":
            lr_scheduler_autoencoder.step()
        if rank==0 and args.autoencoder_train["discriminator_autoencoder"]["lr_scheduler"] != "none":
            lr_scheduler_discriminator.step()

        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

            
        for step, batch in enumerate(train_loader):
            images = batch.to(device)

            # train Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)

            recons_loss = intensity_loss(reconstruction, images)
            kl_loss = KL_loss(z_mu, z_sigma)
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g = loss_g + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                # train Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            # write train loss for each batch into tensorboard
            if rank == 0:
                total_step += 1
                writer.add_scalar("train_recon_loss_iter", recons_loss, total_step)
                writer.add_scalar("train_kl_loss_iter", kl_loss, total_step)
                writer.add_scalar("train_perceptual_loss_iter", p_loss, total_step)
                if epoch > autoencoder_warm_up_n_epochs:
                    writer.add_scalar("train_adv_loss_iter", generator_loss, total_step)
                    writer.add_scalar("train_fake_loss_iter", loss_d_fake, total_step)
                    writer.add_scalar("train_real_loss_iter", loss_d_real, total_step)

        # validation
        if epoch % val_interval == 0:
            autoencoder.eval()
            val_recon_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch.to(device)  # choose only one of Brats channels
                with torch.no_grad():
                    reconstruction, z_mu, z_sigma = autoencoder(images)
                    recons_loss = intensity_loss(
                        reconstruction.float(), images.float()
                    ) + perceptual_weight * loss_perceptual(reconstruction.float(), images.float())

                val_recon_epoch_loss += recons_loss.item()

            val_recon_epoch_loss = val_recon_epoch_loss / (step + 1)
            if rank == 0:
                # save last model
                print(f"Epoch {epoch} val_recon_loss: {val_recon_epoch_loss}")
                if ddp_bool:
                    torch.save(autoencoder.module.state_dict(), trained_g_path_last)
                    torch.save(discriminator.module.state_dict(), trained_d_path_last)
                else:
                    torch.save(autoencoder.state_dict(), trained_g_path_last)
                    torch.save(discriminator.state_dict(), trained_d_path_last)
                # save best model
                if val_recon_epoch_loss < best_val_recon_epoch_loss and rank == 0:
                    best_val_recon_epoch_loss = val_recon_epoch_loss
                    if ddp_bool:
                        torch.save(autoencoder.module.state_dict(), trained_g_path)
                        torch.save(discriminator.module.state_dict(), trained_d_path)
                    else:
                        torch.save(autoencoder.state_dict(), trained_g_path)
                        torch.save(discriminator.state_dict(), trained_d_path)
                    
                    print("Got best val recon loss.")
                    print("Save trained autoencoder to", trained_g_path)
                    print("Save trained discriminator to", trained_d_path)

                # write val loss for each epoch into tensorboard
                writer.add_scalar("val_recon_loss", val_recon_epoch_loss, epoch)
                for axis in range(3):
                    writer.add_image(
                        "val_img_" + str(axis),
                        visualize_one_slice_in_3d_image(images[0, 0, ...], axis).transpose([2, 1, 0]),
                        epoch,
                    )
                    writer.add_image(
                        "val_recon_" + str(axis),
                        visualize_one_slice_in_3d_image(reconstruction[0, 0, ...], axis).transpose([2, 1, 0]),
                        epoch,
                    )
    