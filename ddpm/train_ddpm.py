import argparse
import json
from pathlib import Path
import os
import time
from datetime import timedelta
import sys
sys.path.append("..")
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

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import utils.custom_transforms as custom_transforms
from utils.utils import define_instance
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device

def compute_loss_simplex(images, simplexObj, model, inferer, num_train_timesteps, device):
    with autocast("cuda", enabled=True):
        # Generate random noise
        noise = simplex_ddpm.generate_simplex_noise(simplexObj, images.shape).to(device)

        # Create timesteps
        timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

        # Get model prediction
        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())
        return loss

def compute_loss_gaussian(images, model, inferer, num_train_timesteps, device):
    with autocast("cuda", enabled=True):
        # Generate random noise
        noise = torch.randn_like(images).to(device)

        # Create timesteps
        timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

        # Get model prediction
        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())
        return loss


def launch_train(args):

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


    train_csv = os.path.join(ROOT_DIR, f"AnoDiffExperiments/data_splits_lists/{args.dataset["name"]}/train.csv")
    train_images_path = []

    with open(train_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):
            #print(line)
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

    batch_size = args.dataset["batch_size"]
    num_workers = args.dataset["num_workers"]



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

    model = define_instance(args, "network_def").to(device)

    if args.noise["type"] == "simplex":
        simplexObj = simplex.Simplex_CLASS()
        num_train_timesteps = args.noise["num_train_timesteps"]

        scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=num_train_timesteps)
    elif args.noise["type"] == "gaussian":
        num_train_timesteps = args.noise["num_train_timesteps"]

        scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps,
            beta_start=args.noise["beta_start"],
            beta_end=args.noise["beta_end"],)

    if args.diffusion_train["optimizer"]["type"] == "Adam":
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.diffusion_train["optimizer"]["lr"] * world_size)

    inferer = DiffusionInferer(scheduler)

    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device], output_device=rank, find_unused_parameters=False)
    
        print("STARTING NEW TRAINING")
    
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

        if ddp_bool:
            # if ddp, distribute data across n gpus
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)

        epoch_loss = 0
        #progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        #progress_bar.set_description(f"Epoch {epoch}")

        #for step, batch in progress_bar:
        for step, batch in enumerate(train_loader):
            images = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            if args.noise["type"] == "simplex":
                loss = compute_loss_simplex(images, simplexObj, model, inferer, num_train_timesteps, device)
            elif args.noise["type"] == "gaussian":
                loss = compute_loss_gaussian(images, model, inferer, num_train_timesteps, device)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            

            #progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        if rank==0:
            writer.add_scalar("train_loss", epoch_loss / (step + 1), epoch)

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
                images = batch.to(device)
                
                if args.noise["type"] == "simplex":
                    val_loss = compute_loss_simplex(images, simplexObj, model, inferer, num_train_timesteps, device)
                elif args.noise["type"] == "gaussian":
                    val_loss = compute_loss_gaussian(images, model, inferer, num_train_timesteps, device)
                
                val_epoch_loss += val_loss.item() 

                #progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            
            if rank==0:
                
                writer.add_scalar("val_loss", val_epoch_loss / (step + 1), epoch)

                if val_epoch_loss < best_val_epoch_loss:
                    best_val_epoch_loss = val_epoch_loss
                    best_val_epoch = epoch + 1

                    if ddp_bool:
                        torch.save(model.module.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{SUB_EXPERIMENT_NAME}_best_model.pth"))

                    print("saved new best metric model")
                    print(
                        f"current epoch: {epoch + 1} current val loss: {val_epoch_loss/(step + 1):.4f}"
                        f"\nbest val loss: {best_val_epoch_loss/(step + 1):.4f}"
                        f" at epoch: {best_val_epoch}"
                    )
                    writer.add_scalar("best_val_loss", best_val_epoch_loss/(step + 1), best_val_epoch)

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
        
        print(f"Training complete, best val loss: {best_val_epoch_loss/(step + 1)} at epoch {best_val_epoch}")
    

if __name__ == "__main__":
    main()