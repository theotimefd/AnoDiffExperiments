import argparse
import os
import time
from datetime import timedelta
import sys
sys.path.append("..")

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
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm



#ROOT_DIR = "/home/fehrdelt/bettik/"
ROOT_DIR = "/bettik/PROJECTS/pr-gin5_aini/fehrdelt/"
EXPERIMENT_NAME = "exp_3_10"
MODELS_DIR = ROOT_DIR+"AnoDiffExperiments/best_models/experiment_3/"

IMAGE_SIZE = 128


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device


def compute_loss(images, simplexObj, model, inferer, num_train_timesteps, device):
    with autocast("cuda", enabled=True):
        # Generate random noise
        #noise = torch.randn_like(images).to(device)
        noise = simplex_ddpm.generate_simplex_noise(simplexObj, images.shape).to(device)

        # Create timesteps
        timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

        # Get model prediction
        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

        loss = F.mse_loss(noise_pred.float(), noise.float())
        return loss

def main():
    
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT_NAME} training script")
    parser.add_argument("-g", "--gpus", default=1, type=int, help="number of gpus per node")
    args = parser.parse_args()

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


    train_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/final_flair_dataset_small/train.csv")
    train_images_path = []

    with open(train_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):
            #print(line)
            train_images_path.append(ROOT_DIR+line[0])

    val_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/final_flair_dataset_small/val.csv")
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

    batch_size = 70 # 32
    num_workers = 16 # 4*num_gpus, 4*world_size

    train_transforms = transforms.Compose(
    [
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.RandAffine(prob=0.2, rotate_range=(0.10, 0.10, 0.10)),#+- 0.15 radians for each axis
        transforms.NormalizeIntensity(),
        transforms.ScaleIntensity(),
        #transforms.EnsureType(device=device, track_meta=False),(didn't work error) # convert the data to Tensor without meta, move to GPU and cache to avoid CPU -> GPU sync in every epoch
        custom_transforms.Get2DSliceWithRandomOffset(axis=2, fixed_offset=0, range_offset=10),
        transforms.RandScaleCrop(roi_scale=0.9, max_roi_scale=1.1, random_size=True),
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandScaleIntensity(factors=0.15),
        transforms.RandFlip(prob=0.5, spatial_axis=0),
        custom_transforms.SetBackgroundToZero()
    ]
    )
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms) #TODO datalist[:32]

    val_transforms = transforms.Compose(
        [
            transforms.LoadImage(),
            transforms.EnsureChannelFirst(),
            transforms.NormalizeIntensity(),
            transforms.ScaleIntensity(),
            custom_transforms.Get2DSlice(axis=2),
            transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
            custom_transforms.SetBackgroundToZero(),
            #transforms.EnsureType(device=device, track_meta=False)
        ]
    )
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
    """
    train_loader = ThreadDataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=0, pin_memory=True, sampler=train_sampler
    )
    val_loader = ThreadDataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=val_sampler
    )
    """


    simplexObj = simplex.Simplex_CLASS()


    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(128, 128, 256, 256),
        attention_levels=(False, True, True, True),
        num_head_channels=(0, 128, 128, 256),
    )
    model.to(device)

    num_train_timesteps = 600
    scheduler = simplex_ddpm.SimplexDDPMScheduler(num_train_timesteps=num_train_timesteps)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5 * world_size)

    inferer = DiffusionInferer(scheduler)

    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device], output_device=rank, find_unused_parameters=False)
    
        print("STARTING NEW TRAINING")
    
    if rank==0:
        os.makedirs(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{EXPERIMENT_NAME}", exist_ok=True)
        writer = SummaryWriter(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{EXPERIMENT_NAME}")

    max_epochs = 20000
    val_interval = 4

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

            loss = compute_loss(images, simplexObj, model, inferer, num_train_timesteps, device)

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
                with torch.no_grad(), autocast("cuda", enabled=True):
                    noise = simplex_ddpm.generate_simplex_noise(simplexObj, shape=images.shape).to(device)

                    timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item() 

                #progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

            

            if rank==0:
                
                writer.add_scalar("val_loss", val_epoch_loss / (step + 1), epoch)

                if val_epoch_loss < best_val_epoch_loss:
                    best_val_epoch_loss = val_epoch_loss
                    best_val_epoch = epoch + 1

                    if ddp_bool:
                        torch.save(model.module.state_dict(), os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_best_model.pth"))
                    else:
                        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_best_model.pth"))

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
    

if __name__ == "__main__":
    main()