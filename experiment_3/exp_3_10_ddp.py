import os
import time
import sys


import numpy as np
import csv
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.utils import set_determinism
from monai.data.utils import pad_list_data_collate
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import random

from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler

from torch.nn.parallel import DistributedDataParallel as DDP


import utils.custom_transforms as custom_transforms
import utils.simplex_ddpm as simplex_ddpm



EXPERIMENT_NAME = "exp_3_10"
MODELS_DIR = ROOT_DIR+"AnoDiffExperiments/best_models/experiment_3/"


def setup_ddp(rank, world_size):
    print(f"Running DDP diffusion example on rank {rank}/world_size {world_size}.")
    print(f"Initing to IP {os.environ['MASTER_ADDR']}")
    dist.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(seconds=36000), rank=rank, world_size=world_size
    )  # gloo, nccl
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    return dist, device



def train_epoch(epoch, best_val_epoch_loss, best_val_epoch):
    model.train()

    if ddp_bool:
        # if ddp, distribute data across n gpus
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=DEVICE_TYPE, enabled=True):
            # Generate random noise
            #noise = torch.randn_like(images).to(device)
            noise = generate_simplex_noise(simplexObj, images.shape).to(device)

            # Create timesteps
            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

    if rank==0:
        writer.add_scalar("train_loss", epoch_loss / (step + 1), epoch)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch.to(device)
            with torch.no_grad(), autocast(device_type=DEVICE_TYPE, enabled=True):
                noise = generate_simplex_noise(simplexObj, shape=images.shape).to(device)

                timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item() 

            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

        

        if rank==0:
            
            writer.add_scalar("val_loss", val_epoch_loss / (step + 1), epoch)

            if val_epoch_loss < best_val_epoch_loss:
                best_val_epoch_loss = val_epoch_loss
                best_val_epoch = epoch + 1
                torch.save(
                    model.state_dict(),
                    os.path.join(MODELS_DIR, f"{EXPERIMENT_NAME}_best_model.pth"),
                )
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
    
    return best_val_epoch_loss, best_val_epoch

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

    print_config()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    torch.autograd.set_detect_anomaly(True)

    env_dict = json.load(open(args.environment_file, "r"))
    config_dict = json.load(open(args.config_file, "r"))

    set_determinism(42)



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

    test_reconstruction_csv = os.path.join(ROOT_DIR, "AnoDiffExperiments/data_splits_lists/final_flair_dataset_small/test.csv")
    test_reconstruction_images_path = []

    with open(test_reconstruction_csv, mode='r') as file:
        reader = csv.reader(file)
        for line in tqdm(reader):

            test_reconstruction_images_path.append(ROOT_DIR+line[0])

    #train_datalist = sorted(train_images_path)
    train_datalist = train_images_path

    #val_datalist = sorted(val_images_path)
    val_datalist = val_images_path

    #val_datalist = sorted(val_images_path)
    test_reconstruction_datalist = test_reconstruction_images_path

    #test_unhealthy_datalist = test_unhealthy_images_path

    batch_size = 32
    num_workers = 4

    train_transforms = transforms.Compose(
    [
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.RandAffine(prob=0.2, rotate_range=(0.10, 0.10, 0.10)),#+- 0.15 radians for each axis
        transforms.NormalizeIntensity(),
        transforms.ScaleIntensity(),
        custom_transforms.Get2DSliceWithRandomOffset(axis=2, fixed_offset=0, range_offset=10),
        transforms.RandScaleCrop(roi_scale=0.9, max_roi_scale=1.1, random_size=True),
        transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandScaleIntensity(factors=0.15),
        transforms.RandFlip(prob=0.5, spatial_axis=0),
        SetBackgroundToZero()
    ]
    )
    train_ds = CacheDataset(data=train_datalist, transform=train_transforms) #TODO datalist[:32]
    train_loader = DataLoader(
        #collate_fn=pad_list_data_collate: any tensors are centrally padded to match the shape of the biggest tensor in each dimension
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True, collate_fn=pad_list_data_collate
    )


    val_transforms = transforms.Compose(
        [
            transforms.LoadImage(),
            transforms.EnsureChannelFirst(),
            transforms.NormalizeIntensity(),
            transforms.ScaleIntensity(),
            custom_transforms.Get2DSlice(axis=2),
            transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
            SetBackgroundToZero()
        ]
    )
    val_ds = CacheDataset(data=val_datalist, transform=val_transforms)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True
    )


    test_reconstruction_transforms = transforms.Compose(
        [
            transforms.LoadImage(),
            transforms.EnsureChannelFirst(),
            transforms.NormalizeIntensity(),
            transforms.ScaleIntensity(),
            custom_transforms.Get2DSlice(axis=2),
            transforms.ResizeWithPadOrCrop(spatial_size=(IMAGE_SIZE, IMAGE_SIZE)),
            SetBackgroundToZero()
        ]
    )
    test_reconstruction_ds = CacheDataset(data=test_reconstruction_datalist, transform=test_reconstruction_transforms)
    test_reconstruction_loader = DataLoader(
        test_reconstruction_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True
    )

    if ddp_bool:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, num_replicas=world_size, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(not ddp_bool), num_workers=0, pin_memory=False, sampler=train_sampler
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, sampler=val_sampler
    )


    device = torch.device(DEVICE_TYPE)
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

    optimizer = torch.optim.Adam(params=model.parameters(), lr=2.5e-5)

    inferer = DiffusionInferer(scheduler)

    if ddp_bool:
        # When using DDP, BatchNorm needs to be converted to SyncBatchNorm.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[device], output_device=rank, find_unused_parameters=True)
    
        print("STARTING NEW TRAINING")
    os.makedirs(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{EXPERIMENT_NAME}", exist_ok=True)
    writer = SummaryWriter(ROOT_DIR+f"AnoDiffExperiments/tensorboard/{EXPERIMENT_NAME}")

    max_epochs = 20000
    val_interval = 4

    best_val_epoch_loss = np.inf
    best_val_epoch = 0

    scaler = GradScaler(DEVICE_TYPE)
    total_start = time.time()


    for epoch in range(max_epochs):
        best_val_epoch_loss, best_val_epoch = train_epoch(epoch, best_val_epoch_loss, best_val_epoch)
    

if __name__ == "__main__":
    main()