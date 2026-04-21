import copy
import sys
sys.path.append("../..")

import numpy as np
import torch
import AnoDDPM.simplex as simplex
import utils.simplex_ddpm as simplex_ddpm
import utils.thor_ddpm as thor_ddpm
from utils.utils import scale_intensity_from_histogram_peak
from utils.utils import dtprint


@torch.no_grad()
def my_sample(args, model, device, image, infer_scheduler, timesteps, return_intermediates=False):
    
    simplexObj = simplex.Simplex_CLASS()

    if args.noise["type"] == "simplex":
        noise = simplex_ddpm.generate_simplex_noise(simplexObj, image.shape, normalize=args.noise["normalize"]).to(device)
    if args.noise["type"] == "gaussian":
        noise = torch.randn(image.shape).to(device)
    

    if timesteps >= infer_scheduler.num_train_timesteps:
        dtprint(timesteps, "is too high. Setting to", infer_scheduler.num_train_timesteps-1)

    timesteps_list = torch.Tensor([timesteps for a in range(image.shape[0])]).to(image.device).long()

    image = infer_scheduler.add_noise(image, noise, timesteps_list).to(device) #TODO


    intermediates = []
    intermediates_step = 20

    for t in range(timesteps, 0, -1): # va de timesteps à 0
        
        model_output = model(
            image, timesteps=torch.Tensor((t,)).to(device), context=None
        )
        #print(model_output.shape)
        
        image, _ = infer_scheduler.step(model_output, t, image)
    
        if (t== timesteps-1 or t%intermediates_step == 0) and return_intermediates:
            intermediates.append(image)

    if return_intermediates:
        return image, intermediates
    else:
        return image



@torch.no_grad()
def sample_thor(args, model, device, image, infer_scheduler, timesteps=100, return_intermediates=False):
    
    if timesteps >= infer_scheduler.num_train_timesteps:
        dtprint(timesteps, "is too high. Setting to", infer_scheduler.num_train_timesteps-1)
    
    timesteps_list = torch.Tensor([timesteps for a in range(image.shape[0])]).to(image.device).long()
    timesteps_harmonization = np.linspace(10, timesteps, num=args.thor["nb_timesteps_harmonization"], dtype=int).tolist()


    simplexObj = simplex.Simplex_CLASS()

    original_image = copy.deepcopy(image)

    if args.noise["type"] == "simplex":
        noise = simplex_ddpm.generate_simplex_noise(simplexObj, image.shape, normalize=args.noise["normalize"]).to(device)
    if args.noise["type"] == "gaussian":
        noise = torch.randn(image.shape).to(device)
    

    image = infer_scheduler.add_noise(image, noise, timesteps_list).to(device)

    intermediates_mixed_images_visualize = []
    intermediates_pseudo_anomaly_masks = []
    intermediates_pseudo_anomaly_masks_processed = []

    
    for t in range(timesteps, 0, -1): # goes from timesteps to 0
        
        # compute previous image
        model_output = model(image, timesteps=torch.Tensor((t,)).to(device), context=None)
        image, image_before_step = infer_scheduler.step(model_output, t, image) # here image_before_step is just the image at the timestep+1
            
        
        if t in timesteps_harmonization:
            
            
            pseudo_anomaly_mask, _, _ = thor_ddpm.get_anomaly_mask(copy.deepcopy(image_before_step), copy.deepcopy(original_image), device=device, hist_eq=False)
            
            intermediates_pseudo_anomaly_masks.append(pseudo_anomaly_mask)
            pseudo_anomaly_mask = pseudo_anomaly_mask.cpu().detach().numpy()
            

            pseudo_anomaly_mask_processed = torch.Tensor(thor_ddpm.get_region_anomaly_mask(pseudo_anomaly_mask, kernel_size=4)).to(device).clip(0,1) # simple erosion dilation 
            

            pseudo_anomaly_mask_processed = pseudo_anomaly_mask_processed.clip(0,1) 

            intermediates_pseudo_anomaly_masks_processed.append(pseudo_anomaly_mask_processed)

            image_0 = pseudo_anomaly_mask_processed * image_before_step + (1-pseudo_anomaly_mask_processed) * original_image

            image_0 = torch.clamp(image_0, 0, 1)
            
            image_0 = scale_intensity_from_histogram_peak(image_0, 2.0/7.0) #TODO

            image_0 = torch.clamp(image_0, 0, 1)

            image = infer_scheduler.add_noise(image_0, noise, torch.Tensor((t,)).to(device).long())
            
            intermediates_mixed_images_visualize.append(image)

    if return_intermediates:
        return image, intermediates_mixed_images_visualize, intermediates_pseudo_anomaly_masks, intermediates_pseudo_anomaly_masks_processed
    else:
        return image, intermediates_pseudo_anomaly_masks_processed


@torch.no_grad()
def my_sample_ddim(
    args,
    model,
    device,
    image,
    infer_scheduler,
    timesteps,
    ddim_steps=None,
    eta=0.0,
    return_intermediates=False,
):
    """
    DDIM-style sampler starting from a noised input image at `timesteps`.

    Args:
        timesteps: max diffusion timestep used to corrupt input before denoising.
        ddim_steps: number of reverse steps (<= timesteps). If None, uses full schedule.
        eta: DDIM stochasticity parameter (0.0 => deterministic DDIM, >0 adds noise).
    """
    simplexObj = simplex.Simplex_CLASS()

    if args.noise["type"] == "simplex":
        noise = simplex_ddpm.generate_simplex_noise(
            simplexObj, image.shape, normalize=args.noise["normalize"]
        ).to(device)
    if args.noise["type"] == "gaussian":
        noise = torch.randn(image.shape).to(device)

    if timesteps >= infer_scheduler.num_train_timesteps:
        dtprint(timesteps, "is too high. Setting to", infer_scheduler.num_train_timesteps - 1)
        timesteps = infer_scheduler.num_train_timesteps - 1

    if timesteps <= 0:
        return (image, []) if return_intermediates else image

    timesteps_list = torch.Tensor([timesteps for _ in range(image.shape[0])]).to(image.device).long()
    image = infer_scheduler.add_noise(image, noise, timesteps_list).to(device)

    if ddim_steps is None:
        ddim_steps = timesteps
    ddim_steps = max(1, min(int(ddim_steps), int(timesteps)))

    # Build monotonically increasing DDIM schedule and then iterate in reverse.
    ddim_schedule = np.linspace(0, timesteps, num=ddim_steps + 1, dtype=int)
    ddim_schedule = np.unique(ddim_schedule)
    if ddim_schedule[-1] != timesteps:
        ddim_schedule = np.append(ddim_schedule, timesteps)

    alphas_cumprod = infer_scheduler.alphas_cumprod.to(device=device, dtype=image.dtype)
    clip_sample = getattr(infer_scheduler, "clip_sample", False)
    clip_values = getattr(infer_scheduler, "clip_sample_values", (-1.0, 1.0))
    prediction_type = str(getattr(infer_scheduler, "prediction_type", "epsilon")).lower()

    intermediates = []
    for idx in range(len(ddim_schedule) - 1, 0, -1):
        t = int(ddim_schedule[idx])
        prev_t = int(ddim_schedule[idx - 1])

        model_output = model(image, timesteps=torch.Tensor((t,)).to(device), context=None)

        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[prev_t] if prev_t >= 0 else torch.tensor(1.0, device=device, dtype=image.dtype)
        beta_t = 1.0 - alpha_t

        if prediction_type == "epsilon":
            pred_x0 = (image - beta_t.sqrt() * model_output) / alpha_t.sqrt()
            pred_eps = model_output
        elif prediction_type == "sample":
            pred_x0 = model_output
            pred_eps = (image - alpha_t.sqrt() * pred_x0) / beta_t.sqrt()
        elif prediction_type == "v_prediction":
            pred_x0 = alpha_t.sqrt() * image - beta_t.sqrt() * model_output
            pred_eps = alpha_t.sqrt() * model_output + beta_t.sqrt() * image
        else:
            # Fallback to epsilon prediction for unknown scheduler prediction types.
            pred_x0 = (image - beta_t.sqrt() * model_output) / alpha_t.sqrt()
            pred_eps = model_output

        if clip_sample:
            pred_x0 = torch.clamp(pred_x0, clip_values[0], clip_values[1])

        # DDIM update (Eq. 12 in https://arxiv.org/abs/2010.02502)
        sigma_t = eta * torch.sqrt(((1 - alpha_prev) / (1 - alpha_t)) * (1 - alpha_t / alpha_prev))
        direction = torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t**2, min=0.0)) * pred_eps

        if eta > 0:
            if args.noise["type"] == "simplex":
                step_noise = simplex_ddpm.generate_simplex_noise(
                    simplexObj, image.shape, normalize=args.noise["normalize"]
                ).to(device)
            else:
                step_noise = torch.randn_like(image)
        else:
            step_noise = torch.zeros_like(image)

        image = alpha_prev.sqrt() * pred_x0 + direction + sigma_t * step_noise

        if return_intermediates:
            intermediates.append(image)

    if return_intermediates:
        return image, intermediates
    else:
        return image