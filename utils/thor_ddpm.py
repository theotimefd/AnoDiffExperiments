import sys
sys.path.append("..")

import torch

from skimage import exposure
from scipy.ndimage import gaussian_filter, median_filter, percentile_filter, grey_dilation, grey_closing, maximum_filter, grey_opening

import numpy as np
import copy

from scipy import stats

import lpips



def lpips_loss(anomaly_img, ph_img, device, retPerLayer=False):
    """
    :param anomaly_img: anomaly image
    :param ph_img: pseudo-healthy image
    :param retPerLayer: whether to return the loss per layer
    :return: LPIPS loss
    """
    if len(ph_img.shape) == 2:
        ph_img = torch.unsqueeze(torch.unsqueeze(ph_img, 0), 0)
        anomaly_img = torch.unsqueeze(torch.unsqueeze(anomaly_img, 0), 0)
    

    anomaly_img = ((anomaly_img * 2) - 1).repeat(1,3,1,1)
    ph_img = ((ph_img * 2) - 1).repeat(1,3,1,1)

    l_pips_sq = lpips.LPIPS(pretrained=True, pnet_rand=False, net='squeeze', eval_mode=True, spatial=True, lpips=True).to(device)

    loss_lpips = l_pips_sq(anomaly_img, ph_img, normalize=True, retPerLayer=retPerLayer)
    if retPerLayer:
        loss_lpips = loss_lpips[1][0]
    return loss_lpips.cpu().detach().numpy()

def get_saliency( x, x_rec, device, retPerLayer=False):
    saliency = lpips_loss(x, x_rec, device, retPerLayer)
    saliency = gaussian_filter(saliency, sigma=2)
    return saliency

def compute_residual(x, x_rec, hist_eq=False):
    """
    :param x_rec: reconstructed image
    :param x: original image
    :param hist_eq: whether to perform histogram equalization
    :return: residual image
    """
    if hist_eq:
        x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
        x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
        x_res = np.abs(x_rec_rescale - x_rescale)
    else:
        x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())

    return x_res

def get_anomaly_mask(x, x_rec, device, hist_eq=False, retPerLayer=False):

    x_res = compute_residual(x, x_rec, hist_eq=hist_eq)

    lpips_mask = get_saliency(x, x_rec, device, retPerLayer=retPerLayer).clip(0,1) 

    x_res2 = np.asarray([(x_res[i] / (np.percentile(x_res[i], 95) + 1e-8)) for i in range(x_res.shape[0])]).clip(0, 1)

    combined_mask_np = lpips_mask * x_res #+ x_res) / 2
    combined_mask_np2 = (lpips_mask * x_res) # x_res2
    # # anomalous: high value, healthy: low value

    # combined_mask_np = area_opening((combined_mask_np * 255).astype(np.uint8)) / 255.0#, square(7))
    # combined_mask_np = closing((combined_mask_np * 255).astype(np.uint8), footprint=np.ones(9,9)) / 255.0#, square(7))
    # # combined_mask_np = ndimage.grey_dilation((combined_mask_np * 255).astype(np.uint8), size=(3)) / 255.0#, square(7))
    
    combined_mask = torch.Tensor(combined_mask_np).to(device)
    # combined_mask = dilate_masks(combined_mask)
    
    combined_mask2 = torch.Tensor(combined_mask_np2).to(device)

    combined_mask = (combined_mask / (torch.max(combined_mask) + 1e-8)).clip(0,1)
    # x_res_neg = (x-x_rec)
    return combined_mask, combined_mask2, torch.Tensor(x_res).to(device)
    # return torch.Tensor(x_res).to(device), torch.Tensor(x_res).to(device), torch.Tensor(x_res).to(device)

def get_region_anomaly_mask(ano_map, kernel_size=13):
    # input_image_ = (np.squeeze(copy.deepcopy(input_image).cpu().detach().numpy())*255).astype(np.uint8)
    final_anomaly_map = (grey_closing(ano_map, size=(1,1,kernel_size,kernel_size), mode='nearest'))#+ ano_map)/2
    final_anomaly_map = (grey_dilation(final_anomaly_map, size=(1,1,kernel_size,kernel_size), mode='nearest') + ano_map)/2
    final_anomaly_map = final_anomaly_map.clip(0,1)
    # final_anomaly_map = ((2**final_anomaly_map)-1).clip(0,1)
    return final_anomaly_map