import numpy as np

def histogram_matching(target, source):
    n_bins = 100
    # Calcul des histogrammes et des fonctions de répartition cumulée
    mask_target = target > 1e-5
    hist_target, edge_bin_T = np.histogram(target[mask_target], bins=n_bins, density=True)
    cdf_target = hist_target.cumsum()
    # ic(hist_target,edge_bin_T, cdf_target)

    hist_source, edge_bin_S = np.histogram(source[source > 0], bins=n_bins, density=True)
    cdf_source = hist_source.cumsum()
    
    # Création de la nouvelle image
    new_T = np.zeros_like(target)
    for i, (gt_m, gt_M) in enumerate(zip(edge_bin_T[:-1], edge_bin_T[1:])):
        gs = np.argmin(np.abs(cdf_target[i] - cdf_source))
        mask_gt = np.logical_and((target > gt_m), (target < gt_M))
        np.putmask(new_T, mask_gt, (gs / n_bins))

    new_T[~mask_target] = 0
    return new_T


