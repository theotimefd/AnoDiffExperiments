import numpy as np
from monai.metrics import compute_hausdorff_distance, DiceMetric, compute_iou

def compute_scores(y_pred, y_true, only_dice_iou=False):
    """
    Compute various segmentation metrics between predicted and ground truth masks.
    This function calculates multiple metrics to evaluate the quality of segmentation predictions,
    including IoU, Dice coefficient, Hausdorff distance, precision, recall, and F1 score.
    Args:
        y_pred (torch.Tensor): Predicted segmentation mask in BCHWD format (Batch, Channel, Height, Width, Depth).
            Expected to be a 5D tensor with binary values (0 or 1).
        y_true (torch.Tensor): Ground truth segmentation mask in BCHWD format (Batch, Channel, Height, Width, Depth).
            Expected to be a 5D tensor with binary values (0 or 1).
    Returns:
        tuple: A tuple containing six lists:
            - iou_scores (list): List containing flattened IoU scores with NaN values removed.
            - dice_scores (list): List containing flattened Dice coefficient scores with NaN values removed.
            - hausdorff_distances (list): List containing flattened Hausdorff distances with NaN values removed.
            - precision_scores (list): List containing precision values.
            - recall_scores (list): List containing recall values.
            - f1_scores (list): List containing F1 scores.
    Raises:
        AssertionError: If inputs are not 5D tensors in BCHWD format.
    Note:
        - Background class is excluded from Dice metric computation.
        - NaN values are filtered out from IoU, Dice, and Hausdorff distance calculations.
        - Precision, recall, and F1 are computed on flattened arrays across all dimensions.
    """

    #make sure inputs are in BCHWD format
    assert y_pred.ndim == 5 and y_true.ndim == 5, "Inputs must be 5D tensors in BCHWD format."
    
    dice_metric = DiceMetric(include_background=False, reduction="sum")
    
    iou_scores = []
    dice_scores = []
    hausdorff_distances = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # IoU
    iou_score = compute_iou(y_pred, y_true)
    flattened_iou_score = iou_score.cpu().numpy().flatten()
    flattened_iou_score = flattened_iou_score[~np.isnan(flattened_iou_score)] # remove NaN values
    iou_scores.append(flattened_iou_score)

    # DICE
    dice_score = dice_metric(y_pred, y_true).cpu().numpy().flatten()
    dice_score = dice_score[~np.isnan(dice_score)] # remove NaN values
    dice_scores.append(dice_score)

    if only_dice_iou:
        return iou_scores, dice_scores, [], [], [], []

    # Hausdorff Distance
    hausdorff_distance = compute_hausdorff_distance(y_pred, y_true)
    flattened_hausdorff_distance = hausdorff_distance.cpu().numpy().flatten()
    flattened_hausdorff_distance = flattened_hausdorff_distance[~np.isnan(flattened_hausdorff_distance)] # remove NaN values
    hausdorff_distances.append(flattened_hausdorff_distance)

    # Precision, Recall, F1
    y_pred_np = y_pred.cpu().numpy().flatten()
    y_true_np = y_true.cpu().numpy().flatten()
    
    true_positives = np.sum((y_pred_np == 1) & (y_true_np == 1))
    false_positives = np.sum((y_pred_np == 1) & (y_true_np == 0))
    false_negatives = np.sum((y_pred_np == 0) & (y_true_np == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    return iou_scores, dice_scores, hausdorff_distances, precision_scores, recall_scores, f1_scores


def make_confidence_intervals(scores_list, bootstrap_rounds=200):
    """
    Calculate bootstrap confidence intervals for a list of scores.
    This function performs bootstrap resampling to estimate the mean and 95% confidence
    intervals of the input scores. It uses by default 200 bootstrap rounds with a fixed random seed
    for reproducibility.
    Args:
        scores_list (array-like): A 1D array or list of numerical scores for which to
            calculate confidence intervals.
        bootstrap_rounds (int, optional): The number of bootstrap rounds to perform. Default is 200.
    Returns:
        tuple: A tuple containing three float values:
            - mean (float): The bootstrap mean of the scores.
            - ci_lower (float): The lower bound of the 95% confidence interval (2.5th percentile).
            - ci_upper (float): The upper bound of the 95% confidence interval (97.5th percentile).
    Example:
        >>> scores = np.array([0.8, 0.85, 0.9, 0.82, 0.88])
        >>> mean, lower, upper = make_confidence_intervals(scores)
        >>> print(f"Mean: {mean:.3f}, CI: [{lower:.3f}, {upper:.3f}]")
    """

    rng = np.random.default_rng(42)

    idx = np.arange(len(scores_list))

    scores_boot = []

    for i in range(bootstrap_rounds): 
        # bootstrap rounds: random sampling with replacement of the predictions
        pred_idx = rng.choice(idx, size=len(idx), replace=True)
        
        score_boot = np.mean(scores_list[pred_idx])
        
        scores_boot.append(score_boot)

    # Compute the mean and 95% confidence intervals
    mean = np.mean(scores_boot)
    ci_lower = np.percentile(scores_boot, 2.5)
    ci_upper = np.percentile(scores_boot, 97.5)

    return mean, ci_lower, ci_upper