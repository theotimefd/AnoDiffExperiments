import numpy as np
import matplotlib.pyplot as plt

def compute_adc(S_b0, S_b, b_value):
    """
    Compute the ADC map from DWI images with multiple b-values and b-vectors.

    Parameters:
    - S_b0: signal intensity at b-value 0
    - S_b: signal intensity at b-value
    - b_value: b-value corresponding to S_b

    Returns:
    - adc_map: computed ADC map
    """

    # Ensure there are no zero values in the denominator
    S_b0 = np.maximum(S_b0, 1e-10)
    S_b = np.maximum(S_b, 1e-10)


    # Compute the ADC map
    adc_map = -np.log(S_b / S_b0) / b_value


    return adc_map

def compute_adc_multi_bvec(S_b0, S_b, b_value):
    """
    Compute the ADC map from DWI images with multiple b-values and b-vectors.

    Parameters:
    - S_b0: signal intensity at b-value 0
    - S_b: signal intensities at different b-values and directions
    - b_value: b-value corresponding to S_b

    Returns:
    - adc_map: computed ADC map
    """

    # Ensure there are no zero values in the denominator
    S_b0 = np.maximum(S_b0, 1e-10)
    S_b = np.maximum(S_b, 1e-10)

    # Average signal intensity across directions for each b-value
    S_b_avg = np.mean(S_b, axis=3)

    # Compute the ADC map
    adc_map = -np.log(S_b_avg / S_b0) / b_value


    return adc_map
