import numpy as np
from monai.utils.type_conversion import convert_to_numpy
from monai.bundle import ConfigParser
import sys
import time
sys.path.append("..")
sys.path.append("../..")


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def normalize_image_to_uint8(image):
    """
    Normalize image to uint8
    Args:
        image: numpy array
    """
    draw_img = image
    if np.amin(draw_img) < 0:
        draw_img -= np.amin(draw_img)
    if np.amax(draw_img) > 1:
        draw_img /= np.amax(draw_img)
    draw_img = (255 * draw_img).astype(np.uint8)
    return draw_img


def visualize_one_slice_in_3d_image(image, axis: int = 2):
    """
    Prepare a 2D image slice from a 3D image for visualization.
    Args:
        image: image numpy array, sized (H, W, D)
    """
    image = convert_to_numpy(image)
    # draw image
    center = image.shape[axis] // 2
    if axis == 0:
        draw_img = normalize_image_to_uint8(image[center, :, :])
    elif axis == 1:
        draw_img = normalize_image_to_uint8(image[:, center, :])
    elif axis == 2:
        draw_img = normalize_image_to_uint8(image[:, :, center])
    else:
        raise ValueError("axis should be in [0,1,2]")
    draw_img = np.stack([draw_img, draw_img, draw_img], axis=-1)
    return draw_img

def tprint(text):
    # print with time only
    print(f"[{time.strftime('%H:%M:%S')}] {text}")

def dtprint(text):
    # print with date and time
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}")


def scale_intensity_from_histogram_peak(input_image, target_value=1.0, min_search_value=None):
    # scales the histogram peak (most occurred pixel intensity) to target value
    # to be used only on mri images with intensities between 0 and 1
    try:
        input_np = input_image.cpu().numpy()
    except AttributeError:
        input_np = input_image

    min_search_value = min_search_value if min_search_value is not None else np.max(input_np)/15.0

    hist, bin_edges = np.histogram(input_np.flatten(), bins=100, range=(min_search_value, 0.8))

    peak_value = bin_edges[np.argmax(hist)]

    normalized_image = input_image / peak_value * target_value

    return normalized_image