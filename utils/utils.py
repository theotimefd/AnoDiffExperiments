from monai.bundle import ConfigParser
from monai.data import DataLoader
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
)
import sys
sys.path.append("..")
sys.path.append("../..")
from utils.custom_transforms import Get2DSlice, Get2DSliceWithRandomOffset, SetBackgroundToZero


def define_instance(args, instance_def_key):
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)