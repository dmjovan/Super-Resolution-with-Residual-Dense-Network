import random
from typing import Union

import numpy as np
import torch
from torch import Tensor
from torchvision.transforms import transforms

import logger as logger

_logger = logger.get_logger(__name__)


def rgb_to_y(img: np.ndarray, layout: str = "hwc") -> float:
    """ Converting RGB to YCbCr and reading only Y [luminance] component """

    if layout == "hwc":
        return 16. + (64.738 * img[..., 0] + 129.057 * img[..., 1] + 25.064 * img[..., 2]) / 256.
    elif layout == "chw":
        return 16. + (64.738 * img[0] + 129.057 * img[1] + 25.064 * img[2]) / 256.
    else:
        raise RuntimeError("Only 'chw' and 'hwc' image layouts are supported.")


def denormalize_image(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """ Returning image values into range 0-255"""

    if isinstance(img, torch.Tensor):
        return img.mul(255.0).clamp(0.0, 255.0)

    elif isinstance(img, np.ndarray):
        return np.clip((255 * img).astype(np.uint8), a_min=0, a_max=255)


def randomly_crop_image(img: np.ndarray, crop_size: int = 500) -> np.ndarray:
    """ Cropping image randomly. Max size is 500 x 500. """

    assert crop_size <= 500, f"Random crop oversized - crop_size = {crop_size}"

    img_left = random.randint(0, img.shape[1] - crop_size)
    img_right = img_left + crop_size
    img_top = random.randint(0, img.shape[0] - crop_size)
    img_bottom = img_top + crop_size

    cropped_image = img[img_top:img_bottom, img_left:img_right]

    return cropped_image


def central_crop_image(img: np.ndarray, crop_size: int = 500) -> np.ndarray:
    """ Central crop of image """
    return np.array(transforms.CenterCrop(size=(crop_size, crop_size))(Tensor(img.transpose((2, 0, 1))))).transpose(
        (1, 2, 0)).astype(np.uint8)


def psnr(img1: torch.Tensor, img2: torch.Tensor, max_value: float = 255.0) -> float:
    """ Calculating PSNR (peak-signal-noise-ratio) """
    return 10. * ((max_value ** 2) / ((img1 - img2) ** 2).mean()).log10()


class RunningMean:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
