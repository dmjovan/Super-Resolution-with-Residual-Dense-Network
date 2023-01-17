import numpy as np

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


def denormalize_image(img: np.ndarray) -> np.ndarray:
    """ Returning image values into range 0-255"""

    return img.mul(255.0).clamp(0.0, 255.0)


def psnr(img1: np.ndarray, img2: np.ndarray, max: float = 255.0) -> float:
    """ Calculating PSNR (peak-signal-noise-ratio) """
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()


class Averager:

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

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
