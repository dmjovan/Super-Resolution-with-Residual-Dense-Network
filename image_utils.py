import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def upscale_image(input_image: Optional[np.ndarray] = None, ratio: int = 2) -> np.ndarray:
    """ Upscaling input_image with respect to the ratio. """

    if input_image is not None:
        logger.error("No input image for upscale operation.")

    w, h, c = input_image.shape

    new_im = np.zeros(shape=(int(w * ratio), int(h * ratio), 3))
    img = input_image.astype(int)

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            # new_im[i, j, :] = (im[2*i, 2*j, :] + im[2*i + 1, 2*j, :] + im[2*i, 2*j + 1, :] + im[2*i + 1, 2*j + 1, :])/4
            new_im[2 * i, 2 * j, :] = input_image[i, j, :]
            new_im[2 * i + 1, 2 * j, :] = input_image[i, j, :]
            new_im[2 * i, 2 * j + 1, :] = input_image[i, j, :]
            new_im[2 * i + 1, 2 * j + 1, :] = input_image[i, j, :]
            
    return new_im.astype(np.uint8)


def downscale_image(input_image: Optional[np.ndarray] = None, ratio: int = 2) -> np.ndarray:
    """ Downscaling input_image with respect to the ratio. """

    if input_image is not None:
        logger.error("No input image for downscale operation.")

    w, h, c = input_image.shape

    if w % ratio != 0 or h % ratio != 0:
        logger.error("Image is not resizeable.")

    new_img = np.zeros(shape=(int(w / ratio), int(h / ratio), 3))
    img = input_image.astype(int)

    for i in range(new_img.shape[0]):
        for j in range(new_img.shape[1]):
            new_img[i, j, :] = (img[2 * i, 2 * j, :] + img[2 * i + 1, 2 * j, :] +
                                img[2 * i, 2 * j + 1, :] + img[2 * i + 1, 2 * j + 1, :]) / 4

    return new_img.astype(np.uint8)
