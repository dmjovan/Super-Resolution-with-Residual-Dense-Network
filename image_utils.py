from typing import Optional

import numpy as np

import logger as logger

_logger = logger.get_logger(__name__)

""" Utility for handling images downscaling and upscaling. """


def upscale_image(input_image: Optional[np.ndarray]) -> np.ndarray:
    """
        Upscaling input_image with respect to the ratio of 2.
        Upscaling is done by repeating pixels for upscaled neighbourhood.
    """

    if input_image is None:
        raise RuntimeError("No input image for upscale operation.")

    # Fixed ratio, only upscaling by 2 is supported
    ratio = 2

    h, w, c = input_image.shape

    new_h, new_w, new_c = int(h * ratio), int(w * ratio), 3

    new_img = np.zeros(shape=(new_h, new_w, new_c))

    for i in range(h):
        for j in range(w):
            new_img[2 * i, 2 * j, :] = input_image[i, j, :]
            new_img[2 * i + 1, 2 * j, :] = input_image[i, j, :]
            new_img[2 * i, 2 * j + 1, :] = input_image[i, j, :]
            new_img[2 * i + 1, 2 * j + 1, :] = input_image[i, j, :]

    return new_img.astype(np.uint8)


def downscale_image(input_image: Optional[np.ndarray]) -> np.ndarray:
    """
        Downscaling input_image with respect to the ratio of 2.
        Downscaling is done as a mean value of nearest neighbours.
    """

    if input_image is None:
        raise RuntimeError("No input image for downscale operation.")

    # Fixed ratio, only downscaling by 2 is supported
    ratio = 2

    h, w, c = input_image.shape

    if h % ratio != 0 or w % ratio != 0:
        raise RuntimeError("Image is not resizeable.")

    new_h, new_w, new_c = int(h / ratio), int(w / ratio), 3

    new_img = np.zeros(shape=(new_h, new_w, new_c))
    img = input_image.astype(int)

    for i in range(new_h):
        for j in range(new_w):
            new_img[i, j, :] = (img[2 * i, 2 * j, :] + img[2 * i + 1, 2 * j, :] +
                                img[2 * i, 2 * j + 1, :] + img[2 * i + 1, 2 * j + 1, :]) / 4

    return new_img.astype(np.uint8)
