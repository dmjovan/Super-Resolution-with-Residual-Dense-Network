import os

import PIL.Image as pil_image
import h5py
import numpy as np
from torchvision.transforms import transforms

import logger as logger

_logger = logger.get_logger(__name__)

TRAIN_DATASET_PATH = "datasets/train/images"
VALIDATION_DATASET_PATH = "datasets/validation/images"
SCALE = 2


def prepare_train_dataset_with_five_crop():
    # Creating .h5 file
    h5_file = h5py.File(os.path.join(os.getcwd(), "datasets/train/train_dataset_4000.h5"), "w")

    # Creating low_resolution and high_resolution groups:
    # low_resolution == inputs
    # high_resolution == targets/labels

    lr_group = h5_file.create_group("low_resolution")
    hr_group = h5_file.create_group("high_resolution")

    image_names = sorted(os.listdir(TRAIN_DATASET_PATH), key=lambda x: int(x[:-4]))
    patch_idx = 0

    for i, image_name in enumerate(image_names):

        # Loading high resolution image
        hr = pil_image.open(os.path.join(os.getcwd(), f"{TRAIN_DATASET_PATH}/{image_name}")).convert("RGB")

        # Cropping HR image into 5 cropped images with proper size:
        # top_left_crop, top_right_crop, bottom_left_crop, bottom_right_crop, center_crop
        cropped_hr_images = transforms.FiveCrop(size=(hr.height // SCALE, hr.width // SCALE))(hr)

        for hr in cropped_hr_images:
            # Applying BICUBIC degradation to low-resolution images
            hr = hr.resize(((hr.width // SCALE) * SCALE, (hr.height // SCALE) * SCALE), resample=pil_image.BICUBIC)
            lr = hr.resize((hr.width // SCALE, hr.height // SCALE), resample=pil_image.BICUBIC)

            hr = np.array(hr)
            lr = np.array(lr)

            # Adding images into dataset
            lr_group.create_dataset(str(patch_idx), data=lr)
            hr_group.create_dataset(str(patch_idx), data=hr)

            patch_idx += 1

        _logger.info(f"Finished image {image_name}")

    h5_file.close()


def prepare_train_dataset_with_random_crop():
    # Creating .h5 file
    h5_file = h5py.File(os.path.join(os.getcwd(), "datasets/train/train_dataset_800.h5"), "w")

    # Creating low_resolution and high_resolution groups:
    # low_resolution == inputs
    # high_resolution == targets/labels

    lr_group = h5_file.create_group("low_resolution")
    hr_group = h5_file.create_group("high_resolution")

    image_names = sorted(os.listdir(TRAIN_DATASET_PATH), key=lambda x: int(x[:-4]))
    patch_idx = 0

    for i, image_name in enumerate(image_names):

        # Loading high resolution image
        hr = pil_image.open(os.path.join(os.getcwd(), f"{TRAIN_DATASET_PATH}/{image_name}")).convert("RGB")

        cropped_hr_image = transforms.RandomCrop(size=(hr.height // SCALE, hr.width // SCALE))(hr)

        # Applying BICUBIC degradation to low-resolution images
        hr = hr.resize(((hr.width // SCALE) * SCALE, (hr.height // SCALE) * SCALE), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // SCALE, hr.height // SCALE), resample=pil_image.BICUBIC)

        hr = np.array(hr)
        lr = np.array(lr)

        # Adding images into dataset
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        _logger.info(f"Finished image {image_name}")

    h5_file.close()


def prepare_validation_dataset():
    # Creating .h5 file
    h5_file = h5py.File(os.path.join(os.getcwd(), "datasets/validation/validation_dataset.h5"), 'w')

    # Creating low_resolution and high_resolution groups:
    # low_resolution == inputs
    # high_resolution == targets/labels

    lr_group = h5_file.create_group("low_resolution")
    hr_group = h5_file.create_group("high_resolution")

    image_names = sorted(os.listdir(VALIDATION_DATASET_PATH), key=lambda x: int(x[:-4]))

    for i, image_name in enumerate(image_names):
        # Loading high resolution image
        hr = pil_image.open(os.path.join(os.getcwd(), f"{VALIDATION_DATASET_PATH}/{image_name}")).convert("RGB")

        # Applying BICUBIC degradation to low-resolution images
        hr = hr.resize(((hr.width // SCALE) * SCALE, (hr.height // SCALE) * SCALE), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // SCALE, hr.height // SCALE), resample=pil_image.BICUBIC)

        hr = np.array(hr)
        lr = np.array(lr)

        # Adding images into dataset
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

        _logger.info(f"Finished image {image_name}")

    h5_file.close()


if __name__ == '__main__':
    """ Preparing 2K DIV2K Datasets by converting them properly into .h5 files """

    _logger.info("Running training dataset preparation ...")
    # Preparing train dataset
    prepare_train_dataset_with_five_crop()
    # prepare_train_dataset_with_random_crop()
    _logger.info("Training dataset preparation finished ...")

    _logger.info("Running validation dataset preparation ...")
    # Preparing validation dataset
    prepare_validation_dataset()
    _logger.info("Validation dataset preparation finished ...")
