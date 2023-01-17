import random
from typing import Tuple

import h5py
import numpy as np
from torch.utils.data import Dataset


class SuperResolutionTrainDataset(Dataset):

    def __init__(self, h5_file_path: str, patch_size: int = 32):
        super(SuperResolutionTrainDataset, self).__init__()

        self.h5_file_path = h5_file_path
        self.patch_size = patch_size

    @staticmethod
    def random_crop(input_image: np.ndarray, target_image: np.ndarray, patch_size: int) -> Tuple[
        np.ndarray, np.ndarray]:

        """ Randomly cropping images into (patch_size x patch_size) images """

        input_image_left = random.randint(0, input_image.shape[1] - patch_size)
        input_image_right = input_image_left + patch_size
        input_image_top = random.randint(0, input_image.shape[0] - patch_size)
        input_image_bottom = input_image_top + patch_size

        target_image_left = input_image_left * 2
        target_image_right = input_image_right * 2
        target_image_top = input_image_top * 2
        target_image_bottom = input_image_bottom * 2

        input_image = input_image[input_image_top:input_image_bottom, input_image_left:input_image_right]
        target_image = target_image[target_image_top:target_image_bottom, target_image_left:target_image_right]

        return input_image, target_image

    @staticmethod
    def random_horizontal_flip(input_image: np.ndarray, target_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """ Randomly flip images horizontally """

        if random.random() < 0.5:
            input_image = input_image[:, ::-1, :].copy()
            target_image = target_image[:, ::-1, :].copy()

        return input_image, target_image

    @staticmethod
    def random_vertical_flip(input_image: np.ndarray, target_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """ Randomly flip images vertically """

        if random.random() < 0.5:
            input_image = input_image[::-1, :, :].copy()
            target_image = target_image[::-1, :, :].copy()

        return input_image, target_image

    @staticmethod
    def random_rotate_90(input_image: np.ndarray, target_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        """ Randomly rotate images for 90 degrees """

        if random.random() < 0.5:
            input_image = np.rot90(input_image, axes=(1, 0)).copy()
            target_image = np.rot90(target_image, axes=(1, 0)).copy()

        return input_image, target_image

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_file_path, "r") as f:
            # Reading images from training .h5 file
            input_image = f["low_resolution"][str(idx)][::]
            target_image = f["high_resolution"][str(idx)][::]

            # Applying augmentations on images
            input_image, target_image = self.random_crop(input_image, target_image, self.patch_size)
            input_image, target_image = self.random_horizontal_flip(input_image, target_image)
            input_image, target_image = self.random_vertical_flip(input_image, target_image)
            input_image, target_image = self.random_rotate_90(input_image, target_image)

            # Transposing and scaling images
            input_image = input_image.astype(np.float32).transpose([2, 0, 1]) / 255.0
            target_image = target_image.astype(np.float32).transpose([2, 0, 1]) / 255.0

            return input_image, target_image

    def __len__(self):
        with h5py.File(self.h5_file_path, "r") as h5_file:
            return len(h5_file["low_resolution"])

    def __str__(self) -> str:
        dataset_string = f"Dataset review: \n" \
                         f"############################################\n" \
                         f"Dataset type: TRAIN\n" \
                         f"Total dataset images: {len(self)}\n" \
                         f"############################################"

        return dataset_string


class SuperResolutionValidationDataset(Dataset):

    def __init__(self, h5_file_path: str):
        super(SuperResolutionValidationDataset, self).__init__()

        self.h5_file_path = h5_file_path

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.h5_file_path, "r") as f:
            # Reading images from validation .h5 file
            input_image = f["low_resolution"][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) / 255.0
            target_image = f["high_resolution"][str(idx)][::].astype(np.float32).transpose([2, 0, 1]) / 255.0

            return input_image, target_image

    def __len__(self):
        with h5py.File(self.h5_file_path, "r") as h5_file:
            return len(h5_file["low_resolution"])

    def __str__(self) -> str:
        dataset_string = f"Dataset review: \n" \
                         f"############################################\n" \
                         f"Dataset type: VALIDATION\n" \
                         f"Total dataset images: {len(self)}\n" \
                         f"############################################"

        return dataset_string
