import os
import random
from typing import List

import cv2
import numpy as np
from torch.utils.data import Dataset


# FIXME: Replace this with torch.DataLoader and torch.Dataset and RandomCrop

class SuperResolutionDataset(Dataset):

    def __init__(self, dir_path: str, train_ratio: float = 0.9) -> None:
        """ Creating dataset object for Super Resolution task """

        self.main_dir = dir_path

        # Listing all images names [0001.png -> 0800.png]
        self.dataset_images = sorted(os.listdir(self.main_dir), key=lambda x: int(x[:-4]))

        # Training images
        self.train_len = int(train_ratio * len(self))
        self.train_indices = random.sample(range(len(self)), self.train_len)
        self.train_images = list(map(self.dataset_images.__getitem__, self.train_indices))

        # Testing images
        self.test_indices = list(set(range(len(self))).difference(set(self.train_indices)))
        self.test_len = len(self.test_indices)
        self.test_images = list(map(self.dataset_images.__getitem__, self.test_indices))

    def __len__(self):
        return len(self.dataset_images)

    def __getitem__(self, item):
        pass

    def __str__(self) -> str:
        dataset_string = f"Super Resolution dataset review: \n" \
                         f"############################################\n" \
                         f"Total dataset images: {len(self)}\n" \
                         f"Number of training images: {self.train_len}\n" \
                         f"Number of test images: {self.test_len}\n" \
                         f"############################################"

        return dataset_string

    @staticmethod
    def crop_randomly(image: np.ndarray, crop_height: int = 200, crop_width: int = 200) -> np.ndarray:
        """ Randomly cropping image on both H and W dimensions. """

        h, w, c = image.shape

        max_vertical = h - crop_height
        max_horizontal = w - crop_width

        vertical_start = np.random.randint(0, max_vertical)
        vertical_end = vertical_start + crop_height

        horizontal_start = np.random.randint(0, max_horizontal)
        horizontal_end = horizontal_start + crop_width

        cropped = image[vertical_start:vertical_end, horizontal_start:horizontal_end, :]

        return cropped

    def get_train_batch(self, batch_size: int = 10, crop_size: int = 200) -> List[np.ndarray]:
        """ Getting batch of size batch_size of training images. """

        batch = []
        for i in range(batch_size):
            # Getting one training image name
            image_name = next(iter(random.sample(self.train_images, 1)))

            # Reading one training image
            img = cv2.imread(os.path.join(self.main_dir, image_name))

            # Random cropping of one training image
            random_crop = self.crop_randomly(img, crop_size, crop_size)

            # Adding one training image into list
            batch.append(random_crop)

        return batch

    def get_random_test_image(self, crop_size: int = 200) -> np.ndarray:
        """ Getting one randomly selected image from test dataset. """

        # Getting one testing image name
        image_name = next(iter(random.sample(self.test_images, 1)))

        # Reading one testing image
        img = cv2.imread(os.path.join(self.main_dir, image_name))

        # Random cropping of one testing image
        random_cropped = self.crop_randomly(img, crop_size, crop_size)

        return random_cropped
