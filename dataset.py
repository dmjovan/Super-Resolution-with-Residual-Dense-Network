import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset


class SuperResolutionDataset(Dataset):

    def __init__(self, dir_path: str) -> None:
        self.main_dir = dir_path
        self.images = sorted(os.listdir(self.main_dir), key=lambda x: int(x[:-4]))

        self.train_len = int(0.9 * len(self))
        self.test_len = 10
        self.val_len = len(self) - self.train_len - self.test_len

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        pass

    @staticmethod
    def get_random_crop(image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height

        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)

        crop = image[y: y + crop_height, x: x + crop_width]

        return crop

    def get_train_batch(self, batch_size=10, crop_size=32):
        batch = []
        for i in range(batch_size):
            random_ind = random.randrange(1, self.train_len)
            im = cv2.imread(f"{self.main_dir}//{str(random_ind).zfill(4)}.png")
            random_crop = self.get_random_crop(im, crop_size, crop_size)
            batch.append(random_crop)
        return batch

    def get_random_test_image(self, crop_len=500):
        random_ind = random.randrange(len(self) - self.test_len, len(self))
        im = cv2.imread(f"{self.main_dir}//{str(random_ind).zfill(4)}.png")
        return self.get_random_crop(im, crop_height=crop_len, crop_width=crop_len)
