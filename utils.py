import argparse
import os
import random
import json
import cv2
import natsort
import numpy as np
import torch
from torch.utils.data import Dataset

time_format = '%d_%H_%M_%S'

np_to_tensor = lambda im, device: torch.unsqueeze(torch.permute(torch.Tensor(im), (2, 1, 0)), 0).to(device)

tensor_to_np = lambda im: torch.permute(torch.squeeze(im), (2, 1, 0)).detach().numpy().astype(np.uint8)

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', dest = 'mode', type = str, default = 'train')
	parser.add_argument('--model', dest='model_name', type=str, default="")
	parser.add_argument('--folder', dest='folder_name', type=str, default="")
	parser.add_argument('--test_image', dest='test_image', type=str, default="")

	args = parser.parse_args()

	return args

def save_hyperparameters(folder_name, hyperparams):
    os.chdir(folder_name)
    with open("hyperparameters.txt", "w") as f:
        f.write(json.dumps(hyperparams, indent=2))
    os.chdir("..")

def upscale_image(input_image = None, ratio = 2):
    if input_image is not None:
        AssertionError("No input image")
    w, h, c = input_image.shape
    new_im = np.zeros([int(w*ratio), int(h*ratio), 3])
    im = input_image.astype(int)
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            # new_im[i, j, :] = (im[2*i, 2*j, :] + im[2*i + 1, 2*j, :] + im[2*i, 2*j + 1, :] + im[2*i + 1, 2*j + 1, :])/4
            new_im[2*i, 2*j, :] = input_image[i, j, :]
            new_im[2 * i + 1, 2 * j, :] = input_image[i, j, :]
            new_im[2 * i, 2 * j + 1, :] = input_image[i, j, :]
            new_im[2 * i + 1, 2 * j + 1, :] = input_image[i, j, :]
    return new_im.astype(np.uint8)

def downscale_image(input_image = None, ratio = 2, method = 'simple'):
    if input_image is not None:
        AssertionError("No input image")
    if method != 'simple':
        return
    w, h, c = input_image.shape
    if w%ratio != 0 or h%ratio != 0:
        AssertionError("Image not resizeable")
    new_im = np.zeros([int(w/ratio), int(h/ratio), 3])
    im = input_image.astype(int)
    for i in range(new_im.shape[0]):
        for j in range(new_im.shape[1]):
            new_im[i, j, :] = (im[2*i, 2*j, :] + im[2*i + 1, 2*j, :] + im[2*i, 2*j + 1, :] + im[2*i + 1, 2*j + 1, :])/4
    return new_im.astype(np.uint8)

def get_random_crop(image, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop

class CustomDataSet(Dataset):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)
        self.train_len = int(0.9*len(self))
        self.test_len = 10
        self.val_len = len(self) - self.train_len - self.test_len

    def __len__(self):
        return len(self.total_imgs)

    def get_train_batch(self, batch_size=10, crop_size=32):
        batch = []
        for i in range(batch_size):
            random_ind = random.randrange(1, self.train_len)
            im = cv2.imread(f"{self.main_dir}//{str(random_ind).zfill(4)}.png")
            random_crop = get_random_crop(im, crop_size, crop_size)
            batch.append(random_crop)
        return batch

    def get_random_test_image(self, crop_len = 500):
        random_ind = random.randrange(len(self) - self.test_len, len(self))
        im = cv2.imread(f"{self.main_dir}//{str(random_ind).zfill(4)}.png")
        return get_random_crop(im, crop_height=crop_len, crop_width=crop_len)

def write_loss_to_file(loss):
    with open("loss.txt", "a") as f:
        f.write(str(loss) + '\n')

def save_all(net, optimizer, running_loss, path):
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss
    }, path)

class PSNR:
    def __init__(self):
        self.name = "PSNR"
        self.item = 0

    def __call__(self, img1, img2):
        mse = torch.mean((img2 - img1) ** 2)
        self.item = -20 * torch.log10(255.0 / torch.sqrt(mse))
        return self.item

    def item(self):
        return self.item

class Combined_loss:
    def __init__(self):
        self.name = "Combined"
        self.item = 0
        self.mse = torch.nn.MSELoss()

    def __call__(self, img1, img2):
        mse = torch.mean((img2 - img1) ** 2)
        self.item = -20 * torch.log10(255.0 / torch.sqrt(mse)) + self.mse(img1, img2)
        return self.item

    def item(self):
        return self.item