import os
import time
from typing import Optional

import PIL.Image as pil_image
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import logger as logger
from datasets import SuperResolutionTrainDataset, SuperResolutionValidationDataset
from network import SuperResolutionNetwork
from params import SuperResolutionParams
from utils import Averager, rgb_to_y, denormalize_image, psnr, randomly_crop_image

_logger = logger.get_logger(__name__)

time_format_for_file = "%d_%m_%Y_%H_%M_%S"
time_format_for_log = "%d-%m-%Y %H:%M:%S"


class SuperResolutionHandler:

    def __init__(self, mode: Optional[str], random_crop_train_dataset: Optional[str], validation: Optional[str],
                 model_path: Optional[str],
                 hyper_params_path: Optional[str],
                 test_image_path: Optional[str]) -> None:
        """ Creating SuperResolutionHandler object and validating provided arguments. """

        # Arguments
        self.mode = mode
        self.random_crop_train_dataset = True if random_crop_train_dataset == "true" else False
        self.validation = True if validation == "true" else False
        self.model_path = model_path
        self.hyper_params_path = hyper_params_path
        self.test_image_path = test_image_path
        self._validate_arguments()

        torch.manual_seed(123)

        # Device
        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
            _logger.info(f"Device set to: {torch.cuda.get_device_name(self.device)}")
        else:
            _logger.info("Device set to: CPU")

        # Hyper-parameters
        self.hyper_parameters = SuperResolutionParams(self.hyper_params_path)
        _logger.info(f"Hyper-parameters loaded from the path: {self.hyper_params_path}")

        self.scale = self.hyper_parameters.get_param("scale")

        if self.mode == "train":
            _logger.info(str(self.hyper_parameters))

            # Training dataset
            if self.random_crop_train_dataset:
                self.train_dataset_path = os.path.join(os.getcwd(), "datasets/train/train_dataset_random_crop.h5")
            else:
                self.train_dataset_path = os.path.join(os.getcwd(), "datasets/train/train_dataset_five_crop.h5")
            self.train_dataset = SuperResolutionTrainDataset(h5_file_path=self.train_dataset_path)
            _logger.info(str(self.train_dataset))

            # Train set dataloader
            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                               batch_size=self.hyper_parameters.get_param("batch_size"),
                                               shuffle=True,
                                               num_workers=8,
                                               pin_memory=True)

            if self.validation:
                # Validation dataset
                self.validation_dataset_path = os.path.join(os.getcwd(),
                                                            "datasets/validation/validation_dataset_random_crop.h5")
                self.validation_dataset = SuperResolutionValidationDataset(h5_file_path=self.validation_dataset_path)
                _logger.info(str(self.validation_dataset))

                # Validation set dataloader
                self.validation_dataloader = DataLoader(dataset=self.validation_dataset, batch_size=1)

                # Tensorboard summary writer on the default runs folder
                self.summary_writer = SummaryWriter()

        # Network
        self.net = SuperResolutionNetwork(hyper_parameters=self.hyper_parameters.get_params()).to(self.device)
        _logger.info("Super Resolution Network successfully created")

        # Optimizer
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=self.hyper_parameters.get_param("lr"))
        _logger.info(
            f"Optimizer [Adam] successfully created with learning rate {self.hyper_parameters.get_param('lr')}")

    def _validate_arguments(self) -> None:
        """ Validating arguments from command line. Redefining model_path if provided does not exist. """

        if self.mode not in ["train", "test"]:
            _logger.error("Program argument 'mode' can be either 'train' or 'test'.")
        else:
            _logger.info(f"{self.mode.upper()} mode selected")

        if self.mode == "train" and self.model_path and os.path.exists(self.model_path):
            _logger.warning("Training on pretrained model is not supported. Proceeding with training from scratch.")

        if self.model_path and not os.path.exists(self.model_path):
            self.model_path = os.path.join(os.getcwd(), "models/rdn_x2.pth")
            _logger.warning(
                "Program argument 'model_path' you specified does not exist. Using best model from project.")

        if not self.hyper_params_path or not os.path.exists(self.hyper_params_path):
            self.hyper_params_path = os.path.join(os.getcwd(), "experiments/default_hyper_params.json")
            _logger.warning(
                "Program argument 'hyper_params_path' you specified does not exist. Using default hyper-parameters.")

        if not self.test_image_path or not os.path.exists(self.test_image_path):
            self.test_image_path = os.path.join(os.getcwd(), "demo_images/input.png")

    def run(self) -> None:
        """ Running SuperResolutionHandler """

        _logger.info("Running Super Resolution ...")

        if self.mode == "train":
            self.train()

        elif self.mode == "test":
            self.test()

    def train(self) -> None:
        """ Training SuperResolutionNetwork """

        def decay_learning_rate(curr_epoch: int) -> None:
            """ Decaying learning rate - after each epochs"""

            last_lr = self.hyper_parameters.get_param("lr")
            new_lr = last_lr * (0.1 ** (curr_epoch // int(self.hyper_parameters.get_param("num_epochs") * 0.8)))

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

            _logger.info(f"Optimizer learning rate successfully decayed to {new_lr}")

        # Loss to minimize
        loss_fn = nn.L1Loss()
        try:
            self.hyper_parameters.add_param(key="loss", value=loss_fn.name)
        except AttributeError:
            pass

        start_time = time.strftime(time_format_for_log)
        start_time_for_file = time.strftime(time_format_for_file)
        _logger.info(f"Training started at: {start_time}")

        for epoch in range(1, self.hyper_parameters.get_param("num_epochs") + 1):

            #################### TRAINING ####################

            # Decaying learning rate
            decay_learning_rate(curr_epoch=epoch)

            self.net.train()
            epoch_losses = Averager()

            with tqdm(total=(len(self.train_dataset) -
                             len(self.train_dataset) % self.hyper_parameters.get_param("batch_size")), ncols=80) as t:
                t.set_description(f"Epoch: {epoch}/{self.hyper_parameters.get_param('num_epochs') - 1}")

                for data in self.train_dataloader:
                    # Reading images
                    inputs, targets = data

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Running LR input images through the Super Resolution Residual Dense Network
                    outputs = self.net(inputs)

                    # Calculating loss function
                    loss = loss_fn(outputs, targets)

                    epoch_losses.update(loss.item(), len(inputs))

                    # Zeroing gradients
                    self.optimizer.zero_grad()

                    # Calculating gradients while doing backpropagation
                    loss.backward()

                    # Applying gradients on parameters
                    self.optimizer.step()

                    t.set_postfix(loss="{:.6f}".format(epoch_losses.avg))
                    t.update(len(inputs))

                self.summary_writer.add_scalar("Loss/train", epoch_losses.avg, epoch)

                _logger.info(f"Training loss in epoch {epoch}: {epoch_losses.avg:.6f}")

            # Storing parameters
            if epoch % 10 == 0:
                self.save(folder_name=f"models_{start_time_for_file}", file_name=f"model_{epoch}.pt")

            ################### VALIDATION/EVALUATION AFTER EACH EPOCH ####################

            if self.validation:

                self.net.eval()
                epoch_psnr = Averager()

                for data in self.validation_dataloader:
                    # Reading images
                    inputs, targets = data

                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Running LR input images through the Super Resolution Residual Dense Network
                    with torch.no_grad():
                        outputs = self.net(inputs)

                    # Calculation of PSNR for Y component of outputs and targets
                    outputs = rgb_to_y(denormalize_image(outputs.squeeze(0)), layout="chw")
                    targets = rgb_to_y(denormalize_image(targets.squeeze(0)), layout="chw")

                    outputs = outputs[self.scale:-self.scale, self.scale:-self.scale]
                    targets = targets[self.scale:-self.scale, self.scale:-self.scale]

                    epoch_psnr.update(psnr(outputs, targets), len(inputs))

                self.summary_writer.add_scalar("PSNR/validation", epoch_psnr.avg, epoch)

                _logger.info(f"Validation PSNR in epoch {epoch}: {epoch_psnr.avg:.2f}")

        _logger.info(f"Training finished at: {time.strftime(time_format_for_log)}")

    def test(self):
        """ Testing Super Resolution """

        # Loading checkpoint
        checkpoint = torch.load(self.model_path)

        # Loading model state
        self.net.load_state_dict(checkpoint['model_state_dict'])

        # Loading optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Setting model into evaluation mode
        self.net.eval()

        # Loading test image
        if self.test_image_path and os.path.exists(self.test_image_path):
            test_image = pil_image.open(self.test_image_path).convert('RGB')
            _logger.info(f"Test image loaded from the path: {self.test_image_path}")
        else:
            test_image = pil_image.open(self.test_image_path).convert('RGB')
            _logger.info(f"Specified test image path does not exist. Default test image loaded")

        _logger.info(f"Test image resolution [H x W]:  {test_image.height} x {test_image.width}")

        if test_image.height > 500 and test_image.width > 500:
            _logger.info(f"Randomly cropping test image into 500 x 500 size")
            cropped_test_image = randomly_crop_image(np.array(test_image), crop_size=500)
            test_image = pil_image.fromarray(cropped_test_image)
            test_image.save(self.test_image_path.replace(".", f"_cropped."))

        test_image_width = (test_image.width // self.scale) * self.scale
        test_image_height = (test_image.height // self.scale) * self.scale

        hr = test_image.resize((test_image_width, test_image_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // self.scale, hr.height // self.scale), resample=pil_image.BICUBIC)

        bicubic = lr.resize((lr.width * self.scale, lr.height * self.scale), resample=pil_image.BICUBIC)
        bicubic.save(self.test_image_path.replace(".", f"_bicubic_x{self.scale}."))

        lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        hr = np.expand_dims(np.array(hr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0

        low_res = torch.from_numpy(lr).to(self.device)
        high_res = torch.from_numpy(hr).to(self.device)

        _logger.info(f"Evaluation of Super Resolution model started")

        with torch.no_grad():
            output = self.net(low_res)

        output_y = rgb_to_y(denormalize_image(output.squeeze(0)), layout="chw")
        high_res_y = rgb_to_y(denormalize_image(high_res.squeeze(0)), layout="chw")

        output_y = output_y[self.scale:-self.scale, self.scale:-self.scale]
        high_res_y = high_res_y[self.scale:-self.scale, self.scale:-self.scale]

        psnr_value = psnr(high_res_y, output_y)
        _logger.info(f"Evaluation PSNR: {psnr_value:.2f}")

        img_npy = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        output_img = pil_image.fromarray(denormalize_image(img_npy))
        output_img.save(self.test_image_path.replace(".", f"_output_x{self.scale}."))

        _logger.info(f"Evaluation of Super Resolution model finished")

    def save(self, folder_name: str, file_name: str) -> None:
        """ Saving all model parameters, hyper-parameters and optimizer data. """

        # Creating folder
        if not os.path.exists(os.path.join(os.getcwd(), f"experiments/{folder_name}")):
            os.mkdir(os.path.join(os.getcwd(), f"experiments/{folder_name}"))

        # Saving hyper-parameters
        hyper_params_path = os.path.join(os.getcwd(), f"experiments/{folder_name}/hyper_parameters.json")
        if not os.path.exists(hyper_params_path):
            self.hyper_parameters.save_params(file_path=hyper_params_path)

        # Saving net state, optimizer state and loss value
        model_path = os.path.join(os.getcwd(), f"experiments/{folder_name}/{file_name}")
        torch.save({"model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()}, model_path)

        _logger.info(f"Saved all model and optimizer parameters on the path: {model_path}")
