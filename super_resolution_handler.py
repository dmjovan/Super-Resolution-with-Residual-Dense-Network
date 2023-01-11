import os
import time
from typing import Optional

import cv2
import numpy as np
import torch
from torch import optim

import logger as logger
from dataset import SuperResolutionDataset
from image_utils import downscale_image, upscale_image
from loss import CombinedLoss
from network import SuperResolutionNetwork
from params import SuperResolutionParams

_logger = logger.get_logger(__name__)

time_format = "%d_%m_%Y_%H_%M_%S"


class SuperResolutionHandler:

    def __init__(self, mode: Optional[str], model_path: Optional[str], hyper_params_path: Optional[str],
                 test_image_path: Optional[str]) -> None:
        """ Creating SuperResolutionHandler object and validating provided arguments. """

        # Arguments
        self.mode = mode
        self.model_path = model_path
        self.hyper_params_path = hyper_params_path
        self.test_image_path = test_image_path
        self._validate_arguments()

        # Dataset
        self.dataset_path = os.path.join(os.getcwd(), "images")
        self.dataset = SuperResolutionDataset(self.dataset_path)
        _logger.info(f"Dataset loaded from the path: {self.dataset_path}")

        if self.mode == "train":
            _logger.info(str(self.dataset))

        # Hyper-parameters
        self.hyper_parameters = SuperResolutionParams(self.hyper_params_path)
        _logger.info(f"Hyper-parameters loaded from the path: {self.hyper_params_path}")

        if self.mode == "train":
            _logger.info(str(self.hyper_parameters))

        # Device
        self.device = torch.device("cpu")

        if self.mode == "train" and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            torch.cuda.empty_cache()
            _logger.info(f"Device set to: {torch.cuda.get_device_name(self.device)}")
        else:
            _logger.info("Device set to: CPU")

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
            self.model_path = os.path.join(os.getcwd(), "models/best_model.pt")
            _logger.warning(
                "Program argument 'model_path' you specified does not exist. Using best model from project.")

        if not self.hyper_params_path or not os.path.exists(self.hyper_params_path):
            self.hyper_params_path = os.path.join(os.getcwd(), "experiments/default_hyper_params.json")
            _logger.warning(
                "Program argument 'hyper_params_path' you specified does not exist. Using default hyper-parameters.")

    def run(self) -> None:
        """ Running SuperResolutionHandler """

        _logger.info("Running Super Resolution ...")

        if self.mode == "train":
            self.train()

        elif self.mode == "test":
            self.test()

    def np_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        return torch.unsqueeze(torch.permute(torch.Tensor(img), (2, 1, 0)), 0).to(self.device)

    def tensor_to_np(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.permute(torch.squeeze(tensor), (2, 1, 0)).detach().numpy().astype(np.uint8)

    def train(self) -> None:
        """ Training SuperResolutionNetwork """

        def decay_learning_rate() -> None:
            """ Decaying learning rate - after each epochs"""

            last_lr = self.hyper_parameters.get_param("lr")
            new_lr = last_lr / 5

            self.hyper_parameters.set_param("lr", value=new_lr)

            self.optimizer = optim.Adam(self.net.parameters(), lr=new_lr)
            _logger.info(f"Optimizer learning rate successfully decayed to {new_lr}")

        # Loss to minimize
        loss_fn = CombinedLoss()
        self.hyper_parameters.add_param(key="loss", value=loss_fn.name)

        start_time = time.strftime(time_format)
        _logger.info(f"Training started at: {start_time}")

        self.net.train()

        for epoch in range(self.hyper_parameters.get_param("num_epochs")):

            # Decaying learning rate
            if epoch > 0 and epoch % 40 == 0:  # FIXME
                decay_learning_rate()

            running_loss = 0.0
            for _ in range(self.hyper_parameters.get_param("num_batches_per_epoch")):

                # Getting batch from dataset
                batch = self.dataset.get_train_batch(batch_size=self.hyper_parameters.get_param("batch_size"),
                                                     crop_size=self.hyper_parameters.get_param("crop_size"))

                # Running training network on each image from batch
                for img in batch:
                    # Zeroing gradients
                    self.optimizer.zero_grad()

                    # Downscaling image for smaller resolution
                    input_image = self.np_to_tensor(downscale_image(img))

                    # Storing normal resolution image
                    target = self.np_to_tensor(img)

                    # Gathering output from SuperResolutionNetwork for downscaled input image
                    output = self.net(input_image)

                    # Calculating loss function
                    loss = loss_fn(output, target)

                    # Calculating gradients while doing backpropagation
                    loss.backward()

                    # Applying gradients on parameters
                    self.optimizer.step()

                    # Calculating running loss
                    running_loss += loss.item() / (self.hyper_parameters.get_param(
                        "num_batches_per_epoch") * self.hyper_parameters.get_param("batch_size"))

            _logger.info(f"Finished epoch {epoch} at {time.strftime(time_format)}. Current loss: {running_loss}")

            # Storing parameters
            if epoch % 10 == 0:
                self.save(folder_name=f"models_{start_time}", file_name=f"model_{epoch}.pt", running_loss=running_loss)

        _logger.info(f"Training finished at: {time.strftime(time_format)}")

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
            test_image = cv2.imread(self.test_image_path)
            _logger.info(f"Test image loaded from the path: {self.test_image_path}")
        else:
            test_image = self.dataset.get_random_test_image()
            _logger.info(f"Test image loaded from dataset")

        h, w = test_image.shape[0:2]
        _logger.info(f"Test image resolution [H x W]:  {h} x {w}")

        _logger.info(f"Evaluation of Super Resolution model started")

        with torch.no_grad():
            input_image = downscale_image(test_image)
            upscaled_image = upscale_image(input_image)
            input_image = self.np_to_tensor(input_image)
            output = self.net(input_image)
            output = self.tensor_to_np(output)

            _logger.info(f"Evaluation of Super Resolution model finished")

            # TODO: add better representation and saving of images

            cv2.imwrite("output.png", output)
            cv2.imwrite("target.png", test_image)
            cv2.imwrite("input.png", upscaled_image)

    def save(self, folder_name: str, file_name: str, running_loss: float) -> None:
        """ Saving all model parameters, hyper-parameters, optimizer data and loss. """

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
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": running_loss}, model_path)

        _logger.info(f"Saved all model and optimizer parameters on the path: {model_path}")
