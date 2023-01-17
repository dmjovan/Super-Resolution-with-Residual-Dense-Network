import json
import os
from typing import Dict, Union, Any

import logger as logger

_logger = logger.get_logger(__name__)


class SuperResolutionParams:

    def __init__(self, params_path: str) -> None:
        """ Creating object that handles hyper-parameters of Super Resolution task """

        self.params = None
        self.params_path = params_path

        # Reading JSON file containing hyper-parameters
        self._read_params()

        if "scale" not in self.params or ("scale" in self.params and self.params["scale"] != 2):
            RuntimeError(f"Scale factor not supported. Only scale factor of 2 is supported.")

    def _read_params(self) -> None:
        """ Reading hyper-parameters from JSON file. """

        if os.path.exists(self.params_path):
            with open(self.params_path, mode="r", encoding="utf-8") as file:
                self.params = json.load(file)

                if not isinstance(self.params, dict):
                    RuntimeError("Hyper-parameters are not properly read!")

    def __str__(self) -> str:

        params_string = ""

        if isinstance(self.params, dict) and len(self.params):
            params_string = "Super Resolution params review: \n" \
                            "############################################"
            for key, value in self.params.items():
                params_string += f"\n {key}: {value}"

            params_string += "\n############################################"

        return params_string

    def get_params(self) -> Dict[str, Union[int, float]]:
        """ Getting all hyper-parameters as a Python dictionary """

        return self.params

    def get_param(self, key: str) -> Union[int, float]:
        """ Getting hyper-parameter with provided name (key) """

        if key in self.params.keys():
            return self.params[key]
        else:
            raise KeyError(f"Getting hyper-parameter value failed. There is no hyper-parameter with name {key}")

    def set_param(self, key: str, value: Any):
        """ Setting already existing hyper-parameter """

        if key in self.params.keys():
            self.params[key] = value
        else:
            raise KeyError(f"Setting hyper-parameter value failed. There is no hyper-parameter with name {key}")

    def add_param(self, key: str, value: Any):
        """ Adding new hyper-parameter """

        if key not in self.params.keys():
            self.params[key] = value
        else:
            raise KeyError(f"Hyper-paremeter with name {key} already exists.")

    def save_params(self, file_path: str) -> None:
        """ Saving parameters into provided path, as a JSON file """

        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(self.params, file, indent=2)

        _logger.info(f"Experiment hyper-parameters stored on the path: {file_path}")
