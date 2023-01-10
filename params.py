import json
import logger as logger
import os
from typing import Dict, Union, Any

_logger = logger.get_logger(__name__)


class SuperResolutionParams:

    def __init__(self, params_path: str) -> None:
        self.params = None
        self.params_path = params_path

        self._read_params()

    def _read_params(self) -> None:
        if os.path.exists(self.params_path):
            with open(self.params_path, mode="r", encoding="utf-8") as file:
                self.params = json.load(file)

                if not isinstance(self.params, dict):
                    _logger.error("Hyper-parameters are not properly read!")

    def __str__(self) -> str:

        if isinstance(self.params, dict):
            return str(self.params)

    def get_params(self) -> Dict[str, Union[int, float]]:
        return self.params

    def get_param(self, key: str) -> Union[int, float]:
        if key in self.params.keys():
            return self.params[key]

    def set_param(self, key: str, value: Any):
        if key in self.params.keys():
            self.params[key] = value

    def add_param(self, key: str, value: Any):
        self.params[key] = value

    def save_params(self, file_path: str) -> None:

        with open(file_path, mode="w", encoding="utf-8") as file:
            json.dump(self.params, file, indent=2)

        _logger.info(f"Experiment hyper-parameters stored on the path: {file_path}")
