from typing import NamedTuple, List, Optional, Dict, Any

import numpy as np
import torch
from torch import nn as nn

from dpu_utils.ptutils import BaseComponent


class SampleDatapoint(NamedTuple):
    input_features: List[float]
    target_class: bool


class TensorizedDatapoint(NamedTuple):
    input_features: np.ndarray
    target_class: np.ndarray


class SimpleRegression(BaseComponent[SampleDatapoint, TensorizedDatapoint]):
    """A simple linear regression model used for testing."""
    def __init__(self, name, num_features: int, hyperparameters: Optional[Dict[str, Any]] = None):
        super(SimpleRegression, self).__init__(name, hyperparameters)
        self.__num_features = num_features

    @classmethod
    def default_hyperparameters(cls) -> Dict[str, Any]:
        return {}

    def _load_metadata_from_sample(self, data_to_load: SampleDatapoint) -> None:
        pass  # No metadata in this simple model.

    def _finalize_component_metadata_and_model(self) -> None:
        self.__layer = nn.Linear(self.__num_features, 1, bias=False)
        self.__bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))  # Use a separate bias to allow freezing the weights.
        self.__loss = nn.BCEWithLogitsLoss()

    def load_data_from_sample(self, data_to_load: SampleDatapoint) -> Optional[TensorizedDatapoint]:
        return TensorizedDatapoint(
            input_features=np.array(data_to_load.input_features, dtype=np.float32),
            target_class=np.array(1 if data_to_load.target_class else 0, dtype=np.float32)
        )

    def initialize_minibatch(self) -> Dict[str, Any]:
        return {
            'inputs': [],
            'targets': []
        }

    def extend_minibatch_by_sample(self, datapoint: TensorizedDatapoint, accumulated_minibatch_data: Dict[str, Any]) -> bool:
        accumulated_minibatch_data['inputs'].append(datapoint.input_features)
        accumulated_minibatch_data['targets'].append(datapoint.target_class)
        return True

    def finalize_minibatch(self, accumulated_minibatch_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'inputs': torch.tensor(np.stack(accumulated_minibatch_data['inputs'], axis=0), device=self.device),
            'targets': torch.tensor(np.stack(accumulated_minibatch_data['targets'], axis=0), device=self.device)
        }

    def predict(self, inputs: torch.Tensor):
        predicted = self.__layer(inputs)[:, 0] + self.__bias  # B
        return predicted >= 0

    def forward(self, inputs, targets):
        predicted = self.__layer(inputs)[:, 0] + self.__bias  # B
        loss = self.__loss(input=predicted, target=targets)
        return loss