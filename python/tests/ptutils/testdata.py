from typing import Iterator

import numpy as np

from tests.ptutils.testmodel import SampleDatapoint


class SyntheticData:
    def __init__(self, num_features: int):
        self.__num_features = num_features
        self.__weights = np.random.randn(num_features) * 10

    def generate(self, num_points: int) -> Iterator[SampleDatapoint]:
        for _ in range(num_points):
            # Avoid bias so that there is no obvious class imbalance.
            input_features = np.random.randn(self.__num_features) * 5
            yield SampleDatapoint(
                input_features=list(input_features),
                target_class=sum(input_features * self.__weights) >= 0
            )