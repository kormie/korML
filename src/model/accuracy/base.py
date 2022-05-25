from typing import Union

import numpy as np
from korML import BoolArray, FloatArray


class Accuracy:

    comparisons: FloatArray
    precision: Union[FloatArray, None]

    def __init__(self) -> None:
        raise NotImplementedError()

    def init(self, y, reinit=False):
        raise NotImplementedError()

    # Calculates an accuracy
    # given predictions and ground truth values
    def calculate(self, predictions, y):
        # Get comparison results
        comparisons = self.compare(predictions, y)
        # Calculate an accuracy
        accuracy = np.mean(comparisons)
        # Return accuracy
        return accuracy

    def compare(self, predictions: FloatArray, y: FloatArray) -> BoolArray:
        raise NotImplementedError()
