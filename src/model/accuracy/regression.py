import numpy as np
from korML import FloatArray, BoolArray
from korML.model.accuracy.base import Accuracy


class Regression(Accuracy):

    def __init__(self) -> None:
        self.precision = None

    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def calculate(self, predictions: FloatArray, y: FloatArray) -> FloatArray:
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        return accuracy

    def compare(self, predictions: FloatArray, y: FloatArray) -> BoolArray:
        return np.absolute(predictions - y) < self.precision
