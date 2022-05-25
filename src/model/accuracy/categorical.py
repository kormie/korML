import numpy as np
from .base import Accuracy


class Categorical(Accuracy):
    binary: bool

    def __init__(self, *, binary=False) -> None:
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
