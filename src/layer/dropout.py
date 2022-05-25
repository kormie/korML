import numpy as np
from korML import FloatArray
from korML.layer.base import Layer


class Dropout(Layer):
    rate: float
    binary_mask: FloatArray

    def __init__(self, rate) -> None:
        self.rate = 1 - rate

    def forward(self, inputs, training) -> FloatArray:
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return self.output

        self.binary_mask = np.random.binomial(
            1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
        return self.output

    def backward(self, dvalues: FloatArray):
        self.dinputs = dvalues * self.binary_mask
