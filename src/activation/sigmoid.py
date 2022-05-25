import numpy as np
from korML.activation.base import Activation


class Sigmoid(Activation):

    def predictions(self, outputs):
        return (outputs > 0.5) * 1

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
