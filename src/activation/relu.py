import numpy as np

from korML.activation.base import Activation


class ReLU(Activation):

    def predictions(self, outputs):
        return outputs

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
        return self.dinputs


# if __name__ == '__main__':
#     import nnfs  # type: ignore
#     from nnfs.datasets import spiral_data  # type: ignore
#     from korML.layer import DenseLayer

#     nnfs.init()

#     X, y = spiral_data(samples=100, classes=3)

#     dense1 = DenseLayer(2, 3)

#     activation1 = ReLU()

#     dense1.forward(X)

#     activation1.forward(dense1.output)

#     print(activation1.output[:5])
