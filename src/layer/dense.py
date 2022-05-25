import numpy as np

from korML import FloatArray
from korML.layer.base import Layer


class Dense(Layer):

    def forward(self, inputs, training) -> FloatArray:
        self.output = np.dot(inputs, self.weights) + self.biases
        self.inputs = inputs
        return self.output

    def backward(self, dvalues: FloatArray):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * \
                self.weights

        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * \
                self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)


if __name__ == '__main__':
    import nnfs  # type: ignore
    from nnfs.datasets import spiral_data  # type: ignore

    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    dense1 = Dense(2, 3)

    dense1.forward(X)

    print(dense1.output[:5])
