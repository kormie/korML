import numpy as np

from korML.activation.base import Activation


class SoftMax(Activation):
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):

            # Flatten output array
            single_output = single_output.reshape(-1, 1)

            # Calculate Jacobian matrix
            jacobian_matrix = np.diagflat(single_output) - \
                np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

        return self.dinputs


# if __name__ == '__main__':
#     from nnfs.datasets import spiral_data  # type: ignore
#     from korML.layer import DenseLayer
#     from korML.activation import ReluActivation
#     X, y = spiral_data(samples=100, classes=3)

#     dense1 = DenseLayer(2, 3)
#     activation1 = ReluActivation()

#     dense2 = DenseLayer(3, 3)
#     activation2 = SoftMax()

#     dense1.forward(X)
#     activation1.forward(dense1.output)

#     dense2.forward(activation1.output)

#     activation2.forward(dense2.output)

#     print(activation2.output[:5])
