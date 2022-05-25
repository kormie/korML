from typing import Tuple, Union
import numpy as np
from korML import FloatArray
from korML.activation import SoftMaxActivation
from korML.loss import CatXEntropyLoss


class SoftMaxXCategoricalCrossEntropy():
    activation: SoftMaxActivation
    loss: CatXEntropyLoss
    output: FloatArray

    dinputs: FloatArray

    def backward(self, dvalues: FloatArray, y_true: FloatArray) -> FloatArray:
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        return self.dinputs


# if __name__ == '__main__':
#     import nnfs  # type: ignore
#     from nnfs.datasets import spiral_data  # type: ignore
#     from korML.layer import DenseLayer
#     from korML.activation import ReluActivation

#     nnfs.init()

#     X, y = spiral_data(samples=100, classes=3)
#     dense1 = DenseLayer(2, 3)
#     activation1 = ReluActivation()
#     dense2 = DenseLayer(3, 3)
#     loss_activation = SoftMaxXCategoricalCrossEntropy()

#     dense1.forward(X)
#     activation1.forward(dense1.output)
#     dense2.forward(activation1.output)

#     loss = loss_activation.forward(dense2.output, y)
#     print(loss_activation.output[:5])
#     print('loss:', loss)

#     predictions = np.argmax(loss_activation.output, axis=1)
#     if len(y.shape) == 2:
#         y = np.argmax(y, axis=1)
#     accuracy = np.mean(predictions == y)
#     print('accuracy:', accuracy)

#     loss_activation.backward(loss_activation.output, y)
#     dense2.backward(loss_activation.dinputs)
#     activation1.backward(dense2.dinputs)
#     dense1.backward(activation1.dinputs)
#     print(dense1.dweights)
#     print(dense1.dbiases)
#     print(dense2.dweights)
#     print(dense2.dbiases)
