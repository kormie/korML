import numpy as np
from korML import FloatArray
from korML.loss.base import Loss


class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred: FloatArray, y_true: FloatArray) -> FloatArray:
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1)

        negative_log_liklihoods = -np.log(correct_confidences)

        return negative_log_liklihoods

    def backward(self, dvalues: FloatArray, y_true: FloatArray) -> FloatArray:
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples

        return self.dinputs


# if __name__ == '__main__':
#     import nnfs  # type: ignore
#     from nnfs.datasets import spiral_data  # type: ignore
#     from korML.layer import DenseLayer
#     from korML.activation import ReluActivation
#     from korML.activation import SoftMaxActivation

#     nnfs.init()
#     X, y = spiral_data(samples=100, classes=3)

#     dense1 = DenseLayer(2, 3)

#     activation1 = ReluActivation()

#     dense2 = DenseLayer(3, 3)

#     activation2 = SoftMaxActivation()

#     loss_function = CategoricalCrossEntropy()

#     dense1.forward(X)

#     activation1.forward(dense1.output)

#     dense2.forward(activation1.output)

#     activation2.forward(dense2.output)

#     print(activation2.output[:5])

#     loss = loss_function.calculate(activation2.output, y)

#     print('loss:', loss)

#     predictions = np.argmax(activation2.output, axis=1)
#     if len(y.shape) == 2:
#         y = np.argmax(y, axis=1)
#     accuracy = np.mean(predictions == y)

#     print('acc:', accuracy)
