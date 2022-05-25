from typing import List, Tuple, Union
import numpy as np
from korML.layer.base import Layer
from korML import FloatArray


class Loss:
    dinputs: FloatArray
    trainable_layers: List[Layer]

    def remember_trainable_layers(self, trainable_layers: List[Layer]) -> None:
        self.trainable_layers = trainable_layers

    def calculate(self, output: FloatArray, y, *, include_regularization=False) -> Union[float, Tuple[float, float]]:
        sample_loss = self.forward(output, y)

        data_loss = np.mean(sample_loss)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def forward(self, output: FloatArray, y) -> FloatArray:
        raise NotImplementedError

    def backward(self, dvalues: FloatArray, y_true: FloatArray) -> FloatArray:
        raise NotImplementedError

    def regularization_loss(self) -> float:
        regularization_loss = 0

        # L1 regularization - weights
        # calculate only when factor greater than 0
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                    np.sum(np.abs(layer.weights))

            # L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                    np.sum(layer.weights *
                           layer.weights)

            # L1 regularization - biases
            # calculate only when factor greater than 0
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                    np.sum(np.abs(layer.biases))

            # L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                    np.sum(layer.biases *
                           layer.biases)
        return regularization_loss
