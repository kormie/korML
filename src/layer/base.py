import numpy as np
from korML import FloatArray


class Layer:
    weights: FloatArray
    biases: FloatArray
    output: FloatArray
    inputs: FloatArray
    weight_regularizer_l1: float
    weight_regularizer_l2: float
    bias_regularizer_l1: float
    bias_regularizer_l2: float

    dweights: FloatArray
    dbiases: FloatArray
    dinputs: FloatArray

    weight_momentums: FloatArray
    bias_momentums: FloatArray

    weight_cache: FloatArray
    bias_cache: FloatArray

    def __init__(self, n_inputs: int, n_neurons: int,
                 weight_regularizer_l1: float = 0, weight_regularizer_l2: float = 0,
                 bias_regularizer_l1: float = 0, bias_regularizer_l2: float = 0) -> None:
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs: FloatArray, training: bool) -> FloatArray:
        raise NotImplementedError

    def backward(self, dvalues: FloatArray):
        raise NotImplementedError
