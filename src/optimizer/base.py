import numpy as np
from korML import FloatArray
from korML.layer.base import Layer

# from enum import Enum, unique


class LayerMemory():
    layer: Layer

    def __init__(self, layer: Layer) -> None:
        self.layer = layer


class MomentumMemory(LayerMemory):
    weights: FloatArray
    bias: FloatArray

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer)
        self.weights = np.zeros_like(layer.weights)
        self.bias = np.zeros_like(layer.biases)


class CacheMemory(LayerMemory):
    weights: FloatArray
    bias: FloatArray

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer)
        self.weights = np.zeros_like(layer.weights)
        self.bias = np.zeros_like(layer.biases)


class CacheAndMomentumMemory(LayerMemory):
    cache: CacheMemory
    momentum: MomentumMemory

    def __init__(self, layer: Layer) -> None:
        super().__init__(layer)
        self.cache = CacheMemory(layer)
        self.momentum = MomentumMemory(layer)


class Optimizer:
    learning_rate: float
    decay: float
    current_learning_rate: float
    iterations: int

    def __init__(self, learning_rate=1., decay=0.) -> None:
        self.learning_rate = learning_rate
        self.decay = decay
        self.current_learning_rate = learning_rate
        self.iterations = 0

    def initialize_layer_memory(self, layer: Layer) -> LayerMemory:
        raise NotImplementedError

    def pre_update_params(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer: Layer) -> Layer:
        raise NotImplementedError

    def post_update_params(self) -> None:
        self.iterations += 1
