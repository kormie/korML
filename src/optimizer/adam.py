from korML.layer.base import Layer
from korML.optimizer.base import Optimizer
import numpy as np


class Adam(Optimizer):
    beta1: float
    beta2: float
    epsilon: float

    def __init__(self, learning_rate=1e-3, decay=0., epsilon=1e-7, beta1=0.9, beta2=0.999) -> None:
        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def update_params(self, layer: Layer) -> Layer:
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * \
            layer.weight_momentums + \
            (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * \
            layer.bias_momentums + \
            (1 - self.beta1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass
        # and we need to start with 1 here
        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta1 ** (self.iterations + 1))
        # Update cache with squared current gradients
        layer.weight_cache = self.beta2 * layer.weight_cache + \
            (1 - self.beta2) * layer.dweights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + \
            (1 - self.beta2) * layer.dbiases**2
        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
            weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) +
             self.epsilon)
        layer.biases += -self.current_learning_rate * \
            bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) +
             self.epsilon)

        return layer


if __name__ == '__main__':
    import nnfs  # type: ignore
    from nnfs.datasets import spiral_data  # type: ignore
    from korML.layer import DenseLayer
    from korML.activation import ReluActivation
    from korML.classification import SoftCat

    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    dense1 = DenseLayer(2, 64)

    activation1 = ReluActivation()

    dense2 = DenseLayer(64, 3)

    loss_activation = SoftCat()

    optimizer: Optimizer
    optimizer = Adam(learning_rate=0.05, decay=5e-7)

    for epoc in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoc % 100:
            print(f'epoc: {epoc}, ' +
                  f'accuracy: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')

        # Backpropagation
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update parameters
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()
