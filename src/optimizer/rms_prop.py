import numpy as np
from korML.layer.base import Layer
from korML.optimizer.base import Optimizer


class RMSProp(Optimizer):
    epsilon: float
    rho: float

    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9) -> None:
        super().__init__(learning_rate=learning_rate, decay=decay)
        self.epsilon = epsilon
        self.rho = rho

    def update_params(self, layer: Layer) -> Layer:
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)
        return layer


if __name__ == '__main__':
    import numpy as np
    import nnfs  # type: ignore
    from nnfs.datasets import spiral_data  # type: ignore
    from korML.layer import DenseLayer
    from korML.activation import ReluActivation
    from korML.classification import SoftCat
    from korML.optimizer import SGD

    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    dense1 = DenseLayer(2, 64)

    activation1 = ReluActivation()

    dense2 = DenseLayer(64, 3)

    loss_activation = SoftCat()

    optimizer: Optimizer
    optimizer = RMSProp(learning_rate=0.02, decay=1e-5, rho=0.999)

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
