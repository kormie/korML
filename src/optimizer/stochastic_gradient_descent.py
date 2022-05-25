from korML.layer.base import Layer
import numpy as np

from korML.optimizer.base import Optimizer


class StochasticGradientDescent(Optimizer):
    momentum: float

    def __init__(self, learning_rate=1., decay=0., momentum=0.) -> None:
        super().__init__(learning_rate=learning_rate, decay=decay)
        self.momentum = momentum

    def update_params(self, layer: Layer) -> Layer:
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates
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
    optimizer = StochasticGradientDescent(decay=8e-8, momentum=0.9)

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
