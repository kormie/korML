from korML import FloatArray
from korML.layer.base import Layer
from korML.loss.base import Loss
from korML.loss import *
from korML.model import ModelAccuracy
from korML.optimizer.base import Optimizer
from korML.layer import InputLayer
from korML.activation import *
from korML.classification import SoftCat


class Model:

    loss: Loss
    optimizer: Optimizer
    input_layer: InputLayer
    output_layer_activation: Layer
    accuracy: ModelAccuracy

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None

    def add(self, layer) -> list[Layer]:
        self.layers.append(layer)
        return self.layers

    def set(self, *, loss: Loss, optimizer: Optimizer, accuracy: ModelAccuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            output = self.forward(X, training=True)

            data_loss, regularization_loss = self.loss.calculate(
                output, y, include_regularization=True)
            loss = data_loss + regularization_loss

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'reg_loss: {regularization_loss:.3f}), ' + f'lr: {self.optimizer.current_learning_rate}')

        if validation_data is not None:
            X_val, y_val = validation_data
            output = self.forward(X_val, training=False)
            loss = self.loss.calculate(output, y)
            predictions = self.output_layer_activation.predictions(
                output)
            accuracy = self.accuracy.calculate(predictions, y_val)
            print(f'validation, ' +
                  f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')

    def finalize(self):
        self.input_layer = InputLayer()
        layer_count = len(self.layers)

        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], SoftMaxActivation) and isinstance(self.loss, CatXEntropyLoss):
            self.softmax_classifier_output = SoftCat()

    def forward(self, X: FloatArray, training: bool) -> FloatArray:
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):

        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)

            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
