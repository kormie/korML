from korML.activation.base import Activation


class Linear(Activation):

    def predictions(self, outputs):
        return outputs

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
