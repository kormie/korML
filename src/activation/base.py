from korML import FloatArray


class Activation:

    inputs: FloatArray
    output: FloatArray

    dinputs: FloatArray

    def predictions(self, outputs: FloatArray) -> FloatArray:
        raise NotImplementedError

    def forward(self, inputs: FloatArray, training: bool) -> FloatArray:
        raise NotImplementedError

    def backward(self, dvalues: FloatArray) -> FloatArray:
        raise NotImplementedError
