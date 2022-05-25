from korML import FloatArray


class Input:
    output: FloatArray

    def forward(self, inputs: FloatArray, training) -> FloatArray:
        self.output = inputs
        return self.output
