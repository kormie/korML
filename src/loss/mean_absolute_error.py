import numpy as np
from korML.loss.base import Loss
from korML import FloatArray


class MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true) -> FloatArray:
        # Calculate loss
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses

    # Backward pass
    def backward(self, dvalues, y_true) -> FloatArray:
        # Number of samples
        samples = len(dvalues)
        # Number of outputs in every sample
        # We'll use the first sample to count them
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs
        # Normalize gradient
        self.dinputs = self.dinputs / samples
        return self.dinputs
