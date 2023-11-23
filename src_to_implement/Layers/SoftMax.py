import numpy as np


class BaseLayer:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):

        raise NotImplementedError("Forward pass must be implemented in derived classes.")

    def backward(self, error_tensor):

        raise NotImplementedError("Backward pass must be implemented in derived classes.")


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.probabilities = None

    def forward(self, input_tensor):
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=-1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.probabilities = probabilities
        return probabilities

    def backward(self, error_tensor):
        # Get the number of samples in the batch
        batch_size = error_tensor.shape[0]

        # Compute the derivative of softmax with cross-entropy loss
        gradients = self.probabilities.copy()

        for i in range(batch_size):
            # Compute the inner summation term
            inner_sum = np.sum(error_tensor[i] * self.probabilities[i])
            
            # Apply the formula for each sample
            gradients[i] = self.probabilities[i] * (error_tensor[i] - inner_sum)

        return gradients