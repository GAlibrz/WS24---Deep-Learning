import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.prediction_store = None

    def forward(self, input_tensor):
        # return estimated class probabilities
        output_tensor = np.zeros_like(input_tensor)
        for i in range(input_tensor.shape[0]):  # batch size
            curr = input_tensor[i, :] - np.max(input_tensor[i, :])
            output_tensor[i, :] = np.exp(curr) / np.sum(np.exp(curr))
        self.prediction_store = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        error_n = np.zeros_like(error_tensor)
        for i in range(error_tensor.shape[0]):  # batch size
            y_hat = self.prediction_store[i, :]
            e_n = error_tensor[i, :]
            error_n[i, :] = np.multiply(y_hat, e_n - np.dot(e_n, y_hat))
        return error_n