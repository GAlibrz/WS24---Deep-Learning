import tensorflow as tf
import numpy as np
from Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        exp_values = np.exp(input_tensor - np.max(input_tensor, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]

        jacobian_matrix = np.array([np.diag(p) - np.outer(p, p) for p in self.probabilities])

        grad_input = np.array([np.dot(jacobian_matrix[i], error_tensor[i]) for i in range(batch_size)])

        return grad_input