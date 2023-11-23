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
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        e_x = np.exp(input_tensor - np.max(input_tensor))
        return e_x / e_x.sum(axis=0) # only difference

    def backward(self, error_tensor):
        s = self.input_tensor.reshape(-1, 1)
        # Compute the Jacobian matrix of the softmax function
        jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
        # Compute the product of the Jacobian matrix and the error tensor
        return np.dot(jacobian_matrix, error_tensor)