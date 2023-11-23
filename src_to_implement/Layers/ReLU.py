import numpy as np

class BaseLayer:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):

        raise NotImplementedError("Forward pass must be implemented in derived classes.")

    def backward(self, error_tensor):

        raise NotImplementedError("Backward pass must be implemented in derived classes.")


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0)