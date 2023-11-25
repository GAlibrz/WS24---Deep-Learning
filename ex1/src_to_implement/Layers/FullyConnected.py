import numpy as np
from Optimization import Optimizers
from .Base import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.weights = np.random.uniform(low = 0.0, high = 1.0, size = (self.input_size + 1, self.output_size))
        self.input = None
        self._optimizer = None
        self._gradient_weights = None
        print(self.weights.shape)


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor):
        #print(self.trainable)
        tmp = np.ones((input_tensor.shape[0], 1))
        self.input = np.concatenate((input_tensor, tmp), axis= 1)
        result = np.dot(self.input, self.weights)
        return result
    
    def backward(self, error_tensor):
        gradient_input = np.dot(error_tensor, np.transpose(self.weights))
        self._gradient_weights = np.dot(np.transpose(self.input), error_tensor)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return gradient_input[ : , :self.input_size]