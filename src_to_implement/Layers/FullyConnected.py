import numpy as np
from Optimization import Optimizers
from .Base import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.input = None

        # Use tf.random.normal for weight initialization
        self.weights_no_biases = np.random.normal(size = (input_size, output_size))
        #print(input_size, '  ', output_size, self.weights_no_biases.shape)
        self.biases = np.random.normal(size = (1, output_size))
        self.weights = np.concatenate([self.weights_no_biases, self.biases], axis=0)
        self._optimizer = None
        print(self.weights.shape)

    def forward(self, input_tensor):
        #print(self.trainable)
        self.input = input_tensor
        result_without_biases = np.dot(input_tensor, self.weights_no_biases)
        result_with_biases = result_without_biases + self.biases
        return result_with_biases

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def backward(self, error_tensor):
        self.gradient_weights_without_biases = np.dot(np.transpose(self.input), error_tensor)
        self.biases_gradient = np.sum(error_tensor, axis=0)

        # Concatenate weights and biases into one matrix
        self._gradient_weights = np.concatenate([self.gradient_weights_without_biases, self.biases_gradient], axis=0)

        # Assuming self._optimizer is an instance of the Optimizers class
        self.weights = self._optimizer.calculate_update(self.weights_and_biases, self._gradient_weights)
        self.weights_no_biases = self.weights_and_biases[:-1, :]
        self.biases = self.weights_and_biases[-1, :]