import numpy as np
import tensorflow as tf
from Base import BaseLayer
from Optimization import Optimizers

class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.input = None

        # Use tf.random.normal for weight initialization
        self.weights_no_biases = tf.random.normal((input_size, output_size))
        self.biases = tf.zeros((output_size,))
        self.weights = tf.concat([self.weights_no_biases, self.biases], axis=0)
        self._optimizer = None

    def forward(self, input_tensor):
        self.input = input_tensor
        result_without_biases = tf.matmul(input_tensor, self.weights)
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
        self.gradient_weights_without_biases = tf.matmul(tf.transpose(self.input), error_tensor)
        self.biases_gradient = tf.reduce_sum(error_tensor, axis=0)

        # Concatenate weights and biases into one matrix
        self._gradient_weights = tf.concat([self.gradient_weights_without_biases, self.biases_gradient], axis=0)

        # Assuming self._optimizer is an instance of the Optimizers class
        self.weights = self._optimizer.calculate_update(self.weights_and_biases, self._gradient_weights)
        self.weights_no_biases = self.weights_and_biases[:-1, :]
        self.biases = self.weights_and_biases[-1, :]