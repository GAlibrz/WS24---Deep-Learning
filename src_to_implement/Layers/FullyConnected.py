import numpy as np
import tensorflow as tf
from Base import BaseLayer

class FullyConnected(BaseLayer):


    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # + 1 is added here to represent for the biases, additional care is needed therefor in forward and bachward paths
        self.weights = np.random.rand(output_size, input_size)
        self.biases = np.zeros((output_size, 1))
        
        self._optimizer = None

    def forward(self, input_tensor):
        weights_transpose = np.transpose(self.weights)
        result_without_biases = tf.matmul(weights_transpose, input_tensor)
        result_with_bias = tf.add(result_without_biases, self.biases)
        return result_without_biases

        @property
        def optimizer(self):
            return self._optimizer
        
        @optimizer.setter
        def optimizer(self, optimizer):
            self._optimizer = optimizer
            return self._optimizer
        
