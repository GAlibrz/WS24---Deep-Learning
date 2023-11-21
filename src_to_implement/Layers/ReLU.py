from Base import BaseLayer
import tensorflow as tf


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return tf.nn.relu(input_tensor)

    def backward(self, error_tensor):
        return tf.multiply(tf.cast(tf.greater(self.input_tensor, 0), tf.float32), error_tensor)
