import tensorflow as tf
from Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return tf.nn.softmax(input_tensor)

    def backward(self, error_tensor):
        return tf.multiply(tf.cast(tf.greater(self.input_tensor, 0), tf.float32), error_tensor)