import unittest
from Layers import *
from Optimization import *
import numpy as np
import NeuralNetwork
import argparse


input_size = 4
batch_size = 9
output_size = 3
input_tensor = np.random.rand(batch_size, input_size)
layer = FullyConnected.FullyConnected(input_size, output_size)
output_tensor = layer.forward(input_tensor)
print("backward")
backward = layer.backward(output_tensor)