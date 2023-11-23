import unittest
from Layers import *
from Optimization import *
import numpy as np
import NeuralNetwork
import matplotlib.pyplot as plt
import tabulate
import argparse


input_size = 4
batch_size = 9
output_size = 3
input_tensor = np.random.rand(batch_size, input_size)
layer = FullyConnected.FullyConnected(input_size, output_size)
layer.forward(input_tensor)