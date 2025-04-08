import sys
import os

# Insert the 'app' directory into sys.path if it's not already there
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app_dir = os.path.join(project_root, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)


import unittest
import torch
from CNNModelMNIST import CNNModel
from utils import preprocess_image


class TestCNNModel(unittest.TestCase):
    def setUp(self):
        self.model = CNNModel()
        self.model.eval()

    def test_forward_output_shape(self):
        dummy_input = torch.randn(1, 1, 28, 28)  # MNIST shape
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 10), "Model output should be 1x10 for MNIST classification")
