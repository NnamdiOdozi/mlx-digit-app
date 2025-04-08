import unittest
import torch
from app.CNNModelMNIST import CNNModel

class TestCNNModel(unittest.TestCase):
    def setUp(self):
        self.model = CNNModel()
        self.model.eval()

    def test_forward_output_shape(self):
        dummy_input = torch.randn(1, 1, 28, 28)  # MNIST shape
        output = self.model(dummy_input)
        self.assertEqual(output.shape, (1, 10), "Model output should be 1x10 for MNIST classification")
