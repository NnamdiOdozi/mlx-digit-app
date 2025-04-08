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

from PIL import Image


class TestPreprocessing(unittest.TestCase):
    def test_preprocess_output_shape(self):
        img = Image.new('L', (280, 280), color=255)  # Blank white image
        processed = preprocess_image(img)
        self.assertEqual(processed.shape, (1, 28, 28), "Output should be 1x28x28")