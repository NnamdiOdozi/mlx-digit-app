import unittest
from PIL import Image
from app.main import preprocess_image

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_output_shape(self):
        img = Image.new('L', (280, 280), color=255)  # Blank white image
        processed = preprocess_image(img)
        self.assertEqual(processed.shape, (1, 28, 28), "Output should be 1x28x28")