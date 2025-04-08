# utils.py
from torchvision import transforms

# Define preprocessing transform
transform = transforms.Compose([
transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.LANCZOS),
transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,))
 ])

def preprocess_image(image):
    """Preprocess the image from canvas to match MNIST dataset."""
    transform = transforms.Compose([
        transforms.Resize((28, 28), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = transform(image)  
    return image


