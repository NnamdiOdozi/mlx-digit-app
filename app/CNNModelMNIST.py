# %%
# https://pytorch.org/docs/stable/optim.html
# https://pytorch.org/docs/stable/optim.html#algorithms
# https://pytorch.org/docs/stable/nn.html#loss-functions
# https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
# https://pytorch.org/docs/stable/nn.init.html
# 



# %%
# Import necessary libraries
import torch
import torch.nn as nn                    # For building neural networks
import torch.optim as optim              # For optimization algorithms
import torch.nn.functional as F          # For activation functions and other utilities
from torchvision import datasets, transforms  # For loading and transforming datasets
from torch.utils.data import DataLoader  # For data loading and batching
import matplotlib.pyplot as plt          # For plotting
import numpy as np



# Define the CNN model by subclassing nn.Module
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()  # Initialize the base class
        # Convolutional layer 1: Input channels = 1 (grayscale), Output channels = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Convolutional layer 2: Input channels = 16, Output channels = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # Max pooling layer: Reduces spatial dimensions
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer 1
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # Fully connected layer 2 (Output layer)
        self.fc2 = nn.Linear(128, 10)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Convolutional layer 1 followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))
        # Convolutional layer 2 followed by ReLU activation and pooling
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the output for the fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        # Apply dropout
        x = self.dropout(x)
        # Fully connected layer 1 with ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x
    
 # Define the classes in the Fashion MNIST dataset
classes = [
        'Zero','One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'
]
# Display a few images from the training dataset
def show_images(dataset, num_images=8, title='Training Images'):
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title, fontsize=15)
    for idx in range(num_images):
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
        image, label = dataset[idx]
        image = image / 2 + 0.5  # Unnormalize the image
        np_image = image.numpy()
        plt.imshow(np.transpose(np_image, (1, 2, 0)), cmap='gray')
        ax.set_title(classes[label])

    plt.tight_layout()
    plt.show()


# Training logic should **only run if the script is executed directly**
if __name__ == "__main__" and __file__.endswith("CNNModelMNIST.py"):
    print("Training the CNN model for MNIST...")


    # Check if GPU is available and set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # %%
    # Define transformations for the training and testing data
    transform = transforms.Compose([
        transforms.ToTensor(),                # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
    ])

    # %%
    # Download and load the training data
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)

    # Download and load the test data
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    # %%
    # Define data loaders for batching and shuffling
    batch_size = 128

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False)

   


    # %%
    # import matplotlib.pyplot as plt
    # import numpy as np
    # from torchvision import datasets, transforms

    # # Define transformations for the data
    # transform = transforms.Compose([
    #     transforms.ToTensor(),                # Convert images to PyTorch tensors
    #     transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
    # ])

    # # Download and load the training data
    # train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)

    # # Download and load the test data
    # test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # # Define the classes in the Fashion MNIST dataset
    # classes = [
    #     'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #     'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot'
    # ]

    

    # %%
    # Display images from the training dataset
    show_images(train_dataset, num_images=8, title='Training Images')

    # %%
    # Display images from the test dataset
    show_images(test_dataset, num_images=8, title='Test Images')

    # %%


    # %%


    # %%


    # %%


    # %%


    # %%


    # %%
    # Instantiate the model and move it to the device (CPU or GPU)
    model = CNNModel().to(device)

    # Define the loss function (Cross-Entropy Loss for multi-class classification)
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer (Adam optimizer with a learning rate of 0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs to train
    num_epochs = 10

    # Lists to store training and validation loss and accuracy
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            # Move images and labels to the device
            images = images.to(device)
            labels = labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate the loss
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)  # Get the class with highest probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate average loss and accuracy for the epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        # Validation step
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_images, val_labels in test_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                
                # Forward pass
                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)
                
                # Accumulate validation loss
                val_running_loss += val_loss.item()
                
                # Calculate validation accuracy
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        # Calculate average validation loss and accuracy
        val_epoch_loss = val_running_loss / len(test_loader)
        val_epoch_accuracy = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        
        # Print epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.4f}')
        print(f'Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}\n')

    # %%
    # Plot training and validation loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

    # %%
    # Plot training and validation accuracy over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()

