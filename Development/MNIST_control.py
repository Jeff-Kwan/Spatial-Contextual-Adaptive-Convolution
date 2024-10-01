import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Simple CNN control
        self.convd1 = nn.Conv2d(1, 4, kernel_size=3, padding=1, stride=2)
        self.conv1 = nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
        self.conv4 = nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=1)
        self.convd2 = nn.Conv2d(4, 4, kernel_size=3, padding=1, stride=2)
        self.silu = nn.SiLU()
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 7 * 7, 32),
            nn.SiLU(),
            nn.Linear(32, 10))
    

    def forward(self, x):
        x = self.silu(self.convd1(x))
        x = self.silu(x + self.conv2(self.silu(self.conv1(x))))
        x = self.silu(x + self.conv4(self.silu(self.conv3(x))))
        x = self.silu(self.convd2(x))
        x = self.fc_out(x)
        return x

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformation for training data with data augmentation
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(),  
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])
test_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
n_epochs = 3
b_arr = torch.linspace(0, 1, (n_epochs-1)*len(train_loader)).to(device)
train_losses = []
print('Training the Control CNN model...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        if epoch > 0:
            model.b = b_arr[(epoch-1)*len(train_loader) + i]
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_losses.append(loss.item())

        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0


# Plot training losses over each batch
plt.plot(train_losses)
plt.xlabel('Batch number')
plt.ylabel('Loss')
plt.title('Training Loss over Batches (Control)')
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\MNIST_Control_CNN_loss.png')

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Control CNN accuracy on the 10,000 test images: {accuracy:.2f}%')

# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\MNIST_Control_CNN_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"Control-CNN - Accuracy of the model on the 10,000 test images: {accuracy:.2f}%\n")
    f.write(f"Control-CNN - Final Training loss: {train_losses[-1]}\n")
    f.write(f"Control-CNN - Total trainable parameters: {total_params}\n\n")
    f.write(str(model) + '\n\n')
