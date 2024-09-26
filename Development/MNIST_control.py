import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Simple CNN control
        self.scaconv1 = nn.Conv2d(1, 25, kernel_size=3, padding=1, stride=2)
        self.scaconv2 = nn.Conv2d(25, 50, kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU()
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10))
    

    def forward(self, x):
        x = self.relu(self.scaconv1(x))
        x = self.relu(self.scaconv2(x))
        x = self.fc_out(x)
        return x

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Data loading and transformation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
n_epochs = 3
b_arr = torch.linspace(0, 1, (n_epochs-1)*len(train_loader))
train_losses = []
print('Training the Control CNN model...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
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
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


# Plot training losses over each batch
plt.plot(train_losses)
plt.xlabel('Batch number')
plt.ylabel('Loss')
plt.title('Training Loss over Batches (Control)')
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\Control_CNN_loss.png')
plt.show()

# Test the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Control CNN Accuracy of the model on the 10,000 test images: {accuracy:.2f}%')

# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\Control_CNN_results.txt", "a") as f:
    total_params = count_parameters(model)
    f.write(f"Control-CNN - Accuracy of the model on the 10,000 test images: {accuracy:.2f}%\n")
    f.write(f"Control-CNN - Final Training loss: {train_losses[-1]}\n")
    f.write(f"Control-CNN - Total trainable parameters: {total_params}\n\n")
    f.write(str(model) + '\n\n')
