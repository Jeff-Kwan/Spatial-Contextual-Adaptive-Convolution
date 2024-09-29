import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ControlCNN(nn.Module):
    def __init__(self):
        super(ControlCNN, self).__init__()
        # CNN control for CIFAR10
        self.convs = nn.ModuleList([
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
            ])
        self.down_convs = nn.ModuleList([
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
            ])
        self.relu = nn.ReLU()
        self.norms = nn.ModuleList([
            nn.LayerNorm([16, 32, 32]),
            nn.LayerNorm([32, 16, 16]),
            nn.LayerNorm([64, 8, 8]),
            ])
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10))
    
    def forward(self, x):
        for conv, down, norm in zip(self.convs, self.down_convs, self.norms):
            x = self.relu(down(norm(self.relu(conv(x)))))
        x = self.fc_out(x)
        return x

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and transformation
batch = 128
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_data = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)

# Initialize the model, loss function, and optimizer
model = ControlCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
n_epochs = 10
train_losses = []
print('Training the Control CNN model...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
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
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_Control_CNN_loss.png')

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
print(f'Control CNN accuracy on test images: {accuracy:.2f}%')

# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_Control_CNN_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"Control-CNN - Accuracy of the model on test images: {accuracy:.2f}%\n")
    f.write(f"Control-CNN - Final Training loss: {train_losses[-1]}\n")
    f.write(f"Control-CNN - Total trainable parameters: {total_params}\n\n")
    f.write(str(model) + '\n\n')
