import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from SCAConv_dev import SCAConv

# Simple Global Context Model
class ContextNN(nn.Module):
    def __init__(self, c_len):
        super(ContextNN, self).__init__()
        self.reduce1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
        self.reduce2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
        self.fc_context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7, c_len))
        self.out = nn.Linear(c_len, 10)

    def encode(self, x):
        c = F.relu(self.reduce1(x))
        c = F.relu(self.reduce2(c))
        c = self.fc_context(c)
        return c

    def forward(self, x):
        x = self.encode(x)
        return self.out(x)


# Define the SCA-CNN architecture
class SCACNN(nn.Module):
    def __init__(self, contextNN):
        super(SCACNN, self).__init__()
        self.b = 1
        self.c_len = 8  # Context vector length

        # Global context
        self.context = contextNN

        # SCA-CNN
        self.scaconv1 = SCAConv(1, 16, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len)
        self.scaconv2 = SCAConv(16, 32, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len)
        self.relu = nn.ReLU()
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10))
    

    def forward(self, x):
        # Global context
        c = self.context.encode(x)

        # SCA-CNN with global context
        x = self.relu(self.scaconv1(x, c))
        x = self.relu(self.scaconv2(x, c))
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
contextNN = ContextNN(c_len=8)
model = SCACNN(contextNN)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Context encoder pretraining
n_epochs = 3
context_losses = []
print('Training the global context encoder...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = contextNN(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        context_losses.append(loss.item())
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')


# Train the model
n_epochs = 3
b_arr = torch.linspace(0, 1, (n_epochs-1)*len(train_loader))
train_losses = []
print('Training the SCA-CNN model...')
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

# Plot context pertraining losses over each batch  
plt.plot(context_losses)
plt.xlabel('Batch number')
plt.ylabel('Loss')
plt.title('Context Encoder (Classification) Loss over Batches')
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\Context_loss.png')
plt.show()

# Plot training losses over each batch
plt.plot(train_losses)
plt.xlabel('Batch number')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\SCA_CNN_loss.png')
plt.show()

# Test the context encoder
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = contextNN(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
context_accuracy = 100 * correct / total
print(f'ContextNN Accuracy of the model on the 10,000 test images: {context_accuracy:.2f}%')

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
print(f'SCA-CNN Accuracy of the model on the 10,000 test images: {accuracy:.2f}%')

# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\SCA_CNN_results.txt", "a") as f:
    total_params = count_parameters(model)
    f.write(f"SCA-CNN - Accuracy of the model on the 10,000 test images: {accuracy:.2f}%\n")
    f.write(f"SCA-CNN - Final Training loss: {train_losses[-1]}\n")
    f.write(f"SCA-CNN - Total trainable parameters: {total_params}\n\n")
    f.write(f"ContextNN - Accuracy of the model on the 10,000 test images: {context_accuracy:.2f}%\n")
    f.write(f"ContextNN - Final Training loss: {context_losses[-1]}\n")
    f.write(f"ContextNN - Total trainable parameters: {count_parameters(contextNN)}\n\n")
    f.write(str(contextNN) + '\n\n')
    f.write(str(model) + '\n\n')
