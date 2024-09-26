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
        self.reduce1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=2)
        self.reduce2 = nn.Conv2d(2, 4, kernel_size=3, padding=1, stride=2)
        self.fc_context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 7 * 7, c_len))
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
    def __init__(self, contextNN, c_len):
        super(SCACNN, self).__init__()
        self.b = 1
        self.c_len = c_len  # Context vector length

        # Global context
        self.context = contextNN

        # SCA-CNN
        self.scaconv1 = SCAConv(1, 2, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len, condition=True)
        self.scaconv2 = SCAConv(2, 2, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len, condition=True)
        self.relu = nn.ReLU()
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10))
    

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


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and transformation
batch_size = 128
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
c_len = 8
contextNN = ContextNN(c_len).to(device)
model = SCACNN(contextNN, c_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Context encoder pretraining (optional)
n_epochs = 0
context_losses = [] if n_epochs > 0 else ["Not pretrained"]
print('Pre-training the global context encoder...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = contextNN(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        context_losses.append(loss.item())
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0


# Train the model
n_epochs = 3
# Schedule Adaptive MLP Injection (optional)
b_arr = torch.linspace(0, 1, (n_epochs-1)*len(train_loader))    
train_losses = []
print('Training the SCA-CNN model...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        if epoch > 0:
            model.scaconv1.b = 1#b_arr[(epoch-1)*len(train_loader) + i]
            model.scaconv2.b = 1#b_arr[(epoch-1)*len(train_loader) + i]
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

# Plot context pertraining losses over each batch  
plt.plot(context_losses)
plt.xlabel('Batch number')
plt.ylabel('Loss')
plt.title('Context Encoder (Classification) Loss over Batches')
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\Context_loss.png')
plt.clf()

# Plot training losses over each batch
plt.plot(train_losses)
plt.xlabel('Batch number')
plt.ylabel('Loss')
plt.title('Training Loss over Batches')
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\SCA_CNN_loss.png')
plt.close()

# Test the models
context_correct = 0
SCA_correct = 0
ablation_correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        c_outputs = contextNN(images)
        _, C_pred = torch.max(c_outputs.data, 1)
        context_correct += (C_pred == labels).sum().item()

        # Normal model
        model.scaconv1.b = 1
        model.scaconv2.b = 1
        sca_outputs = model(images)
        _, SCA_pred = torch.max(sca_outputs.data, 1)
        SCA_correct += (SCA_pred == labels).sum().item()

        # Ablation study
        model.scaconv1.b = 0
        model.scaconv2.b = 0
        ablation_outputs = model(images)
        _, ablation_pred = torch.max(ablation_outputs.data, 1)
        ablation_correct += (ablation_pred == labels).sum().item()

        total += labels.size(0)

context_accuracy = 100 * context_correct / total
SCA_accuracy = 100 * SCA_correct / total
ablation_accuracy = 100 * ablation_correct / total
print(f'ContextNN accuracy on the 10,000 test images: {context_accuracy:.2f}%')
print(f'SCA-CNN accuracy on the 10,000 test images: {SCA_accuracy:.2f}%')
print(f'Ablated SCA-CNN accuracy on the 10,000 test images: {ablation_accuracy:.2f}%')

print(f"SCA-CNN scaconv1 mlp norms: {model.scaconv1.mlp[0].weight.norm():.2f}, {model.scaconv1.mlp[2].weight.norm():.2f}")
print(f"SCA-CNN scaconv2 mlp norms: {model.scaconv2.mlp[0].weight.norm():.2f}, {model.scaconv2.mlp[2].weight.norm():.2f}")


# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\SCA_CNN_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"SCA-CNN - Accuracy on the 10,000 test images: {SCA_accuracy:.2f}%\n")
    f.write(f"Ablated SCA-CNN - Accuracy on the 10,000 test images: {ablation_accuracy:.2f}%\n")
    f.write(f"SCA-CNN - Final Training loss: {train_losses[-1]}\n")
    f.write(f"SCA-CNN - Total trainable parameters: {total_params}\n\n")
    f.write(f"ContextNN - Accuracy on the 10,000 test images: {context_accuracy:.2f}%\n")
    f.write(f"ContextNN - Final Training loss: {context_losses[-1]}\n")
    f.write(f"ContextNN - Total trainable parameters: {count_parameters(contextNN)}\n\n")
    f.write(str(contextNN) + '\n\n')
    f.write(str(model) + '\n\n')
