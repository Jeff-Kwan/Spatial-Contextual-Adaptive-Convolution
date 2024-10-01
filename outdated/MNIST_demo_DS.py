import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from SCAC_DS import SCAConv_DS

# Simple Global Context Model
class ContextNN(nn.Module):
    def __init__(self, c_len):
        super(ContextNN, self).__init__()
        # self.reduce1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
        # self.reduce2 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
        # self.fc_context = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(1 * 7 * 7, c_len))
        
        self.conv = nn.Conv1d(1, 1, kernel_size=100, padding=1, stride=98) # c_len=8 only
        # self.linear = nn.Linear(784, c_len)
        self.out = nn.Linear(c_len, 10)

    def encode(self, x):
        # c = F.relu(self.reduce1(x))
        # c = F.relu(self.reduce2(c))
        # c = self.fc_context(c)

        c = self.conv(x.view(x.size(0), 1, -1)).squeeze()

        # c = self.linear(x.view(x.size(0), -1))
        return c

    def forward(self, x, finetune=False):
        if finetune:
            with torch.no_grad():
                x = self.encode(x)
        else:
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
        h = 16 # Hidden units in the MLP of Adaptive Kernel
        self.scaconv1 = SCAConv_DS(1, 2, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True)
        self.scaconv2 = SCAConv_DS(2, 2, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True)
        self.relu = nn.ReLU()
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 7 * 7, 32),
            nn.ReLU(),
            nn.Linear(32, 10))
    

    def forward(self, x):
        # Set b
        self.scaconv1.b = self.b
        self.scaconv2.b = self.b

        # Global context
        # with torch.no_grad():
        #    c = self.context.encode(x)
        c = self.context.encode(x)
        # c = torch.ones((x.shape[0],8))

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
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0


# Train the model
n_epochs = 3
# Schedule Adaptive MLP Injection (optional)
b_arr = torch.linspace(0, 1, (n_epochs-1)*len(train_loader))    
train_losses = []
print('Training the SCA-CNN-DS model...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        if epoch < n_epochs-1:
            model.b = 1#b_arr[(epoch)*len(train_loader) + i]
        else:
            model.b = 1
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
plt.title('Training Loss over Batches')
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\SCA_CNN_DS_loss.png')
plt.close()

# Fine tuning ContextNN linear readout
n_epochs = 2
print('Fine-tuning the global context encoder linear readout...')
for epoch in range(n_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        outputs = contextNN(images, finetune=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0

# Test the models
context_correct = 0
SCA_correct = 0
ablation_correct = 0
adaptive_correct = 0
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
        model.b = 1
        sca_outputs = model(images)
        _, SCA_pred = torch.max(sca_outputs.data, 1)
        SCA_correct += (SCA_pred == labels).sum().item()

        # Adaptive kernel study
        k11 = model.scaconv1.kernel_D.detach().clone()
        k12 = model.scaconv1.kernel_P.detach().clone()
        k21 = model.scaconv2.kernel_D.detach().clone()
        k22 = model.scaconv2.kernel_P.detach().clone()
        model.scaconv1.kernel_D -= k11
        model.scaconv1.kernel_P -= k12
        model.scaconv2.kernel_D -= k21
        model.scaconv2.kernel_P -= k22
        adaptive_outputs = model(images)
        _, adaptive_pred = torch.max(adaptive_outputs.data, 1)
        adaptive_correct += (adaptive_pred == labels).sum().item()
        model.scaconv1.kernel_D += k11
        model.scaconv1.kernel_P += k12
        model.scaconv2.kernel_D += k21
        model.scaconv2.kernel_P += k22

        # Ablation study
        model.b = 0
        ablation_outputs = model(images)
        _, ablation_pred = torch.max(ablation_outputs.data, 1)
        ablation_correct += (ablation_pred == labels).sum().item()

        total += labels.size(0)

context_accuracy = 100 * context_correct / total
SCA_accuracy = 100 * SCA_correct / total
ablation_accuracy = 100 * ablation_correct / total
adaptive_accuracy = 100 * adaptive_correct / total
print(f'ContextNN accuracy on the 10,000 test images: {context_accuracy:.2f}%')
print(f'SCA-CNN-DS accuracy on the 10,000 test images: {SCA_accuracy:.2f}%')
print(f'Ablated SCA-CNN-DS accuracy on the 10,000 test images: {ablation_accuracy:.2f}%')
print(f'Adaptive-only SCA-CNN-DS accuracy on the 10,000 test images: {adaptive_accuracy:.2f}%')

print(f"SCA-CNN-DS scaconv1 mlp_D norms: {model.scaconv1.mlp_D[0].weight.norm():.2f}, {model.scaconv1.mlp_D[2].weight.norm():.2f}")
print(f"SCA-CNN-DS scaconv1 mlp_P norms: {model.scaconv1.mlp_P[0].weight.norm():.2f}, {model.scaconv1.mlp_P[2].weight.norm():.2f}")
print(f"SCA-CNN-DS scaconv2 mlp_D norms: {model.scaconv2.mlp_D[0].weight.norm():.2f}, {model.scaconv2.mlp_D[2].weight.norm():.2f}")
print(f"SCA-CNN-DS scaconv2 mlp_P norms: {model.scaconv2.mlp_P[0].weight.norm():.2f}, {model.scaconv2.mlp_P[2].weight.norm():.2f}")


# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\SCA_CNN_DS_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"SCA-CNN-DS - Accuracy on the 10,000 test images: {SCA_accuracy:.2f}%\n")
    f.write(f"Ablated SCA-CNN-DS - Accuracy on the 10,000 test images: {ablation_accuracy:.2f}%\n")
    f.write(f"Adaptive-only SCA-CNN-DS - Accuracy on the 10,000 test images: {adaptive_accuracy:.2f}%\n\n")
    f.write(f"SCA-CNN-DS - Final Training loss: {train_losses[-1]}\n")
    f.write(f"SCA-CNN-DS - Total trainable parameters: {total_params}\n\n")
    f.write(f"ContextNN - Accuracy on the 10,000 test images: {context_accuracy:.2f}%\n")
    f.write(f"ContextNN - Total trainable parameters: {count_parameters(contextNN)}\n\n")
    f.write(str(contextNN) + '\n\n')
    f.write(str(model) + '\n\n')
