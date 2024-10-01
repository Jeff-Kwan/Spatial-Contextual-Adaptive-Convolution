import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from SCAC_DS import SCAConv_DS

# Global Context Model
class ContextNN(nn.Module):
    def __init__(self, c_len):
        super(ContextNN, self).__init__()
        self.reduce1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=2)
        self.conv1 = nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1)
        self.norm1 = nn.LayerNorm([8, 16, 16])
        self.reduce2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2)
        self.fc_context = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, c_len))
        
        # self.conv = nn.Conv1d(3,1,stride=52,kernel_size=256,padding=6) # c_len=16 only
        self.out = nn.Linear(c_len, 10)

    def encode(self, x):
        c = F.silu(self.reduce1(x))
        c = F.silu(self.norm1(c + self.conv1(c)))
        c = F.silu(self.reduce2(c))
        c = self.fc_context(c)

        # c = self.conv(x.view(x.size(0), 3, -1)).squeeze()
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
        h = 32 # Hidden units in the MLP of Adaptive Kernel
        self.scac_in = SCAConv_DS(3, 8, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True)
        self.scaconvs1 = nn.ModuleList([
            SCAConv_DS(8, 8, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True),
            SCAConv_DS(16, 16, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True),
            SCAConv_DS(32, 32, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True)
            ])
        self.scaconvs2 = nn.ModuleList([
            SCAConv_DS(8, 16, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True),
            SCAConv_DS(16, 32, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True),
            SCAConv_DS(32, 64, kernel_size=3, padding=1, stride=2, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=True)
            ])

        self.norms = nn.ModuleList([
            nn.LayerNorm([8, 32, 32]),
            nn.LayerNorm([16, 16, 16]),
            nn.LayerNorm([32, 8, 8])
            ])

        self.silu = nn.SiLU()
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
            nn.SiLU(),
            nn.Linear(256, 10))
    

    def forward(self, x):
        # Global context
        with torch.no_grad():
           c = self.context.encode(x)
        # c = self.context.encode(x)
        # c = torch.ones((x.shape[0],8))

        # SCA-CNN
        self.scac_in.b = self.b
        x = self.scac_in(x, c)
        for scac1, scac2, norm in zip(self.scaconvs1, self.scaconvs2, self.norms):
            scac1.b = self.b
            scac2.b = self.b
            x = self.silu(norm(x + scac1(x, c)))    # skip connection
            x = self.silu(scac2(x, c)) 
        x = self.fc_out(x)
        return x

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and transformation with data augmentation
batch = 128
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(),  
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
])
train_data = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=False)

def test_model(model, test_loader):
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
    return accuracy

# Initialize the model, loss function, and optimizer
c_len = 32
contextNN = ContextNN(c_len).to(device)
model = SCACNN(contextNN, c_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=0.99)

# Context encoder pretraining (optional)
n_epochs = 5
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
n_epochs = 10
train_losses = torch.zeros(n_epochs)
test_acc = []
print('Training the SCA-CNN-DS model...')
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
        scheduler.step()  # Step the learning rate scheduler
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/100:.4f}')
            running_loss = 0.0
        train_losses[epoch] += loss.item()
    test_acc.append(test_model(model, test_loader))
train_losses = train_losses / len(train_loader)


# Plot training losses over each batch
fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(train_losses, color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Test Accuracy', color=color)
ax2.plot(test_acc, color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Training Loss and Test Accuracy of SCA-CNN-DS on CIFAR 10')
fig.tight_layout() 
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_SCA_CNN_DS_metrics.png')
plt.close()

# Fine tuning ContextNN linear readout
n_epochs = 0
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
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

        # Ablation study
        model.b = 0
        ablation_outputs = model(images)
        _, ablation_pred = torch.max(ablation_outputs.data, 1)
        ablation_correct += (ablation_pred == labels).sum().item()

        total += labels.size(0)

context_accuracy = 100 * context_correct / total
SCA_accuracy = 100 * SCA_correct / total
ablation_accuracy = 100 * ablation_correct / total
print(f'ContextNN accuracy on the 10,000 test images: {context_accuracy:.2f}%')
print(f'SCA-CNN-DS accuracy on the 10,000 test images: {SCA_accuracy:.2f}%')
print(f'Ablated SCA-CNN-DS accuracy on the 10,000 test images: {ablation_accuracy:.2f}%')
print()

print(f"SCA-in Adaptive Importance is {model.scac_in.a.item()}")
for sca in model.scaconvs1:
    print(f"SCAC-1 Adaptive Importance is {sca.a.item()}")
for sca in model.scaconvs2:
    print(f"SCAC-2 Adaptive Importance is {sca.a.item()}")


# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_SCA_CNN_DS_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"SCA-CNN-DS - Accuracy on the 10,000 test images: {SCA_accuracy:.2f}%\n")
    f.write(f"Ablated SCA-CNN-DS - Accuracy on the 10,000 test images: {ablation_accuracy:.2f}%\n")
    f.write(f"SCA-CNN-DS - Final Training loss: {train_losses[-1]}\n")
    f.write(f"SCA-CNN-DS - Total trainable parameters: {total_params}\n\n")
    f.write(f"ContextNN - Accuracy on the 10,000 test images: {context_accuracy:.2f}%\n")
    f.write(f"ContextNN - Total trainable parameters: {count_parameters(contextNN)}\n\n")
    f.write(str(contextNN) + '\n\n')
    f.write(str(model) + '\n\n')
