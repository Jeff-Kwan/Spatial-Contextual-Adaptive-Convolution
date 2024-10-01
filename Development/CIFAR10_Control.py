import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class ControlCNN(nn.Module):
    def __init__(self):
        super(ControlCNN, self).__init__()
        # CNN control for CIFAR10
        self.in_conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1)
            ])
        self.down_convs = nn.ModuleList([
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            ])
        self.silu = nn.SiLU()
        self.norms = nn.ModuleList([
            nn.LayerNorm([16, 32, 32]),
            nn.LayerNorm([32, 16, 16]),
            nn.LayerNorm([64, 8, 8]),
            ])
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.SiLU(),
            nn.Linear(256, 10))
    
    def forward(self, x):
        x = self.in_conv(x)
        for conv, down, norm in zip(self.convs, self.down_convs, self.norms):
            x = self.silu(norm(x + conv(x))) # skip connection
            x = self.silu(down(x)) 
        x = self.fc_out(x)
        return x

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and transformation
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

# Initialize the model, loss function, and optimizer
model = ControlCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=0.98)

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

# Train the model
n_epochs = 10
train_losses = torch.zeros(n_epochs)
test_acc = []
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
        scheduler.step()
        
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 batches
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
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
plt.title('Training Loss and Test Accuracy of Control CNN on CIFAR 10')
fig.tight_layout()
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_Control_CNN_metrics.png')

# Test the model
accuracy = test_model(model, test_loader)
print(f'Control CNN accuracy on test images: {accuracy:.2f}%')

# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_Control_CNN_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"Control-CNN - Accuracy of the model on test images: {accuracy:.2f}%\n")
    f.write(f"Control-CNN - Final Training loss: {train_losses[-1]}\n")
    f.write(f"Control-CNN - Total trainable parameters: {total_params}\n\n")
    f.write(str(model) + '\n\n')




'''
# Complicated formulation
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        return self.conv2(self.conv1(x))

# Define the SCA-CNN with self-context
class ControlCNN(nn.Module):
    def __init__(self):
        super(ControlCNN, self).__init__()
        # CNN - Backbone
        self.conv_in = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.conv_downs = nn.ModuleList([
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),   # 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),  # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),  # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=4, padding=0, stride=1)]) # 4x4 -> 2x2
        self.norms = nn.ModuleList([nn.LayerNorm([16, 32, 32]), 
                                    nn.LayerNorm([32, 16, 16]),
                                    nn.LayerNorm([64, 8, 8]),
                                    nn.LayerNorm([128, 4, 4])])
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 1 * 1, 64),
            nn.SiLU(),
            nn.Linear(64, 10))
        self.silu = nn.SiLU()

        # Residual SCA Blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(16,  16,  3, 1, 1),
            ResBlock(32,  32,  3, 1, 1),
            ResBlock(64,  64,  3, 1, 1),
            ResBlock(128, 128, 3, 1, 1)])

    def forward(self, x):
        x = self.conv_in(x)

        # Residual Blocks
        for i, (block, norm) in enumerate(zip(self.res_blocks, self.norms)):
            x = x + block(x)
            x = self.silu(norm(x))
            x = self.conv_downs[i](x)
        x = self.fc_out(x)
        return x
'''