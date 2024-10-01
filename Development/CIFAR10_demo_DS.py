import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from SCAC_DS import SCAConv_DS

class SCACNN(nn.Module):
    def __init__(self, c_len):
        super(SCACNN, self).__init__()
        self.b = 1
        self.c_len = c_len
        h = 8 # Hidden units in the MLP of Adaptive Kernel
        self.in_conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1)
        self.convs = nn.ModuleList([
            SCAConv_DS(8, 8, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=False),
            SCAConv_DS(16, 16, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=False),
            SCAConv_DS(32, 32, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, mlp_hidden=h, condition=False)
            ])
        self.down_convs = nn.ModuleList([
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            ])
        self.silu = nn.SiLU()
        self.norms = nn.ModuleList([
            nn.LayerNorm([8, 32, 32]),
            nn.LayerNorm([16, 16, 16]),
            nn.LayerNorm([32, 8, 8]),
            ])
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256),
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
batch_size = 128
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  
    transforms.RandomVerticalFlip(),  
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all three channels
])

train_data = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all three channels
]))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
c_len = 8
model = SCACNN(c_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=len(train_loader), gamma=0.9)

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
        #scheduler.step()
        
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
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_SCA_CNN_DS_metrics.png')
plt.close()


# Test the models
SCA_correct = 0
ablation_correct = 0
total = 0
with torch.no_grad():
    model.eval()
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

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

SCA_accuracy = 100 * SCA_correct / total
ablation_accuracy = 100 * ablation_correct / total
print(f'SCA-CNN-DS accuracy on the 10,000 test images: {SCA_accuracy:.2f}%')
print(f'Ablated SCA-CNN-DS accuracy on the 10,000 test images: {ablation_accuracy:.2f}%')

for sca in model.convs:
    print(f"SCA-CNN-DS mlp_D norms: {sca.mlp[0].weight.norm():.2f}, {sca.mlp[2].weight.norm():.2f}")
    print(f"SCA-CNN-DS Adaptive Importance: {sca.a[0].item():.2f}, {sca.a[1].item():.2f}")

#for i, block in enumerate(model.res_blocks):
    #print(f"SCA-CNN-DS block {i} mlp_D norms: {block.conv1.mlp[0].weight.norm():.2f}, {block.conv1.mlp[2].weight.norm():.2f}")
    #print(f"SCA-CNN-DS block {i} mlp_D norms: {block.conv2.mlp[0].weight.norm():.2f}, {block.conv2.mlp[2].weight.norm():.2f}")
    #print(f"SCA-CNN-DS block {i} Adaptive Importance: {block.conv1.a[0].item():.2f}, {block.conv1.a[1].item():.2f}")
    #print(f"SCA-CNN-DS block {i} Adaptive Importance: {block.conv2.a[0].item():.2f}, {block.conv2.a[1].item():.2f}")

# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\CIFAR10_SCA_CNN_DS_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"SCA-CNN-DS - Accuracy on the 10,000 test images: {SCA_accuracy:.2f}%\n")
    f.write(f"Ablated SCA-CNN-DS - Accuracy on the 10,000 test images: {ablation_accuracy:.2f}%\n")
    f.write(f"SCA-CNN-DS - Final Training loss: {train_losses[-1]}\n")
    f.write(f"SCA-CNN-DS - Total trainable parameters: {total_params}\n\n")
    f.write(str(model) + '\n\n')


'''
# Complicated formulation
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, b, c_len, mlp_hidden, condition):
        super(ResBlock, self).__init__()
        self.b = b
        self.conv1 = SCAConv_DS(in_channels, out_channels, kernel_size, padding, stride, b, c_len, mlp_hidden, condition)
        #self.conv2 = SCAConv_DS(in_channels, out_channels, kernel_size, padding, stride, b, c_len, mlp_hidden, condition)
        #self.silu = nn.SiLU()

    def forward(self, x, c):
        self.conv1.b = self.b
        return self.conv1(x, c)
        #self.conv2.b = self.b
        #return self.conv2(self.silu(self.conv1(x, c)))

# Define the SCA-CNN with self-context
class SCACNN(nn.Module):
    def __init__(self, c_len):
        super(SCACNN, self).__init__()
        self.b = 1
        self.c_len = c_len  # Context vector length

        # Number of res_blocks
        self.res_blocks = 4

        # Self-context
        self.contexts = nn.ModuleList([
            nn.Sequential(
            nn.LazyConv2d(1, kernel_size=1, padding=0, stride=1),
            nn.GroupNorm(1, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.LazyLinear(c_len)) for _ in range(self.res_blocks)])

        # CNN - Backbone
        self.conv_in = nn.Conv2d(3, 4, kernel_size=3, padding=1, stride=1)
        self.conv_downs = nn.ModuleList([
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=2),       # 32x32 -> 16x16
            nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2),      # 16x16 -> 8x8
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),      # 8x8 -> 4x4
            nn.Conv2d(32, 64, kernel_size=4, padding=0, stride=1)])    # 4x4 -> 1x1
        self.norms = nn.ModuleList([nn.LayerNorm([4, 32, 32]), 
                                    nn.LayerNorm([8, 16, 16]),
                                    nn.LayerNorm([16, 8, 8]),
                                    nn.LayerNorm([32, 4, 4])])
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 128),
            nn.SiLU(),
            nn.Linear(128, 10))
        self.silu = nn.SiLU()

        # Residual SCA Blocks
        h = 12 # Hidden units in the MLP of Adaptive Kernel
        self.res_blocks = nn.ModuleList([
            ResBlock(4,  4,  3, 1, 1, self.b, self.c_len, h, True),
            ResBlock(8,  8,  3, 1, 1, self.b, self.c_len, h, True),
            ResBlock(16, 16, 3, 1, 1, self.b, self.c_len, h, True),
            ResBlock(32, 32, 3, 1, 1, self.b, self.c_len, h, True)])

    def forward(self, x):
        x = self.conv_in(x)

        # Residual Blocks
        for i, (block, norm) in enumerate(zip(self.res_blocks, self.norms)):
            block.b = self.b
            x = self.silu(norm(x + block(x, self.contexts[i](x))))
            x = self.silu(self.conv_downs[i](x))
        x = self.fc_out(x)
        return x


'''