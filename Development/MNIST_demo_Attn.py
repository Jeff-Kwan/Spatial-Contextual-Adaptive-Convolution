import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from SCAC_DS_Attn import SCAC_DS_Attn

# Define the SCA-CNN with self-context
class SCACNN(nn.Module):
    def __init__(self, c_len):
        super(SCACNN, self).__init__()
        self.b = 1
        self.c_len = c_len  # Context vector length

        # Self-context
        self.c1 = nn.Sequential(
            nn.LazyConv2d(1, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.LazyLinear(c_len)
        )

        # SCA-CNN
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1, stride=2)
        self.sca1 = SCAC_DS_Attn(2, 2, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, condition=True)
        self.sca2 = SCAC_DS_Attn(2, 2, kernel_size=3, padding=1, stride=1, b=self.b, c_len=self.c_len, condition=True)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1, stride=2)
        self.silu = nn.SiLU()
        self.fc_out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * 7 * 7, 32),
            nn.SiLU(),
            nn.Linear(32, 10))
    

    def forward(self, x):
        # Set b
        self.sca1.b = self.b
        self.sca2.b = self.b

        # SCA-CNN with self-context
        x = self.conv1(x)
        x = self.silu(x + self.sca1(x, self.c1(x)))
        x = self.silu(x + self.sca2(x, self.c1(x)))
        x = self.conv2(x)
        x = self.fc_out(x)
        return x

# Function to count model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Set device
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
c_len = 8
model = SCACNN(c_len).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
n_epochs = 3
# Schedule Adaptive MLP Injection (optional)
b_arr = torch.linspace(0, 1, (n_epochs-1)*len(train_loader))    
train_losses = []
print('Training the SCA-CNN-DS-Attn model...')
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
plt.savefig(r'Spatial-Contextual-Adaptive-Convolution\Development\Outputs\MNIST_SCA_CNN_DS_Attn_loss.png')
plt.close()


# Test the models
SCA_correct = 0
ablation_correct = 0
adaptive_correct = 0
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

        # Adaptive kernel study
        k11 = model.sca1.kernel_D.detach().clone()
        k12 = model.sca1.kernel_P.detach().clone()
        k21 = model.sca2.kernel_D.detach().clone()
        k22 = model.sca2.kernel_P.detach().clone()
        model.sca1.kernel_D -= k11
        model.sca1.kernel_P -= k12
        model.sca2.kernel_D -= k21
        model.sca2.kernel_P -= k22
        adaptive_outputs = model(images)
        _, adaptive_pred = torch.max(adaptive_outputs.data, 1)
        adaptive_correct += (adaptive_pred == labels).sum().item()
        model.sca1.kernel_D += k11
        model.sca1.kernel_P += k12
        model.sca2.kernel_D += k21
        model.sca2.kernel_P += k22

        # Ablation study
        model.b = 0
        ablation_outputs = model(images)
        _, ablation_pred = torch.max(ablation_outputs.data, 1)
        ablation_correct += (ablation_pred == labels).sum().item()

        total += labels.size(0)

SCA_accuracy = 100 * SCA_correct / total
ablation_accuracy = 100 * ablation_correct / total
adaptive_accuracy = 100 * adaptive_correct / total
print(f'SCA-CNN-DS-Attn accuracy on the 10,000 test images: {SCA_accuracy:.2f}%')
print(f'Ablated SCA-CNN-DS-Attn accuracy on the 10,000 test images: {ablation_accuracy:.2f}%')
print(f'Adaptive-only SCA-CNN-DS-Attn accuracy on the 10,000 test images: {adaptive_accuracy:.2f}%')

print(f"SCA-DNN-DS-Attn sca1 Adaptive Importance: {model.sca1.a[0].item():.2f}, {model.sca1.a[1].item():.2f}")
print(f"SCA-DNN-DS-Attn sca2 Adaptive Importance: {model.sca2.a[0].item():.2f}, {model.sca2.a[1].item():.2f}")


# Save the test accuracy to the same text file
with open(r"Spatial-Contextual-Adaptive-Convolution\Development\Outputs\MNIST_SCA_CNN_DS_Attn_results.txt", "w") as f:
    total_params = count_parameters(model)
    f.write(f"SCA-CNN-DS-Attn - Accuracy on the 10,000 test images: {SCA_accuracy:.2f}%\n")
    f.write(f"Ablated SCA-CNN-DS-Attn - Accuracy on the 10,000 test images: {ablation_accuracy:.2f}%\n")
    f.write(f"Adaptive-only SCA-CNN-DS-Attn - Accuracy on the 10,000 test images: {adaptive_accuracy:.2f}%\n\n")
    f.write(f"SCA-CNN-DS-Attn - Final Training loss: {train_losses[-1]}\n")
    f.write(f"SCA-CNN-DS-Attn - Total trainable parameters: {total_params}\n\n")
    f.write(str(model) + '\n\n')
